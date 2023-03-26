from collections import OrderedDict
from typing import Tuple
from einops import rearrange, reduce, repeat
from timm.models.layers import trunc_normal_
import torch
from torch import nn
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential
import sys

from models.align import ILA
from models.metrics import cos_similarity_loss, timewise_cos

sys.path.append("../")
from clip.model import LayerNorm, QuickGELU, DropPath


def shift(x, n_segment=3, n_div=8):
    bt, l, h, d = x.size()
    n_batch = bt // n_segment
    fold = d * h // n_div
    x = rearrange(x, "(b t) l h d -> b t (h d) l", b=n_batch, t=n_segment, l=l, h=h, d=d)
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
    out = rearrange(out, "b t (h d) l -> (b t) l h d", b=n_batch, t=n_segment, l=l, h=h, d=d)
    return out


class TemporalCrossAttention(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int] = (14, 14),
        feature_dim: int = 768,
        num_head: int = 12,
        T: int = 8,
    ):
        super().__init__()

        self.spatial_size = spatial_size
        self.num_head = num_head
        self.T = T

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)

        w_size = np.prod([x * 2 - 1 for x in spatial_size])
        self.w1 = nn.Parameter(torch.zeros([w_size, feature_dim]))
        self.w2 = nn.Parameter(torch.zeros([w_size, feature_dim]))

        idx_tensor = torch.zeros([np.prod(spatial_size) for _ in (0, 1)], dtype=torch.long)
        for q in range(np.prod(spatial_size)):
            qi, qj = q // spatial_size[1], q % spatial_size[1]
            for k in range(np.prod(spatial_size)):
                ki, kj = k // spatial_size[1], k % spatial_size[1]
                i_offs = qi - ki + spatial_size[0] - 1
                j_offs = qj - kj + spatial_size[1] - 1
                idx_tensor[q, k] = i_offs * (spatial_size[1] * 2 - 1) + j_offs
        self.idx_tensor = idx_tensor

    def forward_half(self, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        q, k = q[:, :, 1:], k[:, :, 1:] # remove cls token
        assert q.size() == k.size()
        assert q.size(2) == np.prod(self.spatial_size)
        attn = torch.einsum('ntqhd,ntkhd->ntqkh', q / (q.size(-1) ** 0.5), k)
        attn = attn.softmax(dim=-2).mean(dim=-1) # L, L, N, T
        self.idx_tensor = self.idx_tensor.to(w.device)
        w_unroll = w[self.idx_tensor]  # L, L, C

        attn = attn.float()
        w_unroll = w_unroll.float()

        ret = torch.einsum('ntqk,qkc->ntqc', attn, w_unroll)
        return ret

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        NT, L, D = q.size()
        N = NT // self.T
        T = self.T
        assert L == np.prod(self.spatial_size) + 1

        q = self.query_proj(q)
        k = self.key_proj(k)
        q = q.view(NT, L, self.num_head, D // self.num_head)
        k = k.view(NT, L, self.num_head, D // self.num_head)
        q = rearrange(q, "(b t) l h d -> b t l h d", b=NT // self.T, t=self.T, l=L, h=self.num_head, d=D // self.num_head)
        k = rearrange(k, "(b t) l h d -> b t l h d", b=NT // self.T, t=self.T, l=L, h=self.num_head, d=D // self.num_head)

        ret = torch.zeros([N, T, L, self.w1.size(-1)], device='cuda')
        ret[:, 1:, 1:, :] += self.forward_half(q[:, 1:, :, :, :], k[:, :-1, :, :, :], self.w1)
        ret[:, :-1, 1:, :] += self.forward_half(q[:, :-1, :, :, :], k[:, 1:, :, :, :], self.w2)
        ret = rearrange(ret, "b t l c -> (b t) l c", b=N, t=T, l=L, c=D)
        return ret


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
            self, T: int, q_in_dim: int, k_in_dim: int, v_in_dim: int,
            qk_proj_dim: int, v_proj_dim: int, num_heads: int, out_dim: int,
            return_all_features: bool = False,
    ):
        super().__init__()

        self.T = T

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        self.return_all_features = return_all_features
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0)
        assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1)
        assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        # =========================Temporal Shift======================= #
        # k = shift(k, n_segment=self.T)
        # v = shift(v, n_segment=self.T)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        if self.return_all_features:
            return dict(q=q, k=k, v=v, aff=aff, out=out)
        else:
            return out, q, k, v


class MultiAxisAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath=0., T=0, input_resolution=224, patch_size=16, idx=0):
        super().__init__()
        self.T = T
        self.W = input_resolution // patch_size
        self.in_feature_dim = d_model
        self.D_hidden_features = 128
        self.middle_frame_index = self.T // 2
        self.idx = idx

        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.number_step = input_resolution // patch_size

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = ILA(T=self.T, d_model=d_model, patch_size=patch_size,input_resolution=input_resolution)
        self.temporal_skip_fc = nn.Linear(d_model, d_model)
        self.temporal_skip_fc.weight.data.zero_()
        self.temporal_skip_fc.bias.data.zero_()
        self.temporal_skip_align_fc = nn.Linear(d_model, d_model)
        self.temporal_skip_align_fc.weight.data.zero_()
        self.temporal_skip_align_fc.bias.data.zero_()

        self.attn = nn.MultiheadAttention(d_model, n_head, )
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, feature):
        x = feature["input"]
        l, bt, d = x.size()
        B = bt // self.T
        T = self.T
        W = self.W
        H = W

        raw_x = x
        align_x = x.view(l, B, self.T, d)
        align_x = self.message_fc(align_x)
        align_x = self.message_ln(align_x)
        cls_token = align_x[:1, :, :, :]
        align_x = rearrange(align_x, "l b t c -> b t l c", l=l, b=B, t=self.T, c=d)
        support = align_x[:, :-1, :, :]
        query = align_x[:, 1:, :, :]
        pairs, aligned_frame, interactive_feature = self.message_attn(support, query, cls_token)
        cos_loss = cos_similarity_loss(pairs[:, :, :, :self.in_feature_dim], pairs[:, :, :, self.in_feature_dim:])
        feature["cosine"].append(cos_loss)
        aligned_frame = rearrange(aligned_frame, "b t l d -> l (b t) d", l=l, b=B, t=self.T, d=d)
        x = raw_x + self.drop_path(self.temporal_skip_align_fc(aligned_frame))
        x = x + self.drop_path(self.temporal_skip_fc(interactive_feature))

        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x[:l, :, :]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))

        feature["input"] = x

        return feature


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                 use_checkpoint=False, T=8, input_resolution=224, patch_size=16, ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[MultiAxisAttentionBlock(width, heads, attn_mask, droppath[i], T, input_resolution, patch_size, idx=i) for i in
              range(layers)])

    def forward(self, x: torch.Tensor):
        feature = {}
        cos_loss_list = []
        feature["input"] = x
        feature["cosine"] = cos_loss_list
        if not self.use_checkpoint:
            return self.resblocks(feature)
        else:
            return checkpoint_sequential(self.resblocks, 3, feature)


class MultiAxisTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 droppath=None, T=8, use_checkpoint=False, ):
        super().__init__()
        self.T = T
        self.W = input_resolution // patch_size
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T, input_resolution=input_resolution, patch_size=patch_size)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)

        feature = self.transformer(x)
        x = feature["input"]
        x = x.permute(1, 0, 2)

        cls_x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_x = cls_x @ self.proj

        return cls_x, x[:, 1:, :], feature["cosine"]
