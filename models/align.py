import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import numpy as np


def aligned_mask_generation(point, resolution):
    L = resolution
    shape = point.size()[:-1]
    point = point.reshape(-1, 1, 2)
    N = point.size()[0]
    element = torch.arange(0, L).to(point.device) / (L - 1) * 2 - 1
    element = element.reshape(1, L, 1).expand(N, -1, 2)
    element = element - point
    first_condition = (-0.25 <= element) * (element <= 0.25) * 1.0
    second_condition = 1 - (element.abs() - 0.25) * 1.5
    second_condition = (1 - first_condition) * second_condition * (second_condition >= 0)
    second_condition = torch.nn.functional.relu(second_condition)
    aligned_mask_xy = first_condition + second_condition
    aligned_mask_x, aligned_mask_y = aligned_mask_xy[..., 0], aligned_mask_xy[..., 1]
    mask = aligned_mask_y.unsqueeze(-1) * aligned_mask_x.unsqueeze(-2)
    mask = mask.reshape(*shape, resolution, resolution)
    return mask


class SqueezeAndRerange(torch.nn.Module):
    def __init__(self, *dim, T):
        super().__init__()
        self.T = T - 1
        if all(v >= 0 for v in dim):
            self.dim = sorted(dim, reverse=True)
        elif all(v < 0 for v in dim):
            self.dim = sorted(dim)

    def forward(self, x):
        for d in self.dim:
            x = torch.squeeze(x, dim=d)

        bt, d = x.size()
        x = rearrange(x, "(b t) c -> b c t", b=bt // self.T, t=self.T, c=d)
        return x


class Depth_Separable_Convolution(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Depth_Separable_Convolution, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chs,
                out_channels=in_chs,
                kernel_size=(3, 3),
                stride=(1, 1),
                groups=in_chs,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(num_features=in_chs),
            nn.ReLU(),
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chs,
                out_channels=out_chs,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=1,
                padding=(0, 0),
            ),
            nn.BatchNorm2d(num_features=out_chs),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.depthwise_conv(x)
        out = self.pointwise_conv(x)
        return out


class ILA(nn.Module):
    def __init__(self, T=8, d_model=768, patch_size=16, input_resolution=224, is_training=True):
        super().__init__()
        self.T = T
        self.W = input_resolution // patch_size
        self.d_model = d_model
        self.is_training = is_training

        # k400
        self.interactive_block = nn.Sequential(
            Depth_Separable_Convolution(self.d_model * 2, 256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, self.d_model)
        self.conv = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.prediction_block = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            SqueezeAndRerange(-2, -1, T=self.T),
            nn.Conv1d(128, 64, 1, groups=2),
            nn.ReLU(),
            nn.Conv1d(64, 4, 1, groups=2),
            nn.Tanh()
        )
        self.prediction_block[-2].weight.data.zero_()
        self.prediction_block[-2].bias.data.zero_()

        # Something-Something v2
        # self.interactive_block = nn.Sequential(
        #     nn.Conv2d(self.d_model * 2, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        # )
        # self.conv = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        # self.fc = nn.Linear(128, self.d_model)
        #
        # self.prediction_block = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.MaxPool2d((2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.MaxPool2d((2, 2)),
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool2d((1, 1)),
        #     SqueezeAndRerange(-2, -1, T=self.T),
        #     nn.Conv1d(128, 64, 1, groups=2),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 4, 1, groups=2),
        #     nn.Tanh()
        # )
        # self.prediction_block[-2].weight.data.zero_()
        # self.prediction_block[-2].bias.data.zero_()

        delta = 0.2
        self.disturbance_offset = torch.tensor([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]).float() * delta
        self.disturbance_offset = self.disturbance_offset.reshape(1, 1, 9, 2)

    def alignment_parameters(self):
        yield from self.prediction_block.parameters()

    def align_decay(self):
        with torch.no_grad():
            self.disturbance_offset *= 0.5

    def forward(self, pre, cur, cls_token):
        b, t, l, d = pre.size()
        B = b
        T = self.T-1
        W = self.W
        H = W
        D = d
        pre_cls_token = pre[:, :, :1, :]
        cur_cls_token = cur[:, :, :1, :]
        pre = pre[:, :, 1:, :]
        cur = cur[:, :, 1:, :]
        pre_projed = rearrange(pre, "b t (h w) d -> b d t h w", b=B, t=T, d=D, h=H, w=W)
        cur_projed = rearrange(cur, "b t (h w) d -> b d t h w", b=B, t=T, d=D, h=H, w=W)
        pairs = torch.cat([pre_projed, cur_projed], dim=-4)
        pairs = pairs.reshape(-1, D * 2, T, H, W)

        pairs = rearrange(pairs, "b d t h w -> (b t) d h w", b=B, t=T, d=D*2, h=H, w=W)

        interactive_feature = self.interactive_block(pairs)
        points = self.prediction_block(interactive_feature).transpose(1, 2)
        points = points * 0.75

        pre_interactive_feature = interactive_feature[:, :128, :, :]
        cur_interactive_feature = interactive_feature[:, 128:, :, :]
        pre_interactive_feature = rearrange(pre_interactive_feature, "(b t) d h w -> b d t h w", b=B, t=T, d=128, h=H, w=W)
        cur_interactive_feature = rearrange(cur_interactive_feature, "(b t) d h w -> b d t h w", b=B, t=T, d=128, h=H, w=W)
        first_interactive_feature = pre_interactive_feature[:, :, :1, :, :]
        rest_interactive_feature = (pre_interactive_feature[:, :, 1:, :, :]+cur_interactive_feature[:, :, :-1, :, :])/2
        last_interactive_feature = cur_interactive_feature[:, :, -1:, :, :]
        interactive_feature = torch.cat([first_interactive_feature, rest_interactive_feature, last_interactive_feature], dim=2)
        interactive_feature = self.conv(interactive_feature)
        interactive_feature = rearrange(interactive_feature, "b d t h w -> (h w) b t d", b=B, t=T+1, d=128, h=H, w=W)
        interactive_feature = self.fc(interactive_feature)
        interactive_feature = torch.cat([cls_token, interactive_feature], dim=0)
        interactive_feature = rearrange(interactive_feature, "l b t d -> l (b t) d", b=b, t=self.T, d=self.d_model, l=H*W+1)

        if self.is_training:
            point_first_pair = points[:, :, :2]
            point_second_pair = points[:, :, 2:]
            point_first_pair = point_first_pair.unsqueeze(2) + self.disturbance_offset.to(point_first_pair.device)
            point_second_pair = point_second_pair.unsqueeze(2) + self.disturbance_offset.to(point_second_pair.device)

        else:
            point_first_pair = points[:, :, :2]
            point_second_pair = points[:, :, 2:]
            point_first_pair = point_first_pair.unsqueeze(2)
            point_second_pair = point_second_pair.unsqueeze(2)

        aligned_mask = aligned_mask_generation(point_first_pair, H)
        aligned_mask = aligned_mask.mean(2)
        aligned_mask = aligned_mask.reshape(B, 1, T, H, W)
        pre_aligned = aligned_mask * pre_projed
        raw_pre_aligned = pre_aligned
        raw_pre_aligned = rearrange(raw_pre_aligned, "b d t h w -> b t (h w) d", b=B, t=T, d=D, h=H, w=W)
        raw_pre_aligned = torch.cat([pre_cls_token, raw_pre_aligned], dim=2)
        pre_aligned = pre_aligned.sum([-1, -2]).unsqueeze(-1)
        pre_aligned = rearrange(pre_aligned, "b d t l -> l b t d", b=B, t=T, l=1, d=D)

        aligned_mask = aligned_mask_generation(point_second_pair, H)
        aligned_mask = aligned_mask.mean(2)
        aligned_mask = aligned_mask.reshape(B, 1, T, H, W)
        cur_aligned = aligned_mask * cur_projed
        raw_cur_aligned = cur_aligned
        raw_cur_aligned = rearrange(raw_cur_aligned, "b d t h w -> b t (h w) d", b=B, t=T, d=D, h=H, w=W)
        raw_cur_aligned = torch.cat([cur_cls_token, raw_cur_aligned], dim=2)
        cur_aligned = cur_aligned.sum([-1, -2]).unsqueeze(-1)
        cur_aligned = rearrange(cur_aligned, "b d t l -> l b t d", b=B, t=T, l=1, d=D)

        aligned_frame_first = raw_pre_aligned[:, :1, :, :]
        aligned_frame_rest = (raw_pre_aligned[:, 1:, :, :]+raw_cur_aligned[:, :-1, :, :])/2
        aligned_frame_last = raw_cur_aligned[:, -1:, :, :]
        aligned_frame = torch.cat([aligned_frame_first, aligned_frame_rest, aligned_frame_last], dim=1)

        pair = torch.cat([pre_aligned, cur_aligned], dim=3)

        return pair, aligned_frame, interactive_feature


if __name__ == '__main__':
    model = ILA()
    x = torch.randn(8, 7, 197, 768)
    cls_token = torch.randn(1, 8, 8, 768)
    out, frame, feature = model(x, x, cls_token)
    print(out.size())
    print(frame.size())
    print(feature.size())