import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, repeat
import torchvision

from models.mat import MultiAxisTransformer


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        x = self.net(x)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        bt, c, h, w = x.size()
        n_batch = bt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        fold = c // fold_div
        if inplace:
            raise NotImplementedError
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(bt, c, h, w)


class TemporalShiftVit(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShiftVit, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

        # self.residual_fc = nn.Linear(d_model, d_model)
        # self.residual_fc.weight.data.zero_()
        # self.residual_fc.bias.data.zero_()

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        x = self.net(x)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        hw, bt, d = x.size()
        cls_tokens = x[0, :, :].unsqueeze(0)
        x = x[1:, :, :]
        # x = x.permute(1, 2, 0)  # bt, d, hw
        x = rearrange(x, "l bt d -> bt d l", bt=bt, l=hw-1, d=d)
        n_batch = bt // n_segment
        h = int(np.sqrt(hw-1))
        w = h
        x = rearrange(x, "(b t) d (h w) -> b t d h w", b=n_batch, t=n_segment, d=d, h=h, w=w)
        # x = x.contiguous().view(n_batch, n_segment, d, h, w)
        fold = d // fold_div
        if inplace:
            raise NotImplementedError
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        # out = out.contiguous().view(bt, d, h*w)
        # out = out.permute(2, 0, 1)
        out = rearrange(out, "b t d h w -> (h w) (b t) d", b=n_batch, t=n_segment, d=d, h=h, w=w)
        out = torch.cat([cls_tokens, out], dim=0)
        return out


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        bt, d, h, w = x.size()
        n_batch = bt // n_segment
        x = x.view(n_batch, n_segment, d, h, w).transpose(1, 2)   # b, d, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(bt // 2, d, h, w)
        return x


def make_temporal_shift_vit(net, n_segment, n_div=8, place='block', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment]*4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    if isinstance(net, MultiAxisTransformer):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShiftVit(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.transformer.resblocks = make_block_temporal(net.transformer.resblocks, n_segment_list[0])
    else:
        raise NotImplementedError(place)



