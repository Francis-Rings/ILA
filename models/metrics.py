import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def timewise_cos(x, y):
    l, b, t, c = x.size()
    x = rearrange(x, "l b t c -> b t l c", b=b, t=t, l=l, c=c)
    y = rearrange(y, "l b t c -> b t l c", b=b, t=t, l=l, c=c)
    x = x.squeeze()
    y = y.squeeze()
    x = F.normalize(x.reshape(b, t, -1), dim=-1, p=2)
    y = F.normalize(y.reshape(b, t, -1), dim=-1, p=2)
    loss = (1-(x*y).sum(-1)).sum(-1).sum(-1)
    return loss


def cos_similarity_loss(x, y):
    l, b, t, c = x.size()
    x = rearrange(x, "l b t c -> (b t) l c", b=b, t=t, l=l, c=c)
    y = rearrange(y, "l b t c -> (b t) l c", b=b, t=t, l=l, c=c)
    x = x.squeeze()
    y = y.squeeze()
    x = F.normalize(x.reshape(b*t, -1), dim=-1, p=2)
    y = F.normalize(y.reshape(b*t, -1), dim=-1, p=2)
    loss_fn = nn.CosineEmbeddingLoss(reduction='mean')
    loss_flag = torch.ones([b*t])
    loss_flag = loss_flag.to(x.device)
    loss = loss_fn(x, y, loss_flag)
    return loss


if __name__ == '__main__':
    x = torch.ones(1, 8, 16, 768)
    y = torch.ones(1, 8, 16, 768)
    out = timewise_cos(x, y)
    print(out)
    out = cos_similarity_loss(x, y)
    print(out)