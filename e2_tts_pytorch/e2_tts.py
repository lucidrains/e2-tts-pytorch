from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import einsum, rearrange

from x_transformers import (
    Attention,
    FeedForward,
    RMSNorm,
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main class

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        skip_connect_type: Literal['add', 'concat', 'none'] = 'concat',
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        assert divisible_by(depth, 2), 'depth needs to be even'

        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == 'concat'

        self.depth = depth
        self.layers = ModuleList([])

        for _ in range(depth):
            attn_norm = RMSNorm(dim)
            attn = Attention(dim = dim, **attn_kwargs)

            ff_norm = RMSNorm(dim)
            ff = FeedForward(dim = dim, **ff_kwargs)

            skip_proj = nn.Linear(dim * 2, dim, bias = False) if needs_skip_proj else None

            self.layers.append(ModuleList([
                skip_proj,
                attn_norm,
                attn,
                ff_norm,
                ff,
            ]))

        self.to_pred = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(
        self,
        x,
        mask = None,
        target = None
    ):
        skip_connect_type = self.skip_connect_type

        skips = []

        for ind, (maybe_skip_proj, attn_norm, attn, ff_norm, ff) in enumerate(self.layers):
            layer = ind + 1

            # skip connection logic

            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half:
                skip = skips.pop()

                if skip_connect_type == 'concat':
                    # concatenative
                    x = torch.cat((x, skip), dim = -1)
                    x = maybe_skip_proj(x)
                elif skip_connect_type == 'add':
                    # additive
                    x = x + skip

            # attention and feedforward blocks

            x = attn(attn_norm(x)) + x
            x = ff(ff_norm(x)) + x

        assert len(skips) == 0

        pred = self.to_pred(x)

        if not exists(target):
            return pred

        return F.mse_loss(pred, target)
