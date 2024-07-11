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

class E2TTS(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        assert divisible_by(depth, 2), 'depth needs to be even'

        self.depth = depth
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(dim = dim, **attn_kwargs)
            ff = FeedForward(dim = dim, **ff_kwargs)

            self.layers.append(ModuleList([
                attn,
                ff
            ]))

    def forward(
        self,
        x
    ):

        skips = []

        for ind, (attn, ff) in enumerate(self.layers):
            layer = ind + 1

            # skip connection logic
            # first do additive

            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half:
                x = x + skips.pop()

            # attention and feedforward blocks

            x = attn(x) + x
            x = ff(x) + x

        assert len(skips) == 0

        return x
