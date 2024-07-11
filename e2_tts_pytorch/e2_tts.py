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

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x
