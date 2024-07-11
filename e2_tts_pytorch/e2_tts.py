import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

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

# main class

class E2TTS(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
