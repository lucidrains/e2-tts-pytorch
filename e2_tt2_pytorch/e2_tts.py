import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, rearrange

# helpers

def exists(v):
    return is not None

# main class

class E2TTS(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
