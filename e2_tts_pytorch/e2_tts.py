"""
ein notation:
b - batch
n - sequence
d - dimension
"""

from __future__ import annotations
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from torchdiffeq import odeint

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_transformers import (
    Attention,
    FeedForward,
    RMSNorm,
    AdaptiveRMSNorm
)

from e2_tts_pytorch.tensor_typing import (
    Float,
    Int,
    Bool
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def lens_to_mask(
    t: Int['b'],
    length: int | None = None
) -> Bool['b n']:

    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return einx.less('n, b -> b n', seq, t)

def maybe_masked_mean(
    t: Float['b n d'],
    mask: Bool['b n'] = None
) -> Float['b d']:

    if not exists(mask):
        return t.mean(dim = 1)

    t = einx.where('b n, b n d, -> b n d', mask, t, 0.)
    num = reduce(t, 'b n d -> b d', 'sum')
    den = reduce(mask.float(), 'b n -> b', 'sum')

    return einx.divide('b d, b -> b d', num, den.clamp(min = 1.))

# attention and transformer backbone
# for use in both e2tts as well as duration module

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        cond_on_time = False,
        skip_connect_type: Literal['add', 'concat', 'none'] = 'concat',
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        assert divisible_by(depth, 2), 'depth needs to be even'

        self.dim = dim
        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == 'concat'

        self.depth = depth
        self.layers = ModuleList([])

        # time conditioning
        # will use adaptive rmsnorm

        self.cond_on_time = cond_on_time
        rmsnorm_klass = RMSNorm if not cond_on_time else AdaptiveRMSNorm

        self.time_cond_mlp = nn.Identity()

        if cond_on_time:
            self.time_cond_mlp = nn.Sequential(
                Rearrange('... -> ... 1'),
                nn.Linear(1, dim),
                nn.SiLU()
            )

        for _ in range(depth):
            attn_norm = rmsnorm_klass(dim)
            attn = Attention(dim = dim, **attn_kwargs)

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim = dim, **ff_kwargs)

            skip_proj = nn.Linear(dim * 2, dim, bias = False) if needs_skip_proj else None

            self.layers.append(ModuleList([
                skip_proj,
                attn_norm,
                attn,
                ff_norm,
                ff,
            ]))

        self.final_norm = rmsnorm_klass(dim)

    def forward(
        self,
        x: Float['b n d'],
        times: Int['b'] | None = None,
        mask: Bool['b n'] | None = None
    ):
        assert not (exists(times) ^ self.cond_on_time), '`times` must be passed in if `cond_on_time` is set to `True` and vice versa'

        # handle adaptive rmsnorm kwargs

        norm_kwargs = dict()

        if exists(times):
            times = self.time_cond_mlp(times)
            norm_kwargs.update(condition = times)

        # skip connection related stuff

        skip_connect_type = self.skip_connect_type

        skips = []

        # go through the layers

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

            x = attn(attn_norm(x, **norm_kwargs)) + x
            x = ff(ff_norm(x, **norm_kwargs)) + x

        assert len(skips) == 0

        return self.final_norm(x, **norm_kwargs)

# main classes

class DurationPredictor(Module):
    def __init__(
        self,
        transformer: dict | Transformer
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(
                **transformer,
                cond_on_time = False
            )

        self.transformer = transformer

        dim = transformer.dim

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1, bias = False),
            nn.Softplus(),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        x: Float['b n d'],
        *,
        lens: Int['b'] = None,
        mask: Bool['b n'] = None,
        target_duration: Int['b'] = None
    ):
        seq_len = x.shape[1]

        assert not (exists(lens) and exists(mask))

        if exists(lens):
            mask = lens_to_mask(lens, length = seq_len)

        x = self.transformer(x, mask = mask)

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        if not exists(target_duration):
            return pred

        return F.mse_loss(pred, target_duration.float())

class E2TTS(Module):
    def __init__(
        self,
        sigma = 0.,
        transformer: dict | Transformer = None,
        duration_predictor: dict | DurationPredictor | None = None,
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        )
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(
                **transformer,
                cond_on_time = True
            )

        if isinstance(duration_predictor, dict):
            duration_predictor = DurationPredictor(**duration_predictor)

        self.transformer = transformer
        self.duration_predictor = duration_predictor

        dim = transformer.dim
        self.to_pred = nn.Linear(dim, dim)

        # conditional flow related

        self.sigma = sigma

        # sampling

        self.odeint_kwargs = odeint_kwargs

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond,
        *,
        mask: Bool['b n'] | None = None,
        steps = 3
    ):
        self.eval()
        batch = cond.shape[0]

        # neural ode

        def fn(t, x):
            t = repeat(t, '-> b', b = batch)

            return self.transformer(
                cond,
                times = t,
                mask = mask
            )

        y0 = torch.randn_like(cond)
        t = torch.linspace(0, 1, steps, device = self.device)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]

        return sampled

    def forward(
        self,
        inp: Float['b n d'], # is mel in paper
        times: Int['b'] | None = None,
        lens: Int['b'] | None = None,
        mask: Bool['b n'] | None = None
    ):
        batch, seq_len, dtype, σ = *inp.shape[:2], inp.dtype, self.sigma

        assert not (exists(lens) and exists(mask))

        if exists(lens):
            mask = lens_to_mask(lens, length = seq_len)
        elif exists(mask):
            lens = mask.sum(dim = -1).long()

        # get a random span to mask out for training conditionally

        random_span_frac_indices = inp.new_zeros(2, batch).uniform_(0, 1)
        rand_span_indices = (random_span_frac_indices * default(lens, seq_len)).long()
        rand_span_indices = rand_span_indices.sort(dim = 0).values

        seq = torch.arange(seq_len, device = self.device)
        start, end = rand_span_indices[..., None]
        rand_span_mask = (seq >= start) & (seq <= end)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1

        x1 = inp

        # main conditional flow training logic
        # just 4 loc

        # x0 is gaussian noise

        x0 = torch.randn_like(x1)

        # t is random times from above

        times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        # sample xt (w in the paper)

        w = (1 - (1 - σ) * t) * x0 + t * x1

        flow = x1 - (1 - σ) * x0

        # only predict what is within the random mask span for infilling

        w = torch.where(
            rand_span_mask[..., None],
            w, x1
        )

        # transformer and prediction head

        attended = self.transformer(
            w,
            times = times,
            mask = mask
        )

        pred = self.to_pred(attended)

        # flow matching loss

        loss = F.mse_loss(pred, flow, reduction = 'none')

        loss = loss[rand_span_mask]

        return loss.mean()
