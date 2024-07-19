"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Literal, List, Callable
from random import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nn.utils.rnn import pad_sequence

import torchaudio
from torchdiffeq import odeint

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from x_transformers import (
    Attention,
    FeedForward,
    RMSNorm,
    AdaptiveRMSNorm
)

from x_transformers.x_transformers import RotaryEmbedding

from gateloop_transformer import SimpleGateLoopLayer

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

# simple utf-8 tokenizer, since paper went character based

def list_str_to_tensor(
    text: List[str],
    padding_value = -1
) -> Int['b nt']:

    list_tensors = [torch.tensor([*bytes(t, 'UTF-8')]) for t in text]
    text = pad_sequence(list_tensors, padding_value = -1, batch_first = True)
    return text

# tensor helpers

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

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

# to mel spec

class MelSpec(Module):
    def __init__(
        self,
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 80,
        mel_fmin = 0,
        mel_fmax = 8000,
        sampling_rate = 22050,
        normalize = False,
        power = 2,
        norm = "slaney"
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            power=power,
            normalized=normalize,
            sample_rate=sampling_rate,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_mels=n_mel_channels,
            norm=norm,
        )

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = rearrange(inp, 'b 1 nw -> b nw')

        assert len(inp.shape) == 2

        if self.dummy.device != inp.device:
            self.to(inp.device)

        mel = self.mel_stft(inp)
        mel = log(mel)
        return mel

# character embedding

class CharacterEmbed(Module):
    def __init__(
        self,
        dim,
        num_embeds = 256,
        cond_drop_prob = 0.
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_embeds + 1, dim) # will just use 0 as the 'filler token'
        self.combine = nn.Linear(dim * 2, dim)
        self.cond_drop_prob = cond_drop_prob

    def forward(
        self,
        x: Float['b n d'],
        text: Int['b n'],
        drop_text_cond = None
    ):
        drop_text_cond = default(drop_text_cond, self.training and random() < self.cond_drop_prob)

        if drop_text_cond:
            return x

        max_seq_len = x.shape[1]
        text_mask = text == -1

        text = text + 1 # use 0 as filler token
        text = text.masked_fill(text_mask, 0)

        text = text[:, :max_seq_len] # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value = 0)
        text_embed = self.embed(text)

        concatted = torch.cat((x, text_embed), dim = -1)
        assert x.shape[-1] == text_embed.shape[-1] == self.dim, f'expected {self.dim} but received ({x.shape[-1]}, {text_embed.shape[-1]})'
        return self.combine(concatted)

# attention and transformer backbone
# for use in both e2tts as well as duration module

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth = 8,
        cond_on_time = False,
        skip_connect_type: Literal['add', 'concat', 'none'] = 'concat',
        abs_pos_emb = True,
        max_seq_len = 8192,
        heads = 8,
        dim_head = 64,
        num_gateloop_layers = 1,
        dropout = 0.1,
        attn_kwargs: dict = dict(
            gate_value_heads = True
        ),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        assert divisible_by(depth, 2), 'depth needs to be even'

        # absolute positional embedding

        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if abs_pos_emb else None

        self.dim = dim
        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == 'concat'

        self.depth = depth
        self.layers = ModuleList([])

        # rotary embedding

        self.rotary_emb = RotaryEmbedding(dim_head)

        # gateloops

        self.gateloops = ModuleList([SimpleGateLoopLayer(dim = dim) for _ in range(num_gateloop_layers)])

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
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, **attn_kwargs)

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim = dim, glu = True, dropout = dropout, **ff_kwargs)

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
        times: Float['b'] | Float[''] | None = None,
        mask: Bool['b n'] | None = None
    ):
        batch, seq_len, device = *x.shape[:2], x.device

        assert not (exists(times) ^ self.cond_on_time), '`times` must be passed in if `cond_on_time` is set to `True` and vice versa'

        # gateloop layers

        for gateloop in self.gateloops:
            x = gateloop(x) + x

        # handle absolute positions if needed

        if exists(self.abs_pos_emb):
            assert seq_len <= self.max_seq_len, f'{seq_len} exceeds the set `max_seq_len` ({self.max_seq_len}) on Transformer'
            seq = torch.arange(seq_len, device = device)
            x = x + self.abs_pos_emb(seq)

        # handle adaptive rmsnorm kwargs

        norm_kwargs = dict()

        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b = batch)

            times = self.time_cond_mlp(times)
            norm_kwargs.update(condition = times)

        # rotary embedding

        rotary_pos_emb = self.rotary_emb.forward_from_seq_len(seq_len)

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

            x = attn(attn_norm(x, **norm_kwargs), rotary_pos_emb = rotary_pos_emb, mask = mask) + x
            x = ff(ff_norm(x, **norm_kwargs)) + x

        assert len(skips) == 0

        return self.final_norm(x, **norm_kwargs)

# main classes

class DurationPredictor(Module):
    def __init__(
        self,
        transformer: dict | Transformer,
        text_num_embeds = 256,
        mel_spec_kwargs: dict = dict()
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(
                **transformer,
                cond_on_time = False
            )

        self.transformer = transformer
        dim = transformer.dim

        self.dim = dim
        self.embed_text = CharacterEmbed(dim, num_embeds = text_num_embeds)

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1, bias = False),
            nn.Softplus(),
            Rearrange('... 1 -> ...')
        )

        # mel spec

        self.mel_spec = MelSpec(**mel_spec_kwargs)

    def forward(
        self,
        x: Float['b n d'] | Float['b nw'],
        *,
        text: Int['b n'] | List[str] | None = None,
        lens: Int['b'] | None = None,
        return_loss = True
    ):
        # raw wave

        if x.ndim == 2:
            x = self.mel_spec(x)
            x = rearrange(x, 'b d n -> b n d')
            assert x.shape[-1] == self.dim

        batch, seq_len, device = *x.shape[:2], x.device

        # text

        if exists(text):
            if isinstance(text, list):
                text = list_str_to_tensor(text).to(device)
                assert text.shape[0] == batch

            x = self.embed_text(x, text)

        # handle lengths (duration)

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device = device)

        mask = lens_to_mask(lens, length = seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = x.new_zeros(batch).uniform_(0, 1)
            rand_index = (rand_frac_index * lens).long()

            seq = torch.arange(seq_len, device = device)
            mask &= einx.less('n, b -> b n', seq, rand_index)

        # attending

        x = self.transformer(x, mask = mask)

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        # return the prediction if not returning loss

        if not return_loss:
            return pred

        # loss

        return F.mse_loss(pred, lens.float())

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
        ),
        text_num_embeds = 256,
        cond_drop_prob = 0.25,
        mel_spec_module: Module | None = None,
        mel_spec_kwargs: dict = dict(),
        immiscible = False
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

        dim = transformer.dim
        self.dim = dim
        self.to_pred = nn.Linear(dim, dim)

        self.embed_text = CharacterEmbed(dim, num_embeds = text_num_embeds, cond_drop_prob = cond_drop_prob)

        self.duration_predictor = duration_predictor

        # conditional flow related

        self.sigma = sigma

        # sampling

        self.odeint_kwargs = odeint_kwargs

        # mel spec

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))

        # immiscible (diffusion / flow)

        self.immiscible = immiscible

    @property
    def device(self):
        return next(self.parameters()).device

    def transformer_with_pred_head(
        self,
        x: Float['b n d'],
        times: Float['b'],
        mask: Bool['b n'] | None = None,
        text: Int['b nt'] | None = None,
        drop_text_cond: bool | None = None
    ):

        if exists(text):
            x = self.embed_text(x, text, drop_text_cond = drop_text_cond)

        attended = self.transformer(
            x,
            times = times,
            mask = mask
        )

        return self.to_pred(attended)

    def cfg_transformer_with_pred_head(
        self,
        *args,
        cfg_strength: float = 1.,
        **kwargs,
    ):
        
        pred = self.transformer_with_pred_head(*args, drop_text_cond = False, **kwargs)

        if cfg_strength < 1e-5:
            return pred

        null_pred = self.transformer_with_pred_head(*args, drop_text_cond = True, **kwargs)

        return pred + (pred - null_pred) * cfg_strength

    @torch.no_grad()
    def sample(
        self,
        cond: Float['b n d'] | Float['b nw'],
        *,
        text: Int['b n'] | List[str] | None = None,
        lens: Int['b'] | None = None,
        duration: int | Int['b'] | None = None,
        steps = 3,
        cfg_strength = 1.,   # they used a classifier free guidance strenght of 1.
        max_duration = 4096, # in case the duration predictor goes haywire
        vocoder: Callable[Float['b d n'], Float['b nw']] | None = None
    ):
        self.eval()

        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = rearrange(cond, 'b d n -> b n d')
            assert cond.shape[-1] == self.dim

        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        # text

        if isinstance(text, list):
            text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device = device, dtype = torch.long)

        cond_mask = lens_to_mask(lens)

        if exists(duration):
            if isinstance(duration, int):
                duration = torch.full((batch,), duration, device = device, dtype = torch.long)

        elif exists(self.duration_predictor):
            duration = self.duration_predictor(cond, lens = lens).long()

        duration = torch.maximum(lens + 1, duration) # just add one token so something is generated
        duration = duration.clamp(max = max_duration)
        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value = 0.)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_seq_len), value = False)

        mask = lens_to_mask(duration)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed

            x = torch.where(cond_mask[..., None], cond, x)

            # predict flow

            return self.cfg_transformer_with_pred_head(
                x,
                times = t,
                text = text,
                mask = mask,
                cfg_strength = cfg_strength
            )

        y0 = torch.randn_like(cond)
        t = torch.linspace(0, 1, steps, device = self.device)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]

        out = sampled

        if exists(vocoder):
            out = rearrange(out, 'b n d -> b d n')
            out = vocoder(out)

        return out

    def forward(
        self,
        inp: Float['b n d'] | Float['b nw'], # mel or raw wave
        *,
        text: Int['b nt'] | List[str] | None = None,
        times: Int['b'] | None = None,
        lens: Int['b'] | None = None,
    ):
        # handle raw wave

        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, 'b d n -> b n d')
            assert inp.shape[-1] == self.dim

        batch, seq_len, dtype, device, σ = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string

        if isinstance(text, list):
            text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device = device)

        mask = lens_to_mask(lens, length = seq_len)

        # get a random span to mask out for training conditionally

        random_span_frac_indices = inp.new_zeros(2, batch).uniform_(0, 1)
        rand_span_indices = (random_span_frac_indices * default(lens, seq_len)).long()
        rand_span_indices = rand_span_indices.sort(dim = 0).values

        seq = torch.arange(seq_len, device = device)
        start, end = rand_span_indices[..., None]
        rand_span_mask = (seq >= start) & (seq <= end)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1

        x1 = inp

        # main conditional flow training logic
        # just ~5 loc

        # x0 is gaussian noise

        x0 = torch.randn_like(x1)

        # whether to use immiscible flow

        if self.immiscible:
            cost = torch.cdist(x1.flatten(1), x0.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost)
            x0 = x0[reorder_indices]

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

        pred = self.transformer_with_pred_head(w, times = times, text = text)

        # flow matching loss

        loss = F.mse_loss(pred, flow, reduction = 'none')

        loss = loss[rand_span_mask]

        return loss.mean()
