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
from torch import nn, from_numpy
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential, Linear
from torch.nn.utils.rnn import pad_sequence

import torchaudio
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

def mask_from_start_end_indices(
    seq_len: Int['b'],
    start: Int['b'],
    end: Int['b']
):
    max_seq_len = seq_len.max().item()  
    seq = torch.arange(max_seq_len, device = start.device).long()
    return einx.greater_equal('n, b -> b n', seq, start) & einx.less('n, b -> b n', seq, end)

def mask_from_frac_lengths(
    seq_len: Int['b'],
    frac_lengths: Float['b']
):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min = 0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

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
        n_mel_channels = 100,
        sampling_rate = 24_000,
        normalize = False,
        power = 1,
        norm = None,
        center = True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate = sampling_rate,
            n_fft = filter_length,
            win_length = win_length,
            hop_length = hop_length,
            n_mels = n_mel_channels,
            power = power,
            center = center,
            normalized = normalize,
            norm = norm,
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
        cond_drop_prob = 0.,
        num_gateloop_layers = 0
    ):
        super().__init__()
        self.dim = dim
        self.cond_drop_prob = cond_drop_prob

        self.embed = nn.Embedding(num_embeds + 1, dim) # will just use 0 as the 'filler token'

        self.gateloops = ModuleList([Sequential(Linear(dim * 3, dim * 3, bias = False), SimpleGateLoopLayer(dim = dim * 3)) for _ in range(num_gateloop_layers)])

        self.to_cond_gamma_beta = Linear(dim * 3, dim * 2)

        nn.init.zeros_(self.to_cond_gamma_beta.weight)
        nn.init.zeros_(self.to_cond_gamma_beta.bias)

    def forward(
        self,
        x: Float['b n d'],
        cond: Float['b n d'],
        text: Int['b n'],
        drop_text_cond = None
    ):
        drop_text_cond = default(drop_text_cond, self.training and random() < self.cond_drop_prob)

        if drop_text_cond:
            return x

        max_seq_len = x.shape[1]

        text = text + 1 # shift all other token ids up by 1 and use 0 as filler token

        text = text[:, :max_seq_len] # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value = 0)

        text_embed = self.embed(text)

        concatted = torch.cat((x, cond, text_embed), dim = -1)

        for gateloop in self.gateloops:
            concatted = gateloop(concatted) + concatted

        assert x.shape[-1] == text_embed.shape[-1] == self.dim, f'expected {self.dim} but received ({x.shape[-1]}, {text_embed.shape[-1]})'

        gamma, beta = self.to_cond_gamma_beta(concatted).chunk(2, dim = -1)
        return x * (gamma + 1.) + beta

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
            gate_value_heads = True,
            softclamp_logits = True,
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
            self.time_cond_mlp = Sequential(
                Rearrange('... -> ... 1'),
                Linear(1, dim),
                nn.SiLU()
            )

        for ind in range(depth):
            is_later_half = ind >= (depth // 2)

            attn_norm = rmsnorm_klass(dim)
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, **attn_kwargs)

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim = dim, glu = True, dropout = dropout, **ff_kwargs)

            skip_proj = Linear(dim * 2, dim, bias = False) if needs_skip_proj and is_later_half else None

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
        num_channels = None,
        mel_spec_kwargs: dict = dict(),
        char_embed_kwargs: dict = dict(
            num_gateloop_layers = 2
        )
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(
                **transformer,
                cond_on_time = False
            )

        # mel spec

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = default(num_channels, self.mel_spec.n_mel_channels)

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        self.proj_in = Linear(self.num_channels, self.dim)

        self.embed_text = CharacterEmbed(dim, num_embeds = text_num_embeds, **char_embed_kwargs)

        self.to_pred = Sequential(
            Linear(dim, 1, bias = False),
            nn.Softplus(),
            Rearrange('... 1 -> ...')
        )

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

        x = self.proj_in(x)

        batch, seq_len, device = *x.shape[:2], x.device

        # text

        if exists(text):
            if isinstance(text, list):
                text = list_str_to_tensor(text).to(device)
                assert text.shape[0] == batch

            x = self.embed_text(x, x, text)

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
        num_channels = None,
        mel_spec_module: Module | None = None,
        char_embed_kwargs: dict = dict(
            num_gateloop_layers = 2
        ),
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: Tuple[float, float] = (0.7, 1.)
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

        self.frac_lengths_mask = frac_lengths_mask

        self.embed_text = CharacterEmbed(dim, num_embeds = text_num_embeds, cond_drop_prob = cond_drop_prob, **char_embed_kwargs)

        self.duration_predictor = duration_predictor

        # conditional flow related

        self.sigma = sigma

        # sampling

        self.odeint_kwargs = odeint_kwargs

        # mel spec

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
 
        self.num_channels = num_channels

        self.proj_in = Linear(num_channels, dim)
        self.cond_proj_in = Linear(num_channels, dim)
        self.to_pred = Linear(dim, num_channels)

    @property
    def device(self):
        return next(self.parameters()).device

    def transformer_with_pred_head(
        self,
        x: Float['b n d'],
        cond: Float['b n d'],
        times: Float['b'],
        mask: Bool['b n'] | None = None,
        text: Int['b nt'] | None = None,
        drop_text_cond: bool | None = None
    ):
        x = self.proj_in(x)
        cond = self.cond_proj_in(cond)

        if exists(text):
            x = self.embed_text(x, cond, text, drop_text_cond = drop_text_cond)

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
        steps = 32,
        cfg_strength = 1.,   # they used a classifier free guidance strength of 1.
        max_duration = 4096, # in case the duration predictor goes haywire
        vocoder: Callable[Float['b d n'], Float['b nw']] | None = None
    ):
        self.eval()

        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = rearrange(cond, 'b d n -> b n d')
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device = device, dtype = torch.long)

        # text

        if isinstance(text, list):
            text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(dim = -1)
            lens = torch.maximum(text_lens, lens) # make sure lengths are at least those of the text characters

        # duration

        cond_mask = lens_to_mask(lens)

        if exists(duration):
            if isinstance(duration, int):
                duration = torch.full((batch,), duration, device = device, dtype = torch.long)

        elif exists(self.duration_predictor):
            duration = self.duration_predictor(cond, text = text, lens = lens).long()

        duration = torch.maximum(lens + 1, duration) # just add one token so something is generated
        duration = duration.clamp(max = max_duration)
        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value = 0.)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value = False)
        cond_mask = rearrange(cond_mask, '... -> ... 1')

        mask = lens_to_mask(duration)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed

            step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow

            return self.cfg_transformer_with_pred_head(
                x,
                step_cond,
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

        out = torch.where(cond_mask, cond, out)

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
            assert inp.shape[-1] == self.num_channels

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

        frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1

        x1 = inp

        # main conditional flow training logic
        # just ~5 loc

        # x0 is gaussian noise

        x0 = torch.randn_like(x1)

        # t is random times from above

        times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        # sample xt (w in the paper)

        w = (1 - (1 - σ) * t) * x0 + t * x1

        flow = x1 - (1 - σ) * x0

        # only predict what is within the random mask span for infilling

        cond = torch.where(
            rand_span_mask[..., None],
            torch.zeros_like(x1), x1
        )

        # transformer and prediction head

        pred = self.transformer_with_pred_head(w, cond, times = times, text = text)

        # flow matching loss

        loss = F.mse_loss(pred, flow, reduction = 'none')

        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
