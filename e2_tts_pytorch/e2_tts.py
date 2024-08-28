"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
dt - dimension text
"""

from __future__ import annotations
from random import random
from functools import partial
from collections import namedtuple
from typing import Literal, Callable

import torch
from torch import nn, tensor, from_numpy
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential, Linear
from torch.nn.utils.rnn import pad_sequence

import torchaudio
from torchdiffeq import odeint

import einx
from einops.layers.torch import Rearrange
from einops import einsum, rearrange, repeat, reduce, pack, unpack

from scipy.optimize import linear_sum_assignment

from x_transformers import (
    Attention,
    FeedForward,
    RMSNorm,
    AdaptiveRMSNorm,
)

from x_transformers.x_transformers import RotaryEmbedding

from gateloop_transformer import SimpleGateLoopLayer

from e2_tts_pytorch.tensor_typing import (
    Float,
    Int,
    Bool
)

# constants

E2TTSReturn = namedtuple('E2TTS', ['loss', 'cond', 'pred'])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

class Identity(Module):
    def forward(self, x, **kwargs):
        return x

# simple utf-8 tokenizer, since paper went character based

def list_str_to_tensor(
    text: list[str],
    padding_value = -1
) -> Int['b nt']:

    list_tensors = [tensor([*bytes(t, 'UTF-8')]) for t in text]
    padded_tensor = pad_sequence(list_tensors, padding_value = -1, batch_first = True)
    return padded_tensor

# simple english phoneme-based tokenizer

from g2p_en import G2p

def get_g2p_en_encode():
    g2p = G2p()

    def encode(
        text: list[str],
        padding_value = -1
    ) -> Int['b nt']:

        phonemes = [g2p(t) for t in text]
        list_tensors = [tensor([g2p.p2idx[p] for p in one_phoneme]) for one_phoneme in phonemes]
        padded_tensor = pad_sequence(list_tensors, padding_value = -1, batch_first = True)
        return padded_tensor

    return encode

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
    mask: Bool['b n'] | None = None
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

        self.register_buffer('dummy', tensor(0), persistent = False)

    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = rearrange(inp, 'b 1 nw -> b nw')

        assert len(inp.shape) == 2

        if self.dummy.device != inp.device:
            self.to(inp.device)

        mel = self.mel_stft(inp)
        mel = log(mel)
        return mel

# adaln zero from DiT paper

class AdaLNZero(Module):
    def __init__(
        self,
        dim,
        dim_condition = None,
        init_bias_value = -2.
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        gamma = self.to_gamma(condition).sigmoid()
        return x * gamma

# character embedding

class CharacterEmbed(Module):
    def __init__(
        self,
        dim,
        num_embeds = 256,
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_embeds + 1, dim) # will just use 0 as the 'filler token'

    def forward(
        self,
        text: Int['b nt'],
        max_seq_len: int,
    ) -> Float['b n d']:

        text = text + 1 # shift all other token ids up by 1 and use 0 as filler token

        text = text[:, :max_seq_len] # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value = 0)

        return self.embed(text)

class TextAudioCrossCondition(Module):
    def __init__(
        self,
        dim,
        dim_text,
        cond_audio_to_text = True
    ):
        super().__init__()
        self.text_to_audio = nn.Linear(dim_text + dim, dim, bias = False)
        nn.init.zeros_(self.text_to_audio.weight)

        self.cond_audio_to_text = cond_audio_to_text

        if cond_audio_to_text:
            self.audio_to_text = nn.Linear(dim + dim_text, dim_text, bias = False)
            nn.init.zeros_(self.audio_to_text.weight)

    def forward(
        self,
        audio: Float['b n d'],
        text: Float['b n dt']
    ):
        audio_text, _ = pack((audio, text), 'b n *')

        text_cond = self.text_to_audio(audio_text)
        audio_cond = self.audio_to_text(audio_text) if self.cond_audio_to_text else 0.

        return audio + text_cond, text + audio_cond

# attention and transformer backbone
# for use in both e2tts as well as duration module

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        dim_text = None, # will default to half of audio dimension
        depth = 8,
        heads = 8,
        dim_head = 64,
        text_heads = None,
        text_dim_head = None,
        cond_on_time = True,
        skip_connect_type: Literal['add', 'concat', 'none'] = 'concat',
        abs_pos_emb = True,
        max_seq_len = 8192,
        dropout = 0.1,
        num_registers = 32,
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

        dim_text = default(dim_text, dim // 2)
        self.dim_text = dim_text

        text_heads = default(text_heads, heads)
        text_dim_head = default(text_dim_head, dim_head)

        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == 'concat'

        self.depth = depth
        self.layers = ModuleList([])

        # registers

        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(num_registers, dim))
        nn.init.normal_(self.registers, std = 0.02)

        self.text_registers = nn.Parameter(torch.zeros(num_registers, dim_text))
        nn.init.normal_(self.text_registers, std = 0.02)

        # rotary embedding

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.text_rotary_emb = RotaryEmbedding(dim_head)

        # time conditioning
        # will use adaptive rmsnorm

        self.cond_on_time = cond_on_time
        rmsnorm_klass = RMSNorm if not cond_on_time else AdaptiveRMSNorm
        postbranch_klass = Identity if not cond_on_time else partial(AdaLNZero, dim = dim)

        self.time_cond_mlp = Identity()

        if cond_on_time:
            self.time_cond_mlp = Sequential(
                Rearrange('... -> ... 1'),
                Linear(1, dim),
                nn.SiLU()
            )

        for ind in range(depth):
            is_last = ind == (depth - 1)
            is_later_half = ind >= (depth // 2)

            # speech related

            gateloop = SimpleGateLoopLayer(dim = dim)

            attn_norm = rmsnorm_klass(dim)
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, **attn_kwargs)
            attn_adaln_zero = postbranch_klass()

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim = dim, glu = True, dropout = dropout, **ff_kwargs)
            ff_adaln_zero = postbranch_klass()

            skip_proj = Linear(dim * 2, dim, bias = False) if needs_skip_proj and is_later_half else None

            # text related

            text_gateloop = SimpleGateLoopLayer(dim = dim_text)

            text_attn_norm = RMSNorm(dim_text)
            text_attn = Attention(dim = dim_text, heads = text_heads, dim_head = text_dim_head, dropout = dropout, **attn_kwargs)

            text_ff_norm = RMSNorm(dim_text)
            text_ff = FeedForward(dim = dim_text, glu = True, dropout = dropout, **ff_kwargs)

            # cross condition

            cross_condition = TextAudioCrossCondition(dim = dim, dim_text = dim_text, cond_audio_to_text = not is_last)

            self.layers.append(ModuleList([
                gateloop,
                skip_proj,
                attn_norm,
                attn,
                attn_adaln_zero,
                ff_norm,
                ff,
                ff_adaln_zero,
                text_gateloop,
                text_attn_norm,
                text_attn,
                text_ff_norm,
                text_ff,
                cross_condition
            ]))

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x: Float['b n d'],
        times: Float['b'] | Float[''] | None = None,
        mask: Bool['b n'] | None = None,
        text_embed: Float['b n dt'] | None = None,
    ):
        batch, seq_len, device = *x.shape[:2], x.device

        assert not (exists(times) ^ self.cond_on_time), '`times` must be passed in if `cond_on_time` is set to `True` and vice versa'

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

        # register tokens

        registers = repeat(self.registers, 'r d -> b r d', b = batch)
        x, registers_packed_shape = pack((registers, x), 'b * d')

        if exists(mask):
            mask = F.pad(mask, (self.num_registers, 0), value = True)

        # rotary embedding

        rotary_pos_emb = self.rotary_emb.forward_from_seq_len(x.shape[-2])

        # text related

        if exists(text_embed):
            text_rotary_pos_emb = self.text_rotary_emb.forward_from_seq_len(x.shape[-2])

            text_registers = repeat(self.text_registers, 'r d -> b r d', b = batch)
            text_embed, _ = pack((text_registers, text_embed), 'b * d')

        # skip connection related stuff

        skip_connect_type = self.skip_connect_type

        skips = []

        # go through the layers

        for ind, (
            gateloop,
            maybe_skip_proj,
            attn_norm,
            attn,
            maybe_attn_adaln_zero,
            ff_norm,
            ff,
            maybe_ff_adaln_zero,
            text_gateloop,
            text_attn_norm,
            text_attn,
            text_ff_norm,
            text_ff,
            cross_condition
        ) in enumerate(self.layers):

            layer = ind + 1

            # smaller text transformer

            if exists(text_embed):
                text_embed = text_gateloop(text_embed) + text_embed

                text_embed = text_attn(text_attn_norm(text_embed), rotary_pos_emb = text_rotary_pos_emb, mask = mask) + text_embed

                text_embed = text_ff(text_ff_norm(text_embed)) + text_embed

                x, text_embed = cross_condition(x, text_embed)

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

            # associative scan

            x = gateloop(x) + x

            # attention and feedforward blocks

            attn_out = attn(attn_norm(x, **norm_kwargs), rotary_pos_emb = rotary_pos_emb, mask = mask)

            x = x + maybe_attn_adaln_zero(attn_out, **norm_kwargs)

            ff_out = ff(ff_norm(x, **norm_kwargs))

            x = x + maybe_ff_adaln_zero(ff_out, **norm_kwargs)

        assert len(skips) == 0

        _, x = unpack(x, registers_packed_shape, 'b * d')

        return self.final_norm(x)

# main classes

class DurationPredictor(Module):
    def __init__(
        self,
        transformer: dict | Transformer,
        num_channels = None,
        mel_spec_kwargs: dict = dict(),
        char_embed_kwargs: dict = dict(),
        text_num_embeds = None,
        tokenizer: str |  Callable[[list[str]], Int['b nt']] = 'char_utf8'
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

        # tokenizer and text embed

        if callable(tokenizer):
            assert exists(text_num_embeds), '`text_num_embeds` must be given if supplying your own tokenizer encode function'
            self.tokenizer = tokenizer
        elif tokenizer == 'char_utf8':
            text_num_embeds = 256
            self.tokenizer = list_str_to_tensor
        elif tokenizer == 'phoneme_en':
            text_num_embeds = 74
            self.tokenizer = get_g2p_en_encode()
        else:
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        self.embed_text = CharacterEmbed(transformer.dim_text, num_embeds = text_num_embeds, **char_embed_kwargs)

        # to prediction

        self.to_pred = Sequential(
            Linear(dim, 1, bias = False),
            nn.Softplus(),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        x: Float['b n d'] | Float['b nw'],
        *,
        text: Int['b nt'] | list[str] | None = None,
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

        text_embed = None

        if exists(text):
            if isinstance(text, list):
                text = list_str_to_tensor(text).to(device)
                assert text.shape[0] == batch

            text_embed = self.embed_text(text, seq_len)

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

        x = self.transformer(
            x,
            mask = mask,
            text_embed = text_embed,
        )

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
        cond_drop_prob = 0.25,
        num_channels = None,
        mel_spec_module: Module | None = None,
        char_embed_kwargs: dict = dict(),
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.),
        immiscible = False,
        text_num_embeds = None,
        tokenizer: str |  Callable[[list[str]], Int['b nt']] = 'char_utf8'
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
        dim_text = transformer.dim_text

        self.dim = dim
        self.dim_text = dim_text

        self.frac_lengths_mask = frac_lengths_mask

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

        # tokenizer and text embed

        if callable(tokenizer):
            assert exists(text_num_embeds), '`text_num_embeds` must be given if supplying your own tokenizer encode function'
            self.tokenizer = tokenizer
        elif tokenizer == 'char_utf8':
            text_num_embeds = 256
            self.tokenizer = list_str_to_tensor
        elif tokenizer == 'phoneme_en':
            text_num_embeds = 74
            self.tokenizer = get_g2p_en_encode()
        else:
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        self.cond_drop_prob = cond_drop_prob

        self.embed_text = CharacterEmbed(dim_text, num_embeds = text_num_embeds, **char_embed_kwargs)

        # immiscible flow - https://arxiv.org/abs/2406.12303

        self.immiscible = immiscible

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
        seq_len = x.shape[-2]
        drop_text_cond = default(drop_text_cond, self.training and random() < self.cond_drop_prob)

        x = self.proj_in(x)
        cond = self.cond_proj_in(cond)

        # add the condition, given as using voicebox-like scheme

        x = x + cond

        # whether to use a text embedding

        text_embed = None
        if exists(text) and not drop_text_cond:
            text_embed = self.embed_text(text, seq_len)

        # attend

        attended = self.transformer(
            x,
            times = times,
            mask = mask,
            text_embed = text_embed
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
        text: Int['b nt'] | list[str] | None = None,
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
            text = self.tokenizer(text).to(device)
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
        text: Int['b nt'] | list[str] | None = None,
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
            text = self.tokenizer(text).to(device)
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

        # maybe immiscible flow

        if self.immiscible:
            cost = torch.cdist(x1.flatten(1), x0.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.cpu())
            x0 = x0[from_numpy(reorder_indices).to(cost.device)]

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

        loss = loss[rand_span_mask].mean()

        return E2TTSReturn(loss, cond, pred)
