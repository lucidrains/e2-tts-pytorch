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

from pathlib import Path
from random import random
from functools import partial
from itertools import zip_longest
from collections import namedtuple

from typing import Literal, Callable

import jaxtyping
from beartype import beartype

import torch
import torch.nn.functional as F
from torch import nn, tensor, Tensor, from_numpy
from torch.nn import Module, ModuleList, Sequential, Linear
from torch.nn.utils.rnn import pad_sequence

import torchaudio
from torchaudio.functional import DB_to_amplitude
from torchdiffeq import odeint

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from x_transformers import (
    Attention,
    FeedForward,
    RMSNorm,
    AdaptiveRMSNorm,
)

from x_transformers.x_transformers import RotaryEmbedding

from vocos import Vocos

pad_sequence = partial(pad_sequence, batch_first = True)

# constants

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# named tuples

LossBreakdown = namedtuple('LossBreakdown', ['flow', 'velocity_consistency'])

E2TTSReturn = namedtuple('E2TTS', ['loss', 'cond', 'pred_flow', 'pred_data', 'loss_breakdown'])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def divisible_by(num, den):
    return (num % den) == 0

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

class Identity(Module):
    def forward(self, x, **kwargs):
        return x

# tensor helpers

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# simple utf-8 tokenizer, since paper went character based

def list_str_to_tensor(
    text: list[str],
    padding_value = -1
) -> Int['b nt']:

    list_tensors = [tensor([*bytes(t, 'UTF-8')]) for t in text]
    padded_tensor = pad_sequence(list_tensors, padding_value = -1)
    return padded_tensor

# simple english phoneme-based tokenizer

from g2p_en import G2p

def get_g2p_en_encode():
    g2p = G2p()

    # used by @lucasnewman successfully here
    # https://github.com/lucasnewman/e2-tts-pytorch/blob/ljspeech-test/e2_tts_pytorch/e2_tts.py

    phoneme_to_index = g2p.p2idx
    num_phonemes = len(phoneme_to_index)

    extended_chars = [' ', ',', '.', '-', '!', '?', '\'', '"', '...', '..', '. .', '. . .', '. . . .', '. . . . .', '. ...', '... .', '.. ..']
    num_extended_chars = len(extended_chars)

    extended_chars_dict = {p: (num_phonemes + i) for i, p in enumerate(extended_chars)}
    phoneme_to_index = {**phoneme_to_index, **extended_chars_dict}

    def encode(
        text: list[str],
        padding_value = -1
    ) -> Int['b nt']:

        phonemes = [g2p(t) for t in text]
        list_tensors = [tensor([phoneme_to_index[p] for p in one_phoneme]) for one_phoneme in phonemes]
        padded_tensor = pad_sequence(list_tensors, padding_value = -1)
        return padded_tensor

    return encode, (num_phonemes + num_extended_chars)

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
    frac_lengths: Float['b'],
    max_length: int | None = None
):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min = 0)
    end = start + lengths

    out = mask_from_start_end_indices(seq_len, start, end)

    if exists(max_length):
        out = pad_to_length(out, max_length)

    return out

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

def pad_to_length(
    t: Tensor,
    length: int,
    value = None
):
    seq_len = t.shape[-1]
    if length > seq_len:
        t = F.pad(t, (0, length - seq_len), value = value)

    return t[..., :length]

def interpolate_1d(
    x: Tensor,
    length: int,
    mode = 'bilinear'
):
    x = rearrange(x, 'n d -> 1 d n 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, '1 d n 1 -> n d')

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
        self.sampling_rate = sampling_rate

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

# convolutional positional generating module
# taken from https://github.com/lucidrains/voicebox-pytorch/blob/main/voicebox_pytorch/voicebox_pytorch.py#L203

class DepthwiseConv(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert not divisible_by(kernel_size, 2)
        groups = default(groups, dim) # full depthwise conv by default

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.SiLU()
        )

    def forward(
        self,
        x,
        mask = None
    ):

        if exists(mask):
            x = einx.where('b n, b n d, -> b n d', mask, x, 0.)

        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c')

        if exists(mask):
            out = einx.where('b n, b n d, -> b n d', mask, out, 0.)

        return out

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

# random projection fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(self, x):
        freqs = einx.multiply('i, j -> i j', x, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), 'b *')
        return fourier_embed

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
        **kwargs
    ) -> Float['b n d']:

        text = text + 1 # shift all other token ids up by 1 and use 0 as filler token

        text = text[:, :max_seq_len] # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        text = pad_to_length(text, max_seq_len, value = 0)

        return self.embed(text)

class InterpolatedCharacterEmbed(Module):
    def __init__(
        self,
        dim,
        num_embeds = 256,
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_embeds, dim)

        self.abs_pos_mlp = Sequential(
            Rearrange('... -> ... 1'),
            Linear(1, dim),
            nn.SiLU(),
            Linear(dim, dim)
        )

    def forward(
        self,
        text: Int['b nt'],
        max_seq_len: int,
        mask: Bool['b n'] | None = None
    ) -> Float['b n d']:

        device = text.device

        mask = default(mask, (None,))

        interp_embeds = []
        interp_abs_positions = []

        for one_text, one_mask in zip_longest(text, mask):

            valid_text = one_text >= 0
            one_text = one_text[valid_text]
            one_text_embed = self.embed(one_text)

            # save the absolute positions

            text_seq_len = one_text.shape[0]

            # determine audio sequence length from mask

            audio_seq_len = max_seq_len
            if exists(one_mask):
                audio_seq_len = one_mask.sum().long().item()

            # interpolate text embedding to audio embedding length

            interp_text_embed = interpolate_1d(one_text_embed, audio_seq_len)
            interp_abs_pos = torch.linspace(0, text_seq_len, audio_seq_len, device = device)

            interp_embeds.append(interp_text_embed)
            interp_abs_positions.append(interp_abs_pos)

        interp_embeds = pad_sequence(interp_embeds)
        interp_abs_positions = pad_sequence(interp_abs_positions)

        interp_embeds = F.pad(interp_embeds, (0, 0, 0, max_seq_len - interp_embeds.shape[-2]))
        interp_abs_positions = pad_to_length(interp_abs_positions, max_seq_len)

        # pass interp absolute positions through mlp for implicit positions

        interp_embeds = interp_embeds + self.abs_pos_mlp(interp_abs_positions)

        if exists(mask):
            interp_embeds = einx.where('b n, b n d, -> b n d', mask, interp_embeds, 0.)

        return interp_embeds

# text audio cross conditioning in multistream setup

class TextAudioCrossCondition(Module):
    def __init__(
        self,
        dim,
        dim_text,
        cond_audio_to_text = True,
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
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_text = None, # will default to half of audio dimension
        depth = 8,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        text_depth = None,
        text_heads = None,
        text_dim_head = None,
        text_ff_mult = None,
        cond_on_time = True,
        abs_pos_emb = True,
        max_seq_len = 8192,
        kernel_size = 31,
        dropout = 0.1,
        num_registers = 32,
        attn_laser = False,
        attn_kwargs: dict = dict(
            gate_value_heads = True,
            softclamp_logits = True,
        ),
        ff_kwargs: dict = dict(),
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
        text_ff_mult = default(text_ff_mult, ff_mult)
        text_depth = default(text_depth, depth)

        assert 1 <= text_depth <= depth, 'must have at least 1 layer of text conditioning, but less than total number of speech layers'

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
                RandomFourierEmbed(dim),
                Linear(dim + 1, dim),
                nn.SiLU()
            )

        for ind in range(depth):
            is_first_block = ind == 0

            is_later_half = ind >= (depth // 2)
            has_text = ind < text_depth

            # speech related

            speech_conv = DepthwiseConv(dim, kernel_size = kernel_size)

            attn_norm = rmsnorm_klass(dim)
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, learned_value_residual_mix = not is_first_block, laser = attn_laser, **attn_kwargs)
            attn_adaln_zero = postbranch_klass()

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim = dim, glu = True, mult = ff_mult, dropout = dropout, **ff_kwargs)
            ff_adaln_zero = postbranch_klass()

            skip_proj = Linear(dim * 2, dim, bias = False) if is_later_half else None

            speech_modules = ModuleList([
                skip_proj,
                speech_conv,
                attn_norm,
                attn,
                attn_adaln_zero,
                ff_norm,
                ff,
                ff_adaln_zero,
            ])

            text_modules = None

            if has_text:
                # text related

                text_conv = DepthwiseConv(dim_text, kernel_size = kernel_size)

                text_attn_norm = RMSNorm(dim_text)
                text_attn = Attention(dim = dim_text, heads = text_heads, dim_head = text_dim_head, dropout = dropout, learned_value_residual_mix = not is_first_block, laser = attn_laser, **attn_kwargs)

                text_ff_norm = RMSNorm(dim_text)
                text_ff = FeedForward(dim = dim_text, glu = True, mult = text_ff_mult, dropout = dropout, **ff_kwargs)

                # cross condition

                is_last = ind == (text_depth - 1)

                cross_condition = TextAudioCrossCondition(dim = dim, dim_text = dim_text, cond_audio_to_text = not is_last)

                text_modules = ModuleList([
                    text_conv,
                    text_attn_norm,
                    text_attn,
                    text_ff_norm,
                    text_ff,
                    cross_condition
                ])

            self.layers.append(ModuleList([
                speech_modules,
                text_modules
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

        skips = []

        # value residual

        text_attn_first_values = None
        attn_first_values = None

        # go through the layers

        for ind, (speech_modules, text_modules) in enumerate(self.layers):
            layer = ind + 1

            (
                maybe_skip_proj,
                speech_conv,
                attn_norm,
                attn,
                maybe_attn_adaln_zero,
                ff_norm,
                ff,
                maybe_ff_adaln_zero
            ) = speech_modules

            # smaller text transformer

            if exists(text_embed) and exists(text_modules):

                (
                    text_conv,
                    text_attn_norm,
                    text_attn,
                    text_ff_norm,
                    text_ff,
                    cross_condition
                ) = text_modules

                text_embed = text_conv(text_embed, mask = mask) + text_embed

                text_attn_out, text_attn_inter = text_attn(text_attn_norm(text_embed), rotary_pos_emb = text_rotary_pos_emb, mask = mask, return_intermediates = True, value_residual = text_attn_first_values)
                text_embed = text_attn_out + text_embed

                text_attn_first_values = default(text_attn_first_values, text_attn_inter.values)

                text_embed = text_ff(text_ff_norm(text_embed)) + text_embed

                x, text_embed = cross_condition(x, text_embed)

            # skip connection logic

            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half:
                skip = skips.pop()
                x = torch.cat((x, skip), dim = -1)
                x = maybe_skip_proj(x)

            # position generating convolution

            x = speech_conv(x, mask = mask) + x

            # attention and feedforward blocks

            attn_out, attn_inter = attn(attn_norm(x, **norm_kwargs), rotary_pos_emb = rotary_pos_emb, mask = mask, return_intermediates = True, value_residual = attn_first_values)

            attn_first_values = default(attn_first_values, attn_inter.values)

            x = x + maybe_attn_adaln_zero(attn_out, **norm_kwargs)

            ff_out = ff(ff_norm(x, **norm_kwargs))

            x = x + maybe_ff_adaln_zero(ff_out, **norm_kwargs)

        assert len(skips) == 0

        _, x = unpack(x, registers_packed_shape, 'b * d')

        return self.final_norm(x)

# main classes

class DurationPredictor(Module):
    @beartype
    def __init__(
        self,
        transformer: dict | Transformer,
        num_channels = None,
        mel_spec_kwargs: dict = dict(),
        char_embed_kwargs: dict = dict(),
        text_num_embeds = None,
        tokenizer: (
            Literal['char_utf8', 'phoneme_en'] |
            Callable[[list[str]], Int['b nt']]
        ) = 'char_utf8'
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
        dim_text = transformer.dim_text

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
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        else:
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        self.embed_text = CharacterEmbed(dim_text, num_embeds = text_num_embeds, **char_embed_kwargs)

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

    @beartype
    def __init__(
        self,
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
        concat_cond = False,
        interpolated_text = False,
        text_num_embeds: int | None = None,
        tokenizer: (
            Literal['char_utf8', 'phoneme_en'] |
            Callable[[list[str]], Int['b nt']]
        ) = 'char_utf8',
        use_vocos = True,
        pretrained_vocos_path = 'charactr/vocos-mel-24khz',
        sampling_rate: int | None = None,
        velocity_consistency_weight = 0.,
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

        # sampling

        self.odeint_kwargs = odeint_kwargs

        # mel spec

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
 
        self.num_channels = num_channels
        self.sampling_rate = default(sampling_rate, getattr(self.mel_spec, 'sampling_rate', None))

        # whether to concat condition and project rather than project both and sum

        self.concat_cond = concat_cond

        if concat_cond:
            self.proj_in = nn.Linear(num_channels * 2, dim)
        else:
            self.proj_in = nn.Linear(num_channels, dim)
            self.cond_proj_in = nn.Linear(num_channels, dim)

        # to prediction

        self.to_pred = Linear(dim, num_channels)

        # tokenizer and text embed

        if callable(tokenizer):
            assert exists(text_num_embeds), '`text_num_embeds` must be given if supplying your own tokenizer encode function'
            self.tokenizer = tokenizer
        elif tokenizer == 'char_utf8':
            text_num_embeds = 256
            self.tokenizer = list_str_to_tensor
        elif tokenizer == 'phoneme_en':
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        else:
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        self.cond_drop_prob = cond_drop_prob

        # text embedding

        text_embed_klass = CharacterEmbed if not interpolated_text else InterpolatedCharacterEmbed

        self.embed_text = text_embed_klass(dim_text, num_embeds = text_num_embeds, **char_embed_kwargs)

        # weight for velocity consistency

        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        self.velocity_consistency_weight = velocity_consistency_weight

        # default vocos for mel -> audio

        self.vocos = Vocos.from_pretrained(pretrained_vocos_path) if use_vocos else None

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
        drop_text_cond: bool | None = None,
        return_drop_text_cond = False
    ):
        seq_len = x.shape[-2]
        drop_text_cond = default(drop_text_cond, self.training and random() < self.cond_drop_prob)

        if self.concat_cond:
            # concat condition, given as using voicebox-like scheme
            x = torch.cat((cond, x), dim = -1)

        x = self.proj_in(x)

        if not self.concat_cond:
            # an alternative is to simply sum the condition
            # seems to work fine

            cond = self.cond_proj_in(cond)
            x = x + cond

        # whether to use a text embedding

        text_embed = None
        if exists(text) and not drop_text_cond:
            text_embed = self.embed_text(text, seq_len, mask = mask)

        # attend

        attended = self.transformer(
            x,
            times = times,
            mask = mask,
            text_embed = text_embed
        )

        pred =  self.to_pred(attended)

        if not return_drop_text_cond:
            return pred

        return pred, drop_text_cond

    def cfg_transformer_with_pred_head(
        self,
        *args,
        cfg_strength: float = 1.,
        remove_parallel_component: bool = True,
        keep_parallel_frac: float = 0.,
        **kwargs,
    ):
        
        pred = self.transformer_with_pred_head(*args, drop_text_cond = False, **kwargs)

        if cfg_strength < 1e-5:
            return pred

        null_pred = self.transformer_with_pred_head(*args, drop_text_cond = True, **kwargs)

        cfg_update = pred - null_pred

        if remove_parallel_component:
            # https://arxiv.org/abs/2410.02416
            parallel, orthogonal = project(cfg_update, pred)
            cfg_update = orthogonal + parallel * keep_parallel_frac

        return pred + cfg_update * cfg_strength

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
        vocoder: Callable[[Float['b d n']], list[Float['_']]] | None = None,
        return_raw_output: bool | None = None,
        save_to_filename: str | None = None
    ) -> (
        Float['b n d'],
        list[Float['_']]
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
            duration = self.duration_predictor(cond, text = text, lens = lens, return_loss = False).long()

        duration = torch.maximum(lens + 1, duration) # just add one token so something is generated
        duration = duration.clamp(max = max_duration)

        assert duration.shape[0] == batch

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

        # able to return raw untransformed output, if not using mel rep

        if exists(return_raw_output) and return_raw_output:
            return out

        # take care of transforming mel to audio if `vocoder` is passed in, or if `use_vocos` is turned on

        if exists(vocoder):
            assert not exists(self.vocos), '`use_vocos` should not be turned on if you are passing in a custom `vocoder` on sampling'
            out = rearrange(out, 'b n d -> b d n')
            out = vocoder(out)

        elif exists(self.vocos):

            audio = []
            for mel, one_mask in zip(out, mask):
                one_out = DB_to_amplitude(mel[one_mask], ref = 1., power = 0.5)

                one_out = rearrange(one_out, 'n d -> 1 d n')
                one_audio = self.vocos.decode(one_out)
                one_audio = rearrange(one_audio, '1 nw -> nw')
                audio.append(one_audio)

            out = audio

        if exists(save_to_filename):
            assert exists(vocoder) or exists(self.vocos)
            assert exists(self.sampling_rate)

            path = Path(save_to_filename)
            parent_path = path.parents[0]
            parent_path.mkdir(exist_ok = True, parents = True)

            for ind, one_audio in enumerate(out):
                one_audio = rearrange(one_audio, 'nw -> 1 nw')
                save_path = str(parent_path / f'{ind + 1}.{path.name}')
                torchaudio.save(save_path, one_audio.detach().cpu(), sample_rate = self.sampling_rate)

        return out

    def forward(
        self,
        inp: Float['b n d'] | Float['b nw'], # mel or raw wave
        *,
        text: Int['b nt'] | list[str] | None = None,
        times: Int['b'] | None = None,
        lens: Int['b'] | None = None,
        velocity_consistency_model: E2TTS | None = None,
        velocity_consistency_delta = 1e-5
    ):
        need_velocity_loss = exists(velocity_consistency_model) and self.velocity_consistency_weight > 0.

        # handle raw wave

        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, 'b d n -> b n d')
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

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
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, max_length = seq_len)

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

        # if need velocity consistency, make sure time does not exceed 1.

        if need_velocity_loss:
            t = t * (1. - velocity_consistency_delta)

        # sample xt (w in the paper)

        w = (1. - t) * x0 + t * x1

        flow = x1 - x0

        # only predict what is within the random mask span for infilling

        cond = einx.where(
            'b n, b n d, b n d -> b n d',
            rand_span_mask,
            torch.zeros_like(x1), x1
        )

        # transformer and prediction head

        pred, did_drop_text_cond = self.transformer_with_pred_head(
            w,
            cond,
            times = times,
            text = text,
            mask = mask,
            return_drop_text_cond = True
        )

        # maybe velocity consistency loss

        velocity_loss = self.zero

        if need_velocity_loss:

            t_with_delta = t + velocity_consistency_delta
            w_with_delta = (1. - t_with_delta) * x0 + t_with_delta * x1

            with torch.no_grad():
                ema_pred = velocity_consistency_model.transformer_with_pred_head(
                    w_with_delta,
                    cond,
                    times = times + velocity_consistency_delta,
                    text = text,
                    mask = mask,
                    drop_text_cond = did_drop_text_cond
                )

            velocity_loss = F.mse_loss(pred, ema_pred, reduction = 'none')
            velocity_loss = velocity_loss[rand_span_mask].mean()

        # flow matching loss

        loss = F.mse_loss(pred, flow, reduction = 'none')

        loss = loss[rand_span_mask].mean()

        # total loss and get breakdown

        total_loss = (
            loss +
            velocity_loss * self.velocity_consistency_weight
        )

        breakdown = LossBreakdown(loss, velocity_loss)

        # return total loss and bunch of intermediates

        return E2TTSReturn(total_loss, cond, pred, x0 + pred, breakdown)
