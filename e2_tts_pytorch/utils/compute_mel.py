import torch
from torch import nn
import torchaudio
class TorchMelSpectrogram(nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0,
        mel_fmax=8000,
        sampling_rate=22050,
        normalize=False,
    ):
        super().__init__()
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            power=2,
            normalized=normalize,
            sample_rate=sampling_rate,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_mels=n_mel_channels,
            norm="slaney",
        )
    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel
