import torch
from torch.utils.data import Dataset
import torchaudio
from e2_tts_pytorch.utils.compute_mel import TorchMelSpectrogram
from datasets import load_dataset
from tokenizers import Tokenizer
import logging
from einops import rearrange, reduce

logger = logging.getLogger(__name__)

class E2EDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer_path):
        self.data = load_dataset(hf_dataset, split='train')
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.target_sample_rate = 22050
        self.hop_length = 256
        self.mel_spectrogram = TorchMelSpectrogram(sampling_rate=self.target_sample_rate)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        audio = row['audio']['array']
        logger.info(f"Audio shape: {audio.shape}")
        sample_rate = row['audio']['sampling_rate']
        duration = audio.shape[-1] / sample_rate
        
        if duration > 20 or duration < 0.3:
            logger.warning(f"Skipping due to duration out of bound: {duration}")
            return self.__getitem__((index + 1) % len(self.data))
        
        audio_tensor = torch.from_numpy(audio).float()
        
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)
        
        audio_tensor = rearrange(audio_tensor, 't -> 1 t')
        
        mel_spec = self.mel_spectrogram(audio_tensor)
        
        mel_spec = rearrange(mel_spec, '1 d t -> d t')
        
        text = row['transcript']
        text = text.replace(" ", "[SPACE]")
        text_tokens = self.tokenize_text(text)
        
        return {
            'mel_spec': mel_spec,
            'text': text_tokens,
        }

    def tokenize_text(self, text):
        output = self.tokenizer.encode(text)
        return output.ids