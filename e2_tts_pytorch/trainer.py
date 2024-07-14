from __future__ import annotations

import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchaudio

from einops import rearrange
from accelerate import Accelerator

from loguru import logger

from e2_tts_pytorch.e2_tts import (
    E2TTS,
    DurationPredictor,
    MelSpec
)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# collation

def collate_fn(batch):
    mel_specs = [item['mel_spec'].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value = 0)
        padded_mel_specs.append(padded_spec)
    
    mel_specs = torch.stack(padded_mel_specs)

    text = [item['text'] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel = mel_specs,
        mel_lengths = mel_lengths,
        text = text,
        text_lengths = text_lengths,
    )

# dataset

class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate = 22050,
        hop_length = 256
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mel_spectrogram = MelSpec(sampling_rate=target_sample_rate)

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
        
        return dict(
            mel_spec = mel_spec,
            text = text,
        )

# trainer

class E2Trainer:
    def __init__(
        self,
        model: E2TTS,
        optimizer,
        duration_predictor: DurationPredictor | None = None,
        checkpoint_path = None,
        log_file = "logs.txt",
        max_grad_norm = 1.0,
        sample_rate = 22050,
        tensorboard_log_dir = 'runs/e2_tts_experiment',
        accelerate_kwargs: dict = dict()
    ):
        logger.add(log_file)

        self.accelerator = Accelerator(
            log_with="all",
            **accelerate_kwargs
        )

        self.target_sample_rate = sample_rate
        self.model = model
        self.duration_predictor = duration_predictor
        self.optimizer = optimizer
        self.checkpoint_path = default(checkpoint_path, 'model.pth')

        self.mel_spectrogram = MelSpec(sampling_rate=self.target_sample_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.max_grad_norm = max_grad_norm
        
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def save_checkpoint(self, step, finetune=False):

        checkpoint = dict(
            model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
            optimizer_state_dict = self.optimizer.state_dict(),
            step = step
        )

        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        if not exists(self.checkpoint_path) or not os.path.exists(self.checkpoint_path):
            return 0

        checkpoint = torch.load(self.checkpoint_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step']

    def train(self, train_dataset, epochs, batch_size, grad_accumulation_steps=1, num_workers=12, save_step=1000):
        # (todo) gradient accumulation needs to be accounted for

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        train_dataloader = self.accelerator.prepare(train_dataloader)
        start_step = self.load_checkpoint()
        global_step = start_step

        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="step", disable=not self.accelerator.is_local_main_process)
            epoch_loss = 0.0

            for batch in progress_bar:
                text_inputs = batch['text']
                mel_spec = rearrange(batch['mel'], 'b d n -> b n d')
                mel_lengths = batch["mel_lengths"]
                
                if self.duration_predictor is not None:
                    dur_loss = self.duration_predictor(mel_spec, target_duration=batch.get('durations'))
                    self.writer.add_scalar('Duration Loss', dur_loss.item(), global_step)
                
                loss = self.model(mel_spec, text=text_inputs, lens=mel_lengths)
                self.accelerator.backward(loss)
                
                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.accelerator.is_local_main_process:
                    logger.info(f"Step {global_step+1}: Loss = {loss.item():.4f}")
                    self.writer.add_scalar('E2E Loss', loss.item(), global_step)
                
                global_step += 1
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)
            
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss = {epoch_loss:.4f}")
                self.writer.add_scalar('Epoch Average Loss', epoch_loss, epoch)
        
        self.writer.close()
