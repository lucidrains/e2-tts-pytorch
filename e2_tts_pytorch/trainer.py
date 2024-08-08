from __future__ import annotations

import os
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, SequentialLR

import torchaudio

from einops import rearrange

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ema_pytorch import EMA

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

# plot spectrogram 
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig

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
        target_sample_rate = 24_000,
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
        num_warmup_steps=20000,
        grad_accumulation_steps=1,
        duration_predictor: DurationPredictor | None = None,
        checkpoint_path = None,
        log_file = "logs.txt",
        max_grad_norm = 1.0,
        sample_rate = 22050,
        tensorboard_log_dir = 'runs/e2_tts_experiment',
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        logger.add(log_file)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)

        self.accelerator = Accelerator(
            log_with = "all",
            kwargs_handlers = [ddp_kwargs],
            gradient_accumulation_steps = grad_accumulation_steps,
            **accelerate_kwargs
        )

        self.target_sample_rate = sample_rate

        self.model = model

        if self.is_main:
            self.ema_model = EMA(
                model,
                include_online_model = False,
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        self.duration_predictor = duration_predictor
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.checkpoint_path = default(checkpoint_path, 'model.pth')
        self.mel_spectrogram = MelSpec(sampling_rate=self.target_sample_rate)

        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.max_grad_norm = max_grad_norm
        
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict = self.ema_model.state_dict(),
                scheduler_state_dict = self.scheduler.state_dict(),
                step = step
            )

            self.accelerator.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        if not exists(self.checkpoint_path) or not os.path.exists(self.checkpoint_path):
            return 0

        checkpoint = torch.load(self.checkpoint_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']

    def train(self, train_dataset, epochs, batch_size, num_workers=12, save_step=1000):

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_step = self.load_checkpoint()
        global_step = start_step

        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="step", disable=not self.accelerator.is_local_main_process)
            epoch_loss = 0.0

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch['text']
                    mel_spec = rearrange(batch['mel'], 'b d n -> b n d')
                    mel_lengths = batch["mel_lengths"]

                    if self.duration_predictor is not None:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get('durations'))
                        self.writer.add_scalar('duration loss', dur_loss.item(), global_step)

                    loss, cond, pred = self.model(mel_spec, text=text_inputs, lens=mel_lengths)
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                if self.accelerator.is_local_main_process:
                    logger.info(f"step {global_step+1}: loss = {loss.item():.4f}")
                    self.writer.add_scalar('loss', loss.item(), global_step)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_step)
                
                global_step += 1
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)
                    self.writer.add_figure("mel/target", plot_spectrogram(mel_spec[0,:,:].detach().cpu().numpy()), global_step)
                    self.writer.add_figure("mel/mask", plot_spectrogram(cond[0,:,:].detach().cpu().numpy()), global_step)
                    self.writer.add_figure("mel/prediction", plot_spectrogram(pred[0,:,:].detach().cpu().numpy()), global_step)
            
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)
        
        self.writer.close()
