import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from accelerate import Accelerator
from e2_collate import collate_fn
import os
import logging
from utils.compute_mel import TorchMelSpectrogram
from einops import rearrange

class E2Trainer:
    def __init__(self, model, optimizer, duration_predictor=None,
                 checkpoint_path=None, log_file="logs.txt",
                 max_grad_norm=1.0,
                 sample_rate=22050):
        self.target_sample_rate = sample_rate
        self.accelerator = Accelerator(log_with="all")
        self.model = model
        self.duration_predictor = duration_predictor
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.mel_spectrogram = TorchMelSpectrogram(sampling_rate=self.target_sample_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.max_grad_norm = max_grad_norm

    def save_checkpoint(self, step, finetune=False):
        if self.checkpoint_path is None:
            self.checkpoint_path = "model.pth"
        checkpoint = {
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['step']
        return 0

    def train(self, train_dataset, epochs, batch_size, grad_accumulation_steps=1, num_workers=12, save_step=1000):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        train_dataloader = self.accelerator.prepare(train_dataloader)
        start_step = 0
        global_step = start_step
        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="step", disable=not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                text_inputs = batch['text']
                text_lengths = batch['text_lengths']
                mel_spec = rearrange(batch['mel'], 'b d n -> b n d')
                mel_lengths = batch["mel_lengths"]
                print(mel_spec.shape)
                print(text_inputs.shape)
                # duration = batch['durations']
                if self.duration_predictor is not None:
                    dur_loss = self.duration_predictor(mel_spec, target_duration = duration)
                loss = self.model(mel_spec, text_inputs, lens=mel_lengths)
                # mel_loss = torch.nn.functional.mse_loss(masked_mel, masked_mel_hat)
                self.accelerator.backward(loss)
                
                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.accelerator.is_local_main_process:
                    self.logger.info(f"Step {global_step+1}: E2E Loss = {loss.item():.4f}")
                global_step += 1
                progress_bar.set_postfix(loss=loss.item())
                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)
            loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                    self.logger.info(f"Epoch {epoch+1}/{epochs} - E2E Loss = {loss.item():.4f}")