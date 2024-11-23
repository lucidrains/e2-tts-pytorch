import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from datasets import load_dataset

from e2_tts_pytorch.trainer import (
    HFDataset,
    E2Trainer
)

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 512,
        depth = 6,
    )
)

e2tts = E2TTS(
    duration_predictor = duration_predictor,
    transformer = dict(
        dim = 512,
        depth = 12
    ),
)

train_dataset = HFDataset(load_dataset("MushanW/GLOBE")["train"])

trainer = E2Trainer(
    e2tts,
    num_warmup_steps=20000,
    grad_accumulation_steps = 1,
    checkpoint_path = 'e2tts.pt',
    log_file = 'e2tts.txt'
)

epochs = 10
batch_size = 32

trainer.train(train_dataset, epochs, batch_size, save_step=1000)
