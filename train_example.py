import torch
from e2_tts_pytorch import E2TTS, DurationPredictor

from torch.optim import Adam
from datasets import load_dataset

from e2_tts_pytorch.trainer import (
    HFDataset,
    E2Trainer
)

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 80,
        depth = 2,
    )
)

e2tts = E2TTS(
    duration_predictor = duration_predictor,
    transformer = dict(
        dim = 80,
        depth = 4,
        skip_connect_type = 'concat'
    ),
)

train_dataset = HFDataset(load_dataset("MushanW/GLOBE")["train"])

optimizer = Adam(e2tts.parameters(), lr=7.5e-5)

trainer = E2Trainer(
    e2tts,
    optimizer,
    num_warmup_steps=20000,
    checkpoint_path = 'e2tts.pt',
    log_file = 'e2tts.txt'
)

epochs = 10
batch_size = 32
grad_accumulation_steps = 1

trainer.train(train_dataset, epochs, batch_size, grad_accumulation_steps, save_step=1000)
