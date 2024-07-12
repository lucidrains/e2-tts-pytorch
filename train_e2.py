import torch
import torch.nn.init as init
from torch.optim import Adam
from e2_dataset import E2EDataset
from e2_tts_pytorch.e2_tts import E2TTS, DurationPredictor
from e2_trainer import E2Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_path = "/home/azureuser/xtts/assets/vocab.json"

train_dataset = E2EDataset("MushanW/GLOBE", tokenizer_path)

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 512,
        depth = 2,
    )
).to(device)

e2tts = E2TTS(
    duration_predictor = duration_predictor,
    transformer = dict(
        dim = 80,
        depth = 4,
        skip_connect_type = 'concat'
    ),
).to(device)


optimizer = Adam(e2tts.parameters(), lr=1e-4)

checkpoint_path = 'e2e.pt'
log_file = 'e2e.txt'

trainer = E2Trainer(
    e2tts,
    optimizer,
    checkpoint_path=checkpoint_path,
    log_file=log_file
)

epochs = 10
batch_size = 8
grad_accumulation_steps = 1

trainer.train(train_dataset, epochs, batch_size, grad_accumulation_steps, save_step=1000)