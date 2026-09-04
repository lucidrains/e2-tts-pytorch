import ast
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


TRAINER_PATH = Path(__file__).parents[1] / 'e2_tts_pytorch' / 'trainer.py'
TRAIN_EXAMPLE_PATH = Path(__file__).parents[1] / 'train_example.py'


def load_trainer_without_audio_extensions():
    """Load the trainer with lightweight stand-ins for optional extensions."""

    package = types.ModuleType('e2_tts_pytorch')
    package.__path__ = []
    e2_tts = types.ModuleType('e2_tts_pytorch.e2_tts')
    e2_tts.E2TTS = object
    e2_tts.DurationPredictor = object
    e2_tts.MelSpec = object

    torchaudio = types.ModuleType('torchaudio')
    torchaudio.transforms = types.SimpleNamespace()

    accelerate = types.ModuleType('accelerate')
    accelerate.Accelerator = object
    accelerate_utils = types.ModuleType('accelerate.utils')
    accelerate_utils.DistributedDataParallelKwargs = object

    adopt_package = types.ModuleType('adam_atan2_pytorch')
    adopt_module = types.ModuleType('adam_atan2_pytorch.adopt')
    adopt_module.Adopt = torch.optim.SGD

    ema_module = types.ModuleType('ema_pytorch')
    ema_module.EMA = object

    loguru_module = types.ModuleType('loguru')
    loguru_module.logger = types.SimpleNamespace(
        add=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )

    replacements = {
        'e2_tts_pytorch': package,
        'e2_tts_pytorch.e2_tts': e2_tts,
        'torchaudio': torchaudio,
        'accelerate': accelerate,
        'accelerate.utils': accelerate_utils,
        'adam_atan2_pytorch': adopt_package,
        'adam_atan2_pytorch.adopt': adopt_module,
        'ema_pytorch': ema_module,
        'loguru': loguru_module,
    }

    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)

    module_name = 'trainer_under_test'
    spec = importlib.util.spec_from_file_location(module_name, TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class FakeAccelerator:
    is_local_main_process = True
    is_main_process = True
    sync_gradients = True

    def __init__(self):
        self.backward_calls = []
        self.accumulate_calls = 0

    def accumulate(self, _model):
        self.accumulate_calls += 1
        return nullcontext()

    def backward(self, loss):
        self.backward_calls.append(loss.detach().clone())
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def prepare(self, *objects):
        return objects if len(objects) > 1 else objects[0]

    def unwrap_model(self, model):
        return model


class FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))

    def add_figure(self, *args, **kwargs):
        pass

    def close(self):
        pass


class ToyModel(nn.Module):
    def __init__(self, duration_predictor):
        super().__init__()
        self.flow_weight = nn.Parameter(torch.tensor(0.0))
        self.duration_predictor = duration_predictor

    def forward(self, mel, **kwargs):
        loss = (self.flow_weight - 1.).square()
        return loss, torch.zeros_like(mel), mel, mel


class ToyDurationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.duration_weight = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, mel, *, lens=None):
        self.calls += 1
        return (self.duration_weight - 2.).square()


class TinyLoader:
    def __init__(self, *args, **kwargs):
        self.batches = [
            {
                'mel': torch.zeros(1, 1, 2),
                'mel_lengths': torch.tensor([2]),
                'text': ['a'],
            },
            {
                'mel': torch.zeros(1, 1, 2),
                'mel_lengths': torch.tensor([2]),
                'text': ['b'],
            },
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_postfix(self, **kwargs):
        pass


def test_duration_loss_reaches_both_models_and_preserves_accumulation(monkeypatch):
    trainer_module = load_trainer_without_audio_extensions()
    monkeypatch.setattr(trainer_module, 'DataLoader', TinyLoader)
    monkeypatch.setattr(trainer_module, 'tqdm', lambda iterable, **kwargs: iterable)

    duration_predictor = ToyDurationPredictor()
    model = ToyModel(duration_predictor)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = trainer_module.E2Trainer.__new__(trainer_module.E2Trainer)
    trainer.model = model
    trainer.duration_predictor = duration_predictor
    trainer.duration_loss_weight = 0.5
    trainer.use_switch_ema = False
    trainer.optimizer = optimizer
    trainer.accelerator = FakeAccelerator()
    trainer.ema_model = types.SimpleNamespace(
        initted=False,
        update=lambda: None,
        state_dict=lambda: {},
    )
    trainer.need_velocity_consistent_loss = False
    trainer.num_warmup_steps = 1
    trainer.max_grad_norm = 0.
    trainer.writer = FakeWriter()
    trainer.load_checkpoint = lambda: 0

    trainer.train([], epochs=1, batch_size=1, num_workers=0, save_step=100)

    assert trainer.accelerator.accumulate_calls == 2
    assert len(trainer.accelerator.backward_calls) == 2
    assert duration_predictor.calls == 2
    assert model.flow_weight.item() > 0.
    assert duration_predictor.duration_weight.item() > 0.

    names = [name for name, _, _ in trainer.writer.scalars]
    assert 'duration loss' in names
    assert 'flow loss' in names
    assert 'loss' in names


def test_duration_loss_weight_and_optional_path_have_expected_gradients():
    trainer_module = load_trainer_without_audio_extensions()

    flow = torch.tensor(1., requires_grad=True)
    duration = torch.tensor(2., requires_grad=True)
    combined = trainer_module.combine_training_losses(flow, duration, 0.25)
    combined.backward()

    assert flow.grad == 1.
    assert duration.grad == 0.25

    flow_without_duration = torch.tensor(3., requires_grad=True)
    unchanged = trainer_module.combine_training_losses(flow_without_duration)
    assert unchanged is flow_without_duration


def test_train_example_passes_duration_predictor_to_trainer():
    tree = ast.parse(TRAIN_EXAMPLE_PATH.read_text(encoding='utf-8'))

    trainer_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'E2Trainer'
    ]

    assert len(trainer_calls) == 1
    assert any(
        keyword.arg == 'duration_predictor'
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == 'duration_predictor'
        for keyword in trainer_calls[0].keywords
    )
