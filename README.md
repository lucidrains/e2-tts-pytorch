
<img src="./e2-tts.png" width="400px"></img>

## E2 TTS - Pytorch (wip)

Implementation of E2-TTS, <a href="https://arxiv.org/abs/2406.18009v1">Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS</a>, in Pytorch

You can chat with other researchers about this work <a href="https://discord.gg/XXHAarGSEH">here</a>

## Install

```bash
$ pip install e2-tts-pytorch
```

## Usage

```python
import torch
from e2_tts_pytorch import (
    E2TTS,
    DurationPredictor
)

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 512,
        depth = 2,
    )
)

x = torch.randn(1, 1024, 512)
duration = torch.randn(1,)

loss = duration_predictor(x)
loss.backward()

e2tts = E2TTS(
    duration_predictor = duration_predictor,
    transformer = dict(
        dim = 512,
        depth = 4,
        skip_connect_type = 'concat'
    ),
)

loss = e2tts(x)
loss.backward()

sampled = e2tts.sample(x)
```

## Citations

```bibtex
@inproceedings{Eskimez2024E2TE,
    title   = {E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS},
    author  = {Sefik Emre Eskimez and Xiaofei Wang and Manthan Thakker and Canrun Li and Chung-Hsien Tsai and Zhen Xiao and Hemin Yang and Zirun Zhu and Min Tang and Xu Tan and Yanqing Liu and Sheng Zhao and Naoyuki Kanda},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270738197}
}
```
