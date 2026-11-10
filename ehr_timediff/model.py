"""Builds the TimeDiff model from the vendored code, for training and for sampling.

Hyperparameter defaults are the paper's (2310.15290, and etdiff_train.py):
bidirectional LSTM, hidden 256, 3 layers, embed 64, time embedding 256, mixed
diffusion with 1000 steps and a cosine schedule, categorical loss weight 0.8.
"""
from __future__ import annotations

import random
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .arms import PROJECT_ROOT
from .layout import LAYOUT, N_HOURS, Layout

# The vendored package lives at <project>/timediff; make `import timediff...` work
# no matter where the CLI is started from.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timediff.models.ETDiff.blocks import RNN  # noqa: E402
from timediff.models.ETDiff.mixed_diffusion import MixedDiffusion  # noqa: E402


@dataclass
class HParams:
    model: str = "lstm"
    hidden: int = 256
    num_layers: int = 3
    bidirectional: bool = True
    embed_dim: int = 64
    time_dim: int = 256
    dropout: float = 0.0
    timesteps: int = 1000
    loss_lambda: float = 0.8
    batch_size: int = 32
    learning_rate: float = 8e-5
    gradient_accumulate_every: int = 2
    ema_decay: float = 0.995
    num_steps: int = 2000
    save_every: int = 5000
    seed: int = 2023

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HParams":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def build_diffusion(hp: HParams, layout: Layout = LAYOUT) -> MixedDiffusion:
    # MixedDiffusion feeds the network numericals plus one one-hot block per
    # categorical channel: 7 + 9*2 = 25 channels for this layout.
    model_channels = layout.n_channels + len(layout.categorical)
    network = RNN(
        input_channels=model_channels,
        hidden_channels=hp.hidden,
        output_channels=model_channels,
        layers=hp.num_layers,
        model=hp.model,
        dropout=hp.dropout,
        bidirectional=hp.bidirectional,
        self_condition=False,
        embed_dim=hp.embed_dim,
        time_dim=hp.time_dim,
    )
    return MixedDiffusion(
        model=network,
        channels=model_channels,
        seq_length=N_HOURS,
        timesteps=hp.timesteps,
        auto_normalize=True,            # [0,1] -> [-1,1]; export.py stops at [0,1]
        numerical_features_indices=layout.numerical,
        categorical_features_indices=layout.categorical,
        categorical_num_classes=layout.categorical_num_classes,
        loss_lambda=hp.loss_lambda,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
