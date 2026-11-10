"""Train TimeDiff on one pipeline arm.

    python -m ehr_timediff.train --dataset mimic-iii --aggregation mean --arm raw --num-steps 700000

Exports the arm first if prepared.pt is missing. Resumes from model.pt if one
exists in the run directory (pass --restart to throw it away). The paper trains
for 700k steps; the default here is a smoke-test-sized 2000.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import config
from .arms import normalize_aggregation, normalize_arm, run_dir
from .export import PREPARED_NAME, add_arm_arguments, load_prepared, prepare_arm
from .model import HParams, build_diffusion, count_params, seed_everything

from timediff.models.ETDiff.et_diff import ETDiff  # noqa: E402  (path set up by .model)

MODEL_NAME = "model.pt"
TRAIN_CONFIG_NAME = "train_config.json"


class PreparedDataset(Dataset):
    """In-memory stand-in for TimeDiff's TimeSeriesDataset.

    The original cached to data/datasets/<name>/ relative to the working directory
    and never checked whether the cache was stale, so a second arm would silently
    train on the first arm's data. This one holds exactly the tensor it is given.

    ETDiff reads `.min`/`.max` only to unscale samples in a plotting hook that is
    disabled; unscaling happens in generate.py with the real Scaler.
    """

    def __init__(self, x: torch.Tensor):
        self.data = x.float()
        self.min = torch.zeros(1)
        self.max = torch.ones(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].clone()

    @property
    def channels(self):
        return self.data.shape[1]

    @property
    def seq_length(self):
        return self.data.shape[2]


def train_arm(directory: Path, hp: HParams, restart: bool = False, wandb_project: str = None) -> Path:
    prepared = load_prepared(directory / PREPARED_NAME)
    dataset = PreparedDataset(prepared["x_train"])
    if len(dataset) < hp.batch_size:
        raise ValueError(f"training split has {len(dataset)} records, fewer than batch size {hp.batch_size}")

    checkpoint = directory / MODEL_NAME
    config_path = directory / TRAIN_CONFIG_NAME
    if checkpoint.exists() and not restart and config_path.exists():
        previous_config = json.load(open(config_path))
        if previous_config.get("prepared_meta") != prepared["meta"]:
            raise SystemExit("cannot resume: prepared.pt was rebuilt (source pickle, split or mask changed) "
                             "since model.pt was trained on it. Pass --restart to train on the new data.")
        previous = HParams.from_dict(previous_config["hparams"])
        # Architecture must match to resume; the step budget may grow.
        for key in ("model", "hidden", "num_layers", "bidirectional", "embed_dim", "time_dim", "timesteps"):
            if getattr(previous, key) != getattr(hp, key):
                raise SystemExit(f"cannot resume: {key} changed ({getattr(previous, key)} -> {getattr(hp, key)}). "
                                 "Pass --restart to start over.")
    if restart and checkpoint.exists():
        checkpoint.unlink()

    seed_everything(hp.seed)
    diffusion = build_diffusion(hp)

    wandb = None
    if wandb_project:
        import wandb as _wandb  # optional dependency
        _wandb.init(project=wandb_project, name=directory.name, config=hp.to_dict())
        wandb = _wandb

    trainer = ETDiff(
        diffusion_model=diffusion,
        dataset=dataset,
        sample_every=hp.save_every,          # ETDiff saves a checkpoint every `sample_every` steps
        train_batch_size=hp.batch_size,
        diff_lr=hp.learning_rate,
        diff_num_steps=hp.num_steps,
        gradient_accumulate_every=hp.gradient_accumulate_every,
        ema_decay=hp.ema_decay,
        amp=False,
        wandb=wandb,
        check_point_path=str(checkpoint),
        run_id=directory.name,
    )
    if checkpoint.exists():
        trainer.load(str(checkpoint))
        print(f"[train] resuming from step {trainer.step:,}/{hp.num_steps:,}")

    run_config = {
        "hparams": hp.to_dict(),
        "n_train": len(dataset),
        "parameters": count_params(diffusion),
        "device": str(trainer.device),
        "torch": torch.__version__,
        "prepared_meta": prepared["meta"],
    }
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[train] {directory}: {len(dataset):,} records, {run_config['parameters']:,} parameters, "
          f"device {trainer.device}, {hp.num_steps:,} steps")

    t0 = time.time()
    trainer.train()  # saves to model.pt at the end, and every save_every steps
    run_config["train_seconds_this_session"] = round(time.time() - t0, 1)
    run_config["final_step"] = trainer.step
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[train] done at step {trainer.step:,} in {time.time() - t0:.0f}s -> {checkpoint}")
    return checkpoint


def add_hparam_arguments(parser: argparse.ArgumentParser) -> None:
    d = HParams()
    parser.add_argument("--num-steps", type=int, default=d.num_steps, help="paper: 700000")
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--learning-rate", type=float, default=d.learning_rate)
    parser.add_argument("--hidden", type=int, default=d.hidden)
    parser.add_argument("--num-layers", type=int, default=d.num_layers)
    parser.add_argument("--rnn", choices=["lstm", "gru"], default=d.model)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=d.embed_dim)
    parser.add_argument("--time-dim", type=int, default=d.time_dim)
    parser.add_argument("--dropout", type=float, default=d.dropout)
    parser.add_argument("--timesteps", type=int, default=d.timesteps, help="diffusion steps")
    parser.add_argument("--loss-lambda", type=float, default=d.loss_lambda)
    parser.add_argument("--gradient-accumulate-every", type=int, default=d.gradient_accumulate_every)
    parser.add_argument("--ema-decay", type=float, default=d.ema_decay)
    parser.add_argument("--save-every", type=int, default=d.save_every)
    parser.add_argument("--seed", type=int, default=d.seed)


def hparams_from_args(args) -> HParams:
    return HParams(
        model=args.rnn, hidden=args.hidden, num_layers=args.num_layers,
        bidirectional=not args.no_bidirectional, embed_dim=args.embed_dim, time_dim=args.time_dim,
        dropout=args.dropout, timesteps=args.timesteps, loss_lambda=args.loss_lambda,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every, ema_decay=args.ema_decay,
        num_steps=args.num_steps, save_every=args.save_every, seed=args.seed,
    )


def resolve_run_dir(args) -> Path:
    """Run directory for the arm; exports it first if prepared.pt is missing."""
    runs = config.runs_root(args.runs_root)
    directory = run_dir(runs, args.dataset, normalize_aggregation(args.aggregation), normalize_arm(args.arm))
    if not (directory / PREPARED_NAME).exists():
        print("[train] no prepared.pt yet; exporting the arm first")
        prepare_arm(config.data_root(args.data_root), runs, args.dataset, args.aggregation, args.arm)
    return directory


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arm_arguments(parser)
    add_hparam_arguments(parser)
    parser.add_argument("--restart", action="store_true", help="discard model.pt and train from scratch")
    parser.add_argument("--wandb-project", default=None, help="log to Weights & Biases (optional)")
    args = parser.parse_args(argv)
    train_arm(resolve_run_dir(args), hparams_from_args(args), args.restart, args.wandb_project)


if __name__ == "__main__":
    sys.exit(main())
