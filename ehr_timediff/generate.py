"""Sample synthetic patients from a trained arm, in the pipeline's own format.

    python -m ehr_timediff.generate --dataset mimic-iii --aggregation mean --arm raw

Writes into the arm's run directory:
  <arm>_synthetic_dataset_ehr.pkl   list[(Tensor[24,7] float32, Tensor[icu, mortality] int64)]
                                    Same format as the pipeline's cached arms, so
                                    DatasetEHR.load() and every classifier in the
                                    pipeline work on it unchanged.
  synthetic_mask.npy                (N, 24, 7) bool, which cells the model generated as measured
  synthetic_raw.npy                 (N, 16, 24) raw model output, before unscaling
  generate_manifest.json            counts, label prevalence and missingness vs. the real data

Default sample count is the size of the arm's real training split, so synthetic
and real classifiers are trained on the same amount of data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from tqdm import tqdm

from .arms import VITAL_NAMES, arm_slug
from .export import PREPARED_NAME, add_arm_arguments, load_prepared
from .io import save_pipeline_dataset
from .layout import Scaler, from_timediff
from .model import HParams, build_diffusion, seed_everything
from .train import MODEL_NAME, TRAIN_CONFIG_NAME, resolve_run_dir


def synthetic_pickle_name(arm: str) -> str:
    return f"{arm_slug(arm)}_synthetic_dataset_ehr.pkl"


def load_sampler(directory: Path, device: torch.device):
    hp = HParams.from_dict(json.load(open(directory / TRAIN_CONFIG_NAME))["hparams"])
    diffusion = build_diffusion(hp)
    ema = EMA(diffusion, beta=hp.ema_decay, update_every=10)
    state = torch.load(directory / MODEL_NAME, map_location="cpu", weights_only=False)
    ema.load_state_dict(state["ema"])
    sampler = ema.ema_model.to(device).eval()
    return sampler, hp, int(state["diff_step"])


@torch.no_grad()
def sample(sampler, n: int, batch_size: int) -> np.ndarray:
    chunks, remaining = [], n
    with tqdm(total=n, desc="sampling patients") as bar:
        while remaining > 0:
            b = min(batch_size, remaining)
            chunks.append(sampler.sample(batch_size=b).cpu().numpy())
            remaining -= b
            bar.update(b)
    return np.concatenate(chunks, axis=0)


def generate_arm(directory: Path, num_samples: int = None, batch_size: int = 1000, seed: int = 0,
                 device: str = None) -> Path:
    prepared = load_prepared(directory / PREPARED_NAME)
    meta = prepared["meta"]
    scaler = Scaler.from_dict(prepared["scaler"])
    n = num_samples or len(prepared["train"]["labels"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    seed_everything(seed)
    sampler, hp, step = load_sampler(directory, device)
    if step < hp.num_steps:
        print(f"[generate] WARNING checkpoint is at step {step:,} of {hp.num_steps:,}; training did not finish")

    t0 = time.time()
    raw = sample(sampler, n, batch_size)
    values, mask, labels, agreement = from_timediff(raw, scaler)

    out_pickle = directory / synthetic_pickle_name(meta["arm"])
    save_pipeline_dataset(values, labels, out_pickle)
    np.save(directory / "synthetic_mask.npy", mask)
    np.save(directory / "synthetic_raw.npy", raw.astype(np.float32))

    real_labels, real_mask = prepared["train"]["labels"], prepared["train"]["mask"]
    manifest = {
        **meta,
        "checkpoint_step": step,
        "n_synthetic": int(n),
        "sample_seed": seed,
        "sample_seconds": round(time.time() - t0, 1),
        "label_prevalence": {
            "real_train": {"icu": float(real_labels[:, 0].mean()), "mortality": float(real_labels[:, 1].mean())},
            "synthetic": {"icu": float(labels[:, 0].mean()), "mortality": float(labels[:, 1].mean())},
        },
        # 1.0 means the model kept each patient's label constant across all 24 hours,
        # as it is in the real data. Low values mean the labels are unreliable.
        "label_hour_agreement_mean": {"icu": float(agreement[:, 0].mean()), "mortality": float(agreement[:, 1].mean())},
        "missing_rate": {
            "real_train": {v: float(1 - real_mask[:, :, i].mean()) for i, v in enumerate(VITAL_NAMES)},
            "synthetic": {v: float(1 - mask[:, :, i].mean()) for i, v in enumerate(VITAL_NAMES)},
        },
        "output": str(out_pickle),
    }
    with open(directory / "generate_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[generate] {n:,} synthetic patients in {time.time() - t0:.0f}s -> {out_pickle}")
    print(f"[generate] mortality prevalence real {manifest['label_prevalence']['real_train']['mortality']:.3f} "
          f"vs synthetic {manifest['label_prevalence']['synthetic']['mortality']:.3f}")
    return out_pickle


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arm_arguments(parser)
    parser.add_argument("--num-samples", type=int, default=None, help="default: size of the real training split")
    parser.add_argument("--batch-size", type=int, default=1000, help="patients sampled per reverse-diffusion pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    generate_arm(resolve_run_dir(args), args.num_samples, args.batch_size, args.seed, args.device)


if __name__ == "__main__":
    main()
