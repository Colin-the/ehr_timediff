"""Reading and writing the pipeline's on-disk formats, plus the train/test split.

Deliberately imports nothing from the pipeline's Managers: those pull in cupy and
cuML at import time. A cached arm is a plain `torch.save`d
`list[(Tensor[24,7] float32, Tensor[2] int64)]`, so torch alone can read it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .layout import N_HOURS, N_VITALS


def load_arm(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Cached arm pickle -> (values (N,24,7) float32, labels (N,2) int64 [icu, mortality])."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached arm at {path}. Build it in the pipeline first "
            "(Experiments/apply_dataset_filter.py <dataset> <aggregation>)."
        )
    # weights_only=False: these are tuples of tensors, not a state dict, and newer
    # torch defaults to weights_only=True. Only load pickles you produced yourself.
    data = torch.load(path, map_location="cpu", weights_only=False)
    if len(data) == 0:
        return np.zeros((0, N_HOURS, N_VITALS), np.float32), np.zeros((0, 2), np.int64)
    values = torch.stack([entry[0] for entry in data]).numpy().astype(np.float32)
    labels = torch.stack([entry[1] for entry in data]).numpy().astype(np.int64)
    if values.shape[1:] != (N_HOURS, N_VITALS):
        raise ValueError(f"{path}: expected 24×7 records, got {values.shape[1:]}")
    if labels.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 labels [icu, mortality], got {labels.shape[1]}")
    return values, labels


def load_ids(path: Path, expected_len: int) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no admission-id sidecar at {path}. Generate it in the pipeline with "
            "`python rerun/regen_admission_ids.py <aggregation>`. It is required: the "
            "train/test split is keyed on admission id so it is identical across arms."
        )
    ids = np.load(path).astype(np.int64)
    if len(ids) != expected_len:
        raise ValueError(f"{path} holds {len(ids):,} ids but the arm has {expected_len:,} records")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: admission ids are not unique")
    return ids


def save_pipeline_dataset(values: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Write records exactly as DatasetEHR.save would, so DatasetEHR.load reads them.

    Same dtypes as RecordEHR.to_tensor: float32 timeseries, int64 labels.
    """
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    data = [(torch.from_numpy(values[i].copy()), torch.from_numpy(labels[i].copy())) for i in range(len(values))]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64: a well-mixed, platform-independent 64-bit hash."""
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def in_test_split(ids: np.ndarray, test_fraction: float = 0.2, salt: int = 2023) -> np.ndarray:
    """True where an admission belongs to the held-out test split.

    Decided by a hash of the admission id alone, so the same admission is on the
    same side of the split in every arm, aggregation and run. That gives two
    things a per-arm random split cannot: no arm's generator ever trains on a
    patient that another arm is tested on, and a common test set across arms.

    Not stratified by label. Stratifying would make membership depend on which
    other patients are in the arm, which breaks cross-arm consistency. Because
    the hash is independent of the label, prevalence matches in expectation.
    """
    with np.errstate(over="ignore"):
        mixed = _splitmix64(np.asarray(ids, dtype=np.int64).astype(np.uint64) ^ _splitmix64(np.array([salt]))[0])
    unit = (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return unit < test_fraction
