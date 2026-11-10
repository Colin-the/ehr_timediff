"""Which cells of a cached arm were actually measured.

The cached pickles do not say. `RecordEHR.to_tensor` turns every missing hour
into 0.0, so the only trace of missingness left is the zeros. Two ways back:

exact  Replay the arm from `processed_record_ehr.pkl` (the list-per-cell records
       before aggregation) through the same filter and the same pre/post-
       aggregation order the pipeline used, and read off which aggregated cells
       are NaN. Correct for every aggregation. Costs one aggregation pass
       (a few minutes for ~46k records) and needs the processed records file.

zero   Treat value == 0 as missing. Correct for mean and median, because every
       vital's lower bound is >= 1 so a real mean is never 0. Wrong for the three
       deviation aggregations, where an hour with a single reading has a
       deviation of exactly 0. Refused for those unless forced.

Either way the result is validated against the cached tensors: every cell the
mask calls missing must hold 0, or the cache and the replay disagree.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .arms import (VITALS, ZERO_MEANS_MISSING, ensure_pipeline_importable, filter_function,
                   normalize_aggregation, normalize_arm)

MASK_MODES = ("auto", "exact", "zero")


def zero_mask(values: np.ndarray, aggregation: str, force: bool = False) -> np.ndarray:
    aggregation = normalize_aggregation(aggregation)
    if aggregation not in ZERO_MEANS_MISSING and not force:
        raise ValueError(
            f"'{aggregation}' gives 0 for any hour with a single reading, so 0 cannot be "
            "read as 'missing'. Use mask mode 'exact' (needs processed_record_ehr.pkl) "
            "or pass --force-zero-mask to accept the conflation."
        )
    return np.asarray(values) != 0


def exact_mask(processed_records: Path, aggregation: str, arm: str, ids: np.ndarray) -> np.ndarray:
    """Replay one arm and return its (N, 24, 7) observed mask, in `ids` order."""
    aggregation = normalize_aggregation(aggregation)
    arm = normalize_arm(arm)
    ensure_pipeline_importable()
    from Managers.ehr_filter_manager import aggregate_filter  # noqa: E402

    with open(processed_records, "rb") as f:
        records = pickle.load(f)

    function, pre_aggregate = filter_function(arm)
    # Same branching as dataset_manager.create_filter_dataset.
    if function is None:
        out = aggregate_filter(records, aggregation)
    elif pre_aggregate:
        filtered, _ = function(records, VITALS)
        out = aggregate_filter(filtered, aggregation)
    else:
        out = aggregate_filter(records, aggregation)
        out, _ = function(out, VITALS)

    by_id = {int(r.admission_id): r for r in out}
    missing = [int(i) for i in ids if int(i) not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing):,} admission ids in the arm's sidecar are not produced by "
            f"replaying '{arm}' (e.g. {missing[:5]}). The cache and the filter code disagree."
        )
    mask = np.stack([~by_id[int(i)].timeseries.isna().to_numpy() for i in ids])
    return mask.astype(bool)


def build_mask(values: np.ndarray, ids: np.ndarray, aggregation: str, arm: str, mode: str,
               processed_records: Optional[Path], force_zero: bool = False) -> Tuple[np.ndarray, str]:
    """Pick a mask strategy, build it, validate it. Returns (mask, mode actually used)."""
    if mode not in MASK_MODES:
        raise ValueError(f"mask mode must be one of {MASK_MODES}")
    have_records = processed_records is not None and Path(processed_records).exists()
    if mode == "exact" and not have_records:
        raise FileNotFoundError(f"mask mode 'exact' needs {processed_records}")
    if mode == "exact" or (mode == "auto" and have_records):
        mask, used = exact_mask(Path(processed_records), aggregation, arm, ids), "exact"
    else:
        mask, used = zero_mask(values, aggregation, force=force_zero), "zero"
    validate_mask(values, mask)
    return mask, used


def validate_mask(values: np.ndarray, mask: np.ndarray) -> None:
    if mask.shape != values.shape:
        raise ValueError(f"mask shape {mask.shape} != values shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("cached values contain NaN/inf; to_tensor should have zero-filled them")
    bad = (~mask) & (values != 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum()):,} cells are marked missing but hold a non-zero value. The "
            "cached pickle was built with different filter code than the vendored copy; "
            "rebuild the arm or re-vendor the pipeline."
        )
