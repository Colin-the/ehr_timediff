"""Turn one pipeline arm into a TimeDiff training file.

    python -m ehr_timediff.export --dataset mimic-iii --aggregation mean --arm raw

Reads   <data_root>/<dataset>/<aggregation>/<arm pickle> + <arm>_admission_ids.npy
Writes  <runs_root>/<dataset>/<aggregation>/<arm>/prepared.pt
        <runs_root>/<dataset>/<aggregation>/<arm>/export_manifest.json

prepared.pt holds the training tensor TimeDiff consumes, the scaler needed to
undo it, and the real train/test records (in pipeline format) that evaluation
compares synthetic data against. Rebuilt whenever the source pickle's sha256
changes, which is the one cache-invalidation signal the pipeline itself lacks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from . import config
from .arms import (VITAL_NAMES, arm_ids_path, arm_pickle_path, normalize_aggregation, normalize_arm,
                   processed_records_path, run_dir)
from .io import load_arm, load_ids, sha256, in_test_split
from .layout import LAYOUT, Scaler, to_timediff
from .masks import MASK_MODES, build_mask

PREPARED_NAME = "prepared.pt"
SCALINGS = ("minmax", "robust")
DEFAULT_CLIP_PERCENTILES = (0.5, 99.5)


def _missingness(mask: np.ndarray) -> dict:
    if len(mask) == 0:
        return {}
    return {v: round(float(1 - mask[:, :, i].mean()), 4) for i, v in enumerate(VITAL_NAMES)}


def _prevalence(labels: np.ndarray) -> dict:
    if len(labels) == 0:
        return {"icu": None, "mortality": None}
    return {"icu": round(float(labels[:, 0].mean()), 4), "mortality": round(float(labels[:, 1].mean()), 4)}


def _scaling_warnings(values: np.ndarray, mask: np.ndarray, scaler: Scaler) -> list:
    """Flag vitals whose min/max are set by a few extreme readings.

    Min/max scaling is what TimeDiff does, and it is kept here so the generator is
    the paper's. But in an unfiltered arm a single charting error (heart rate
    9999) stretches the range so far that every real reading lands in a sliver of
    [0, 1]. That is part of what a filter changes, so it is reported rather than
    silently fixed.
    """
    warnings = []
    for i, vital in enumerate(VITAL_NAMES):
        observed = values[:, :, i][mask[:, :, i]]
        if observed.size < 10:
            continue
        lo, hi = np.percentile(observed, [0.1, 99.9])
        spread = max(hi - lo, 1e-9)
        if (scaler.maximum[i] - scaler.minimum[i]) > 5 * spread:
            warnings.append(
                f"{vital}: training range [{scaler.minimum[i]:.4g}, {scaler.maximum[i]:.4g}] is "
                f">5x the 0.1-99.9 percentile range [{lo:.4g}, {hi:.4g}]. Real readings will "
                "occupy a sliver of the model's scale, and on test data the model then failed to "
                "learn the vitals at all. Consider --scaling robust."
            )
    return warnings


def _clipped_cells(values: np.ndarray, mask: np.ndarray, scaler: Scaler) -> int:
    if not scaler.clip:
        return 0
    outside = (values < scaler.minimum) | (values > scaler.maximum)
    return int((outside & mask).sum())


def prepare_arm(data_root: Path, runs_root: Path, dataset: str, aggregation: str, arm: str,
                mask_mode: str = "auto", force_zero_mask: bool = False, test_fraction: float = 0.2,
                split_salt: int = 2023, processed_records: Optional[Path] = None,
                force: bool = False, scaling: str = "minmax",
                clip_percentiles: Tuple[float, float] = DEFAULT_CLIP_PERCENTILES) -> Path:
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}")
    percentiles = tuple(float(q) for q in clip_percentiles) if scaling == "robust" else None
    aggregation, arm = normalize_aggregation(aggregation), normalize_arm(arm)
    out_dir = run_dir(runs_root, dataset, aggregation, arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = out_dir / PREPARED_NAME

    source = arm_pickle_path(data_root, dataset, aggregation, arm)
    source_sha = sha256(source) if source.exists() else None
    if prepared_path.exists() and not force:
        meta = torch.load(prepared_path, map_location="cpu", weights_only=False)["meta"]
        if source_sha is None:
            # e.g. runs/ copied to a machine without the pipeline data: keep what we have.
            print(f"[export] WARNING {source} not found; keeping existing {prepared_path} unchecked")
            return prepared_path
        same_mask = mask_mode == "auto" or mask_mode == meta["mask_mode"]
        same_scaling = meta.get("scaling", "minmax") == scaling and \
            (percentiles is None or tuple(meta.get("clip_percentiles") or ()) == percentiles)
        if meta["source_sha256"] == source_sha and meta["split_salt"] == split_salt \
                and meta["test_fraction"] == test_fraction and same_mask and same_scaling:
            print(f"[export] {prepared_path} is current (source sha matches); skipping")
            return prepared_path
        print("[export] source pickle, split, mask or scaling settings changed; rebuilding")

    t0 = time.time()
    values, labels = load_arm(source)
    ids = load_ids(arm_ids_path(data_root, dataset, aggregation, arm), len(values))
    if processed_records is None:
        processed_records = processed_records_path(data_root, dataset)
    mask, mask_used = build_mask(values, ids, aggregation, arm, mask_mode, processed_records, force_zero_mask)

    is_test = in_test_split(ids, test_fraction, split_salt)
    train, test = ~is_test, is_test
    if train.sum() == 0:
        raise ValueError("the arm's training split is empty")

    scaler = Scaler.fit(values[train], mask[train], percentiles)
    x_train = to_timediff(values[train], mask[train], labels[train], scaler)

    prepared = {
        "x_train": torch.from_numpy(x_train),
        "scaler": scaler.to_dict(),
        "layout": LAYOUT.describe(),
        "train": {"ids": ids[train], "values": values[train], "mask": mask[train], "labels": labels[train]},
        "test": {"ids": ids[test], "values": values[test], "mask": mask[test], "labels": labels[test]},
        "meta": {
            "dataset": dataset, "aggregation": aggregation, "arm": arm,
            "source": str(source), "source_sha256": source_sha,
            "mask_mode": mask_used, "test_fraction": test_fraction, "split_salt": split_salt,
            "scaling": scaling, "clip_percentiles": list(percentiles) if percentiles else None,
        },
    }
    torch.save(prepared, prepared_path)

    manifest = {
        **prepared["meta"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_total": int(len(values)), "n_train": int(train.sum()), "n_test": int(test.sum()),
        "label_prevalence_train": _prevalence(labels[train]),
        "label_prevalence_test": _prevalence(labels[test]),
        "missing_rate_train": _missingness(mask[train]),
        "observed_zero_cells_train": int(((values[train] == 0) & mask[train]).sum()),
        "scaler": scaler.to_dict(),
        "layout": LAYOUT.describe(),
        "clipped_cells_train": int(_clipped_cells(values[train], mask[train], scaler)),
        "warnings": _scaling_warnings(values[train], mask[train], scaler) if scaling == "minmax" else [],
    }
    if len(values) < 5000:
        manifest["warnings"].append(
            f"only {len(values):,} records; a diffusion model this size will tend to memorise "
            "them. Read the privacy metrics before trusting this arm's utility numbers."
        )
    with open(out_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[export] {dataset}/{aggregation}/{arm}: {len(values):,} records "
          f"({train.sum():,} train / {test.sum():,} test), mask={mask_used}, "
          f"{time.time() - t0:.1f}s -> {prepared_path}")
    for w in manifest["warnings"]:
        print(f"[export] WARNING {w}")
    return prepared_path


def load_prepared(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="mimic-iii", help="mimic-iii | mimic-iv | eicu (pipeline folder name)")
    parser.add_argument("--aggregation", default="mean")
    parser.add_argument("--arm", default="raw", help="'raw' or a filter name, e.g. 'long gap' / long_gap")
    parser.add_argument("--data-root", default=None, help="pipeline Data/ directory")
    parser.add_argument("--runs-root", default=None)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arm_arguments(parser)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default="auto")
    parser.add_argument("--force-zero-mask", action="store_true",
                        help="allow value==0 as 'missing' for deviation aggregations")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-salt", type=int, default=2023)
    parser.add_argument("--processed-records", default=None,
                        help="override path to processed_record_ehr.pkl (exact masks)")
    parser.add_argument("--scaling", choices=SCALINGS, default="minmax",
                        help="minmax = TimeDiff's choice; robust = percentile range with clipping")
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=DEFAULT_CLIP_PERCENTILES,
                        metavar=("LO", "HI"), help="range for --scaling robust")
    parser.add_argument("--force", action="store_true", help="rebuild even if prepared.pt is current")
    args = parser.parse_args(argv)
    prepare_arm(config.data_root(args.data_root), config.runs_root(args.runs_root), args.dataset,
                args.aggregation, args.arm, args.mask_mode, args.force_zero_mask, args.test_fraction,
                args.split_salt, Path(args.processed_records) if args.processed_records else None, args.force,
                args.scaling, args.clip_percentiles)


if __name__ == "__main__":
    main()
