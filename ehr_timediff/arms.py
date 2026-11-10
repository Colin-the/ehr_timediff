"""What an "arm" is, and where the pipeline keeps its files.

An arm is one cached dataset the pipeline built: `raw` (no filter) or one of the
twelve filters, under one aggregation method. This module is the only place that
knows the pipeline's on-disk naming, so if that ever changes it changes here.

The filter table mirrors `rerun/_common.py` in the pipeline (the definition of
record: `Managers.ehr_filter_manager.FILTERS` omits 'all vitals' and carries no
pre/post-aggregation flags). `VITALS` is a seventh copy of the range table the
pipeline already keeps six of (bug register R-30); `tests/test_layout.py` parses
the vendored `Experiments/apply_dataset_filter.py` and fails if they drift.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = PROJECT_ROOT / "ehr_pipeline"

# Column order of every 24×7 matrix the pipeline writes (Processing/*.py
# VITAL_FEATURES, and the hardcoded column_index 0-6 in _remove_outliers).
VITAL_NAMES = (
    "heart rate",
    "systolic blood pressure",
    "diastolic blood pressure",
    "mean blood pressure",
    "respiration rate",
    "temperature",
    "oxygen saturation",
)

# Closed intervals, same as the pipeline (post 2026-08-30 fix, bug register F-02).
VITALS = {
    "heart rate": [(1, 600), "bpm"],
    "systolic blood pressure": [(1, 400), "mmHg"],
    "diastolic blood pressure": [(1, 300), "mmHg"],
    "mean blood pressure": [(1, 300), "mmHg"],
    "respiration rate": [(1, 70), "breaths/min"],
    "temperature": [(21, 50), "C"],
    "oxygen saturation": [(1, 100), "%"],
}

AGGREGATIONS = ("mean", "median", "standard deviation", "mean deviation", "maximum deviation")

# Aggregations where a cell value of exactly 0 can only mean "nothing measured".
# Every vital's lower bound is >= 1, so a mean or median of real readings is never
# 0 after outlier filtering. The deviation methods give 0 for any hour with a
# single reading, so for them 0 is ambiguous. See masks.py.
ZERO_MEANS_MISSING = ("mean", "median")

# arm name -> (filter function name in ehr_filter_manager, runs before aggregation?)
ARMS: Dict[str, Tuple[Optional[str], bool]] = {
    "raw": (None, False),
    "heart rate": ("heart_rate_filter", True),
    "systolic blood pressure": ("systolic_blood_pressure_filter", True),
    "diastolic blood pressure": ("diastolic_blood_pressure_filter", True),
    "mean blood pressure": ("mean_blood_pressure_filter", True),
    "respiration rate": ("respiration_rate_filter", True),
    "temperature": ("temperature_filter", True),
    "oxygen saturation": ("oxygen_saturation_filter", True),
    "fill missing data": ("fill_missing_data_filter", False),
    "long missing segment": ("long_missing_segment_filter", False),
    "long gap": ("long_gap_filter", False),
    "high invalid data": ("high_invalid_data_filter", False),
    "all vitals": ("all_vitals_filter", True),
}


def normalize_arm(arm: str) -> str:
    """Accept 'long_gap', 'Long Gap' or 'long gap' and return the canonical name."""
    name = arm.replace("_", " ").strip().lower()
    if name not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    return name


def normalize_aggregation(aggregation: str) -> str:
    name = aggregation.replace("_", " ").strip().lower()
    if name not in AGGREGATIONS:
        # The pipeline silently falls back to mean on an unknown method. Don't.
        raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    return name


def arm_slug(arm: str) -> str:
    """On-disk spelling, matching dataset_manager's `.replace(' ', '_').lower()`."""
    return normalize_arm(arm).replace(" ", "_")


def aggregation_slug(aggregation: str) -> str:
    return normalize_aggregation(aggregation).replace(" ", "_")


def arm_pickle_path(data_root: Path, dataset: str, aggregation: str, arm: str) -> Path:
    """The cached `list[(Tensor[24,7], Tensor[icu, mortality])]` for one arm."""
    directory = Path(data_root) / dataset / normalize_aggregation(aggregation)
    if normalize_arm(arm) == "raw":
        return directory / "raw_dataset_ehr.pkl"
    return directory / f"{arm_slug(arm)}_filtered_dataset_ehr.pkl"


def arm_ids_path(data_root: Path, dataset: str, aggregation: str, arm: str) -> Path:
    """Admission-id sidecar written by the pipeline's rerun/regen_admission_ids.py."""
    directory = Path(data_root) / dataset / normalize_aggregation(aggregation)
    return directory / f"{arm_slug(arm)}_admission_ids.npy"


def processed_records_path(data_root: Path, dataset: str) -> Path:
    """Pre-aggregation list-per-cell records; the only source of exact masks."""
    return Path(data_root) / dataset / "processed_record_ehr.pkl"


def run_dir(runs_root: Path, dataset: str, aggregation: str, arm: str) -> Path:
    return Path(runs_root) / dataset / aggregation_slug(aggregation) / arm_slug(arm)


def ensure_pipeline_importable() -> None:
    """Put the vendored pipeline on sys.path.

    Needed to import its filters, and also to unpickle `processed_record_ehr.pkl`,
    whose records were pickled as `Entities.ehr_record.RecordEHR`.
    """
    path = str(PIPELINE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def filter_function(arm: str) -> Tuple[Optional[Callable], bool]:
    """(filter callable or None for raw, pre_aggregate flag)."""
    function_name, pre_aggregate = ARMS[normalize_arm(arm)]
    if function_name is None:
        return None, False
    ensure_pipeline_importable()
    from Managers import ehr_filter_manager  # noqa: E402  (vendored, CPU-only)

    return getattr(ehr_filter_manager, function_name), pre_aggregate
