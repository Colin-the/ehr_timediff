"""A fake pipeline Data/ directory, for testing without MIMIC access.

    python -m ehr_timediff.fake --out runs/_fake_data --n 3000

Builds list-per-cell records shaped like processed_record_ehr.pkl (charting
gaps, several readings per hour, the odd impossible value), then runs them
through the *vendored pipeline's own* aggregation and filter code to write the
arm pickles and admission-id sidecars. The caches therefore match what the real
pipeline would produce for the same records, which is what lets the exact-mask
replay be tested honestly.

Labels carry a deliberate signal (mortality rises with heart rate and falls with
SpO2) so TSTR/TRTR numbers are meaningfully above 0.5. Nothing here resembles
real patients beyond the rough vital ranges.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .arms import (VITAL_NAMES, VITALS, arm_ids_path, arm_pickle_path, ensure_pipeline_importable,
                   filter_function, normalize_aggregation, processed_records_path)
from .io import save_pipeline_dataset

DATASET = "fake"
# (typical value, spread, chance an hour has any reading)
PROFILE = {
    "heart rate": (85, 15, 0.95),
    "systolic blood pressure": (120, 18, 0.85),
    "diastolic blood pressure": (62, 10, 0.85),
    "mean blood pressure": (80, 12, 0.85),
    "respiration rate": (19, 4, 0.9),
    "temperature": (37.0, 0.6, 0.3),
    "oxygen saturation": (97, 2, 0.9),
}


def make_records(n: int, seed: int = 0):
    ensure_pipeline_importable()
    from Entities.ehr_record import RecordEHR  # pickled under this module path, like the real file

    rng = np.random.default_rng(seed)
    records = []
    for k in range(n):
        severity = rng.normal()
        frame = pd.DataFrame([[[] for _ in VITAL_NAMES] for _ in range(24)], index=range(24),
                             columns=list(VITAL_NAMES), dtype=object)
        # Some patients go dark on a vital for a long stretch (long missing segment),
        # some leave the unit for a few hours (long gap), and some are charted
        # densely enough to survive the high-invalid-data filter.
        dark = {v: (rng.integers(0, 14), rng.integers(10, 16)) if rng.random() < 0.25 else None for v in VITAL_NAMES}
        off_unit = (rng.integers(0, 20), rng.integers(2, 5)) if rng.random() < 0.3 else None
        dense = rng.random() < 0.15
        if dense:
            dark = {v: None for v in VITAL_NAMES}
        for v, (mu, sd, p_hour) in PROFILE.items():
            p_hour = 0.99 if dense else p_hour
            shift = {"heart rate": 8, "oxygen saturation": -1.5, "respiration rate": 2}.get(v, 0) * severity
            level = mu + shift + rng.normal(0, sd * 0.5)
            for h in range(24):
                if dark[v] is not None and dark[v][0] <= h < dark[v][0] + dark[v][1]:
                    continue
                if off_unit is not None and off_unit[0] <= h < off_unit[0] + off_unit[1]:
                    continue
                if rng.random() > p_hour:
                    continue
                readings = list(np.round(level + rng.normal(0, sd * 0.3, rng.integers(1, 4)), 1))
                if rng.random() < 0.01:          # charting error, outside the closed range
                    readings.append(float(VITALS[v][0][1] * 3))
                frame.at[h, v] = [float(x) for x in readings]
        mortality = int(rng.random() < 1 / (1 + np.exp(-(1.4 * severity - 2.2))))
        icu = int(rng.random() < 1 / (1 + np.exp(-(0.8 * severity - 0.3))))
        records.append(RecordEHR(100000 + 3 * k, frame, icu, mortality))
    return records


def build(out: Path, n: int, aggregations=("mean",), arms=("raw", "all vitals", "long gap", "high invalid data"),
          seed: int = 0) -> Path:
    ensure_pipeline_importable()
    from Managers.ehr_filter_manager import aggregate_filter

    records = make_records(n, seed)
    processed = processed_records_path(out, DATASET)
    processed.parent.mkdir(parents=True, exist_ok=True)
    with open(processed, "wb") as f:
        pickle.dump(records, f)

    for aggregation in aggregations:
        aggregation = normalize_aggregation(aggregation)
        aggregated = aggregate_filter(records, aggregation)
        for arm in arms:
            function, pre = filter_function(arm)
            if function is None:
                kept = aggregated
            elif pre:
                kept = aggregate_filter(function(records, VITALS)[0], aggregation)
            else:
                kept = function(aggregated, VITALS)[0]
            if not kept:
                print(f"[fake] {aggregation}/{arm}: filter kept nothing; not written")
                continue
            tensors = [r.to_tensor() for r in kept]
            values = np.stack([t[0].numpy() for t in tensors])
            labels = np.stack([t[1].numpy() for t in tensors])
            save_pipeline_dataset(values, labels, arm_pickle_path(out, DATASET, aggregation, arm))
            np.save(arm_ids_path(out, DATASET, aggregation, arm),
                    np.array([r.admission_id for r in kept], dtype=np.int64))
            print(f"[fake] {aggregation}/{arm}: {len(kept):,} records")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory to use as a fake pipeline Data/ root")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--aggregations", nargs="+", default=["mean"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    build(Path(args.out), args.n, args.aggregations, seed=args.seed)


if __name__ == "__main__":
    main()
