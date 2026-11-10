"""Run the whole chain for one arm: export -> train -> generate -> evaluate.

    # one arm, explicitly
    python -m ehr_timediff.run --dataset mimic-iii --aggregation mean --arm raw --num-steps 700000

    # row N of config/arms.yaml (what the SLURM array does)
    python -m ehr_timediff.run --index 3

    # just list the rows
    python -m ehr_timediff.run --list

Every stage skips or resumes sensibly when re-run: export rebuilds only if the
source pickle changed, training resumes from model.pt, generate and evaluate are
cheap enough to just redo. --stages picks a subset (e.g. --stages generate evaluate).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import config
from .arms import PROJECT_ROOT, normalize_aggregation, normalize_arm, run_dir
from .model import HParams

STAGES = ("export", "train", "generate", "evaluate")
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "arms.yaml"


def load_rows(path: Path) -> list:
    with open(path) as f:
        spec = yaml.safe_load(f)
    defaults = spec.get("defaults", {}) or {}
    rows = []
    for row in spec["runs"]:
        merged = {**defaults, **row}
        merged["aggregation"] = normalize_aggregation(merged["aggregation"])
        merged["arm"] = normalize_arm(merged["arm"])
        rows.append(merged)
    return rows


def run(row: dict, stages, data_root, runs_root) -> None:
    from . import evaluate, export, generate, train

    dataset, aggregation, arm = row["dataset"], row["aggregation"], row["arm"]
    directory = run_dir(runs_root, dataset, aggregation, arm)
    print(f"===== {dataset} / {aggregation} / {arm}  ->  {directory}")
    if "export" in stages:
        export.prepare_arm(data_root(), runs_root, dataset, aggregation, arm,
                           mask_mode=row.get("mask_mode", "auto"),
                           force_zero_mask=row.get("force_zero_mask", False),
                           scaling=row.get("scaling", "minmax"),
                           clip_percentiles=row.get("clip_percentiles", export.DEFAULT_CLIP_PERCENTILES))
    if "train" in stages:
        hp = HParams.from_dict({k: v for k, v in row.items() if k in HParams.__dataclass_fields__})
        train.train_arm(directory, hp, restart=row.get("restart", False))
    if "generate" in stages:
        generate.generate_arm(directory, row.get("num_samples"))
    if "evaluate" in stages:
        try:
            root = data_root()
        except SystemExit:
            root = None
        evaluate.evaluate_arm(directory, runs_root, root, skip_privacy=row.get("skip_privacy", False))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--index", type=int, default=None, help="row of the config's runs list (0-based)")
    parser.add_argument("--list", action="store_true", help="print the config's rows and exit")
    parser.add_argument("--dataset")
    parser.add_argument("--aggregation")
    parser.add_argument("--arm")
    parser.add_argument("--num-steps", type=int, default=None, help="override the step budget")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--runs-root", default=None)
    args = parser.parse_args(argv)

    if args.list:
        for i, row in enumerate(load_rows(Path(args.config))):
            print(f"{i:3d}  {row['dataset']:10s} {row['aggregation']:20s} {row['arm']:25s} steps={row.get('num_steps')}")
        return

    if args.index is not None:
        rows = load_rows(Path(args.config))
        if not 0 <= args.index < len(rows):
            raise SystemExit(f"--index {args.index} out of range; {args.config} has {len(rows)} rows")
        row = rows[args.index]
    elif args.dataset and args.aggregation and args.arm:
        row = {"dataset": args.dataset, "aggregation": normalize_aggregation(args.aggregation),
               "arm": normalize_arm(args.arm)}
    else:
        raise SystemExit("give --index N, or all of --dataset/--aggregation/--arm")
    if args.num_steps is not None:
        row["num_steps"] = args.num_steps

    run(row, args.stages, lambda: config.data_root(args.data_root), config.runs_root(args.runs_root))


if __name__ == "__main__":
    main()
