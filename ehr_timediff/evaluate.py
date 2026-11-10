"""Score one arm's synthetic data against real held-out patients.

    python -m ehr_timediff.evaluate --dataset mimic-iii --aggregation mean --arm raw

Utility (both tasks, icu and mortality):
  TRTR  train on the arm's real training split, test on real held-out patients
  TSTR  train on the synthetic patients, test on the same real held-out patients
  Two test sets: the arm's own held-out split, and the *common* test set (the
  held-out admissions of the raw arm, same aggregation). The arm's own split
  answers "is the synthetic data as useful as the real data it came from"; the
  common one puts every arm on the same patients, but favours arms whose
  population is close to raw.
  Classifiers match the existing code: the pipeline's random forest (300 trees)
  on the 168 flattened features, and the comparison notebook's standardised
  logistic regression on 7 across-hour features.

Fidelity:
  discriminative score  |0.5 - accuracy| of a GRU trained to tell real from
                        synthetic (0 = indistinguishable). Re-implemented in
                        torch: the TimeDiff original needs TensorFlow 1.x.
  per-vital             mean/std of measured values, missing rate, and the KS
                        statistic between real and synthetic measured values.

Privacy:
  NNAA                  TimeDiff's nearest-neighbour adversarial accuracy
                        (vendored timediff/evaluate/privacy.py). Privacy loss
                        near 0 is good; large means the model is closer to its
                        training patients than to unseen ones.
  DCR                   median distance from each synthetic patient to the
                        closest training patient, vs. the same for held-out real
                        patients. A ratio well under 1 suggests memorisation.

Writes evaluation.json in the run directory and upserts a row in
<runs_root>/results.csv so arms can be compared side by side.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .arms import VITAL_NAMES, arm_ids_path, arm_pickle_path, run_dir
from .export import PREPARED_NAME, add_arm_arguments, load_prepared
from .generate import synthetic_pickle_name
from .io import load_arm, load_ids, in_test_split
from .layout import LABEL_NAMES, LAYOUT, Scaler, to_timediff
from .train import resolve_run_dir

from timediff.evaluate.privacy import NNAA  # noqa: E402  (path set up via .train -> .model)


# --------------------------------------------------------------------------- utility

def flatten(values: np.ndarray) -> np.ndarray:
    """24×7 -> 168 zero-filled features: what the pipeline's random forest sees."""
    return values.reshape(len(values), -1)


def across_hours(values: np.ndarray, mask: np.ndarray, aggregation: str) -> np.ndarray:
    """24×7 -> 7: each vital collapsed over its measured hours, with the same method
    as the arm's aggregation. Mirrors aggregate_hourly_matrix in the comparison
    notebook (pandas skipna semantics; std with ddof=1). NaN where a vital was never
    measured; the LR pipeline mean-imputes those."""
    x = np.where(mask, values, np.nan).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns -> NaN, expected
        if aggregation == "mean":
            return np.nanmean(x, axis=1)
        if aggregation == "median":
            return np.nanmedian(x, axis=1)
        if aggregation == "standard deviation":
            return np.nanstd(x, axis=1, ddof=1)
        centred = np.abs(x - np.nanmean(x, axis=1, keepdims=True))
        if aggregation == "mean deviation":
            return np.nanmean(centred, axis=1)
        if aggregation == "maximum deviation":
            return np.nanmax(centred, axis=1)
    raise ValueError(f"unknown aggregation {aggregation!r}")


def classifiers(seed: int, aggregation: str) -> dict:
    """name -> (model factory, featuriser over (values, mask)).

    rf  the pipeline's model: random forest, 300 trees, on the 168 flattened features
        (cuML there, sklearn here).
    lr  the comparison notebook's model: standardised logistic regression on 7
        across-hour features. Linear models do badly on the 168 zero-filled
        features, where every missing hour looks like a huge drop in the vital.
    """
    return {
        "rf": (lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
               lambda v, m: flatten(v)),
        "lr": (lambda: make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler(),
                                     LogisticRegression(max_iter=500, solver="liblinear", random_state=0)),
               lambda v, m: across_hours(v, m, aggregation)),
    }


def score(clf_factory, x_train, y_train, x_test, y_test) -> dict:
    if len(np.unique(y_train)) < 2:
        return {"auroc": None, "auprc": None, "note": "training labels contain one class only"}
    if len(np.unique(y_test)) < 2:
        return {"auroc": None, "auprc": None, "note": "test labels contain one class only"}
    clf = clf_factory()
    clf.fit(x_train, y_train)
    p = clf.predict_proba(x_test)[:, 1]
    return {"auroc": float(roc_auc_score(y_test, p)), "auprc": float(average_precision_score(y_test, p))}


def utility(real_train, synth, test_sets: dict, aggregation: str, seed: int) -> dict:
    """Each argument / test set is a (values, mask, labels) triple in pipeline format."""
    results = {}
    for clf_name, (factory, featurise) in classifiers(seed, aggregation).items():
        x_real, x_synth = featurise(real_train[0], real_train[1]), featurise(synth[0], synth[1])
        x_tests = {name: featurise(v, m) for name, (v, m, _) in test_sets.items()}
        for task_index, task in enumerate(LABEL_NAMES):
            for set_name, (_, _, yl) in test_sets.items():
                key = f"{task}/{clf_name}/{set_name}"
                y_test = yl[:, task_index]
                trtr = score(factory, x_real, real_train[2][:, task_index], x_tests[set_name], y_test)
                tstr = score(factory, x_synth, synth[2][:, task_index], x_tests[set_name], y_test)
                results[key] = {"trtr": trtr, "tstr": tstr}
                print(f"[evaluate] {key:32s} TRTR auroc {_fmt(trtr['auroc'])}  TSTR auroc {_fmt(tstr['auroc'])}")
    return results


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


# --------------------------------------------------------------------------- fidelity

class _Discriminator(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (B, T, C)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def discriminative_score(real: np.ndarray, synth: np.ndarray, seed: int, max_n: int = 10000,
                         epochs: int = 20, device: Optional[str] = None) -> dict:
    """|0.5 - held-out accuracy| of a post-hoc GRU classifier (TimeGAN / TimeDiff metric).

    Inputs are TimeDiff-layout tensors (N, C, 24): scaled values plus observed
    indicators, so the discriminator sees missingness patterns as well as values.
    Hidden size is channels/2, as in the original.
    """
    rng = np.random.default_rng(seed)
    n = min(len(real), len(synth), max_n)
    real = real[rng.permutation(len(real))[:n]]
    synth = synth[rng.permutation(len(synth))[:n]]
    x = np.concatenate([real, synth]).transpose(0, 2, 1).astype(np.float32)   # (2n, T, C)
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)
    order = rng.permutation(2 * n)
    x, y = x[order], y[order]
    cut = int(0.8 * len(x))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    model = _Discriminator(x.shape[2], max(x.shape[2] // 2, 1)).to(device)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x[:cut]).to(device), torch.from_numpy(y[:cut]).to(device)
    for _ in range(epochs):
        for idx in torch.randperm(len(xt), device=device).split(128):
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x[cut:]).to(device)).cpu().numpy()
    accuracy = float(((logits > 0) == (y[cut:] > 0.5)).mean())
    # Accuracy is kept alongside the score: |0.5 - acc| reads the same for 1.0
    # (synthetic trivially detectable) and 0.0 (a broken discriminator).
    return {"score": abs(0.5 - accuracy), "accuracy": accuracy, "n_per_class": int(n)}


def per_vital(real_values, real_mask, synth_values, synth_mask, seed: int, max_points: int = 200000) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i, vital in enumerate(VITAL_NAMES):
        r = real_values[:, :, i][real_mask[:, :, i]]
        s = synth_values[:, :, i][synth_mask[:, :, i]]
        entry = {
            "missing_rate_real": float(1 - real_mask[:, :, i].mean()),
            "missing_rate_synth": float(1 - synth_mask[:, :, i].mean()),
        }
        if len(r) and len(s):
            r_sub = r if len(r) <= max_points else rng.choice(r, max_points, replace=False)
            s_sub = s if len(s) <= max_points else rng.choice(s, max_points, replace=False)
            entry.update({
                "mean_real": float(r.mean()), "mean_synth": float(s.mean()),
                "std_real": float(r.std()), "std_synth": float(s.std()),
                "ks": float(ks_2samp(r_sub, s_sub).statistic),
            })
        out[vital] = entry
    return out


# --------------------------------------------------------------------------- privacy

def _privacy_features(values, mask, labels, scaler) -> np.ndarray:
    return to_timediff(values, mask, labels, scaler).reshape(len(values), -1).astype(np.float64)


def _nearest_distance(a: np.ndarray, b: np.ndarray, batch: int = 2048) -> np.ndarray:
    """For each row of a, Euclidean distance to its nearest row of b."""
    ta, tb = torch.from_numpy(a).float(), torch.from_numpy(b).float()
    if torch.cuda.is_available():
        ta, tb = ta.cuda(), tb.cuda()
    out = [torch.cdist(ta[i:i + batch], tb).min(dim=1).values.cpu() for i in range(0, len(ta), batch)]
    return torch.cat(out).numpy()


def privacy(train_f, test_f, synth_f, seed: int, max_test: int = 3000) -> dict:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    test_sub = test_f[rng.permutation(len(test_f))[:max_test]]
    loss, aa_test, aa_train = NNAA(synth_f, train_f, test_sub)()
    dcr_synth = float(np.median(_nearest_distance(synth_f[:max_test * 2], train_f)))
    dcr_real = float(np.median(_nearest_distance(test_sub, train_f)))
    return {
        "nnaa_privacy_loss": float(loss), "nnaa_aa_test": float(aa_test), "nnaa_aa_train": float(aa_train),
        "dcr_median_synth_to_train": dcr_synth, "dcr_median_test_to_train": dcr_real,
        "dcr_ratio": dcr_synth / dcr_real if dcr_real > 0 else None,
    }


# --------------------------------------------------------------------------- driver

def common_test_set(runs_root: Path, data_root: Optional[Path], dataset: str, aggregation: str,
                    test_fraction: float, salt: int):
    """Held-out admissions of the raw arm (same aggregation): the shared yardstick."""
    raw_prepared = run_dir(runs_root, dataset, aggregation, "raw") / PREPARED_NAME
    if raw_prepared.exists():
        test = load_prepared(raw_prepared)["test"]
        return test["values"], test["mask"], test["labels"]
    if data_root is None:
        return None
    pickle_path = arm_pickle_path(data_root, dataset, aggregation, "raw")
    if not pickle_path.exists():
        return None
    values, labels = load_arm(pickle_path)
    ids = load_ids(arm_ids_path(data_root, dataset, aggregation, "raw"), len(values))
    keep = in_test_split(ids, test_fraction, salt)
    # No exact mask without exporting the raw arm; value != 0 is only used by the
    # 7-feature LR, and is exact for mean/median.
    return values[keep], values[keep] != 0, labels[keep]


def evaluate_arm(directory: Path, runs_root: Path, data_root: Optional[Path], seed: int = 0,
                 skip_privacy: bool = False) -> dict:
    t0 = time.time()
    prepared = load_prepared(directory / PREPARED_NAME)
    meta, scaler = prepared["meta"], Scaler.from_dict(prepared["scaler"])
    train, test = prepared["train"], prepared["test"]
    synth_values, synth_labels = load_arm(directory / synthetic_pickle_name(meta["arm"]))
    synth_mask = np.load(directory / "synthetic_mask.npy")

    test_sets = {"own_test": (test["values"], test["mask"], test["labels"])}
    common = common_test_set(runs_root, data_root, meta["dataset"], meta["aggregation"],
                             meta["test_fraction"], meta["split_salt"])
    if common is not None:
        test_sets["common_test"] = common
    else:
        print("[evaluate] no raw arm available; skipping the common test set")

    result = {
        **{k: meta[k] for k in ("dataset", "aggregation", "arm", "mask_mode")},
        "n_train": int(len(train["labels"])), "n_test": int(len(test["labels"])),
        "n_synthetic": int(len(synth_labels)),
        "n_common_test": int(len(common[2])) if common is not None else 0,
        "label_prevalence": {
            "real_train": dict(zip(LABEL_NAMES, train["labels"].mean(0).tolist())),
            "synthetic": dict(zip(LABEL_NAMES, synth_labels.mean(0).tolist())),
        },
    }
    result["utility"] = utility((train["values"], train["mask"], train["labels"]),
                                (synth_values, synth_mask, synth_labels), test_sets, meta["aggregation"], seed)

    real_td = to_timediff(train["values"], train["mask"], train["labels"], scaler)
    synth_td = to_timediff(synth_values, synth_mask, synth_labels, scaler)
    result["discriminative"] = discriminative_score(real_td, synth_td, seed)
    print(f"[evaluate] discriminative score {result['discriminative']['score']:.3f} "
          f"(held-out accuracy {result['discriminative']['accuracy']:.3f}; score 0 = indistinguishable)")
    result["per_vital"] = per_vital(train["values"], train["mask"], synth_values, synth_mask, seed)

    if not skip_privacy and len(test["labels"]) > 0:
        result["privacy"] = privacy(
            _privacy_features(train["values"], train["mask"], train["labels"], scaler),
            _privacy_features(test["values"], test["mask"], test["labels"], scaler),
            synth_td.reshape(len(synth_td), -1).astype(np.float64), seed)
        print(f"[evaluate] NNAA privacy loss {result['privacy']['nnaa_privacy_loss']:.3f}, "
              f"DCR ratio {_fmt(result['privacy']['dcr_ratio'])}")

    result["seconds"] = round(time.time() - t0, 1)
    with open(directory / "evaluation.json", "w") as f:
        json.dump(result, f, indent=2)
    upsert_results_row(runs_root / "results.csv", flatten_result(result))
    print(f"[evaluate] done in {result['seconds']:.0f}s -> {directory / 'evaluation.json'}")
    return result


def flatten_result(result: dict) -> dict:
    row = {k: result[k] for k in ("dataset", "aggregation", "arm", "mask_mode", "n_train", "n_test",
                                   "n_synthetic", "n_common_test")}
    for task in LABEL_NAMES:
        row[f"prev_{task}_real"] = round(result["label_prevalence"]["real_train"][task], 4)
        row[f"prev_{task}_synth"] = round(result["label_prevalence"]["synthetic"][task], 4)
    for key, value in result["utility"].items():
        slug = key.replace("/", "_")
        row[f"{slug}_trtr_auroc"] = value["trtr"]["auroc"]
        row[f"{slug}_tstr_auroc"] = value["tstr"]["auroc"]
    row["discriminative_score"] = result["discriminative"]["score"]
    row["discriminator_accuracy"] = result["discriminative"]["accuracy"]
    row["mean_ks"] = float(np.mean([v["ks"] for v in result["per_vital"].values() if "ks" in v] or [np.nan]))
    if "privacy" in result:
        row["nnaa_privacy_loss"] = result["privacy"]["nnaa_privacy_loss"]
        row["dcr_ratio"] = result["privacy"]["dcr_ratio"]
    return row


def upsert_results_row(path: Path, row: dict) -> None:
    key = ("dataset", "aggregation", "arm")
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        same = np.logical_and.reduce([table[k] == row[k] for k in key])
        table = table[~same]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.sort_values(list(key)).to_csv(path, index=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arm_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-privacy", action="store_true")
    args = parser.parse_args(argv)
    try:
        data_root = config.data_root(args.data_root)
    except SystemExit:
        data_root = None
    evaluate_arm(resolve_run_dir(args), config.runs_root(args.runs_root), data_root, args.seed, args.skip_privacy)


if __name__ == "__main__":
    main()
