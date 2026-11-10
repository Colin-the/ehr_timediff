"""Bridge tests. No MIMIC needed: everything runs on generated data.

    .venv/bin/python -m pytest tests -q
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ehr_timediff.arms import PIPELINE_ROOT, VITAL_NAMES, VITALS, ensure_pipeline_importable  # noqa: E402
from ehr_timediff.io import load_arm, save_pipeline_dataset, in_test_split  # noqa: E402
from ehr_timediff.layout import LABEL_NAMES, LAYOUT, Scaler, from_timediff, to_timediff  # noqa: E402
from ehr_timediff.masks import validate_mask, zero_mask  # noqa: E402


def _random_arm(n=64, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.uniform(30, 150, size=(n, 24, 7)).astype(np.float32)
    mask = rng.random((n, 24, 7)) > 0.3
    values[~mask] = 0.0
    labels = rng.integers(0, 2, size=(n, 2))
    return values, mask, labels


def test_vitals_table_matches_vendored_pipeline():
    """arms.VITALS is the seventh copy of the range table (bug register R-30); catch drift."""
    source = (PIPELINE_ROOT / "Experiments" / "apply_dataset_filter.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "VITALS" for t in node.targets):
            pipeline_vitals = ast.literal_eval(node.value)
            break
    else:
        pytest.fail("VITALS not found in vendored apply_dataset_filter.py")
    assert {k: (tuple(v[0]), v[1]) for k, v in pipeline_vitals.items()} == \
           {k: (tuple(v[0]), v[1]) for k, v in VITALS.items()}
    assert tuple(pipeline_vitals.keys()) == VITAL_NAMES


def test_label_order_matches_pipeline():
    ensure_pipeline_importable()
    from Entities.ehr_dataset import DatasetEHR
    assert DatasetEHR().label_index_map == {name: i for i, name in enumerate(LABEL_NAMES)}


def test_layout_indices():
    assert LAYOUT.n_channels == 16
    assert LAYOUT.numerical == [0, 2, 4, 6, 8, 10, 12]
    assert LAYOUT.categorical == [1, 3, 5, 7, 9, 11, 13, 14, 15]
    assert sorted(LAYOUT.numerical + LAYOUT.categorical) == list(range(16))


def test_round_trip_is_exact():
    values, mask, labels = _random_arm()
    scaler = Scaler.fit(values, mask)
    x = to_timediff(values, mask, labels, scaler)
    assert x.shape == (64, 16, 24)
    numericals = x[:, LAYOUT.numerical, :]
    assert numericals.min() >= 0 and numericals.max() <= 1 + 1e-6
    back_values, back_mask, back_labels, agreement = from_timediff(x, scaler)
    assert np.array_equal(back_mask, mask)
    assert np.array_equal(back_labels, labels)
    assert np.allclose(back_values, values, atol=1e-3)
    assert np.all(back_values[~back_mask] == 0)          # same convention as RecordEHR.to_tensor
    assert np.all(agreement == 1.0)


def test_robust_scaling_clips_outliers():
    values, mask, labels = _random_arm(n=200)
    values[0, 0, 0], mask[0, 0, 0] = 5000.0, True            # one impossible reading
    minmax = Scaler.fit(values, mask)
    robust = Scaler.fit(values, mask, percentiles=(1, 99))
    assert minmax.maximum[0] == 5000.0 and robust.maximum[0] < 200
    x = to_timediff(values, mask, labels, robust)
    assert x[:, LAYOUT.numerical, :].max() <= 1.0
    back, *_ = from_timediff(x, Scaler.from_dict(robust.to_dict()))
    assert back[0, 0, 0] == pytest.approx(robust.maximum[0], rel=1e-5)   # clipped, not 5000


def test_labels_are_majority_vote_over_hours():
    values, mask, labels = _random_arm(n=2)
    scaler = Scaler.fit(values, mask)
    x = to_timediff(values, mask, np.array([[0, 0], [1, 1]]), scaler)
    x[0, LAYOUT.label_channel(1), :5] = 1                 # 5 of 24 hours flip: still 0
    x[1, LAYOUT.label_channel(1), :20] = 0                # 20 of 24 hours flip: now 0
    _, _, out, agreement = from_timediff(x, scaler)
    assert out.tolist() == [[0, 0], [1, 0]]
    assert agreement[0, 1] == pytest.approx(19 / 24)


def test_split_is_keyed_on_admission_id_only():
    ids = np.arange(100000, 160000, 3)
    full = in_test_split(ids)
    subset = ids[::7]
    assert np.array_equal(in_test_split(subset), full[::7])   # same patient, same side, any arm
    assert abs(full.mean() - 0.2) < 0.01
    assert not np.array_equal(in_test_split(ids, salt=1), full)


def test_zero_mask_refuses_deviation_aggregations():
    values, _, _ = _random_arm()
    zero_mask(values, "mean")
    with pytest.raises(ValueError):
        zero_mask(values, "standard deviation")
    zero_mask(values, "standard deviation", force=True)


def test_validate_mask_catches_disagreement():
    values, mask, _ = _random_arm()
    validate_mask(values, mask)
    broken = mask.copy()
    broken[0, 0, 0] = False
    values[0, 0, 0] = 80.0
    with pytest.raises(ValueError):
        validate_mask(values, broken)


def test_pipeline_format_round_trip(tmp_path):
    values, _, labels = _random_arm()
    path = tmp_path / "x_synthetic_dataset_ehr.pkl"
    save_pipeline_dataset(values, labels, path)
    ensure_pipeline_importable()
    from Entities.ehr_dataset import DatasetEHR
    ds = DatasetEHR()
    ds.load(str(path))                                   # the pipeline's own loader, default torch.load
    x, y = ds[3]
    assert x.dtype == torch.float32 and tuple(x.shape) == (24, 7)
    assert y.dtype == torch.int64 and y.tolist() == labels[3].tolist()
    v2, l2 = load_arm(path)
    assert np.array_equal(v2, values) and np.array_equal(l2, labels)


@pytest.mark.slow
def test_end_to_end_on_fake_data(tmp_path):
    """fake pipeline data -> export -> train (tiny) -> generate -> evaluate."""
    from ehr_timediff import evaluate, export, fake, generate, train
    from ehr_timediff.arms import run_dir
    from ehr_timediff.model import HParams

    data, runs = tmp_path / "data", tmp_path / "runs"
    fake.build(data, n=300, aggregations=("mean",), arms=("raw", "long gap"))
    for arm in ("raw", "long gap"):
        export.prepare_arm(data, runs, fake.DATASET, "mean", arm)
        directory = run_dir(runs, fake.DATASET, "mean", arm)
        train.train_arm(directory, HParams(num_steps=30, timesteps=20, hidden=32, save_every=1000))
        generate.generate_arm(directory, num_samples=120, batch_size=60)
        result = evaluate.evaluate_arm(directory, runs, data, skip_privacy=(arm == "raw"))
        assert result["n_synthetic"] == 120
        assert "mortality/rf/own_test" in result["utility"]
    assert "mortality/rf/common_test" in result["utility"]  # long gap was scored on raw's test set
    assert "privacy" in result
    manifest = json.load(open(run_dir(runs, fake.DATASET, "mean", "long gap") / "export_manifest.json"))
    assert manifest["mask_mode"] == "exact"
    assert (runs / "results.csv").exists()
