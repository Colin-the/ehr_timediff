"""The TimeDiff tensor layout, and the conversions to and from pipeline format.

Pipeline format, per patient: a 24×7 float matrix (hour × vital) with 0 wherever
nothing was measured, plus labels `[icu, mortality]` in that order.

TimeDiff format: `(channels, 24)`, value channels each followed by their
observed-indicator, then the two labels held constant across all 24 hours. This
is the convention the paper uses for MIMIC-IV/eICU (value, indicator, value,
indicator, ..., label):

    ch  0  heart rate              numerical
    ch  1  heart rate observed     categorical (0/1)
    ch  2  systolic bp             numerical
    ...
    ch 12  oxygen saturation       numerical
    ch 13  oxygen sat. observed    categorical (0/1)
    ch 14  icu  (> 3 day stay)     categorical (0/1)
    ch 15  mortality               categorical (0/1)

Carrying the labels as channels means every synthetic patient comes out with its
own labels, rather than being labelled afterwards by a classifier trained on real
data (which is what the earlier TimeDiff runs did).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .arms import VITAL_NAMES

N_HOURS = 24
N_VITALS = len(VITAL_NAMES)
# Same order as RecordEHR.to_tensor and DatasetEHR.label_index_map. The pipeline
# indexes these positionally; swapping them trains against the wrong target with
# no error.
LABEL_NAMES = ("icu", "mortality")


@dataclass(frozen=True)
class Layout:
    n_vitals: int = N_VITALS
    label_names: tuple = LABEL_NAMES

    def value_channel(self, vital: int) -> int:
        return 2 * vital

    def mask_channel(self, vital: int) -> int:
        return 2 * vital + 1

    def label_channel(self, label: int) -> int:
        return 2 * self.n_vitals + label

    @property
    def n_channels(self) -> int:
        return 2 * self.n_vitals + len(self.label_names)

    @property
    def numerical(self) -> List[int]:
        return [self.value_channel(i) for i in range(self.n_vitals)]

    @property
    def mask_channels(self) -> List[int]:
        return [self.mask_channel(i) for i in range(self.n_vitals)]

    @property
    def label_channels(self) -> List[int]:
        return [self.label_channel(i) for i in range(len(self.label_names))]

    @property
    def categorical(self) -> List[int]:
        return self.mask_channels + self.label_channels

    @property
    def categorical_num_classes(self) -> List[int]:
        return [2] * len(self.categorical)

    def describe(self) -> dict:
        names = {}
        for i, vital in enumerate(VITAL_NAMES[: self.n_vitals]):
            names[self.value_channel(i)] = vital
            names[self.mask_channel(i)] = f"{vital} observed"
        for i, label in enumerate(self.label_names):
            names[self.label_channel(i)] = label
        return {
            "n_channels": self.n_channels,
            "numerical": self.numerical,
            "categorical": self.categorical,
            "categorical_num_classes": self.categorical_num_classes,
            "channel_names": {str(k): v for k, v in sorted(names.items())},
        }


LAYOUT = Layout()


@dataclass
class Scaler:
    """Per-vital scaling to [0, 1], fit on observed training cells only.

    TimeDiff's MixedDiffusion maps [0, 1] -> [-1, 1] itself (auto_normalize), so
    this must stop at [0, 1]. The old preprocess_mimic3.py scaled to [-1, 1] and
    then MixedDiffusion scaled again.

    minmax  (TimeDiff's choice) the range is the observed min..max.
    robust  the range is a pair of percentiles, and values outside it are clipped.
            Needed for arms with charting errors left in: under min/max a single
            impossible reading compresses every real value into a sliver of the
            scale, and on test data the model then never learned the vitals at all
            (noise-prediction error stayed at 1.0 for 30k steps, vs 0.1 by ~12k
            steps once the range was percentile-based).
    """

    minimum: np.ndarray = field(default_factory=lambda: np.zeros(N_VITALS))
    maximum: np.ndarray = field(default_factory=lambda: np.ones(N_VITALS))
    fill: np.ndarray = field(default_factory=lambda: np.zeros(N_VITALS))
    clip: bool = False

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray, percentiles=None) -> "Scaler":
        """values/mask: (N, 24, 7). Unobserved cells are ignored.

        percentiles=None is min/max scaling; (lo, hi) is robust scaling with clipping.
        """
        observed = np.where(mask, values, np.nan)
        if percentiles is None:
            minimum = np.nanmin(observed, axis=(0, 1))
            maximum = np.nanmax(observed, axis=(0, 1))
        else:
            with np.errstate(all="ignore"):
                minimum = np.array([np.nanpercentile(observed[:, :, i], percentiles[0])
                                    if mask[:, :, i].any() else np.nan for i in range(values.shape[2])])
                maximum = np.array([np.nanpercentile(observed[:, :, i], percentiles[1])
                                    if mask[:, :, i].any() else np.nan for i in range(values.shape[2])])
            observed = np.clip(observed, minimum, maximum)       # fill value from the clipped data
        fill = np.nanmean(observed, axis=(0, 1))
        # A vital never observed in training (possible in a tiny arm) would give NaN
        # stats; fall back to 0..1 so scaling is a no-op instead of NaN everywhere.
        never = np.isnan(minimum)
        minimum[never], maximum[never], fill[never] = 0.0, 1.0, 0.0
        return cls(minimum.astype(np.float64), maximum.astype(np.float64), fill.astype(np.float64),
                   clip=percentiles is not None)

    @property
    def span(self) -> np.ndarray:
        span = self.maximum - self.minimum
        return np.where(span > 0, span, 1.0)

    def scale(self, values: np.ndarray) -> np.ndarray:
        scaled = (values - self.minimum) / self.span
        return np.clip(scaled, 0.0, 1.0) if self.clip else scaled

    def unscale(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.span + self.minimum

    def to_dict(self) -> dict:
        return {"minimum": self.minimum.tolist(), "maximum": self.maximum.tolist(), "fill": self.fill.tolist(),
                "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(np.asarray(d["minimum"], float), np.asarray(d["maximum"], float), np.asarray(d["fill"], float),
                   bool(d.get("clip", False)))


def to_timediff(values: np.ndarray, mask: np.ndarray, labels: np.ndarray, scaler: Scaler,
                layout: Layout = LAYOUT) -> np.ndarray:
    """Pipeline arrays -> TimeDiff tensor.

    values: (N, 24, 7) in clinical units; anything where mask is False is ignored.
    mask:   (N, 24, 7) bool, True = measured.
    labels: (N, 2) int, [icu, mortality].
    returns (N, 16, 24) float32 with numericals in [0, 1] (for training rows).

    Unobserved cells get the training mean, as TimeDiff's replace_nan_with_mean
    does. The model sees the indicator channel, so the fill value only needs to be
    plausible, not meaningful.
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(labels)
    n = len(values)
    assert values.shape == (n, N_HOURS, layout.n_vitals), values.shape
    assert mask.shape == values.shape, mask.shape
    assert labels.shape == (n, len(layout.label_names)), labels.shape

    filled = np.where(mask, values, scaler.fill[None, None, :])
    scaled = scaler.scale(filled)

    out = np.zeros((n, layout.n_channels, N_HOURS), dtype=np.float32)
    for i in range(layout.n_vitals):
        out[:, layout.value_channel(i), :] = scaled[:, :, i]
        out[:, layout.mask_channel(i), :] = mask[:, :, i]
    for j in range(len(layout.label_names)):
        out[:, layout.label_channel(j), :] = labels[:, j, None]
    return out


def from_timediff(x: np.ndarray, scaler: Scaler, layout: Layout = LAYOUT, clip: bool = True):
    """TimeDiff tensor -> (values (N,24,7), mask (N,24,7) bool, labels (N,2) int64, label_agreement (N,2)).

    - Numericals are unscaled to clinical units. With clip=True they are clipped to
      the training [min, max] first, which is the range the model was trained on.
    - Unobserved cells are set to 0, which is exactly what RecordEHR.to_tensor
      produces for a missing hour, so the output is indistinguishable in format
      from a real pipeline record.
    - Labels are generated per hour but are constant in the real data, so the
      patient label is the majority vote across the 24 hours. `label_agreement` is
      the fraction of hours that agree with that vote (1.0 = the model kept the
      label constant, as the data does). It is a cheap sanity signal.
    """
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 3 and x.shape[1] == layout.n_channels and x.shape[2] == N_HOURS, x.shape

    scaled = np.stack([x[:, layout.value_channel(i), :] for i in range(layout.n_vitals)], axis=-1)
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    values = scaler.unscale(scaled)
    mask = np.stack([np.round(x[:, layout.mask_channel(i), :]) >= 1 for i in range(layout.n_vitals)], axis=-1)
    values = np.where(mask, values, 0.0)

    per_hour = np.stack([np.round(x[:, ch, :]) >= 1 for ch in layout.label_channels], axis=1)  # (N, 2, 24)
    share_positive = per_hour.mean(axis=2)
    labels = (share_positive >= 0.5).astype(np.int64)
    agreement = np.where(labels == 1, share_positive, 1.0 - share_positive)
    return values.astype(np.float32), mask, labels, agreement
