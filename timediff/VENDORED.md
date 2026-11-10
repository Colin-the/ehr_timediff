# Vendored TimeDiff

Source: `4900/CIS*4900/TimeDiff/` (fork of the code for Tian et al., *Reliable generation of
privacy-preserving synthetic EHR time series via diffusion models*, JAMIA 2024,
arXiv 2310.15290; paper PDF at the project root).

Only the model and one evaluation module are copied:

| file | from | changes |
|---|---|---|
| `models/ETDiff/blocks.py` | same path | `import models.ETDiff.utils` -> relative import; `torchcde` optional (only the unused NeuralCDE block needs it) |
| `models/ETDiff/utils.py` | same path | `torchcde` optional |
| `models/ETDiff/mixed_diffusion.py` | same path | relative import; `torch.cuda.amp.autocast` -> `torch.amp.autocast("cuda", ...)`; raw docstrings for `\epsilon` / `\theta` |
| `models/ETDiff/gaussian_diffusion.py` | same path | same as mixed_diffusion.py |
| `models/ETDiff/et_diff.py` | same path | relative import; matplotlib/seaborn imported lazily inside the (disabled, eICU-specific) `check_point_regular` plotter |
| `evaluate/privacy.py` | same path | none (NNAA, MIR) |

No model maths was changed.

Deliberately **not** copied:

- `etdiff_train.py`: replaced by `ehr_timediff/train.py`. The original hard-codes per-dataset
  channel layouts (`COLUMNS_DICT`) and sample paths under `/home/undergrad/...`, and its
  `TimeSeriesDataset` caches to `data/datasets/<name>/` relative to the working directory
  without ever checking whether the cache is stale.
- `evaluate/utility.py`: its `DiscriminativeScore` / `PredictiveScore` need TensorFlow 1.x,
  which doesn't install on Python 3.12+. `ehr_timediff/evaluate.py` re-implements the
  discriminative score in torch (same design: GRU, hidden = channels/2, |0.5 − accuracy|).
- `evaluate/utils.py` (TensorFlow), `evaluate/nn_classifier.py`, `eval_samples.py`, all
  `baselines/` and the other generators (`models/{ehr_m_gan,gt_gan,halo,...}`), every
  `preprocess*/`, `train_*`, `evaluate_*`, `optimize_*` script, logs, `.md` notes,
  checkpoints and `samples/`.

Results from the old folder (`TimeDiff/samples/*.npy`, `downstream_results/`) should not be
compared with this project's results: those samples had no labels (they were pseudo-labelled
by a classifier trained on real data), `preprocess_mimic3.py` normalised twice, and
`run_downstream_mimic3.py` read labels as `[mortality, icu]` when the pipeline stores
`[icu, mortality]`.
