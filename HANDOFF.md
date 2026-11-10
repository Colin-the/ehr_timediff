# Handoff: generating synthetic data on nibi

Written 2026-11-10 on the local workstation (RTX 5090) before the folder moved to nibi.
This file covers status, decisions and next steps. `README.md` covers how the code works.

## Goal

Train one TimeDiff model per pipeline **arm** on the EHR-Dataset-Processing output, generate
synthetic patients in the pipeline's own format, and compare arms. The question is whether a
filter makes the synthetic data better or worse.

**Scope the user cares about:** `mean` and `median` aggregations, with the arms `raw`, the
7 per-vital outlier filters and `all vitals`. That is 9 arms × 2 aggregations = 18 models.
The structural filters (`long missing segment`, `long gap`, `high invalid data`, `fill missing
data`) and the deviation aggregations are **out of scope**.

## Status

- All code is done and tested locally on generated data. 11 pytest tests pass, including an
  end-to-end export → train → generate → evaluate run.
- Nothing has been run on real MIMIC yet. The local machine only has the `*_admission_ids.npy`
  files, not the dataset pickles.
- `config/arms.yaml` **still lists the old first sweep** (mean: raw, all vitals, fill missing
  data, long missing segment, long gap, high invalid data). **Replace it with the 18 in-scope
  rows before submitting** (see step 4).

## Transfer

Don't copy `.venv/` (built for the 5090; rebuild it on nibi) or `runs/` (only fake test output,
~350 MB). For example:

```bash
rsync -av --exclude .venv --exclude runs --exclude logs --exclude __pycache__ \
    /ssd/colin/ehr_timediff/ nibi:<dest>/ehr_timediff/
```

## Steps on nibi

1. **Environment** (on a login node; compute nodes have no internet):
   `bash scripts/setup_nibi.sh`. This loads `StdEnv/2023 python/3.12`, builds `.venv`, and
   installs torch from the wheelhouse plus `requirements.txt`. `ema-pytorch` probably comes from
   PyPI. Then run `.venv/bin/python -m pytest` (~15 s, needs no MIMIC) to check the install.
2. **Paths:** `cp config/paths.example.yaml config/paths.yaml`, then set
   - `data_root`: the pipeline's `Data/` directory, which holds `mimic-iii/`;
   - `runs_root`: somewhere under `$SCRATCH` or project space, not `$HOME` (~65 MB per model
     plus samples).

   The env vars `EHR_DATA_ROOT` and `EHR_RUNS_ROOT` override this file.
3. **SLURM account:** replace `def-CHANGE_ME` in `scripts/slurm/train_array.sbatch` and
   `scripts/slurm/evaluate.sbatch`. The defaults are 1× H100, 8 CPUs, 48 GB and 10 h.
4. **Choose the scaling, then rewrite `config/arms.yaml`** (see "Decisions still open").
   Arm names: `raw`, `heart rate`, `systolic blood pressure`, `diastolic blood pressure`,
   `mean blood pressure`, `respiration rate`, `temperature`, `oxygen saturation`, `all vitals`.
5. **Pilot one arm before the sweep:**
   ```bash
   .venv/bin/python -m ehr_timediff.run --dataset mimic-iii --aggregation mean --arm raw --num-steps 20000
   ```
   Run this inside `salloc --gpus-per-node=h100:1 --account=...` or as a one-off sbatch.
   Then read `runs/mimic-iii/mean/raw/`:
   - `export_manifest.json`:
     - `warnings`: are there min/max range warnings on the raw arm?
     - `mask_mode`: should be `exact` if `processed_record_ehr.pkl` exists, otherwise `zero`. Both are valid for mean/median.
     - `observed_zero_cells_train`, `clipped_cells_train`, label prevalence and missing rates.
   - `train_config.json` / training log: eps-mse must fall well below 1.0. If it sits at about
     1.0, the model isn't learning the vitals (the outlier/scaling problem below).
   - `generate_manifest.json`: label prevalence close to the real one, and
     `label_hour_agreement_mean` near 1.
   - `evaluation.json`: TSTR close to TRTR, discriminator accuracy well below 1.0, sensible per-vital means.
6. **Sweep:** run `.venv/bin/python -m ehr_timediff.run --list`, then
   `bash scripts/slurm/submit.sh` (one array task per row). A task that times out resumes from
   `model.pt`: `sbatch --array=<i> scripts/slurm/train_array.sbatch`.
7. **Results:** each arm gets a row in `<runs_root>/results.csv`. The synthetic data is at
   `<runs_root>/mimic-iii/<agg>/<arm>/<arm>_synthetic_dataset_ehr.pkl` and loads with the
   pipeline's `DatasetEHR.load`.

## Decisions still open

- **Scaling: `minmax` vs `robust`. This is the most important decision.** It has to be the same
  for every arm compared. On fake data with ~1% impossible readings, the raw arm under `minmax`
  never learned the vitals:

  | setting | mortality RF TSTR / TRTR | discriminator accuracy | mean heart rate (real 85) |
  |---|---|---|---|
  | `minmax` | 0.44 / 0.80 | 1.00 | 500 |
  | `robust` 1/98 | 0.77 / 0.80 | 0.58 | 86 |

  `raw` and the per-vital arms keep impossible readings in the vitals they don't filter, so
  they are at risk. If the pilot shows range warnings or eps-mse stuck near 1.0, use
  `scaling: robust` with `clip_percentiles` set so the upper percentile sits below the outlier
  rate. Check `clipped_cells_train`. The fake data needed `[1, 98]`; real MIMIC outliers are
  probably far rarer, so the default `[0.5, 99.5]` may be enough. Clipping is a mild outlier
  filter in its own right, so record which setting was used.
- **Step budget:** the default is the paper's 700k. That took about 3–4 h per arm on the 5090,
  and should be similar or faster on an H100. 18 arms fit comfortably in the 10 h limit.
- **Sample count:** `num_samples` defaults to the arm's training size.

## Things already settled (don't re-litigate)

- **Labels are channels.** The tensor has 16 channels: 7 vitals, 7 "measured" flags, then
  `icu` and `mortality`. The label order `[icu, mortality]` matches the pipeline. The old
  TimeDiff work may have swapped them.
- **Zeros for missing/filtered cells are fine for mean/median.** Every vital's lower bound is
  ≥ 1, so 0 always means "no reading". The generator sees the "measured" flags, not the 0s.
  An outlier filter only empties an hour when all of its readings were out of range.
  - The one subtlety: a charted 0 (e.g. heart rate 0) as the only reading in an hour survives
    in raw and in the unfiltered vitals of the per-vital arms. The `exact` mask keeps it as a
    reading; the `zero` mask treats it as missing. The effect is small; see
    `observed_zero_cells_train`.
- **Split** is a hash of admission id, with 20% held out. Each patient lands on the same side
  in every arm, and the raw arm's test patients form a common test set.
- **Evaluation:**
  - TSTR vs TRTR with an RF (300 trees, 168 features) and LR (7 across-hour features);
  - a GRU discriminator (score and accuracy);
  - per-vital KS statistics;
  - NNAA and distance-to-closest-record privacy metrics.
- The old `TimeDiff/samples/*` results are not comparable: no labels, a probable label swap,
  double normalization. Don't use them as a baseline.

## Known limitations to report

- Filters change who is in the cohort, so TSTR on an arm's own test set mixes generator
  quality with the population change. The common test set controls for this.
- Generation is unconditional, so it reproduces mortality prevalence (~10–15%) rather than
  oversampling positives.
- The pipeline has already binned the data to a fixed 24 h × 7 grid, so TimeDiff's
  irregular-sampling ability goes unused.
- Synthetic data derived from MIMIC remains under the PhysioNet DUA. Keep `runs/` private.

## Where to look

| need | file |
|---|---|
| full design, channel layout, outliers table | `README.md` |
| arm list and per-row options | `config/arms.yaml`, `ehr_timediff/run.py` |
| export, masks, scaling | `ehr_timediff/export.py`, `masks.py`, `layout.py` |
| hyperparameters (paper defaults) | `ehr_timediff/model.py` (`HParams`) |
| metrics | `ehr_timediff/evaluate.py` |
| vendored code and patches | `timediff/VENDORED.md`, `ehr_pipeline/VENDORED.md` |
| paper | `2310.15290v6.pdf` |
