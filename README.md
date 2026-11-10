# ehr_timediff

Trains TimeDiff ([Tian et al., JAMIA 2024](https://doi.org/10.1093/jamia/ocae229), arXiv 2310.15290, PDF in this folder) on the **output of the EHR-Dataset-Processing pipeline** instead of on a raw database. Then it hands the synthetic patients back in the pipeline's own format.

The pipeline produces one cached dataset per *arm* (no filter, or one of 12 filters) per aggregation method. This project trains one diffusion model per arm, so you can ask the pipeline's usual question of synthetic data: **does a filter make the synthetic data better or worse?**

```
pipeline arm pickle ──export──▶ prepared.pt ──train──▶ model.pt ──generate──▶ <arm>_synthetic_dataset_ehr.pkl ──evaluate──▶ evaluation.json
(24×7 + [icu, mortality])        (N,16,24)                                     (same format as the input)          + runs/results.csv
```

## Layout

| path | what |
|---|---|
| `ehr_timediff/` | the bridge: `export`, `train`, `generate`, `evaluate`, `run` (all four in order), `fake` (test data) |
| `timediff/` | TimeDiff's model code, vendored with import fixes only. See `timediff/VENDORED.md` |
| `ehr_pipeline/` | unmodified copy of the pipeline's code, needed to replay filters for exact masks. See `ehr_pipeline/VENDORED.md` |
| `config/arms.yaml` | which arms to train, and step budgets |
| `config/paths.example.yaml` | copy to `config/paths.yaml`; points at the pipeline's `Data/` |
| `scripts/run_arm.sh` | local runner |
| `scripts/setup_nibi.sh`, `scripts/slurm/` | venv setup and SLURM array for nibi |
| `tests/` | `pytest`; runs entirely on generated data |
| `runs/` | outputs (gitignored) |

## Running it

### Locally, without MIMIC (checks the install)

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 5090 needs cu128+
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                       # ~15 s, includes a tiny end-to-end run
scripts/run_arm.sh --fake --num-steps 5000       # fake records through the real pipeline filters
```

### On nibi, against the real cached arms

```bash
bash scripts/setup_nibi.sh                       # on a login node, once
cp config/paths.example.yaml config/paths.yaml   # set data_root to .../EHR-Dataset-Processing/Data
# set --account in scripts/slurm/*.sbatch
.venv/bin/python -m ehr_timediff.run --list      # rows of config/arms.yaml
bash scripts/slurm/submit.sh                     # one GPU array task per row
```

Each task runs export → train → generate → evaluate. Training resumes from `model.pt`, so a task that hits its time limit can just be resubmitted (`sbatch --array=3 scripts/slurm/train_array.sbatch`).

Before the full sweep, run one arm with a smaller budget and read its manifests:

```bash
.venv/bin/python -m ehr_timediff.run --dataset mimic-iii --aggregation mean --arm raw --num-steps 20000
```

### What the cached arms must look like

For `<Data>/<dataset>/<aggregation>/` this project reads:

- `raw_dataset_ehr.pkl` or `<arm>_filtered_dataset_ehr.pkl`: built by `Experiments/apply_dataset_filter.py`
- `<arm>_admission_ids.npy`: built by `rerun/regen_admission_ids.py`. **Required**: the train/test split is keyed on admission id.
- `<Data>/<dataset>/processed_record_ehr.pkl`: optional for mean/median, **required** for the three deviation aggregations (see masks below).

## How the data is converted

**Channels.** Each patient becomes a `(16, 24)` tensor, laid out the way the paper does MIMIC-IV/eICU:

| channels | content | diffusion type |
|---|---|---|
| 0, 2, …, 12 | the 7 vitals, pipeline column order | Gaussian |
| 1, 3, …, 13 | "was this vital measured this hour" | multinomial (0/1) |
| 14 | `icu` (stay > 3 days), repeated over 24 h | multinomial (0/1) |
| 15 | `mortality`, repeated over 24 h | multinomial (0/1) |

Labels are channels, so every synthetic patient is generated **with** its labels. The model makes a label per hour; the patient's label is the majority vote, and `generate_manifest.json` reports how consistently the model held it constant (`label_hour_agreement_mean`).

**Missingness masks.** The cached pickles store missing hours as `0.0` (`RecordEHR.to_tensor`), so "measured" has to be recovered:

- `exact` (default when `processed_record_ehr.pkl` exists) replays the arm's filter over the pre-aggregation records, in the same pre/post-aggregation order the pipeline uses, and reads off the empty cells. It is validated against the cache and fails if the two disagree.
- `zero` treats `0` as missing. This is exact for mean/median, since every vital's lower bound is ≥ 1, and **refused** for standard / mean / maximum deviation. There, any hour with a single reading is also 0. On test data that conflation mislabels about a quarter of all cells.

**Split.** Test membership is a hash of the admission id (20%). The same admission is on the same side in every arm and aggregation. No generator ever trains on a patient another arm is tested on, and the raw arm's held-out admissions make a **common test set** for all arms.

**Scaling.** Per vital, over measured training cells, to [0, 1]. TimeDiff's `MixedDiffusion` takes it to [−1, 1] itself. Unmeasured cells get the training mean (the indicator channel tells the model they're fill). Two modes, set with `--scaling` or `scaling:` in `arms.yaml`:

- `minmax` (default, TimeDiff's own choice): the range is the observed min..max.
- `robust`: the range is a percentile pair (`--clip-percentiles`, default 0.5 / 99.5), and readings outside it are clipped. **Needed for any arm that still contains impossible readings.** See "Outliers" below.

**Output.** Vitals are unscaled and clipped to the training range. Unmeasured cells are set to 0, labels go back to `[icu, mortality]`, and the result is written with the same dtypes as `to_tensor`. The pipeline reads it directly:

```python
from Entities.ehr_dataset import DatasetEHR
ds = DatasetEHR(); ds.load("runs/mimic-iii/mean/raw/raw_synthetic_dataset_ehr.pkl")
```

## Evaluation (`evaluation.json`, one row per arm in `runs/results.csv`)

- **TSTR vs TRTR**, both tasks: a model trained on synthetic data vs one trained on the real training split, both scored on (a) the arm's held-out real patients and (b) the common raw-arm test set. Two models, matching existing code: the pipeline's random forest (300 trees, 168 flattened features) and the comparison notebook's logistic regression (7 across-hour features, imputed and standardised).
- **Discriminative score**: |0.5 − accuracy| of a GRU telling real from synthetic. 0 is best. The raw accuracy is stored too.
- **Per vital**: mean/std of measured values, missing rate, and the KS statistic, real vs synthetic.
- **Privacy**: NNAA privacy loss (TimeDiff's metric, near 0 is good) and a distance-to-closest-record ratio. A ratio well under 1 means synthetic patients sit closer to training patients than unseen real patients do, i.e. memorisation.

## Things that will bite you

- **Outliers: choose the scaling before the sweep.** Under `minmax`, one impossible reading stretches a vital's range, and real values crowd into a sliver of it. That doesn't just blur the output. On test data (charting errors planted at 3× the upper bound in ~1% of cells) the raw arm's model **never learned the vitals**: its noise-prediction error stayed at 1.0, i.e. the same as always predicting zero, through 30k steps, and its synthetic heart rates spread over the whole 0–950 range. Same data, same 30k steps:

  | raw arm, fake data | mortality TSTR / TRTR (RF) | discriminator accuracy | synthetic mean HR (real 85) |
  |---|---|---|---|
  | `minmax` | 0.44 / 0.80 | 1.00 | 500 |
  | `robust`, percentiles 1 / 98 | 0.77 / 0.80 | 0.58 | 86 |
  | `all vitals` arm, `minmax` | 0.77 / 0.79 | 0.56 | 86 |

  `export_manifest.json` warns when a vital's min/max range is >5× its 0.1–99.9 percentile span. Check those warnings on the real raw arm first. The tradeoff: `robust` clips the outliers, which is a mild outlier filter in its own right, so filter comparisons become "on top of percentile clipping". Whichever you choose, use it for **every** arm being compared. If you use `robust`, set the percentiles so the upper one sits below the outlier rate (`clipped_cells_train` in the manifest shows how much was clipped).
- **Arm sizes differ by 30×.** MIMIC-III `mean`: 46k (raw), 31k (long missing segment), 20k (long gap), **1.4k (high invalid data)**. A model on 1.4k patients will memorise; read its privacy numbers before its utility numbers.
- **Filters change who is in the cohort.** TSTR on an arm's own test set mixes generator quality with the population change. The common test set fixes the patients, but favours arms close to raw.
- **The pipeline itself treats 0 as missing.** `long gap` and `high invalid data` count `value == 0` as invalid even under deviation aggregations. On test data, `standard deviation` + `high invalid data` kept *zero* records. That's upstream behaviour, but it shapes which deviation arms exist.
- **`fill missing data`** looks almost fully measured (KNN-imputed), so its generator learns the imputer's curves and its indicator channels are nearly all 1.
- **Class imbalance.** Mortality is around 10–15%. Generation is unconditional, so it reproduces the prevalence but can't be asked for more positives; that needs a label-conditional model. Check `label_prevalence` in `generate_manifest.json`.
- **Cost.** About 60 training steps/s on an RTX 5090 (4.1M parameters, batch 32 × 2 accumulation), so the paper's 700k steps take about 3–4 h per arm. Sampling 37k patients is a few minutes.
- **Fixed hourly grid.** TimeDiff can model irregular sampling; the pipeline has already binned to 24 hours, so that isn't used. MIMIC hour 0 is hospital admission and eICU's is ICU admission, so don't compare across datasets.
- **Stale caches.** The pipeline never invalidates its pickles. `export` stores the source pickle's sha256 and rebuilds `prepared.pt` when it changes; `--force` rebuilds regardless (e.g. after changing `--mask-mode`).
- **Data use.** Synthetic data from MIMIC/eICU is still under the PhysioNet DUA unless you have grounds to say otherwise. `runs/` is gitignored for that reason.
