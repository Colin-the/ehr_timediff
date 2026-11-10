# Vendored EHR-Dataset-Processing

An unmodified copy of the code from
`clinical_preprocessing_toolbox/pipelines/EHR-Dataset-Processing/` (as of 2026-11-10):
`Entities/`, `Managers/`, `Processing/`, `Docker/`, `Experiments/apply_dataset_filter.py`,
`README.md`, `requirements.txt`. No `Data/`, `rerun/`, figures or notebooks.

Why it's here:

1. `masks.py` replays a filter on `processed_record_ehr.pkl` to recover exactly which cells
   were measured. That only works if the replay uses the same filter code that built the
   cached pickles. The replay is validated against the cache and fails loudly if they disagree.
2. Unpickling `processed_record_ehr.pkl` needs `Entities.ehr_record.RecordEHR` importable.
3. Someone with Postgres access can rebuild the arm pickles from this folder alone (see its
   README; that path needs the pipeline's cuDF environment, not this project's venv).

The bridge imports only `Entities/` and `Managers/ehr_filter_manager.py`, which are plain
pandas/numpy/sklearn. It never imports `Managers/dataset_manager.py` or
`evaluation_manager.py` (cupy/cuML).

If the pipeline's filters change, re-copy this folder; otherwise exact masks will (correctly)
refuse to build against the new caches. `tests/test_bridge.py` checks that the `VITALS`
range table in `ehr_timediff/arms.py` still matches the copy here.
