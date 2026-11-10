#!/usr/bin/env bash
# Run export -> train -> generate -> evaluate for one arm on this machine.
#
#   scripts/run_arm.sh --index 0                                  # row 0 of config/arms.yaml
#   scripts/run_arm.sh --dataset mimic-iii --aggregation mean --arm "long gap" --num-steps 50000
#   scripts/run_arm.sh --fake --num-steps 2000                    # no MIMIC: generated test data
#   ARM="all vitals" scripts/run_arm.sh --fake --num-steps 2000
#
# Everything after the script name goes to `python -m ehr_timediff.run` (see its --help).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

if [[ "${1:-}" == "--fake" ]]; then
    shift
    export EHR_DATA_ROOT="$PWD/runs/_fake_data" EHR_RUNS_ROOT="$PWD/runs/_fake_runs"
    [[ -d "$EHR_DATA_ROOT/fake" ]] || "$PY" -m ehr_timediff.fake --out "$EHR_DATA_ROOT" --n 3000
    exec "$PY" -m ehr_timediff.run --dataset fake --aggregation mean --arm "${ARM:-raw}" "$@"
fi
exec "$PY" -m ehr_timediff.run "$@"
