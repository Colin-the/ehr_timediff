#!/bin/bash
# Re-run evaluation only (no GPU needed), e.g. after changing evaluate.py:
#   sbatch --array=0-5 scripts/slurm/evaluate.sbatch
#SBATCH --job-name=timediff-eval
#SBATCH --account=def-CHANGE_ME
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
module load StdEnv/2023 python/3.12
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
python -m ehr_timediff.run --index "$SLURM_ARRAY_TASK_ID" --stages evaluate
