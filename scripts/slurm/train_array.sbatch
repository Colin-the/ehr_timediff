#!/bin/bash
# One array task per row of config/arms.yaml: export -> train -> generate -> evaluate.
# Submit with scripts/slurm/submit.sh (it sizes --array from arms.yaml), or by hand:
#   sbatch --array=0-5 scripts/slurm/train_array.sbatch
#
# Training resumes from model.pt, so if a task hits the time limit just resubmit
# that index: sbatch --array=3 scripts/slurm/train_array.sbatch
#SBATCH --job-name=timediff
#SBATCH --account=def-CHANGE_ME
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
module load StdEnv/2023 python/3.12
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m ehr_timediff.run --index "$SLURM_ARRAY_TASK_ID"
