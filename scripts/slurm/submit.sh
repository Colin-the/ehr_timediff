#!/usr/bin/env bash
# Submit one array task per row of config/arms.yaml.
#   bash scripts/slurm/submit.sh                 # train_array.sbatch
#   bash scripts/slurm/submit.sh evaluate.sbatch # evaluation only
set -euo pipefail
cd "$(dirname "$0")/../.."
module load StdEnv/2023 python/3.12 2>/dev/null || true
source .venv/bin/activate
python -m ehr_timediff.run --list
N=$(python -m ehr_timediff.run --list | wc -l)
mkdir -p logs
sbatch --array="0-$((N - 1))" "scripts/slurm/${1:-train_array.sbatch}"
