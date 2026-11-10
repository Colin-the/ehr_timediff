#!/usr/bin/env bash
# Build the project venv on a nibi LOGIN node (compute nodes have no internet).
# Run once:  bash scripts/setup_nibi.sh
set -euo pipefail
cd "$(dirname "$0")/.."
module load StdEnv/2023 python/3.12
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# torch from the Alliance wheelhouse (CUDA bundled, H100-ready); everything else from it
# where available, falling back to PyPI (ema-pytorch usually is not in the wheelhouse).
pip install --no-index torch || pip install torch
pip install -r requirements.txt
python -c "import torch, accelerate, ema_pytorch, einops, sklearn; print('ok, torch', torch.__version__)"
echo "Now: cp config/paths.example.yaml config/paths.yaml and set data_root."
