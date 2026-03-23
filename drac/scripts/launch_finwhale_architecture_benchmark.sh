#!/bin/bash
# Launch a focused architecture benchmark sweep for FinWhale CNN models.
#
# This is a thin wrapper around launch_finwhale_training_sweep.sh that:
# - benchmarks a small/medium/large CNN family plus ResNet18/34/50
# - keeps every run in the same W&B project
# - uses a conservative single-setting hyperparameter config by default
#
# Example:
#   bash drac/scripts/launch_finwhale_architecture_benchmark.sh \
#     --tar-path /project/rpp-kmoran/merileo/data/finwhales/all_mat_files.tar \
#     --wandb-project finwhale_cnn_benchmark \
#     --dataset-tag finwhale_trainstyle

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
SWEEP_SCRIPT="$REPO_ROOT/drac/scripts/launch_finwhale_training_sweep.sh"

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
  echo "Error: sweep launcher not found: $SWEEP_SCRIPT"
  exit 1
fi

DEFAULT_MODELS="SmallCNN,DeepCNN:w32:d4,DeepCNN:w64:d6,DeepCNN:w96:d8,resnet18,resnet34,resnet50"
DEFAULT_WANDB_PROJECT="finwhale_cnn_architecture_benchmark"
DEFAULT_WANDB_GROUP_PREFIX="finwhale-architecture-benchmark"

exec bash "$SWEEP_SCRIPT" \
  --models "$DEFAULT_MODELS" \
  --batch-size 64 \
  --epochs 100 \
  --num-workers 4 \
  --seeds 42 \
  --lrs 1e-3 \
  --balances weighted \
  --center-bias-list 0.25 \
  --min-gap-list 120 \
  --main-metric f1 \
  --device cuda \
  --split-strategy time_separated \
  --wandb-project "$DEFAULT_WANDB_PROJECT" \
  --wandb-group-prefix "$DEFAULT_WANDB_GROUP_PREFIX" \
  "$@"
