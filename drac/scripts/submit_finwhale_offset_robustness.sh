#!/bin/bash
# Submit deterministic offset-robustness evaluation on DRAC.
#
# Example:
# sbatch $HOME/whale-call-analysis/drac/scripts/submit_finwhale_offset_robustness.sh \
#   --tar-path /path/to/all_mat_files.tar \
#   --checkpoint /path/to/best.pt \
#   --split-file /path/to/splits/test.txt \
#   --out-dir /scratch/$USER/finwhale_offset_eval/run1

#SBATCH --account=def-kmoran
#SBATCH --job-name=finwhale_offset_eval
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Detect repo root
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/drac/scripts/submit_finwhale_offset_robustness.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
fi
PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"

TAR_PATH=""
POS_DIR=""
NEG_DIR=""
CHECKPOINT=""
OUT_DIR=""
SPLIT_FILE=""
MODEL=""
CROP_SIZE=""
CROP_TIME_SECONDS=""
CROP_FREQ_MIN=""
CROP_FREQ_MAX=""
THRESHOLD="0.5"
THRESHOLD_HIGH="0.7"
TARGET_RECALL="0.95"
TARGET_PRECISION="0.95"
THRESHOLD_STEP="0.01"
OFFSET_FRACS=""
MAX_SAMPLES="0"
MAX_NEG_SAMPLES="0"
SEED="42"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tar-path) TAR_PATH="$2"; shift 2 ;;
    --pos-dir) POS_DIR="$2"; shift 2 ;;
    --neg-dir) NEG_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --split-file) SPLIT_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --crop-time-seconds) CROP_TIME_SECONDS="$2"; shift 2 ;;
    --crop-freq-range-hz) CROP_FREQ_MIN="$2"; CROP_FREQ_MAX="$3"; shift 3 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --threshold-high) THRESHOLD_HIGH="$2"; shift 2 ;;
    --target-recall) TARGET_RECALL="$2"; shift 2 ;;
    --target-precision) TARGET_PRECISION="$2"; shift 2 ;;
    --threshold-step) THRESHOLD_STEP="$2"; shift 2 ;;
    --offset-fracs) OFFSET_FRACS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --max-neg-samples) MAX_NEG_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$OUT_DIR" ]]; then
  echo "Error: --checkpoint and --out-dir are required"
  exit 1
fi
if [[ -z "$TAR_PATH" && ( -z "$POS_DIR" || -z "$NEG_DIR" ) ]]; then
  echo "Error: provide either --tar-path or both --pos-dir and --neg-dir"
  exit 1
fi

LOG_DIR="$SCRATCH/whale-call-analysis/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/finwhale_offset_eval_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/finwhale_offset_eval_${SLURM_JOB_ID:-$$}.err" >&2)

module load python/3.10
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Error: venv not found at $VENV_PATH/bin/activate"
  exit 2
fi
source "$VENV_PATH/bin/activate"

echo "Copying project to node-local storage..."
rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/whale_project/"

if [[ -n "$TAR_PATH" ]]; then
  echo "Extracting data archive..."
  mkdir -p "$SLURM_TMPDIR/finwhale_data"
  if [[ "$TAR_PATH" == *.tar.gz || "$TAR_PATH" == *.tgz ]]; then
    tar -xzf "$TAR_PATH" -C "$SLURM_TMPDIR/finwhale_data"
  elif [[ "$TAR_PATH" == *.tar ]]; then
    tar -xf "$TAR_PATH" -C "$SLURM_TMPDIR/finwhale_data"
  elif [[ "$TAR_PATH" == *.zip ]]; then
    unzip -q "$TAR_PATH" -d "$SLURM_TMPDIR/finwhale_data"
  else
    echo "Unsupported archive format: $TAR_PATH"
    exit 1
  fi

  if [[ -d "$SLURM_TMPDIR/finwhale_data/mat_files" && -d "$SLURM_TMPDIR/finwhale_data/neg_mat_files" ]]; then
    POS_ARG="$SLURM_TMPDIR/finwhale_data/mat_files"
    NEG_ARG="$SLURM_TMPDIR/finwhale_data/neg_mat_files"
  else
    ROOT_SUBDIR=$(find "$SLURM_TMPDIR/finwhale_data" -maxdepth 2 -type d -name mat_files -print -quit)
    if [[ -n "$ROOT_SUBDIR" ]]; then
      POS_ARG="$ROOT_SUBDIR"
      NEG_ARG="$(dirname "$ROOT_SUBDIR")/neg_mat_files"
      [[ -d "$NEG_ARG" ]] || { echo "Missing neg_mat_files next to $ROOT_SUBDIR"; exit 1; }
    else
      echo "Could not locate mat_files/neg_mat_files in archive"
      exit 1
    fi
  fi
else
  POS_ARG="$POS_DIR"
  NEG_ARG="$NEG_DIR"
fi

mkdir -p "$OUT_DIR"

export PYTHONPATH="$PYTHONPATH:$SLURM_TMPDIR/whale_project/src"
cd "$SLURM_TMPDIR/whale_project"

CMD=(
  python -u scripts/diagnostics/evaluate_offset_robustness.py
  --pos-dir "$POS_ARG"
  --neg-dir "$NEG_ARG"
  --checkpoint "$CHECKPOINT"
  --out-dir "$OUT_DIR"
  --threshold "$THRESHOLD"
  --threshold-high "$THRESHOLD_HIGH"
  --target-recall "$TARGET_RECALL"
  --target-precision "$TARGET_PRECISION"
  --threshold-step "$THRESHOLD_STEP"
  --max-samples "$MAX_SAMPLES"
  --max-neg-samples "$MAX_NEG_SAMPLES"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ -n "$SPLIT_FILE" ]]; then
  CMD+=( --split-file "$SPLIT_FILE" )
fi
if [[ -n "$MODEL" ]]; then
  CMD+=( --model "$MODEL" )
fi
if [[ -n "$CROP_SIZE" ]]; then
  CMD+=( --crop-size "$CROP_SIZE" )
fi
if [[ -n "$CROP_TIME_SECONDS" ]]; then
  CMD+=( --crop-time-seconds "$CROP_TIME_SECONDS" )
fi
if [[ -n "$CROP_FREQ_MIN" && -n "$CROP_FREQ_MAX" ]]; then
  CMD+=( --crop-freq-range-hz "$CROP_FREQ_MIN" "$CROP_FREQ_MAX" )
fi
if [[ -n "$OFFSET_FRACS" ]]; then
  CMD+=( --offset-fracs "$OFFSET_FRACS" )
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

