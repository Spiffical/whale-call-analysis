#!/bin/bash
# Submit a fin-whale RT-DETR bbox experiment.

#SBATCH --account=def-kmoran
#SBATCH --job-name=fin_rtdetr
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/drac/scripts/submit_finwhale_rtdetr.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  if [[ -L "$SCRIPT_PATH" ]]; then
    SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
  fi
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
  if [[ -d "$SCRIPT_DIR/../.." && -f "$SCRIPT_DIR/../../scripts/train/train_finwhale_rtdetr.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
  else
    REPO_ROOT="$HOME/whale-call-analysis"
  fi
fi

PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/scratch/$USER}/whale-call-analysis/finwhale_bbox_runs}"
CONFIG_PATH="${CONFIG_PATH:-config/dataset_config.yaml}"
AUDIO_DIR=""
RUN_TAG="joint_v1"
MODEL_NAME="PekingU/rtdetr_r50vd"
EPOCHS=20
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
NUM_WORKERS=4
GRAD_ACCUM=1
LEARNING_RATE="5e-5"
WEIGHT_DECAY="1e-4"
WARMUP_RATIO="0.1"
PURE_ZERO_RATIO="0.5"
NEGATIVE_MARGIN_S="2.0"
CENTER_BIAS_SIGMA_FRAC="0.25"
FREQ_MIN_HZ="1.0"
FREQ_MAX_HZ="200.0"
EDGE_BUFFER_S="2.0"
IMAGE_SIZE=640
SEED=42
MAX_TRAIN_IMAGES=0
MAX_EVAL_IMAGES=0
QC_LIMIT=24
INSTALL_DETECTION_DEPS="true"
SMOKE_MODE="false"

usage() {
  cat <<'USAGE'
Usage:
  sbatch drac/scripts/submit_finwhale_rtdetr.sh --audio-dir /path/to/raw_audio [options]

Required:
  --audio-dir PATH

Options:
  --run-tag TAG
  --output-root PATH
  --project-path PATH
  --venv-path PATH
  --config-path PATH
  --model-name NAME                    (default: PekingU/rtdetr_r50vd)
  --epochs N                           (default: 20)
  --train-batch-size N                 (default: 4)
  --eval-batch-size N                  (default: 4)
  --num-workers N                      (default: 4)
  --gradient-accumulation-steps N      (default: 1)
  --learning-rate VALUE                (default: 5e-5)
  --weight-decay VALUE                 (default: 1e-4)
  --warmup-ratio VALUE                 (default: 0.1)
  --pure-zero-ratio VALUE              (default: 0.5)
  --negative-margin-s VALUE            (default: 2.0)
  --center-bias-sigma-frac VALUE       (default: 0.25)
  --freq-min-hz VALUE                  (default: 1.0)
  --freq-max-hz VALUE                  (default: 200.0)
  --edge-buffer-s VALUE                (default: 2.0)
  --image-size N                       (default: 640)
  --seed N                             (default: 42)
  --max-train-images N                 (default: 0 = all)
  --max-eval-images N                  (default: 0 = all)
  --qc-limit N                         (default: 24)
  --install-detection-deps
  --skip-install-detection-deps
  --smoke-mode                         Use lighter defaults for a quick pilot
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --project-path) PROJECT_PATH="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --config-path) CONFIG_PATH="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --gradient-accumulation-steps) GRAD_ACCUM="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --warmup-ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --pure-zero-ratio) PURE_ZERO_RATIO="$2"; shift 2 ;;
    --negative-margin-s) NEGATIVE_MARGIN_S="$2"; shift 2 ;;
    --center-bias-sigma-frac) CENTER_BIAS_SIGMA_FRAC="$2"; shift 2 ;;
    --freq-min-hz) FREQ_MIN_HZ="$2"; shift 2 ;;
    --freq-max-hz) FREQ_MAX_HZ="$2"; shift 2 ;;
    --edge-buffer-s) EDGE_BUFFER_S="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-train-images) MAX_TRAIN_IMAGES="$2"; shift 2 ;;
    --max-eval-images) MAX_EVAL_IMAGES="$2"; shift 2 ;;
    --qc-limit) QC_LIMIT="$2"; shift 2 ;;
    --install-detection-deps) INSTALL_DETECTION_DEPS="true"; shift ;;
    --skip-install-detection-deps) INSTALL_DETECTION_DEPS="false"; shift ;;
    --smoke-mode) SMOKE_MODE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$AUDIO_DIR" || ! -d "$AUDIO_DIR" ]]; then
  echo "Error: --audio-dir is required and must exist"
  exit 1
fi
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Error: venv not found at $VENV_PATH/bin/activate"
  exit 1
fi

if [[ "$SMOKE_MODE" == "true" ]]; then
  MODEL_NAME="PekingU/rtdetr_r18vd"
  EPOCHS=1
  TRAIN_BATCH_SIZE=2
  EVAL_BATCH_SIZE=2
  MAX_TRAIN_IMAGES=128
  MAX_EVAL_IMAGES=64
  QC_LIMIT=12
fi

LOG_DIR="${SCRATCH:-/scratch/$USER}/whale-call-analysis/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
exec > >(tee -a "$LOG_DIR/fin_rtdetr_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/fin_rtdetr_${SLURM_JOB_ID:-$$}.err" >&2)

module load python/3.11.5
source "$VENV_PATH/bin/activate"

echo "Staging project into $SLURM_TMPDIR ..."
rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/whale_project/"
export PYTHONPATH="${PYTHONPATH:-}:$SLURM_TMPDIR/whale_project/src"
cd "$SLURM_TMPDIR/whale_project"

if [[ "$INSTALL_DETECTION_DEPS" == "true" ]]; then
  pip install -r "$SLURM_TMPDIR/whale_project/requirements-detection.txt"
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_SLUG="finwhale_rtdetr_${RUN_TAG}_${RUN_STAMP}"
TMP_ROOT="$SLURM_TMPDIR/finwhale_rtdetr_pipeline"
MANIFEST_DIR="$TMP_ROOT/manifests"
SPLIT_DIR="$TMP_ROOT/splits"
EXPORT_DIR="$SLURM_TMPDIR/finwhale_rtdetr_data"
TRAIN_DIR="$TMP_ROOT/train"
EVAL_DIR="$TMP_ROOT/eval_best"
FINAL_DIR="$OUTPUT_ROOT/$RUN_SLUG"

mkdir -p "$TMP_ROOT" "$EXPORT_DIR" "$TRAIN_DIR" "$EVAL_DIR" "$FINAL_DIR"

python -u scripts/data/detection/build_finwhale_bbox_manifests.py \
  --output-dir "$MANIFEST_DIR"

python -u scripts/data/detection/build_finwhale_bbox_splits.py \
  --annotation-manifest "$MANIFEST_DIR/unified_annotations.csv" \
  --clip-manifest "$MANIFEST_DIR/clip_manifest.csv" \
  --output-dir "$SPLIT_DIR"

python -u scripts/data/detection/export_finwhale_bbox_dataset.py \
  --annotation-manifest "$MANIFEST_DIR/unified_annotations.csv" \
  --clip-manifest "$MANIFEST_DIR/clip_manifest.csv" \
  --split-assignments "$SPLIT_DIR/assignments.csv" \
  --audio-dir "$AUDIO_DIR" \
  --output-dir "$EXPORT_DIR" \
  --config-path "$CONFIG_PATH" \
  --pure-zero-ratio "$PURE_ZERO_RATIO" \
  --negative-margin-s "$NEGATIVE_MARGIN_S" \
  --center-bias-sigma-frac "$CENTER_BIAS_SIGMA_FRAC" \
  --freq-min-hz "$FREQ_MIN_HZ" \
  --freq-max-hz "$FREQ_MAX_HZ" \
  --edge-buffer-s "$EDGE_BUFFER_S" \
  --image-size "$IMAGE_SIZE" \
  --seed "$SEED" \
  --qc-limit "$QC_LIMIT"

python -u scripts/train/train_finwhale_rtdetr.py \
  --dataset-dir "$EXPORT_DIR" \
  --output-dir "$TRAIN_DIR" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --warmup-ratio "$WARMUP_RATIO" \
  --max-train-images "$MAX_TRAIN_IMAGES" \
  --max-eval-images "$MAX_EVAL_IMAGES" \
  --seed "$SEED" \
  --device cuda

python -u scripts/train/eval_finwhale_rtdetr.py \
  --dataset-dir "$EXPORT_DIR" \
  --checkpoint-dir "$TRAIN_DIR/best" \
  --output-dir "$EVAL_DIR" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-images "$MAX_EVAL_IMAGES" \
  --device cuda

mkdir -p "$FINAL_DIR/manifests" "$FINAL_DIR/splits" "$FINAL_DIR/export_metadata" "$FINAL_DIR/train" "$FINAL_DIR/eval_best"
rsync -a "$MANIFEST_DIR/" "$FINAL_DIR/manifests/"
rsync -a "$SPLIT_DIR/" "$FINAL_DIR/splits/"
rsync -a \
  --include='*/' \
  --include='summary.json' \
  --include='context_manifest.csv' \
  --include='crop_manifest.csv' \
  --include='*.coco.json' \
  --include='qc/***' \
  --exclude='*' \
  "$EXPORT_DIR/" "$FINAL_DIR/export_metadata/"
rsync -a "$TRAIN_DIR/" "$FINAL_DIR/train/"
rsync -a "$EVAL_DIR/" "$FINAL_DIR/eval_best/"

cat > "$FINAL_DIR/run_info.json" <<EOF
{
  "run_slug": "$RUN_SLUG",
  "audio_dir": "$AUDIO_DIR",
  "project_path": "$PROJECT_PATH",
  "model_name": "$MODEL_NAME",
  "epochs": $EPOCHS,
  "train_batch_size": $TRAIN_BATCH_SIZE,
  "eval_batch_size": $EVAL_BATCH_SIZE,
  "smoke_mode": $([[ "$SMOKE_MODE" == "true" ]] && echo "true" || echo "false")
}
EOF

echo "RT-DETR run complete."
echo "Durable outputs copied to: $FINAL_DIR"
