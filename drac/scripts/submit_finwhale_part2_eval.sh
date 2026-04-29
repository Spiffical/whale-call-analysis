#!/bin/bash
# Submit Part 2 inference/evaluation on Nibi/DRAC.
#
# Expected input archive is the VM-prepared bundle from:
#   scripts/data/part2/prepare_part2_vm_dataset.py

#SBATCH --account=def-kmoran
#SBATCH --job-name=finwhale_part2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/drac/scripts/submit_finwhale_part2_eval.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  if [[ -L "$SCRIPT_PATH" ]]; then
    SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
  fi
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
  if [[ -d "$SCRIPT_DIR/../.." && -f "$SCRIPT_DIR/../../scripts/inference/evaluate_part2_predictions.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
  else
    REPO_ROOT="$HOME/whale-call-analysis"
  fi
fi
PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"

PART2_ARCHIVE=""
CHECKPOINT=""
OUT_DIR=""
WINDOW_STEPS="24,48"
LOW_THRESHOLDS="0.70,0.75,0.80"
HIGH_THRESHOLDS="0.82,0.85,0.90"
MIN_MEMBERS_VALUES="2,3"
MAX_GAP_VALUES="auto,10,15"
MATCH_COLLAR_S="1.0"
CLASS_HIERARCHY=""
DEVICE="cuda"
BATCH_SIZE="128"
NUM_WORKERS="4"
CROP_SIZE=""
MERGE_EVENT_MEDIA="false"
EXPORT_EXAMPLE_IMAGES="true"
MAX_EXAMPLES_PER_GROUP="8"
METADATA_RELPATH="metadata.json"
MAT_DIR_RELPATH="mat_files"
RAW_AUDIO_RELPATH="raw_audio"
ANNOTATIONS_RELPATH="manifests/fin_annotations.csv"
CLIP_MANIFEST_RELPATH="manifests/clip_manifest.csv"
BASELINE_TAR=""
BASELINE_POS_DIR=""
BASELINE_NEG_DIR=""
BASELINE_SPLITS_DIR=""
BASELINE_EVAL_SPLIT="test"
USE_WANDB="false"
WANDB_PROJECT="whale-call-analysis"
WANDB_ENTITY=""
WANDB_GROUP=""
WANDB_NAME_PREFIX=""
WANDB_TAGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --part2-archive) PART2_ARCHIVE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --window-steps) WINDOW_STEPS="$2"; shift 2 ;;
    --low-thresholds) LOW_THRESHOLDS="$2"; shift 2 ;;
    --high-thresholds) HIGH_THRESHOLDS="$2"; shift 2 ;;
    --min-members-values) MIN_MEMBERS_VALUES="$2"; shift 2 ;;
    --max-gap-values) MAX_GAP_VALUES="$2"; shift 2 ;;
    --match-collar-s) MATCH_COLLAR_S="$2"; shift 2 ;;
    --class-hierarchy) CLASS_HIERARCHY="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --merge-event-media) MERGE_EVENT_MEDIA="true"; shift ;;
    --skip-example-images) EXPORT_EXAMPLE_IMAGES="false"; shift ;;
    --max-examples-per-group) MAX_EXAMPLES_PER_GROUP="$2"; shift 2 ;;
    --metadata-relpath) METADATA_RELPATH="$2"; shift 2 ;;
    --mat-dir-relpath) MAT_DIR_RELPATH="$2"; shift 2 ;;
    --raw-audio-relpath) RAW_AUDIO_RELPATH="$2"; shift 2 ;;
    --annotations-relpath) ANNOTATIONS_RELPATH="$2"; shift 2 ;;
    --clip-manifest-relpath) CLIP_MANIFEST_RELPATH="$2"; shift 2 ;;
    --baseline-tar) BASELINE_TAR="$2"; shift 2 ;;
    --baseline-pos-dir) BASELINE_POS_DIR="$2"; shift 2 ;;
    --baseline-neg-dir) BASELINE_NEG_DIR="$2"; shift 2 ;;
    --baseline-splits-dir) BASELINE_SPLITS_DIR="$2"; shift 2 ;;
    --baseline-eval-split) BASELINE_EVAL_SPLIT="$2"; shift 2 ;;
    --use-wandb) USE_WANDB="true"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-name-prefix) WANDB_NAME_PREFIX="$2"; shift 2 ;;
    --wandb-tags) WANDB_TAGS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$PART2_ARCHIVE" || -z "$CHECKPOINT" || -z "$OUT_DIR" ]]; then
  echo "Error: --part2-archive, --checkpoint, and --out-dir are required"
  exit 1
fi

LOG_DIR="$SCRATCH/whale-call-analysis/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/finwhale_part2_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/finwhale_part2_${SLURM_JOB_ID:-$$}.err" >&2)

module load python/3.10
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Error: venv not found at $VENV_PATH/bin/activate"
  exit 2
fi
source "$VENV_PATH/bin/activate"

if [[ "$USE_WANDB" == "true" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -f "$HOME/.wandb_api_key" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
    echo "Loaded WANDB_API_KEY from ~/.wandb_api_key"
  elif [[ -f "$PROJECT_PATH/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_PATH/.env" | xargs)
  else
    echo "Warning: WandB requested but WANDB_API_KEY is not configured."
  fi
fi

rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/whale_project/"
export PYTHONPATH="${PYTHONPATH:-}:$SLURM_TMPDIR/whale_project/src"
cd "$SLURM_TMPDIR/whale_project"

mkdir -p "$SLURM_TMPDIR/part2_bundle"
if [[ "$PART2_ARCHIVE" == *.tar.gz || "$PART2_ARCHIVE" == *.tgz ]]; then
  tar -xzf "$PART2_ARCHIVE" -C "$SLURM_TMPDIR/part2_bundle"
elif [[ "$PART2_ARCHIVE" == *.tar.zst ]]; then
  tar --use-compress-program=unzstd -xf "$PART2_ARCHIVE" -C "$SLURM_TMPDIR/part2_bundle"
elif [[ "$PART2_ARCHIVE" == *.tar ]]; then
  tar -xf "$PART2_ARCHIVE" -C "$SLURM_TMPDIR/part2_bundle"
elif [[ "$PART2_ARCHIVE" == *.zip ]]; then
  unzip -q "$PART2_ARCHIVE" -d "$SLURM_TMPDIR/part2_bundle"
else
  echo "Unsupported archive format: $PART2_ARCHIVE"
  exit 1
fi

BUNDLE_ROOT="$SLURM_TMPDIR/part2_bundle"
if [[ ! -f "$BUNDLE_ROOT/$METADATA_RELPATH" ]]; then
  FOUND_META="$(find "$BUNDLE_ROOT" -maxdepth 2 -type f -name "$(basename "$METADATA_RELPATH")" | head -n 1 || true)"
  if [[ -n "$FOUND_META" ]]; then
    BUNDLE_ROOT="$(dirname "$FOUND_META")"
  fi
fi

MAT_DIR="$BUNDLE_ROOT/$MAT_DIR_RELPATH"
METADATA_PATH="$BUNDLE_ROOT/$METADATA_RELPATH"
RAW_AUDIO_DIR="$BUNDLE_ROOT/$RAW_AUDIO_RELPATH"
ANNOTATIONS_CSV="$BUNDLE_ROOT/$ANNOTATIONS_RELPATH"
CLIP_MANIFEST_CSV="$BUNDLE_ROOT/$CLIP_MANIFEST_RELPATH"
ALL_ANNOTATIONS_CSV="$BUNDLE_ROOT/manifests/annotations_all.csv"

if [[ ! -d "$MAT_DIR" || ! -f "$METADATA_PATH" || ! -f "$ANNOTATIONS_CSV" || ! -f "$CLIP_MANIFEST_CSV" ]]; then
  echo "Resolved bundle root: $BUNDLE_ROOT"
  echo "Missing one or more required bundle artifacts."
  echo "  MAT_DIR=$MAT_DIR"
  echo "  METADATA_PATH=$METADATA_PATH"
  echo "  ANNOTATIONS_CSV=$ANNOTATIONS_CSV"
  echo "  CLIP_MANIFEST_CSV=$CLIP_MANIFEST_CSV"
  exit 1
fi

RUN_OUT_DIR="$OUT_DIR/finwhale_part2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_OUT_DIR"
BASELINE_METRICS_JSON=""

if [[ -n "$BASELINE_TAR" || ( -n "$BASELINE_POS_DIR" && -n "$BASELINE_NEG_DIR" ) ]]; then
  if [[ -n "$BASELINE_TAR" ]]; then
    mkdir -p "$SLURM_TMPDIR/baseline_data"
    if [[ "$BASELINE_TAR" == *.tar.gz || "$BASELINE_TAR" == *.tgz ]]; then
      tar -xzf "$BASELINE_TAR" -C "$SLURM_TMPDIR/baseline_data"
    elif [[ "$BASELINE_TAR" == *.tar.zst ]]; then
      tar --use-compress-program=unzstd -xf "$BASELINE_TAR" -C "$SLURM_TMPDIR/baseline_data"
    elif [[ "$BASELINE_TAR" == *.tar ]]; then
      tar -xf "$BASELINE_TAR" -C "$SLURM_TMPDIR/baseline_data"
    elif [[ "$BASELINE_TAR" == *.zip ]]; then
      unzip -q "$BASELINE_TAR" -d "$SLURM_TMPDIR/baseline_data"
    else
      echo "Unsupported baseline archive format: $BASELINE_TAR"
      exit 1
    fi
    if [[ -d "$SLURM_TMPDIR/baseline_data/mat_files" && -d "$SLURM_TMPDIR/baseline_data/neg_mat_files" ]]; then
      BASELINE_POS_DIR="$SLURM_TMPDIR/baseline_data/mat_files"
      BASELINE_NEG_DIR="$SLURM_TMPDIR/baseline_data/neg_mat_files"
    else
      ROOT_SUBDIR="$(find "$SLURM_TMPDIR/baseline_data" -maxdepth 2 -type d -name mat_files | head -n 1 || true)"
      if [[ -n "$ROOT_SUBDIR" && -d "$(dirname "$ROOT_SUBDIR")/neg_mat_files" ]]; then
        BASELINE_POS_DIR="$ROOT_SUBDIR"
        BASELINE_NEG_DIR="$(dirname "$ROOT_SUBDIR")/neg_mat_files"
      else
        echo "Could not locate mat_files/neg_mat_files in baseline archive"
        exit 1
      fi
    fi
  fi
  BASELINE_OUT="$RUN_OUT_DIR/historical_baseline"
  mkdir -p "$BASELINE_OUT"
  if [[ -z "$BASELINE_SPLITS_DIR" && -d "$(dirname "$CHECKPOINT")/splits" ]]; then
    BASELINE_SPLITS_DIR="$(dirname "$CHECKPOINT")/splits"
  fi
  BASELINE_CMD=(
    python -u scripts/train/test_cnn.py
    --checkpoint "$CHECKPOINT"
    --pos-dir "$BASELINE_POS_DIR"
    --neg-dir "$BASELINE_NEG_DIR"
    --out-dir "$BASELINE_OUT"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device "$DEVICE"
  )
  if [[ -n "$BASELINE_SPLITS_DIR" ]]; then
    BASELINE_CMD+=( --splits-dir "$BASELINE_SPLITS_DIR" --eval-split "$BASELINE_EVAL_SPLIT" )
  fi
  if [[ -n "$CROP_SIZE" ]]; then
    BASELINE_CMD+=( --crop-size "$CROP_SIZE" )
  fi
  if [[ "$USE_WANDB" == "true" ]]; then
    BASELINE_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      BASELINE_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
    if [[ -n "$WANDB_GROUP" ]]; then
      BASELINE_CMD+=( --wandb-group "$WANDB_GROUP" )
    fi
    if [[ -n "$WANDB_NAME_PREFIX" ]]; then
      BASELINE_CMD+=( --wandb-name "${WANDB_NAME_PREFIX}_baseline" )
    fi
  fi
  echo "Running historical baseline..."
  "${BASELINE_CMD[@]}"
  BASELINE_METRICS_JSON="$(find "$BASELINE_OUT" -maxdepth 2 -type f -name metrics.json | head -n 1 || true)"
fi

IFS=',' read -r -a WINDOW_STEP_VALUES <<< "$WINDOW_STEPS"
for STEP in "${WINDOW_STEP_VALUES[@]}"; do
  STEP_TRIM="$(echo "$STEP" | xargs)"
  [[ -n "$STEP_TRIM" ]] || continue
  STEP_DIR="$RUN_OUT_DIR/window_step_${STEP_TRIM}"
  mkdir -p "$STEP_DIR"

  PRED_JSON="$STEP_DIR/predictions_window.json"
  INFER_CMD=(
    python -u scripts/inference/run_inference.py
    --mat-dir "$MAT_DIR"
    --checkpoint "$CHECKPOINT"
    --dataset-metadata "$METADATA_PATH"
    --output-json "$PRED_JSON"
    --sliding-window
    --window-step "$STEP_TRIM"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device "$DEVICE"
    --raw-audio-dir "$RAW_AUDIO_DIR"
  )
  if [[ -n "$CROP_SIZE" ]]; then
    INFER_CMD+=( --crop-size "$CROP_SIZE" )
  fi

  echo "Running inference for window_step=$STEP_TRIM"
  "${INFER_CMD[@]}"

  EVAL_CMD=(
    python -u scripts/inference/evaluate_part2_predictions.py
    --window-predictions-json "$PRED_JSON"
    --annotations-csv "$ANNOTATIONS_CSV"
    --clip-manifest-csv "$CLIP_MANIFEST_CSV"
    --output-dir "$STEP_DIR/evaluation"
    --match-collar-s "$MATCH_COLLAR_S"
    --window-step-label "$STEP_TRIM"
    --low-thresholds "$LOW_THRESHOLDS"
    --high-thresholds "$HIGH_THRESHOLDS"
    --min-members-values "$MIN_MEMBERS_VALUES"
    --max-gap-values "$MAX_GAP_VALUES"
  )
  if [[ -f "$ALL_ANNOTATIONS_CSV" ]]; then
    EVAL_CMD+=( --all-annotations-csv "$ALL_ANNOTATIONS_CSV" )
  fi
  if [[ "$EXPORT_EXAMPLE_IMAGES" == "true" ]]; then
    EVAL_CMD+=(
      --example-mat-dir "$MAT_DIR"
      --export-example-images
      --max-examples-per-group "$MAX_EXAMPLES_PER_GROUP"
    )
  fi
  if [[ -n "$CLASS_HIERARCHY" ]]; then
    EVAL_CMD+=( --class-hierarchy "$CLASS_HIERARCHY" )
  fi
  if [[ -n "$BASELINE_METRICS_JSON" ]]; then
    EVAL_CMD+=( --baseline-metrics-json "$BASELINE_METRICS_JSON" )
  fi
  if [[ "$MERGE_EVENT_MEDIA" == "true" ]]; then
    EVAL_CMD+=( --merge-event-media )
  fi
  if [[ "$USE_WANDB" == "true" ]]; then
    EVAL_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      EVAL_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
    if [[ -n "$WANDB_GROUP" ]]; then
      EVAL_CMD+=( --wandb-group "$WANDB_GROUP" )
    fi
    if [[ -n "$WANDB_NAME_PREFIX" ]]; then
      EVAL_CMD+=( --wandb-name "${WANDB_NAME_PREFIX}_part2_ws${STEP_TRIM}" )
    fi
    if [[ -n "$WANDB_TAGS" ]]; then
      EVAL_CMD+=( --wandb-tags "$WANDB_TAGS,part2,window_step_${STEP_TRIM}" )
    else
      EVAL_CMD+=( --wandb-tags "part2,window_step_${STEP_TRIM}" )
    fi
  fi

  echo "Running Part 2 evaluation for window_step=$STEP_TRIM"
  "${EVAL_CMD[@]}"
done

echo "Part 2 evaluation complete."
echo "Outputs: $RUN_OUT_DIR"
