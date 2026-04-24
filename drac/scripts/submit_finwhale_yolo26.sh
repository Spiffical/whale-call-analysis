#!/bin/bash
# Submit a fin-whale YOLO26 bbox experiment.

#SBATCH --account=def-kmoran
#SBATCH --job-name=fin_yolo26
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/drac/scripts/submit_finwhale_yolo26.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  if [[ -L "$SCRIPT_PATH" ]]; then
    SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
  fi
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
  if [[ -d "$SCRIPT_DIR/../.." && -f "$SCRIPT_DIR/../../scripts/train/train_finwhale_yolo26.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
  else
    REPO_ROOT="$HOME/whale-call-analysis"
  fi
fi

PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-${SCRATCH:-/scratch/$USER}/whale-call-analysis/.venvs/finwhale_yolo26}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/scratch/$USER}/whale-call-analysis/finwhale_bbox_runs}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
CONFIG_PATH="${CONFIG_PATH:-config/dataset_config.yaml}"
AUDIO_DIR=""
AUDIO_BUNDLE_TAR=""
ALLOWED_FILENAMES_TXT=""
AVAILABLE_AUDIO_FILENAMES_TXT=""
RUN_TAG="joint_v1"
MODEL_NAME="yolo26m.pt"
EPOCHS=30
BATCH_SIZE=8
NUM_WORKERS=4
PATIENCE=20
PURE_ZERO_RATIO="0.5"
NEGATIVE_MARGIN_S="2.0"
CENTER_BIAS_SIGMA_FRAC="0.25"
FREQ_MIN_HZ="1.0"
FREQ_MAX_HZ="200.0"
EDGE_BUFFER_S="2.0"
IMAGE_SIZE=640
SEED=42
QC_LIMIT=24
INSTALL_DETECTION_DEPS="true"
SMOKE_MODE="false"
USE_WANDB="false"
WANDB_PROJECT="finwhale-bbox"
WANDB_ENTITY=""
WANDB_GROUP="finwhale-yolo26"
WANDB_NAME=""
WANDB_TAGS="bbox,yolo26,finwhale"

usage() {
  cat <<'USAGE'
Usage:
  sbatch drac/scripts/submit_finwhale_yolo26.sh (--audio-dir /path/to/raw_audio | --audio-bundle-tar /path/to/bundle.tar) [options]

Required:
  One of:
    --audio-dir PATH
    --audio-bundle-tar PATH        Archive containing raw_audio/<clip> files (.tar, .tar.gz, .tar.zst)

Options:
  --run-tag TAG
  --output-root PATH
  --log-dir PATH
  --project-path PATH
  --venv-path PATH
  --config-path PATH
  --allowed-filenames-txt PATH
  --available-audio-filenames-txt PATH
  --model-name NAME                    (default: yolo26m.pt)
  --epochs N                           (default: 30)
  --batch-size N                       (default: 8)
  --workers N                          (default: 4)
  --patience N                         (default: 20)
  --pure-zero-ratio VALUE              (default: 0.5)
  --negative-margin-s VALUE            (default: 2.0)
  --center-bias-sigma-frac VALUE       (default: 0.25)
  --freq-min-hz VALUE                  (default: 1.0)
  --freq-max-hz VALUE                  (default: 200.0)
  --edge-buffer-s VALUE                (default: 2.0)
  --image-size N                       (default: 640)
  --seed N                             (default: 42)
  --qc-limit N                         (default: 24)
  --use-wandb
  --wandb-project NAME                 (default: finwhale-bbox)
  --wandb-entity NAME
  --wandb-group NAME                   (default: finwhale-yolo26)
  --wandb-name NAME
  --wandb-tags CSV                     (default: bbox,yolo26,finwhale)
  --install-detection-deps
  --skip-install-detection-deps
  --smoke-mode                         Use lighter defaults for a quick pilot
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --audio-bundle-tar) AUDIO_BUNDLE_TAR="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --project-path) PROJECT_PATH="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --config-path) CONFIG_PATH="$2"; shift 2 ;;
    --allowed-filenames-txt) ALLOWED_FILENAMES_TXT="$2"; shift 2 ;;
    --available-audio-filenames-txt) AVAILABLE_AUDIO_FILENAMES_TXT="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) NUM_WORKERS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --pure-zero-ratio) PURE_ZERO_RATIO="$2"; shift 2 ;;
    --negative-margin-s) NEGATIVE_MARGIN_S="$2"; shift 2 ;;
    --center-bias-sigma-frac) CENTER_BIAS_SIGMA_FRAC="$2"; shift 2 ;;
    --freq-min-hz) FREQ_MIN_HZ="$2"; shift 2 ;;
    --freq-max-hz) FREQ_MAX_HZ="$2"; shift 2 ;;
    --edge-buffer-s) EDGE_BUFFER_S="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --qc-limit) QC_LIMIT="$2"; shift 2 ;;
    --use-wandb) USE_WANDB="true"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-name) WANDB_NAME="$2"; shift 2 ;;
    --wandb-tags) WANDB_TAGS="$2"; shift 2 ;;
    --install-detection-deps) INSTALL_DETECTION_DEPS="true"; shift ;;
    --skip-install-detection-deps) INSTALL_DETECTION_DEPS="false"; shift ;;
    --smoke-mode) SMOKE_MODE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$AUDIO_DIR" && -n "$AUDIO_BUNDLE_TAR" ]]; then
  echo "Error: use either --audio-dir or --audio-bundle-tar, not both"
  exit 1
fi
if [[ -z "$AUDIO_DIR" && -z "$AUDIO_BUNDLE_TAR" ]]; then
  echo "Error: one of --audio-dir or --audio-bundle-tar is required"
  exit 1
fi
if [[ -n "$AUDIO_DIR" && ! -d "$AUDIO_DIR" ]]; then
  echo "Error: --audio-dir must exist: $AUDIO_DIR"
  exit 1
fi
if [[ -n "$AUDIO_BUNDLE_TAR" && ! -f "$AUDIO_BUNDLE_TAR" ]]; then
  echo "Error: --audio-bundle-tar must exist: $AUDIO_BUNDLE_TAR"
  exit 1
fi
if [[ -n "$ALLOWED_FILENAMES_TXT" && ! -f "$ALLOWED_FILENAMES_TXT" ]]; then
  echo "Error: --allowed-filenames-txt must exist: $ALLOWED_FILENAMES_TXT"
  exit 1
fi
if [[ -n "$AVAILABLE_AUDIO_FILENAMES_TXT" && ! -f "$AVAILABLE_AUDIO_FILENAMES_TXT" ]]; then
  echo "Error: --available-audio-filenames-txt must exist: $AVAILABLE_AUDIO_FILENAMES_TXT"
  exit 1
fi

if [[ "$SMOKE_MODE" == "true" ]]; then
  MODEL_NAME="yolo26n.pt"
  EPOCHS=1
  BATCH_SIZE=4
  QC_LIMIT=12
fi

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
exec > >(tee -a "$LOG_DIR/fin_yolo26_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/fin_yolo26_${SLURM_JOB_ID:-$$}.err" >&2)

module load StdEnv/2023 gcc/12.3 python/3.11.5 opencv/4.11.0

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Bootstrapping Python venv at $VENV_PATH ..."
  mkdir -p "$(dirname "$VENV_PATH")"
  python -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

if [[ "$USE_WANDB" == "true" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Error: --use-wandb was requested but WANDB_API_KEY is not set and ~/.wandb_api_key was not found."
    exit 1
  fi
fi

echo "Staging project into $SLURM_TMPDIR ..."
rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/whale_project/"
export PYTHONPATH="${PYTHONPATH:-}:$SLURM_TMPDIR/whale_project/src"
cd "$SLURM_TMPDIR/whale_project"

REQ_STAMP_PATH="$VENV_PATH/.finwhale_yolo26_requirements.sha256"
REQ_FILES=("$SLURM_TMPDIR/whale_project/requirements.txt")
if [[ "$INSTALL_DETECTION_DEPS" == "true" ]]; then
  REQ_FILES+=("$SLURM_TMPDIR/whale_project/requirements-detection.txt")
fi
REQ_HASH="$(cat "${REQ_FILES[@]}" | sha256sum | awk '{print $1}')"
INSTALLED_REQ_HASH=""
if [[ -f "$REQ_STAMP_PATH" ]]; then
  INSTALLED_REQ_HASH="$(cat "$REQ_STAMP_PATH")"
fi

if [[ "$INSTALLED_REQ_HASH" != "$REQ_HASH" ]]; then
  echo "Installing Python requirements into $VENV_PATH ..."
  python -m pip install --upgrade pip setuptools wheel
  pip install -r "$SLURM_TMPDIR/whale_project/requirements.txt"
  if [[ "$INSTALL_DETECTION_DEPS" == "true" ]]; then
    pip install -r "$SLURM_TMPDIR/whale_project/requirements-detection.txt"
  fi
  printf '%s\n' "$REQ_HASH" > "$REQ_STAMP_PATH"
else
  echo "Python requirements already match stamp $REQ_HASH; reusing $VENV_PATH"
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_SLUG="finwhale_yolo26_${RUN_TAG}_${RUN_STAMP}"
TMP_ROOT="$SLURM_TMPDIR/finwhale_yolo26_pipeline"
MANIFEST_DIR="$TMP_ROOT/manifests"
SPLIT_DIR="$TMP_ROOT/splits"
EXPORT_DIR="$SLURM_TMPDIR/finwhale_detector_export"
YOLO_DIR="$SLURM_TMPDIR/finwhale_yolo26_data"
EXTRACT_DIR="$SLURM_TMPDIR/finwhale_bbox_audio"
TRAIN_DIR="$TMP_ROOT/train"
EVAL_DIR="$TMP_ROOT/eval_best"
FINAL_DIR="$OUTPUT_ROOT/$RUN_SLUG"

mkdir -p "$TMP_ROOT" "$EXPORT_DIR" "$YOLO_DIR" "$EXTRACT_DIR" "$TRAIN_DIR" "$EVAL_DIR" "$FINAL_DIR"

python -u scripts/data/detection/build_finwhale_bbox_manifests.py \
  --output-dir "$MANIFEST_DIR"

python -u scripts/data/detection/build_finwhale_bbox_splits.py \
  --annotation-manifest "$MANIFEST_DIR/unified_annotations.csv" \
  --clip-manifest "$MANIFEST_DIR/clip_manifest.csv" \
  --output-dir "$SPLIT_DIR"

if [[ "$SMOKE_MODE" == "true" && -z "$ALLOWED_FILENAMES_TXT" ]]; then
  ALLOWED_FILENAMES_TXT="$TMP_ROOT/smoke_allowed_filenames.txt"
  python - "$SPLIT_DIR/assignments.csv" "$ALLOWED_FILENAMES_TXT" "${AVAILABLE_AUDIO_FILENAMES_TXT:-}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

assignments_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
available_path = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

assignments = pd.read_csv(assignments_path)
if available_path and available_path.exists():
    available = {line.strip() for line in available_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    assignments = assignments[assignments["filename"].astype(str).isin(available)].copy()

quotas = {
    "train": 16,
    "val_2025": 6,
    "test_2025": 6,
    "val_hist": 4,
    "test_hist": 4,
}
chosen: list[str] = []
seen: set[str] = set()

for split_name, quota in quotas.items():
    group = assignments[assignments["split_name"].astype(str) == split_name].copy()
    if group.empty:
        continue
    group["_priority"] = (
        1000 * group["is_fin_positive"].fillna(0).astype(int)
        + 100 * group["is_annotated_non_fin"].fillna(0).astype(int)
        + group["fin_annotation_count"].fillna(0).astype(int)
    )
    group = group.sort_values(
        ["_priority", "fin_annotation_count", "non_fin_annotation_count", "filename"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    for filename in group["filename"].astype(str).tolist():
        if filename in seen:
            continue
        chosen.append(filename)
        seen.add(filename)
        if sum(1 for item in chosen if item in set(group["filename"].astype(str))) >= int(quota):
            break

if not chosen:
    raise SystemExit("smoke-mode filename allowlist is empty")

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(chosen) + "\n", encoding="utf-8")
print(f"Wrote {len(chosen)} smoke-mode allowed filenames to {output_path}")
PY
fi

if [[ -n "$AUDIO_BUNDLE_TAR" ]]; then
  AUDIO_EXTRACT_ARGS=(
    --annotation-manifest "$MANIFEST_DIR/unified_annotations.csv"
    --clip-manifest "$MANIFEST_DIR/clip_manifest.csv"
    --split-assignments "$SPLIT_DIR/assignments.csv"
    --output-path "$TMP_ROOT/audio_extract_members.txt"
    --tar-prefix raw_audio
    --context-duration-s 40.0
    --clip-duration-s 300.0
    --edge-buffer-s "$EDGE_BUFFER_S"
    --pure-zero-ratio "$PURE_ZERO_RATIO"
    --negative-margin-s "$NEGATIVE_MARGIN_S"
    --summary-path "$TMP_ROOT/audio_extract_summary.json"
  )
  if [[ -n "$ALLOWED_FILENAMES_TXT" ]]; then
    AUDIO_EXTRACT_ARGS+=(--allowed-filenames-txt "$ALLOWED_FILENAMES_TXT")
  fi
  if [[ -n "$AVAILABLE_AUDIO_FILENAMES_TXT" ]]; then
    AUDIO_EXTRACT_ARGS+=(--available-audio-filenames-txt "$AVAILABLE_AUDIO_FILENAMES_TXT")
  fi
  python -u scripts/data/detection/build_finwhale_bbox_audio_extract_list.py "${AUDIO_EXTRACT_ARGS[@]}"
  case "$AUDIO_BUNDLE_TAR" in
    *.tar.zst|*.tzst)
      tar --zstd -xf "$AUDIO_BUNDLE_TAR" -C "$EXTRACT_DIR" -T "$TMP_ROOT/audio_extract_members.txt"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$AUDIO_BUNDLE_TAR" -C "$EXTRACT_DIR" -T "$TMP_ROOT/audio_extract_members.txt"
      ;;
    *)
      tar -xf "$AUDIO_BUNDLE_TAR" -C "$EXTRACT_DIR" -T "$TMP_ROOT/audio_extract_members.txt"
      ;;
  esac
  AUDIO_DIR="$EXTRACT_DIR/raw_audio"
fi

EXPORT_ARGS=(
  --annotation-manifest "$MANIFEST_DIR/unified_annotations.csv"
  --clip-manifest "$MANIFEST_DIR/clip_manifest.csv"
  --split-assignments "$SPLIT_DIR/assignments.csv"
  --audio-dir "$AUDIO_DIR"
  --output-dir "$EXPORT_DIR"
  --config-path "$CONFIG_PATH"
  --pure-zero-ratio "$PURE_ZERO_RATIO"
  --negative-margin-s "$NEGATIVE_MARGIN_S"
  --center-bias-sigma-frac "$CENTER_BIAS_SIGMA_FRAC"
  --freq-min-hz "$FREQ_MIN_HZ"
  --freq-max-hz "$FREQ_MAX_HZ"
  --edge-buffer-s "$EDGE_BUFFER_S"
  --image-size "$IMAGE_SIZE"
  --seed "$SEED"
  --qc-limit "$QC_LIMIT"
)
if [[ -n "$ALLOWED_FILENAMES_TXT" ]]; then
  EXPORT_ARGS+=(--allowed-filenames-txt "$ALLOWED_FILENAMES_TXT")
fi
python -u scripts/data/detection/export_finwhale_bbox_dataset.py "${EXPORT_ARGS[@]}"

python -u scripts/data/detection/build_finwhale_yolo_dataset.py \
  --coco-export-dir "$EXPORT_DIR" \
  --output-dir "$YOLO_DIR" \
  --link-mode hardlink

if [[ "$USE_WANDB" == "true" ]]; then
  TRAIN_WANDB_NAME="${WANDB_NAME:-$RUN_SLUG-train}"
  python -u scripts/train/train_finwhale_yolo26.py \
    --data-yaml "$YOLO_DIR/yamls/data_train_val2025.yaml" \
    --output-dir "$TRAIN_DIR" \
    --model-name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --imgsz "$IMAGE_SIZE" \
    --device 0 \
    --workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --patience "$PATIENCE" \
    --project-name train \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-name "$TRAIN_WANDB_NAME" \
    --wandb-tags "$WANDB_TAGS"
else
  python -u scripts/train/train_finwhale_yolo26.py \
    --data-yaml "$YOLO_DIR/yamls/data_train_val2025.yaml" \
    --output-dir "$TRAIN_DIR" \
    --model-name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --imgsz "$IMAGE_SIZE" \
    --device 0 \
    --workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --patience "$PATIENCE" \
    --project-name train
fi

EVAL_YAMLS=""
for split_name in val_2025 test_2025 val_hist test_hist; do
  yaml_path="$YOLO_DIR/yamls/data_eval_${split_name}.yaml"
  if [[ -f "$yaml_path" ]] && compgen -G "$YOLO_DIR/images/$split_name/*" > /dev/null; then
    if [[ -n "$EVAL_YAMLS" ]]; then
      EVAL_YAMLS+=","
    fi
    EVAL_YAMLS+="${split_name}=${yaml_path}"
  fi
done

if [[ -n "$EVAL_YAMLS" && "$USE_WANDB" == "true" ]]; then
  EVAL_WANDB_NAME="${WANDB_NAME:-$RUN_SLUG-eval}"
  python -u scripts/train/eval_finwhale_yolo26.py \
    --weights "$TRAIN_DIR/best.pt" \
    --eval-yamls "$EVAL_YAMLS" \
    --output-dir "$EVAL_DIR" \
    --batch-size "$BATCH_SIZE" \
    --imgsz "$IMAGE_SIZE" \
    --device 0 \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-name "$EVAL_WANDB_NAME" \
    --wandb-tags "$WANDB_TAGS,eval"
elif [[ -n "$EVAL_YAMLS" ]]; then
  python -u scripts/train/eval_finwhale_yolo26.py \
    --weights "$TRAIN_DIR/best.pt" \
    --eval-yamls "$EVAL_YAMLS" \
    --output-dir "$EVAL_DIR" \
    --batch-size "$BATCH_SIZE" \
    --imgsz "$IMAGE_SIZE" \
    --device 0
else
  echo "No non-empty eval splits were exported; skipping eval step."
fi

mkdir -p "$FINAL_DIR/manifests" "$FINAL_DIR/splits" "$FINAL_DIR/export_metadata" "$FINAL_DIR/yolo_dataset" "$FINAL_DIR/train" "$FINAL_DIR/eval_best"
rsync -a "$MANIFEST_DIR/" "$FINAL_DIR/manifests/"
rsync -a "$SPLIT_DIR/" "$FINAL_DIR/splits/"
if [[ -f "$TMP_ROOT/audio_extract_members.txt" ]]; then
  cp "$TMP_ROOT/audio_extract_members.txt" "$FINAL_DIR/export_metadata/audio_extract_members.txt"
fi
if [[ -f "$TMP_ROOT/audio_extract_summary.json" ]]; then
  cp "$TMP_ROOT/audio_extract_summary.json" "$FINAL_DIR/export_metadata/audio_extract_summary.json"
fi
rsync -a \
  --include='*/' \
  --include='summary.json' \
  --include='context_manifest.csv' \
  --include='crop_manifest.csv' \
  --include='*.coco.json' \
  --include='qc/***' \
  --exclude='*' \
  "$EXPORT_DIR/" "$FINAL_DIR/export_metadata/"
rsync -a \
  --include='*/' \
  --include='summary.json' \
  --include='yamls/***' \
  --exclude='*' \
  "$YOLO_DIR/" "$FINAL_DIR/yolo_dataset/"
rsync -a "$TRAIN_DIR/" "$FINAL_DIR/train/"
rsync -a "$EVAL_DIR/" "$FINAL_DIR/eval_best/"

cat > "$FINAL_DIR/run_info.json" <<EOF
{
  "run_slug": "$RUN_SLUG",
  "audio_dir": "$AUDIO_DIR",
  "audio_bundle_tar": "${AUDIO_BUNDLE_TAR:-}",
  "allowed_filenames_txt": "${ALLOWED_FILENAMES_TXT:-}",
  "available_audio_filenames_txt": "${AVAILABLE_AUDIO_FILENAMES_TXT:-}",
  "project_path": "$PROJECT_PATH",
  "model_name": "$MODEL_NAME",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "smoke_mode": $([[ "$SMOKE_MODE" == "true" ]] && echo "true" || echo "false")
}
EOF

echo "YOLO26 run complete."
