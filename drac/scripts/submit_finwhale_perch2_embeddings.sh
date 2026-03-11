#!/bin/bash
# NOTE:
#   sbatch drac/scripts/submit_finwhale_perch2_embeddings.sh [args]
#
# Local smoke mode (no sbatch):
#   bash drac/scripts/submit_finwhale_perch2_embeddings.sh --local-test-mode [args]

#SBATCH --account=def-kmoran
#SBATCH --job-name=finwhale_perch2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/drac/scripts/submit_finwhale_perch2_embeddings.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/train/train_perch2_embeddings.py" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
  if [[ -d "$SCRIPT_DIR/../.." && -f "$SCRIPT_DIR/../../scripts/train/train_perch2_embeddings.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  else
    REPO_ROOT="$HOME/whale-call-analysis"
  fi
fi

PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
DEFAULT_EXP_DIR="${SCRATCH:-/scratch/$USER}/whale-call-analysis/perch2_training_runs"
EXP_DIR="${EXP_DIR:-$DEFAULT_EXP_DIR}"

CONTEXT_DATASET_TAR=""
CONTEXT_DATASET_DIR=""
CONTEXT_MANIFEST_RELPATH="context_window_manifest.csv"
CONTEXT_AUDIO_RELDIR="context_audio"
COPY_CONTEXT_DIR_TO_TMP="true"

PERCH_MODEL="perch_v2_gpu"
PERCH_MODEL_EXPLICIT="false"
BATCH_SIZE=16
CONTEXT_SECONDS="40"
TRAIN_CLIP_SECONDS="10"
EVAL_CLIP_SECONDS="10"
TRAIN_RATIO="0.8"
VAL_RATIO="0.1"
MIN_GAP_SECONDS="120"
LOGREG_C="1.0"
MAX_ITER=3000
TRAIN_POS_AUGMENT_COPIES=1
TRAIN_NEG_AUGMENT_COPIES=1
CENTER_BIAS_SIGMA_FRAC="0.25"
SEED=42
RUN_TAG=""
NOTE=""
DISABLE_GPU="false"
SKIP_SAVE_EMBEDDINGS="false"
INSTALL_PERCH_DEPS="false"
GIT_BRANCH="main"
AUTO_SWITCH_BRANCH="false"
LOCAL_TEST_MODE="false"

usage() {
  cat <<'USAGE'
Usage:
  sbatch drac/scripts/submit_finwhale_perch2_embeddings.sh [options]
  bash drac/scripts/submit_finwhale_perch2_embeddings.sh --local-test-mode [options]

Required (choose one):
  --context-dataset-tar PATH
  --context-dataset-dir PATH

Optional:
  --context-manifest-relpath PATH   Relative path in extracted dataset (default: context_window_manifest.csv)
  --context-audio-reldir PATH       Relative path in extracted dataset (default: context_audio)
  --no-copy-context-dir             In dir mode, do not rsync context dataset to SLURM_TMPDIR
  --perch-model NAME                perch_v2 | perch_v2_gpu | perch_v2_cpu (default: perch_v2_gpu)
  --batch-size N                    (default: 16)
  --context-seconds SEC             (default: 40)
  --train-clip-seconds SEC          (default: 10)
  --eval-clip-seconds SEC           (default: 10)
  --train-ratio R                   (default: 0.8)
  --val-ratio R                     (default: 0.1)
  --min-gap-seconds SEC             (default: 120)
  --logreg-c VALUE                  (default: 1.0)
  --max-iter N                      (default: 3000)
  --train-pos-augment-copies N      (default: 1)
  --train-neg-augment-copies N      (default: 1)
  --center-bias-sigma-frac VALUE    (default: 0.25)
  --seed N                          (default: 42)
  --run-tag TAG
  --note TEXT
  --exp-dir PATH                    (default: $SCRATCH/whale-call-analysis/perch2_training_runs)
  --project-path PATH               (default: detected repo root)
  --venv-path PATH                  (default: <repo>/.venv)
  --disable-gpu                     Force CPU mode for TensorFlow
  --skip-save-embeddings            Do not save embeddings.npz
  --install-perch-deps              pip install -r requirements-perch.txt in job venv
  --git-branch NAME                 Enforce branch in project path (default: main)
  --auto-switch-branch              Auto checkout required branch in project path
  --local-test-mode                 Run directly without SLURM module assumptions
  -h, --help
USAGE
}

resolve_file_in_dataset() {
  local dataset_root="$1"
  local relpath="$2"
  if [[ -f "$dataset_root/$relpath" ]]; then
    echo "$dataset_root/$relpath"
    return 0
  fi
  local found
  found="$(find "$dataset_root" -maxdepth 5 -type f -path "*/$relpath" -print -quit)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi
  return 1
}

resolve_dir_in_dataset() {
  local dataset_root="$1"
  local reldir="$2"
  if [[ -d "$dataset_root/$reldir" ]]; then
    echo "$dataset_root/$reldir"
    return 0
  fi
  local found
  found="$(find "$dataset_root" -maxdepth 5 -type d -path "*/$reldir" -print -quit)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context-dataset-tar) CONTEXT_DATASET_TAR="$2"; shift 2 ;;
    --context-dataset-dir) CONTEXT_DATASET_DIR="$2"; shift 2 ;;
    --context-manifest-relpath) CONTEXT_MANIFEST_RELPATH="$2"; shift 2 ;;
    --context-audio-reldir) CONTEXT_AUDIO_RELDIR="$2"; shift 2 ;;
    --no-copy-context-dir) COPY_CONTEXT_DIR_TO_TMP="false"; shift ;;
    --perch-model) PERCH_MODEL="$2"; PERCH_MODEL_EXPLICIT="true"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --context-seconds) CONTEXT_SECONDS="$2"; shift 2 ;;
    --train-clip-seconds) TRAIN_CLIP_SECONDS="$2"; shift 2 ;;
    --eval-clip-seconds) EVAL_CLIP_SECONDS="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --min-gap-seconds) MIN_GAP_SECONDS="$2"; shift 2 ;;
    --logreg-c) LOGREG_C="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --train-pos-augment-copies) TRAIN_POS_AUGMENT_COPIES="$2"; shift 2 ;;
    --train-neg-augment-copies) TRAIN_NEG_AUGMENT_COPIES="$2"; shift 2 ;;
    --center-bias-sigma-frac) CENTER_BIAS_SIGMA_FRAC="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --note) NOTE="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --project-path) PROJECT_PATH="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --disable-gpu) DISABLE_GPU="true"; shift ;;
    --skip-save-embeddings) SKIP_SAVE_EMBEDDINGS="true"; shift ;;
    --install-perch-deps) INSTALL_PERCH_DEPS="true"; shift ;;
    --git-branch) GIT_BRANCH="$2"; shift 2 ;;
    --auto-switch-branch) AUTO_SWITCH_BRANCH="true"; shift ;;
    --local-test-mode) LOCAL_TEST_MODE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$CONTEXT_DATASET_TAR" && -z "$CONTEXT_DATASET_DIR" ]]; then
  echo "Error: provide one of --context-dataset-tar or --context-dataset-dir"
  exit 1
fi
if [[ -n "$CONTEXT_DATASET_TAR" && -n "$CONTEXT_DATASET_DIR" ]]; then
  echo "Error: --context-dataset-tar and --context-dataset-dir are mutually exclusive"
  exit 1
fi
if [[ -n "$CONTEXT_DATASET_TAR" && ! -f "$CONTEXT_DATASET_TAR" ]]; then
  echo "Error: context dataset archive not found: $CONTEXT_DATASET_TAR"
  exit 1
fi
if [[ -n "$CONTEXT_DATASET_DIR" && ! -d "$CONTEXT_DATASET_DIR" ]]; then
  echo "Error: context dataset directory not found: $CONTEXT_DATASET_DIR"
  exit 1
fi
if [[ "$DISABLE_GPU" == "true" && "$PERCH_MODEL" == "perch_v2_gpu" ]]; then
  if [[ "$PERCH_MODEL_EXPLICIT" == "true" ]]; then
    echo "Error: --disable-gpu cannot be combined with --perch-model perch_v2_gpu."
    echo "       Use --perch-model perch_v2_cpu (recommended) or remove --disable-gpu."
    exit 1
  fi
  echo "Info: --disable-gpu enabled; switching Perch model from perch_v2_gpu to perch_v2_cpu."
  PERCH_MODEL="perch_v2_cpu"
fi
if [[ "$LOCAL_TEST_MODE" == "true" && "$EXP_DIR" == "$DEFAULT_EXP_DIR" ]]; then
  EXP_DIR="$REPO_ROOT/output/drac_local_exp"
fi
if [[ "$EXP_DIR" != /* ]]; then
  EXP_DIR="$PROJECT_PATH/$EXP_DIR"
fi

if [[ "$LOCAL_TEST_MODE" == "true" ]]; then
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    SLURM_TMPDIR="$(mktemp -d "/tmp/finwhale_perch2_local_${USER}_XXXXXX")"
  fi
else
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "Error: SLURM_TMPDIR is not set. Use sbatch or --local-test-mode."
    exit 1
  fi
fi

if [[ -n "${SCRATCH:-}" ]]; then
  LOG_DIR="$SCRATCH/whale-call-analysis/perch2_training_logs"
else
  LOG_DIR="$REPO_ROOT/output/drac_local_logs"
fi
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/finwhale_perch2_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/finwhale_perch2_${SLURM_JOB_ID:-$$}.err" >&2)

echo "Using REPO_ROOT: $REPO_ROOT"
echo "Using PROJECT_PATH: $PROJECT_PATH"
echo "Using SLURM_TMPDIR: $SLURM_TMPDIR"
echo "Using VENV_PATH: $VENV_PATH"
echo "Using EXP_DIR: $EXP_DIR"
echo "Using LOG_DIR: $LOG_DIR"
echo "Local test mode: $LOCAL_TEST_MODE"

if [[ "$LOCAL_TEST_MODE" != "true" ]]; then
  module load python/3.10
fi
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Error: venv not found at $VENV_PATH/bin/activate"
  exit 2
fi
source "$VENV_PATH/bin/activate"

KAGGLE_ROOT="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
if [[ ! -f "$KAGGLE_ROOT/kaggle.json" ]]; then
  echo "Warning: Kaggle credentials were not found at $KAGGLE_ROOT/kaggle.json"
  echo "         First Perch checkpoint download may fail unless credentials are configured."
fi

if git -C "$PROJECT_PATH" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  CURRENT_BRANCH="$(git -C "$PROJECT_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
  if [[ "$CURRENT_BRANCH" != "$GIT_BRANCH" ]]; then
    echo "Repository at $PROJECT_PATH is on branch '$CURRENT_BRANCH', but '$GIT_BRANCH' is required."
    if [[ "$AUTO_SWITCH_BRANCH" == "true" ]]; then
      echo "Auto-switching to '$GIT_BRANCH' ..."
      git -C "$PROJECT_PATH" fetch origin "$GIT_BRANCH" || true
      git -C "$PROJECT_PATH" checkout "$GIT_BRANCH" 2>/dev/null \
        || git -C "$PROJECT_PATH" checkout -B "$GIT_BRANCH" "origin/$GIT_BRANCH" \
        || { echo "Error: failed to checkout '$GIT_BRANCH' in $PROJECT_PATH"; exit 1; }
      git -C "$PROJECT_PATH" pull --ff-only || true
    else
      echo "Error: wrong branch. Re-run with --auto-switch-branch or switch manually:"
      echo "  cd $PROJECT_PATH && git checkout $GIT_BRANCH && git pull --ff-only"
      exit 1
    fi
  fi
else
  echo "Warning: $PROJECT_PATH is not a git repository; proceeding without branch enforcement."
fi

echo "Copying project to node-local storage..."
rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/whale_project/"

if [[ "$INSTALL_PERCH_DEPS" == "true" ]]; then
  echo "Installing Perch dependencies in active venv..."
  pip install -r "$SLURM_TMPDIR/whale_project/requirements-perch.txt"
fi

CONTEXT_ROOT=""
if [[ -n "$CONTEXT_DATASET_TAR" ]]; then
  CONTEXT_ROOT="$SLURM_TMPDIR/perch2_context_dataset"
  rm -rf "$CONTEXT_ROOT"
  mkdir -p "$CONTEXT_ROOT"
  echo "Extracting context dataset archive to $CONTEXT_ROOT ..."

  if [[ "$CONTEXT_DATASET_TAR" == *.tar.gz || "$CONTEXT_DATASET_TAR" == *.tgz ]]; then
    if command -v pigz >/dev/null 2>&1; then
      tar --use-compress-program=pigz -xf "$CONTEXT_DATASET_TAR" -C "$CONTEXT_ROOT"
    else
      tar -xzf "$CONTEXT_DATASET_TAR" -C "$CONTEXT_ROOT"
    fi
  elif [[ "$CONTEXT_DATASET_TAR" == *.tar.zst || "$CONTEXT_DATASET_TAR" == *.tzst ]]; then
    if command -v unzstd >/dev/null 2>&1; then
      tar --use-compress-program=unzstd -xf "$CONTEXT_DATASET_TAR" -C "$CONTEXT_ROOT"
    elif command -v zstd >/dev/null 2>&1; then
      zstd -dc "$CONTEXT_DATASET_TAR" | tar -xf - -C "$CONTEXT_ROOT"
    else
      echo "Error: cannot extract $CONTEXT_DATASET_TAR (need unzstd or zstd)"
      exit 1
    fi
  elif [[ "$CONTEXT_DATASET_TAR" == *.tar ]]; then
    tar -xf "$CONTEXT_DATASET_TAR" -C "$CONTEXT_ROOT"
  elif [[ "$CONTEXT_DATASET_TAR" == *.zip ]]; then
    if command -v unzip >/dev/null 2>&1; then
      unzip -q "$CONTEXT_DATASET_TAR" -d "$CONTEXT_ROOT"
    else
      echo "Error: unzip not found on PATH"
      exit 1
    fi
  else
    echo "Error: unsupported context archive format: $CONTEXT_DATASET_TAR"
    exit 1
  fi
else
  if [[ "$COPY_CONTEXT_DIR_TO_TMP" == "true" ]]; then
    CONTEXT_ROOT="$SLURM_TMPDIR/perch2_context_dataset"
    rm -rf "$CONTEXT_ROOT"
    mkdir -p "$CONTEXT_ROOT"
    echo "Copying context dataset directory to node-local storage..."
    rsync -a "$CONTEXT_DATASET_DIR/" "$CONTEXT_ROOT/"
  else
    CONTEXT_ROOT="$CONTEXT_DATASET_DIR"
  fi
fi

CONTEXT_MANIFEST_PATH="$(resolve_file_in_dataset "$CONTEXT_ROOT" "$CONTEXT_MANIFEST_RELPATH" || true)"
if [[ -z "$CONTEXT_MANIFEST_PATH" || ! -f "$CONTEXT_MANIFEST_PATH" ]]; then
  echo "Error: context manifest not found in dataset root: $CONTEXT_ROOT"
  echo "       looked for relpath: $CONTEXT_MANIFEST_RELPATH"
  exit 1
fi
CONTEXT_AUDIO_DIR="$(resolve_dir_in_dataset "$CONTEXT_ROOT" "$CONTEXT_AUDIO_RELDIR" || true)"
if [[ -z "$CONTEXT_AUDIO_DIR" || ! -d "$CONTEXT_AUDIO_DIR" ]]; then
  echo "Error: context audio directory not found in dataset root: $CONTEXT_ROOT"
  echo "       looked for relpath: $CONTEXT_AUDIO_RELDIR"
  exit 1
fi

safe_tag() {
  echo "$1" | tr -cs '[:alnum:]_-' '_' | sed 's/^_//;s/_$//'
}

fmt_num() {
  printf '%.3f' "$1" | sed 's/0*$//' | sed 's/\.$//' | tr '.' 'p'
}

MODEL_TAG="$(safe_tag "$PERCH_MODEL")"
CTX_TAG="$(fmt_num "$CONTEXT_SECONDS")"
TRCLIP_TAG="$(fmt_num "$TRAIN_CLIP_SECONDS")"
EVCLIP_TAG="$(fmt_num "$EVAL_CLIP_SECONDS")"
CBS_TAG="$(fmt_num "$CENTER_BIAS_SIGMA_FRAC")"
GAP_TAG="$(fmt_num "$MIN_GAP_SECONDS")"
BASE_FOLDER="finwhale-perch2-${MODEL_TAG}-b${BATCH_SIZE}-ctx${CTX_TAG}-tr${TRCLIP_TAG}-ev${EVCLIP_TAG}-cbs${CBS_TAG}-gap${GAP_TAG}-seed${SEED}"
if [[ -n "$RUN_TAG" ]]; then
  BASE_FOLDER="${BASE_FOLDER}-$(safe_tag "$RUN_TAG")"
fi
EXP_PATH="${EXP_DIR}/finwhale/perch2/${BASE_FOLDER}"
mkdir -p "$EXP_PATH"

PYTHON_CMD=(
  python -u scripts/train/train_perch2_embeddings.py
  --context-manifest-csv "$CONTEXT_MANIFEST_PATH"
  --context-audio-dir "$CONTEXT_AUDIO_DIR"
  --output-dir "$EXP_PATH"
  --perch-model "$PERCH_MODEL"
  --batch-size "$BATCH_SIZE"
  --context-seconds "$CONTEXT_SECONDS"
  --train-clip-seconds "$TRAIN_CLIP_SECONDS"
  --eval-clip-seconds "$EVAL_CLIP_SECONDS"
  --train-ratio "$TRAIN_RATIO"
  --val-ratio "$VAL_RATIO"
  --min-gap-seconds "$MIN_GAP_SECONDS"
  --seed "$SEED"
  --logreg-c "$LOGREG_C"
  --max-iter "$MAX_ITER"
  --train-pos-augment-copies "$TRAIN_POS_AUGMENT_COPIES"
  --train-neg-augment-copies "$TRAIN_NEG_AUGMENT_COPIES"
  --center-bias-sigma-frac "$CENTER_BIAS_SIGMA_FRAC"
)

if [[ "$DISABLE_GPU" == "true" ]]; then
  PYTHON_CMD+=( --disable-gpu )
fi
if [[ "$SKIP_SAVE_EMBEDDINGS" == "true" ]]; then
  PYTHON_CMD+=( --skip-save-embeddings )
fi
if [[ -n "$NOTE" ]]; then
  PYTHON_CMD+=( --note "$NOTE" )
fi

export PYTHONPATH="${PYTHONPATH:-}:$SLURM_TMPDIR/whale_project/src"
cd "$SLURM_TMPDIR/whale_project"

echo "Submitting FinWhale Perch2 embedding job"
if [[ -n "$CONTEXT_DATASET_TAR" ]]; then
  echo "  context-dataset-archive: $CONTEXT_DATASET_TAR"
else
  echo "  context-dataset-dir: $CONTEXT_DATASET_DIR"
fi
echo "  resolved-context-root: $CONTEXT_ROOT"
echo "  resolved-context-manifest: $CONTEXT_MANIFEST_PATH"
echo "  resolved-context-audio-dir: $CONTEXT_AUDIO_DIR"
echo "  perch-model: $PERCH_MODEL | batch-size: $BATCH_SIZE"
echo "  context: $CONTEXT_SECONDS | train-clip: $TRAIN_CLIP_SECONDS | eval-clip: $EVAL_CLIP_SECONDS"
echo "  train_pos_augment_copies: $TRAIN_POS_AUGMENT_COPIES | train_neg_augment_copies: $TRAIN_NEG_AUGMENT_COPIES"
echo "  center_bias_sigma_frac: $CENTER_BIAS_SIGMA_FRAC"
echo "  output-root: $EXP_PATH"
echo "Running: ${PYTHON_CMD[*]}"
"${PYTHON_CMD[@]}"

if [[ "$LOCAL_TEST_MODE" == "true" ]]; then
  echo "Local test mode completed. Temporary directory: $SLURM_TMPDIR"
fi
