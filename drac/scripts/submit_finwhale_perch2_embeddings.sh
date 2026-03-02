#!/bin/bash
# NOTE: Run with:
#   sbatch /path/to/whale-call-analysis/drac/scripts/submit_finwhale_perch2_embeddings.sh [args]
#
# Local smoke mode (no sbatch, for validation only):
#   bash drac/scripts/submit_finwhale_perch2_embeddings.sh --local-test-mode [args]

#SBATCH --account=def-kmoran
#SBATCH --job-name=finwhale_perch2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Detect repo root.
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
EXP_DIR="${EXP_DIR:-/exp}"

AUDIO_DIR=""
declare -a EXCEL_FILES=()
PERCH_MODEL="perch_v2_gpu"
BATCH_SIZE=16
CONTEXT_SECONDS="40"
TRAIN_CLIP_SECONDS="10"
EVAL_CLIP_SECONDS="10"
ASSUMED_CLIP_DURATION_SECONDS="300"
NEGATIVES_PER_POSITIVE=1
NEGATIVE_MARGIN_SECONDS="2"
MAX_POSITIVES=""
MAX_AUDIO_FILES=""
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
COPY_AUDIO_TO_TMP="false"
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

Required:
  --audio-dir PATH

Optional:
  --excel-file PATH               Repeatable; required unless repo default files exist.
  --excel-files-csv CSV           Comma-separated Excel paths.
  --perch-model NAME              perch_v2 | perch_v2_gpu | perch_v2_cpu (default: perch_v2_gpu)
  --batch-size N                  (default: 16)
  --context-seconds SEC           (default: 40)
  --train-clip-seconds SEC        (default: 10)
  --eval-clip-seconds SEC         (default: 10)
  --assumed-clip-duration-seconds SEC   (default: 300)
  --negatives-per-positive N      (default: 1)
  --negative-margin-seconds SEC   (default: 2)
  --max-positives N
  --max-audio-files N
  --train-ratio R                 (default: 0.8)
  --val-ratio R                   (default: 0.1)
  --min-gap-seconds SEC           (default: 120)
  --logreg-c VALUE                (default: 1.0)
  --max-iter N                    (default: 3000)
  --train-pos-augment-copies N    (default: 1)
  --train-neg-augment-copies N    (default: 1)
  --center-bias-sigma-frac VALUE  (default: 0.25)
  --seed N                        (default: 42)
  --run-tag TAG
  --note TEXT
  --exp-dir PATH                  (default: /exp)
  --project-path PATH             (default: detected repo root)
  --venv-path PATH                (default: <repo>/.venv)
  --copy-audio-to-tmp             Copy full audio dir to node local storage.
  --disable-gpu                   Pass --disable-gpu to training script.
  --skip-save-embeddings          Pass --skip-save-embeddings to training script.
  --install-perch-deps            pip install -r requirements-perch.txt inside job venv.
  --git-branch NAME               Enforce branch in project-path (default: main)
  --auto-switch-branch            Auto-checkout required branch in project-path.
  --local-test-mode               Run directly without SLURM module assumptions.
  -h, --help
USAGE
}

split_csv_to_array() {
  local raw="$1"
  raw="${raw// /}"
  IFS=',' read -r -a _parts <<< "$raw"
  for x in "${_parts[@]}"; do
    if [[ -n "$x" ]]; then
      EXCEL_FILES+=("$x")
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --excel-file) EXCEL_FILES+=("$2"); shift 2 ;;
    --excel-files-csv) split_csv_to_array "$2"; shift 2 ;;
    --perch-model) PERCH_MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --context-seconds) CONTEXT_SECONDS="$2"; shift 2 ;;
    --train-clip-seconds) TRAIN_CLIP_SECONDS="$2"; shift 2 ;;
    --eval-clip-seconds) EVAL_CLIP_SECONDS="$2"; shift 2 ;;
    --assumed-clip-duration-seconds) ASSUMED_CLIP_DURATION_SECONDS="$2"; shift 2 ;;
    --negatives-per-positive) NEGATIVES_PER_POSITIVE="$2"; shift 2 ;;
    --negative-margin-seconds) NEGATIVE_MARGIN_SECONDS="$2"; shift 2 ;;
    --max-positives) MAX_POSITIVES="$2"; shift 2 ;;
    --max-audio-files) MAX_AUDIO_FILES="$2"; shift 2 ;;
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
    --copy-audio-to-tmp) COPY_AUDIO_TO_TMP="true"; shift ;;
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

if [[ -z "$AUDIO_DIR" ]]; then
  echo "Error: --audio-dir is required"
  exit 1
fi
if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "Error: audio directory does not exist: $AUDIO_DIR"
  exit 1
fi
if [[ "$EXP_DIR" != /* ]]; then
  EXP_DIR="$PROJECT_PATH/$EXP_DIR"
fi

if [[ "${#EXCEL_FILES[@]}" -eq 0 ]]; then
  default_excel_1="$PROJECT_PATH/data/finwhales/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx"
  default_excel_2="$PROJECT_PATH/data/finwhales/Clayoquot_40Hz_Annotations_Rannankari.xlsx"
  if [[ -f "$default_excel_1" && -f "$default_excel_2" ]]; then
    EXCEL_FILES=("$default_excel_1" "$default_excel_2")
  else
    echo "Error: no Excel annotations provided and repo defaults were not found."
    echo "Pass --excel-file for each annotation workbook."
    exit 1
  fi
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
  LOG_DIR="$SCRATCH/whale-call-analysis/logs"
else
  LOG_DIR="$REPO_ROOT/output/drac_local_logs"
fi
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/finwhale_perch2_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/finwhale_perch2_${SLURM_JOB_ID:-$$}.err" >&2)

echo "Using REPO_ROOT: $REPO_ROOT"
echo "Using PROJECT_PATH: $PROJECT_PATH"
echo "Using SLURM_TMPDIR: $SLURM_TMPDIR"
echo "Using VENV_PATH: $VENV_PATH"
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

declare -a RESOLVED_EXCEL_FILES=()
for excel_path in "${EXCEL_FILES[@]}"; do
  if [[ "$excel_path" == /* ]]; then
    resolved="$excel_path"
  else
    resolved="$SLURM_TMPDIR/whale_project/$excel_path"
    if [[ ! -f "$resolved" ]]; then
      resolved="$PROJECT_PATH/$excel_path"
    fi
  fi
  if [[ ! -f "$resolved" ]]; then
    echo "Error: Excel file does not exist: $excel_path (resolved: $resolved)"
    exit 1
  fi
  RESOLVED_EXCEL_FILES+=("$resolved")
done

AUDIO_ARG="$AUDIO_DIR"
if [[ "$COPY_AUDIO_TO_TMP" == "true" ]]; then
  echo "Copying audio directory to node-local storage (can be very large)..."
  mkdir -p "$SLURM_TMPDIR/finwhale_audio"
  rsync -a "$AUDIO_DIR/" "$SLURM_TMPDIR/finwhale_audio/"
  AUDIO_ARG="$SLURM_TMPDIR/finwhale_audio"
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
BASE_FOLDER="finwhale-perch2-${MODEL_TAG}-b${BATCH_SIZE}-ctx${CTX_TAG}-tr${TRCLIP_TAG}-ev${EVCLIP_TAG}-npp${NEGATIVES_PER_POSITIVE}-cbs${CBS_TAG}-gap${GAP_TAG}-seed${SEED}"
if [[ -n "$RUN_TAG" ]]; then
  BASE_FOLDER="${BASE_FOLDER}-$(safe_tag "$RUN_TAG")"
fi
EXP_PATH="${EXP_DIR}/finwhale/perch2/${BASE_FOLDER}"
mkdir -p "$EXP_PATH"

PYTHON_CMD=(
  python -u scripts/train/train_perch2_embeddings.py
  --excel-files "${RESOLVED_EXCEL_FILES[@]}"
  --audio-dir "$AUDIO_ARG"
  --output-dir "$EXP_PATH"
  --perch-model "$PERCH_MODEL"
  --batch-size "$BATCH_SIZE"
  --context-seconds "$CONTEXT_SECONDS"
  --train-clip-seconds "$TRAIN_CLIP_SECONDS"
  --eval-clip-seconds "$EVAL_CLIP_SECONDS"
  --assumed-clip-duration-seconds "$ASSUMED_CLIP_DURATION_SECONDS"
  --negatives-per-positive "$NEGATIVES_PER_POSITIVE"
  --negative-margin-seconds "$NEGATIVE_MARGIN_SECONDS"
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

if [[ -n "$MAX_POSITIVES" ]]; then
  PYTHON_CMD+=( --max-positives "$MAX_POSITIVES" )
fi
if [[ -n "$MAX_AUDIO_FILES" ]]; then
  PYTHON_CMD+=( --max-audio-files "$MAX_AUDIO_FILES" )
fi
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
echo "  audio-dir: $AUDIO_DIR"
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
