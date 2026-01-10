#!/bin/bash
# NOTE: Run with: sbatch /path/to/whale-call-analysis/drac/scripts/submit_finwhale_cnn.sh [args]
# Logs will be created in $SCRATCH/whale-call-analysis/logs/

#SBATCH --account=def-kmoran                    # DRAC project account
#SBATCH --job-name=finwhale_cnn                 # Job name
#SBATCH --time=08:00:00                         # Max runtime (HH:MM:SS)
#SBATCH --gres=gpu:h100:1                      # GPU type: adjust if needed (e.g., a100:1)
#SBATCH --cpus-per-task=4                       # CPU cores
#SBATCH --mem=32G                               # Memory per node

# Detect repo root - use SLURM_SUBMIT_DIR if available (set by sbatch),
# otherwise try to resolve from the original script path
if [[ -n "$SLURM_SUBMIT_DIR" && -f "$SLURM_SUBMIT_DIR/drac/scripts/submit_finwhale_cnn.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -n "$SLURM_SUBMIT_DIR" && -f "$SLURM_SUBMIT_DIR/scripts/train_cnn.py" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  # Fallback: assume script was called with absolute path, resolve it
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  if [[ -L "$SCRIPT_PATH" ]]; then
    SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
  fi
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
  if [[ -d "$SCRIPT_DIR/../.." && -f "$SCRIPT_DIR/../../scripts/train_cnn.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  else
    # Last resort: assume ~/whale-call-analysis
    REPO_ROOT="$HOME/whale-call-analysis"
  fi
fi
echo "Using REPO_ROOT: $REPO_ROOT"

# Create log directories and set up logging
LOG_DIR="$SCRATCH/whale-call-analysis/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/finwhale_cnn_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/finwhale_cnn_${SLURM_JOB_ID:-$$}.err" >&2)

# Parameters (with defaults)
POS_DIR=""
NEG_DIR=""
# WandB settings
USE_WANDB="false"
WANDB_PROJECT="finwhale_cnn"
WANDB_GROUP="supervised_cnn"
WANDB_ENTITY=""
EPOCHS=20
BATCH_SIZE=64
NUM_WORKERS=4
BALANCE="weighted"      # weighted | oversample | none
LR=1e-3
TRAIN_RATIO=0.8
VAL_RATIO=0.1
CROP_SIZE=""            # Empty = full freq range (square). Can be "96" or "96,96" for [freq,time]
DEVICE="cuda"
PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"   # Path to this repo on DRAC login node
EXP_DIR="/exp"                # Base experiment dir (shared scratch/project recommended)
COPY_TO_TMP="true"           # Default to true as requested
GIT_BRANCH="main"        # Required branch in PROJECT_PATH
AUTO_SWITCH_BRANCH="false"    # If true, auto checkout required branch in PROJECT_PATH
SEED=42
TAR_PATH=""
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
SPLIT_STRATEGY="internal"
MIN_GAP_SECONDS=120
MODEL="SmallCNN"
MAIN_METRIC="f1"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pos-dir) POS_DIR="$2"; shift 2 ;;
    --neg-dir) NEG_DIR="$2"; shift 2 ;;
    --use-wandb) USE_WANDB="true"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --balance) BALANCE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --copy-to-tmp) COPY_TO_TMP="true"; shift ;;
    --no-copy) COPY_TO_TMP="false"; shift ;;
    --git-branch) GIT_BRANCH="$2"; shift 2 ;;
    --auto-switch-branch) AUTO_SWITCH_BRANCH="true"; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --tar-path) TAR_PATH="$2"; shift 2 ;;
    --split-strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --min-gap-seconds) MIN_GAP_SECONDS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --main-metric) MAIN_METRIC="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Validate required args
if [[ -z "$POS_DIR" || -z "$NEG_DIR" ]]; then
  if [[ -n "$TAR_PATH" ]]; then
    echo "Using --tar-path=$TAR_PATH to populate POS/NEG directories"
  else
    echo "Error: --pos-dir and --neg-dir are required unless --tar-path is provided"
    exit 1
  fi
fi


echo "Submitting FinWhale CNN job"
echo "  pos-dir: $POS_DIR"
echo "  neg-dir: $NEG_DIR"
echo "  project: $WANDB_PROJECT | group: $WANDB_GROUP | entity: ${WANDB_ENTITY:-<default>}"
echo "  epochs: $EPOCHS | batch: $BATCH_SIZE | lr: $LR | balance: $BALANCE"
echo "  train_ratio: $TRAIN_RATIO | val_ratio: $VAL_RATIO | crop: $CROP_SIZE"
echo "  copy_to_tmp: $COPY_TO_TMP"

# Load modules and venv
module load python/3.10
if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "Error: venv not found at $VENV_PATH/bin/activate"
  exit 2
fi
source "$VENV_PATH/bin/activate"

# WandB API key setup (required for logging from compute nodes)
# Can be set via: environment variable, ~/.wandb_api_key file, or --wandb-key arg
if [[ -z "$WANDB_API_KEY" ]]; then
  if [[ -f "$HOME/.wandb_api_key" ]]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_api_key")
    echo "Loaded WANDB_API_KEY from ~/.wandb_api_key"
  else
    echo "Warning: WANDB_API_KEY not set. WandB logging may fail."
    echo "  Set it via: export WANDB_API_KEY=your_key"
    echo "  Or create: ~/.wandb_api_key with your API key"
  fi
fi

# Load W&B API key from .env if present
if [[ -f "$PROJECT_PATH/.env" ]]; then
  export $(grep -v '^#' "$PROJECT_PATH/.env" | xargs)
fi

# Prepare project in local node scratch and ensure requested branch from PROJECT_PATH
echo "Preparing project in $SLURM_TMPDIR ..."

if git -C "$PROJECT_PATH" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  CURRENT_BRANCH=$(git -C "$PROJECT_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
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

rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/whale_project/"

# Optionally extract tar archive into node-local storage and set POS/NEG
if [[ -n "$TAR_PATH" ]]; then
  echo "Extracting tar archive to $SLURM_TMPDIR ..."
  mkdir -p "$SLURM_TMPDIR/finwhale_data"
  # Support .tar, .tar.gz/.tgz and .zip
  if [[ "$TAR_PATH" == *.tar.gz || "$TAR_PATH" == *.tgz ]]; then
    tar -xzf "$TAR_PATH" -C "$SLURM_TMPDIR/finwhale_data"
  elif [[ "$TAR_PATH" == *.tar ]]; then
    tar -xf "$TAR_PATH" -C "$SLURM_TMPDIR/finwhale_data"
  elif [[ "$TAR_PATH" == *.zip ]]; then
    if command -v unzip >/dev/null 2>&1; then
      unzip -q "$TAR_PATH" -d "$SLURM_TMPDIR/finwhale_data"
    else
      echo "Error: unzip not found on system PATH"
      exit 1
    fi
  else
    echo "Error: Unsupported archive format for --tar-path: $TAR_PATH"
    exit 1
  fi
  # Detect mat_files and neg_mat_files inside the extracted tree
  # Prefer top-level if present
  if [[ -d "$SLURM_TMPDIR/finwhale_data/mat_files" && -d "$SLURM_TMPDIR/finwhale_data/neg_mat_files" ]]; then
    POS_ARG="$SLURM_TMPDIR/finwhale_data/mat_files"
    NEG_ARG="$SLURM_TMPDIR/finwhale_data/neg_mat_files"
  else
    # Search one level down
    ROOT_SUBDIR=$(find "$SLURM_TMPDIR/finwhale_data" -maxdepth 2 -type d -name mat_files -print -quit)
    if [[ -n "$ROOT_SUBDIR" ]]; then
      POS_ARG="$ROOT_SUBDIR"
      # infer neg path next to it
      CANDIDATE_NEG_DIR="$(dirname "$ROOT_SUBDIR")/neg_mat_files"
      if [[ -d "$CANDIDATE_NEG_DIR" ]]; then
        NEG_ARG="$CANDIDATE_NEG_DIR"
      else
        echo "Error: Could not find neg_mat_files adjacent to $ROOT_SUBDIR"
        exit 1
      fi
    else
      echo "Error: Could not find mat_files and neg_mat_files in extracted archive"
      exit 1
    fi
  fi
  echo "Resolved POS_ARG=$POS_ARG"
  echo "Resolved NEG_ARG=$NEG_ARG"
else
  # Optionally copy data to node-local storage (CAUTION: may be huge)
  POS_ARG="$POS_DIR"
  NEG_ARG="$NEG_DIR"
  if [[ "$COPY_TO_TMP" == "true" ]]; then
    echo "Copying data to node-local storage (this may take a long time) ..."
    mkdir -p "$SLURM_TMPDIR/finwhale_data/pos" "$SLURM_TMPDIR/finwhale_data/neg"
    rsync -a "$POS_DIR/" "$SLURM_TMPDIR/finwhale_data/pos/"
    rsync -a "$NEG_DIR/" "$SLURM_TMPDIR/finwhale_data/neg/"
    POS_ARG="$SLURM_TMPDIR/finwhale_data/pos"
    NEG_ARG="$SLURM_TMPDIR/finwhale_data/neg"
  fi
fi

# Build experiment directory and python command
BASE_FOLDER="finwhale-cnn-b${BATCH_SIZE}-lr${LR}-tr$(printf '%.1f' ${TRAIN_RATIO})-${WANDB_GROUP}"
EXP_PATH="${EXP_DIR}/finwhale/${BASE_FOLDER}"
mkdir -p "$EXP_PATH"

PYTHON_SCRIPT="$SLURM_TMPDIR/whale_project/scripts/train_cnn.py"
PYTHON_CMD=(
  python -u -W ignore "$PYTHON_SCRIPT"
  --pos-dir "$POS_ARG" --neg-dir "$NEG_ARG"
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
  --lr "$LR" --balance "$BALANCE"
  --train-ratio "$TRAIN_RATIO" --val-ratio "$VAL_RATIO"
  --device "$DEVICE"
  --exp_dir "$EXP_PATH" --save-path "$EXP_PATH/best.pt" --seed "$SEED"
  --split-strategy "$SPLIT_STRATEGY" --min-gap-seconds "$MIN_GAP_SECONDS" --model "$MODEL"
  --main-metric "$MAIN_METRIC"
)

# WandB arguments
if [[ "$USE_WANDB" == "true" ]]; then
  PYTHON_CMD+=( --use_wandb --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" )
  if [[ -n "$WANDB_ENTITY" ]]; then
    PYTHON_CMD+=( --wandb_entity "$WANDB_ENTITY" )
  fi
fi

# Add optional crop-size if specified
if [[ -n "$CROP_SIZE" ]]; then
  PYTHON_CMD+=( --crop-size "$CROP_SIZE" )
fi

# Ensure src is importable
export PYTHONPATH="$PYTHONPATH:$SLURM_TMPDIR/whale_project/src"

echo "Running: ${PYTHON_CMD[*]}"
cd "$SLURM_TMPDIR/whale_project"
"${PYTHON_CMD[@]}"
