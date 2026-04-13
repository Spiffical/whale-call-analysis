#!/bin/bash
# Submit CAM / attribution localization experiments for the Part 2 fin-whale bundle.

#SBATCH --account=def-kmoran
#SBATCH --job-name=fin_attn
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/drac/scripts/submit_finwhale_attention_experiment.sh" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
  if [[ -L "$SCRIPT_PATH" ]]; then
    SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
  fi
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
  if [[ -d "$SCRIPT_DIR/../.." && -f "$SCRIPT_DIR/../../scripts/analysis/run_finwhale_attention_experiment.py" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
  else
    REPO_ROOT="$HOME/whale-call-analysis"
  fi
fi

PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
BUNDLE_TAR=""
OUT_DIR=""
MODE="pilot"
METHODS="gradcampp,hirescam,layercam,scorecam,integrated_gradients"
SPLIT_DIR=""
ALL_ANNOTATIONS_RELPATH="manifests/annotations_all.csv"
FIN_ANNOTATIONS_RELPATH="manifests/fin_annotations.csv"
MAT_DIR_RELPATH="mat_files"
LAYER_PRESET="last"
DEVICE="cuda"
CHECKPOINT_SPECS=()
EXTRA_ARGS=()

extract_archive_member() {
  local archive_path="$1"
  local member_path="$2"
  local dest_root="$3"
  mkdir -p "$dest_root"
  case "$archive_path" in
    *.tar.gz|*.tgz)
      tar -xzf "$archive_path" -C "$dest_root" "$member_path"
      ;;
    *.tar.zst)
      tar --use-compress-program=unzstd -xf "$archive_path" -C "$dest_root" "$member_path"
      ;;
    *.tar)
      tar -xf "$archive_path" -C "$dest_root" "$member_path"
      ;;
    *.zip)
      unzip -q "$archive_path" "$member_path" -d "$dest_root"
      ;;
    *)
      echo "Unsupported checkpoint archive format: $archive_path"
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-tar) BUNDLE_TAR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --split-dir) SPLIT_DIR="$2"; shift 2 ;;
    --layer-preset) LAYER_PRESET="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --checkpoint-spec) CHECKPOINT_SPECS+=("$2"); shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$BUNDLE_TAR" || -z "$OUT_DIR" ]]; then
  echo "Error: --bundle-tar and --out-dir are required"
  exit 1
fi
if [[ ${#CHECKPOINT_SPECS[@]} -eq 0 ]]; then
  echo "Error: provide at least one --checkpoint-spec label=/path/to/best.pt"
  exit 1
fi

LOG_DIR="$SCRATCH/whale-call-analysis/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/fin_attn_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/fin_attn_${SLURM_JOB_ID:-$$}.err" >&2)

module load python/3.11.5
module load opencv/4.11.0
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Error: venv not found at $VENV_PATH/bin/activate"
  exit 2
fi
source "$VENV_PATH/bin/activate"

rsync -a --delete --exclude='.git' "$PROJECT_PATH/" "$SLURM_TMPDIR/whale_project/"
cd "$SLURM_TMPDIR/whale_project"

mkdir -p "$SLURM_TMPDIR/part2_bundle"
if [[ "$BUNDLE_TAR" == *.tar.gz || "$BUNDLE_TAR" == *.tgz ]]; then
  tar -xzf "$BUNDLE_TAR" -C "$SLURM_TMPDIR/part2_bundle"
elif [[ "$BUNDLE_TAR" == *.tar.zst ]]; then
  tar --use-compress-program=unzstd -xf "$BUNDLE_TAR" -C "$SLURM_TMPDIR/part2_bundle"
elif [[ "$BUNDLE_TAR" == *.tar ]]; then
  tar -xf "$BUNDLE_TAR" -C "$SLURM_TMPDIR/part2_bundle"
else
  echo "Unsupported archive format: $BUNDLE_TAR"
  exit 1
fi

BUNDLE_ROOT="$SLURM_TMPDIR/part2_bundle"
if [[ ! -d "$BUNDLE_ROOT/$MAT_DIR_RELPATH" ]]; then
  FOUND_MAT_DIR="$(find "$BUNDLE_ROOT" -maxdepth 2 -type d -name "$(basename "$MAT_DIR_RELPATH")" | head -n 1 || true)"
  if [[ -n "$FOUND_MAT_DIR" ]]; then
    BUNDLE_ROOT="$(dirname "$FOUND_MAT_DIR")"
  fi
fi

MAT_DIR="$BUNDLE_ROOT/$MAT_DIR_RELPATH"
ALL_ANNOTATIONS_CSV="$BUNDLE_ROOT/$ALL_ANNOTATIONS_RELPATH"
FIN_ANNOTATIONS_CSV="$BUNDLE_ROOT/$FIN_ANNOTATIONS_RELPATH"

if [[ -n "$SPLIT_DIR" ]]; then
  FIN_ANNOTATIONS_CSV="$SPLIT_DIR/fin_annotations.csv"
fi

if [[ ! -d "$MAT_DIR" || ! -f "$FIN_ANNOTATIONS_CSV" ]]; then
  echo "Resolved bundle root: $BUNDLE_ROOT"
  echo "Missing required attention experiment inputs."
  echo "  MAT_DIR=$MAT_DIR"
  echo "  FIN_ANNOTATIONS_CSV=$FIN_ANNOTATIONS_CSV"
  exit 1
fi

RUN_OUT_DIR="$OUT_DIR/finwhale_attention_${MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_OUT_DIR"

CMD=(
  python -u scripts/analysis/run_finwhale_attention_experiment.py
  --mode "$MODE"
  --methods "$METHODS"
  --mat-dir "$MAT_DIR"
  --fin-annotations-csv "$FIN_ANNOTATIONS_CSV"
  --output-dir "$RUN_OUT_DIR"
  --device "$DEVICE"
  --layer-preset "$LAYER_PRESET"
)

if [[ -f "$ALL_ANNOTATIONS_CSV" ]]; then
  CMD+=( --all-annotations-csv "$ALL_ANNOTATIONS_CSV" )
fi

RESOLVED_CHECKPOINT_SPECS=()
for spec in "${CHECKPOINT_SPECS[@]}"; do
  label="${spec%%=*}"
  raw_path="${spec#*=}"
  if [[ "$raw_path" == *"::"* ]]; then
    archive_path="${raw_path%%::*}"
    member_path="${raw_path#*::}"
    checkpoint_stage_root="$SLURM_TMPDIR/checkpoints/$label"
    extract_archive_member "$archive_path" "$member_path" "$checkpoint_stage_root"
    resolved_path="$checkpoint_stage_root/$member_path"
    if [[ ! -f "$resolved_path" ]]; then
      echo "Failed to extract checkpoint member."
      echo "  archive: $archive_path"
      echo "  member:  $member_path"
      exit 1
    fi
    RESOLVED_CHECKPOINT_SPECS+=( "$label=$resolved_path" )
  else
    if [[ ! -f "$raw_path" ]]; then
      echo "Checkpoint not found: $raw_path"
      exit 1
    fi
    RESOLVED_CHECKPOINT_SPECS+=( "$spec" )
  fi
done

for spec in "${RESOLVED_CHECKPOINT_SPECS[@]}"; do
  CMD+=( --checkpoint "$spec" )
done

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=( "${EXTRA_ARGS[@]}" )
fi

echo "Running attention experiment:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Attention outputs written to: $RUN_OUT_DIR"
