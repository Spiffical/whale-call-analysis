#!/bin/bash
# Launch the recommended first YOLO26 fin-whale bbox experiment.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
SUBMIT_SCRIPT="$REPO_ROOT/drac/scripts/submit_finwhale_yolo26.sh"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 1
fi

AUDIO_DIR=""
AUDIO_BUNDLE_TAR=""
RUN_TAG="joint_v1"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_yolo26_first_experiment.sh (--audio-dir /path/to/raw_audio | --audio-bundle-tar /path/to/bundle.tar) [options]

Required:
  One of:
    --audio-dir PATH
    --audio-bundle-tar PATH

Optional:
  --run-tag TAG
  --smoke-mode
  Any additional arguments are passed through to submit_finwhale_yolo26.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --audio-bundle-tar) AUDIO_BUNDLE_TAR="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
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

CMD=(
  sbatch
  "$SUBMIT_SCRIPT"
  --run-tag "$RUN_TAG"
  --model-name yolo26m.pt
  --epochs 30
  --batch-size 8
  --workers 4
  --patience 20
  --pure-zero-ratio 0.5
  --center-bias-sigma-frac 0.25
  --freq-min-hz 1
  --freq-max-hz 200
  --use-wandb
  --wandb-project finwhale-bbox
  --wandb-group finwhale-yolo26-joint-v1
  --install-detection-deps
)

if [[ -n "$AUDIO_DIR" ]]; then
  CMD+=( --audio-dir "$AUDIO_DIR" )
else
  CMD+=( --audio-bundle-tar "$AUDIO_BUNDLE_TAR" )
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=( "${EXTRA_ARGS[@]}" )
fi

echo "Submitting YOLO26 experiment:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"
