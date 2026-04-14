#!/bin/bash
# Launch the recommended first RT-DETR fin-whale bbox experiment.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
SUBMIT_SCRIPT="$REPO_ROOT/drac/scripts/submit_finwhale_rtdetr.sh"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 1
fi

AUDIO_DIR=""
RUN_TAG="joint_v1"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_rtdetr_first_experiment.sh --audio-dir /path/to/raw_audio [options]

Required:
  --audio-dir PATH

Optional:
  --run-tag TAG
  --smoke-mode
  Any additional arguments are passed through to submit_finwhale_rtdetr.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$AUDIO_DIR" ]]; then
  echo "Error: --audio-dir is required"
  exit 1
fi

CMD=(
  sbatch
  "$SUBMIT_SCRIPT"
  --audio-dir "$AUDIO_DIR"
  --run-tag "$RUN_TAG"
  --epochs 20
  --train-batch-size 4
  --eval-batch-size 4
  --gradient-accumulation-steps 1
  --learning-rate 5e-5
  --weight-decay 1e-4
  --warmup-ratio 0.1
  --pure-zero-ratio 0.5
  --center-bias-sigma-frac 0.25
  --freq-min-hz 1
  --freq-max-hz 200
  --install-detection-deps
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=( "${EXTRA_ARGS[@]}" )
fi

echo "Submitting RT-DETR experiment:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"
