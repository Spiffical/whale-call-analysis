#!/bin/bash
# Build an ONC common-row evaluation H5 for E128 SSAMBA binary-gate scoring.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_REL="repo_e24_expert_hparam_68be99f"
MANIFEST_REL="manifests/e100_onc_only_blocked_nov_validation_20260611T020900Z/E101_stage2_ONConly_blocked_nov20_25_30_val/standardized_manifest.csv"
REPO_ON_NIBI=""
MANIFEST_CSV=""
DATASET_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502/mat_archives/multiband40s_20260514T002301Z/extracted"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT=""
OUTPUT_H5=""
OUTPUT_SUMMARY=""
PYTHON_BIN="python3"
DEPENDENCY=""
DRY_RUN="false"
REPO_SET="false"
MANIFEST_SET="false"

BAND="low"
BAND_CROP_SHAPE="391x50"
OUTPUT_SHAPE="512x512"
SPLITS="val,test"
NON_TARGET_MODE="normal"
AMBIGUOUS_MODE="skip"
MAX_NORMAL="0"
MAX_PER_TARGET="0"
NORMAL_CROPS_PER_ROW="1"
CONTEXT_SECONDS="40.0"
CROP_TIME_SECONDS="10.0"
SEED="2026"
COMPRESSION="lzf"
SBATCH_TIME="03:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="48G"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e128_eval_h5.sh [options]

Build an E123/E126-compatible H5 containing ONC common validation/test rows for
production-style E128 SSL binary-gate scoring. This is a build-only CPU job; it
does not submit SSAMBA training.

Options:
  --manifest-csv PATH         ONC common-row manifest.
  --repo-root PATH            Whale-call repo root on Nibi.
  --dataset-root PATH         Multiband MAT extraction root.
  --output-root PATH          Default: $weekend_root/datasets/e128_onc_eval_h5_$stamp
  --output-h5 PATH            Default: $output_root/e128_onc_common_eval_low_$stamp.h5
  --output-summary PATH       Default: $output_h5 with .summary.json suffix
  --weekend-root PATH         Default: /scratch/.../multispecies_weekend_20260502
  --python-bin NAME           Default: python3
  --dependency SPEC           Optional Slurm dependency for the build job
  --band BAND                 Default: low
  --band-crop-shape FxT       Default: 391x50
  --output-shape FxT          Default: 512x512
  --splits CSV                Default: val,test
  --non-target-mode MODE      skip or normal. Default: normal
  --ambiguous-mode MODE       skip, first, or semicolon. Default: skip
  --max-normal N              Default: 0 (all normal/background rows)
  --max-per-target N          Default: 0 (all target rows)
  --normal-crops-per-row N    Default: 1
  --context-seconds X         Default: 40.0
  --crop-time-seconds X       Default: 10.0
  --seed N                    Default: 2026
  --compression NAME          Default: lzf
  --time HH:MM:SS             Default: 03:00:00
  --cpus-per-task N           Default: 4
  --mem MEM                   Default: 48G
  --dry-run                   Write script and print sbatch command only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-csv) MANIFEST_CSV="$2"; MANIFEST_SET="true"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; REPO_SET="true"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --output-h5) OUTPUT_H5="$2"; shift 2 ;;
    --output-summary) OUTPUT_SUMMARY="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --band) BAND="$2"; shift 2 ;;
    --band-crop-shape) BAND_CROP_SHAPE="$2"; shift 2 ;;
    --output-shape) OUTPUT_SHAPE="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --non-target-mode) NON_TARGET_MODE="$2"; shift 2 ;;
    --ambiguous-mode) AMBIGUOUS_MODE="$2"; shift 2 ;;
    --max-normal) MAX_NORMAL="$2"; shift 2 ;;
    --max-per-target) MAX_PER_TARGET="$2"; shift 2 ;;
    --normal-crops-per-row) NORMAL_CROPS_PER_ROW="$2"; shift 2 ;;
    --context-seconds) CONTEXT_SECONDS="$2"; shift 2 ;;
    --crop-time-seconds) CROP_TIME_SECONDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --compression) COMPRESSION="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$REPO_SET" != "true" ]]; then
  REPO_ON_NIBI="$WEEKEND_ROOT/$REPO_REL"
fi
if [[ "$MANIFEST_SET" != "true" ]]; then
  MANIFEST_CSV="$WEEKEND_ROOT/$MANIFEST_REL"
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$WEEKEND_ROOT/datasets/e128_onc_eval_h5_${STAMP}"
fi
if [[ -z "$OUTPUT_H5" ]]; then
  OUTPUT_H5="$OUTPUT_ROOT/e128_onc_common_eval_${BAND}_${STAMP}.h5"
fi
if [[ -z "$OUTPUT_SUMMARY" ]]; then
  OUTPUT_SUMMARY="${OUTPUT_H5%.h5}.summary.json"
fi

if [[ ! -d "$REPO_ON_NIBI" ]]; then
  echo "Missing repo root: $REPO_ON_NIBI" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "true" && ! -f "$MANIFEST_CSV" ]]; then
  echo "Missing manifest CSV: $MANIFEST_CSV" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs" "$(dirname "$OUTPUT_H5")" "$(dirname "$OUTPUT_SUMMARY")"
JOB_SCRIPT="$OUTPUT_ROOT/logs/E128_eval_h5.sbatch"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=E128evalH5
#SBATCH --output=$OUTPUT_ROOT/logs/e128_eval_h5_%j.out
#SBATCH --error=$OUTPUT_ROOT/logs/e128_eval_h5_%j.err
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail
REPO="$REPO_ON_NIBI"
MANIFEST="$MANIFEST_CSV"
DATASET_ROOT="$DATASET_ROOT"
OUT_H5="$OUTPUT_H5"
OUT_SUMMARY="$OUTPUT_SUMMARY"

cd "\$REPO"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"
mkdir -p "\$(dirname "\$OUT_H5")" "\$(dirname "\$OUT_SUMMARY")"

$PYTHON_BIN -u scripts/data/multilabel/build_e123_ssl_h5_dataset.py \\
  --manifest-csv "\$MANIFEST" \\
  --dataset-root "\$DATASET_ROOT" \\
  --output-h5 "\$OUT_H5" \\
  --output-summary "\$OUT_SUMMARY" \\
  --band "$BAND" \\
  --band-crop-shape "$BAND_CROP_SHAPE" \\
  --output-shape "$OUTPUT_SHAPE" \\
  --splits "$SPLITS" \\
  --non-target-mode "$NON_TARGET_MODE" \\
  --ambiguous-mode "$AMBIGUOUS_MODE" \\
  --max-normal "$MAX_NORMAL" \\
  --max-per-target "$MAX_PER_TARGET" \\
  --normal-crops-per-row "$NORMAL_CROPS_PER_ROW" \\
  --context-seconds "$CONTEXT_SECONDS" \\
  --crop-time-seconds "$CROP_TIME_SECONDS" \\
  --seed "$SEED" \\
  --compression "$COMPRESSION"
EOF

cmd=(sbatch --parsable)
if [[ -n "$DEPENDENCY" ]]; then
  cmd+=(--dependency="$DEPENDENCY")
fi
cmd+=("$JOB_SCRIPT")

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY RUN: ${cmd[*]}"
  job_id=""
else
  job_id="$("${cmd[@]}")"
  echo "Submitted E128 eval H5 build job: $job_id"
fi

cat <<EOF
E128 ONC evaluation H5 build prepared.
Output root: $OUTPUT_ROOT
Output H5: $OUTPUT_H5
Output summary: $OUTPUT_SUMMARY
Job script: $JOB_SCRIPT
Job ID: $job_id
Manifest: $MANIFEST_CSV
Splits: $SPLITS
Non-target mode: $NON_TARGET_MODE
EOF
