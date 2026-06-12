#!/bin/bash
# Submit a CPU audit job for an E123/E126 SSAMBA H5 dataset.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
INPUT_H5=""
BUILDER_SUMMARY_JSON=""
OUTPUT_DIR=""
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="python3"
SBATCH_TIME="00:30:00"
SBATCH_CPUS="1"
SBATCH_MEM="8G"
DEPENDENCY=""
LEDGER_PATH=""
LEDGER_ENTRY_ID=""
MIN_NORMAL_ROWS="10000"
MIN_NORMAL_TRAIN_ROWS="10000"
MIN_NORMAL_MONTHS="12"
MIN_NORMAL_TRAIN_MONTHS="12"
ALLOW_MISSING_H5="false"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e126_ssl_h5_audit.sh --input-h5 PATH [options]

Submit a small CPU job to audit an E123/E126 SSAMBA H5 dataset for normal
spectrogram coverage, split balance, label counts, and month spread.

Options:
  --input-h5 PATH             Required H5 dataset path
  --builder-summary-json PATH Optional summary JSON from H5 builder
  --output-dir PATH           Default: $weekend_root/pipeline_runs/e126_ssl_h5_audit_$stamp
  --weekend-root PATH         Default: /scratch/.../multispecies_weekend_20260502
  --repo-root PATH            Default: $weekend_root/repo_e24_expert_hparam_68be99f
  --python-bin NAME           Default: python3
  --time HH:MM:SS             Default: 00:30:00
  --cpus-per-task N           Default: 1
  --mem MEM                   Default: 8G
  --dependency SPEC           Passed to sbatch, e.g. afterok:123
  --allow-missing-h5          Allow queueing before dependency-created H5 exists
  --min-normal-rows N         Default: 10000
  --min-normal-train-rows N   Default: 10000
  --min-normal-months N       Default: 12
  --min-normal-train-months N Default: 12
  --ledger-path PATH          Optional living results ledger to append/update
  --ledger-entry-id ID        Optional stable ledger block id
  --dry-run                   Write script but do not submit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-h5) INPUT_H5="$2"; shift 2 ;;
    --builder-summary-json) BUILDER_SUMMARY_JSON="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --allow-missing-h5) ALLOW_MISSING_H5="true"; shift ;;
    --min-normal-rows) MIN_NORMAL_ROWS="$2"; shift 2 ;;
    --min-normal-train-rows) MIN_NORMAL_TRAIN_ROWS="$2"; shift 2 ;;
    --min-normal-months) MIN_NORMAL_MONTHS="$2"; shift 2 ;;
    --min-normal-train-months) MIN_NORMAL_TRAIN_MONTHS="$2"; shift 2 ;;
    --ledger-path) LEDGER_PATH="$2"; shift 2 ;;
    --ledger-entry-id) LEDGER_ENTRY_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$INPUT_H5" ]]; then
  echo "Missing required --input-h5" >&2
  usage
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$WEEKEND_ROOT/pipeline_runs/e126_ssl_h5_audit_${STAMP}"
fi
if [[ ! -d "$REPO_ON_NIBI" ]]; then
  echo "Missing repo root: $REPO_ON_NIBI" >&2
  exit 1
fi
if [[ "$ALLOW_MISSING_H5" != "true" && ! -f "$INPUT_H5" ]]; then
  echo "Missing H5 dataset: $INPUT_H5" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
JOB_SCRIPT="$OUTPUT_DIR/logs/E126sslH5audit.sbatch"

args=(
  --input-h5 "$INPUT_H5"
  --output-dir "$OUTPUT_DIR"
  --min-normal-rows "$MIN_NORMAL_ROWS"
  --min-normal-train-rows "$MIN_NORMAL_TRAIN_ROWS"
  --min-normal-months "$MIN_NORMAL_MONTHS"
  --min-normal-train-months "$MIN_NORMAL_TRAIN_MONTHS"
)
if [[ -n "$BUILDER_SUMMARY_JSON" ]]; then
  args+=(--builder-summary-json "$BUILDER_SUMMARY_JSON")
fi
if [[ -n "$LEDGER_PATH" ]]; then
  args+=(--ledger-path "$LEDGER_PATH")
fi
if [[ -n "$LEDGER_ENTRY_ID" ]]; then
  args+=(--ledger-entry-id "$LEDGER_ENTRY_ID")
fi
printf '%s\0' "${args[@]}" > "$OUTPUT_DIR/logs/audit_args.nul"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=E126h5audit
#SBATCH --output=$OUTPUT_DIR/logs/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail
cd "$REPO_ON_NIBI"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$REPO_ON_NIBI:\${PYTHONPATH:-}"

AUDIT_ARGS=()
mapfile -d '' -t AUDIT_ARGS < "$OUTPUT_DIR/logs/audit_args.nul"

$PYTHON_BIN -u scripts/analysis/e126_ssl_h5_audit_report.py "\${AUDIT_ARGS[@]}"
EOF

echo "E126 H5 audit output dir: $OUTPUT_DIR"
echo "E126 H5 audit job script: $JOB_SCRIPT"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run; not submitting."
else
  submit_args=()
  if [[ -n "$DEPENDENCY" ]]; then
    submit_args+=(--dependency="$DEPENDENCY")
  fi
  job_id="$(sbatch --parsable "${submit_args[@]}" "$JOB_SCRIPT")"
  echo "Submitted E126 H5 audit job: $job_id"
fi
