#!/bin/bash
# Submit a small CPU job to build the E124 production-candidate leaderboard.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
PYTHON_BIN="python3"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR=""
SBATCH_TIME="00:30:00"
SBATCH_CPUS="1"
SBATCH_MEM="4G"
DEPENDENCY=""
DRY_RUN="false"
SUMMARY_JSONS=()
SUMMARY_GLOBS=()
CANDIDATES=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e124_candidate_leaderboard.sh [options]

Build a compact leaderboard across completed E119/E121/E122/E26 reports.
This is a CPU-only report job; it does not run training or inference.

Options:
  --summary-json PATH       Summary JSON; may be repeated
  --summary-glob GLOB       Summary JSON glob; may be repeated
  --candidate NAME=PATH     Named summary JSON; may be repeated
  --weekend-root PATH       Default: /scratch/.../multispecies_weekend_20260502
  --repo-root PATH          Default: $weekend_root/repo_e24_expert_hparam_68be99f
  --python-bin NAME         Default: python3
  --output-dir PATH         Default: $weekend_root/pipeline_runs/e124_candidate_leaderboard_$stamp
  --stamp STAMP             Default: current UTC stamp
  --time HH:MM:SS           Default: 00:30:00
  --cpus-per-task N         Default: 1
  --mem MEM                 Default: 4G
  --dependency SPEC         Passed to sbatch, e.g. afterany:123:124
  --dry-run                 Write the sbatch script but do not submit it
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --summary-json) SUMMARY_JSONS+=("$2"); shift 2 ;;
    --summary-glob) SUMMARY_GLOBS+=("$2"); shift 2 ;;
    --candidate) CANDIDATES+=("$2"); shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$WEEKEND_ROOT/pipeline_runs/e124_candidate_leaderboard_${STAMP}"
fi

if [[ ! -d "$REPO_ON_NIBI" ]]; then
  echo "Missing repo root: $REPO_ON_NIBI" >&2
  exit 1
fi

if [[ "${#SUMMARY_JSONS[@]}" -eq 0 && "${#SUMMARY_GLOBS[@]}" -eq 0 && "${#CANDIDATES[@]}" -eq 0 ]]; then
  SUMMARY_GLOBS+=(
    "$WEEKEND_ROOT/pipeline_runs/e119*/e119_summary.json"
    "$WEEKEND_ROOT/pipeline_runs/e119*/*/e119_summary.json"
    "$WEEKEND_ROOT/pipeline_runs/e121*/e121_summary.json"
    "$WEEKEND_ROOT/pipeline_runs/e121*/*/e121_summary.json"
    "$WEEKEND_ROOT/pipeline_runs/e122*/e122_summary.json"
    "$WEEKEND_ROOT/pipeline_runs/e122*/*/e122_summary.json"
    "$WEEKEND_ROOT/pipeline_runs/e26*/diagnostic_summary.json"
    "$WEEKEND_ROOT/pipeline_runs/e26*/*/diagnostic_summary.json"
  )
fi

mkdir -p "$OUTPUT_DIR/logs"
JOB_SCRIPT="$OUTPUT_DIR/logs/E124leaderboard.sbatch"

args=()
for value in "${SUMMARY_JSONS[@]}"; do
  args+=(--summary-json "$value")
done
for value in "${SUMMARY_GLOBS[@]}"; do
  args+=(--summary-glob "$value")
done
for value in "${CANDIDATES[@]}"; do
  args+=(--candidate "$value")
done

printf '%s\0' "${args[@]}" > "$OUTPUT_DIR/logs/summary_args.nul"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=E124leader
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

SUMMARY_ARGS=()
if [[ -f "$OUTPUT_DIR/logs/summary_args.nul" ]]; then
  mapfile -d '' -t SUMMARY_ARGS < "$OUTPUT_DIR/logs/summary_args.nul"
fi

$PYTHON_BIN scripts/analysis/e124_compare_production_candidates.py \\
  "\${SUMMARY_ARGS[@]}" \\
  --output-dir "$OUTPUT_DIR"
EOF

echo "E124 output dir: $OUTPUT_DIR"
echo "E124 job script: $JOB_SCRIPT"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run; not submitting."
else
  submit_args=()
  if [[ -n "$DEPENDENCY" ]]; then
    submit_args+=(--dependency="$DEPENDENCY")
  fi
  job_id="$(sbatch --parsable "${submit_args[@]}" "$JOB_SCRIPT")"
  echo "Submitted E124 leaderboard job: $job_id"
fi
