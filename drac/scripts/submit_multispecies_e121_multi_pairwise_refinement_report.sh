#!/bin/bash
# Submit the E121 multi-pairwise refinement report as a small CPU Slurm job.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
OUTPUT_DIR=""
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E121multiPair"
SBATCH_TIME="01:00:00"
SBATCH_CPUS="2"
SBATCH_MEM="16G"
BASE_DECISION_MODE="calibrated"
THRESHOLD_GRID="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"
MARGIN_GRID="-0.25,0.0,0.25"
BIAS_GRID="-0.30,-0.15,0.0,0.15,0.30"
DEPENDENCY=""
DRY_RUN="false"
BASE_RUN_DIRS=()
BASE_RUN_GLOBS=()
PAIRWISE_RUN_DIRS=()
PAIRWISE_RUN_GLOBS=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e121_multi_pairwise_refinement_report.sh [options]

Submit E121: evaluate multiple pairwise specialists as one conservative
refinement layer on top of one or more multiclass base runs.

Required:
  --base-run-dir PATH          Base run directory; may be repeated
  --base-run-glob GLOB         Glob for base run directories; may be repeated
  --pairwise-run-dir PATH      Pairwise specialist run directory; may be repeated
  --pairwise-run-glob GLOB     Glob for pairwise specialist run directories; may be repeated

Options:
  --weekend-root PATH          Default: /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502
  --repo-root PATH             Default: $weekend_root/repo_e24_expert_hparam_68be99f
  --output-dir PATH            Default: $weekend_root/pipeline_runs/e121_multi_pairwise_refinement_$stamp
  --stamp STAMP                Default: current UTC stamp
  --run-name NAME              Default: E121multiPair
  --dependency SPEC            Passed to sbatch, e.g. afterany:15956168
  --base-decision-mode MODE    existing or calibrated. Default: calibrated
  --threshold-grid CSV         Base calibrated threshold grid
  --margin-grid CSV            Base calibrated margin grid. Use --margin-grid=-0.25,0.0,0.25 for negatives
  --bias-grid CSV              Base calibrated species-bias grid. Use --bias-grid=-0.30,0.0,0.30 for negatives
  --time HH:MM:SS              Default: 01:00:00
  --cpus-per-task N            Default: 2
  --mem MEM                    Default: 16G
  --dry-run                    Write sbatch script but do not submit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --base-run-dir) BASE_RUN_DIRS+=("$2"); shift 2 ;;
    --base-run-glob) BASE_RUN_GLOBS+=("$2"); shift 2 ;;
    --pairwise-run-dir) PAIRWISE_RUN_DIRS+=("$2"); shift 2 ;;
    --pairwise-run-glob) PAIRWISE_RUN_GLOBS+=("$2"); shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --base-decision-mode) BASE_DECISION_MODE="$2"; shift 2 ;;
    --threshold-grid|--base-calibration-threshold-grid) THRESHOLD_GRID="$2"; shift 2 ;;
    --margin-grid|--base-calibration-margin-grid)
      MARGIN_GRID="$2"; shift 2 ;;
    --margin-grid=*|--base-calibration-margin-grid=*)
      MARGIN_GRID="${1#*=}"; shift ;;
    --bias-grid|--base-calibration-bias-grid)
      BIAS_GRID="$2"; shift 2 ;;
    --bias-grid=*|--base-calibration-bias-grid=*)
      BIAS_GRID="${1#*=}"; shift ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

for base_run_glob in "${BASE_RUN_GLOBS[@]}"; do
  mapfile -t matches < <(compgen -G "$base_run_glob" | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No base runs matched glob: $base_run_glob" >&2
    exit 1
  fi
  BASE_RUN_DIRS+=("${matches[@]}")
done
for pairwise_run_glob in "${PAIRWISE_RUN_GLOBS[@]}"; do
  mapfile -t matches < <(compgen -G "$pairwise_run_glob" | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No pairwise runs matched glob: $pairwise_run_glob" >&2
    exit 1
  fi
  PAIRWISE_RUN_DIRS+=("${matches[@]}")
done
if [[ "${#BASE_RUN_DIRS[@]}" -eq 0 ]]; then
  echo "Provide at least one --base-run-dir or --base-run-glob" >&2
  usage >&2
  exit 1
fi
if [[ "${#PAIRWISE_RUN_DIRS[@]}" -eq 0 ]]; then
  echo "Provide at least one --pairwise-run-dir or --pairwise-run-glob" >&2
  usage >&2
  exit 1
fi
if [[ "$BASE_DECISION_MODE" != "existing" && "$BASE_DECISION_MODE" != "calibrated" ]]; then
  echo "--base-decision-mode must be existing or calibrated" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$WEEKEND_ROOT/pipeline_runs/e121_multi_pairwise_refinement_${STAMP}"
fi
LOG_DIR="$OUTPUT_DIR/logs"
JOB_SCRIPT="$LOG_DIR/e121_multi_pairwise_refinement_${STAMP}.sbatch"
mkdir -p "$LOG_DIR"

base_args=""
for base_run_dir in "${BASE_RUN_DIRS[@]}"; do
  base_args+=" --base-run-dir \"$base_run_dir\""
done
pairwise_args=""
for pairwise_run_dir in "${PAIRWISE_RUN_DIRS[@]}"; do
  pairwise_args+=" --pairwise-run-dir \"$pairwise_run_dir\""
done

echo "Base run dirs:"
printf '  %s\n' "${BASE_RUN_DIRS[@]}"
echo "Pairwise run dirs:"
printf '  %s\n' "${PAIRWISE_RUN_DIRS[@]}"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail

echo "Started E121 multi-pairwise refinement report at \$(date -Is)"
echo "Host: \$(hostname)"

WEEKEND="$WEEKEND_ROOT"
REPO="$REPO_ON_NIBI"
OUTPUT_DIR="$OUTPUT_DIR"

cd "\$REPO"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
mkdir -p "\$OUTPUT_DIR"

echo "Repo: \$REPO"
git rev-parse HEAD || true
echo "Output dir: \$OUTPUT_DIR"
timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true

python scripts/analysis/e121_multi_pairwise_refinement_report.py \\
  --name E121_multi_pairwise_refinement \\
  --base-decision-mode "$BASE_DECISION_MODE" \\
  --base-calibration-threshold-grid "$THRESHOLD_GRID" \\
  --base-calibration-margin-grid="$MARGIN_GRID" \\
  --base-calibration-bias-grid="$BIAS_GRID" \\
  --output-dir "\$OUTPUT_DIR" \\
  $base_args \\
  $pairwise_args

echo "Completed E121 multi-pairwise refinement report at \$(date -Is)"
find "\$OUTPUT_DIR" -maxdepth 2 -type f -printf '%p\\n' | sort
EOF

echo "Wrote $JOB_SCRIPT"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run; not submitting."
  exit 0
fi

submit_args=()
if [[ -n "$DEPENDENCY" ]]; then
  submit_args+=(--dependency="$DEPENDENCY")
fi
job_id="$(sbatch --parsable "${submit_args[@]}" "$JOB_SCRIPT")"
echo "Submitted E121 report job: $job_id"
echo "Output dir: $OUTPUT_DIR"
