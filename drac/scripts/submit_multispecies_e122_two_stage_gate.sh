#!/bin/bash
# Build and submit the E122 binary whale-call gate plus optional two-stage report.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
DATASET_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502/mat_archives/multiband40s_20260514T002301Z/extracted"
SOURCE_MANIFEST="$WEEKEND_ROOT/manifests/e100_onc_only_blocked_nov_validation_20260611T020900Z/E101_stage2_ONConly_blocked_nov20_25_30_val/standardized_manifest.csv"
FIN_INIT_CKPT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"
PYTHON_BIN="python3"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST_ROOT=""
RUN_DIR=""
REPORT_DIR=""
GATE_LABEL="task:whale_call"
POSITIVE_LABELS="species:Bp,species:Bm,species:Mn"
EPOCHS="35"
LR="0.0003"
DROPOUT="0.3"
BATCH_SIZE="32"
NUM_WORKERS="4"
STOP_AFTER_SECONDS="9000"
SBATCH_TIME="03:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="48G"
SBATCH_GRES="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
REPORT_TIME="01:00:00"
REPORT_MEM="16G"
DEPENDENCY=""
BASE_DECISION_MODE="calibrated"
SPECIES_STAGE_MODE="force_species_argmax"
DRY_RUN="false"
BASE_RUN_DIRS=()
BASE_RUN_GLOBS=()
SOURCE_KINDS=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e122_two_stage_gate.sh [options]

Build a binary whale-call vs background gate manifest, submit a small resumable
MIG training chain, and optionally submit the E122 two-stage report against one
or more multiclass species base runs.

Options:
  --base-run-dir PATH          Species classifier base run; may be repeated
  --base-run-glob GLOB         Glob for species classifier base runs; may be repeated
  --source-kind KIND           Keep only source kind in gate manifest; may be repeated
  --weekend-root PATH          Default: /scratch/.../multispecies_weekend_20260502
  --repo-root PATH             Default: $weekend_root/repo_e24_expert_hparam_68be99f
  --dataset-root PATH          Multiband MAT extracted dataset root
  --source-manifest PATH       Source standardized manifest for gate training
  --fin-init-checkpoint PATH   Initialization checkpoint
  --python-bin NAME            Default: python3
  --manifest-root PATH         Default: $weekend_root/manifests/e122_two_stage_gate_$stamp
  --run-dir PATH               Default: $weekend_root/runs/E122_whale_call_gate_ONConly_3band_lr3e4_$stamp
  --report-dir PATH            Default: $weekend_root/pipeline_runs/e122_two_stage_gate_$stamp
  --stamp STAMP                Default: current UTC stamp
  --dependency SPEC            Passed to initial sbatch, e.g. afterany:123
  --gate-label LABEL           Default: task:whale_call
  --positive-labels CSV        Default: species:Bp,species:Bm,species:Mn
  --species-stage-mode MODE    base_pred or force_species_argmax. Default: force_species_argmax
  --base-decision-mode MODE    existing or calibrated. Default: calibrated
  --epochs N                   Default: 35
  --lr LR                      Default: 0.0003
  --dropout P                  Default: 0.3
  --batch-size N               Default: 32
  --num-workers N              Default: 4
  --stop-after-seconds N       Default: 9000
  --time HH:MM:SS              Default: 03:00:00
  --cpus-per-task N            Default: 4
  --mem MEM                    Default: 48G
  --gres GRES                  Default: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
  --dry-run                    Write manifests/scripts but do not submit jobs
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-run-dir) BASE_RUN_DIRS+=("$2"); shift 2 ;;
    --base-run-glob) BASE_RUN_GLOBS+=("$2"); shift 2 ;;
    --source-kind) SOURCE_KINDS+=("$2"); shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --source-manifest) SOURCE_MANIFEST="$2"; shift 2 ;;
    --fin-init-checkpoint) FIN_INIT_CKPT="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --report-dir) REPORT_DIR="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --gate-label) GATE_LABEL="$2"; shift 2 ;;
    --positive-labels) POSITIVE_LABELS="$2"; shift 2 ;;
    --species-stage-mode) SPECIES_STAGE_MODE="$2"; shift 2 ;;
    --base-decision-mode) BASE_DECISION_MODE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --stop-after-seconds) STOP_AFTER_SECONDS="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
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
if [[ "$BASE_DECISION_MODE" != "existing" && "$BASE_DECISION_MODE" != "calibrated" ]]; then
  echo "--base-decision-mode must be existing or calibrated" >&2
  exit 1
fi
if [[ "$SPECIES_STAGE_MODE" != "base_pred" && "$SPECIES_STAGE_MODE" != "force_species_argmax" ]]; then
  echo "--species-stage-mode must be base_pred or force_species_argmax" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "Missing source manifest: $SOURCE_MANIFEST" >&2
  exit 1
fi
if [[ -z "$MANIFEST_ROOT" ]]; then
  MANIFEST_ROOT="$WEEKEND_ROOT/manifests/e122_two_stage_gate_${STAMP}"
fi
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$WEEKEND_ROOT/runs/E122_whale_call_gate_ONConly_3band_lr3e4_${STAMP}"
fi
if [[ -z "$REPORT_DIR" ]]; then
  REPORT_DIR="$WEEKEND_ROOT/pipeline_runs/e122_two_stage_gate_${STAMP}"
fi

VARIANT_DIR="$MANIFEST_ROOT/E122_whale_call_gate"
MANIFEST="$VARIANT_DIR/standardized_manifest.csv"
VOCAB="$VARIANT_DIR/label_vocabulary.json"
mkdir -p "$VARIANT_DIR" "$RUN_DIR/logs" "$RUN_DIR/train"

source_args=()
for source_kind in "${SOURCE_KINDS[@]}"; do
  source_args+=(--source-kind "$source_kind")
done

cd "$REPO_ON_NIBI"
"$PYTHON_BIN" scripts/data/multilabel/build_e122_two_stage_gate_manifest.py \
  --input-manifest "$SOURCE_MANIFEST" \
  --output-dir "$VARIANT_DIR" \
  --positive-labels "$POSITIVE_LABELS" \
  --gate-label "$GATE_LABEL" \
  "${source_args[@]}"

JOB_SCRIPT="$RUN_DIR/logs/E122gate.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=E122gate
#SBATCH --output=$RUN_DIR/logs/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM
#SBATCH --gres=$SBATCH_GRES

set -euo pipefail
REPO="$REPO_ON_NIBI"
RUN="$RUN_DIR"
DATA="$DATASET_ROOT"
MAN="$MANIFEST"
VOC="$VOCAB"
FINCKPT="$FIN_INIT_CKPT"
TARGET_EPOCHS="$EPOCHS"

cd "\$REPO"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"

SUMMARY="\$RUN/train/run_summary.json"
if $PYTHON_BIN - "\$SUMMARY" "\$TARGET_EPOCHS" <<'PYCHK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
target = int(sys.argv[2])
if not p.exists():
    raise SystemExit(1)
history = json.loads(p.read_text()).get("history") or []
raise SystemExit(0 if max((int(row.get("epoch", 0)) for row in history), default=0) >= target else 1)
PYCHK
then
  echo "E122 gate complete; skipping"
  exit 0
fi

EXTRA=()
if [[ -f "\$RUN/train/last.pt" ]]; then
  EXTRA=(--resume-checkpoint "\$RUN/train/last.pt")
else
  EXTRA=(--init-all-branches-checkpoint "\$FINCKPT")
fi

$PYTHON_BIN -u scripts/train/train_multiband_multilabel.py \\
  --manifest-csv "\$MAN" \\
  --vocab-json "\$VOC" \\
  --dataset-root "\$DATA" \\
  --exp-dir "\$RUN/train" \\
  --bands low,mid,high \\
  --band-crop-shapes low:391x50,mid:256x100,high:256x312 \\
  --encoder resnet18 \\
  --fusion gated \\
  --head-type shared \\
  --dropout "$DROPOUT" \\
  --epochs "$EPOCHS" \\
  --batch-size "$BATCH_SIZE" \\
  --num-workers "$NUM_WORKERS" \\
  --weight-decay 0.0001 \\
  --crop-time-seconds 10 \\
  --context-seconds 40 \\
  --center-bias-sigma-frac 0.25 \\
  --positive-crop-mode centered_gaussian \\
  --band-availability-mode all \\
  --class-band-mask-mode none \\
  --device cuda \\
  --seed 2026 \\
  --stop-after-seconds "$STOP_AFTER_SECONDS" \\
  --lr "$LR" \\
  --loss-mode balanced_bce \\
  "\${EXTRA[@]}"
EOF

echo "Wrote gate job: $JOB_SCRIPT"
initial_job=""
continuation_job=""
report_job=""
if [[ "$DRY_RUN" == "false" ]]; then
  submit_args=()
  if [[ -n "$DEPENDENCY" ]]; then
    submit_args+=(--dependency="$DEPENDENCY")
  fi
  initial_job="$(sbatch --parsable "${submit_args[@]}" "$JOB_SCRIPT")"
  continuation_job="$(sbatch --parsable --dependency=afterany:"$initial_job" "$JOB_SCRIPT")"
  echo "Submitted E122 gate initial=$initial_job continuation=$continuation_job"
else
  echo "Dry run; not submitting gate jobs."
fi

if [[ "${#BASE_RUN_DIRS[@]}" -gt 0 ]]; then
  mkdir -p "$REPORT_DIR/logs"
  report_base_args=""
  for base_run_dir in "${BASE_RUN_DIRS[@]}"; do
    report_base_args+=" --base-run-dir \"$base_run_dir\""
  done
  REPORT_SCRIPT="$REPORT_DIR/logs/E122report.sbatch"
  cat > "$REPORT_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=E122report
#SBATCH --output=$REPORT_DIR/logs/slurm-%j.out
#SBATCH --time=$REPORT_TIME
#SBATCH --cpus-per-task=2
#SBATCH --mem=$REPORT_MEM

set -euo pipefail
cd "$REPO_ON_NIBI"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$REPO_ON_NIBI:\${PYTHONPATH:-}"
mkdir -p "$REPORT_DIR"
python scripts/analysis/e122_two_stage_gate_report.py \\
  --name E122_two_stage_gate \\
  --gate-run-dir "$RUN_DIR" \\
  --gate-label "$GATE_LABEL" \\
  --species-stage-mode "$SPECIES_STAGE_MODE" \\
  --base-decision-mode "$BASE_DECISION_MODE" \\
  --output-dir "$REPORT_DIR" \\
  $report_base_args
EOF
  echo "Wrote report job: $REPORT_SCRIPT"
  if [[ "$DRY_RUN" == "false" ]]; then
    report_dep="$continuation_job"
    report_job="$(sbatch --parsable --dependency=afterany:"$report_dep" "$REPORT_SCRIPT")"
    echo "Submitted E122 report job: $report_job"
  fi
fi

SUBMITTED_TSV="$MANIFEST_ROOT/e122_two_stage_submitted.tsv"
echo -e "gate_initial_job_id\tgate_continuation_job_id\treport_job_id\trun_dir\tmanifest\tvocab\treport_dir\tjob_script" > "$SUBMITTED_TSV"
echo -e "${initial_job}\t${continuation_job}\t${report_job}\t${RUN_DIR}\t${MANIFEST}\t${VOCAB}\t${REPORT_DIR}\t${JOB_SCRIPT}" >> "$SUBMITTED_TSV"
echo "E122 plan: $SUBMITTED_TSV"
