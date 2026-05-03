#!/bin/bash
# Submit the bounded weekend DCLDE Oo-repair GPU experiment on Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
PROJECT_EXP_ROOT="$FINAL2025_ROOT/multispecies_calltype_experiments"
WEEKEND_ROOT="${SCRATCH:-/scratch/$USER}/whale-call-analysis/multispecies_weekend_20260502"
ONC_RUN="$PROJECT_EXP_ROOT/prep_runs/multispecies_prep_balanced100_availaudio_20260430T120219Z"
ONC_SOURCE_NAME="onc_bal100"
BIODCASE_TRAIN50_RUN="$PROJECT_EXP_ROOT/prep_runs/biodcase_task2_prep_train50_resume_20260501T182239Z"
DCLDE_RUN="$WEEKEND_ROOT/prep_runs/dclde_orca_cap200_prep_20260502T202327Z"
BASE_CHECKPOINT="$FINAL2025_ROOT/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

EXPERIMENT="E06_onc_biod_dclde_oo_repair_species_call"
MODE="species_call"
MODEL="resnet18"
EPOCHS="20"
BATCH_SIZE="32"
NUM_WORKERS="8"
LR="1e-4"
WEIGHT_DECAY="1e-4"
CROP_SIZE="96"
CROP_TIME_SECONDS="10"
FREQ_MIN_HZ="5"
FREQ_MAX_HZ="100"
CENTER_BIAS_SIGMA_FRAC="0.25"
POSITIVE_CROP_MODE="edge_mix"
SEED="20260503"
USE_POS_WEIGHT="true"
USE_WANDB="true"
WANDB_PROJECT="whale-multispecies-calltype"
WANDB_ENTITY=""
WANDB_GROUP="weekend-20260502-dclde"
SBATCH_GRES="gpu:h100:1"
SBATCH_PARTITION=""
SBATCH_TIME="04:00:00"
SBATCH_CPUS="8"
SBATCH_MEM="56G"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_weekend_dclde_oo_repair.sh [options]

Submits one bounded GPU job:
  E06_onc_biod_dclde_oo_repair_species_call

Options:
  --weekend-root PATH
  --onc-run PATH
  --onc-source-name NAME          Default: onc_bal100
  --biodcase-train50-run PATH
  --dclde-run PATH
  --base-checkpoint PATH
  --experiment NAME
  --mode species|species_call     Default: species_call
  --epochs N                     Default: 20
  --batch-size N                 Default: 32
  --lr X                         Default: 1e-4
  --no-pos-weight
  --no-wandb
  --wandb-project NAME
  --wandb-entity NAME
  --wandb-group NAME
  --gres SPEC                    Default: gpu:h100:1
  --partition NAME
  --time HH:MM:SS                Default: 04:00:00
  --cpus-per-task N              Default: 8
  --mem SIZE                     Default: 56G
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --onc-run) ONC_RUN="$2"; shift 2 ;;
    --onc-source-name) ONC_SOURCE_NAME="$2"; shift 2 ;;
    --biodcase-train50-run) BIODCASE_TRAIN50_RUN="$2"; shift 2 ;;
    --dclde-run) DCLDE_RUN="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --no-pos-weight) USE_POS_WEIGHT="false"; shift ;;
    --no-wandb) USE_WANDB="false"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --partition) SBATCH_PARTITION="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$MODE" != "species" && "$MODE" != "species_call" ]]; then
  echo "--mode must be species or species_call, got $MODE" >&2
  exit 1
fi

first_existing() {
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

ONC_SPLIT="$(first_existing \
  "$ONC_RUN/splits_label_balanced_v2/split_manifest.csv" \
  "$ONC_RUN/splits/split_manifest.csv")" || {
  echo "Could not find ONC split manifest under $ONC_RUN" >&2
  exit 1
}
BIOD_SPLIT="$(first_existing "$BIODCASE_TRAIN50_RUN/splits/split_manifest.csv")" || {
  echo "Could not find BioDCASE train50 split manifest under $BIODCASE_TRAIN50_RUN" >&2
  exit 1
}
DCLDE_SPLIT="$(first_existing "$DCLDE_RUN/splits/split_manifest.csv")" || {
  echo "Could not find DCLDE split manifest under $DCLDE_RUN" >&2
  exit 1
}
[[ -e "$BASE_CHECKPOINT" ]] || { echo "Missing base checkpoint: $BASE_CHECKPOINT" >&2; exit 1; }

mkdir -p "$WEEKEND_ROOT"/{manifests,runs,logs}
PLAN_TSV="$WEEKEND_ROOT/weekend_dclde_oo_repair_plan.tsv"
SUBMITTED_TSV="$WEEKEND_ROOT/weekend_dclde_oo_repair_submitted.tsv"
echo -e "experiment\tmode\tmanifest\tvocab\trun_dir\twandb_group" > "$PLAN_TSV"
echo -e "job_id\texperiment\trun_dir\tjob_script" > "$SUBMITTED_TSV"

echo "Queue check before submission"
squeue -u "$USER" || true
sacct -u "$USER" --starttime now-7days || true

cd "$REPO_ROOT"
source .venv/bin/activate

MANIFEST_DIR="$WEEKEND_ROOT/manifests/$EXPERIMENT"
mkdir -p "$MANIFEST_DIR"
python -u scripts/data/multilabel/standardize_multilabel_manifest.py \
  --output-dir "$MANIFEST_DIR" \
  --mode "$MODE" \
  --vocab-min-count 1 \
  --dedupe-key mat_path,split \
  --input "$ONC_SOURCE_NAME|$ONC_SPLIT|$ONC_RUN" \
  --input "biodcase_train50|$BIOD_SPLIT|$BIODCASE_TRAIN50_RUN" \
  --input "dclde_orca_cap200|$DCLDE_SPLIT|$DCLDE_RUN" \
  | tee "$MANIFEST_DIR/standardize_stdout.json" >&2

RUN_DIR="$WEEKEND_ROOT/runs/${EXPERIMENT}_$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$RUN_DIR/logs"
TRAIN_DIR="$RUN_DIR/train"
mkdir -p "$LOG_DIR" "$TRAIN_DIR"
JOB_SCRIPT="$LOG_DIR/${EXPERIMENT}.sbatch"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${EXPERIMENT}
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM
EOF
if [[ -n "$SBATCH_PARTITION" ]]; then
  echo "#SBATCH --partition=$SBATCH_PARTITION" >> "$JOB_SCRIPT"
fi
if [[ -n "$SBATCH_GRES" ]]; then
  echo "#SBATCH --gres=$SBATCH_GRES" >> "$JOB_SCRIPT"
fi

cat >> "$JOB_SCRIPT" <<EOF

set -euo pipefail
echo "Started at \$(date -Is)"
echo "Host: \$(hostname)"
cd "$REPO_ROOT"
source .venv/bin/activate
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/$USER/.cache}"
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export WANDB_DATA_DIR="$RUN_DIR/wandb_data"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_DIR" "\$WANDB_CACHE_DIR" "\$WANDB_DATA_DIR" "\$WANDB_CONFIG_DIR"

cat > "$RUN_DIR/run_metadata.json" <<META
{
  "experiment": "$EXPERIMENT",
  "mode": "$MODE",
  "run_dir": "$RUN_DIR",
  "manifest_dir": "$MANIFEST_DIR",
  "onc_run": "$ONC_RUN",
  "biodcase_train50_run": "$BIODCASE_TRAIN50_RUN",
  "dclde_run": "$DCLDE_RUN",
  "base_checkpoint": "$BASE_CHECKPOINT",
  "wandb_project": "$WANDB_PROJECT",
  "wandb_group": "$WANDB_GROUP",
  "dedupe_key": "mat_path,split",
  "gate": "ONC primary macro >0.6372 or close with Oo/Mn improvement, background FP <=0.50"
}
META

train_cmd=(
  python -u scripts/train/train_multilabel_resnet_smoke.py
  --manifest-csv "$MANIFEST_DIR/standardized_manifest.csv"
  --vocab-json "$MANIFEST_DIR/label_vocabulary.json"
  --exp-dir "$TRAIN_DIR"
  --model "$MODEL"
  --init-checkpoint "$BASE_CHECKPOINT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --crop-size "$CROP_SIZE"
  --crop-time-seconds "$CROP_TIME_SECONDS"
  --freq-min-hz "$FREQ_MIN_HZ"
  --freq-max-hz "$FREQ_MAX_HZ"
  --center-bias-sigma-frac "$CENTER_BIAS_SIGMA_FRAC"
  --positive-crop-mode "$POSITIVE_CROP_MODE"
  --device cuda
  --seed "$SEED"
  --max-example-images 96
  --wandb-project "$WANDB_PROJECT"
  --wandb-group "$WANDB_GROUP"
  --wandb-name "$EXPERIMENT"
  --wandb-tags "weekend-20260502,$EXPERIMENT,$MODE,dclde,oo-repair,multilabel,resnet,nibi"
)
if [[ "$USE_POS_WEIGHT" == "true" ]]; then
  train_cmd+=(--use-pos-weight)
fi
if [[ "$USE_WANDB" == "true" ]]; then
  train_cmd+=(--use-wandb)
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  train_cmd+=(--wandb-entity "$WANDB_ENTITY")
fi
printf '%q ' "\${train_cmd[@]}" > "$RUN_DIR/train_command.sh"
printf '\n' >> "$RUN_DIR/train_command.sh"
"\${train_cmd[@]}"

echo "Finished at \$(date -Is)"
EOF

chmod +x "$JOB_SCRIPT"
echo -e "${EXPERIMENT}\t${MODE}\t${MANIFEST_DIR}/standardized_manifest.csv\t${MANIFEST_DIR}/label_vocabulary.json\t${RUN_DIR}\t${WANDB_GROUP}" >> "$PLAN_TSV"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: $EXPERIMENT -> $JOB_SCRIPT"
  echo "Wrote plan: $PLAN_TSV"
  exit 0
fi

SBATCH_OUT="$(sbatch "$JOB_SCRIPT")"
echo "$SBATCH_OUT"
JOB_ID="$(echo "$SBATCH_OUT" | awk '{print $NF}')"
echo -e "${JOB_ID}\t${EXPERIMENT}\t${RUN_DIR}\t${JOB_SCRIPT}" >> "$SUBMITTED_TSV"
echo "Wrote plan: $PLAN_TSV"
echo "Wrote submissions: $SUBMITTED_TSV"
