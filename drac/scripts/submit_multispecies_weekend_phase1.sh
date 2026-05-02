#!/bin/bash
# Submit bounded weekend Phase 1 multi-species/call-type GPU experiments on Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
PROJECT_EXP_ROOT="$FINAL2025_ROOT/multispecies_calltype_experiments"
WEEKEND_ROOT="${SCRATCH:-/scratch/$USER}/whale-call-analysis/multispecies_weekend_20260502"
ONC_BAL100_RUN="$PROJECT_EXP_ROOT/prep_runs/multispecies_prep_balanced100_availaudio_20260430T120219Z"
BIODCASE_TRAIN50_RUN="$PROJECT_EXP_ROOT/prep_runs/biodcase_task2_prep_train50_resume_20260501T182239Z"
BASE_CHECKPOINT="$FINAL2025_ROOT/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

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
SEED="20260502"
USE_POS_WEIGHT="true"
USE_WANDB="true"
WANDB_PROJECT="whale-multispecies-calltype"
WANDB_ENTITY=""
SBATCH_GRES="gpu:h100:1"
SBATCH_PARTITION=""
SBATCH_TIME="04:00:00"
SBATCH_CPUS="8"
SBATCH_MEM="48G"
DRY_RUN="false"
ONLY_EXPERIMENTS=""

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_weekend_phase1.sh [options]

Submits these bounded GPU jobs:
  E01_onc_bal100_control
  E02_biodcase_train50_only
  E03_onc_biod_train50_species
  E04_onc_biod_train50_species_call

Options:
  --weekend-root PATH
  --onc-bal100-run PATH
  --biodcase-train50-run PATH
  --base-checkpoint PATH
  --only CSV                     Example: E01,E03
  --epochs N                     Default: 20
  --batch-size N                 Default: 32
  --lr X                         Default: 1e-4
  --no-pos-weight
  --no-wandb
  --wandb-project NAME
  --wandb-entity NAME
  --gres SPEC                    Default: gpu:h100:1
  --partition NAME
  --time HH:MM:SS                Default: 04:00:00
  --cpus-per-task N              Default: 8
  --mem SIZE                     Default: 48G
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --onc-bal100-run) ONC_BAL100_RUN="$2"; shift 2 ;;
    --biodcase-train50-run) BIODCASE_TRAIN50_RUN="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --only) ONLY_EXPERIMENTS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --no-pos-weight) USE_POS_WEIGHT="false"; shift ;;
    --no-wandb) USE_WANDB="false"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
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

first_existing() {
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

should_run() {
  local experiment="$1"
  [[ -z "$ONLY_EXPERIMENTS" ]] && return 0
  IFS=',' read -ra requested <<< "$ONLY_EXPERIMENTS"
  for item in "${requested[@]}"; do
    [[ "$experiment" == "$item" ]] && return 0
  done
  return 1
}

ONC_SPLIT="$(first_existing \
  "$ONC_BAL100_RUN/splits_label_balanced_v2/split_manifest.csv" \
  "$ONC_BAL100_RUN/splits/split_manifest.csv")" || {
  echo "Could not find ONC balanced100 split manifest under $ONC_BAL100_RUN" >&2
  exit 1
}
BIOD_SPLIT="$(first_existing "$BIODCASE_TRAIN50_RUN/splits/split_manifest.csv")" || {
  echo "Could not find BioDCASE train50 split manifest under $BIODCASE_TRAIN50_RUN" >&2
  exit 1
}
[[ -e "$BASE_CHECKPOINT" ]] || { echo "Missing base checkpoint: $BASE_CHECKPOINT" >&2; exit 1; }

mkdir -p "$WEEKEND_ROOT"/{manifests,runs,logs}
PLAN_TSV="$WEEKEND_ROOT/weekend_phase1_plan.tsv"
SUBMITTED_TSV="$WEEKEND_ROOT/weekend_phase1_submitted.tsv"
echo -e "experiment\tmode\tmanifest\tvocab\trun_dir\twandb_group" > "$PLAN_TSV"
echo -e "job_id\texperiment\trun_dir\tjob_script" > "$SUBMITTED_TSV"

echo "Queue check before submissions"
squeue -u "$USER" || true
sacct -u "$USER" --starttime now-7days || true

build_manifest() {
  local experiment="$1"
  local mode="$2"
  shift 2
  local out_dir="$WEEKEND_ROOT/manifests/$experiment"
  mkdir -p "$out_dir"
  local cmd=(
    python -u scripts/data/multilabel/standardize_multilabel_manifest.py
    --output-dir "$out_dir"
    --mode "$mode"
    --vocab-min-count 1
  )
  while [[ $# -gt 0 ]]; do
    cmd+=(--input "$1")
    shift
  done
  "${cmd[@]}" | tee "$out_dir/standardize_stdout.json" >&2
  echo "$out_dir"
}

write_and_submit_job() {
  local experiment="$1"
  local mode="$2"
  local manifest_dir="$3"
  local wandb_group="$4"
  local run_dir="$WEEKEND_ROOT/runs/${experiment}_$(date -u +%Y%m%dT%H%M%SZ)"
  local log_dir="$run_dir/logs"
  local train_dir="$run_dir/train"
  mkdir -p "$log_dir" "$train_dir"
  local job_script="$log_dir/${experiment}.sbatch"

  cat > "$job_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${experiment}
#SBATCH --output=$log_dir/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM
EOF
  if [[ -n "$SBATCH_PARTITION" ]]; then
    echo "#SBATCH --partition=$SBATCH_PARTITION" >> "$job_script"
  fi
  if [[ -n "$SBATCH_GRES" ]]; then
    echo "#SBATCH --gres=$SBATCH_GRES" >> "$job_script"
  fi

  cat >> "$job_script" <<EOF

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
export WANDB_DIR="$run_dir/wandb"
export WANDB_CACHE_DIR="$run_dir/wandb_cache"
export WANDB_DATA_DIR="$run_dir/wandb_data"
export WANDB_CONFIG_DIR="$run_dir/wandb_config"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_DIR" "\$WANDB_CACHE_DIR" "\$WANDB_DATA_DIR" "\$WANDB_CONFIG_DIR"

train_cmd=(
  python -u scripts/train/train_multilabel_resnet_smoke.py
  --manifest-csv "$manifest_dir/standardized_manifest.csv"
  --vocab-json "$manifest_dir/label_vocabulary.json"
  --exp-dir "$train_dir"
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
  --wandb-group "$wandb_group"
  --wandb-name "$experiment"
  --wandb-tags "weekend-20260502,$experiment,$mode,multilabel,resnet,nibi"
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
"\${train_cmd[@]}"

cat > "$run_dir/run_metadata.json" <<META
{
  "experiment": "$experiment",
  "mode": "$mode",
  "run_dir": "$run_dir",
  "manifest_dir": "$manifest_dir",
  "base_checkpoint": "$BASE_CHECKPOINT",
  "wandb_project": "$WANDB_PROJECT",
  "wandb_group": "$wandb_group"
}
META
echo "Finished at \$(date -Is)"
EOF

  chmod +x "$job_script"
  echo -e "${experiment}\t${mode}\t${manifest_dir}/standardized_manifest.csv\t${manifest_dir}/label_vocabulary.json\t${run_dir}\t${wandb_group}" >> "$PLAN_TSV"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY_RUN: $experiment -> $job_script"
    return 0
  fi
  local sbatch_out job_id
  sbatch_out="$(sbatch "$job_script")"
  echo "$sbatch_out"
  job_id="$(echo "$sbatch_out" | awk '{print $NF}')"
  echo -e "${job_id}\t${experiment}\t${run_dir}\t${job_script}" >> "$SUBMITTED_TSV"
}

cd "$REPO_ROOT"
source .venv/bin/activate

if should_run "E01"; then
  manifest_dir="$(build_manifest "E01_onc_bal100_control" "species" "onc_bal100|$ONC_SPLIT|$ONC_BAL100_RUN")"
  write_and_submit_job "E01_onc_bal100_control" "species" "$manifest_dir" "weekend-20260502-onc-controls"
fi

if should_run "E02"; then
  manifest_dir="$(build_manifest "E02_biodcase_train50_only" "species_call" "biodcase_train50|$BIOD_SPLIT|$BIODCASE_TRAIN50_RUN")"
  write_and_submit_job "E02_biodcase_train50_only" "species_call" "$manifest_dir" "weekend-20260502-biodcase"
fi

if should_run "E03"; then
  manifest_dir="$(build_manifest "E03_onc_biod_train50_species" "species" \
    "onc_bal100|$ONC_SPLIT|$ONC_BAL100_RUN" \
    "biodcase_train50|$BIOD_SPLIT|$BIODCASE_TRAIN50_RUN")"
  write_and_submit_job "E03_onc_biod_train50_species" "species" "$manifest_dir" "weekend-20260502-combined"
fi

if should_run "E04"; then
  manifest_dir="$(build_manifest "E04_onc_biod_train50_species_call" "species_call" \
    "onc_bal100|$ONC_SPLIT|$ONC_BAL100_RUN" \
    "biodcase_train50|$BIOD_SPLIT|$BIODCASE_TRAIN50_RUN")"
  write_and_submit_job "E04_onc_biod_train50_species_call" "species_call" "$manifest_dir" "weekend-20260502-combined"
fi

echo "Wrote plan: $PLAN_TSV"
echo "Wrote submissions: $SUBMITTED_TSV"
