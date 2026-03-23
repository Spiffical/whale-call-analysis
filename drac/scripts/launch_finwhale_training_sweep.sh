#!/bin/bash
# Launch a structured sweep of FinWhale training jobs on DRAC.
#
# Run from login node (not via sbatch), e.g.:
#   bash drac/scripts/launch_finwhale_training_sweep.sh \
#     --tar-path /path/to/all_mat_files.tar \
#     --wandb-project finwhale-resnet \
#     --wandb-group-prefix finwhale-training-audit

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
SUBMIT_SCRIPT="$REPO_ROOT/drac/scripts/submit_finwhale_cnn.sh"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 1
fi

TAR_PATH=""
POS_DIR=""
NEG_DIR=""
DATASET_TARS_CSV=""      # CSV list: tag=/path/to/all_mat_files.tar,tag2=/path/to/all_mat_files.tar
DATASET_TAG_SINGLE=""    # Optional tag for single data source
MODEL="resnet18"
MODELS_CSV=""
MODEL_PRESET=""
BATCH_SIZE=64
EPOCHS=100
NUM_WORKERS=4
TRAIN_RATIO=0.8
VAL_RATIO=0.1
MAIN_METRIC="f1"
DEVICE="cuda"
SPLIT_STRATEGY="time_separated"
SEEDS_CSV="42"
LRS_CSV="1e-3"
BALANCES_CSV="weighted,none"
CENTER_BIAS_CSV="0.25,0.45,0.65"
MIN_GAPS_CSV="120,180"
USE_WANDB="true"
WANDB_PROJECT="finwhale-resnet"
WANDB_GROUP_PREFIX="finwhale-training-sweep"
WANDB_ENTITY=""
SWEEP_ID=""
EXP_ROOT="${SCRATCH:-/scratch/$USER}/finwhale_sweeps"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_training_sweep.sh [options]

Required (choose one data source):
  --dataset-tars CSV
  --tar-path PATH
  --pos-dir PATH --neg-dir PATH

Options:
  --dataset-tag TAG                Optional tag for single data source (default: auto)
  --dataset-tars CSV               Multi-dataset tar list: tag=/path/a.tar,tag2=/path/b.tar
  --model NAME                     Single model name (default: resnet18)
  --models CSV                     Comma-separated model list
  --model-preset NAME              One of: architecture_benchmark, resnets
  --batch-size N                   (default: 64)
  --epochs N                       (default: 100)
  --num-workers N                  (default: 4)
  --train-ratio R                  (default: 0.8)
  --val-ratio R                    (default: 0.1)
  --main-metric NAME               (default: f1)
  --device NAME                    (default: cuda)
  --split-strategy NAME            (default: time_separated)
  --seeds CSV                      (default: 42)
  --lrs CSV                        (default: 1e-3)
  --balances CSV                   (default: weighted,none)
  --center-bias-list CSV           (default: 0.25,0.45,0.65)
  --min-gap-list CSV               (default: 120,180)
  --exp-root PATH                  (default: $SCRATCH/finwhale_sweeps)
  --sweep-id ID                    (default: auto timestamp)
  --wandb-project NAME             (default: finwhale-resnet)
  --wandb-group-prefix NAME        (default: finwhale-training-sweep)
  --wandb-entity NAME
  --no-wandb
  --dry-run
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-tars) DATASET_TARS_CSV="$2"; shift 2 ;;
    --dataset-tag) DATASET_TAG_SINGLE="$2"; shift 2 ;;
    --tar-path) TAR_PATH="$2"; shift 2 ;;
    --pos-dir) POS_DIR="$2"; shift 2 ;;
    --neg-dir) NEG_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --models) MODELS_CSV="$2"; shift 2 ;;
    --model-preset) MODEL_PRESET="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --main-metric) MAIN_METRIC="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --split-strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --lrs) LRS_CSV="$2"; shift 2 ;;
    --balances) BALANCES_CSV="$2"; shift 2 ;;
    --center-bias-list) CENTER_BIAS_CSV="$2"; shift 2 ;;
    --min-gap-list) MIN_GAPS_CSV="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group-prefix) WANDB_GROUP_PREFIX="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --no-wandb) USE_WANDB="false"; shift ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --sweep-id) SWEEP_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$DATASET_TARS_CSV" && ( -n "$TAR_PATH" || -n "$POS_DIR" || -n "$NEG_DIR" ) ]]; then
  echo "Error: --dataset-tars cannot be combined with --tar-path or --pos-dir/--neg-dir"
  exit 1
fi
if [[ -z "$DATASET_TARS_CSV" && -z "$TAR_PATH" && ( -z "$POS_DIR" || -z "$NEG_DIR" ) ]]; then
  echo "Error: provide one of: --dataset-tars, --tar-path, or --pos-dir/--neg-dir"
  exit 1
fi

if [[ -z "$SWEEP_ID" ]]; then
  SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -n "$MODELS_CSV" && -n "$MODEL_PRESET" ]]; then
  echo "Error: use either --models or --model-preset, not both"
  exit 1
fi

split_csv() {
  local raw="$1"
  raw="${raw// /}"
  IFS=',' read -r -a parts <<< "$raw"
  for x in "${parts[@]}"; do
    if [[ -n "$x" ]]; then
      echo "$x"
    fi
  done
}

emit_model_preset() {
  local preset="$1"
  case "$preset" in
    architecture_benchmark|benchmark|diverse)
      cat <<'EOF'
SmallCNN
DeepCNN:w32:d4
DeepCNN:w64:d6
DeepCNN:w96:d8
resnet18
resnet34
resnet50
EOF
      ;;
    resnets|resnet_family)
      cat <<'EOF'
resnet18
resnet34
resnet50
EOF
      ;;
    *)
      echo "Error: unknown --model-preset '$preset'" >&2
      return 1
      ;;
  esac
}

readarray -t SEED_LIST < <(split_csv "$SEEDS_CSV")
readarray -t LR_LIST < <(split_csv "$LRS_CSV")
readarray -t BALANCE_LIST < <(split_csv "$BALANCES_CSV")
readarray -t CENTER_BIAS_LIST < <(split_csv "$CENTER_BIAS_CSV")
readarray -t MIN_GAP_LIST < <(split_csv "$MIN_GAPS_CSV")
if [[ -n "$MODELS_CSV" ]]; then
  readarray -t MODEL_LIST < <(split_csv "$MODELS_CSV")
elif [[ -n "$MODEL_PRESET" ]]; then
  readarray -t MODEL_LIST < <(emit_model_preset "$MODEL_PRESET")
else
  MODEL_LIST=("$MODEL")
fi

if [[ ${#SEED_LIST[@]} -eq 0 || ${#LR_LIST[@]} -eq 0 || ${#BALANCE_LIST[@]} -eq 0 || ${#CENTER_BIAS_LIST[@]} -eq 0 || ${#MIN_GAP_LIST[@]} -eq 0 || ${#MODEL_LIST[@]} -eq 0 ]]; then
  echo "Error: one of the parameter lists is empty."
  exit 1
fi

SWEEP_DIR="$EXP_ROOT/$SWEEP_ID"
RUNS_DIR="$SWEEP_DIR/runs"
mkdir -p "$RUNS_DIR"

PLAN_TSV="$SWEEP_DIR/plan.tsv"
SUBMITTED_TSV="$SWEEP_DIR/submitted_jobs.tsv"
REPLAY_SH="$SWEEP_DIR/replay_commands.sh"
WANDB_GROUP_BASE="${WANDB_GROUP_PREFIX}-${SWEEP_ID}"

echo -e "run_slug\tdataset_tag\tdataset_source\tmodel\tlr\tbalance\tcenter_bias_sigma_frac\tmin_gap_seconds\tseed\texp_dir\twandb_group" > "$PLAN_TSV"
echo -e "job_id\trun_slug\tdataset_tag\tdataset_source\tmodel\tlr\tbalance\tcenter_bias_sigma_frac\tmin_gap_seconds\tseed\texp_dir\twandb_group" > "$SUBMITTED_TSV"
echo "#!/bin/bash" > "$REPLAY_SH"
echo "set -euo pipefail" >> "$REPLAY_SH"

to_tag() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//+/}"
  echo "$v" | tr -cs '[:alnum:]_-' '_' | sed 's/^_//;s/_$//'
}

declare -a DATASET_TAG_LIST=()
declare -a DATASET_SOURCE_LIST=()
declare -a DATASET_MODE_LIST=()

if [[ -n "$DATASET_TARS_CSV" ]]; then
  readarray -t DATASET_PAIR_LIST < <(split_csv "$DATASET_TARS_CSV")
  if [[ ${#DATASET_PAIR_LIST[@]} -eq 0 ]]; then
    echo "Error: --dataset-tars is empty"
    exit 1
  fi
  for pair in "${DATASET_PAIR_LIST[@]}"; do
    if [[ "$pair" != *=* ]]; then
      echo "Error: invalid --dataset-tars entry '$pair' (expected tag=/path/to/archive.tar)"
      exit 1
    fi
    tag_raw="${pair%%=*}"
    src="${pair#*=}"
    tag="$(to_tag "$tag_raw")"
    if [[ -z "$tag" || -z "$src" ]]; then
      echo "Error: invalid --dataset-tars entry '$pair'"
      exit 1
    fi
    DATASET_TAG_LIST+=("$tag")
    DATASET_SOURCE_LIST+=("$src")
    DATASET_MODE_LIST+=("tar")
  done
elif [[ -n "$TAR_PATH" ]]; then
  default_tag="$(basename "$(dirname "$TAR_PATH")")"
  tag="$(to_tag "${DATASET_TAG_SINGLE:-$default_tag}")"
  if [[ -z "$tag" ]]; then
    tag="dataset"
  fi
  DATASET_TAG_LIST+=("$tag")
  DATASET_SOURCE_LIST+=("$TAR_PATH")
  DATASET_MODE_LIST+=("tar")
else
  tag="$(to_tag "${DATASET_TAG_SINGLE:-dirs}")"
  if [[ -z "$tag" ]]; then
    tag="dataset"
  fi
  DATASET_TAG_LIST+=("$tag")
  DATASET_SOURCE_LIST+=("$POS_DIR|$NEG_DIR")
  DATASET_MODE_LIST+=("dirs")
fi

if [[ ${#DATASET_TAG_LIST[@]} -eq 0 ]]; then
  echo "Error: no datasets resolved for sweep"
  exit 1
fi

RUN_COUNT=0
for d_idx in "${!DATASET_TAG_LIST[@]}"; do
  dataset_tag="${DATASET_TAG_LIST[$d_idx]}"
  dataset_source="${DATASET_SOURCE_LIST[$d_idx]}"
  dataset_mode="${DATASET_MODE_LIST[$d_idx]}"
  dataset_group="${WANDB_GROUP_BASE}-${dataset_tag}"

  for model in "${MODEL_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      for lr in "${LR_LIST[@]}"; do
        for balance in "${BALANCE_LIST[@]}"; do
          for cbs in "${CENTER_BIAS_LIST[@]}"; do
            for gap in "${MIN_GAP_LIST[@]}"; do
              RUN_COUNT=$((RUN_COUNT + 1))
              run_slug=$(printf "r%03d_d%s_%s_lr%s_bal%s_cbs%s_gap%s_s%s" \
                "$RUN_COUNT" \
                "$(to_tag "$dataset_tag")" \
                "$(to_tag "$model")" \
                "$(to_tag "$lr")" \
                "$(to_tag "$balance")" \
                "$(to_tag "$cbs")" \
                "$(to_tag "$gap")" \
                "$(to_tag "$seed")")

              run_exp_dir="$RUNS_DIR/$dataset_tag/$run_slug"
              mkdir -p "$run_exp_dir"

              echo -e "${run_slug}\t${dataset_tag}\t${dataset_source}\t${model}\t${lr}\t${balance}\t${cbs}\t${gap}\t${seed}\t${run_exp_dir}\t${dataset_group}" >> "$PLAN_TSV"

              cmd=(
                sbatch --parsable "$SUBMIT_SCRIPT"
                --model "$model"
                --batch-size "$BATCH_SIZE"
                --epochs "$EPOCHS"
                --num-workers "$NUM_WORKERS"
                --lr "$lr"
                --balance "$balance"
                --train-ratio "$TRAIN_RATIO"
                --val-ratio "$VAL_RATIO"
                --main-metric "$MAIN_METRIC"
                --device "$DEVICE"
                --exp-dir "$run_exp_dir"
                --seed "$seed"
                --split-strategy "$SPLIT_STRATEGY"
                --min-gap-seconds "$gap"
                --center-bias-sigma-frac "$cbs"
                --run-tag "$run_slug"
              )

              if [[ "$dataset_mode" == "tar" ]]; then
                cmd+=( --tar-path "$dataset_source" )
              else
                ds_pos="${dataset_source%%|*}"
                ds_neg="${dataset_source#*|}"
                cmd+=( --pos-dir "$ds_pos" --neg-dir "$ds_neg" )
              fi

              if [[ "$USE_WANDB" == "true" ]]; then
                cmd+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$dataset_group" )
                if [[ -n "$WANDB_ENTITY" ]]; then
                  cmd+=( --wandb-entity "$WANDB_ENTITY" )
                fi
              fi

              {
                printf '%q ' "${cmd[@]}"
                printf '\n'
              } >> "$REPLAY_SH"

              if [[ "$DRY_RUN" == "true" ]]; then
                job_id="DRYRUN_${RUN_COUNT}"
                echo "[dry-run] $job_id $run_slug dataset=$dataset_tag model=$model"
              else
                sbatch_out="$("${cmd[@]}")"
                job_id="${sbatch_out%%;*}"
                echo "[submitted] job_id=${job_id} run_slug=${run_slug} dataset=${dataset_tag} model=${model}"
              fi

              echo -e "${job_id}\t${run_slug}\t${dataset_tag}\t${dataset_source}\t${model}\t${lr}\t${balance}\t${cbs}\t${gap}\t${seed}\t${run_exp_dir}\t${dataset_group}" >> "$SUBMITTED_TSV"
            done
          done
        done
      done
    done
  done
done

chmod +x "$REPLAY_SH"

echo ""
echo "Sweep prepared: $SWEEP_DIR"
echo "Datasets:"
for d_idx in "${!DATASET_TAG_LIST[@]}"; do
  echo "  - ${DATASET_TAG_LIST[$d_idx]} -> ${DATASET_SOURCE_LIST[$d_idx]}"
done
echo "Models:"
for model in "${MODEL_LIST[@]}"; do
  echo "  - $model"
done
echo "Planned runs: $RUN_COUNT"
echo "Plan file: $PLAN_TSV"
echo "Submitted jobs: $SUBMITTED_TSV"
echo "Replay commands: $REPLAY_SH"
echo ""
echo "When jobs finish, summarize with:"
echo "  python $REPO_ROOT/drac/scripts/summarize_finwhale_sweep.py --sweep-dir $SWEEP_DIR"
