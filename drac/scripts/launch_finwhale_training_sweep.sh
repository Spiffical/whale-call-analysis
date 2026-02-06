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
MODEL="resnet18"
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
  --tar-path PATH
  --pos-dir PATH --neg-dir PATH

Options:
  --model NAME                     (default: resnet18)
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
    --tar-path) TAR_PATH="$2"; shift 2 ;;
    --pos-dir) POS_DIR="$2"; shift 2 ;;
    --neg-dir) NEG_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
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

if [[ -z "$TAR_PATH" && ( -z "$POS_DIR" || -z "$NEG_DIR" ) ]]; then
  echo "Error: provide either --tar-path or both --pos-dir and --neg-dir"
  exit 1
fi

if [[ -z "$SWEEP_ID" ]]; then
  SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)"
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

readarray -t SEED_LIST < <(split_csv "$SEEDS_CSV")
readarray -t LR_LIST < <(split_csv "$LRS_CSV")
readarray -t BALANCE_LIST < <(split_csv "$BALANCES_CSV")
readarray -t CENTER_BIAS_LIST < <(split_csv "$CENTER_BIAS_CSV")
readarray -t MIN_GAP_LIST < <(split_csv "$MIN_GAPS_CSV")

if [[ ${#SEED_LIST[@]} -eq 0 || ${#LR_LIST[@]} -eq 0 || ${#BALANCE_LIST[@]} -eq 0 || ${#CENTER_BIAS_LIST[@]} -eq 0 || ${#MIN_GAP_LIST[@]} -eq 0 ]]; then
  echo "Error: one of the parameter lists is empty."
  exit 1
fi

SWEEP_DIR="$EXP_ROOT/$SWEEP_ID"
RUNS_DIR="$SWEEP_DIR/runs"
mkdir -p "$RUNS_DIR"

PLAN_TSV="$SWEEP_DIR/plan.tsv"
SUBMITTED_TSV="$SWEEP_DIR/submitted_jobs.tsv"
REPLAY_SH="$SWEEP_DIR/replay_commands.sh"

WANDB_GROUP="${WANDB_GROUP_PREFIX}-${SWEEP_ID}"

echo -e "run_slug\tmodel\tlr\tbalance\tcenter_bias_sigma_frac\tmin_gap_seconds\tseed\texp_dir\twandb_group" > "$PLAN_TSV"
echo -e "job_id\trun_slug\tmodel\tlr\tbalance\tcenter_bias_sigma_frac\tmin_gap_seconds\tseed\texp_dir\twandb_group" > "$SUBMITTED_TSV"
echo "#!/bin/bash" > "$REPLAY_SH"
echo "set -euo pipefail" >> "$REPLAY_SH"

to_tag() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//+/}"
  echo "$v" | tr -cs '[:alnum:]_-' '_' | sed 's/^_//;s/_$//'
}

RUN_COUNT=0
for seed in "${SEED_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for balance in "${BALANCE_LIST[@]}"; do
      for cbs in "${CENTER_BIAS_LIST[@]}"; do
        for gap in "${MIN_GAP_LIST[@]}"; do
          RUN_COUNT=$((RUN_COUNT + 1))
          run_slug=$(printf "r%03d_%s_lr%s_bal%s_cbs%s_gap%s_s%s" \
            "$RUN_COUNT" \
            "$(to_tag "$MODEL")" \
            "$(to_tag "$lr")" \
            "$(to_tag "$balance")" \
            "$(to_tag "$cbs")" \
            "$(to_tag "$gap")" \
            "$(to_tag "$seed")")

          run_exp_dir="$RUNS_DIR/$run_slug"
          mkdir -p "$run_exp_dir"

          echo -e "${run_slug}\t${MODEL}\t${lr}\t${balance}\t${cbs}\t${gap}\t${seed}\t${run_exp_dir}\t${WANDB_GROUP}" >> "$PLAN_TSV"

          cmd=(
            sbatch --parsable "$SUBMIT_SCRIPT"
            --model "$MODEL"
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

          if [[ -n "$TAR_PATH" ]]; then
            cmd+=( --tar-path "$TAR_PATH" )
          else
            cmd+=( --pos-dir "$POS_DIR" --neg-dir "$NEG_DIR" )
          fi

          if [[ "$USE_WANDB" == "true" ]]; then
            cmd+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" )
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
            echo "[dry-run] $job_id $run_slug"
          else
            sbatch_out="$("${cmd[@]}")"
            job_id="${sbatch_out%%;*}"
            echo "[submitted] job_id=${job_id} run_slug=${run_slug}"
          fi

          echo -e "${job_id}\t${run_slug}\t${MODEL}\t${lr}\t${balance}\t${cbs}\t${gap}\t${seed}\t${run_exp_dir}\t${WANDB_GROUP}" >> "$SUBMITTED_TSV"
        done
      done
    done
  done
done

chmod +x "$REPLAY_SH"

echo ""
echo "Sweep prepared: $SWEEP_DIR"
echo "Planned runs: $RUN_COUNT"
echo "Plan file: $PLAN_TSV"
echo "Submitted jobs: $SUBMITTED_TSV"
echo "Replay commands: $REPLAY_SH"
echo ""
echo "When jobs finish, summarize with:"
echo "  python $REPO_ROOT/drac/scripts/summarize_finwhale_sweep.py --sweep-dir $SWEEP_DIR"
