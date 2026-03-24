#!/bin/bash
# Launch a Part 2 fin-whale fine-tuning learning-curve experiment on DRAC/Nibi.
#
# This submits, for each budget/sampling-mode run:
#  1. fine-tune training from the existing checkpoint
#  2. historical retention evaluation on the 2018/2019 held-out baseline
#  3. Part 2 evaluation on a fixed held-out Part 2 bundle

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
TRAIN_SUBMIT="$REPO_ROOT/drac/scripts/submit_finwhale_cnn.sh"
TEST_SUBMIT="$REPO_ROOT/drac/scripts/submit_finwhale_test.sh"
PART2_SUBMIT="$REPO_ROOT/drac/scripts/submit_finwhale_part2_eval.sh"

for required in "$TRAIN_SUBMIT" "$TEST_SUBMIT" "$PART2_SUBMIT"; do
  [[ -f "$required" ]] || { echo "Missing required submit script: $required"; exit 1; }
done

FINE_TUNE_TAR=""
FINE_TUNE_POS_DIR=""
FINE_TUNE_NEG_DIR=""
SPLITS_ROOT=""
BASE_CHECKPOINT=""
HISTORICAL_BASELINE_TAR=""
HISTORICAL_SPLITS_DIR=""
PART2_TEST_ARCHIVE=""
CURVE_ID=""
OUT_ROOT="${SCRATCH:-/scratch/$USER}/finwhale_part2_finetune_curve"
MODEL="resnet18"
BATCH_SIZE=64
EPOCHS=20
NUM_WORKERS=4
LR="1e-4"
BALANCE="none"
CROP_SIZE="96"
DEVICE="cuda"
CENTER_BIAS_SIGMA_FRAC="0.25"
WANDB_PROJECT="finwhale-resnet"
WANDB_GROUP_PREFIX="finwhale-part2-curve"
WANDB_ENTITY=""
USE_WANDB="true"
PART2_WINDOW_STEPS="24"
PART2_LOW_THRESHOLDS="0.80"
PART2_HIGH_THRESHOLDS="0.82"
PART2_MIN_MEMBERS="2"
PART2_MAX_GAP_VALUES="auto"
PART2_MATCH_COLLAR_S="1.0"
MERGE_EVENT_MEDIA="false"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_part2_finetune_curve.sh [options]

Required:
  --splits-root PATH                 Output directory from prepare_part2_finetune_learning_curve.py
  --base-checkpoint PATH             Pretrained checkpoint to fine-tune from
  --historical-baseline-tar PATH     Original training-domain dataset tar (all_mat_files.tar)
  --part2-test-archive PATH          Fixed held-out Part 2 bundle/archive for evaluation

Data source (choose one):
  --fine-tune-tar PATH
  --fine-tune-pos-dir PATH --fine-tune-neg-dir PATH

Optional:
  --historical-splits-dir PATH       Defaults to <dirname(base-checkpoint)>/splits
  --curve-id ID                      Defaults to UTC timestamp
  --out-root PATH                    Defaults to $SCRATCH/finwhale_part2_finetune_curve
  --model NAME                       Default: resnet18
  --batch-size N                     Default: 64
  --epochs N                         Default: 20
  --num-workers N                    Default: 4
  --lr VALUE                         Default: 1e-4
  --balance VALUE                    Default: none
  --crop-size VALUE                  Default: 96
  --device NAME                      Default: cuda
  --center-bias-sigma-frac VALUE     Default: 0.25
  --part2-window-steps CSV           Default: 24
  --part2-low-thresholds CSV         Default: 0.80
  --part2-high-thresholds CSV        Default: 0.82
  --part2-min-members-values CSV     Default: 2
  --part2-max-gap-values CSV         Default: auto
  --part2-match-collar-s VALUE       Default: 1.0
  --merge-event-media
  --wandb-project NAME               Default: finwhale-resnet
  --wandb-group-prefix NAME          Default: finwhale-part2-curve
  --wandb-entity NAME
  --no-wandb
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fine-tune-tar) FINE_TUNE_TAR="$2"; shift 2 ;;
    --fine-tune-pos-dir) FINE_TUNE_POS_DIR="$2"; shift 2 ;;
    --fine-tune-neg-dir) FINE_TUNE_NEG_DIR="$2"; shift 2 ;;
    --splits-root) SPLITS_ROOT="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --historical-baseline-tar) HISTORICAL_BASELINE_TAR="$2"; shift 2 ;;
    --historical-splits-dir) HISTORICAL_SPLITS_DIR="$2"; shift 2 ;;
    --part2-test-archive) PART2_TEST_ARCHIVE="$2"; shift 2 ;;
    --curve-id) CURVE_ID="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --balance) BALANCE="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --center-bias-sigma-frac) CENTER_BIAS_SIGMA_FRAC="$2"; shift 2 ;;
    --part2-window-steps) PART2_WINDOW_STEPS="$2"; shift 2 ;;
    --part2-low-thresholds) PART2_LOW_THRESHOLDS="$2"; shift 2 ;;
    --part2-high-thresholds) PART2_HIGH_THRESHOLDS="$2"; shift 2 ;;
    --part2-min-members-values) PART2_MIN_MEMBERS="$2"; shift 2 ;;
    --part2-max-gap-values) PART2_MAX_GAP_VALUES="$2"; shift 2 ;;
    --part2-match-collar-s) PART2_MATCH_COLLAR_S="$2"; shift 2 ;;
    --merge-event-media) MERGE_EVENT_MEDIA="true"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group-prefix) WANDB_GROUP_PREFIX="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --no-wandb) USE_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SPLITS_ROOT" || -z "$BASE_CHECKPOINT" || -z "$HISTORICAL_BASELINE_TAR" || -z "$PART2_TEST_ARCHIVE" ]]; then
  echo "Error: --splits-root, --base-checkpoint, --historical-baseline-tar, and --part2-test-archive are required"
  exit 1
fi
if [[ -z "$FINE_TUNE_TAR" && ( -z "$FINE_TUNE_POS_DIR" || -z "$FINE_TUNE_NEG_DIR" ) ]]; then
  echo "Error: provide either --fine-tune-tar or --fine-tune-pos-dir/--fine-tune-neg-dir"
  exit 1
fi
if [[ -n "$FINE_TUNE_TAR" && ( -n "$FINE_TUNE_POS_DIR" || -n "$FINE_TUNE_NEG_DIR" ) ]]; then
  echo "Error: --fine-tune-tar cannot be combined with --fine-tune-pos-dir/--fine-tune-neg-dir"
  exit 1
fi

SPLITS_ROOT="$(realpath "$SPLITS_ROOT")"
BASE_CHECKPOINT="$(realpath "$BASE_CHECKPOINT")"
HISTORICAL_BASELINE_TAR="$(realpath "$HISTORICAL_BASELINE_TAR")"
PART2_TEST_ARCHIVE="$(realpath "$PART2_TEST_ARCHIVE")"
[[ -n "$FINE_TUNE_TAR" ]] && FINE_TUNE_TAR="$(realpath "$FINE_TUNE_TAR")"
[[ -n "$FINE_TUNE_POS_DIR" ]] && FINE_TUNE_POS_DIR="$(realpath "$FINE_TUNE_POS_DIR")"
[[ -n "$FINE_TUNE_NEG_DIR" ]] && FINE_TUNE_NEG_DIR="$(realpath "$FINE_TUNE_NEG_DIR")"

if [[ -z "$HISTORICAL_SPLITS_DIR" ]]; then
  HISTORICAL_SPLITS_DIR="$(dirname "$BASE_CHECKPOINT")/splits"
fi
HISTORICAL_SPLITS_DIR="$(realpath "$HISTORICAL_SPLITS_DIR")"

PLAN_CSV="$SPLITS_ROOT/learning_curve_plan.csv"
[[ -f "$PLAN_CSV" ]] || { echo "Missing learning curve plan: $PLAN_CSV"; exit 1; }

if [[ -z "$CURVE_ID" ]]; then
  CURVE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

CURVE_DIR="$OUT_ROOT/$CURVE_ID"
mkdir -p "$CURVE_DIR"
PLAN_TSV="$CURVE_DIR/plan.tsv"
SUBMITTED_TSV="$CURVE_DIR/submitted_jobs.tsv"
REPLAY_SH="$CURVE_DIR/replay_commands.sh"

echo -e "run_id\tsampling_mode\tactual_budget_calls\ttrain_fin_clip_count\trepeat_index\ttrain_exp_path\tcheckpoint_path" > "$PLAN_TSV"
echo -e "train_job_id\tretention_job_id\tpart2_job_id\trun_id\tsampling_mode\tactual_budget_calls\ttrain_fin_clip_count\trepeat_index\tcheckpoint_path" > "$SUBMITTED_TSV"
echo "#!/bin/bash" > "$REPLAY_SH"
echo "set -euo pipefail" >> "$REPLAY_SH"

WANDB_GROUP="${WANDB_GROUP_PREFIX}-${CURVE_ID}"

while IFS=$'\t' read -r RUN_ID SAMPLING_MODE ACTUAL_BUDGET_CALLS TRAIN_FIN_CLIP_COUNT REPEAT_INDEX; do
  [[ -n "$RUN_ID" ]] || continue
  RUN_SPLIT_DIR="$SPLITS_ROOT/runs/$RUN_ID"
  [[ -d "$RUN_SPLIT_DIR" ]] || { echo "Missing run split dir: $RUN_SPLIT_DIR"; exit 1; }

  RUN_DIR="$CURVE_DIR/$RUN_ID"
  TRAIN_EXP_PATH="$RUN_DIR/train"
  TRAIN_CHECKPOINT="$TRAIN_EXP_PATH/best.pt"
  RETENTION_OUT_DIR="$RUN_DIR/historical_retention"
  PART2_OUT_DIR="$RUN_DIR/part2_eval"
  WANDB_NAME_BASE="p2ft_${RUN_ID}"
  WANDB_TAGS="part2-finetune,learning-curve,${SAMPLING_MODE},budget_${ACTUAL_BUDGET_CALLS}"

  echo -e "${RUN_ID}\t${SAMPLING_MODE}\t${ACTUAL_BUDGET_CALLS}\t${TRAIN_FIN_CLIP_COUNT}\t${REPEAT_INDEX}\t${TRAIN_EXP_PATH}\t${TRAIN_CHECKPOINT}" >> "$PLAN_TSV"

  TRAIN_CMD=(
    sbatch --parsable "$TRAIN_SUBMIT"
    --model "$MODEL"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --balance "$BALANCE"
    --device "$DEVICE"
    --crop-size "$CROP_SIZE"
    --center-bias-sigma-frac "$CENTER_BIAS_SIGMA_FRAC"
    --splits-dir "$RUN_SPLIT_DIR"
    --init-checkpoint "$BASE_CHECKPOINT"
    --exp-path "$TRAIN_EXP_PATH"
    --run-tag "$RUN_ID"
    --experiment-id "$CURVE_ID"
    --sampling-mode "$SAMPLING_MODE"
    --budget-calls "$ACTUAL_BUDGET_CALLS"
    --budget-clips "$TRAIN_FIN_CLIP_COUNT"
    --repeat-index "$REPEAT_INDEX"
  )
  if [[ -n "$FINE_TUNE_TAR" ]]; then
    TRAIN_CMD+=( --tar-path "$FINE_TUNE_TAR" )
  else
    TRAIN_CMD+=( --pos-dir "$FINE_TUNE_POS_DIR" --neg-dir "$FINE_TUNE_NEG_DIR" )
  fi
  if [[ "$USE_WANDB" == "true" ]]; then
    TRAIN_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name "${WANDB_NAME_BASE}_train" --wandb-tags "$WANDB_TAGS,train" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      TRAIN_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
  fi

  TEST_CMD=(
    sbatch --parsable --dependency=afterok:%TRAIN_JOB% "$TEST_SUBMIT"
    --tar-path "$HISTORICAL_BASELINE_TAR"
    --checkpoint "$TRAIN_CHECKPOINT"
    --out-dir "$RETENTION_OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --crop-size "$CROP_SIZE"
    --device "$DEVICE"
    --splits-dir "$HISTORICAL_SPLITS_DIR"
    --eval-split test
  )
  if [[ "$USE_WANDB" == "true" ]]; then
    TEST_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name "${WANDB_NAME_BASE}_retention" --wandb-tags "$WANDB_TAGS,retention" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      TEST_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
  fi

  PART2_CMD=(
    sbatch --parsable --dependency=afterok:%TRAIN_JOB% "$PART2_SUBMIT"
    --part2-archive "$PART2_TEST_ARCHIVE"
    --checkpoint "$TRAIN_CHECKPOINT"
    --out-dir "$PART2_OUT_DIR"
    --window-steps "$PART2_WINDOW_STEPS"
    --low-thresholds "$PART2_LOW_THRESHOLDS"
    --high-thresholds "$PART2_HIGH_THRESHOLDS"
    --min-members-values "$PART2_MIN_MEMBERS"
    --max-gap-values "$PART2_MAX_GAP_VALUES"
    --match-collar-s "$PART2_MATCH_COLLAR_S"
    --crop-size "$CROP_SIZE"
  )
  if [[ "$MERGE_EVENT_MEDIA" == "true" ]]; then
    PART2_CMD+=( --merge-event-media )
  fi
  if [[ "$USE_WANDB" == "true" ]]; then
    PART2_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name-prefix "${WANDB_NAME_BASE}" --wandb-tags "$WANDB_TAGS,part2-eval" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      PART2_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
  fi

  printf '%q ' "${TRAIN_CMD[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"
  TEST_CMD_REPLAY=("${TEST_CMD[@]/%TRAIN_JOB%/JOBID}")
  printf '%q ' "${TEST_CMD_REPLAY[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"
  PART2_CMD_REPLAY=("${PART2_CMD[@]/%TRAIN_JOB%/JOBID}")
  printf '%q ' "${PART2_CMD_REPLAY[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: ${TRAIN_CMD[*]}"
    echo "DRY RUN: ${TEST_CMD[*]}"
    echo "DRY RUN: ${PART2_CMD[*]}"
    continue
  fi

  TRAIN_JOB_ID="$("${TRAIN_CMD[@]}")"
  TEST_CMD_ACTUAL=("${TEST_CMD[@]/%TRAIN_JOB%/$TRAIN_JOB_ID}")
  PART2_CMD_ACTUAL=("${PART2_CMD[@]/%TRAIN_JOB%/$TRAIN_JOB_ID}")
  RETENTION_JOB_ID="$("${TEST_CMD_ACTUAL[@]}")"
  PART2_JOB_ID="$("${PART2_CMD_ACTUAL[@]}")"

  echo -e "${TRAIN_JOB_ID}\t${RETENTION_JOB_ID}\t${PART2_JOB_ID}\t${RUN_ID}\t${SAMPLING_MODE}\t${ACTUAL_BUDGET_CALLS}\t${TRAIN_FIN_CLIP_COUNT}\t${REPEAT_INDEX}\t${TRAIN_CHECKPOINT}" >> "$SUBMITTED_TSV"
  echo "Submitted $RUN_ID: train=$TRAIN_JOB_ID retention=$RETENTION_JOB_ID part2=$PART2_JOB_ID"
done < <(
  python - "$PLAN_CSV" <<'PY'
import csv
import sys

plan_csv = sys.argv[1]
with open(plan_csv, "r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        print(
            "\t".join(
                [
                    row.get("run_id", ""),
                    row.get("sampling_mode", ""),
                    row.get("actual_budget_calls", ""),
                    row.get("train_fin_clip_count", ""),
                    row.get("repeat_index", ""),
                ]
            )
        )
PY
)

echo "Learning-curve submissions complete."
echo "Plan: $PLAN_TSV"
echo "Submitted jobs: $SUBMITTED_TSV"
