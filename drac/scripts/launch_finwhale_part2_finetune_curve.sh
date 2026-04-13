#!/bin/bash
# Launch a coherent Part 2 fin-whale fine-tuning experiment on DRAC/Nibi.
#
# This submits, for each learning-curve run:
#   1. fine-tune training from the selected base checkpoint
#   2. historical retention evaluation on 2018/2019 Clayoquot
#   3. 2025 Clayoquot validation-bundle evaluation
#
# Optionally, it can also submit:
#   4. final held-out 2025 Clayoquot test evaluations for a shortlist of run IDs
#   5. the unfine-tuned baseline checkpoint on that same final test bundle

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
BASELINE_TEST_CHECKPOINT=""
HISTORICAL_BASELINE_TAR=""
HISTORICAL_SPLITS_DIR=""
PART2_EVAL_ARCHIVE=""
FINAL_TEST_ARCHIVE=""
FINAL_TEST_RUN_IDS=""
INCLUDE_RUN_IDS=""
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

TRAIN_PARTITION=""
TRAIN_GRES=""
TRAIN_TIME=""
TRAIN_CPUS=""
TRAIN_MEM=""
RETENTION_PARTITION=""
RETENTION_GRES=""
RETENTION_TIME=""
RETENTION_CPUS=""
RETENTION_MEM=""
PART2_PARTITION=""
PART2_GRES=""
PART2_TIME=""
PART2_CPUS=""
PART2_MEM=""

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_part2_finetune_curve.sh [options]

Required:
  --splits-root PATH                 Output directory from prepare_part2_finetune_learning_curve.py
  --base-checkpoint PATH             Pretrained checkpoint to fine-tune from
  --historical-baseline-tar PATH     Original training-domain dataset tar (all_mat_files.tar)
  --part2-eval-archive PATH          Fixed validation bundle/archive for learning-curve selection

Data source (choose one):
  --fine-tune-tar PATH
  --fine-tune-pos-dir PATH --fine-tune-neg-dir PATH

Optional:
  --historical-splits-dir PATH       Defaults to <dirname(base-checkpoint)>/splits
  --baseline-test-checkpoint PATH    Defaults to --base-checkpoint
  --final-test-archive PATH          Fixed final held-out 2025 bundle for shortlisted test comparisons
  --final-test-run-ids CSV           Run IDs to evaluate on --final-test-archive
  --include-run-ids CSV              Optional run IDs to submit from the learning-curve plan
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

  Resource overrides (all optional; sbatch CLI overrides script defaults):
    --train-partition NAME
    --train-gres SPEC
    --train-time HH:MM:SS
    --train-cpus N
    --train-mem SIZE
    --retention-partition NAME
    --retention-gres SPEC
    --retention-time HH:MM:SS
    --retention-cpus N
    --retention-mem SIZE
    --part2-partition NAME
    --part2-gres SPEC
    --part2-time HH:MM:SS
    --part2-cpus N
    --part2-mem SIZE

  Legacy compatibility:
    --part2-test-archive PATH        Alias for --part2-eval-archive when the latter is omitted

  Other:
    --dry-run
USAGE
}

normalize_job_id() {
  local raw="${1:-}"
  raw="${raw//$'\r'/}"
  raw="${raw//$'\n'/}"
  raw="${raw%%;*}"
  raw="$(echo "$raw" | tr -d '[:space:]')"
  if [[ ! "$raw" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse sbatch job id from output: '$1'" >&2
    return 1
  fi
  printf '%s' "$raw"
}

csv_contains() {
  local csv="${1:-}"
  local needle="${2:-}"
  [[ -n "$needle" ]] || return 1
  [[ -n "$csv" ]] || return 1
  IFS=',' read -r -a _csv_tokens <<<"$csv"
  for token in "${_csv_tokens[@]}"; do
    if [[ "$(echo "$token" | xargs)" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

append_resource_args() {
  local array_name="$1"
  local partition="$2"
  local gres="$3"
  local time_limit="$4"
  local cpus="$5"
  local mem="$6"
  local -n out="$array_name"
  [[ -n "$partition" ]] && out+=( --partition "$partition" )
  [[ -n "$gres" ]] && out+=( --gres "$gres" )
  [[ -n "$time_limit" ]] && out+=( --time "$time_limit" )
  [[ -n "$cpus" ]] && out+=( --cpus-per-task "$cpus" )
  [[ -n "$mem" ]] && out+=( --mem "$mem" )
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fine-tune-tar) FINE_TUNE_TAR="$2"; shift 2 ;;
    --fine-tune-pos-dir) FINE_TUNE_POS_DIR="$2"; shift 2 ;;
    --fine-tune-neg-dir) FINE_TUNE_NEG_DIR="$2"; shift 2 ;;
    --splits-root) SPLITS_ROOT="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --baseline-test-checkpoint) BASELINE_TEST_CHECKPOINT="$2"; shift 2 ;;
    --historical-baseline-tar) HISTORICAL_BASELINE_TAR="$2"; shift 2 ;;
    --historical-splits-dir) HISTORICAL_SPLITS_DIR="$2"; shift 2 ;;
    --part2-eval-archive) PART2_EVAL_ARCHIVE="$2"; shift 2 ;;
    --part2-test-archive)
      if [[ -z "$PART2_EVAL_ARCHIVE" ]]; then
        PART2_EVAL_ARCHIVE="$2"
      else
        FINAL_TEST_ARCHIVE="$2"
      fi
      shift 2
      ;;
    --final-test-archive) FINAL_TEST_ARCHIVE="$2"; shift 2 ;;
    --final-test-run-ids) FINAL_TEST_RUN_IDS="$2"; shift 2 ;;
    --include-run-ids) INCLUDE_RUN_IDS="$2"; shift 2 ;;
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
    --train-partition) TRAIN_PARTITION="$2"; shift 2 ;;
    --train-gres) TRAIN_GRES="$2"; shift 2 ;;
    --train-time) TRAIN_TIME="$2"; shift 2 ;;
    --train-cpus) TRAIN_CPUS="$2"; shift 2 ;;
    --train-mem) TRAIN_MEM="$2"; shift 2 ;;
    --retention-partition) RETENTION_PARTITION="$2"; shift 2 ;;
    --retention-gres) RETENTION_GRES="$2"; shift 2 ;;
    --retention-time) RETENTION_TIME="$2"; shift 2 ;;
    --retention-cpus) RETENTION_CPUS="$2"; shift 2 ;;
    --retention-mem) RETENTION_MEM="$2"; shift 2 ;;
    --part2-partition) PART2_PARTITION="$2"; shift 2 ;;
    --part2-gres) PART2_GRES="$2"; shift 2 ;;
    --part2-time) PART2_TIME="$2"; shift 2 ;;
    --part2-cpus) PART2_CPUS="$2"; shift 2 ;;
    --part2-mem) PART2_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SPLITS_ROOT" || -z "$BASE_CHECKPOINT" || -z "$HISTORICAL_BASELINE_TAR" || -z "$PART2_EVAL_ARCHIVE" ]]; then
  echo "Error: --splits-root, --base-checkpoint, --historical-baseline-tar, and --part2-eval-archive are required"
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
PART2_EVAL_ARCHIVE="$(realpath "$PART2_EVAL_ARCHIVE")"
[[ -n "$FINAL_TEST_ARCHIVE" ]] && FINAL_TEST_ARCHIVE="$(realpath "$FINAL_TEST_ARCHIVE")"
[[ -n "$FINE_TUNE_TAR" ]] && FINE_TUNE_TAR="$(realpath "$FINE_TUNE_TAR")"
[[ -n "$FINE_TUNE_POS_DIR" ]] && FINE_TUNE_POS_DIR="$(realpath "$FINE_TUNE_POS_DIR")"
[[ -n "$FINE_TUNE_NEG_DIR" ]] && FINE_TUNE_NEG_DIR="$(realpath "$FINE_TUNE_NEG_DIR")"

if [[ -z "$BASELINE_TEST_CHECKPOINT" ]]; then
  BASELINE_TEST_CHECKPOINT="$BASE_CHECKPOINT"
fi
BASELINE_TEST_CHECKPOINT="$(realpath "$BASELINE_TEST_CHECKPOINT")"

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
FINAL_TEST_TSV="$CURVE_DIR/final_test_jobs.tsv"
REPLAY_SH="$CURVE_DIR/replay_commands.sh"

echo -e "run_id\tsampling_mode\tactual_budget_calls\ttrain_fin_clip_count\trepeat_index\ttrain_exp_path\tcheckpoint_path" > "$PLAN_TSV"
echo -e "train_job_id\tretention_job_id\tpart2_eval_job_id\tfinal_test_job_id\trun_id\tsampling_mode\tactual_budget_calls\ttrain_fin_clip_count\trepeat_index\tcheckpoint_path" > "$SUBMITTED_TSV"
echo -e "job_id\trun_id\tcheckpoint_path\tarchive\tout_dir" > "$FINAL_TEST_TSV"
echo "#!/bin/bash" > "$REPLAY_SH"
echo "set -euo pipefail" >> "$REPLAY_SH"

WANDB_GROUP="${WANDB_GROUP_PREFIX}-${CURVE_ID}"

TRAIN_SBATCH_RESOURCE_ARGS=()
RETENTION_SBATCH_RESOURCE_ARGS=()
PART2_SBATCH_RESOURCE_ARGS=()
append_resource_args TRAIN_SBATCH_RESOURCE_ARGS "$TRAIN_PARTITION" "$TRAIN_GRES" "$TRAIN_TIME" "$TRAIN_CPUS" "$TRAIN_MEM"
append_resource_args RETENTION_SBATCH_RESOURCE_ARGS "$RETENTION_PARTITION" "$RETENTION_GRES" "$RETENTION_TIME" "$RETENTION_CPUS" "$RETENTION_MEM"
append_resource_args PART2_SBATCH_RESOURCE_ARGS "$PART2_PARTITION" "$PART2_GRES" "$PART2_TIME" "$PART2_CPUS" "$PART2_MEM"

BASELINE_FINAL_TEST_JOB_ID=""
if [[ -n "$FINAL_TEST_ARCHIVE" ]]; then
  BASELINE_FINAL_TEST_OUT="$CURVE_DIR/baseline_final_test"
  BASELINE_FINAL_TEST_CMD=(
    sbatch --parsable
    "${PART2_SBATCH_RESOURCE_ARGS[@]}"
    "$PART2_SUBMIT"
    --part2-archive "$FINAL_TEST_ARCHIVE"
    --checkpoint "$BASELINE_TEST_CHECKPOINT"
    --out-dir "$BASELINE_FINAL_TEST_OUT"
    --window-steps "$PART2_WINDOW_STEPS"
    --low-thresholds "$PART2_LOW_THRESHOLDS"
    --high-thresholds "$PART2_HIGH_THRESHOLDS"
    --min-members-values "$PART2_MIN_MEMBERS"
    --max-gap-values "$PART2_MAX_GAP_VALUES"
    --match-collar-s "$PART2_MATCH_COLLAR_S"
    --crop-size "$CROP_SIZE"
  )
  if [[ "$MERGE_EVENT_MEDIA" == "true" ]]; then
    BASELINE_FINAL_TEST_CMD+=( --merge-event-media )
  fi
  if [[ "$USE_WANDB" == "true" ]]; then
    BASELINE_FINAL_TEST_CMD+=(
      --use-wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-group "$WANDB_GROUP"
      --wandb-name-prefix "baseline_final_test"
      --wandb-tags "part2-finetune,baseline,final-test"
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
      BASELINE_FINAL_TEST_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
  fi
  printf '%q ' "${BASELINE_FINAL_TEST_CMD[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: ${BASELINE_FINAL_TEST_CMD[*]}"
  else
    BASELINE_FINAL_TEST_RAW="$("${BASELINE_FINAL_TEST_CMD[@]}")"
    BASELINE_FINAL_TEST_JOB_ID="$(normalize_job_id "$BASELINE_FINAL_TEST_RAW")"
    echo -e "${BASELINE_FINAL_TEST_JOB_ID}\tbaseline\t${BASELINE_TEST_CHECKPOINT}\t${FINAL_TEST_ARCHIVE}\t${BASELINE_FINAL_TEST_OUT}" >> "$FINAL_TEST_TSV"
    echo "Submitted baseline final test: job=$BASELINE_FINAL_TEST_JOB_ID"
  fi
fi

while IFS=$'\t' read -r RUN_ID SAMPLING_MODE ACTUAL_BUDGET_CALLS TRAIN_FIN_CLIP_COUNT REPEAT_INDEX; do
  [[ -n "$RUN_ID" ]] || continue
  if [[ -n "$INCLUDE_RUN_IDS" ]] && ! csv_contains "$INCLUDE_RUN_IDS" "$RUN_ID"; then
    continue
  fi

  RUN_SPLIT_DIR="$SPLITS_ROOT/runs/$RUN_ID"
  [[ -d "$RUN_SPLIT_DIR" ]] || { echo "Missing run split dir: $RUN_SPLIT_DIR"; exit 1; }

  RUN_DIR="$CURVE_DIR/$RUN_ID"
  TRAIN_EXP_PATH="$RUN_DIR/train"
  TRAIN_CHECKPOINT="$TRAIN_EXP_PATH/best.pt"
  RETENTION_OUT_DIR="$RUN_DIR/historical_retention"
  PART2_EVAL_OUT_DIR="$RUN_DIR/part2_eval_val"
  FINAL_TEST_OUT_DIR="$RUN_DIR/part2_eval_test"
  WANDB_NAME_BASE="p2ft_${RUN_ID}"
  WANDB_TAGS="part2-finetune,learning-curve,${SAMPLING_MODE},budget_${ACTUAL_BUDGET_CALLS}"

  echo -e "${RUN_ID}\t${SAMPLING_MODE}\t${ACTUAL_BUDGET_CALLS}\t${TRAIN_FIN_CLIP_COUNT}\t${REPEAT_INDEX}\t${TRAIN_EXP_PATH}\t${TRAIN_CHECKPOINT}" >> "$PLAN_TSV"

  TRAIN_CMD=(
    sbatch --parsable
    "${TRAIN_SBATCH_RESOURCE_ARGS[@]}"
    "$TRAIN_SUBMIT"
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

  TRAIN_DEP_PLACEHOLDER="__TRAIN_JOB_ID__"
  RETENTION_CMD=(
    sbatch --parsable
    "${RETENTION_SBATCH_RESOURCE_ARGS[@]}"
    --dependency=afterok:${TRAIN_DEP_PLACEHOLDER}
    "$TEST_SUBMIT"
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
    RETENTION_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name "${WANDB_NAME_BASE}_retention" --wandb-tags "$WANDB_TAGS,retention" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      RETENTION_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
  fi

  PART2_EVAL_CMD=(
    sbatch --parsable
    "${PART2_SBATCH_RESOURCE_ARGS[@]}"
    --dependency=afterok:${TRAIN_DEP_PLACEHOLDER}
    "$PART2_SUBMIT"
    --part2-archive "$PART2_EVAL_ARCHIVE"
    --checkpoint "$TRAIN_CHECKPOINT"
    --out-dir "$PART2_EVAL_OUT_DIR"
    --window-steps "$PART2_WINDOW_STEPS"
    --low-thresholds "$PART2_LOW_THRESHOLDS"
    --high-thresholds "$PART2_HIGH_THRESHOLDS"
    --min-members-values "$PART2_MIN_MEMBERS"
    --max-gap-values "$PART2_MAX_GAP_VALUES"
    --match-collar-s "$PART2_MATCH_COLLAR_S"
    --crop-size "$CROP_SIZE"
  )
  if [[ "$MERGE_EVENT_MEDIA" == "true" ]]; then
    PART2_EVAL_CMD+=( --merge-event-media )
  fi
  if [[ "$USE_WANDB" == "true" ]]; then
    PART2_EVAL_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name-prefix "${WANDB_NAME_BASE}_val" --wandb-tags "$WANDB_TAGS,part2-val" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      PART2_EVAL_CMD+=( --wandb-entity "$WANDB_ENTITY" )
    fi
  fi

  FINAL_TEST_CMD=()
  if [[ -n "$FINAL_TEST_ARCHIVE" && -n "$FINAL_TEST_RUN_IDS" ]] && csv_contains "$FINAL_TEST_RUN_IDS" "$RUN_ID"; then
    FINAL_TEST_CMD=(
      sbatch --parsable
      "${PART2_SBATCH_RESOURCE_ARGS[@]}"
      --dependency=afterok:${TRAIN_DEP_PLACEHOLDER}
      "$PART2_SUBMIT"
      --part2-archive "$FINAL_TEST_ARCHIVE"
      --checkpoint "$TRAIN_CHECKPOINT"
      --out-dir "$FINAL_TEST_OUT_DIR"
      --window-steps "$PART2_WINDOW_STEPS"
      --low-thresholds "$PART2_LOW_THRESHOLDS"
      --high-thresholds "$PART2_HIGH_THRESHOLDS"
      --min-members-values "$PART2_MIN_MEMBERS"
      --max-gap-values "$PART2_MAX_GAP_VALUES"
      --match-collar-s "$PART2_MATCH_COLLAR_S"
      --crop-size "$CROP_SIZE"
    )
    if [[ "$MERGE_EVENT_MEDIA" == "true" ]]; then
      FINAL_TEST_CMD+=( --merge-event-media )
    fi
    if [[ "$USE_WANDB" == "true" ]]; then
      FINAL_TEST_CMD+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" --wandb-name-prefix "${WANDB_NAME_BASE}_test" --wandb-tags "$WANDB_TAGS,part2-final-test" )
      if [[ -n "$WANDB_ENTITY" ]]; then
        FINAL_TEST_CMD+=( --wandb-entity "$WANDB_ENTITY" )
      fi
    fi
  fi

  printf '%q ' "${TRAIN_CMD[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"
  RETENTION_CMD_REPLAY=("${RETENTION_CMD[@]//${TRAIN_DEP_PLACEHOLDER}/JOBID}")
  printf '%q ' "${RETENTION_CMD_REPLAY[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"
  PART2_EVAL_CMD_REPLAY=("${PART2_EVAL_CMD[@]//${TRAIN_DEP_PLACEHOLDER}/JOBID}")
  printf '%q ' "${PART2_EVAL_CMD_REPLAY[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"
  if [[ ${#FINAL_TEST_CMD[@]} -gt 0 ]]; then
    FINAL_TEST_CMD_REPLAY=("${FINAL_TEST_CMD[@]//${TRAIN_DEP_PLACEHOLDER}/JOBID}")
    printf '%q ' "${FINAL_TEST_CMD_REPLAY[@]}" >> "$REPLAY_SH"; echo >> "$REPLAY_SH"
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: ${TRAIN_CMD[*]}"
    echo "DRY RUN: ${RETENTION_CMD[*]}"
    echo "DRY RUN: ${PART2_EVAL_CMD[*]}"
    if [[ ${#FINAL_TEST_CMD[@]} -gt 0 ]]; then
      echo "DRY RUN: ${FINAL_TEST_CMD[*]}"
    fi
    continue
  fi

  TRAIN_JOB_RAW="$("${TRAIN_CMD[@]}")"
  TRAIN_JOB_ID="$(normalize_job_id "$TRAIN_JOB_RAW")"
  RETENTION_CMD_ACTUAL=("${RETENTION_CMD[@]//${TRAIN_DEP_PLACEHOLDER}/$TRAIN_JOB_ID}")
  PART2_EVAL_CMD_ACTUAL=("${PART2_EVAL_CMD[@]//${TRAIN_DEP_PLACEHOLDER}/$TRAIN_JOB_ID}")
  RETENTION_JOB_RAW="$("${RETENTION_CMD_ACTUAL[@]}")"
  RETENTION_JOB_ID="$(normalize_job_id "$RETENTION_JOB_RAW")"
  PART2_EVAL_JOB_RAW="$("${PART2_EVAL_CMD_ACTUAL[@]}")"
  PART2_EVAL_JOB_ID="$(normalize_job_id "$PART2_EVAL_JOB_RAW")"

  FINAL_TEST_JOB_ID=""
  if [[ ${#FINAL_TEST_CMD[@]} -gt 0 ]]; then
    FINAL_TEST_CMD_ACTUAL=("${FINAL_TEST_CMD[@]//${TRAIN_DEP_PLACEHOLDER}/$TRAIN_JOB_ID}")
    FINAL_TEST_JOB_RAW="$("${FINAL_TEST_CMD_ACTUAL[@]}")"
    FINAL_TEST_JOB_ID="$(normalize_job_id "$FINAL_TEST_JOB_RAW")"
    echo -e "${FINAL_TEST_JOB_ID}\t${RUN_ID}\t${TRAIN_CHECKPOINT}\t${FINAL_TEST_ARCHIVE}\t${FINAL_TEST_OUT_DIR}" >> "$FINAL_TEST_TSV"
  fi

  echo -e "${TRAIN_JOB_ID}\t${RETENTION_JOB_ID}\t${PART2_EVAL_JOB_ID}\t${FINAL_TEST_JOB_ID}\t${RUN_ID}\t${SAMPLING_MODE}\t${ACTUAL_BUDGET_CALLS}\t${TRAIN_FIN_CLIP_COUNT}\t${REPEAT_INDEX}\t${TRAIN_CHECKPOINT}" >> "$SUBMITTED_TSV"
  echo "Submitted $RUN_ID: train=$TRAIN_JOB_ID retention=$RETENTION_JOB_ID val=$PART2_EVAL_JOB_ID${FINAL_TEST_JOB_ID:+ final_test=$FINAL_TEST_JOB_ID}"
done < <(
  python3 - "$PLAN_CSV" <<'PY'
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
if [[ -n "$FINAL_TEST_ARCHIVE" ]]; then
  echo "Final test jobs: $FINAL_TEST_TSV"
fi
