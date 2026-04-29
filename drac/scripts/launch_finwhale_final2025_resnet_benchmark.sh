#!/bin/bash
# Launch the focused final-2025 ResNet benchmark on DRAC/Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
TRAIN_SUBMIT="$REPO_ROOT/drac/scripts/submit_finwhale_cnn.sh"
PART2_SUBMIT="$REPO_ROOT/drac/scripts/submit_finwhale_part2_eval.sh"

for required in "$TRAIN_SUBMIT" "$PART2_SUBMIT"; do
  [[ -f "$required" ]] || { echo "Missing required submit script: $required"; exit 1; }
done

HISTORICAL_POS_DIR=""
HISTORICAL_NEG_DIR=""
HISTORICAL_SPLITS_DIR=""
JOINT_SPLITS_DIR=""
BASE_CHECKPOINT=""
PART2_VAL_ARCHIVE=""
PART2_TEST_ARCHIVE=""
RUN_FINAL_TEST="false"

MODEL="resnet18"
CROP_SIZE="96"
TRAIN_BATCH_SIZE="64"
EVAL_BATCH_SIZE="128"
NUM_WORKERS="4"
DEVICE="cuda"
MAIN_METRIC="f1"
CENTER_BIAS_SIGMA_FRAC="0.25"
POSITIVE_CROP_MODE="edge_mix"
MIN_GAP_SECONDS="120"

SCRATCH_LR="3e-4"
SCRATCH_EPOCHS="20"
WARMSTART_LR="1e-4"
WARMSTART_EPOCHS="10"
SEEDS_CSV="1337"

WINDOW_STEPS="24,48"
LOW_THRESHOLDS="0.70,0.75,0.80"
HIGH_THRESHOLDS="0.82,0.85,0.90"
MIN_MEMBERS_VALUES="2,3"
MAX_GAP_VALUES="auto,10,15"
MATCH_COLLAR_S="1.0"
MERGE_EVENT_MEDIA="false"

USE_WANDB="true"
WANDB_PROJECT="finwhale-resnet"
WANDB_GROUP_PREFIX="finwhale-final2025-benchmark"
WANDB_ENTITY=""

OUT_ROOT="${SCRATCH:-/scratch/$USER}/finwhale_final2025_benchmark"
BENCHMARK_ID=""
GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"

TRAIN_PARTITION=""
TRAIN_GRES=""
TRAIN_TIME=""
TRAIN_CPUS=""
TRAIN_MEM=""
EVAL_PARTITION=""
EVAL_GRES=""
EVAL_TIME=""
EVAL_CPUS=""
EVAL_MEM=""
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_final2025_resnet_benchmark.sh [options]

Required:
  --historical-pos-dir PATH
  --historical-neg-dir PATH
  --historical-splits-dir PATH
  --joint-splits-dir PATH
  --base-checkpoint PATH
  --part2-val-archive PATH

Optional:
  --part2-test-archive PATH
  --run-final-test
  --seeds CSV                      Default: 1337
  --out-root PATH                  Default: $SCRATCH/finwhale_final2025_benchmark
  --benchmark-id ID                Default: UTC timestamp
  --git-branch NAME                Default: current repo branch
  --model NAME                     Default: resnet18
  --crop-size VALUE                Default: 96
  --train-batch-size N             Default: 64
  --eval-batch-size N              Default: 128
  --num-workers N                  Default: 4
  --device NAME                    Default: cuda
  --positive-crop-mode NAME        Default: edge_mix
  --window-steps CSV               Default: 24,48
  --low-thresholds CSV             Default: 0.70,0.75,0.80
  --high-thresholds CSV            Default: 0.82,0.85,0.90
  --min-members-values CSV         Default: 2,3
  --max-gap-values CSV             Default: auto,10,15
  --match-collar-s VALUE           Default: 1.0
  --merge-event-media
  --wandb-project NAME             Default: finwhale-resnet
  --wandb-group-prefix NAME        Default: finwhale-final2025-benchmark
  --wandb-entity NAME
  --no-wandb

Recommended recipe defaults:
  joint_scratch:   lr=3e-4 epochs=20 balance=none cbs=0.25 crop_mode=edge_mix gap=120
  joint_warmstart: lr=1e-4 epochs=10 init=base-checkpoint

Resource overrides:
  --train-partition NAME
  --train-gres SPEC
  --train-time HH:MM:SS
  --train-cpus N
  --train-mem SIZE
  --eval-partition NAME
  --eval-gres SPEC
  --eval-time HH:MM:SS
  --eval-cpus N
  --eval-mem SIZE

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --historical-pos-dir) HISTORICAL_POS_DIR="$2"; shift 2 ;;
    --historical-neg-dir) HISTORICAL_NEG_DIR="$2"; shift 2 ;;
    --historical-splits-dir) HISTORICAL_SPLITS_DIR="$2"; shift 2 ;;
    --joint-splits-dir) JOINT_SPLITS_DIR="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --part2-val-archive) PART2_VAL_ARCHIVE="$2"; shift 2 ;;
    --part2-test-archive) PART2_TEST_ARCHIVE="$2"; shift 2 ;;
    --run-final-test) RUN_FINAL_TEST="true"; shift ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --benchmark-id) BENCHMARK_ID="$2"; shift 2 ;;
    --git-branch) GIT_BRANCH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --positive-crop-mode) POSITIVE_CROP_MODE="$2"; shift 2 ;;
    --window-steps) WINDOW_STEPS="$2"; shift 2 ;;
    --low-thresholds) LOW_THRESHOLDS="$2"; shift 2 ;;
    --high-thresholds) HIGH_THRESHOLDS="$2"; shift 2 ;;
    --min-members-values) MIN_MEMBERS_VALUES="$2"; shift 2 ;;
    --max-gap-values) MAX_GAP_VALUES="$2"; shift 2 ;;
    --match-collar-s) MATCH_COLLAR_S="$2"; shift 2 ;;
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
    --eval-partition) EVAL_PARTITION="$2"; shift 2 ;;
    --eval-gres) EVAL_GRES="$2"; shift 2 ;;
    --eval-time) EVAL_TIME="$2"; shift 2 ;;
    --eval-cpus) EVAL_CPUS="$2"; shift 2 ;;
    --eval-mem) EVAL_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$HISTORICAL_POS_DIR" || -z "$HISTORICAL_NEG_DIR" || -z "$HISTORICAL_SPLITS_DIR" || -z "$JOINT_SPLITS_DIR" || -z "$BASE_CHECKPOINT" || -z "$PART2_VAL_ARCHIVE" ]]; then
  echo "Error: missing one or more required arguments"
  usage
  exit 1
fi
if [[ "$RUN_FINAL_TEST" == "true" && -z "$PART2_TEST_ARCHIVE" ]]; then
  echo "Error: --run-final-test requires --part2-test-archive"
  exit 1
fi

if [[ -z "$BENCHMARK_ID" ]]; then
  BENCHMARK_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

readarray -t SEED_LIST < <(split_csv "$SEEDS_CSV")
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "Error: --seeds resolved to an empty list"
  exit 1
fi

BENCHMARK_DIR="$OUT_ROOT/$BENCHMARK_ID"
RUNS_DIR="$BENCHMARK_DIR/runs"
mkdir -p "$RUNS_DIR"

PLAN_TSV="$BENCHMARK_DIR/plan.tsv"
SUBMITTED_TSV="$BENCHMARK_DIR/submitted_jobs.tsv"
REPLAY_SH="$BENCHMARK_DIR/replay_commands.sh"
WANDB_GROUP_BASE="${WANDB_GROUP_PREFIX}-${BENCHMARK_ID}"

echo -e "phase\trecipe\tseed\tcheckpoint_path\ttrain_dir\teval_archive\teval_split\tout_dir\twandb_group" > "$PLAN_TSV"
echo -e "phase\trecipe\tseed\tjob_id\tdepends_on\tcheckpoint_path\tout_dir\twandb_group" > "$SUBMITTED_TSV"
echo "#!/bin/bash" > "$REPLAY_SH"
echo "set -euo pipefail" >> "$REPLAY_SH"

TRAIN_SBATCH_RESOURCE_ARGS=()
EVAL_SBATCH_RESOURCE_ARGS=()
append_resource_args TRAIN_SBATCH_RESOURCE_ARGS "$TRAIN_PARTITION" "$TRAIN_GRES" "$TRAIN_TIME" "$TRAIN_CPUS" "$TRAIN_MEM"
append_resource_args EVAL_SBATCH_RESOURCE_ARGS "$EVAL_PARTITION" "$EVAL_GRES" "$EVAL_TIME" "$EVAL_CPUS" "$EVAL_MEM"

submit_or_echo() {
  local -n cmd_ref="$1"
  if [[ "$DRY_RUN" == "true" ]]; then
    printf 'DRYRUN'
    return
  fi
  local raw
  raw="$("${cmd_ref[@]}")"
  normalize_job_id "$raw"
}

record_replay() {
  local -n cmd_ref="$1"
  printf '%q ' "${cmd_ref[@]}" >> "$REPLAY_SH"
  printf '\n' >> "$REPLAY_SH"
}

submit_eval_job() {
  local phase="$1"
  local recipe="$2"
  local seed="$3"
  local dependency="$4"
  local checkpoint_path="$5"
  local archive_path="$6"
  local baseline_split="$7"
  local out_dir="$8"
  local wandb_group="$9"
  local wandb_name_prefix="${10}"
  local wandb_tags="${11}"

  local dep_value="$dependency"
  if [[ "$dep_value" == "DRYRUN" ]]; then
    dep_value="JOBID"
  fi
  local dep_args=()
  if [[ -n "$dep_value" ]]; then
    dep_args=( --dependency "afterok:${dep_value}" )
  fi

  local cmd=(
    sbatch --parsable
    "${dep_args[@]}"
    "${EVAL_SBATCH_RESOURCE_ARGS[@]}"
    "$PART2_SUBMIT"
    --part2-archive "$archive_path"
    --checkpoint "$checkpoint_path"
    --out-dir "$out_dir"
    --window-steps "$WINDOW_STEPS"
    --low-thresholds "$LOW_THRESHOLDS"
    --high-thresholds "$HIGH_THRESHOLDS"
    --min-members-values "$MIN_MEMBERS_VALUES"
    --max-gap-values "$MAX_GAP_VALUES"
    --match-collar-s "$MATCH_COLLAR_S"
    --baseline-pos-dir "$HISTORICAL_POS_DIR"
    --baseline-neg-dir "$HISTORICAL_NEG_DIR"
    --baseline-splits-dir "$HISTORICAL_SPLITS_DIR"
    --baseline-eval-split "$baseline_split"
    --batch-size "$EVAL_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device "$DEVICE"
    --crop-size "$CROP_SIZE"
  )
  if [[ "$MERGE_EVENT_MEDIA" == "true" ]]; then
    cmd+=( --merge-event-media )
  fi
  if [[ "$USE_WANDB" == "true" ]]; then
    cmd+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$wandb_group" --wandb-name-prefix "$wandb_name_prefix" --wandb-tags "$wandb_tags" )
    if [[ -n "$WANDB_ENTITY" ]]; then
      cmd+=( --wandb-entity "$WANDB_ENTITY" )
    fi
  fi

  record_replay cmd
  local job_id
  job_id="$(submit_or_echo cmd)"
  echo -e "${phase}\t${recipe}\t${seed}\t${job_id}\t${dependency}\t${checkpoint_path}\t${out_dir}\t${wandb_group}" >> "$SUBMITTED_TSV"
  echo "[${phase}] recipe=${recipe} seed=${seed} job_id=${job_id} out=${out_dir}"
}

BASELINE_VAL_DIR="$BENCHMARK_DIR/baseline_val"
BASELINE_VAL_GROUP="${WANDB_GROUP_BASE}-baseline"
echo -e "baseline_val\tbaseline\tbase\t${BASE_CHECKPOINT}\t\t${PART2_VAL_ARCHIVE}\tval\t${BASELINE_VAL_DIR}\t${BASELINE_VAL_GROUP}" >> "$PLAN_TSV"
submit_eval_job \
  "baseline_val" \
  "baseline" \
  "base" \
  "" \
  "$BASE_CHECKPOINT" \
  "$PART2_VAL_ARCHIVE" \
  "val" \
  "$BASELINE_VAL_DIR" \
  "$BASELINE_VAL_GROUP" \
  "baseline_val" \
  "final2025-benchmark,baseline,val"

if [[ "$RUN_FINAL_TEST" == "true" ]]; then
  BASELINE_TEST_DIR="$BENCHMARK_DIR/baseline_test"
  echo -e "baseline_test\tbaseline\tbase\t${BASE_CHECKPOINT}\t\t${PART2_TEST_ARCHIVE}\ttest\t${BASELINE_TEST_DIR}\t${BASELINE_VAL_GROUP}" >> "$PLAN_TSV"
  submit_eval_job \
    "baseline_test" \
    "baseline" \
    "base" \
    "" \
    "$BASE_CHECKPOINT" \
    "$PART2_TEST_ARCHIVE" \
    "test" \
    "$BASELINE_TEST_DIR" \
    "$BASELINE_VAL_GROUP" \
    "baseline_test" \
    "final2025-benchmark,baseline,test"
fi

for seed in "${SEED_LIST[@]}"; do
  for recipe in joint_scratch joint_warmstart; do
    run_dir="$RUNS_DIR/${recipe}_seed${seed}"
    train_dir="$run_dir/train"
    val_eval_dir="$run_dir/val_eval"
    test_eval_dir="$run_dir/test_eval"
    mkdir -p "$train_dir" "$val_eval_dir"

    lr="$SCRATCH_LR"
    epochs="$SCRATCH_EPOCHS"
    init_checkpoint=""
    if [[ "$recipe" == "joint_warmstart" ]]; then
      lr="$WARMSTART_LR"
      epochs="$WARMSTART_EPOCHS"
      init_checkpoint="$BASE_CHECKPOINT"
    fi

    checkpoint_path="$train_dir/best.pt"
    wandb_group="${WANDB_GROUP_BASE}-${recipe}"
    wandb_name="${recipe}_seed${seed}_train"
    wandb_tags="final2025-benchmark,${recipe},train"

    echo -e "train\t${recipe}\t${seed}\t${checkpoint_path}\t${train_dir}\t\t\t${train_dir}\t${wandb_group}" >> "$PLAN_TSV"

    train_cmd=(
      sbatch --parsable
      "${TRAIN_SBATCH_RESOURCE_ARGS[@]}"
      "$TRAIN_SUBMIT"
      --pos-dir "$HISTORICAL_POS_DIR"
      --neg-dir "$HISTORICAL_NEG_DIR"
      --splits-dir "$JOINT_SPLITS_DIR"
      --no-copy
      --git-branch "$GIT_BRANCH"
      --model "$MODEL"
      --batch-size "$TRAIN_BATCH_SIZE"
      --epochs "$epochs"
      --num-workers "$NUM_WORKERS"
      --lr "$lr"
      --balance none
      --crop-size "$CROP_SIZE"
      --device "$DEVICE"
      --exp-dir "$train_dir"
      --seed "$seed"
      --split-strategy time_separated
      --min-gap-seconds "$MIN_GAP_SECONDS"
      --main-metric "$MAIN_METRIC"
      --center-bias-sigma-frac "$CENTER_BIAS_SIGMA_FRAC"
      --positive-crop-mode "$POSITIVE_CROP_MODE"
      --run-tag "${recipe}_seed${seed}"
    )
    if [[ -n "$init_checkpoint" ]]; then
      train_cmd+=( --init-checkpoint "$init_checkpoint" )
    fi
    if [[ "$USE_WANDB" == "true" ]]; then
      train_cmd+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$wandb_group" --wandb-name "$wandb_name" --wandb-tags "$wandb_tags" )
      if [[ -n "$WANDB_ENTITY" ]]; then
        train_cmd+=( --wandb-entity "$WANDB_ENTITY" )
      fi
    fi

    record_replay train_cmd
    train_job_id="$(submit_or_echo train_cmd)"
    echo -e "train\t${recipe}\t${seed}\t${train_job_id}\t\t${checkpoint_path}\t${train_dir}\t${wandb_group}" >> "$SUBMITTED_TSV"
    echo "[train] recipe=${recipe} seed=${seed} job_id=${train_job_id} dir=${train_dir}"

    echo -e "val_eval\t${recipe}\t${seed}\t${checkpoint_path}\t${train_dir}\t${PART2_VAL_ARCHIVE}\tval\t${val_eval_dir}\t${wandb_group}" >> "$PLAN_TSV"
    submit_eval_job \
      "val_eval" \
      "$recipe" \
      "$seed" \
      "$train_job_id" \
      "$checkpoint_path" \
      "$PART2_VAL_ARCHIVE" \
      "val" \
      "$val_eval_dir" \
      "$wandb_group" \
      "${recipe}_seed${seed}_val" \
      "final2025-benchmark,${recipe},val"

    if [[ "$RUN_FINAL_TEST" == "true" ]]; then
      mkdir -p "$test_eval_dir"
      echo -e "test_eval\t${recipe}\t${seed}\t${checkpoint_path}\t${train_dir}\t${PART2_TEST_ARCHIVE}\ttest\t${test_eval_dir}\t${wandb_group}" >> "$PLAN_TSV"
      submit_eval_job \
        "test_eval" \
        "$recipe" \
        "$seed" \
        "$train_job_id" \
        "$checkpoint_path" \
        "$PART2_TEST_ARCHIVE" \
        "test" \
        "$test_eval_dir" \
        "$wandb_group" \
        "${recipe}_seed${seed}_test" \
        "final2025-benchmark,${recipe},test"
    fi
  done
done

chmod +x "$REPLAY_SH"

echo ""
echo "Benchmark prepared: $BENCHMARK_DIR"
echo "Plan file: $PLAN_TSV"
echo "Submitted jobs: $SUBMITTED_TSV"
echo "Replay commands: $REPLAY_SH"
echo ""
echo "Summarize with:"
echo "  python $REPO_ROOT/drac/scripts/summarize_finwhale_final2025_benchmark.py --benchmark-dir $BENCHMARK_DIR"
