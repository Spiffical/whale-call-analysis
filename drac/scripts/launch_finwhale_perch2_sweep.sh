#!/bin/bash
# Launch a structured Perch2 embedding sweep.
#
# Run from login node:
#   bash drac/scripts/launch_finwhale_perch2_sweep.sh \
#     --audio-dir /path/to/audio \
#     --dry-run

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
SUBMIT_SCRIPT="$REPO_ROOT/drac/scripts/submit_finwhale_perch2_embeddings.sh"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 1
fi

AUDIO_DIR=""
declare -a EXCEL_FILES=()
PERCH_MODEL="perch_v2_gpu"
BATCH_SIZE=16
CONTEXT_SECONDS="40"
TRAIN_CLIP_SECONDS="10"
EVAL_CLIP_SECONDS="10"
ASSUMED_CLIP_DURATION_SECONDS="300"
NEGATIVES_PER_POSITIVE=1
NEGATIVE_MARGIN_SECONDS="2"
MAX_POSITIVES=""
MAX_AUDIO_FILES=""
TRAIN_RATIO="0.8"
VAL_RATIO="0.1"
SEEDS_CSV="42"
LOGREG_C_LIST_CSV="1.0"
CENTER_BIAS_LIST_CSV="0.25"
MIN_GAP_LIST_CSV="120"
TRAIN_POS_AUGMENT_COPIES=1
TRAIN_NEG_AUGMENT_COPIES=1
MAX_ITER=3000
DISABLE_GPU="false"
SKIP_SAVE_EMBEDDINGS="true"
COPY_AUDIO_TO_TMP="false"
INSTALL_PERCH_DEPS="false"
EXP_ROOT="${SCRATCH:-/scratch/$USER}/finwhale_perch2_sweeps"
SWEEP_ID=""
RUN_TAG_PREFIX="perch2"
NOTE_PREFIX=""
PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
LOCAL_TEST_MODE="false"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_perch2_sweep.sh [options]

Required:
  --audio-dir PATH

Optional:
  --excel-file PATH               Repeatable. If omitted, submit script defaults are used.
  --excel-files-csv CSV           Comma-separated Excel paths.
  --perch-model NAME              (default: perch_v2_gpu)
  --batch-size N                  (default: 16)
  --context-seconds SEC           (default: 40)
  --train-clip-seconds SEC        (default: 10)
  --eval-clip-seconds SEC         (default: 10)
  --assumed-clip-duration-seconds SEC   (default: 300)
  --negatives-per-positive N      (default: 1)
  --negative-margin-seconds SEC   (default: 2)
  --max-positives N
  --max-audio-files N
  --train-ratio R                 (default: 0.8)
  --val-ratio R                   (default: 0.1)
  --seeds CSV                     (default: 42)
  --logreg-c-list CSV             (default: 1.0)
  --center-bias-list CSV          (default: 0.25)
  --min-gap-list CSV              (default: 120)
  --train-pos-augment-copies N    (default: 1)
  --train-neg-augment-copies N    (default: 1)
  --max-iter N                    (default: 3000)
  --disable-gpu
  --no-skip-save-embeddings       Save embeddings.npz in sweep runs.
  --copy-audio-to-tmp
  --install-perch-deps
  --exp-root PATH                 (default: $SCRATCH/finwhale_perch2_sweeps)
  --sweep-id ID                   (default: UTC timestamp)
  --run-tag-prefix TAG            (default: perch2)
  --note-prefix TEXT
  --project-path PATH
  --venv-path PATH
  --local-test-mode               Run submit script directly instead of sbatch.
  --dry-run
  -h, --help
USAGE
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

append_excel_csv() {
  local raw="$1"
  while IFS= read -r v; do
    EXCEL_FILES+=("$v")
  done < <(split_csv "$raw")
}

to_tag() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//+/}"
  echo "$v" | tr -cs '[:alnum:]_-' '_' | sed 's/^_//;s/_$//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --excel-file) EXCEL_FILES+=("$2"); shift 2 ;;
    --excel-files-csv) append_excel_csv "$2"; shift 2 ;;
    --perch-model) PERCH_MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --context-seconds) CONTEXT_SECONDS="$2"; shift 2 ;;
    --train-clip-seconds) TRAIN_CLIP_SECONDS="$2"; shift 2 ;;
    --eval-clip-seconds) EVAL_CLIP_SECONDS="$2"; shift 2 ;;
    --assumed-clip-duration-seconds) ASSUMED_CLIP_DURATION_SECONDS="$2"; shift 2 ;;
    --negatives-per-positive) NEGATIVES_PER_POSITIVE="$2"; shift 2 ;;
    --negative-margin-seconds) NEGATIVE_MARGIN_SECONDS="$2"; shift 2 ;;
    --max-positives) MAX_POSITIVES="$2"; shift 2 ;;
    --max-audio-files) MAX_AUDIO_FILES="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --logreg-c-list) LOGREG_C_LIST_CSV="$2"; shift 2 ;;
    --center-bias-list) CENTER_BIAS_LIST_CSV="$2"; shift 2 ;;
    --min-gap-list) MIN_GAP_LIST_CSV="$2"; shift 2 ;;
    --train-pos-augment-copies) TRAIN_POS_AUGMENT_COPIES="$2"; shift 2 ;;
    --train-neg-augment-copies) TRAIN_NEG_AUGMENT_COPIES="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --disable-gpu) DISABLE_GPU="true"; shift ;;
    --no-skip-save-embeddings) SKIP_SAVE_EMBEDDINGS="false"; shift ;;
    --copy-audio-to-tmp) COPY_AUDIO_TO_TMP="true"; shift ;;
    --install-perch-deps) INSTALL_PERCH_DEPS="true"; shift ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --sweep-id) SWEEP_ID="$2"; shift 2 ;;
    --run-tag-prefix) RUN_TAG_PREFIX="$2"; shift 2 ;;
    --note-prefix) NOTE_PREFIX="$2"; shift 2 ;;
    --project-path) PROJECT_PATH="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --local-test-mode) LOCAL_TEST_MODE="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$AUDIO_DIR" ]]; then
  echo "Error: --audio-dir is required"
  exit 1
fi
if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "Error: audio directory does not exist: $AUDIO_DIR"
  exit 1
fi

if [[ -z "$SWEEP_ID" ]]; then
  SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

readarray -t SEED_LIST < <(split_csv "$SEEDS_CSV")
readarray -t LOGREG_C_LIST < <(split_csv "$LOGREG_C_LIST_CSV")
readarray -t CENTER_BIAS_LIST < <(split_csv "$CENTER_BIAS_LIST_CSV")
readarray -t MIN_GAP_LIST < <(split_csv "$MIN_GAP_LIST_CSV")

if [[ ${#SEED_LIST[@]} -eq 0 || ${#LOGREG_C_LIST[@]} -eq 0 || ${#CENTER_BIAS_LIST[@]} -eq 0 || ${#MIN_GAP_LIST[@]} -eq 0 ]]; then
  echo "Error: one of seeds/logreg-c/center-bias/min-gap lists is empty."
  exit 1
fi

SWEEP_DIR="$EXP_ROOT/$SWEEP_ID"
RUNS_DIR="$SWEEP_DIR/runs"
mkdir -p "$RUNS_DIR"

PLAN_TSV="$SWEEP_DIR/plan.tsv"
SUBMITTED_TSV="$SWEEP_DIR/submitted_jobs.tsv"
REPLAY_SH="$SWEEP_DIR/replay_commands.sh"

echo -e "run_slug\tperch_model\tlogreg_c\tcenter_bias_sigma_frac\tmin_gap_seconds\tseed\trun_exp_dir" > "$PLAN_TSV"
echo -e "job_id\trun_slug\tperch_model\tlogreg_c\tcenter_bias_sigma_frac\tmin_gap_seconds\tseed\trun_exp_dir" > "$SUBMITTED_TSV"
echo "#!/bin/bash" > "$REPLAY_SH"
echo "set -euo pipefail" >> "$REPLAY_SH"

RUN_COUNT=0
for seed in "${SEED_LIST[@]}"; do
  for logreg_c in "${LOGREG_C_LIST[@]}"; do
    for cbs in "${CENTER_BIAS_LIST[@]}"; do
      for gap in "${MIN_GAP_LIST[@]}"; do
        RUN_COUNT=$((RUN_COUNT + 1))
        run_slug=$(printf "r%03d_%s_lr%s_cbs%s_gap%s_s%s" \
          "$RUN_COUNT" \
          "$(to_tag "$RUN_TAG_PREFIX")" \
          "$(to_tag "$logreg_c")" \
          "$(to_tag "$cbs")" \
          "$(to_tag "$gap")" \
          "$(to_tag "$seed")")

        run_exp_dir="$RUNS_DIR/$run_slug"
        mkdir -p "$run_exp_dir"

        echo -e "${run_slug}\t${PERCH_MODEL}\t${logreg_c}\t${cbs}\t${gap}\t${seed}\t${run_exp_dir}" >> "$PLAN_TSV"

        cmd=()
        if [[ "$LOCAL_TEST_MODE" == "true" ]]; then
          cmd+=( bash "$SUBMIT_SCRIPT" --local-test-mode )
        else
          cmd+=( sbatch --parsable "$SUBMIT_SCRIPT" )
        fi

        cmd+=(
          --audio-dir "$AUDIO_DIR"
          --perch-model "$PERCH_MODEL"
          --batch-size "$BATCH_SIZE"
          --context-seconds "$CONTEXT_SECONDS"
          --train-clip-seconds "$TRAIN_CLIP_SECONDS"
          --eval-clip-seconds "$EVAL_CLIP_SECONDS"
          --assumed-clip-duration-seconds "$ASSUMED_CLIP_DURATION_SECONDS"
          --negatives-per-positive "$NEGATIVES_PER_POSITIVE"
          --negative-margin-seconds "$NEGATIVE_MARGIN_SECONDS"
          --train-ratio "$TRAIN_RATIO"
          --val-ratio "$VAL_RATIO"
          --min-gap-seconds "$gap"
          --logreg-c "$logreg_c"
          --max-iter "$MAX_ITER"
          --train-pos-augment-copies "$TRAIN_POS_AUGMENT_COPIES"
          --train-neg-augment-copies "$TRAIN_NEG_AUGMENT_COPIES"
          --center-bias-sigma-frac "$cbs"
          --seed "$seed"
          --run-tag "$run_slug"
          --exp-dir "$run_exp_dir"
          --project-path "$PROJECT_PATH"
          --venv-path "$VENV_PATH"
        )

        if [[ -n "$MAX_POSITIVES" ]]; then
          cmd+=( --max-positives "$MAX_POSITIVES" )
        fi
        if [[ -n "$MAX_AUDIO_FILES" ]]; then
          cmd+=( --max-audio-files "$MAX_AUDIO_FILES" )
        fi
        if [[ "$DISABLE_GPU" == "true" ]]; then
          cmd+=( --disable-gpu )
        fi
        if [[ "$SKIP_SAVE_EMBEDDINGS" == "true" ]]; then
          cmd+=( --skip-save-embeddings )
        fi
        if [[ "$COPY_AUDIO_TO_TMP" == "true" ]]; then
          cmd+=( --copy-audio-to-tmp )
        fi
        if [[ "$INSTALL_PERCH_DEPS" == "true" ]]; then
          cmd+=( --install-perch-deps )
        fi
        if [[ -n "$NOTE_PREFIX" ]]; then
          cmd+=( --note "${NOTE_PREFIX}:${run_slug}" )
        fi
        for excel_path in "${EXCEL_FILES[@]}"; do
          cmd+=( --excel-file "$excel_path" )
        done

        {
          printf '%q ' "${cmd[@]}"
          printf '\n'
        } >> "$REPLAY_SH"

        if [[ "$DRY_RUN" == "true" ]]; then
          job_id="DRYRUN_${RUN_COUNT}"
          echo "[dry-run] $job_id $run_slug"
        else
          if [[ "$LOCAL_TEST_MODE" == "true" ]]; then
            "${cmd[@]}"
            job_id="LOCAL_${RUN_COUNT}"
            echo "[local] run_slug=${run_slug}"
          else
            sbatch_out="$("${cmd[@]}")"
            job_id="${sbatch_out%%;*}"
            echo "[submitted] job_id=${job_id} run_slug=${run_slug}"
          fi
        fi

        echo -e "${job_id}\t${run_slug}\t${PERCH_MODEL}\t${logreg_c}\t${cbs}\t${gap}\t${seed}\t${run_exp_dir}" >> "$SUBMITTED_TSV"
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
