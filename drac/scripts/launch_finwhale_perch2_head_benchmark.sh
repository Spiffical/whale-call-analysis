#!/bin/bash
# Launch a focused benchmark of classifier heads on top of fixed Perch2 embeddings.
#
# This benchmarks a small family of heads on the same prebuilt Perch context dataset:
# - logreg baseline
# - MLP small / medium / large
# - residual dense ("ResNet-style") small / medium / large
#
# Run from a login node, e.g.:
#   bash drac/scripts/launch_finwhale_perch2_head_benchmark.sh \
#     --context-dataset-tar /project/rpp-kmoran/merileo/data/finwhales/perch2_context_dataset_20260310T210754Z.tar.zst \
#     --container-image "$SCRATCH/whale-call-analysis/containers/tensorflow_2.20.0_gpu.sif" \
#     --container-venv-path "$SCRATCH/whale-call-analysis/venvs/perch2_tf220" \
#     --apptainer-module apptainer \
#     --use-wandb

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"
SUBMIT_SCRIPT="$REPO_ROOT/drac/scripts/submit_finwhale_perch2_embeddings.sh"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 1
fi

CONTEXT_DATASET_TAR=""
CONTEXT_DATASET_DIR=""
CONTEXT_MANIFEST_RELPATH="context_window_manifest.csv"
CONTEXT_AUDIO_RELDIR="context_audio"
COPY_CONTEXT_DIR_TO_TMP="true"

USE_WANDB="false"
WANDB_PROJECT="finwhale_perch2"
WANDB_GROUP_PREFIX="finwhale-perch2-head-benchmark"
WANDB_ENTITY=""

PERCH_MODEL="perch_v2_gpu"
BATCH_SIZE=16
CONTEXT_SECONDS="40"
TRAIN_CLIP_SECONDS="10"
EVAL_CLIP_SECONDS="10"
TRAIN_RATIO="0.8"
VAL_RATIO="0.1"
MIN_GAP_SECONDS="120"
TRAIN_POS_AUGMENT_COPIES=1
TRAIN_NEG_AUGMENT_COPIES=1
CENTER_BIAS_SIGMA_FRAC="0.25"
SEEDS_CSV="42"

HEAD_SPECS_CSV="logreg,mlp_small,mlp_medium,mlp_large,resnet_small,resnet_medium,resnet_large"
EPOCHS=30
HEAD_BATCH_SIZE=256
EARLY_STOPPING_PATIENCE=5
MAX_ITER=3000
DISABLE_GPU="false"
SKIP_SAVE_EMBEDDINGS="true"
INSTALL_PERCH_DEPS="false"

EXP_ROOT="${SCRATCH:-/scratch/$USER}/whale-call-analysis/perch2_head_benchmarks"
SWEEP_ID=""
RUN_TAG_PREFIX="perch2-head-benchmark"
NOTE_PREFIX=""
PROJECT_PATH="${PROJECT_PATH:-$REPO_ROOT}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
PYTHON_MODULE=""
CONTAINER_IMAGE=""
CONTAINER_VENV_PATH=""
APPTAINER_MODULE=""
LOCAL_TEST_MODE="false"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/launch_finwhale_perch2_head_benchmark.sh [options]

Required (choose one):
  --context-dataset-tar PATH
  --context-dataset-dir PATH

Optional:
  --context-manifest-relpath PATH  (default: context_window_manifest.csv)
  --context-audio-reldir PATH      (default: context_audio)
  --no-copy-context-dir
  --use-wandb
  --wandb-project NAME             (default: finwhale_perch2)
  --wandb-group-prefix NAME        (default: finwhale-perch2-head-benchmark)
  --wandb-entity NAME
  --perch-model NAME               (default: perch_v2_gpu)
  --batch-size N                   (default: 16)
  --context-seconds SEC            (default: 40)
  --train-clip-seconds SEC         (default: 10)
  --eval-clip-seconds SEC          (default: 10)
  --train-ratio R                  (default: 0.8)
  --val-ratio R                    (default: 0.1)
  --min-gap-seconds SEC            (default: 120)
  --train-pos-augment-copies N     (default: 1)
  --train-neg-augment-copies N     (default: 1)
  --center-bias-sigma-frac VALUE   (default: 0.25)
  --seeds CSV                      (default: 42)
  --head-specs CSV                 (default: logreg,mlp_small,mlp_medium,mlp_large,resnet_small,resnet_medium,resnet_large)
  --epochs N                       (default: 30)
  --head-batch-size N              (default: 256)
  --early-stopping-patience N      (default: 5)
  --max-iter N                     (default: 3000)
  --disable-gpu
  --no-skip-save-embeddings
  --install-perch-deps
  --exp-root PATH                  (default: $SCRATCH/whale-call-analysis/perch2_head_benchmarks)
  --sweep-id ID                    (default: UTC timestamp)
  --run-tag-prefix TAG             (default: perch2-head-benchmark)
  --note-prefix TEXT
  --project-path PATH
  --venv-path PATH
  --python-module NAME
  --container-image PATH
  --container-venv-path PATH
  --apptainer-module NAME
  --local-test-mode
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

to_tag() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//+/}"
  echo "$v" | tr -cs '[:alnum:]_-' '_' | sed 's/^_//;s/_$//'
}

build_head_args() {
  local spec="$1"
  case "$spec" in
    logreg)
      printf '%s\n' \
        --classifier-head logreg \
        --logreg-c 1.0 \
        --max-iter "$MAX_ITER"
      ;;
    mlp_small)
      printf '%s\n' \
        --classifier-head mlp \
        --epochs "$EPOCHS" \
        --mlp-batch-size "$HEAD_BATCH_SIZE" \
        --mlp-hidden-dims 256 \
        --mlp-dropout 0.10 \
        --mlp-learning-rate 0.001 \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      ;;
    mlp_medium)
      printf '%s\n' \
        --classifier-head mlp \
        --epochs "$EPOCHS" \
        --mlp-batch-size "$HEAD_BATCH_SIZE" \
        --mlp-hidden-dims 512,128 \
        --mlp-dropout 0.20 \
        --mlp-learning-rate 0.001 \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      ;;
    mlp_large)
      printf '%s\n' \
        --classifier-head mlp \
        --epochs "$EPOCHS" \
        --mlp-batch-size "$HEAD_BATCH_SIZE" \
        --mlp-hidden-dims 1024,512,256 \
        --mlp-dropout 0.30 \
        --mlp-learning-rate 0.0005 \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      ;;
    resnet_small)
      printf '%s\n' \
        --classifier-head resnet \
        --epochs "$EPOCHS" \
        --mlp-batch-size "$HEAD_BATCH_SIZE" \
        --resnet-width 256 \
        --resnet-blocks 2 \
        --mlp-dropout 0.10 \
        --mlp-learning-rate 0.001 \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      ;;
    resnet_medium)
      printf '%s\n' \
        --classifier-head resnet \
        --epochs "$EPOCHS" \
        --mlp-batch-size "$HEAD_BATCH_SIZE" \
        --resnet-width 512 \
        --resnet-blocks 3 \
        --mlp-dropout 0.20 \
        --mlp-learning-rate 0.001 \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      ;;
    resnet_large)
      printf '%s\n' \
        --classifier-head resnet \
        --epochs "$EPOCHS" \
        --mlp-batch-size "$HEAD_BATCH_SIZE" \
        --resnet-width 1024 \
        --resnet-blocks 4 \
        --mlp-dropout 0.30 \
        --mlp-learning-rate 0.0005 \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      ;;
    *)
      echo "Error: unknown head spec '$spec'" >&2
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context-dataset-tar) CONTEXT_DATASET_TAR="$2"; shift 2 ;;
    --context-dataset-dir) CONTEXT_DATASET_DIR="$2"; shift 2 ;;
    --context-manifest-relpath) CONTEXT_MANIFEST_RELPATH="$2"; shift 2 ;;
    --context-audio-reldir) CONTEXT_AUDIO_RELDIR="$2"; shift 2 ;;
    --no-copy-context-dir) COPY_CONTEXT_DIR_TO_TMP="false"; shift ;;
    --use-wandb) USE_WANDB="true"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group-prefix) WANDB_GROUP_PREFIX="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --perch-model) PERCH_MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --context-seconds) CONTEXT_SECONDS="$2"; shift 2 ;;
    --train-clip-seconds) TRAIN_CLIP_SECONDS="$2"; shift 2 ;;
    --eval-clip-seconds) EVAL_CLIP_SECONDS="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --min-gap-seconds) MIN_GAP_SECONDS="$2"; shift 2 ;;
    --train-pos-augment-copies) TRAIN_POS_AUGMENT_COPIES="$2"; shift 2 ;;
    --train-neg-augment-copies) TRAIN_NEG_AUGMENT_COPIES="$2"; shift 2 ;;
    --center-bias-sigma-frac) CENTER_BIAS_SIGMA_FRAC="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --head-specs) HEAD_SPECS_CSV="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --head-batch-size) HEAD_BATCH_SIZE="$2"; shift 2 ;;
    --early-stopping-patience) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --disable-gpu) DISABLE_GPU="true"; shift ;;
    --no-skip-save-embeddings) SKIP_SAVE_EMBEDDINGS="false"; shift ;;
    --install-perch-deps) INSTALL_PERCH_DEPS="true"; shift ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --sweep-id) SWEEP_ID="$2"; shift 2 ;;
    --run-tag-prefix) RUN_TAG_PREFIX="$2"; shift 2 ;;
    --note-prefix) NOTE_PREFIX="$2"; shift 2 ;;
    --project-path) PROJECT_PATH="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --python-module) PYTHON_MODULE="$2"; shift 2 ;;
    --container-image) CONTAINER_IMAGE="$2"; shift 2 ;;
    --container-venv-path) CONTAINER_VENV_PATH="$2"; shift 2 ;;
    --apptainer-module) APPTAINER_MODULE="$2"; shift 2 ;;
    --local-test-mode) LOCAL_TEST_MODE="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$CONTEXT_DATASET_TAR" && -z "$CONTEXT_DATASET_DIR" ]]; then
  echo "Error: provide one of --context-dataset-tar or --context-dataset-dir"
  exit 1
fi
if [[ -n "$CONTEXT_DATASET_TAR" && -n "$CONTEXT_DATASET_DIR" ]]; then
  echo "Error: --context-dataset-tar and --context-dataset-dir are mutually exclusive"
  exit 1
fi
if [[ -z "$SWEEP_ID" ]]; then
  SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

readarray -t SEED_LIST < <(split_csv "$SEEDS_CSV")
readarray -t HEAD_SPEC_LIST < <(split_csv "$HEAD_SPECS_CSV")
if [[ ${#SEED_LIST[@]} -eq 0 || ${#HEAD_SPEC_LIST[@]} -eq 0 ]]; then
  echo "Error: seeds or head-specs list is empty."
  exit 1
fi

SWEEP_DIR="$EXP_ROOT/$SWEEP_ID"
RUNS_DIR="$SWEEP_DIR/runs"
mkdir -p "$RUNS_DIR"
WANDB_GROUP_BASE="${WANDB_GROUP_PREFIX}-${SWEEP_ID}"

PLAN_TSV="$SWEEP_DIR/plan.tsv"
SUBMITTED_TSV="$SWEEP_DIR/submitted_jobs.tsv"
REPLAY_SH="$SWEEP_DIR/replay_commands.sh"

echo -e "run_slug\thead_spec\tperch_model\tseed\trun_exp_dir" > "$PLAN_TSV"
echo -e "job_id\trun_slug\thead_spec\tperch_model\tseed\trun_exp_dir" > "$SUBMITTED_TSV"
echo "#!/bin/bash" > "$REPLAY_SH"
echo "set -euo pipefail" >> "$REPLAY_SH"

RUN_COUNT=0
for seed in "${SEED_LIST[@]}"; do
  for head_spec in "${HEAD_SPEC_LIST[@]}"; do
    RUN_COUNT=$((RUN_COUNT + 1))
    run_slug=$(printf "r%03d_%s_%s_s%s" \
      "$RUN_COUNT" \
      "$(to_tag "$RUN_TAG_PREFIX")" \
      "$(to_tag "$head_spec")" \
      "$(to_tag "$seed")")

    run_exp_dir="$RUNS_DIR/$run_slug"
    mkdir -p "$run_exp_dir"

    echo -e "${run_slug}\t${head_spec}\t${PERCH_MODEL}\t${seed}\t${run_exp_dir}" >> "$PLAN_TSV"

    cmd=()
    if [[ "$LOCAL_TEST_MODE" == "true" ]]; then
      cmd+=( bash "$SUBMIT_SCRIPT" --local-test-mode )
    else
      cmd+=( sbatch --parsable "$SUBMIT_SCRIPT" )
    fi

    cmd+=(
      --context-manifest-relpath "$CONTEXT_MANIFEST_RELPATH"
      --context-audio-reldir "$CONTEXT_AUDIO_RELDIR"
      --perch-model "$PERCH_MODEL"
      --batch-size "$BATCH_SIZE"
      --context-seconds "$CONTEXT_SECONDS"
      --train-clip-seconds "$TRAIN_CLIP_SECONDS"
      --eval-clip-seconds "$EVAL_CLIP_SECONDS"
      --train-ratio "$TRAIN_RATIO"
      --val-ratio "$VAL_RATIO"
      --min-gap-seconds "$MIN_GAP_SECONDS"
      --train-pos-augment-copies "$TRAIN_POS_AUGMENT_COPIES"
      --train-neg-augment-copies "$TRAIN_NEG_AUGMENT_COPIES"
      --center-bias-sigma-frac "$CENTER_BIAS_SIGMA_FRAC"
      --seed "$seed"
      --run-tag "$run_slug"
      --exp-dir "$run_exp_dir"
      --project-path "$PROJECT_PATH"
      --venv-path "$VENV_PATH"
    )

    readarray -t HEAD_ARGS < <(build_head_args "$head_spec")
    cmd+=( "${HEAD_ARGS[@]}" )

    if [[ -n "$PYTHON_MODULE" ]]; then
      cmd+=( --python-module "$PYTHON_MODULE" )
    fi
    if [[ -n "$CONTAINER_IMAGE" ]]; then
      cmd+=( --container-image "$CONTAINER_IMAGE" )
    fi
    if [[ -n "$CONTAINER_VENV_PATH" ]]; then
      cmd+=( --container-venv-path "$CONTAINER_VENV_PATH" )
    fi
    if [[ -n "$APPTAINER_MODULE" ]]; then
      cmd+=( --apptainer-module "$APPTAINER_MODULE" )
    fi
    if [[ -n "$CONTEXT_DATASET_TAR" ]]; then
      cmd+=( --context-dataset-tar "$CONTEXT_DATASET_TAR" )
    else
      cmd+=( --context-dataset-dir "$CONTEXT_DATASET_DIR" )
      if [[ "$COPY_CONTEXT_DIR_TO_TMP" == "false" ]]; then
        cmd+=( --no-copy-context-dir )
      fi
    fi
    if [[ "$DISABLE_GPU" == "true" ]]; then
      cmd+=( --disable-gpu )
    fi
    if [[ "$USE_WANDB" == "true" ]]; then
      cmd+=( --use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP_BASE" )
      if [[ -n "$WANDB_ENTITY" ]]; then
        cmd+=( --wandb-entity "$WANDB_ENTITY" )
      fi
    fi
    if [[ "$SKIP_SAVE_EMBEDDINGS" == "true" ]]; then
      cmd+=( --skip-save-embeddings )
    fi
    if [[ "$INSTALL_PERCH_DEPS" == "true" ]]; then
      cmd+=( --install-perch-deps )
    fi
    if [[ -n "$NOTE_PREFIX" ]]; then
      cmd+=( --note "${NOTE_PREFIX}:${run_slug}" )
    fi

    {
      printf '%q ' "${cmd[@]}"
      printf '\n'
    } >> "$REPLAY_SH"

    if [[ "$DRY_RUN" == "true" ]]; then
      job_id="DRYRUN_${RUN_COUNT}"
      echo "[dry-run] $job_id $run_slug head=$head_spec"
    else
      if [[ "$LOCAL_TEST_MODE" == "true" ]]; then
        "${cmd[@]}"
        job_id="LOCAL_${RUN_COUNT}"
        echo "[local] run_slug=${run_slug} head=${head_spec}"
      else
        sbatch_out="$("${cmd[@]}")"
        job_id="${sbatch_out%%;*}"
        echo "[submitted] job_id=${job_id} run_slug=${run_slug} head=${head_spec}"
      fi
    fi

    echo -e "${job_id}\t${run_slug}\t${head_spec}\t${PERCH_MODEL}\t${seed}\t${run_exp_dir}" >> "$SUBMITTED_TSV"
  done
done

chmod +x "$REPLAY_SH"

echo ""
echo "Perch head benchmark prepared: $SWEEP_DIR"
echo "Head specs:"
for head_spec in "${HEAD_SPEC_LIST[@]}"; do
  echo "  - $head_spec"
done
echo "Planned runs: $RUN_COUNT"
echo "Plan file: $PLAN_TSV"
echo "Submitted jobs: $SUBMITTED_TSV"
echo "Replay commands: $REPLAY_SH"
