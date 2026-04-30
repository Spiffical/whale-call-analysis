#!/bin/bash
# Submit the first multi-species/call-type ResNet smoke experiment on Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
EXP_ROOT="$FINAL2025_ROOT/multispecies_calltype_experiments"
ANNOTATIONS_CSV="$FINAL2025_ROOT/part2/full_bundle/manifests/annotations_all.csv"
MAT_DIR="$FINAL2025_ROOT/part2/finetune_dataset/mat_files"
BASE_CHECKPOINT="$FINAL2025_ROOT/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

RUN_NAME="multispecies_resnet_finetune_smoke"
MODEL="resnet18"
EPOCHS="3"
BATCH_SIZE="32"
NUM_WORKERS="4"
LR="1e-4"
WEIGHT_DECAY="1e-4"
MANIFEST_LIMIT="5000"
VOCAB_MIN_COUNT="5"
MAX_TRAIN_SAMPLES="2048"
MAX_VAL_SAMPLES="512"
CROP_SIZE="96"
CROP_TIME_SECONDS="10"
FREQ_MIN_HZ="5"
FREQ_MAX_HZ="100"
CENTER_BIAS_SIGMA_FRAC="0.25"
POSITIVE_CROP_MODE="edge_mix"
SEED="2026"
USE_POS_WEIGHT="false"
USE_WANDB="true"
WANDB_PROJECT="whale-multispecies-calltype"
WANDB_ENTITY=""
WANDB_GROUP="multispecies-calltype-v1"
WANDB_TAGS="multilabel,resnet,nibi,smoke,species"

SBATCH_PARTITION=""
SBATCH_GRES="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
SBATCH_TIME="02:00:00"
SBATCH_CPUS="8"
SBATCH_MEM="48G"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_resnet_smoke.sh [options]

Key options:
  --run-name NAME
  --final2025-root PATH
  --exp-root PATH
  --annotations-csv PATH
  --mat-dir PATH
  --base-checkpoint PATH
  --manifest-limit N              Default: 5000
  --vocab-min-count N             Default: 5
  --max-train-samples N           Default: 2048
  --max-val-samples N             Default: 512
  --epochs N                      Default: 3
  --batch-size N                  Default: 32
  --lr X                          Default: 1e-4
  --use-pos-weight
  --no-wandb
  --wandb-project NAME            Default: whale-multispecies-calltype
  --wandb-entity NAME
  --wandb-group NAME              Default: multispecies-calltype-v1
  --wandb-tags CSV

SBATCH:
  --partition NAME
  --gres SPEC                     Default: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
  --time HH:MM:SS                 Default: 02:00:00
  --cpus-per-task N               Default: 8
  --mem SIZE                      Default: 48G
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --final2025-root) FINAL2025_ROOT="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --annotations-csv) ANNOTATIONS_CSV="$2"; shift 2 ;;
    --mat-dir) MAT_DIR="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --manifest-limit) MANIFEST_LIMIT="$2"; shift 2 ;;
    --vocab-min-count) VOCAB_MIN_COUNT="$2"; shift 2 ;;
    --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
    --max-val-samples) MAX_VAL_SAMPLES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --use-pos-weight) USE_POS_WEIGHT="true"; shift ;;
    --no-wandb) USE_WANDB="false"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-tags) WANDB_TAGS="$2"; shift 2 ;;
    --partition) SBATCH_PARTITION="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

for required in "$ANNOTATIONS_CSV" "$MAT_DIR" "$BASE_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

RUN_ID="${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$EXP_ROOT/runs/$RUN_ID"
MANIFEST_DIR="$OUT_DIR/manifest"
SPLIT_DIR="$OUT_DIR/splits"
TRAIN_DIR="$OUT_DIR/train"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

JOB_SCRIPT="$LOG_DIR/${RUN_ID}.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
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
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_DIR="$OUT_DIR/wandb"
export WANDB_CACHE_DIR="$OUT_DIR/wandb_cache"
export WANDB_DATA_DIR="$OUT_DIR/wandb_data"
export WANDB_CONFIG_DIR="$OUT_DIR/wandb_config"
mkdir -p "\$WANDB_DIR" "\$WANDB_CACHE_DIR" "\$WANDB_DATA_DIR" "\$WANDB_CONFIG_DIR" "$MANIFEST_DIR" "$SPLIT_DIR" "$TRAIN_DIR"

python -u scripts/data/multilabel/build_call_mat_manifest.py \\
  --annotations-csv "$ANNOTATIONS_CSV" \\
  --mat-dir "$MAT_DIR" \\
  --output-dir "$MANIFEST_DIR" \\
  --dataset-name "final2025_part2_call_trainstyle" \\
  --match-tolerance-s 0.25 \\
  --vocab-min-count "$VOCAB_MIN_COUNT" \\
  --limit "$MANIFEST_LIMIT"

python -u scripts/data/multilabel/build_candidate_splits.py \\
  --manifest-csv "$MANIFEST_DIR/call_multilabel_manifest.csv" \\
  --output-dir "$SPLIT_DIR"

train_cmd=(
  python -u scripts/train/train_multilabel_resnet_smoke.py
  --manifest-csv "$SPLIT_DIR/split_manifest.csv"
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
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-val-samples "$MAX_VAL_SAMPLES"
  --wandb-project "$WANDB_PROJECT"
  --wandb-group "$WANDB_GROUP"
  --wandb-name "$RUN_ID"
  --wandb-tags "$WANDB_TAGS"
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

cat > "$OUT_DIR/run_metadata.json" <<META
{
  "run_id": "$RUN_ID",
  "run_name": "$RUN_NAME",
  "repo_root": "$REPO_ROOT",
  "annotations_csv": "$ANNOTATIONS_CSV",
  "mat_dir": "$MAT_DIR",
  "base_checkpoint": "$BASE_CHECKPOINT",
  "manifest_dir": "$MANIFEST_DIR",
  "split_dir": "$SPLIT_DIR",
  "train_dir": "$TRAIN_DIR",
  "wandb_project": "$WANDB_PROJECT",
  "wandb_group": "$WANDB_GROUP"
}
META
echo "Finished at \$(date -Is)"
EOF

chmod +x "$JOB_SCRIPT"
echo "Job script: $JOB_SCRIPT"
echo "Output dir: $OUT_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: not submitting"
  exit 0
fi

sbatch_out="$(sbatch "$JOB_SCRIPT")"
echo "$sbatch_out"
job_id="$(echo "$sbatch_out" | awk '{print $NF}')"
echo -e "job_id\trun_id\tout_dir\tjob_script" > "$OUT_DIR/submitted_jobs.tsv"
echo -e "${job_id}\t${RUN_ID}\t${OUT_DIR}\t${JOB_SCRIPT}" >> "$OUT_DIR/submitted_jobs.tsv"
echo "$job_id"
