#!/bin/bash
# Launch archive-backed ONC calibration experiments without regenerating MATs.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_sourcecal_20260507_fc53054"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E15onc_calib"

ARCHIVE_STAMP="20260507T195430Z"
EXTRACT_DIR="$WEEKEND_ROOT/mat_archives/scaleup40s_${ARCHIVE_STAMP}/extracted"
ARCHIVE_PATH="$WEEKEND_ROOT/mat_archives/scaleup40s_${ARCHIVE_STAMP}/scaleup40s_mat_cache.tar"
MANIFEST_ROOT="$WEEKEND_ROOT/manifests"
E12_INPUT="$MANIFEST_ROOT/E12large_onc_dclde_40sctx_autoneg_species"
E13_INPUT="$MANIFEST_ROOT/E13large_onc_biod_dclde_40sctx_autoneg_species"
VARIANT_ROOT="$MANIFEST_ROOT/onc_calibration_variants_${STAMP}"
PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/onc_calibration_ladder_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
SUBMITTED_TSV="$PIPELINE_DIR/onc_calibration_training_submitted.tsv"
PLAN_TSV="$PIPELINE_DIR/onc_calibration_training_plan.tsv"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
BASE_CKPT="$FINAL2025_ROOT/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

ONC_RARE_TARGET="10000"
EPOCHS="40"
BATCH_SIZE="64"
NUM_WORKERS="8"
SEED="2026"
WEIGHT_DECAY="0.0001"
GPU_TIME="08:00:00"
GPU_MEM="72G"
SBATCH_GRES="gpu:h100:1"
WANDB_GROUP="weekend-20260502-onc-calibration"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_onc_calibration_ladder.sh [options]

This reuses the E14 40s MAT archive, builds train-only ONC rare-label
oversampling variants, and submits a bounded H100 ladder.

Options:
  --repo-root PATH             Repo used on Nibi inside jobs
  --weekend-root PATH          Default: /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502
  --stamp STAMP                Override UTC stamp
  --archive-stamp STAMP        Existing MAT archive stamp, default 20260507T195430Z
  --onc-rare-target N          Train target for ONC Bm/Mn/Oo, default 10000
  --epochs N                   Default: 40
  --dry-run                    Build variants/scripts but do not submit jobs
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --archive-stamp) ARCHIVE_STAMP="$2"; shift 2 ;;
    --onc-rare-target) ONC_RARE_TARGET="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --gpu-time) GPU_TIME="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

# Recompute dependent paths after option parsing.
EXTRACT_DIR="$WEEKEND_ROOT/mat_archives/scaleup40s_${ARCHIVE_STAMP}/extracted"
ARCHIVE_PATH="$WEEKEND_ROOT/mat_archives/scaleup40s_${ARCHIVE_STAMP}/scaleup40s_mat_cache.tar"
MANIFEST_ROOT="$WEEKEND_ROOT/manifests"
E12_INPUT="$MANIFEST_ROOT/E12large_onc_dclde_40sctx_autoneg_species"
E13_INPUT="$MANIFEST_ROOT/E13large_onc_biod_dclde_40sctx_autoneg_species"
VARIANT_ROOT="$MANIFEST_ROOT/onc_calibration_variants_${STAMP}"
PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/onc_calibration_ladder_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
SUBMITTED_TSV="$PIPELINE_DIR/onc_calibration_training_submitted.tsv"
PLAN_TSV="$PIPELINE_DIR/onc_calibration_training_plan.tsv"

mkdir -p "$LOG_DIR" "$VARIANT_ROOT"
cd "$REPO_ON_NIBI"
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -d "$EXTRACT_DIR" ]]; then
  echo "Missing archive extraction dir: $EXTRACT_DIR" >&2
  exit 2
fi
if [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "Missing MAT archive: $ARCHIVE_PATH" >&2
  exit 2
fi

echo "Repo: $REPO_ON_NIBI"
git rev-parse HEAD
echo "Variant root: $VARIANT_ROOT"

build_variant() {
  local input_dir="$1"
  local variant_name="$2"
  shift 2
  local out_dir="$VARIANT_ROOT/$variant_name"
  python -u scripts/data/multilabel/build_onc_calibration_manifest_variants.py \
    --manifest-csv "$input_dir/standardized_manifest.csv" \
    --vocab-json "$input_dir/label_vocabulary.json" \
    --output-dir "$out_dir" \
    --variant-name "$variant_name" \
    --seed "$SEED" \
    "$@"
}

COMMON_ONC_OVERSAMPLE=(
  --oversample-train-source-label "ONC:species:Bm:$ONC_RARE_TARGET"
  --oversample-train-source-label "ONC:species:Mn:$ONC_RARE_TARGET"
  --oversample-train-source-label "ONC:species:Oo:$ONC_RARE_TARGET"
)

echo "Building ONC rare-label full-source variant"
build_variant "$E13_INPUT" "E15_e13_oncrare_full" "${COMMON_ONC_OVERSAMPLE[@]}"

echo "Building ONC rare-label external-capped full-source variant"
build_variant "$E13_INPUT" "E15_e13_oncrare_extcap" \
  "${COMMON_ONC_OVERSAMPLE[@]}" \
  --train-source-label-cap "BioDCASE:species:Bm:8000" \
  --train-source-label-cap "BioDCASE:species:Bp:8000" \
  --train-source-label-cap "BioDCASE:<background>:1000" \
  --train-source-label-cap "DCLDE:species:Mn:3000" \
  --train-source-label-cap "DCLDE:species:Oo:3000" \
  --train-source-label-cap "DCLDE:<background>:3000"

echo "Building ONC rare-label DCLDE-capped no-BioDCASE variant"
build_variant "$E12_INPUT" "E15_e12_oncrare_dcldecap" \
  "${COMMON_ONC_OVERSAMPLE[@]}" \
  --train-source-label-cap "DCLDE:species:Mn:2500" \
  --train-source-label-cap "DCLDE:species:Oo:2500" \
  --train-source-label-cap "DCLDE:<background>:2500"

echo "Queue/accounting check before GPU submissions"
squeue -u merileo || true
sacct -u merileo --starttime now-7days || true

echo -e "experiment\tmanifest\tvocab\tdataset_root\tmodel\tlr\tweight_decay\tuse_pos_weight\trun_dir\tarchive_path\twandb_group" > "$PLAN_TSV"
echo -e "job_id\texperiment\trun_dir\tjob_script" > "$SUBMITTED_TSV"

submit_train() {
  local variant_name="$1"
  local use_pos_weight="$2"
  local suffix="$3"
  local model_name="$4"
  local train_lr="$5"
  local train_weight_decay="$6"
  local variant_dir="$VARIANT_ROOT/$variant_name"
  local run_exp="${variant_name}_${suffix}"
  local run_dir="$WEEKEND_ROOT/runs/${run_exp}_$(date -u +%Y%m%dT%H%M%SZ)"
  local run_log_dir="$run_dir/logs"
  local train_dir="$run_dir/train"
  mkdir -p "$run_log_dir" "$train_dir"
  local job_script="$run_log_dir/${run_exp}.sbatch"
  cat > "$job_script" <<TRAIN_EOF
#!/bin/bash
#SBATCH --job-name=$run_exp
#SBATCH --output=$run_log_dir/slurm-%j.out
#SBATCH --time=$GPU_TIME
#SBATCH --cpus-per-task=8
#SBATCH --mem=$GPU_MEM
#SBATCH --gres=$SBATCH_GRES

set -euo pipefail
cd "$REPO_ON_NIBI"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export WANDB_PROJECT=whale-multispecies-calltype
export WANDB_DIR="$run_dir/wandb"
export WANDB_CACHE_DIR="$run_dir/wandb_cache"
export WANDB_DATA_DIR="$run_dir/wandb_data"
export WANDB_CONFIG_DIR="$run_dir/wandb_config"
mkdir -p "$train_dir" "\$WANDB_DIR" "\$WANDB_CACHE_DIR" "\$WANDB_DATA_DIR" "\$WANDB_CONFIG_DIR"
train_cmd=(
  python -u scripts/train/train_multilabel_resnet_smoke.py
  --manifest-csv "$variant_dir/standardized_manifest.csv"
  --vocab-json "$variant_dir/label_vocabulary.json"
  --dataset-root "$EXTRACT_DIR"
  --exp-dir "$train_dir"
  --model "$model_name"
  --init-checkpoint "$BASE_CKPT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "$train_lr"
  --weight-decay "$train_weight_decay"
  --crop-size 96
  --crop-time-seconds 10
  --freq-min-hz 5
  --freq-max-hz 100
  --center-bias-sigma-frac 0.25
  --positive-crop-mode edge_mix
  --device cuda
  --seed "$SEED"
  --use-wandb
  --wandb-project whale-multispecies-calltype
  --wandb-group "$WANDB_GROUP"
  --wandb-name "$run_exp"
  --wandb-tags "multilabel,species,scaleup40s,mat-archive,onc-calibration,$variant_name,$model_name"
)
if [[ "$use_pos_weight" == "true" ]]; then
  train_cmd+=(--use-pos-weight)
fi
"\${train_cmd[@]}"
python -u scripts/analysis/summarize_multilabel_predictions.py \
  --validation-csv "$train_dir/validation_predictions.csv" \
  --test-csv "$train_dir/test_predictions.csv" \
  --output-dir "$train_dir/onc_calibrated_eval" \
  --calibration-source-kind ONC \
  --eval-source-kind ONC
cat > "$run_dir/run_metadata.json" <<META
{
  "experiment": "$run_exp",
  "variant": "$variant_name",
  "model": "$model_name",
  "manifest_csv": "$variant_dir/standardized_manifest.csv",
  "vocab_json": "$variant_dir/label_vocabulary.json",
  "dataset_root": "$EXTRACT_DIR",
  "mat_archive": "$ARCHIVE_PATH",
  "train_dir": "$train_dir",
  "use_pos_weight": "$use_pos_weight",
  "lr": "$train_lr",
  "weight_decay": "$train_weight_decay",
  "crop_time_seconds": 10,
  "positive_crop_mode": "edge_mix",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "onc_rare_target": $ONC_RARE_TARGET
}
META
TRAIN_EOF
  echo -e "$run_exp\t$variant_dir/standardized_manifest.csv\t$variant_dir/label_vocabulary.json\t$EXTRACT_DIR\t$model_name\t$train_lr\t$train_weight_decay\t$use_pos_weight\t$run_dir\t$ARCHIVE_PATH\t$WANDB_GROUP" >> "$PLAN_TSV"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "DRY_RUN\t$run_exp\t$run_dir\t$job_script" >> "$SUBMITTED_TSV"
    echo "DRY_RUN: wrote $job_script"
  else
    local job_id
    job_id="$(sbatch "$job_script" | awk '{print $4}')"
    echo -e "$job_id\t$run_exp\t$run_dir\t$job_script" >> "$SUBMITTED_TSV"
    echo "Submitted $run_exp as $job_id"
  fi
}

submit_train "E15_e13_oncrare_full" false "resnet18_noposw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"
submit_train "E15_e13_oncrare_full" true "resnet18_posw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"
submit_train "E15_e13_oncrare_extcap" false "resnet18_noposw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"
submit_train "E15_e13_oncrare_extcap" true "resnet18_posw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"
submit_train "E15_e13_oncrare_extcap" false "resnet34_noposw_lr2e4" "resnet34" "0.0002" "$WEIGHT_DECAY"
submit_train "E15_e12_oncrare_dcldecap" true "resnet18_posw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"

echo "Training plan:"
cat "$PLAN_TSV"
echo "Submitted jobs:"
cat "$SUBMITTED_TSV"
