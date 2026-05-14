#!/bin/bash
# Reuse the E16 combined multiband MAT archive and launch source-aware E17 ablations.

set -euo pipefail

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
DEF_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e17_multiband"
SOURCE_VARIANT_STAMP="20260514T002301Z"
CACHE_STAMP="20260514T002301Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E17reuse_submit"

EPOCHS="45"
BATCH_SIZE="32"
NUM_WORKERS="8"
SEED="2026"
WEIGHT_DECAY="0.0001"
SBATCH_TIME="06:00:00"
SBATCH_CPUS="8"
SBATCH_MEM="48G"
GPU_TIME="12:00:00"
GPU_MEM="96G"
SBATCH_GRES="gpu:h100:1"
WANDB_GROUP="weekend-20260502-e17-multiband-source-aware"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e17_multiband_reuse_ladder.sh [options]

This submits one small CPU coordination job that reuses the E16 combined
multiband MAT tar archive, re-extracts it if needed, then submits a bounded
parallel H100 ladder testing source-aware band masks, class-band routing, and
more centered positive crops.

Options:
  --repo-root PATH          Repo used inside Nibi jobs
  --source-variant-stamp S  Default: 20260514T002301Z
  --cache-stamp S           Default: 20260514T002301Z
  --stamp STAMP             New run stamp
  --epochs N                Default: 45
  --batch-size N            Default: 32
  --dry-run                 Write sbatch scripts but do not submit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --final2025-root) FINAL2025_ROOT="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --def-root) DEF_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --source-variant-stamp) SOURCE_VARIANT_STAMP="$2"; shift 2 ;;
    --cache-stamp) CACHE_STAMP="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --gpu-time) GPU_TIME="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/e17_multiband_reuse_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
JOB_SCRIPT="$LOG_DIR/e17_multiband_reuse_${STAMP}.sbatch"
mkdir -p "$LOG_DIR"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail

echo "Started E17 multiband reuse ladder at \$(date -Is)"
echo "Host: \$(hostname)"

FINAL2025="$FINAL2025_ROOT"
WEEKEND="$WEEKEND_ROOT"
DEF_ROOT="$DEF_ROOT"
REPO="$REPO_ON_NIBI"
SOURCE_VARIANT_STAMP="$SOURCE_VARIANT_STAMP"
CACHE_STAMP="$CACHE_STAMP"
STAMP="$STAMP"
PIPELINE_DIR="$PIPELINE_DIR"
VARIANT_ROOT="\$WEEKEND/manifests/multiband_variants_\$SOURCE_VARIANT_STAMP"
CACHE_DIR="\$DEF_ROOT/mat_archives/multiband40s_\$CACHE_STAMP"
ARCHIVE_PATH="\$CACHE_DIR/multiband40s_mat_cache.tar"
EXTRACT_DIR="\$CACHE_DIR/extracted"
SUBMITTED_TSV="\$PIPELINE_DIR/e17_training_submitted.tsv"
PLAN_TSV="\$PIPELINE_DIR/e17_training_plan.tsv"
BASE_CKPT="\$FINAL2025/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

mkdir -p "\$PIPELINE_DIR"
cd "\$REPO"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_CACHE_DIR" "\$PIP_CACHE_DIR"

echo "Repo: \$REPO"
git rev-parse HEAD || true
echo "Variant root: \$VARIANT_ROOT"
echo "Cache dir: \$CACHE_DIR"
timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true

if [[ ! -f "\$ARCHIVE_PATH" ]]; then
  echo "Missing archive: \$ARCHIVE_PATH" >&2
  exit 2
fi
if [[ ! -f "\$VARIANT_ROOT/E16_e13_multiband_oncrare_extcap/standardized_manifest.csv" ]]; then
  echo "Missing E16 variant manifests under \$VARIANT_ROOT" >&2
  exit 2
fi

expected_count="\$(python - "\$CACHE_DIR/archive_meta/archive_summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
print(payload.get("unique_mat_count") or payload.get("input_rows") or 128889)
PY
)"
existing_count=0
if [[ -d "\$EXTRACT_DIR/mat_files" ]]; then
  existing_count="\$(find "\$EXTRACT_DIR/mat_files" -type f -name '*.mat' 2>/dev/null | wc -l | tr -d ' ')"
fi
echo "Extracted MAT count: \$existing_count / expected \$expected_count"
if [[ "\$existing_count" -lt "\$expected_count" ]]; then
  echo "Re-extracting E16 combined MAT archive at \$(date -Is)"
  rm -rf "\$EXTRACT_DIR"
  mkdir -p "\$EXTRACT_DIR"
  timeout 180 diskusage_report || true
  df -ih /project/def-kmoran /scratch || true
  tar -xf "\$ARCHIVE_PATH" -C "\$EXTRACT_DIR"
  find "\$EXTRACT_DIR/mat_files" -type f -name '*.mat' | wc -l > "\$EXTRACT_DIR/.mat_count"
  echo "Extraction complete at \$(date -Is): \$(cat "\$EXTRACT_DIR/.mat_count") MAT files"
fi

echo "Queue/accounting check before GPU submissions"
squeue -u merileo || true
sacct -u merileo --starttime now-7days || true
timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true

echo -e "job_id\texperiment\trun_dir\tjob_script" > "\$SUBMITTED_TSV"
echo -e "experiment\tvariant\tmanifest\tvocab\tdataset_root\tencoder\tfusion\tlr\tweight_decay\tuse_pos_weight\tband_availability\tclass_band_mask\tcrop_mode\tbands\trun_dir\tarchive_path\twandb_group" > "\$PLAN_TSV"

submit_train() {
  local variant_name="\$1"
  local suffix="\$2"
  local encoder="\$3"
  local train_lr="\$4"
  local use_pos_weight="\$5"
  local band_availability="\$6"
  local class_band_mask="\$7"
  local crop_mode="\$8"
  local bands_arg="\$9"
  local variant_dir="\$VARIANT_ROOT/\$variant_name"
  local run_exp="E17_\${variant_name#E16_}_\${suffix}"
  local run_dir="\$WEEKEND/runs/\${run_exp}_\$(date -u +%Y%m%dT%H%M%SZ)"
  local run_log_dir="\$run_dir/logs"
  local train_dir="\$run_dir/train"
  mkdir -p "\$run_log_dir" "\$train_dir"
  local job_script="\$run_log_dir/\${run_exp}.sbatch"
  cat > "\$job_script" <<TRAIN_EOF
#!/bin/bash
#SBATCH --job-name=\$run_exp
#SBATCH --output=\$run_log_dir/slurm-%j.out
#SBATCH --time=$GPU_TIME
#SBATCH --cpus-per-task=8
#SBATCH --mem=$GPU_MEM
#SBATCH --gres=$SBATCH_GRES

set -euo pipefail
cd "$REPO_ON_NIBI"
source .venv/bin/activate
export PYTHONPATH="\\\$PWD:\\\${PYTHONPATH:-}"
export WANDB_PROJECT=whale-multispecies-calltype
export WANDB_DIR="\$run_dir/wandb"
export WANDB_CACHE_DIR="\$run_dir/wandb_cache"
export WANDB_DATA_DIR="\$run_dir/wandb_data"
export WANDB_CONFIG_DIR="\$run_dir/wandb_config"
mkdir -p "\$train_dir" "\\\$WANDB_DIR" "\\\$WANDB_CACHE_DIR" "\\\$WANDB_DATA_DIR" "\\\$WANDB_CONFIG_DIR"
train_cmd=(
  python -u scripts/train/train_multiband_multilabel.py
  --manifest-csv "\$variant_dir/standardized_manifest.csv"
  --vocab-json "\$variant_dir/label_vocabulary.json"
  --dataset-root "\$EXTRACT_DIR"
  --exp-dir "\$train_dir"
  --bands "\$bands_arg"
  --band-crop-shapes low:391x50,mid:256x100,high:256x312
  --encoder "\$encoder"
  --fusion gated
  --init-low-checkpoint "\$BASE_CKPT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "\$train_lr"
  --weight-decay "$WEIGHT_DECAY"
  --crop-time-seconds 10
  --context-seconds 40
  --center-bias-sigma-frac 0.25
  --positive-crop-mode "\$crop_mode"
  --band-availability-mode "\$band_availability"
  --class-band-mask-mode "\$class_band_mask"
  --device cuda
  --seed "$SEED"
  --use-wandb
  --wandb-project whale-multispecies-calltype
  --wandb-group "$WANDB_GROUP"
  --wandb-name "\$run_exp"
  --wandb-tags "multilabel,species,multiband,E17,\$variant_name,\$encoder,\$band_availability,\$class_band_mask,\$crop_mode"
)
if [[ "\$use_pos_weight" == "true" ]]; then
  train_cmd+=(--use-pos-weight)
fi
"\\\${train_cmd[@]}"
python -u scripts/analysis/summarize_multilabel_predictions.py \\
  --validation-csv "\$train_dir/validation_predictions.csv" \\
  --test-csv "\$train_dir/test_predictions.csv" \\
  --output-dir "\$train_dir/onc_calibrated_eval" \\
  --calibration-source-kind ONC \\
  --eval-source-kind ONC
cat > "\$run_dir/run_metadata.json" <<META
{
  "experiment": "\$run_exp",
  "variant": "\$variant_name",
  "encoder": "\$encoder",
  "fusion": "gated",
  "manifest_csv": "\$variant_dir/standardized_manifest.csv",
  "vocab_json": "\$variant_dir/label_vocabulary.json",
  "dataset_root": "\$EXTRACT_DIR",
  "mat_archive": "\$ARCHIVE_PATH",
  "train_dir": "\$train_dir",
  "use_pos_weight": "\$use_pos_weight",
  "lr": "\$train_lr",
  "weight_decay": "$WEIGHT_DECAY",
  "bands": "\$bands_arg",
  "band_availability_mode": "\$band_availability",
  "class_band_mask_mode": "\$class_band_mask",
  "crop_time_seconds": 10,
  "positive_crop_mode": "\$crop_mode",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE
}
META
TRAIN_EOF
  echo -e "\$run_exp\t\$variant_name\t\$variant_dir/standardized_manifest.csv\t\$variant_dir/label_vocabulary.json\t\$EXTRACT_DIR\t\$encoder\tgated\t\$train_lr\t$WEIGHT_DECAY\t\$use_pos_weight\t\$band_availability\t\$class_band_mask\t\$crop_mode\t\$bands_arg\t\$run_dir\t\$ARCHIVE_PATH\t$WANDB_GROUP" >> "\$PLAN_TSV"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "DRY_RUN\t\$run_exp\t\$run_dir\t\$job_script" >> "\$SUBMITTED_TSV"
    echo "DRY_RUN: wrote \$job_script"
  else
    local job_id
    job_id="\$(sbatch "\$job_script" | awk '{print \$4}')"
    echo -e "\$job_id\t\$run_exp\t\$run_dir\t\$job_script" >> "\$SUBMITTED_TSV"
    echo "Submitted \$run_exp as \$job_id"
  fi
}

submit_train "E16_e13_multiband_oncrare_extcap" "srcmask_r18_posw_edge" "resnet18" "0.0003" "true" "source" "none" "edge_mix" "low,mid,high"
submit_train "E16_e13_multiband_oncrare_extcap" "srcmask_auditv1_r18_posw_edge" "resnet18" "0.0003" "true" "source" "audit_v1" "edge_mix" "low,mid,high"
submit_train "E16_e13_multiband_oncrare_extcap" "srcmask_auditv2_r18_posw_center" "resnet18" "0.0003" "true" "source" "audit_v2" "centered_gaussian" "low,mid,high"
submit_train "E16_e13_multiband_oncrare_extcap" "srcmask_auditv2_r18_noposw_center" "resnet18" "0.0003" "false" "source" "audit_v2" "centered_gaussian" "low,mid,high"
submit_train "E16_e12_multiband_oncrare_dcldecap" "srcmask_auditv2_r18_posw_center" "resnet18" "0.0003" "true" "source" "audit_v2" "centered_gaussian" "low,mid,high"
submit_train "E16_e13_multiband_oncrare_full" "srcmask_auditv2_r18_posw_center" "resnet18" "0.0003" "true" "source" "audit_v2" "centered_gaussian" "low,mid,high"
submit_train "E16_e13_multiband_oncrare_extcap" "lowmid_auditv2_r18_posw_center" "resnet18" "0.0003" "true" "all" "audit_v2" "centered_gaussian" "low,mid"

echo "Training plan:"
cat "\$PLAN_TSV"
echo "Submitted jobs:"
cat "\$SUBMITTED_TSV"
echo "Finished E17 multiband reuse ladder at \$(date -Is)"
EOF

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: wrote $JOB_SCRIPT"
else
  job_id="$(sbatch "$JOB_SCRIPT" | awk '{print $4}')"
  echo "Submitted $RUN_NAME as $job_id"
  echo "Pipeline dir: $PIPELINE_DIR"
  echo "Slurm log: $LOG_DIR/slurm-$job_id.out"
fi
