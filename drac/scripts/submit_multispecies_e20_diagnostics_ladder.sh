#!/bin/bash
# Launch E20 diagnostic follow-ups after the E19 staged species/band ladder.

set -euo pipefail

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
DEF_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e20_diagnostics"
SOURCE_VARIANT_STAMP="20260514T002301Z"
CACHE_STAMP="20260514T002301Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E20diag_submit"

EPOCHS="45"
BATCH_SIZE="64"
NUM_WORKERS="8"
SEED="2026"
WEIGHT_DECAY="0.0001"
COORDINATOR_TIME="02:00:00"
COORDINATOR_CPUS="4"
COORDINATOR_MEM="24G"
GPU_TIME="12:00:00"
GPU_MEM="96G"
GPU_GRES="gpu:h100:1"
WANDB_GROUP="weekend-20260502-e20-diagnostics"
DRY_RUN="false"
AS_COORDINATOR="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e20_diagnostics_ladder.sh [options]

E20 reuses the E16 combined multiband MAT cache and the E19 outputs. It first
runs a CPU-only post-hoc audit of E19 PR/AP and a single-target ensemble, then
submits diagnostic H100 jobs:

  Interference / balancing:
    - Bp+Mn low+mid
    - Bm+Mn low+mid
    - Bp+Bm+Mn low+mid with label caps
    - Bp+Bm+Mn low+mid with external source-label caps
    - full low+mid+high with no class-band mask
    - full low+mid+high with label caps

  Oo-focused:
    - Oo mid-only
    - Oo high-only
    - Oo low+mid+high
    - Oo mid+high with 20s crops
    - Oo mid+high with full 40s crops
    - Oo ONC-only mid+high
    - Oo DCLDE-only mid+high in-domain check

Options:
  --repo-root PATH          Repo used inside Nibi jobs
  --source-variant-stamp S  Default: 20260514T002301Z
  --cache-stamp S           Default: 20260514T002301Z
  --stamp STAMP             New run stamp
  --epochs N                Default: 45
  --batch-size N            Default: 64
  --dry-run                 Write scripts/variants but do not submit GPU jobs
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --as-coordinator) AS_COORDINATOR="true"; shift ;;
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
    --coordinator-time) COORDINATOR_TIME="$2"; shift 2 ;;
    --coordinator-cpus) COORDINATOR_CPUS="$2"; shift 2 ;;
    --coordinator-mem) COORDINATOR_MEM="$2"; shift 2 ;;
    --gpu-time) GPU_TIME="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --gres) GPU_GRES="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/e20_diagnostics_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
VARIANT_ROOT="$WEEKEND_ROOT/manifests/e20_diagnostics_${STAMP}"

if [[ "$AS_COORDINATOR" != "true" ]]; then
  mkdir -p "$LOG_DIR"
  submit_args=(
    "$0"
    --as-coordinator
    --final2025-root "$FINAL2025_ROOT"
    --weekend-root "$WEEKEND_ROOT"
    --def-root "$DEF_ROOT"
    --repo-root "$REPO_ON_NIBI"
    --source-variant-stamp "$SOURCE_VARIANT_STAMP"
    --cache-stamp "$CACHE_STAMP"
    --stamp "$STAMP"
    --run-name "$RUN_NAME"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --gpu-time "$GPU_TIME"
    --gpu-mem "$GPU_MEM"
    --gres "$GPU_GRES"
    --wandb-group "$WANDB_GROUP"
  )
  if [[ "$DRY_RUN" == "true" ]]; then
    submit_args+=(--dry-run)
  fi
  job_id="$(
    sbatch \
      --job-name="$RUN_NAME" \
      --output="$LOG_DIR/slurm-%j.out" \
      --time="$COORDINATOR_TIME" \
      --cpus-per-task="$COORDINATOR_CPUS" \
      --mem="$COORDINATOR_MEM" \
      "${submit_args[@]}" | awk '{print $4}'
  )"
  echo "Submitted $RUN_NAME as $job_id"
  echo "Pipeline dir: $PIPELINE_DIR"
  echo "Slurm log: $LOG_DIR/slurm-$job_id.out"
  exit 0
fi

echo "Started E20 diagnostic ladder at $(date -Is)"
echo "Host: $(hostname)"

FINAL2025="$FINAL2025_ROOT"
WEEKEND="$WEEKEND_ROOT"
DEF_ROOT="$DEF_ROOT"
REPO="$REPO_ON_NIBI"
SOURCE_VARIANT_ROOT="$WEEKEND/manifests/multiband_variants_$SOURCE_VARIANT_STAMP"
INPUT_VARIANT="$SOURCE_VARIANT_ROOT/E16_e13_multiband_oncrare_full"
CACHE_DIR="$DEF_ROOT/mat_archives/multiband40s_$CACHE_STAMP"
ARCHIVE_PATH="$CACHE_DIR/multiband40s_mat_cache.tar"
EXTRACT_DIR="$CACHE_DIR/extracted"
SUBMITTED_TSV="$PIPELINE_DIR/e20_training_submitted.tsv"
PLAN_TSV="$PIPELINE_DIR/e20_training_plan.tsv"
POSTHOC_DIR="$PIPELINE_DIR/e20_posthoc_e19"
BASE_CKPT="$FINAL2025/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

mkdir -p "$PIPELINE_DIR" "$LOG_DIR" "$VARIANT_ROOT" "$POSTHOC_DIR"
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
mkdir -p "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$PIP_CACHE_DIR"

echo "Repo: $REPO"
git rev-parse HEAD || true
echo "Input variant: $INPUT_VARIANT"
echo "Cache dir: $CACHE_DIR"
timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "Missing archive: $ARCHIVE_PATH" >&2
  exit 2
fi
if [[ ! -f "$INPUT_VARIANT/standardized_manifest.csv" ]]; then
  echo "Missing input manifest: $INPUT_VARIANT/standardized_manifest.csv" >&2
  exit 2
fi

expected_count="$(
  python - "$CACHE_DIR/archive_meta/archive_summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
print(payload.get("unique_mat_count") or payload.get("input_rows") or 128889)
PY
)"
existing_count=0
if [[ -d "$EXTRACT_DIR/mat_files" ]]; then
  existing_count="$(find "$EXTRACT_DIR/mat_files" -type f -name '*.mat' 2>/dev/null | wc -l | tr -d ' ')"
fi
echo "Extracted MAT count: $existing_count / expected $expected_count"
if [[ "$existing_count" -lt "$expected_count" ]]; then
  echo "Re-extracting E16 combined MAT archive at $(date -Is)"
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  timeout 180 diskusage_report || true
  df -ih /project/def-kmoran /scratch || true
  tar -xf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"
  find "$EXTRACT_DIR/mat_files" -type f -name '*.mat' | wc -l > "$EXTRACT_DIR/.mat_count"
  echo "Extraction complete at $(date -Is): $(cat "$EXTRACT_DIR/.mat_count") MAT files"
fi

echo "Running E19 post-hoc PR/AP and single-target ensemble audit"
python -u scripts/analysis/e20_posthoc_e19_diagnostics.py \
  --weekend-root "$WEEKEND" \
  --output-dir "$POSTHOC_DIR"

echo "Building E20 diagnostic variants"
python -u scripts/data/multilabel/build_e20_diagnostic_variants.py \
  --input-manifest "$INPUT_VARIANT/standardized_manifest.csv" \
  --input-vocab "$INPUT_VARIANT/label_vocabulary.json" \
  --output-root "$VARIANT_ROOT" \
  --seed "$SEED"

echo "Queue/accounting check before GPU submissions"
squeue -u merileo || true
sacct -u merileo --starttime now-7days || true
timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true

echo -e "job_id\texperiment\trun_dir\tjob_script" > "$SUBMITTED_TSV"
echo -e "experiment\tvariant\tmanifest\tvocab\tdataset_root\tencoder\tfusion\tlr\tweight_decay\tuse_pos_weight\tband_availability\tclass_band_mask\tcrop_mode\tcrop_seconds\tband_crop_shapes\tbatch_size\tbands\teval_label_ids\tcalibration_source_kind\teval_source_kind\trun_dir\tarchive_path\twandb_group" > "$PLAN_TSV"

submit_train() {
  local variant_name="$1"
  local suffix="$2"
  local encoder="$3"
  local train_lr="$4"
  local use_pos_weight="$5"
  local band_availability="$6"
  local class_band_mask="$7"
  local crop_mode="$8"
  local crop_seconds="$9"
  local band_crop_shapes="${10}"
  local batch_size="${11}"
  local bands_arg="${12}"
  local eval_label_ids="${13}"
  local calibration_source_kind="${14}"
  local eval_source_kind="${15}"
  local example_band="${16}"
  local variant_dir="$VARIANT_ROOT/$variant_name"
  local run_exp="${variant_name}_${suffix}"
  local run_dir="$WEEKEND/runs/${run_exp}_$(date -u +%Y%m%dT%H%M%SZ)"
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
#SBATCH --gres=$GPU_GRES

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
  python -u scripts/train/train_multiband_multilabel.py
  --manifest-csv "$variant_dir/standardized_manifest.csv"
  --vocab-json "$variant_dir/label_vocabulary.json"
  --dataset-root "$EXTRACT_DIR"
  --exp-dir "$train_dir"
  --bands "$bands_arg"
  --band-crop-shapes "$band_crop_shapes"
  --encoder "$encoder"
  --fusion gated
  --init-all-branches-checkpoint "$BASE_CKPT"
  --epochs "$EPOCHS"
  --batch-size "$batch_size"
  --num-workers "$NUM_WORKERS"
  --lr "$train_lr"
  --weight-decay "$WEIGHT_DECAY"
  --crop-time-seconds "$crop_seconds"
  --context-seconds 40
  --center-bias-sigma-frac 0.25
  --positive-crop-mode "$crop_mode"
  --band-availability-mode "$band_availability"
  --class-band-mask-mode "$class_band_mask"
  --device cuda
  --seed "$SEED"
  --example-image-band "$example_band"
  --use-wandb
  --wandb-project whale-multispecies-calltype
  --wandb-group "$WANDB_GROUP"
  --wandb-name "$run_exp"
  --wandb-tags "multilabel,species,E20,diagnostics,$variant_name,$encoder,$bands_arg,$band_availability,$class_band_mask,$crop_mode,crop${crop_seconds}"
)
if [[ "$use_pos_weight" == "true" ]]; then
  train_cmd+=(--use-pos-weight)
fi
"\${train_cmd[@]}"
python -u scripts/analysis/summarize_multilabel_predictions.py \\
  --validation-csv "$train_dir/validation_predictions.csv" \\
  --test-csv "$train_dir/test_predictions.csv" \\
  --output-dir "$train_dir/onc_calibrated_eval" \\
  --calibration-source-kind "$calibration_source_kind" \\
  --eval-source-kind "$eval_source_kind" \\
  --label-ids "$eval_label_ids"
cat > "$run_dir/run_metadata.json" <<META
{
  "experiment": "$run_exp",
  "variant": "$variant_name",
  "encoder": "$encoder",
  "fusion": "gated",
  "manifest_csv": "$variant_dir/standardized_manifest.csv",
  "vocab_json": "$variant_dir/label_vocabulary.json",
  "dataset_root": "$EXTRACT_DIR",
  "mat_archive": "$ARCHIVE_PATH",
  "train_dir": "$train_dir",
  "use_pos_weight": "$use_pos_weight",
  "lr": "$train_lr",
  "weight_decay": "$WEIGHT_DECAY",
  "bands": "$bands_arg",
  "eval_label_ids": "$eval_label_ids",
  "calibration_source_kind": "$calibration_source_kind",
  "eval_source_kind": "$eval_source_kind",
  "band_availability_mode": "$band_availability",
  "class_band_mask_mode": "$class_band_mask",
  "crop_time_seconds": $crop_seconds,
  "band_crop_shapes": "$band_crop_shapes",
  "positive_crop_mode": "$crop_mode",
  "epochs": $EPOCHS,
  "batch_size": $batch_size
}
META
TRAIN_EOF
  echo -e "$run_exp\t$variant_name\t$variant_dir/standardized_manifest.csv\t$variant_dir/label_vocabulary.json\t$EXTRACT_DIR\t$encoder\tgated\t$train_lr\t$WEIGHT_DECAY\t$use_pos_weight\t$band_availability\t$class_band_mask\t$crop_mode\t$crop_seconds\t$band_crop_shapes\t$batch_size\t$bands_arg\t$eval_label_ids\t$calibration_source_kind\t$eval_source_kind\t$run_dir\t$ARCHIVE_PATH\t$WANDB_GROUP" >> "$PLAN_TSV"
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

SHAPES_10S="low:391x50,mid:256x100,high:256x312"
SHAPES_20S="low:391x100,mid:256x200,high:256x624"
SHAPES_40S="low:391x200,mid:256x400,high:256x1248"

submit_train "E20_bp_mn_lowmid_cumulative" "r18_noposw" "resnet18" "0.0003" "false" "all" "audit_v2" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid" "species:Bp,species:Mn" "ONC" "ONC" "mid"
submit_train "E20_bm_mn_lowmid_cumulative" "r18_noposw" "resnet18" "0.0003" "false" "all" "audit_v2" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid" "species:Bm,species:Mn" "ONC" "ONC" "mid"
submit_train "E20_bp_bm_mn_lowmid_labelcap" "r18_noposw" "resnet18" "0.0003" "false" "all" "audit_v2" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid" "species:Bp,species:Bm,species:Mn" "ONC" "ONC" "mid"
submit_train "E20_bp_bm_mn_lowmid_sourcecap" "r18_noposw" "resnet18" "0.0003" "false" "all" "audit_v2" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid" "species:Bp,species:Bm,species:Mn" "ONC" "ONC" "mid"
submit_train "E20_full_allbands_nomask" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "none" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid,high" "species:Bp,species:Bm,species:Mn,species:Oo" "ONC" "ONC" "high"
submit_train "E20_full_allbands_labelcap" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "odont_high" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid,high" "species:Bp,species:Bm,species:Mn,species:Oo" "ONC" "ONC" "high"
submit_train "E20_oo_mid_only" "r18_noposw" "resnet18" "0.0003" "false" "all" "none" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "mid" "species:Oo" "ONC" "ONC" "mid"
submit_train "E20_oo_high_only" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "none" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "high" "species:Oo" "ONC" "ONC" "high"
submit_train "E20_oo_allbands" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "none" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid,high" "species:Oo" "ONC" "ONC" "high"
submit_train "E20_oo_midhigh_wide20" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "none" "centered_gaussian" "20" "$SHAPES_20S" "32" "mid,high" "species:Oo" "ONC" "ONC" "high"
submit_train "E20_oo_midhigh_full40" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "none" "centered_gaussian" "40" "$SHAPES_40S" "16" "mid,high" "species:Oo" "ONC" "ONC" "high"
submit_train "E20_oo_onc_only_midhigh" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "none" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "mid,high" "species:Oo" "ONC" "ONC" "high"
submit_train "E20_oo_dclde_only_midhigh" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "none" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "mid,high" "species:Oo" "DCLDE" "DCLDE" "high"

echo "Training plan:"
cat "$PLAN_TSV"
echo "Submitted jobs:"
cat "$SUBMITTED_TSV"
echo "Post-hoc outputs:"
find "$POSTHOC_DIR" -maxdepth 3 -type f | sort
echo "Finished E20 diagnostic ladder submission at $(date -Is)"
