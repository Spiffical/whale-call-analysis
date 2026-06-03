#!/bin/bash
# Launch E26 ONC-only per-species expert ablation.

set -euo pipefail

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
DEF_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
SOURCE_VARIANT_STAMP="20260514T002301Z"
CACHE_STAMP="20260514T002301Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E26oncOnly_submit"

EPOCHS="45"
BATCH_SIZE="32"
NUM_WORKERS="4"
SEED="2026"
WEIGHT_DECAY="0.0001"
COORDINATOR_TIME="01:00:00"
COORDINATOR_CPUS="2"
COORDINATOR_MEM="16G"
GPU_TIME="03:00:00"
GPU_CPUS="4"
GPU_MEM="48G"
GPU_GRES="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GPU_EXCLUDE=""
CHAIN_DEPTH="4"
STOP_AFTER_SECONDS="9000"
WANDB_GROUP="weekend-20260502-e26-onc-only"
DRY_RUN="false"
AS_COORDINATOR="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e26_onc_only_experts.sh [options]

E26 repeats the best E24 expert ensemble setup while training only on ONC
annotations. It submits three resumable 3-hour MIG chains, one each for fin
whale, blue whale, and humpback whale, followed by an ONC-calibrated report.

Options:
  --repo-root PATH          Repo used inside Nibi jobs
  --source-variant-stamp S  Default: 20260514T002301Z
  --cache-stamp S           Default: 20260514T002301Z
  --stamp STAMP             New run stamp
  --epochs N                Default: 45
  --batch-size N            Default: 32
  --num-workers N           Default: 4
  --chain-depth N           Default: 4
  --stop-after-seconds N    Default: 9000
  --gres SPEC               Default: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
  --gpu-exclude LIST        Default: empty
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
    --chain-depth) CHAIN_DEPTH="$2"; shift 2 ;;
    --stop-after-seconds) STOP_AFTER_SECONDS="$2"; shift 2 ;;
    --coordinator-time) COORDINATOR_TIME="$2"; shift 2 ;;
    --coordinator-cpus) COORDINATOR_CPUS="$2"; shift 2 ;;
    --coordinator-mem) COORDINATOR_MEM="$2"; shift 2 ;;
    --gpu-time) GPU_TIME="$2"; shift 2 ;;
    --gpu-cpus) GPU_CPUS="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --gres) GPU_GRES="$2"; shift 2 ;;
    --gpu-exclude) GPU_EXCLUDE="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/e26_onc_only_experts_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
VARIANT_ROOT="$WEEKEND_ROOT/manifests/e26_onc_only_experts_${STAMP}"

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
    --chain-depth "$CHAIN_DEPTH"
    --stop-after-seconds "$STOP_AFTER_SECONDS"
    --gpu-time "$GPU_TIME"
    --gpu-cpus "$GPU_CPUS"
    --gpu-mem "$GPU_MEM"
    --gres "$GPU_GRES"
    --gpu-exclude "$GPU_EXCLUDE"
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

echo "Started E26 ONC-only expert ablation at $(date -Is)"
echo "Host: $(hostname)"

FINAL2025="$FINAL2025_ROOT"
WEEKEND="$WEEKEND_ROOT"
REPO="$REPO_ON_NIBI"
SOURCE_VARIANT_ROOT="$WEEKEND/manifests/multiband_variants_$SOURCE_VARIANT_STAMP"
INPUT_VARIANT="$SOURCE_VARIANT_ROOT/E16_e13_multiband_oncrare_full"
CACHE_DIR="$DEF_ROOT/mat_archives/multiband40s_$CACHE_STAMP"
ARCHIVE_PATH="$CACHE_DIR/multiband40s_mat_cache.tar"
EXTRACT_DIR="$CACHE_DIR/extracted"
SUBMITTED_TSV="$PIPELINE_DIR/e26_training_submitted.tsv"
PLAN_TSV="$PIPELINE_DIR/e26_training_plan.tsv"
REPORT_DIR="$PIPELINE_DIR/e26_report"
BASE_CKPT="$FINAL2025/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

mkdir -p "$PIPELINE_DIR" "$LOG_DIR" "$VARIANT_ROOT" "$REPORT_DIR"
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
if [[ ! -d "$EXTRACT_DIR/mat_files" ]]; then
  echo "Missing extracted MAT cache: $EXTRACT_DIR/mat_files" >&2
  echo "Refusing to re-extract automatically because /project inode use can be tight." >&2
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
existing_count="$(find "$EXTRACT_DIR/mat_files" -type f -name '*.mat' 2>/dev/null | wc -l | tr -d ' ')"
echo "Extracted MAT count: $existing_count / expected $expected_count"
if [[ "$existing_count" -lt "$expected_count" ]]; then
  echo "Extracted cache is incomplete; refusing to submit E26 until cache is repaired." >&2
  exit 2
fi

echo "Running E26 smoke checks"
python -u scripts/analysis/e26_collect_onc_only_report.py \
  --submitted-tsv "$PIPELINE_DIR/does_not_exist.tsv" \
  --plan-tsv "$PIPELINE_DIR/does_not_exist.tsv" \
  --output-dir "$PIPELINE_DIR/e26_report_smoke"
python -u scripts/train/train_multiband_multilabel.py --help >/dev/null

echo "Building E26 ONC-only variants"
python -u scripts/data/multilabel/build_e26_onc_only_expert_variants.py \
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
echo -e "experiment\tvariant\tmanifest\tvocab\tdataset_root\tencoder\tfusion\thead_type\tlr\tweight_decay\tdropout\tuse_pos_weight\tloss_mode\tband_availability\tclass_band_mask\tcrop_mode\tcrop_seconds\tband_crop_shapes\tbatch_size\tbands\teval_label_ids\tcalibration_source_kind\teval_source_kind\trun_dir\tarchive_path\twandb_group" > "$PLAN_TSV"
final_stage_job_ids=()

submit_chain() {
  local variant_name="$1"
  local suffix="$2"
  local encoder="$3"
  local fusion="$4"
  local head_type="$5"
  local train_lr="$6"
  local use_pos_weight="$7"
  local loss_mode="$8"
  local band_availability="$9"
  local class_band_mask="${10}"
  local crop_mode="${11}"
  local crop_seconds="${12}"
  local band_crop_shapes="${13}"
  local batch_size="${14}"
  local bands_arg="${15}"
  local eval_label_ids="${16}"
  local calibration_source_kind="${17}"
  local eval_source_kind="${18}"
  local example_band="${19}"
  local dropout="${20}"
  local variant_dir="$VARIANT_ROOT/$variant_name"
  local run_exp="${variant_name}_${suffix}"
  local run_dir="$WEEKEND/runs/${run_exp}_${STAMP}"
  local run_log_dir="$run_dir/logs"
  local train_dir="$run_dir/train"
  local exclude_directive=""
  if [[ -n "$GPU_EXCLUDE" ]]; then
    exclude_directive="#SBATCH --exclude=$GPU_EXCLUDE"
  fi
  mkdir -p "$run_log_dir" "$train_dir"
  local job_script="$run_log_dir/${run_exp}.sbatch"
  cat > "$job_script" <<TRAIN_EOF
#!/bin/bash
#SBATCH --job-name=$run_exp
#SBATCH --output=$run_log_dir/slurm-%j.out
#SBATCH --time=$GPU_TIME
#SBATCH --cpus-per-task=$GPU_CPUS
#SBATCH --mem=$GPU_MEM
#SBATCH --gres=$GPU_GRES
$exclude_directive

set -euo pipefail
echo "Starting $run_exp stage \${E26_STAGE:-1} at \$(date -Is)"
cd "$REPO_ON_NIBI"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
mkdir -p "$train_dir"
if python - "$train_dir/run_summary.json" "$EPOCHS" "$train_dir/onc_calibrated_eval/onc_calibrated_metrics_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
epochs = int(sys.argv[2])
metrics = Path(sys.argv[3])
if not summary.exists() or not metrics.exists():
    raise SystemExit(1)
payload = json.loads(summary.read_text())
history = payload.get("history") or []
max_epoch = max((int(item.get("epoch", 0)) for item in history), default=0)
raise SystemExit(0 if max_epoch >= epochs else 1)
PY
then
  echo "$run_exp already complete through $EPOCHS epochs; skipping."
  exit 0
fi
train_cmd=(
  python -u scripts/train/train_multiband_multilabel.py
  --manifest-csv "$variant_dir/standardized_manifest.csv"
  --vocab-json "$variant_dir/label_vocabulary.json"
  --dataset-root "$EXTRACT_DIR"
  --exp-dir "$train_dir"
  --bands "$bands_arg"
  --band-crop-shapes "$band_crop_shapes"
  --encoder "$encoder"
  --fusion "$fusion"
  --head-type "$head_type"
  --dropout "$dropout"
  --loss-mode "$loss_mode"
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
  --stop-after-seconds "$STOP_AFTER_SECONDS"
)
if [[ "$use_pos_weight" == "true" ]]; then
  train_cmd+=(--use-pos-weight)
fi
if [[ -f "$train_dir/last.pt" ]]; then
  train_cmd+=(--resume-checkpoint "$train_dir/last.pt")
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
  "fusion": "$fusion",
  "head_type": "$head_type",
  "dropout": $dropout,
  "manifest_csv": "$variant_dir/standardized_manifest.csv",
  "vocab_json": "$variant_dir/label_vocabulary.json",
  "dataset_root": "$EXTRACT_DIR",
  "mat_archive": "$ARCHIVE_PATH",
  "train_dir": "$train_dir",
  "use_pos_weight": "$use_pos_weight",
  "loss_mode": "$loss_mode",
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
  "batch_size": $batch_size,
  "chain_depth": $CHAIN_DEPTH,
  "stop_after_seconds": $STOP_AFTER_SECONDS
}
META
echo "Finished $run_exp stage \${E26_STAGE:-1} at \$(date -Is)"
TRAIN_EOF
  echo -e "$run_exp\t$variant_name\t$variant_dir/standardized_manifest.csv\t$variant_dir/label_vocabulary.json\t$EXTRACT_DIR\t$encoder\t$fusion\t$head_type\t$train_lr\t$WEIGHT_DECAY\t$dropout\t$use_pos_weight\t$loss_mode\t$band_availability\t$class_band_mask\t$crop_mode\t$crop_seconds\t$band_crop_shapes\t$batch_size\t$bands_arg\t$eval_label_ids\t$calibration_source_kind\t$eval_source_kind\t$run_dir\t$ARCHIVE_PATH\t$WANDB_GROUP" >> "$PLAN_TSV"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "DRY_RUN\t$run_exp\t$run_dir\t$job_script" >> "$SUBMITTED_TSV"
    echo "DRY_RUN: wrote $job_script"
    return
  fi
  local previous_job=""
  local last_job=""
  local stage
  for stage in $(seq 1 "$CHAIN_DEPTH"); do
    local sbatch_args=(--export=ALL,E26_STAGE="$stage")
    if [[ -n "$previous_job" ]]; then
      sbatch_args+=(--dependency=afterany:"$previous_job")
    fi
    last_job="$(sbatch "${sbatch_args[@]}" "$job_script" | awk '{print $4}')"
    previous_job="$last_job"
    echo "Submitted $run_exp stage $stage as $last_job"
  done
  final_stage_job_ids+=("$last_job")
  echo -e "$last_job\t$run_exp\t$run_dir\t$job_script" >> "$SUBMITTED_TSV"
}

submit_report_job() {
  if [[ "$DRY_RUN" == "true" || "${#final_stage_job_ids[@]}" -eq 0 ]]; then
    return
  fi
  local dep
  dep="$(IFS=:; echo "${final_stage_job_ids[*]}")"
  local report_script="$LOG_DIR/e26_collect_report.sbatch"
  cat > "$report_script" <<REPORT_EOF
#!/bin/bash
#SBATCH --job-name=E26collect_report
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "$REPO_ON_NIBI"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
python -u scripts/analysis/e26_collect_onc_only_report.py \\
  --submitted-tsv "$SUBMITTED_TSV" \\
  --plan-tsv "$PLAN_TSV" \\
  --variant-root "$VARIANT_ROOT" \\
  --output-dir "$REPORT_DIR"
REPORT_EOF
  local report_job
  report_job="$(sbatch --dependency=afterany:$dep "$report_script" | awk '{print $4}')"
  echo -e "$report_job\tE26_collect_report\t$REPORT_DIR\t$report_script" >> "$PIPELINE_DIR/e26_report_submitted.tsv"
  echo "Submitted E26 report collector as $report_job after jobs: $dep"
}

SHAPES_10S="low:391x50,mid:256x100,high:256x312"

submit_chain "E26_fin_whale_low_onc_only" "r18_lr3e4_c10_d03" "resnet18" "gated" "shared" "0.0003" "false" "balanced_bce" "all" "audit_v2" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low" "species:Bp" "ONC" "ONC" "low" "0.3"
submit_chain "E26_blue_whale_low_onc_only" "r18_lr3e4_c10_d03" "resnet18" "gated" "shared" "0.0003" "false" "balanced_bce" "all" "audit_v2" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low" "species:Bm" "ONC" "ONC" "low" "0.3"
submit_chain "E26_humpback_whale_lowmid_onc_only" "r18_lr1e4_c10_d03" "resnet18" "gated" "shared" "0.0001" "false" "balanced_bce" "all" "audit_v2" "centered_gaussian" "10" "$SHAPES_10S" "$BATCH_SIZE" "low,mid" "species:Mn" "ONC" "ONC" "mid" "0.3"

submit_report_job

echo "Training plan:"
cat "$PLAN_TSV"
echo "Submitted jobs:"
cat "$SUBMITTED_TSV"
if [[ -f "$PIPELINE_DIR/e26_report_submitted.tsv" ]]; then
  echo "Report collector:"
  cat "$PIPELINE_DIR/e26_report_submitted.tsv"
fi
echo "Report dir: $REPORT_DIR"
echo "Finished E26 ONC-only expert ablation submission at $(date -Is)"
