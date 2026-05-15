#!/bin/bash
# Reuse the E16 combined multiband MAT archive and launch a low-band
# incremental external-species ladder.

set -euo pipefail

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
DEF_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e18_species_ladder"
SOURCE_VARIANT_STAMP="20260514T002301Z"
CACHE_STAMP="20260514T002301Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E18species_submit"

EPOCHS="45"
BATCH_SIZE="64"
NUM_WORKERS="8"
SEED="2026"
WEIGHT_DECAY="0.0001"
SBATCH_TIME="02:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="24G"
GPU_TIME="12:00:00"
GPU_MEM="80G"
SBATCH_GRES="gpu:h100:1"
WANDB_GROUP="weekend-20260502-e18-incremental-species-lowband"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e18_incremental_species_ladder.sh [options]

This submits one small CPU coordination job that reuses the E16 combined
multiband archive, builds low-band-only manifest variants that add external
species one source/label at a time, then launches a bounded H100 ladder.

Options:
  --repo-root PATH          Repo used inside Nibi jobs
  --source-variant-stamp S  Default: 20260514T002301Z
  --cache-stamp S           Default: 20260514T002301Z
  --stamp STAMP             New run stamp
  --epochs N                Default: 45
  --batch-size N            Default: 64
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

PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/e18_incremental_species_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
JOB_SCRIPT="$LOG_DIR/e18_incremental_species_${STAMP}.sbatch"
mkdir -p "$LOG_DIR"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail

echo "Started E18 incremental species ladder at \$(date -Is)"
echo "Host: \$(hostname)"

FINAL2025="$FINAL2025_ROOT"
WEEKEND="$WEEKEND_ROOT"
DEF_ROOT="$DEF_ROOT"
REPO="$REPO_ON_NIBI"
SOURCE_VARIANT_STAMP="$SOURCE_VARIANT_STAMP"
CACHE_STAMP="$CACHE_STAMP"
STAMP="$STAMP"
PIPELINE_DIR="$PIPELINE_DIR"
SOURCE_VARIANT_ROOT="\$WEEKEND/manifests/multiband_variants_\$SOURCE_VARIANT_STAMP"
INPUT_VARIANT="\$SOURCE_VARIANT_ROOT/E16_e13_multiband_oncrare_full"
VARIANT_ROOT="\$WEEKEND/manifests/e18_incremental_species_\$STAMP"
CACHE_DIR="\$DEF_ROOT/mat_archives/multiband40s_\$CACHE_STAMP"
ARCHIVE_PATH="\$CACHE_DIR/multiband40s_mat_cache.tar"
EXTRACT_DIR="\$CACHE_DIR/extracted"
SUBMITTED_TSV="\$PIPELINE_DIR/e18_training_submitted.tsv"
PLAN_TSV="\$PIPELINE_DIR/e18_training_plan.tsv"
BASE_CKPT="\$FINAL2025/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

mkdir -p "\$PIPELINE_DIR" "\$VARIANT_ROOT"
cd "\$REPO"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_CACHE_DIR" "\$PIP_CACHE_DIR"

echo "Repo: \$REPO"
git rev-parse HEAD || true
echo "Input variant: \$INPUT_VARIANT"
echo "Cache dir: \$CACHE_DIR"
timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true

if [[ ! -f "\$ARCHIVE_PATH" ]]; then
  echo "Missing archive: \$ARCHIVE_PATH" >&2
  exit 2
fi
if [[ ! -f "\$INPUT_VARIANT/standardized_manifest.csv" ]]; then
  echo "Missing input manifest: \$INPUT_VARIANT/standardized_manifest.csv" >&2
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

echo "Building E18 low-band incremental species variants"
python - "\$INPUT_VARIANT/standardized_manifest.csv" "\$INPUT_VARIANT/label_vocabulary.json" "\$VARIANT_ROOT" <<'PY'
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

manifest_csv = Path(sys.argv[1])
vocab_json = Path(sys.argv[2])
variant_root = Path(sys.argv[3])
variant_root.mkdir(parents=True, exist_ok=True)

variants = [
    ("E18_onc_only_low", {}),
    ("E18_onc_biod_bp_low", {"BioDCASE": ["species:Bp"]}),
    ("E18_onc_biod_bp_bm_low", {"BioDCASE": ["species:Bp", "species:Bm"]}),
    ("E18_onc_biod_bp_bm_dclde_mn_low", {"BioDCASE": ["species:Bp", "species:Bm"], "DCLDE": ["species:Mn"]}),
    ("E18_onc_biod_bp_bm_dclde_mn_oo_low", {"BioDCASE": ["species:Bp", "species:Bm"], "DCLDE": ["species:Mn", "species:Oo"]}),
]

def clean(value):
    return str(value or "").strip()

def labels(row):
    for key in ("label_ids", "target_label_ids", "canonical_label_ids", "analysis_label_ids"):
        value = clean(row.get(key))
        if value:
            return tuple(token.strip() for token in value.split("|") if token.strip())
    return tuple()

def label_key(row):
    labs = labels(row)
    return "|".join(labs) if labs else "<background>"

date_patterns = [
    re.compile(r"(20\\d{2})[-_]?([01]\\d)[-_]?([0-3]\\d)"),
    re.compile(r"(20\\d{2})[-_]?([01]\\d)"),
]

def month_bin(row):
    text = " ".join(clean(row.get(key)) for key in ("clip", "source_audio", "item_id", "mat_path", "expected_mat_name"))
    for pattern in date_patterns:
        match = pattern.search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
    return "<unknown>"

with manifest_csv.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
fieldnames = list(rows[0].keys())

def keep_row(row, allowed):
    source = clean(row.get("source_kind"))
    if source == "ONC":
        return True
    source_allowed = set(allowed.get(source, []))
    if not source_allowed:
        return False
    labs = set(labels(row))
    if not labs:
        return True
    return bool(labs.intersection(source_allowed))

def summarize(selected_rows, allowed):
    split_counts = Counter(clean(r.get("split")) or "<blank>" for r in selected_rows)
    split_source_label = Counter(
        (clean(r.get("split")) or "<blank>", clean(r.get("source_kind")) or "<blank>", label_key(r), clean(r.get("negative_bucket")) or "")
        for r in selected_rows
    )
    time_counts = Counter(
        (clean(r.get("split")) or "<blank>", clean(r.get("source_kind")) or "<blank>", label_key(r), month_bin(r))
        for r in selected_rows
    )
    return {
        "row_count": len(selected_rows),
        "allowed_external_labels": allowed,
        "split_counts": dict(split_counts.most_common()),
        "split_source_label_counts": [
            {
                "split": split,
                "source_kind": source,
                "label": label,
                "negative_bucket": bucket,
                "rows": count,
            }
            for (split, source, label, bucket), count in split_source_label.most_common()
        ],
        "time_counts": [
            {
                "split": split,
                "source_kind": source,
                "label": label,
                "month": month,
                "rows": count,
            }
            for (split, source, label, month), count in time_counts.most_common()
        ],
    }

index = []
for name, allowed in variants:
    out_dir = variant_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [dict(row) for row in rows if keep_row(row, allowed)]
    with (out_dir / "standardized_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    shutil.copy2(vocab_json, out_dir / "label_vocabulary.json")
    summary = summarize(selected, allowed)
    summary.update({
        "variant_name": name,
        "input_manifest": str(manifest_csv),
        "manifest_csv": str(out_dir / "standardized_manifest.csv"),
        "vocab_json": str(out_dir / "label_vocabulary.json"),
    })
    (out_dir / "manifest_variant_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    index.append(summary)
    print(f"{name}: {len(selected)} rows")

(variant_root / "variant_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
PY

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
  local crop_mode="\$6"
  local variant_dir="\$VARIANT_ROOT/\$variant_name"
  local run_exp="\${variant_name}_\${suffix}"
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
  --bands low
  --band-crop-shapes low:391x50
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
  --band-availability-mode all
  --class-band-mask-mode none
  --device cuda
  --seed "$SEED"
  --example-image-band low
  --use-wandb
  --wandb-project whale-multispecies-calltype
  --wandb-group "$WANDB_GROUP"
  --wandb-name "\$run_exp"
  --wandb-tags "multilabel,species,E18,lowband,incremental_species,\$variant_name,\$encoder,\$crop_mode"
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
  "bands": "low",
  "band_availability_mode": "all",
  "class_band_mask_mode": "none",
  "crop_time_seconds": 10,
  "positive_crop_mode": "\$crop_mode",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE
}
META
TRAIN_EOF
  echo -e "\$run_exp\t\$variant_name\t\$variant_dir/standardized_manifest.csv\t\$variant_dir/label_vocabulary.json\t\$EXTRACT_DIR\t\$encoder\tgated\t\$train_lr\t$WEIGHT_DECAY\t\$use_pos_weight\tall\tnone\t\$crop_mode\tlow\t\$run_dir\t\$ARCHIVE_PATH\t$WANDB_GROUP" >> "\$PLAN_TSV"
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

submit_train "E18_onc_only_low" "r18_noposw_center" "resnet18" "0.0003" "false" "centered_gaussian"
submit_train "E18_onc_biod_bp_low" "r18_noposw_center" "resnet18" "0.0003" "false" "centered_gaussian"
submit_train "E18_onc_biod_bp_bm_low" "r18_noposw_center" "resnet18" "0.0003" "false" "centered_gaussian"
submit_train "E18_onc_biod_bp_bm_dclde_mn_low" "r18_noposw_center" "resnet18" "0.0003" "false" "centered_gaussian"
submit_train "E18_onc_biod_bp_bm_dclde_mn_oo_low" "r18_noposw_center" "resnet18" "0.0003" "false" "centered_gaussian"
submit_train "E18_onc_biod_bp_bm_dclde_mn_oo_low" "r18_posw_center" "resnet18" "0.0003" "true" "centered_gaussian"

echo "Training plan:"
cat "\$PLAN_TSV"
echo "Submitted jobs:"
cat "\$SUBMITTED_TSV"
echo "Finished E18 incremental species ladder at \$(date -Is)"
EOF

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: wrote $JOB_SCRIPT"
else
  job_id="$(sbatch "$JOB_SCRIPT" | awk '{print $4}')"
  echo "Submitted $RUN_NAME as $job_id"
  echo "Pipeline dir: $PIPELINE_DIR"
  echo "Slurm log: $LOG_DIR/slurm-$job_id.out"
fi
