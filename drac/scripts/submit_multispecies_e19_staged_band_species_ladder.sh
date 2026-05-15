#!/bin/bash
# Reuse the E16 combined multiband MAT archive and launch a staged
# species/band ladder that validates the fusion architecture one step at a time.

set -euo pipefail

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
DEF_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e19_staged_ladder"
SOURCE_VARIANT_STAMP="20260514T002301Z"
CACHE_STAMP="20260514T002301Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E19staged_submit"

EPOCHS="45"
BATCH_SIZE="64"
NUM_WORKERS="8"
SEED="2026"
WEIGHT_DECAY="0.0001"
SBATCH_TIME="02:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="24G"
GPU_TIME="12:00:00"
GPU_MEM="96G"
SBATCH_GRES="gpu:h100:1"
WANDB_GROUP="weekend-20260502-e19-staged-band-species"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e19_staged_band_species_ladder.sh [options]

This submits one small CPU coordination job that reuses the E16 combined
multiband archive, builds clean staged variants with reduced label
vocabularies, then launches a bounded H100 ladder.

The ladder is designed to test one source/band/species complexity at a time:
  1. Bp-only low-band probe with ONC + BioDCASE fin whales.
  2. Bm-only low-band probe with ONC + BioDCASE blue whales.
  3. Bp+Bm low-band cumulative baleen probe.
  4. Mn-only low+mid-band probe with ONC + DCLDE humpbacks.
  5. Bp+Bm+Mn low+mid cumulative probe.
  6. Oo-only mid+high probe with ONC + DCLDE killer whales.
  7. Full low+mid+high routed fusion probe.
  8. Full low+mid+high routed fusion with positive weights as a control.

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

PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/e19_staged_band_species_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
JOB_SCRIPT="$LOG_DIR/e19_staged_band_species_${STAMP}.sbatch"
mkdir -p "$LOG_DIR"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail

echo "Started E19 staged band/species ladder at \$(date -Is)"
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
VARIANT_ROOT="\$WEEKEND/manifests/e19_staged_band_species_\$STAMP"
CACHE_DIR="\$DEF_ROOT/mat_archives/multiband40s_\$CACHE_STAMP"
ARCHIVE_PATH="\$CACHE_DIR/multiband40s_mat_cache.tar"
EXTRACT_DIR="\$CACHE_DIR/extracted"
SUBMITTED_TSV="\$PIPELINE_DIR/e19_training_submitted.tsv"
PLAN_TSV="\$PIPELINE_DIR/e19_training_plan.tsv"
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
import json
import sys
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

echo "Building E19 staged species/band variants"
python - "\$INPUT_VARIANT/standardized_manifest.csv" "\$INPUT_VARIANT/label_vocabulary.json" "\$VARIANT_ROOT" <<'PY'
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

manifest_csv = Path(sys.argv[1])
vocab_json = Path(sys.argv[2])
variant_root = Path(sys.argv[3])
variant_root.mkdir(parents=True, exist_ok=True)

variants = [
    {
        "name": "E19_bp_low_probe",
        "active_label_ids": ["species:Bp"],
        "eval_label_ids": ["species:Bp"],
        "sources": ["ONC", "BioDCASE"],
        "bands": ["low"],
        "description": "Fin-whale-only low-band probe with ONC + BioDCASE Bp.",
    },
    {
        "name": "E19_bm_low_probe",
        "active_label_ids": ["species:Bm"],
        "eval_label_ids": ["species:Bm"],
        "sources": ["ONC", "BioDCASE"],
        "bands": ["low"],
        "description": "Blue-whale-only low-band probe with ONC + BioDCASE Bm.",
    },
    {
        "name": "E19_bp_bm_low_cumulative",
        "active_label_ids": ["species:Bp", "species:Bm"],
        "eval_label_ids": ["species:Bp", "species:Bm"],
        "sources": ["ONC", "BioDCASE"],
        "bands": ["low"],
        "description": "Cumulative low-band baleen probe with Bp and Bm.",
    },
    {
        "name": "E19_mn_lowmid_probe",
        "active_label_ids": ["species:Mn"],
        "eval_label_ids": ["species:Mn"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["low", "mid"],
        "description": "Humpback-only low+mid probe with ONC + DCLDE Mn.",
    },
    {
        "name": "E19_bp_bm_mn_lowmid_cumulative",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "description": "Cumulative low+mid probe after adding Mn.",
    },
    {
        "name": "E19_oo_midhigh_probe",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid", "high"],
        "description": "Killer-whale-only mid+high probe with ONC + DCLDE Oo.",
    },
    {
        "name": "E19_full_routed_allbands",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn", "species:Oo"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn", "species:Oo"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid", "high"],
        "description": "Full routed low+mid+high fusion probe.",
    },
]

def clean(value):
    return str(value or "").strip()

def split_pipe(value):
    return [token.strip() for token in clean(value).split("|") if token.strip()]

def labels(row):
    for key in ("label_ids", "target_label_ids", "canonical_label_ids", "analysis_label_ids"):
        value = clean(row.get(key))
        if value:
            return tuple(split_pipe(value))
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

def subset_vocab(vocab_payload, active_labels):
    active = set(active_labels)
    labels_out = [dict(label) for label in vocab_payload.get("labels", []) if str(label.get("id")) in active]
    missing = sorted(active.difference(str(label.get("id")) for label in labels_out))
    if missing:
        raise SystemExit(f"Active labels missing from vocabulary: {missing}")
    return {"schema_version": vocab_payload.get("schema_version", "multilabel-v1"), "labels": labels_out}

def rewrite_label_fields(row, active_labels):
    active = set(active_labels)
    labs = [label for label in labels(row) if label in active]
    out = dict(row)
    text = "|".join(labs)
    for key in ("label_ids", "canonical_label_ids", "target_label_ids"):
        if key in out:
            out[key] = text
    if "is_background" in out:
        out["is_background"] = "0" if labs else "1"
    return out

def keep_row(row, variant):
    source = clean(row.get("source_kind"))
    if source not in set(variant["sources"]):
        return False
    active = set(variant["active_label_ids"])
    labs = set(labels(row))
    if not labs:
        return True
    return bool(labs.intersection(active))

def summarize(selected_rows, variant, fieldnames):
    split_counts = Counter(clean(r.get("split")) or "<blank>" for r in selected_rows)
    split_source_label = Counter(
        (clean(r.get("split")) or "<blank>", clean(r.get("source_kind")) or "<blank>", label_key(r), clean(r.get("negative_bucket")) or "")
        for r in selected_rows
    )
    time_counts = Counter(
        (clean(r.get("split")) or "<blank>", clean(r.get("source_kind")) or "<blank>", label_key(r), month_bin(r))
        for r in selected_rows
    )
    missing_by_band = {
        band: sum(1 for row in selected_rows if not clean(row.get(f"{band}_mat_path")))
        for band in variant["bands"]
    }
    return {
        "variant_name": variant["name"],
        "description": variant["description"],
        "active_label_ids": variant["active_label_ids"],
        "eval_label_ids": variant["eval_label_ids"],
        "sources": variant["sources"],
        "bands": variant["bands"],
        "row_count": len(selected_rows),
        "split_counts": dict(split_counts.most_common()),
        "missing_mat_path_by_band": missing_by_band,
        "columns": fieldnames,
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

with manifest_csv.open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
fieldnames = list(rows[0].keys())
vocab_payload = json.loads(vocab_json.read_text(encoding="utf-8"))

index = []
for variant in variants:
    out_dir = variant_root / variant["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        rewrite_label_fields(row, variant["active_label_ids"])
        for row in rows
        if keep_row(row, variant)
    ]
    with (out_dir / "standardized_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    (out_dir / "label_vocabulary.json").write_text(
        json.dumps(subset_vocab(vocab_payload, variant["active_label_ids"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = summarize(selected, variant, fieldnames)
    summary.update({
        "input_manifest": str(manifest_csv),
        "manifest_csv": str(out_dir / "standardized_manifest.csv"),
        "vocab_json": str(out_dir / "label_vocabulary.json"),
    })
    (out_dir / "manifest_variant_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    index.append(summary)
    print(f"{variant['name']}: {len(selected)} rows; splits={summary['split_counts']}; labels={variant['active_label_ids']}; bands={variant['bands']}")

(variant_root / "variant_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "Queue/accounting check before GPU submissions"
squeue -u merileo || true
sacct -u merileo --starttime now-7days || true
timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true

echo -e "job_id\texperiment\trun_dir\tjob_script" > "\$SUBMITTED_TSV"
echo -e "experiment\tvariant\tmanifest\tvocab\tdataset_root\tencoder\tfusion\tlr\tweight_decay\tuse_pos_weight\tband_availability\tclass_band_mask\tcrop_mode\tbands\teval_label_ids\trun_dir\tarchive_path\twandb_group" > "\$PLAN_TSV"

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
  local eval_label_ids="\${10}"
  local example_band="\${11}"
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
  --bands "\$bands_arg"
  --band-crop-shapes low:391x50,mid:256x100,high:256x312
  --encoder "\$encoder"
  --fusion gated
  --init-all-branches-checkpoint "\$BASE_CKPT"
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
  --example-image-band "\$example_band"
  --use-wandb
  --wandb-project whale-multispecies-calltype
  --wandb-group "$WANDB_GROUP"
  --wandb-name "\$run_exp"
  --wandb-tags "multilabel,species,E19,staged_band_species,\$variant_name,\$encoder,\$bands_arg,\$band_availability,\$class_band_mask,\$crop_mode"
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
  --eval-source-kind ONC \\
  --label-ids "\$eval_label_ids"
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
  "eval_label_ids": "\$eval_label_ids",
  "band_availability_mode": "\$band_availability",
  "class_band_mask_mode": "\$class_band_mask",
  "crop_time_seconds": 10,
  "positive_crop_mode": "\$crop_mode",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE
}
META
TRAIN_EOF
  echo -e "\$run_exp\t\$variant_name\t\$variant_dir/standardized_manifest.csv\t\$variant_dir/label_vocabulary.json\t\$EXTRACT_DIR\t\$encoder\tgated\t\$train_lr\t$WEIGHT_DECAY\t\$use_pos_weight\t\$band_availability\t\$class_band_mask\t\$crop_mode\t\$bands_arg\t\$eval_label_ids\t\$run_dir\t\$ARCHIVE_PATH\t$WANDB_GROUP" >> "\$PLAN_TSV"
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

submit_train "E19_bp_low_probe" "r18_noposw" "resnet18" "0.0003" "false" "all" "none" "centered_gaussian" "low" "species:Bp" "low"
submit_train "E19_bm_low_probe" "r18_noposw" "resnet18" "0.0003" "false" "all" "none" "centered_gaussian" "low" "species:Bm" "low"
submit_train "E19_bp_bm_low_cumulative" "r18_noposw" "resnet18" "0.0003" "false" "all" "none" "centered_gaussian" "low" "species:Bp,species:Bm" "low"
submit_train "E19_mn_lowmid_probe" "r18_noposw" "resnet18" "0.0003" "false" "all" "none" "centered_gaussian" "low,mid" "species:Mn" "mid"
submit_train "E19_bp_bm_mn_lowmid_cumulative" "r18_noposw" "resnet18" "0.0003" "false" "all" "audit_v2" "centered_gaussian" "low,mid" "species:Bp,species:Bm,species:Mn" "mid"
submit_train "E19_oo_midhigh_probe" "r18_noposw" "resnet18" "0.0003" "false" "all" "none" "centered_gaussian" "mid,high" "species:Oo" "high"
submit_train "E19_full_routed_allbands" "r18_noposw" "resnet18" "0.0003" "false" "source_or_metadata" "odont_high" "centered_gaussian" "low,mid,high" "species:Bp,species:Bm,species:Mn,species:Oo" "high"
submit_train "E19_full_routed_allbands" "r18_posw" "resnet18" "0.0003" "true" "source_or_metadata" "odont_high" "centered_gaussian" "low,mid,high" "species:Bp,species:Bm,species:Mn,species:Oo" "high"

echo "Training plan:"
cat "\$PLAN_TSV"
echo "Submitted jobs:"
cat "\$SUBMITTED_TSV"
echo "Finished E19 staged band/species ladder at \$(date -Is)"
EOF

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: wrote $JOB_SCRIPT"
else
  job_id="$(sbatch "$JOB_SCRIPT" | awk '{print $4}')"
  echo "Submitted $RUN_NAME as $job_id"
  echo "Pipeline dir: $PIPELINE_DIR"
  echo "Slurm log: $LOG_DIR/slurm-$job_id.out"
fi
