#!/bin/bash
# Build a reusable low/mid/high 40s MAT archive and launch bounded full-source training jobs.

set -euo pipefail

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
DEF_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_multiband_20260513"
SOURCE_STAMP="20260507T195430Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="E16prep_multiband"

EPOCHS="45"
BATCH_SIZE="32"
NUM_WORKERS="8"
SEED="2026"
WEIGHT_DECAY="0.0001"
ONC_RARE_TARGET="10000"
PREP_PARALLEL_JOBS="12"
SBATCH_TIME="24:00:00"
SBATCH_CPUS="24"
SBATCH_MEM="160G"
GPU_TIME="12:00:00"
GPU_MEM="96G"
SBATCH_GRES="gpu:h100:1"
WANDB_GROUP="weekend-20260502-multiband-fusion"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_multiband_pipeline.sh [options]

This submits one CPU job that reuses the full source-calibrated source CSVs,
creates low/mid/high 40s MATs, packages them into one reusable tar archive
under def-kmoran project storage, extracts the archive once, then submits a
bounded multiband H100 ladder.

Options:
  --repo-root PATH          Repo used inside Nibi jobs
  --source-stamp STAMP      Source manifest stamp, default 20260507T195430Z
  --stamp STAMP             New run stamp
  --def-root PATH           Durable/project cache root
  --epochs N                Default: 45
  --batch-size N            Default: 32
  --prep-parallel-jobs N    Default: 12
  --dry-run                 Write sbatch scripts but do not submit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --final2025-root) FINAL2025_ROOT="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --def-root) DEF_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --source-stamp) SOURCE_STAMP="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prep-parallel-jobs) PREP_PARALLEL_JOBS="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --gpu-time) GPU_TIME="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --onc-rare-target) ONC_RARE_TARGET="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/multiband_fusion_${STAMP}"
LOG_DIR="$PIPELINE_DIR/logs"
JOB_SCRIPT="$LOG_DIR/multiband_fusion_${STAMP}.sbatch"
mkdir -p "$LOG_DIR"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail

echo "Started multiband fusion pipeline at \$(date -Is)"
echo "Host: \$(hostname)"

FINAL2025="$FINAL2025_ROOT"
WEEKEND="$WEEKEND_ROOT"
DEF_ROOT="$DEF_ROOT"
REPO="$REPO_ON_NIBI"
SOURCE_STAMP="$SOURCE_STAMP"
STAMP="$STAMP"
PIPELINE_DIR="$PIPELINE_DIR"
SOURCE_DIR="\$WEEKEND/manifests/scaleup40s_sources_\$SOURCE_STAMP"
MANIFEST_ROOT="\$WEEKEND/manifests"
SOURCE_E12="\$MANIFEST_ROOT/E12large_onc_dclde_40sctx_autoneg_species"
SOURCE_E13="\$MANIFEST_ROOT/E13large_onc_biod_dclde_40sctx_autoneg_species"
CACHE_DIR="\$DEF_ROOT/mat_archives/multiband40s_\$STAMP"
BUILD_DIR="\$CACHE_DIR/build"
REPORT_DIR="\$BUILD_DIR/reports"
ARCHIVE_META_DIR="\$CACHE_DIR/archive_meta"
ARCHIVE_PATH="\$CACHE_DIR/multiband40s_mat_cache.tar"
EXTRACT_DIR="\$CACHE_DIR/extracted"
VARIANT_ROOT="\$MANIFEST_ROOT/multiband_variants_\$STAMP"
SUBMITTED_TSV="\$PIPELINE_DIR/multiband_training_submitted.tsv"
PLAN_TSV="\$PIPELINE_DIR/multiband_training_plan.tsv"

ONC_AUDIO="\$FINAL2025/part2/full_bundle/raw_audio"
BIOD_ROOT="\$FINAL2025/multispecies_calltype_experiments/external_data/biodcase2026_task2/extracted/2026_BioDCASE_development_set/train"
BASE_CKPT="\$FINAL2025/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

mkdir -p "\$PIPELINE_DIR" "\$REPORT_DIR" "\$ARCHIVE_META_DIR" "\$EXTRACT_DIR" "\$VARIANT_ROOT"
cd "\$REPO"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_CACHE_DIR" "\$PIP_CACHE_DIR"

echo "Repo: \$REPO"
git rev-parse HEAD || true
echo "Source dir: \$SOURCE_DIR"
echo "Cache dir: \$CACHE_DIR"
diskusage_report || true

if [[ ! -f "\$SOURCE_E13/standardized_manifest.csv" ]]; then
  echo "Missing source E13 manifest: \$SOURCE_E13/standardized_manifest.csv" >&2
  exit 2
fi
if [[ ! -f "\$SOURCE_DIR/onc/selected_calls.csv" ]]; then
  echo "Missing source CSVs under \$SOURCE_DIR" >&2
  exit 2
fi

split_csv() {
  local input_csv="\$1"
  local out_dir="\$2"
  local jobs="\$3"
  rm -rf "\$out_dir"
  mkdir -p "\$out_dir"
  python - "\$input_csv" "\$out_dir" "\$jobs" <<'PY'
import csv
import math
import sys
from pathlib import Path

input_csv = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
jobs = max(1, int(sys.argv[3]))
with input_csv.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
if not rows:
    raise SystemExit(f"No rows in {input_csv}")
chunk_count = min(jobs, len(rows))
chunk_size = int(math.ceil(len(rows) / chunk_count))
for idx in range(chunk_count):
    chunk = rows[idx * chunk_size : (idx + 1) * chunk_size]
    if not chunk:
        continue
    path = out_dir / f"chunk_{idx:03d}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunk)
PY
}

combine_reports() {
  local label="\$1"
  local chunk_out="\$2"
  local combined="\$REPORT_DIR/\${label}_multiband_report.csv"
  python - "\$chunk_out" "\$combined" <<'PY'
import csv
import sys
from pathlib import Path

chunk_out = Path(sys.argv[1])
combined = Path(sys.argv[2])
rows = []
fieldnames = []
for report in sorted(chunk_out.glob("*/multiband_report.csv")):
    with report.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for field in reader.fieldnames or []:
            if field not in fieldnames:
                fieldnames.append(field)
        rows.extend(reader)
if not rows:
    raise SystemExit(f"No multiband reports found under {chunk_out}")
combined.parent.mkdir(parents=True, exist_ok=True)
with combined.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(combined)
PY
}

run_multiband_prepare() {
  local label="\$1"
  local calls_csv="\$2"
  local audio_dir="\$3"
  local chunk_root="\$BUILD_DIR/chunks/\$label"
  local chunk_csv_dir="\$chunk_root/csv"
  local chunk_out_dir="\$chunk_root/out"
  rm -rf "\$chunk_root"
  mkdir -p "\$chunk_csv_dir" "\$chunk_out_dir"
  split_csv "\$calls_csv" "\$chunk_csv_dir" "$PREP_PARALLEL_JOBS"
  echo "Preparing \$label multiband MATs from \$(find "\$chunk_csv_dir" -type f -name '*.csv' | wc -l) chunks"
  local pids=()
  for chunk_csv in "\$chunk_csv_dir"/chunk_*.csv; do
    [[ -f "\$chunk_csv" ]] || continue
    local chunk_name
    chunk_name="\$(basename "\$chunk_csv" .csv)"
    local out_dir="\$chunk_out_dir/\$chunk_name"
    mkdir -p "\$out_dir"
    (
      python -u scripts/data/multilabel/prepare_multiband_context_windows.py \\
        --calls-csv "\$chunk_csv" \\
        --audio-dir "\$audio_dir" \\
        --out-dir "\$out_dir" \\
        --window-s 40 \\
        > "\$out_dir/stdout.log" 2> "\$out_dir/stderr.log"
    ) &
    pids+=(\$!)
  done
  local status=0
  for pid in "\${pids[@]}"; do
    if ! wait "\$pid"; then
      status=1
    fi
  done
  if [[ "\$status" -ne 0 ]]; then
    echo "At least one \$label multiband prep chunk failed" >&2
    find "\$chunk_out_dir" -name stderr.log -maxdepth 2 -type f -print -exec tail -80 {} \\;
    exit "\$status"
  fi
  combine_reports "\$label" "\$chunk_out_dir"
}

stage_biodcase_audio() {
  local raw_dir="\$CACHE_DIR/biodcase_raw_audio"
  mkdir -p "\$raw_dir"
  python - "\$SOURCE_DIR/biodcase/selected_calls.csv" "\$BIOD_ROOT/audio" "\$raw_dir" <<'PY'
import csv
import json
import shutil
import sys
from pathlib import Path

selected = Path(sys.argv[1])
audio_root = Path(sys.argv[2])
raw_dir = Path(sys.argv[3])
raw_dir.mkdir(parents=True, exist_ok=True)
staged = []
missing = []
with selected.open(newline="", encoding="utf-8-sig") as handle:
    for row in csv.DictReader(handle):
        clip = row["clip"]
        src = audio_root / row["source_dataset"] / row["source_audio"]
        dst = raw_dir / clip
        if not src.exists():
            missing.append({"clip": clip, "source": str(src)})
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
        staged.append({"clip": clip, "source": str(src)})
summary = {"selected_count": len(staged) + len(missing), "staged_count": len(staged), "missing_count": len(missing), "missing": missing[:25], "raw_dir": str(raw_dir)}
(raw_dir / "staging_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if missing:
    raise SystemExit(f"{len(missing)} BioDCASE audio files missing")
PY
  echo "\$raw_dir"
}

stage_dclde_audio() {
  local raw_dir="\$CACHE_DIR/dclde_raw_audio"
  mkdir -p "\$raw_dir"
  python - "\$SOURCE_DIR/dclde/selected_calls.csv" "\$raw_dir" "\$SOURCE_DIR/dclde_audio_download_report_multiband.csv" <<'PY'
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import soundfile as sf

selected = Path(sys.argv[1])
raw_dir = Path(sys.argv[2])
report_csv = Path(sys.argv[3])
raw_dir.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(selected.open(newline="", encoding="utf-8-sig")))
by_clip = {}
for row in rows:
    by_clip.setdefault(row["clip"], row)
results = []
for idx, (clip, row) in enumerate(sorted(by_clip.items()), start=1):
    dest = raw_dir / clip
    url = row.get("https_url", "")
    status = "failed"
    error = ""
    for attempt in range(1, 4):
        try:
            if not dest.exists() or dest.stat().st_size == 0:
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                print(f"Downloading DCLDE {idx}/{len(by_clip)} attempt {attempt}: {clip}", flush=True)
                with urllib.request.urlopen(url, timeout=180) as resp, tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                tmp.replace(dest)
            info = sf.info(str(dest))
            status = "ok"
            results.append({"clip": clip, "status": status, "path": str(dest), "bytes": dest.stat().st_size, "samplerate": info.samplerate, "channels": info.channels, "duration": float(info.duration), "url": url})
            break
        except Exception as exc:
            error = repr(exc)
            try:
                if dest.exists() and dest.stat().st_size == 0:
                    dest.unlink()
            except FileNotFoundError:
                pass
            time.sleep(2 * attempt)
    if status != "ok":
        results.append({"clip": clip, "status": "failed", "path": str(dest), "error": error, "url": url})
failed = {row["clip"] for row in results if row["status"] != "ok"}
fieldnames = sorted({key for row in results for key in row})
with report_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
summary = {"audio_count": len(results), "failed_count": len(failed), "failed_examples": sorted(failed)[:20], "report_csv": str(report_csv), "raw_dir": str(raw_dir)}
(report_csv.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if failed:
    raise SystemExit(f"{len(failed)} DCLDE audio files failed")
PY
  echo "\$raw_dir"
}

echo "Preparing multiband MATs"
run_multiband_prepare "onc" "\$SOURCE_DIR/onc/selected_calls.csv" "\$ONC_AUDIO"
run_multiband_prepare "onc_primary_adjacent_gap" "\$SOURCE_DIR/onc_primary_adjacent_gaps_selected.csv" "\$ONC_AUDIO"
BIOD_RAW="\$(stage_biodcase_audio | tail -1)"
run_multiband_prepare "biodcase" "\$SOURCE_DIR/biodcase/selected_calls.csv" "\$BIOD_RAW"
DCLDE_RAW="\$(stage_dclde_audio | tail -1)"
run_multiband_prepare "dclde" "\$SOURCE_DIR/dclde/selected_calls.csv" "\$DCLDE_RAW"
rm -rf "\$BIOD_RAW" "\$DCLDE_RAW"

echo "Building multiband manifests"
E13_MB="\$MANIFEST_ROOT/E16large_onc_biod_dclde_multiband40s_species"
E12_MB="\$MANIFEST_ROOT/E16large_onc_dclde_multiband40s_species"
mkdir -p "\$E13_MB/raw_absolute" "\$E12_MB"
python -u scripts/data/multilabel/build_multiband_manifest.py \\
  --manifest-csv "\$SOURCE_E13/standardized_manifest.csv" \\
  --output-dir "\$E13_MB/raw_absolute" \\
  --report-csv "\$REPORT_DIR/onc_multiband_report.csv" \\
  --report-csv "\$REPORT_DIR/onc_primary_adjacent_gap_multiband_report.csv" \\
  --report-csv "\$REPORT_DIR/biodcase_multiband_report.csv" \\
  --report-csv "\$REPORT_DIR/dclde_multiband_report.csv" \\
  > "\$E13_MB/raw_absolute/build_multiband_manifest_stdout.json"
cp "\$SOURCE_E13/label_vocabulary.json" "\$E13_MB/raw_absolute/label_vocabulary.json"

echo "Creating reusable multiband tar archive in def-kmoran"
python -u scripts/data/multilabel/create_multiband_mat_archive.py \\
  --manifest-csv "\$E13_MB/raw_absolute/standardized_manifest.csv" \\
  --output-dir "\$ARCHIVE_META_DIR" \\
  --archive-path "\$ARCHIVE_PATH" \\
  --archive-format tar \\
  --include-file "\$E13_MB/raw_absolute/label_vocabulary.json" \\
  > "\$ARCHIVE_META_DIR/create_archive_stdout.json"

echo "Removing unarchived build MAT files before extraction to control project file count"
rm -rf "\$BUILD_DIR/chunks"

echo "Extracting reusable archive"
tar -xf "\$ARCHIVE_PATH" -C "\$EXTRACT_DIR"
cp "\$EXTRACT_DIR/archive_manifest.csv" "\$E13_MB/standardized_manifest.csv"
cp "\$SOURCE_E13/label_vocabulary.json" "\$E13_MB/label_vocabulary.json"

python - "\$E13_MB" "\$E12_MB" "\$EXTRACT_DIR" "\$ARCHIVE_PATH" <<'PY'
from pathlib import Path
from src.dataset.multilabel import read_csv_rows, write_csv_rows
import json
import sys

e13 = Path(sys.argv[1])
e12 = Path(sys.argv[2])
extract_dir = sys.argv[3]
archive_path = sys.argv[4]
rows = read_csv_rows(e13 / "standardized_manifest.csv")
e12_rows = [row for row in rows if row.get("source_kind") != "BioDCASE"]
e12.mkdir(parents=True, exist_ok=True)
write_csv_rows(e12 / "standardized_manifest.csv", e12_rows)
(e13 / "multiband_archive_summary.json").write_text(json.dumps({"rows": len(rows), "dataset_root": extract_dir, "archive_path": archive_path}, indent=2), encoding="utf-8")
(e12 / "multiband_archive_summary.json").write_text(json.dumps({"rows": len(e12_rows), "dataset_root": extract_dir, "archive_path": archive_path}, indent=2), encoding="utf-8")
PY
cp "\$SOURCE_E13/label_vocabulary.json" "\$E12_MB/label_vocabulary.json"

build_variant() {
  local input_dir="\$1"
  local variant_name="\$2"
  shift 2
  local out_dir="\$VARIANT_ROOT/\$variant_name"
  python -u scripts/data/multilabel/build_onc_calibration_manifest_variants.py \\
    --manifest-csv "\$input_dir/standardized_manifest.csv" \\
    --vocab-json "\$input_dir/label_vocabulary.json" \\
    --output-dir "\$out_dir" \\
    --variant-name "\$variant_name" \\
    --seed "$SEED" \\
    "\$@"
}

COMMON_ONC_OVERSAMPLE=(
  --oversample-train-source-label "ONC:species:Bm:$ONC_RARE_TARGET"
  --oversample-train-source-label "ONC:species:Mn:$ONC_RARE_TARGET"
  --oversample-train-source-label "ONC:species:Oo:$ONC_RARE_TARGET"
)

echo "Building train-manifest variants"
build_variant "\$E13_MB" "E16_e13_multiband_oncrare_full" "\${COMMON_ONC_OVERSAMPLE[@]}"
build_variant "\$E13_MB" "E16_e13_multiband_oncrare_extcap" \\
  "\${COMMON_ONC_OVERSAMPLE[@]}" \\
  --train-source-label-cap "BioDCASE:species:Bm:8000" \\
  --train-source-label-cap "BioDCASE:species:Bp:8000" \\
  --train-source-label-cap "BioDCASE:<background>:1000" \\
  --train-source-label-cap "DCLDE:species:Mn:3000" \\
  --train-source-label-cap "DCLDE:species:Oo:3000" \\
  --train-source-label-cap "DCLDE:<background>:3000"
build_variant "\$E12_MB" "E16_e12_multiband_oncrare_dcldecap" \\
  "\${COMMON_ONC_OVERSAMPLE[@]}" \\
  --train-source-label-cap "DCLDE:species:Mn:2500" \\
  --train-source-label-cap "DCLDE:species:Oo:2500" \\
  --train-source-label-cap "DCLDE:<background>:2500"

echo "Queue/accounting check before GPU submissions"
squeue -u merileo || true
sacct -u merileo --starttime now-7days || true
diskusage_report || true

echo -e "experiment\tmanifest\tvocab\tdataset_root\tencoder\tfusion\tlr\tweight_decay\tuse_pos_weight\trun_dir\tarchive_path\twandb_group" > "\$PLAN_TSV"
echo -e "job_id\texperiment\trun_dir\tjob_script" > "\$SUBMITTED_TSV"

submit_train() {
  local variant_name="\$1"
  local use_pos_weight="\$2"
  local suffix="\$3"
  local encoder="\$4"
  local train_lr="\$5"
  local train_weight_decay="\$6"
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
  --bands low,mid,high
  --band-crop-shapes low:391x50,mid:256x100,high:256x312
  --encoder "\$encoder"
  --fusion gated
  --init-low-checkpoint "\$BASE_CKPT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "\$train_lr"
  --weight-decay "\$train_weight_decay"
  --crop-time-seconds 10
  --context-seconds 40
  --center-bias-sigma-frac 0.25
  --positive-crop-mode edge_mix
  --device cuda
  --seed "$SEED"
  --use-wandb
  --wandb-project whale-multispecies-calltype
  --wandb-group "$WANDB_GROUP"
  --wandb-name "\$run_exp"
  --wandb-tags "multilabel,species,multiband,40sctx,gated,\$variant_name,\$encoder"
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
  "weight_decay": "\$train_weight_decay",
  "bands": ["low", "mid", "high"],
  "band_crop_shapes": {"low": [391, 50], "mid": [256, 100], "high": [256, 312]},
  "crop_time_seconds": 10,
  "positive_crop_mode": "edge_mix",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE,
  "onc_rare_target": $ONC_RARE_TARGET
}
META
TRAIN_EOF
  echo -e "\$run_exp\t\$variant_dir/standardized_manifest.csv\t\$variant_dir/label_vocabulary.json\t\$EXTRACT_DIR\t\$encoder\tgated\t\$train_lr\t\$train_weight_decay\t\$use_pos_weight\t\$run_dir\t\$ARCHIVE_PATH\t$WANDB_GROUP" >> "\$PLAN_TSV"
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

submit_train "E16_e13_multiband_oncrare_extcap" false "gated_resnet18_noposw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"
submit_train "E16_e13_multiband_oncrare_extcap" true "gated_resnet18_posw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"
submit_train "E16_e12_multiband_oncrare_dcldecap" true "gated_resnet18_posw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"
submit_train "E16_e13_multiband_oncrare_extcap" false "gated_resnet34_noposw_lr2e4" "resnet34" "0.0002" "$WEIGHT_DECAY"
submit_train "E16_e13_multiband_oncrare_full" true "gated_resnet18_posw_lr3e4" "resnet18" "0.0003" "$WEIGHT_DECAY"

echo "Training plan:"
cat "\$PLAN_TSV"
echo "Submitted jobs:"
cat "\$SUBMITTED_TSV"
echo "Finished multiband fusion pipeline at \$(date -Is)"
EOF

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: wrote $JOB_SCRIPT"
else
  job_id="$(sbatch "$JOB_SCRIPT" | awk '{print $4}')"
  echo "Submitted $RUN_NAME as $job_id"
  echo "Pipeline dir: $PIPELINE_DIR"
  echo "Slurm log: $LOG_DIR/slurm-$job_id.out"
fi
