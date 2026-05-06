#!/bin/bash
# Build a larger species-first 40s MAT cache/archive and launch bounded training jobs.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
PROJECT_ROOT="$FINAL2025_ROOT/multispecies_calltype_experiments"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
RUN_NAME="E12prep_scale40s"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

ONC_MAX_FIN="5000"
ONC_MAX_PER_SPECIES="0"
ONC_MAX_BACKGROUND_CLIPS="1000"
ONC_BACKGROUND_WINDOWS_PER_CLIP="2"
ONC_GAP_MAX_WINDOWS_PER_CLIP="8"
# Gap rows become 40s context MATs and are randomly cropped to ~10s during
# training, so the full visible 40s context needs to stay clear of primary calls.
ONC_GAP_EXCLUSION_BUFFER_S="20"
BIODCASE_MAX_PER_LABEL="1000"
BIODCASE_MAX_BACKGROUND="1500"
DCLDE_MAX_POSITIVE="2000"
DCLDE_MAX_HARD_NEGATIVE="2000"
DCLDE_MAX_AUDIO_FAILURE_FRAC="0.05"

EPOCHS="30"
BATCH_SIZE="64"
NUM_WORKERS="8"
LR="0.0003"
WEIGHT_DECAY="0.0001"
SEED="2026"
SBATCH_TIME="12:00:00"
SBATCH_CPUS="16"
SBATCH_MEM="96G"
SBATCH_GRES="gpu:h100:1"
GPU_TIME="08:00:00"
GPU_MEM="72G"
WANDB_GROUP="weekend-20260502-scaleup40s"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_scaleup_40s_archive_pipeline.sh [options]

This submits one CPU prep pipeline. The pipeline builds 40s MAT files, archives
them into one tar cache, extracts that cache once, then submits E12/E13 GPU jobs.

Key options:
  --onc-max-fin N                       Default: 5000
  --onc-max-per-species N               Default: 0 (all available non-fin rows)
  --onc-max-background-clips N          Default: 1000
  --onc-gap-max-windows-per-clip N      Default: 8
  --onc-gap-exclusion-buffer-s N        Default: 20 (keeps 40s context primary-free)
  --biodcase-max-per-label N            Default: 1000
  --biodcase-max-background N           Default: 1500
  --dclde-max-positive N                Default: 2000
  --dclde-max-hard-negative N           Default: 2000
  --epochs N                            Default: 30
  --batch-size N                        Default: 64
  --dry-run                             Write the sbatch but do not submit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --final2025-root) FINAL2025_ROOT="$2"; PROJECT_ROOT="$2/multispecies_calltype_experiments"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --onc-max-fin) ONC_MAX_FIN="$2"; shift 2 ;;
    --onc-max-per-species) ONC_MAX_PER_SPECIES="$2"; shift 2 ;;
    --onc-max-background-clips) ONC_MAX_BACKGROUND_CLIPS="$2"; shift 2 ;;
    --onc-background-windows-per-clip) ONC_BACKGROUND_WINDOWS_PER_CLIP="$2"; shift 2 ;;
    --onc-gap-max-windows-per-clip) ONC_GAP_MAX_WINDOWS_PER_CLIP="$2"; shift 2 ;;
  --onc-gap-exclusion-buffer-s) ONC_GAP_EXCLUSION_BUFFER_S="$2"; shift 2 ;;
    --biodcase-max-per-label) BIODCASE_MAX_PER_LABEL="$2"; shift 2 ;;
    --biodcase-max-background) BIODCASE_MAX_BACKGROUND="$2"; shift 2 ;;
    --dclde-max-positive) DCLDE_MAX_POSITIVE="$2"; shift 2 ;;
    --dclde-max-hard-negative) DCLDE_MAX_HARD_NEGATIVE="$2"; shift 2 ;;
    --dclde-max-audio-failure-frac) DCLDE_MAX_AUDIO_FAILURE_FRAC="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
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

REPO_ON_NIBI="$PROJECT_ROOT/repo"
PIPELINE_DIR="$WEEKEND_ROOT/pipeline_runs/scaleup40s_archive_${STAMP}"
SOURCE_DIR="$WEEKEND_ROOT/manifests/scaleup40s_sources_${STAMP}"
CACHE_DIR="$WEEKEND_ROOT/mat_archives/scaleup40s_${STAMP}"
BUILD_DIR="$CACHE_DIR/build"
BUILD_MAT_DIR="$BUILD_DIR/mat_files"
ARCHIVE_META_DIR="$CACHE_DIR/archive_meta"
ARCHIVE_PATH="$CACHE_DIR/scaleup40s_mat_cache.tar"
EXTRACT_DIR="$CACHE_DIR/extracted"
MANIFEST_ROOT="$WEEKEND_ROOT/manifests"
LOG_DIR="$PIPELINE_DIR/logs"
JOB_SCRIPT="$LOG_DIR/scaleup40s_archive_${STAMP}.sbatch"
mkdir -p "$LOG_DIR"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM

set -euo pipefail

echo "Started scale-up 40s archive pipeline at \$(date -Is)"
echo "Host: \$(hostname)"

REPO="$REPO_ON_NIBI"
FINAL2025="$FINAL2025_ROOT"
PROJECT="$PROJECT_ROOT"
WEEKEND="$WEEKEND_ROOT"
STAMP="$STAMP"
PIPELINE_DIR="$PIPELINE_DIR"
SOURCE_DIR="\$WEEKEND/manifests/scaleup40s_sources_\$STAMP"
CACHE_DIR="\$WEEKEND/mat_archives/scaleup40s_\$STAMP"
BUILD_DIR="\$CACHE_DIR/build"
BUILD_MAT_DIR="\$BUILD_DIR/mat_files"
ARCHIVE_META_DIR="\$CACHE_DIR/archive_meta"
ARCHIVE_PATH="\$CACHE_DIR/scaleup40s_mat_cache.tar"
EXTRACT_DIR="\$CACHE_DIR/extracted"
MANIFEST_ROOT="\$WEEKEND/manifests"

ONC_ANNOTATIONS="\$FINAL2025/part2/full_bundle/manifests/annotations_all.csv"
ONC_CLIP_MANIFEST="\$FINAL2025/part2/full_bundle/manifests/clip_manifest.csv"
ONC_AUDIO="\$FINAL2025/part2/full_bundle/raw_audio"
DATASET_DOC="\$FINAL2025/historical/training_dataset/dataset_documentation.json"
DCLDE_ANNOTATIONS="\$WEEKEND/audits/dclde_2027_killer_whales/Annotations.csv"
DCLDE_GCS_OBJECTS="\$WEEKEND/audits/dclde_2027_killer_whales/gcs_objects.txt"
BIOD_ROOT="\$PROJECT/external_data/biodcase2026_task2/extracted/2026_BioDCASE_development_set/train"
BASE_CKPT="\$FINAL2025/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"

mkdir -p "\$PIPELINE_DIR" "\$SOURCE_DIR" "\$BUILD_MAT_DIR" "\$ARCHIVE_META_DIR" "\$EXTRACT_DIR"
cd "\$REPO"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_CACHE_DIR" "\$PIP_CACHE_DIR"

git rev-parse HEAD

echo "Building ONC source manifest"
mkdir -p "\$SOURCE_DIR/onc"
python -u scripts/data/multilabel/build_multispecies_prep_manifest.py \\
  --annotations-csv "\$ONC_ANNOTATIONS" \\
  --clip-manifest-csv "\$ONC_CLIP_MANIFEST" \\
  --output-dir "\$SOURCE_DIR/onc" \\
  --dataset-name "final2025_onc_scaleup40s" \\
  --species "OD,Mn,Bm,Oo" \\
  --include-fin \\
  --max-fin "$ONC_MAX_FIN" \\
  --max-per-species "$ONC_MAX_PER_SPECIES" \\
  --max-background "$ONC_MAX_BACKGROUND_CLIPS" \\
  --background-windows-per-clip "$ONC_BACKGROUND_WINDOWS_PER_CLIP" \\
  --context-s 40 \\
  --edge-context-s 10.5 \\
  --available-audio-dir "\$ONC_AUDIO" \\
  > "\$SOURCE_DIR/onc/build_stdout.json"

echo "Generating ONC 40s MATs"
python -u scripts/data/part2/prepare_trainstyle_windows.py \\
  --calls-csv "\$SOURCE_DIR/onc/selected_calls.csv" \\
  --audio-dir "\$ONC_AUDIO" \\
  --dataset-doc "\$DATASET_DOC" \\
  --out-dir "\$BUILD_MAT_DIR" \\
  --spec-backend auto \\
  --window-s 40 \\
  --edge-context-s 10.5

echo "Building ONC primary-adjacent gap rows"
python - <<'PY'
import csv
from pathlib import Path
clip_manifest = Path("$FINAL2025_ROOT/part2/full_bundle/manifests/clip_manifest.csv")
out = Path("$SOURCE_DIR/onc_clip_durations.csv")
with clip_manifest.open(newline="", encoding="utf-8-sig") as src, out.open("w", newline="", encoding="utf-8") as dst:
    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=["filename", "source_audio", "clip", "duration_s"])
    writer.writeheader()
    for row in reader:
        name = (row.get("filename") or "").strip()
        if name:
            writer.writerow({"filename": name, "source_audio": name, "clip": name, "duration_s": "300.0"})
PY
python -u scripts/data/multilabel/build_negative_window_manifest.py \\
  --annotations-csv "\$ONC_ANNOTATIONS" \\
  --clip-duration-csv "\$SOURCE_DIR/onc_clip_durations.csv" \\
  --output-csv "\$SOURCE_DIR/onc_primary_adjacent_gap_candidates.csv" \\
  --window-s 10 \\
  --exclusion-buffer-s "$ONC_GAP_EXCLUSION_BUFFER_S" \\
  --step-s 10 \\
  --max-windows-per-clip "$ONC_GAP_MAX_WINDOWS_PER_CLIP"
python - <<'PY'
import csv
import json
from pathlib import Path

source = Path("$SOURCE_DIR")
inp = source / "onc_primary_adjacent_gap_candidates.csv"
out = source / "onc_primary_adjacent_gaps_selected.csv"
model_labels = Path("$REPO_ON_NIBI/experiments/weekend_20260502_analysis/negative_review_visual_sample/tables/onc_primary_adjacent_gap_model_assisted_labels.csv")
excluded_labels = {}
if model_labels.exists():
    with model_labels.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            excluded_labels[row.get("item_id", "")] = row.get("model_assisted_review_label", "")
rows = []
excluded = []
with inp.open(newline="", encoding="utf-8-sig") as handle:
    for row in csv.DictReader(handle):
        if row.get("negative_bucket") != "primary_adjacent_gap":
            continue
        if excluded_labels.get(row.get("item_id", "")) == "unlabeled_signal_suspect":
            row["exclude_reason"] = "model_assisted_obvious_signal"
            excluded.append(row)
            continue
        clip = row["clip"]
        begin = float(row["begin_s"])
        end = float(row["end_s"])
        expected = f"{clip}_{begin:.1f}s_{end:.1f}s_trainstyle.mat"
        row["expected_mat_name"] = expected
        row["mat_path"] = f"mat_files/{expected}"
        row["source_kind"] = "ONC"
        row["source_dataset"] = "onc_primary_adjacent_gap_scaleup40s"
        row["label_ids"] = ""
        row["canonical_label_ids"] = ""
        row["source_label_ids"] = ""
        row["analysis_label_ids"] = ""
        row["is_background"] = "1"
        rows.append(row)
fieldnames = []
for row in rows:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
if excluded:
    ex_path = source / "onc_primary_adjacent_gaps_excluded.csv"
    ex_fields = []
    for row in excluded:
        for key in row:
            if key not in ex_fields:
                ex_fields.append(key)
    with ex_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ex_fields)
        writer.writeheader()
        writer.writerows(excluded)
summary = {"selected_gap_rows": len(rows), "excluded_gap_rows": len(excluded), "output_csv": str(out)}
(source / "onc_primary_adjacent_gap_selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
python -u scripts/data/part2/prepare_trainstyle_windows.py \\
  --calls-csv "\$SOURCE_DIR/onc_primary_adjacent_gaps_selected.csv" \\
  --audio-dir "\$ONC_AUDIO" \\
  --dataset-doc "\$DATASET_DOC" \\
  --out-dir "\$BUILD_MAT_DIR" \\
  --spec-backend auto \\
  --window-s 40 \\
  --edge-context-s 10.5

echo "Building BioDCASE source manifest"
mkdir -p "\$SOURCE_DIR/biodcase" "\$SOURCE_DIR/biodcase_raw_audio"
mapfile -t biod_annotations < <(find "\$BIOD_ROOT/annotations" -maxdepth 1 -type f -name '*.csv' | sort)
if [[ "\${#biod_annotations[@]}" -eq 0 ]]; then
  echo "No BioDCASE train annotations found under \$BIOD_ROOT/annotations" >&2
  exit 2
fi
biod_cmd=(python -u scripts/data/multilabel/build_biodcase_task2_manifest.py
  --output-dir "\$SOURCE_DIR/biodcase"
  --dataset-name "biodcase2026_task2_train_scaleup40s"
  --audio-root "\$BIOD_ROOT/audio"
  --require-existing-audio
  --max-per-label "$BIODCASE_MAX_PER_LABEL"
  --max-background "$BIODCASE_MAX_BACKGROUND"
  --clip-name-mode dataset_prefix)
for ann in "\${biod_annotations[@]}"; do
  biod_cmd+=(--annotations-csv "\$ann")
done
"\${biod_cmd[@]}" > "\$SOURCE_DIR/biodcase/build_stdout.json"
python - <<'PY'
import csv
import json
import shutil
from pathlib import Path

selected_csv = Path("$SOURCE_DIR/biodcase/selected_calls.csv")
audio_root = Path("$PROJECT_ROOT/external_data/biodcase2026_task2/extracted/2026_BioDCASE_development_set/train/audio")
raw_dir = Path("$SOURCE_DIR/biodcase_raw_audio")
raw_dir.mkdir(parents=True, exist_ok=True)
staged = []
missing = []
with selected_csv.open(newline="", encoding="utf-8-sig") as handle:
    for row in csv.DictReader(handle):
        clip = row["clip"]
        dataset = row["source_dataset"]
        source_audio = row["source_audio"]
        src = audio_root / dataset / source_audio
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
python -u scripts/data/part2/prepare_trainstyle_windows.py \\
  --calls-csv "\$SOURCE_DIR/biodcase/selected_calls.csv" \\
  --audio-dir "\$SOURCE_DIR/biodcase_raw_audio" \\
  --dataset-doc "\$DATASET_DOC" \\
  --out-dir "\$BUILD_MAT_DIR" \\
  --spec-backend auto \\
  --window-s 40 \\
  --edge-context-s 10.5

echo "Building DCLDE source manifest"
mkdir -p "\$SOURCE_DIR/dclde" "\$SOURCE_DIR/dclde_raw_audio"
python -u scripts/data/multilabel/build_dclde_killer_whale_manifest.py \\
  --annotations-csv "\$DCLDE_ANNOTATIONS" \\
  --output-dir "\$SOURCE_DIR/dclde" \\
  --gcs-object-list "\$DCLDE_GCS_OBJECTS" \\
  --require-gcs-audio \\
  --max-positive "$DCLDE_MAX_POSITIVE" \\
  --max-hard-negative "$DCLDE_MAX_HARD_NEGATIVE" \\
  --hard-negative-classes "UndBio,AB" \\
  > "\$SOURCE_DIR/dclde/build_stdout.json"
python - <<'PY'
import csv
import json
import time
import urllib.request
from pathlib import Path

import soundfile as sf

selected = Path("$SOURCE_DIR/dclde/selected_calls.csv")
raw_dir = Path("$SOURCE_DIR/dclde_raw_audio")
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
                dest.unlink()
            except FileNotFoundError:
                pass
            time.sleep(2 * attempt)
    if status != "ok":
        results.append({"clip": clip, "status": "failed", "path": str(dest), "error": error, "url": url})
failed = {row["clip"] for row in results if row["status"] != "ok"}
report = Path("$SOURCE_DIR/dclde_audio_download_report.csv")
fieldnames = sorted({key for row in results for key in row})
with report.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
failure_frac = len(failed) / max(1, len(by_clip))
summary = {"audio_count": len(results), "failed_count": len(failed), "failure_frac": failure_frac, "failed_examples": sorted(failed)[:20], "report_csv": str(report)}
Path("$SOURCE_DIR/dclde_audio_download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if failure_frac > float("$DCLDE_MAX_AUDIO_FAILURE_FRAC"):
    raise SystemExit(f"DCLDE audio failure fraction {failure_frac:.3f} exceeds limit")
if failed:
    for csv_name in ["selected_calls.csv", "positive_calls.csv", "hard_negative_windows.csv", "expected_multilabel_manifest.csv", "required_audio_sources.csv"]:
        path = Path("$SOURCE_DIR/dclde") / csv_name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            kept = [row for row in reader if row.get("clip") not in failed]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
PY
python -u scripts/data/part2/prepare_trainstyle_windows.py \\
  --calls-csv "\$SOURCE_DIR/dclde/selected_calls.csv" \\
  --audio-dir "\$SOURCE_DIR/dclde_raw_audio" \\
  --dataset-doc "\$DATASET_DOC" \\
  --out-dir "\$BUILD_MAT_DIR" \\
  --spec-backend auto \\
  --window-s 40 \\
  --edge-context-s 0

echo "Standardizing large manifests"
standardize_raw() {
  local exp="\$1"
  local include_bio="\$2"
  local out_dir="\$MANIFEST_ROOT/\$exp"
  mkdir -p "\$out_dir"
  local cmd=(python -u scripts/data/multilabel/standardize_multilabel_manifest.py
    --output-dir "\$out_dir/raw_absolute"
    --mode species
    --vocab-min-count 1
    --dedupe-key mat_path
    --input "onc_scale40s|\$SOURCE_DIR/onc/selected_calls.csv|\$BUILD_DIR"
    --input "onc_primary_adjacent_gap|\$SOURCE_DIR/onc_primary_adjacent_gaps_selected.csv|\$BUILD_DIR"
    --input "dclde_scale40s|\$SOURCE_DIR/dclde/selected_calls.csv|\$BUILD_DIR")
  if [[ "\$include_bio" == "true" ]]; then
    cmd+=(--input "biodcase_scale40s|\$SOURCE_DIR/biodcase/selected_calls.csv|\$BUILD_DIR")
  fi
  "\${cmd[@]}" > "\$out_dir/raw_absolute/standardize_stdout.json"
  python - <<PY
import csv, json, os, sys
from pathlib import Path
p = Path("\$out_dir/raw_absolute/standardized_manifest.csv")
missing = []
with p.open(newline="", encoding="utf-8-sig") as handle:
    for i, row in enumerate(csv.DictReader(handle), start=2):
        mp = row.get("mat_path", "")
        if mp and not os.path.exists(mp):
            missing.append((i, mp, row.get("item_id", "")))
summary = json.loads((Path("\$out_dir/raw_absolute/standardization_summary.json")).read_text())
print(json.dumps({"manifest": str(p), "rows": summary["row_count"], "labels": summary["canonical_label_counts"], "missing": len(missing), "missing_examples": missing[:5]}, indent=2, sort_keys=True))
if missing:
    sys.exit(2)
PY
}

E12_EXP="E12large_onc_dclde_40sctx_autoneg_species"
E13_EXP="E13large_onc_biod_dclde_40sctx_autoneg_species"
standardize_raw "\$E12_EXP" false
standardize_raw "\$E13_EXP" true

echo "Creating reusable MAT tar archive"
python -u scripts/data/multilabel/create_manifest_mat_archive.py \\
  --manifest-csv "\$MANIFEST_ROOT/\$E13_EXP/raw_absolute/standardized_manifest.csv" \\
  --output-dir "\$ARCHIVE_META_DIR" \\
  --archive-path "\$ARCHIVE_PATH" \\
  --archive-format tar \\
  --member-prefix mat_files \\
  --include-file "\$MANIFEST_ROOT/\$E13_EXP/raw_absolute/label_vocabulary.json" \\
  > "\$ARCHIVE_META_DIR/create_archive_stdout.json"

echo "Extracting archive for current training jobs"
tar -xf "\$ARCHIVE_PATH" -C "\$EXTRACT_DIR"

python - <<'PY'
import csv
import json
from pathlib import Path

from src.dataset.multilabel import build_vocabulary_from_rows, read_csv_rows, write_csv_rows

extract_dir = Path("$EXTRACT_DIR")
manifest_root = Path("$MANIFEST_ROOT")
rows_all = read_csv_rows(extract_dir / "archive_manifest.csv")
datasets = {
    "E12large_onc_dclde_40sctx_autoneg_species": [row for row in rows_all if row.get("source_kind") != "BioDCASE"],
    "E13large_onc_biod_dclde_40sctx_autoneg_species": rows_all,
}
for exp, rows in datasets.items():
    out = manifest_root / exp
    out.mkdir(parents=True, exist_ok=True)
    unsplit = out / "archive_manifest_unsplit.csv"
    write_csv_rows(unsplit, rows)
    vocab = build_vocabulary_from_rows(rows, min_count=1)
    vocab.save(out / "label_vocabulary.json")
    counts = {}
    for row in rows:
        source_kind = row.get("source_kind") or "<blank>"
        counts[source_kind] = counts.get(source_kind, 0) + 1
    summary = {"experiment": exp, "row_count": len(rows), "source_kind_counts": counts, "vocabulary_label_ids": list(vocab.label_ids), "archive_path": "$ARCHIVE_PATH", "dataset_root": str(extract_dir)}
    (out / "archive_manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY

for exp in "\$E12_EXP" "\$E13_EXP"; do
  python -u scripts/data/multilabel/build_candidate_splits.py \\
    --manifest-csv "\$MANIFEST_ROOT/\$exp/archive_manifest_unsplit.csv" \\
    --output-dir "\$MANIFEST_ROOT/\$exp/splits" \\
    --strategy label_balanced \\
    --seed "$SEED" \\
    > "\$MANIFEST_ROOT/\$exp/splits/build_splits_stdout.json"
  cp "\$MANIFEST_ROOT/\$exp/splits/split_manifest.csv" "\$MANIFEST_ROOT/\$exp/standardized_manifest.csv"
done

echo "Archive/cache summary"
python - <<'PY'
import json
from pathlib import Path
payload = {
    "archive_summary": json.loads(Path("$ARCHIVE_META_DIR/archive_summary.json").read_text()),
    "e12": json.loads(Path("$MANIFEST_ROOT/E12large_onc_dclde_40sctx_autoneg_species/archive_manifest_summary.json").read_text()),
    "e13": json.loads(Path("$MANIFEST_ROOT/E13large_onc_biod_dclde_40sctx_autoneg_species/archive_manifest_summary.json").read_text()),
}
Path("$PIPELINE_DIR/scaleup40s_cache_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Removing generated build MAT files after archive extraction to reduce file count"
find "\$BUILD_MAT_DIR" -maxdepth 1 -type f -name '*.mat' -delete

echo "Queue/accounting check before GPU submissions"
squeue -u merileo || true
sacct -u merileo --starttime now-7days || true

SUBMITTED_TSV="\$PIPELINE_DIR/scaleup40s_training_submitted.tsv"
PLAN_TSV="\$PIPELINE_DIR/scaleup40s_training_plan.tsv"
echo -e "experiment\tmanifest\tvocab\tdataset_root\tuse_pos_weight\trun_dir\tarchive_path\twandb_group" > "\$PLAN_TSV"
echo -e "job_id\texperiment\trun_dir\tjob_script" > "\$SUBMITTED_TSV"

submit_train() {
  local exp="\$1"
  local use_pos_weight="\$2"
  local suffix="\$3"
  local manifest_dir="\$MANIFEST_ROOT/\$exp"
  local run_exp="\${exp}_\$suffix"
  local run_dir="\$WEEKEND/runs/\${run_exp}_\$(date -u +%Y%m%dT%H%M%SZ)"
  local log_dir="\$run_dir/logs"
  local train_dir="\$run_dir/train"
  mkdir -p "\$log_dir" "\$train_dir"
  local job_script="\$log_dir/\${run_exp}.sbatch"
  cat > "\$job_script" <<TRAIN_EOF
#!/bin/bash
#SBATCH --job-name=\$run_exp
#SBATCH --output=\$log_dir/slurm-%j.out
#SBATCH --time=$GPU_TIME
#SBATCH --cpus-per-task=8
#SBATCH --mem=$GPU_MEM
#SBATCH --gres=$SBATCH_GRES

set -euo pipefail
cd "\$REPO"
source .venv/bin/activate
export PYTHONPATH="\$PWD:\${PYTHONPATH:-}"
export WANDB_PROJECT=whale-multispecies-calltype
export WANDB_DIR="\$run_dir/wandb"
export WANDB_CACHE_DIR="\$run_dir/wandb_cache"
export WANDB_DATA_DIR="\$run_dir/wandb_data"
export WANDB_CONFIG_DIR="\$run_dir/wandb_config"
mkdir -p "\$train_dir" "\$WANDB_DIR" "\$WANDB_CACHE_DIR" "\$WANDB_DATA_DIR" "\$WANDB_CONFIG_DIR"
train_cmd=(
  python -u scripts/train/train_multilabel_resnet_smoke.py
  --manifest-csv "\$manifest_dir/standardized_manifest.csv"
  --vocab-json "\$manifest_dir/label_vocabulary.json"
  --dataset-root "\$EXTRACT_DIR"
  --exp-dir "\$train_dir"
  --model resnet18
  --init-checkpoint "\$BASE_CKPT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
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
  --wandb-name "\$run_exp"
  --wandb-tags "multilabel,resnet,species,scaleup40s,mat-archive"
)
if [[ "\$use_pos_weight" == "true" ]]; then
  train_cmd+=(--use-pos-weight)
fi
"\${train_cmd[@]}"
cat > "\$run_dir/run_metadata.json" <<META
{
  "experiment": "\$run_exp",
  "base_experiment": "\$exp",
  "manifest_csv": "\$manifest_dir/standardized_manifest.csv",
  "vocab_json": "\$manifest_dir/label_vocabulary.json",
  "dataset_root": "\$EXTRACT_DIR",
  "mat_archive": "\$ARCHIVE_PATH",
  "train_dir": "\$train_dir",
  "use_pos_weight": "\$use_pos_weight",
  "crop_time_seconds": 10,
  "positive_crop_mode": "edge_mix",
  "epochs": $EPOCHS,
  "batch_size": $BATCH_SIZE
}
META
TRAIN_EOF
  local job_id
  job_id=\$(sbatch "\$job_script" | awk '{print \$4}')
  echo -e "\$run_exp\t\$manifest_dir/standardized_manifest.csv\t\$manifest_dir/label_vocabulary.json\t\$EXTRACT_DIR\t\$use_pos_weight\t\$run_dir\t\$ARCHIVE_PATH\t$WANDB_GROUP" >> "\$PLAN_TSV"
  echo -e "\$job_id\t\$run_exp\t\$run_dir\t\$job_script" >> "\$SUBMITTED_TSV"
  echo "Submitted \$run_exp as \$job_id"
}

submit_train "\$E12_EXP" true "posw"
submit_train "\$E12_EXP" false "noposw"
submit_train "\$E13_EXP" true "posw"
submit_train "\$E13_EXP" false "noposw"

echo "Submitted training jobs:"
cat "\$SUBMITTED_TSV"
echo "Completed scale-up 40s archive pipeline at \$(date -Is)"
EOF

chmod +x "$JOB_SCRIPT"
echo "Job script: $JOB_SCRIPT"
echo "Pipeline dir: $PIPELINE_DIR"

echo "Queue/accounting check before CPU pipeline submission"
squeue -u merileo || true
sacct -u merileo --starttime now-7days || true

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: not submitting"
  exit 0
fi

sbatch_out="$(sbatch "$JOB_SCRIPT")"
echo "$sbatch_out"
job_id="$(echo "$sbatch_out" | awk '{print $NF}')"
echo "$job_id" > "$PIPELINE_DIR/pipeline_job_id.txt"
echo -e "job_id\tpipeline_dir\tjob_script" > "$PIPELINE_DIR/submitted.txt"
echo -e "$job_id\t$PIPELINE_DIR\t$JOB_SCRIPT" >> "$PIPELINE_DIR/submitted.txt"
echo "$job_id"
