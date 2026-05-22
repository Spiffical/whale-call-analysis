#!/bin/bash
# Submit multispecies high-confidence sliding-window inference for an unreviewed month.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

MONTH="2025-04"
DEVICE_CODE="ICLISTENHF6016"
FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
ARCHIVE="/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio.tar.zst"
AVAILABLE_FILENAMES="/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio_available_filenames.txt"
REVIEWED_WORKBOOK=""
SOURCE_RAW_DIR=""
OUT_DIR=""
RUN_LABEL="multispecies_e24_expert_highconf_unreviewed_low0p7_high0p9_min3_gap15"

FIN_EXPERT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/runs/E24_fin_whale_low_sourcecap_r18_lr3e4_c10_d03_20260522T183143Z"
BLUE_EXPERT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/runs/E24_blue_whale_low_expert_r18_lr3e4_c10_d03_20260522T183143Z"
HUMPBACK_EXPERT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/runs/E24_humpback_whale_lowmid_expert_r18_lr1e4_c10_d03_20260522T183145Z"

CLIP_SECONDS="300"
CROP_SECONDS="10"
STEP_SECONDS="10"
ADJACENT_MINUTES="5"
LOW_THRESHOLD="0.70"
HIGH_THRESHOLD="0.90"
MIN_MEMBERS="3"
MAX_GAP_SECONDS="15"
MAX_EVENTS_PER_LABEL="0"
MAX_TARGET_FILES="0"
BATCH_SIZE="128"
DEVICE="cuda"

SBATCH_TIME="18:00:00"
SBATCH_CPUS="12"
SBATCH_MEM="96G"
SBATCH_GRES="gpu:h100:1"
SBATCH_EXCLUDE="g19"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_april_highconf_inference.sh [options]

Purpose:
  Score unreviewed Clayoquot month audio with the E24 fin whale, blue whale,
  and humpback whale expert ensemble. The job streams raw audio directly,
  stitches previous/current/next 5-minute files before FFTs to avoid boundary
  artifacts, scores 10 s sliding windows, clusters strict high-confidence
  events, and caches only retained event crops.

Key options:
  --month YYYY-MM                 Default: 2025-04
  --out-dir PATH                  Default: /scratch/.../pipeline_runs/e25_april_multispecies_highconf_<timestamp>
  --source-raw-dir PATH           Default: prior full-month fin-whale raw_audio cache
  --include-reviewed              Do not exclude workbook-reviewed clips
  --max-target-files N            Smoke-test cap after reviewed exclusion
  --low-threshold X               Default: 0.70
  --high-threshold X              Default: 0.90
  --min-members N                 Default: 3
  --max-gap-seconds X             Default: 15
  --dry-run                       Write job script but do not submit
USAGE
}

INCLUDE_REVIEWED="false"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --month) MONTH="$2"; shift 2 ;;
    --device-code) DEVICE_CODE="$2"; shift 2 ;;
    --final2025-root) FINAL2025_ROOT="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --available-filenames) AVAILABLE_FILENAMES="$2"; shift 2 ;;
    --reviewed-workbook) REVIEWED_WORKBOOK="$2"; shift 2 ;;
    --source-raw-dir) SOURCE_RAW_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --run-label) RUN_LABEL="$2"; shift 2 ;;
    --fin-expert) FIN_EXPERT="$2"; shift 2 ;;
    --blue-expert) BLUE_EXPERT="$2"; shift 2 ;;
    --humpback-expert) HUMPBACK_EXPERT="$2"; shift 2 ;;
    --clip-seconds) CLIP_SECONDS="$2"; shift 2 ;;
    --crop-seconds) CROP_SECONDS="$2"; shift 2 ;;
    --step-seconds) STEP_SECONDS="$2"; shift 2 ;;
    --adjacent-minutes) ADJACENT_MINUTES="$2"; shift 2 ;;
    --low-threshold) LOW_THRESHOLD="$2"; shift 2 ;;
    --high-threshold) HIGH_THRESHOLD="$2"; shift 2 ;;
    --min-members) MIN_MEMBERS="$2"; shift 2 ;;
    --max-gap-seconds) MAX_GAP_SECONDS="$2"; shift 2 ;;
    --max-events-per-label) MAX_EVENTS_PER_LABEL="$2"; shift 2 ;;
    --max-target-files) MAX_TARGET_FILES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --exclude) SBATCH_EXCLUDE="$2"; shift 2 ;;
    --include-reviewed) INCLUDE_REVIEWED="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
  echo "Error: --month must be YYYY-MM" >&2
  usage
  exit 1
fi

if [[ -z "$REVIEWED_WORKBOOK" ]]; then
  REVIEWED_WORKBOOK="$REPO_ROOT/data/finwhales/ONC_ClayoquotSlope2025_Annotations_Cetaceans_Instrument_EQ_Sonar_Unknown.xlsx"
fi
if [[ -z "$SOURCE_RAW_DIR" ]]; then
  SOURCE_RAW_DIR="$FINAL2025_ROOT/high_confidence_predictions/${MONTH}_fullmonth_unreviewed_joint_scratch_highconf_ws48_low0p7_high0p9_min3_gap15/raw_audio"
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$WEEKEND_ROOT/pipeline_runs/e25_april_multispecies_highconf_${STAMP}"
fi

for required in "$ARCHIVE" "$AVAILABLE_FILENAMES" "$FIN_EXPERT/train/best.pt" "$BLUE_EXPERT/train/best.pt" "$HUMPBACK_EXPERT/train/best.pt"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
if [[ "$INCLUDE_REVIEWED" != "true" && ! -e "$REVIEWED_WORKBOOK" ]]; then
  echo "Missing reviewed workbook: $REVIEWED_WORKBOOK" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/logs"
JOB_SCRIPT="$OUT_DIR/logs/run_multispecies_highconf_${MONTH}.sbatch"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=E25multiHC_${MONTH}
#SBATCH --output=$OUT_DIR/logs/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM
EOF

if [[ -n "$SBATCH_GRES" ]]; then
  echo "#SBATCH --gres=$SBATCH_GRES" >> "$JOB_SCRIPT"
fi
if [[ -n "$SBATCH_EXCLUDE" ]]; then
  echo "#SBATCH --exclude=$SBATCH_EXCLUDE" >> "$JOB_SCRIPT"
fi

cat >> "$JOB_SCRIPT" <<'EOF'

set -euo pipefail

echo "Started multispecies high-confidence month inference at $(date -Is)"
echo "Host: $(hostname)"
cd "__REPO_ROOT__"
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
mkdir -p "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$PIP_CACHE_DIR"

MONTH="__MONTH__"
DEVICE_CODE="__DEVICE_CODE__"
ARCHIVE="__ARCHIVE__"
AVAILABLE_FILENAMES="__AVAILABLE_FILENAMES__"
REVIEWED_WORKBOOK="__REVIEWED_WORKBOOK__"
SOURCE_RAW_DIR="__SOURCE_RAW_DIR__"
OUT_DIR="__OUT_DIR__"
INCLUDE_REVIEWED="__INCLUDE_REVIEWED__"
FIN_EXPERT="__FIN_EXPERT__"
BLUE_EXPERT="__BLUE_EXPERT__"
HUMPBACK_EXPERT="__HUMPBACK_EXPERT__"
CLIP_SECONDS="__CLIP_SECONDS__"
CROP_SECONDS="__CROP_SECONDS__"
STEP_SECONDS="__STEP_SECONDS__"
ADJACENT_MINUTES="__ADJACENT_MINUTES__"
LOW_THRESHOLD="__LOW_THRESHOLD__"
HIGH_THRESHOLD="__HIGH_THRESHOLD__"
MIN_MEMBERS="__MIN_MEMBERS__"
MAX_GAP_SECONDS="__MAX_GAP_SECONDS__"
MAX_EVENTS_PER_LABEL="__MAX_EVENTS_PER_LABEL__"
MAX_TARGET_FILES="__MAX_TARGET_FILES__"
BATCH_SIZE="__BATCH_SIZE__"
DEVICE="__DEVICE__"

RAW_DIR="$OUT_DIR/raw_audio"
MANIFEST_DIR="$OUT_DIR/manifests"
PREDICTION_DIR="$OUT_DIR/predictions"
METADATA_JSON="$OUT_DIR/package_metadata.json"
mkdir -p "$RAW_DIR" "$MANIFEST_DIR" "$PREDICTION_DIR"

git rev-parse HEAD || true
python -m py_compile \
  scripts/data/multilabel/build_month_unreviewed_sliding_windows.py \
  scripts/inference/run_multiband_expert_ensemble_inference.py

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to run full-month inference on CPU")
print("CUDA device:", torch.cuda.get_device_name(0))
PY

selection_args=(
  --month "$MONTH"
  --device-code "$DEVICE_CODE"
  --available-filenames "$AVAILABLE_FILENAMES"
  --out-dir "$MANIFEST_DIR"
  --clip-seconds "$CLIP_SECONDS"
  --crop-seconds "$CROP_SECONDS"
  --step-seconds "$STEP_SECONDS"
  --adjacent-minutes "$ADJACENT_MINUTES"
)
if [[ "$INCLUDE_REVIEWED" == "true" ]]; then
  selection_args+=(--include-reviewed)
else
  selection_args+=(--reviewed-workbook "$REVIEWED_WORKBOOK")
fi
if [[ "$MAX_TARGET_FILES" != "0" ]]; then
  selection_args+=(--max-target-files "$MAX_TARGET_FILES")
fi

echo "Building unreviewed month target list and 10 s windows..."
python -u scripts/data/multilabel/build_month_unreviewed_sliding_windows.py "${selection_args[@]}"

echo "Linking selected raw audio from prior cache where available..."
python - "$MANIFEST_DIR/selected_filenames.txt" "$SOURCE_RAW_DIR" "$RAW_DIR" "$MANIFEST_DIR/missing_archive_members.txt" <<'PY'
import os
import sys
from pathlib import Path

selected_path, source_raw_dir, raw_dir, missing_path = [Path(arg) for arg in sys.argv[1:5]]
raw_dir.mkdir(parents=True, exist_ok=True)
source_raw_dir = source_raw_dir.resolve()
linked = 0
present = 0
missing = []
for name in selected_path.read_text(encoding="utf-8").splitlines():
    name = name.strip()
    if not name:
        continue
    dst = raw_dir / name
    if dst.exists() or dst.is_symlink():
        present += 1
        continue
    src = source_raw_dir / name
    if src.exists():
        os.symlink(src, dst)
        linked += 1
    else:
        missing.append(f"raw_audio/{name}")
missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
print(f"Raw audio already present: {present}")
print(f"Raw audio symlinked from prior cache: {linked}")
print(f"Raw audio missing from prior cache: {len(missing)}")
PY

if [[ -s "$MANIFEST_DIR/missing_archive_members.txt" ]]; then
  echo "Extracting missing selected raw audio from archive..."
  tar --use-compress-program=unzstd -xf "$ARCHIVE" -C "$OUT_DIR" -T "$MANIFEST_DIR/missing_archive_members.txt"
else
  echo "All selected raw audio is available without archive extraction."
fi

python - "$MANIFEST_DIR/target_clip_list.txt" "$MANIFEST_DIR/selected_filenames.txt" "$RAW_DIR" <<'PY'
import sys
from pathlib import Path

target_path, selected_path, raw_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
missing_targets = [
    name for name in target_path.read_text(encoding="utf-8").splitlines()
    if name.strip() and not (raw_dir / name.strip()).exists()
]
missing_selected = [
    name for name in selected_path.read_text(encoding="utf-8").splitlines()
    if name.strip() and not (raw_dir / name.strip()).exists()
]
if missing_targets:
    print("Missing target audio examples:")
    for name in missing_targets[:20]:
        print(name)
    raise SystemExit(f"{len(missing_targets)} target audio files missing")
if missing_selected:
    print("Missing selected context audio examples:")
    for name in missing_selected[:20]:
        print(name)
    raise SystemExit(f"{len(missing_selected)} selected audio/context files missing")
print("All target and selected context audio files are present.")
PY

echo "Running stitched-audio multispecies expert ensemble inference..."
python -u scripts/inference/run_multiband_expert_ensemble_inference.py \
  --audio-list "$MANIFEST_DIR/target_clip_list.txt" \
  --audio-dir "$RAW_DIR" \
  --raw-audio-dir "$RAW_DIR" \
  --expert "species:Bp=$FIN_EXPERT" \
  --expert "species:Bm=$BLUE_EXPERT" \
  --expert "species:Mn=$HUMPBACK_EXPERT" \
  --output-dir "$PREDICTION_DIR" \
  --clip-seconds "$CLIP_SECONDS" \
  --crop-seconds "$CROP_SECONDS" \
  --step-seconds "$STEP_SECONDS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --low-threshold "$LOW_THRESHOLD" \
  --high-threshold "$HIGH_THRESHOLD" \
  --min-members "$MIN_MEMBERS" \
  --max-gap-seconds "$MAX_GAP_SECONDS" \
  --max-events-per-label "$MAX_EVENTS_PER_LABEL" \
  --cache-audio-crops

python - "$OUT_DIR" "$MANIFEST_DIR/selection_summary.json" "$PREDICTION_DIR/prediction_summary.json" "$METADATA_JSON" <<'PY'
import json
import sys
from pathlib import Path

out_dir, selection_json, prediction_json, metadata_json = [Path(arg) for arg in sys.argv[1:5]]
selection = json.loads(selection_json.read_text(encoding="utf-8"))
prediction = json.loads(prediction_json.read_text(encoding="utf-8"))
metadata = {
    "pipeline": "multispecies_april_highconf_stitched_audio",
    "out_dir": str(out_dir),
    "selection": selection,
    "prediction_summary": prediction,
}
metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
with (out_dir / "package_summary.md").open("w", encoding="utf-8") as handle:
    handle.write("# Multispecies April High-Confidence Cache\n\n")
    handle.write(f"- target audio clips: `{selection.get('unreviewed_target_files')}`\n")
    handle.write(f"- selected audio plus neighbors: `{selection.get('selected_files_including_adjacent')}`\n")
    handle.write(f"- scored windows: `{prediction.get('window_rows')}`\n")
    handle.write(f"- kept events: `{prediction.get('event_count')}`\n")
    handle.write(f"- events by label: `{prediction.get('events_by_label')}`\n")
    handle.write(f"- high-confidence cache: `{prediction.get('outputs', {}).get('cache_tar')}`\n")
    handle.write(f"- full prediction summary: `{prediction_json}`\n")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY

timeout 180 diskusage_report || true
df -ih /project/def-kmoran /scratch || true
echo "Finished at $(date -Is)"
EOF

python - "$JOB_SCRIPT" \
  "__REPO_ROOT__=$REPO_ROOT" \
  "__MONTH__=$MONTH" \
  "__DEVICE_CODE__=$DEVICE_CODE" \
  "__ARCHIVE__=$ARCHIVE" \
  "__AVAILABLE_FILENAMES__=$AVAILABLE_FILENAMES" \
  "__REVIEWED_WORKBOOK__=$REVIEWED_WORKBOOK" \
  "__SOURCE_RAW_DIR__=$SOURCE_RAW_DIR" \
  "__OUT_DIR__=$OUT_DIR" \
  "__INCLUDE_REVIEWED__=$INCLUDE_REVIEWED" \
  "__FIN_EXPERT__=$FIN_EXPERT" \
  "__BLUE_EXPERT__=$BLUE_EXPERT" \
  "__HUMPBACK_EXPERT__=$HUMPBACK_EXPERT" \
  "__CLIP_SECONDS__=$CLIP_SECONDS" \
  "__CROP_SECONDS__=$CROP_SECONDS" \
  "__STEP_SECONDS__=$STEP_SECONDS" \
  "__ADJACENT_MINUTES__=$ADJACENT_MINUTES" \
  "__LOW_THRESHOLD__=$LOW_THRESHOLD" \
  "__HIGH_THRESHOLD__=$HIGH_THRESHOLD" \
  "__MIN_MEMBERS__=$MIN_MEMBERS" \
  "__MAX_GAP_SECONDS__=$MAX_GAP_SECONDS" \
  "__MAX_EVENTS_PER_LABEL__=$MAX_EVENTS_PER_LABEL" \
  "__MAX_TARGET_FILES__=$MAX_TARGET_FILES" \
  "__BATCH_SIZE__=$BATCH_SIZE" \
  "__DEVICE__=$DEVICE" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for pair in sys.argv[2:]:
    key, value = pair.split("=", 1)
    text = text.replace(key, value)
path.write_text(text, encoding="utf-8")
PY

chmod +x "$JOB_SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run job script: $JOB_SCRIPT"
  echo "Output directory: $OUT_DIR"
  exit 0
fi

JOB_OUTPUT="$(sbatch "$JOB_SCRIPT")"
echo "$JOB_OUTPUT"
echo "Job script: $JOB_SCRIPT"
echo "Output directory: $OUT_DIR"
