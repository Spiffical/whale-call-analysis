#!/bin/bash
# Submit a full-month high-confidence fin whale inference package job on Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

MONTH=""
DEVICE_CODE="ICLISTENHF6016"
FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
ARCHIVE="/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio.tar.zst"
AVAILABLE_FILENAMES="/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio_available_filenames.txt"
CHECKPOINT=""
DATASET_DOC=""
REVIEWED_WORKBOOK=""
OUT_DIR=""
RUN_LABEL="joint_scratch_highconf_unreviewed_ws48_low0p7_high0p9_min3_gap15"
EXCLUDE_REVIEWED="true"

WINDOW_STEP="48"
LOW_THRESHOLD="0.70"
HIGH_THRESHOLD="0.90"
MIN_MEMBERS="3"
MAX_GAP_SECONDS="15"
CROP_SIZE="96"
BATCH_SIZE="128"
NUM_WORKERS="4"
DEVICE="cuda"
SPEC_BACKEND="auto"
ADJACENT_MINUTES="5"

SBATCH_PARTITION=""
SBATCH_GRES="gpu:1"
SBATCH_TIME="24:00:00"
SBATCH_CPUS="8"
SBATCH_MEM="64G"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_finwhale_highconf_month_inference.sh --month YYYY-MM [options]

Purpose:
  Extract one 2025 month of Clayoquot raw audio, prepare full-clip train-style MATs,
  run the selected fin whale detector, cluster high-confidence overlapping windows,
  merge event media, and write strict O3 predictions.json plus app review JSON.

Key options:
  --month YYYY-MM
  --device-code CODE                 Default: ICLISTENHF6016
  --final2025-root PATH              Default: /project/.../final2025_resnet_20260423
  --archive PATH                     Default: canonical clayoquot_raw_audio.tar.zst
  --available-filenames PATH         Default: canonical available filenames list
  --checkpoint PATH                  Default: winning joint_scratch checkpoint under final2025-root
  --dataset-doc PATH                 Default: historical training dataset_documentation.json
  --reviewed-workbook PATH           Default: final 2025 annotation workbook in repo data/finwhales
  --include-reviewed                 Predict every available month clip instead of excluding workbook-reviewed clips
  --out-dir PATH                     Default: final2025-root/high_confidence_predictions/YYYY-MM_<run-label>
  --run-label LABEL                  Default: joint_scratch_highconf_unreviewed_ws48_low0p7_high0p9_min3_gap15

Inference/postprocess options:
  --window-step N                    Default: 48
  --low-threshold X                  Default: 0.70
  --high-threshold X                 Default: 0.90
  --min-members N                    Default: 3
  --max-gap-seconds X                Default: 15
  --spec-backend auto|scipy|torch    Default: auto

SBATCH options:
  --partition NAME
  --gres SPEC                        Default: gpu:1
  --time HH:MM:SS                    Default: 24:00:00
  --cpus-per-task N                  Default: 8
  --mem SIZE                         Default: 64G
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --month) MONTH="$2"; shift 2 ;;
    --device-code) DEVICE_CODE="$2"; shift 2 ;;
    --final2025-root) FINAL2025_ROOT="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --available-filenames) AVAILABLE_FILENAMES="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset-doc) DATASET_DOC="$2"; shift 2 ;;
    --reviewed-workbook) REVIEWED_WORKBOOK="$2"; shift 2 ;;
    --include-reviewed) EXCLUDE_REVIEWED="false"; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --run-label) RUN_LABEL="$2"; shift 2 ;;
    --window-step) WINDOW_STEP="$2"; shift 2 ;;
    --low-threshold) LOW_THRESHOLD="$2"; shift 2 ;;
    --high-threshold) HIGH_THRESHOLD="$2"; shift 2 ;;
    --min-members) MIN_MEMBERS="$2"; shift 2 ;;
    --max-gap-seconds) MAX_GAP_SECONDS="$2"; shift 2 ;;
    --spec-backend) SPEC_BACKEND="$2"; shift 2 ;;
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

if [[ ! "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
  echo "Error: --month must be YYYY-MM" >&2
  usage
  exit 1
fi

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$FINAL2025_ROOT/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"
fi
if [[ -z "$DATASET_DOC" ]]; then
  DATASET_DOC="$FINAL2025_ROOT/historical/training_dataset/dataset_documentation.json"
fi
if [[ -z "$REVIEWED_WORKBOOK" ]]; then
  REVIEWED_WORKBOOK="$REPO_ROOT/data/finwhales/ONC_ClayoquotSlope2025_Annotations_Cetaceans_Instrument_EQ_Sonar_Unknown.xlsx"
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$FINAL2025_ROOT/high_confidence_predictions/${MONTH}_${RUN_LABEL}"
fi

for required in "$ARCHIVE" "$AVAILABLE_FILENAMES" "$CHECKPOINT" "$DATASET_DOC"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
if [[ "$EXCLUDE_REVIEWED" == "true" && ! -e "$REVIEWED_WORKBOOK" ]]; then
  echo "Missing reviewed-workbook path needed for unreviewed-only inference: $REVIEWED_WORKBOOK" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/logs"
JOB_SCRIPT="$OUT_DIR/logs/run_highconf_month_${MONTH}.sbatch"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=finwhale_hc_${MONTH}
#SBATCH --output=$OUT_DIR/logs/slurm-%j.out
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

cat >> "$JOB_SCRIPT" <<'EOF'

set -euo pipefail

echo "Started at $(date -Is)"
cd "__REPO_ROOT__"
source .venv/bin/activate

MONTH="__MONTH__"
DEVICE_CODE="__DEVICE_CODE__"
ARCHIVE="__ARCHIVE__"
AVAILABLE_FILENAMES="__AVAILABLE_FILENAMES__"
CHECKPOINT="__CHECKPOINT__"
DATASET_DOC="__DATASET_DOC__"
REVIEWED_WORKBOOK="__REVIEWED_WORKBOOK__"
EXCLUDE_REVIEWED="__EXCLUDE_REVIEWED__"
OUT_DIR="__OUT_DIR__"
WINDOW_STEP="__WINDOW_STEP__"
LOW_THRESHOLD="__LOW_THRESHOLD__"
HIGH_THRESHOLD="__HIGH_THRESHOLD__"
MIN_MEMBERS="__MIN_MEMBERS__"
MAX_GAP_SECONDS="__MAX_GAP_SECONDS__"
CROP_SIZE="__CROP_SIZE__"
BATCH_SIZE="__BATCH_SIZE__"
NUM_WORKERS="__NUM_WORKERS__"
DEVICE="__DEVICE__"
SPEC_BACKEND="__SPEC_BACKEND__"
ADJACENT_MINUTES="__ADJACENT_MINUTES__"

RAW_DIR="$OUT_DIR/raw_audio"
MAT_DIR="$OUT_DIR/mat_files"
MANIFEST_DIR="$OUT_DIR/manifests"
EXPORT_DIR="$OUT_DIR/exported_windows"
EVENT_MEDIA_DIR="$OUT_DIR/predictions_postprocessed_events_media"
WINDOW_JSON="$OUT_DIR/predictions_window.json"
APP_JSON="$OUT_DIR/predictions_postprocessed.app.json"
STRICT_JSON="$OUT_DIR/predictions_postprocessed.o3.json"
APP_CANONICAL_JSON="$OUT_DIR/predictions.json"
EVENTS_CSV="$OUT_DIR/predictions_postprocessed_events.csv"
SUMMARY_MD="$OUT_DIR/predictions_postprocessed_summary.md"
DEBUG_JSON="$OUT_DIR/predictions_postprocessed_debug.json"
METADATA_JSON="$OUT_DIR/metadata.json"

mkdir -p "$RAW_DIR" "$MAT_DIR" "$MANIFEST_DIR" "$EXPORT_DIR" "$EVENT_MEDIA_DIR"

python - "$MONTH" "$DEVICE_CODE" "$AVAILABLE_FILENAMES" "$MANIFEST_DIR" "$ADJACENT_MINUTES" "$REVIEWED_WORKBOOK" "$EXCLUDE_REVIEWED" <<'PY'
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

month, device, available_path, manifest_dir, adjacent_minutes, reviewed_workbook, exclude_reviewed = sys.argv[1:8]
adjacent = timedelta(minutes=float(adjacent_minutes))
year, mon = [int(part) for part in month.split("-")]
start = datetime(year, mon, 1, tzinfo=timezone.utc)
if mon == 12:
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
else:
    end = datetime(year, mon + 1, 1, tzinfo=timezone.utc)

ts_re = re.compile(r"^(?P<device>[^_]+)_(?P<ts>\d{8}T\d{6})(?:\.\d+)?Z")

def file_key(name: str) -> str | None:
    match = ts_re.search(Path(name).name)
    if not match:
        return None
    return f"{match.group('device')}_{match.group('ts')}"

def file_ts(name: str) -> datetime | None:
    match = ts_re.search(Path(name).name)
    if not match:
        return None
    return datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

reviewed_keys = set()
reviewed_names = set()
if exclude_reviewed.lower() == "true":
    from src.dataset.part2_annotations import load_workbook_sheets, normalize_audio_filename

    for sheet in load_workbook_sheets(Path(reviewed_workbook)):
        if sheet.name == "READ ME":
            continue
        for row in sheet.rows:
            name = normalize_audio_filename(row.get("filename", ""))
            if not name or not name.startswith(device + "_"):
                continue
            ts = file_ts(name)
            if ts is None or not (start <= ts < end):
                continue
            key = file_key(name)
            if key:
                reviewed_keys.add(key)
            reviewed_names.add(Path(name).name)

all_target = []
target = []
excluded_reviewed = []
selected_by_key = {}
available_by_key = {}

with open(available_path, "r", encoding="utf-8") as handle:
    for raw in handle:
        name = Path(raw.strip()).name
        if not name or not name.startswith(device + "_"):
            continue
        ts = file_ts(name)
        key = file_key(name)
        if ts is None or key is None:
            continue
        available_by_key.setdefault(key, (ts, name))
        if start <= ts < end:
            all_target.append((ts, name))
            if exclude_reviewed.lower() == "true" and (key in reviewed_keys or name in reviewed_names):
                excluded_reviewed.append((ts, name))
            else:
                target.append((ts, name))

all_target = sorted(set(all_target))
target = sorted(set(target))
excluded_reviewed = sorted(set(excluded_reviewed))
for ts, name in target:
    key = file_key(name)
    if key:
        selected_by_key[key] = (ts, name)
    for delta in (-adjacent, adjacent):
        adj_ts = ts + delta
        adj_key = f"{device}_{adj_ts.strftime('%Y%m%dT%H%M%S')}"
        selected_by_key.setdefault(adj_key, available_by_key.get(adj_key))
selected = sorted(set(value for value in selected_by_key.values() if value is not None))

if not all_target:
    raise SystemExit(f"No available month audio files found for {device} {month}")
if not target:
    raise SystemExit(f"All {len(all_target)} available {device} {month} files were marked reviewed/examined")
if not selected:
    raise SystemExit(f"No selected audio files found for {device} {month}")

manifest_dir = Path(manifest_dir)
manifest_dir.mkdir(parents=True, exist_ok=True)
with open(manifest_dir / "target_clip_list_all_available.txt", "w", encoding="utf-8") as out:
    for _, name in all_target:
        out.write(name + "\n")
with open(manifest_dir / "target_clip_list.txt", "w", encoding="utf-8") as out:
    for _, name in target:
        out.write(name + "\n")
with open(manifest_dir / "excluded_reviewed_clip_list.txt", "w", encoding="utf-8") as out:
    for _, name in excluded_reviewed:
        out.write(name + "\n")
with open(manifest_dir / "reviewed_workbook_clip_keys.txt", "w", encoding="utf-8") as out:
    for key in sorted(reviewed_keys):
        out.write(key + "\n")
with open(manifest_dir / "selected_archive_members.txt", "w", encoding="utf-8") as out:
    for _, name in selected:
        out.write("raw_audio/" + name + "\n")
with open(manifest_dir / "selected_filenames.txt", "w", encoding="utf-8") as out:
    for _, name in selected:
        out.write(name + "\n")
with open(manifest_dir / "selection_summary.txt", "w", encoding="utf-8") as out:
    out.write(f"month={month}\n")
    out.write(f"device={device}\n")
    out.write(f"exclude_reviewed={exclude_reviewed}\n")
    out.write(f"reviewed_workbook={reviewed_workbook}\n")
    out.write(f"available_month_files={len(all_target)}\n")
    out.write(f"reviewed_or_examined_files_excluded={len(excluded_reviewed)}\n")
    out.write(f"unreviewed_target_files={len(target)}\n")
    out.write(f"selected_files_including_adjacent={len(selected)}\n")
    out.write(f"target_start={target[0][0].isoformat()}\n")
    out.write(f"target_end={target[-1][0].isoformat()}\n")

print(f"Available month files: {len(all_target)}")
print(f"Reviewed/examined files excluded: {len(excluded_reviewed)}")
print(f"Unreviewed target month files: {len(target)}")
print(f"Files selected for extraction including adjacent context: {len(selected)}")
PY

python - "$MANIFEST_DIR/selected_archive_members.txt" "$RAW_DIR" "$MANIFEST_DIR/missing_archive_members.txt" <<'PY'
import sys
from pathlib import Path

members_path, raw_dir, missing_path = [Path(arg) for arg in sys.argv[1:4]]
raw_dir.mkdir(parents=True, exist_ok=True)
missing = []
with open(members_path, "r", encoding="utf-8") as handle:
    for line in handle:
        member = line.strip()
        if not member:
            continue
        name = Path(member).name
        if not (raw_dir / name).exists():
            missing.append(member)
with open(missing_path, "w", encoding="utf-8") as out:
    for member in missing:
        out.write(member + "\n")
print(f"Missing extracted audio files: {len(missing)}")
PY

if [[ -s "$MANIFEST_DIR/missing_archive_members.txt" ]]; then
  echo "Extracting missing audio from archive..."
  tar --use-compress-program=unzstd -xf "$ARCHIVE" -C "$OUT_DIR" -T "$MANIFEST_DIR/missing_archive_members.txt"
else
  echo "All selected audio files already extracted."
fi

python - "$MANIFEST_DIR/target_clip_list.txt" "$RAW_DIR" <<'PY'
import sys
from pathlib import Path

target_list = Path(sys.argv[1])
raw_dir = Path(sys.argv[2])
missing = []
for line in target_list.read_text(encoding="utf-8").splitlines():
    name = line.strip()
    if name and not (raw_dir / name).exists():
        missing.append(name)
if missing:
    print("Missing target audio after extraction:")
    for name in missing[:20]:
        print(name)
    raise SystemExit(f"{len(missing)} target files missing")
print("All target audio files are present.")
PY

echo "Preparing full-clip MAT files with spec-backend=$SPEC_BACKEND..."
python -u scripts/data/part2/prepare_trainstyle_windows.py \
  --slide \
  --clip-list "$MANIFEST_DIR/target_clip_list.txt" \
  --audio-dir "$RAW_DIR" \
  --dataset-doc "$DATASET_DOC" \
  --out-dir "$MAT_DIR" \
  --device "$DEVICE_CODE" \
  --spec-backend "$SPEC_BACKEND" \
  --window-s 300 \
  --step-s 300

python - "$MAT_DIR" "$OUT_DIR" "$DEVICE_CODE" "$MONTH" "$DATASET_DOC" "$METADATA_JSON" "$SPEC_BACKEND" <<'PY'
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

mat_dir, out_dir, device, month, dataset_doc, metadata_json, spec_backend = sys.argv[1:8]
mat_dir = Path(mat_dir)
out_dir = Path(out_dir)
metadata_json = Path(metadata_json)
stem_re = re.compile(r"^(?P<src>.+)_(?P<start>-?\d+(?:\.\d+)?)s_(?P<end>-?\d+(?:\.\d+)?)s_window$")
ts_re = re.compile(r"_(?P<ts>\d{8}T\d{6})(?:\.\d+)?Z")
rows = []
starts = []
ends = []

for mat_path in sorted(mat_dir.glob("*.mat")):
    match = stem_re.match(mat_path.stem)
    if not match:
        continue
    source_audio = match.group("src")
    start_s = float(match.group("start"))
    end_s = float(match.group("end"))
    ts_match = ts_re.search(source_audio)
    audio_timestamp = ""
    audio_end = ""
    if ts_match:
        base = datetime.strptime(ts_match.group("ts"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        start_dt = base + timedelta(seconds=start_s)
        end_dt = base + timedelta(seconds=end_s)
        audio_timestamp = start_dt.isoformat()
        audio_end = end_dt.isoformat()
        starts.append(start_dt)
        ends.append(end_dt)
    raw_audio_rel = f"raw_audio/{source_audio}" if (out_dir / "raw_audio" / source_audio).exists() else None
    rows.append({
        "file_id": mat_path.stem,
        "mat_path": str(mat_path.relative_to(out_dir)),
        "source_audio": source_audio,
        "raw_audio_path": raw_audio_rel,
        "segment_start_sec": start_s,
        "segment_end_sec": end_s,
        "audio_timestamp": audio_timestamp,
        "audio_end_time": audio_end,
    })

if not rows:
    raise SystemExit(f"No MAT metadata rows generated from {mat_dir}")

metadata = {
    "version": "1.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "data_source": {
        "device_code": device,
        "date_from": min(starts).isoformat() if starts else f"{month}-01T00:00:00+00:00",
        "date_to": max(ends).isoformat() if ends else "",
    },
    "spectrogram_config": {
        "context_duration": 300.0,
        "window_duration": 1.0,
        "overlap": 0.9,
        "frequency_limits": {"min": 5.0, "max": 100.0},
        "crop_size": 96,
        "edge_context": 10.5,
        "source": {
            "type": "computed",
            "pipeline": "submit_finwhale_highconf_month_inference.full_clip_spectrograms",
            "dataset_doc": str(dataset_doc),
            "backend": spec_backend,
        },
    },
    "files": rows,
    "month_package": {
        "month": month,
        "target_selection": "unreviewed_only",
        "mat_count": len(rows),
        "target_audio_count": len(rows),
    },
}
metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Wrote metadata: {metadata_json}")
print(f"MAT files in metadata: {len(rows)}")
PY

echo "Running inference and high-confidence event postprocessing..."
python -u scripts/inference/run_inference.py \
  --mat-dir "$MAT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --dataset-metadata "$METADATA_JSON" \
  --output-json "$WINDOW_JSON" \
  --sliding-window \
  --window-step "$WINDOW_STEP" \
  --crop-size "$CROP_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --export-crops \
  --export-threshold "$LOW_THRESHOLD" \
  --export-dir "$EXPORT_DIR" \
  --raw-audio-dir "$RAW_DIR" \
  --postprocess \
  --postprocess-output-json "$APP_JSON" \
  --postprocess-low-threshold "$LOW_THRESHOLD" \
  --postprocess-high-threshold "$HIGH_THRESHOLD" \
  --postprocess-min-members "$MIN_MEMBERS" \
  --postprocess-max-gap-seconds "$MAX_GAP_SECONDS" \
  --postprocess-events-csv "$EVENTS_CSV" \
  --postprocess-summary-md "$SUMMARY_MD" \
  --postprocess-debug-json "$DEBUG_JSON" \
  --postprocess-merge-event-media \
  --postprocess-event-media-dir "$EVENT_MEDIA_DIR" \
  --postprocess-replace-items-with-events \
  --postprocess-merge-across-source-audio

python scripts/inference/transform_predictions_to_o3.py \
  --input-json "$APP_JSON" \
  --output-json "$STRICT_JSON" \
  --overwrite

cp "$STRICT_JSON" "$APP_CANONICAL_JSON"

python - "$OUT_DIR" "$APP_JSON" "$STRICT_JSON" "$WINDOW_JSON" <<'PY'
import json
import sys
from pathlib import Path

out_dir, app_json, strict_json, window_json = [Path(arg) for arg in sys.argv[1:5]]
app = json.loads(app_json.read_text(encoding="utf-8"))
strict = json.loads(strict_json.read_text(encoding="utf-8"))
items = app.get("items", [])
missing_paths = []
for item in items:
    paths = item.get("paths") if isinstance(item, dict) else None
    if not isinstance(paths, dict):
        missing_paths.append((item.get("item_id", "unknown"), "paths"))
        continue
    for key in ("audio_path", "spectrogram_mat_path"):
        value = paths.get(key)
        if not value:
            missing_paths.append((item.get("item_id", "unknown"), key))
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = app_json.parent / candidate
        if not candidate.exists():
            missing_paths.append((item.get("item_id", "unknown"), key))

summary = {
    "predictions_json": str(out_dir / "predictions.json"),
    "postprocessed_app_json": str(app_json),
    "strict_o3_json": str(strict_json),
    "window_predictions_json": str(window_json),
    "schema_version": strict.get("schema_version"),
    "event_items": len(items),
    "strict_o3_items": len(strict.get("items", [])),
    "events_metadata_count": len(app.get("events", [])) if isinstance(app.get("events"), list) else None,
    "missing_event_media_paths": len(missing_paths),
}
(out_dir / "package_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with open(out_dir / "package_summary.md", "w", encoding="utf-8") as handle:
    handle.write("# High-Confidence Fin Whale Package\n\n")
    for key, value in summary.items():
        handle.write(f"- {key}: `{value}`\n")
    if missing_paths:
        handle.write("\n## Missing Paths\n")
        for item_id, key in missing_paths[:50]:
            handle.write(f"- `{item_id}` missing `{key}`\n")
if missing_paths:
    raise SystemExit(f"{len(missing_paths)} event media paths are missing; see package_summary.md")
print(json.dumps(summary, indent=2))
PY

echo "Finished at $(date -Is)"
EOF

python - "$JOB_SCRIPT" \
  "__REPO_ROOT__=$REPO_ROOT" \
  "__MONTH__=$MONTH" \
  "__DEVICE_CODE__=$DEVICE_CODE" \
  "__ARCHIVE__=$ARCHIVE" \
  "__AVAILABLE_FILENAMES__=$AVAILABLE_FILENAMES" \
  "__CHECKPOINT__=$CHECKPOINT" \
  "__DATASET_DOC__=$DATASET_DOC" \
  "__REVIEWED_WORKBOOK__=$REVIEWED_WORKBOOK" \
  "__EXCLUDE_REVIEWED__=$EXCLUDE_REVIEWED" \
  "__OUT_DIR__=$OUT_DIR" \
  "__WINDOW_STEP__=$WINDOW_STEP" \
  "__LOW_THRESHOLD__=$LOW_THRESHOLD" \
  "__HIGH_THRESHOLD__=$HIGH_THRESHOLD" \
  "__MIN_MEMBERS__=$MIN_MEMBERS" \
  "__MAX_GAP_SECONDS__=$MAX_GAP_SECONDS" \
  "__CROP_SIZE__=$CROP_SIZE" \
  "__BATCH_SIZE__=$BATCH_SIZE" \
  "__NUM_WORKERS__=$NUM_WORKERS" \
  "__DEVICE__=$DEVICE" \
  "__SPEC_BACKEND__=$SPEC_BACKEND" \
  "__ADJACENT_MINUTES__=$ADJACENT_MINUTES" <<'PY'
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

SBATCH_CMD=(sbatch)
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run job script: $JOB_SCRIPT"
  echo "Output directory: $OUT_DIR"
  exit 0
fi

JOB_OUTPUT="$("${SBATCH_CMD[@]}" "$JOB_SCRIPT")"
echo "$JOB_OUTPUT"
echo "Job script: $JOB_SCRIPT"
echo "Output directory: $OUT_DIR"
