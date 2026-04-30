#!/bin/bash
# Build a tiny multi-species/background train-style MAT prep bundle on Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
EXP_ROOT="$FINAL2025_ROOT/multispecies_calltype_experiments"
ANNOTATIONS_CSV="$FINAL2025_ROOT/part2/full_bundle/manifests/annotations_all.csv"
CLIP_MANIFEST_CSV="$FINAL2025_ROOT/part2/full_bundle/manifests/clip_manifest.csv"
DATASET_DOC="$FINAL2025_ROOT/historical/training_dataset/dataset_documentation.json"
SOURCE_AUDIO_DIR="$FINAL2025_ROOT/part2/full_bundle/raw_audio"
ARCHIVE_PATH="/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio.tar.zst"
AVAILABLE_FILENAMES="/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio_available_filenames.txt"

RUN_NAME="multispecies_prep_tiny"
SPECIES="OD,Mn,Bm,Oo"
MAX_PER_SPECIES="5"
INCLUDE_FIN="true"
MAX_FIN="5"
MAX_BACKGROUND="5"
BACKGROUND_WINDOWS_PER_CLIP="1"
WINDOW_S="40"
EDGE_CONTEXT_S="10.5"
SPEC_BACKEND="torch"
VOCAB_MIN_COUNT="1"
SAVE_IMAGES="true"
SPLIT_STRATEGY="label_balanced"
SPLIT_SEED="2026"

SBATCH_PARTITION=""
SBATCH_TIME="01:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="24G"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_prep_tiny.sh [options]

Builds a bounded non-training prep job:
  1. Selects a small mixed-species/background call manifest.
  2. Stages required raw audio from an existing directory, or falls back to archive extraction.
  3. Generates train-style 40s MAT windows with the existing Part 2 prep code.
  4. Builds the multi-label MAT manifest and grouped candidate splits.

Key options:
  --run-name NAME                  Default: multispecies_prep_tiny
  --final2025-root PATH
  --exp-root PATH
  --annotations-csv PATH
  --clip-manifest-csv PATH
  --dataset-doc PATH
  --source-audio-dir PATH           Default: Part 2 full_bundle/raw_audio; set empty to use archive extraction
  --archive-path PATH
  --available-filenames PATH
  --species CSV                    Default: OD,Mn,Bm,Oo
  --max-per-species N              Default: 5
  --no-fin                         Exclude Bp examples
  --max-fin N                      Default: 5
  --max-background N               Default: 5
  --background-windows-per-clip N  Default: 1
  --window-s SECONDS               Default: 40
  --edge-context-s SECONDS         Default: 10.5
  --spec-backend auto|scipy|torch  Default: torch
  --split-strategy temporal|label_balanced
                                   Default: label_balanced
  --split-seed N                   Default: 2026
  --no-images                      Do not save diagnostic spectrogram images

SBATCH:
  --partition NAME
  --time HH:MM:SS                  Default: 01:00:00
  --cpus-per-task N                Default: 4
  --mem SIZE                       Default: 24G
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --final2025-root) FINAL2025_ROOT="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --annotations-csv) ANNOTATIONS_CSV="$2"; shift 2 ;;
    --clip-manifest-csv) CLIP_MANIFEST_CSV="$2"; shift 2 ;;
    --dataset-doc) DATASET_DOC="$2"; shift 2 ;;
    --source-audio-dir) SOURCE_AUDIO_DIR="$2"; shift 2 ;;
    --archive-path) ARCHIVE_PATH="$2"; shift 2 ;;
    --available-filenames) AVAILABLE_FILENAMES="$2"; shift 2 ;;
    --species) SPECIES="$2"; shift 2 ;;
    --max-per-species) MAX_PER_SPECIES="$2"; shift 2 ;;
    --no-fin) INCLUDE_FIN="false"; shift ;;
    --max-fin) MAX_FIN="$2"; shift 2 ;;
    --max-background) MAX_BACKGROUND="$2"; shift 2 ;;
    --background-windows-per-clip) BACKGROUND_WINDOWS_PER_CLIP="$2"; shift 2 ;;
    --window-s) WINDOW_S="$2"; shift 2 ;;
    --edge-context-s) EDGE_CONTEXT_S="$2"; shift 2 ;;
    --spec-backend) SPEC_BACKEND="$2"; shift 2 ;;
    --split-strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --split-seed) SPLIT_SEED="$2"; shift 2 ;;
    --no-images) SAVE_IMAGES="false"; shift ;;
    --partition) SBATCH_PARTITION="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

for required in "$ANNOTATIONS_CSV" "$CLIP_MANIFEST_CSV" "$DATASET_DOC"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
if [[ -n "$SOURCE_AUDIO_DIR" ]]; then
  [[ -d "$SOURCE_AUDIO_DIR" ]] || { echo "Missing source audio dir: $SOURCE_AUDIO_DIR" >&2; exit 1; }
else
  for required in "$ARCHIVE_PATH" "$AVAILABLE_FILENAMES"; do
    [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
  done
fi

RUN_ID="${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$EXP_ROOT/prep_runs/$RUN_ID"
PREP_DIR="$OUT_DIR/prep_manifest"
RAW_DIR="$OUT_DIR/raw_audio"
MAT_DIR="$OUT_DIR/mat_files"
MANIFEST_DIR="$OUT_DIR/multilabel_manifest"
SPLIT_DIR="$OUT_DIR/splits"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

JOB_SCRIPT="$LOG_DIR/${RUN_ID}.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM
EOF

if [[ -n "$SBATCH_PARTITION" ]]; then
  echo "#SBATCH --partition=$SBATCH_PARTITION" >> "$JOB_SCRIPT"
fi

cat >> "$JOB_SCRIPT" <<EOF

set -euo pipefail
echo "Started at \$(date -Is)"
echo "Host: \$(hostname)"
cd "$REPO_ROOT"
source .venv/bin/activate

mkdir -p "$PREP_DIR" "$RAW_DIR" "$MAT_DIR" "$MANIFEST_DIR" "$SPLIT_DIR"

prep_cmd=(
  python -u scripts/data/multilabel/build_multispecies_prep_manifest.py
  --annotations-csv "$ANNOTATIONS_CSV"
  --clip-manifest-csv "$CLIP_MANIFEST_CSV"
  --output-dir "$PREP_DIR"
  --species "$SPECIES"
  --max-per-species "$MAX_PER_SPECIES"
  --max-background "$MAX_BACKGROUND"
  --background-windows-per-clip "$BACKGROUND_WINDOWS_PER_CLIP"
  --context-s "$WINDOW_S"
  --edge-context-s "$EDGE_CONTEXT_S"
)
if [[ "$INCLUDE_FIN" == "true" ]]; then
  prep_cmd+=(--include-fin --max-fin "$MAX_FIN")
fi
"\${prep_cmd[@]}"

if [[ -n "$SOURCE_AUDIO_DIR" ]]; then
  python - "$PREP_DIR/required_audio_filenames.txt" "$SOURCE_AUDIO_DIR" "$RAW_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

required_txt = Path(sys.argv[1])
source_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
required = [line.strip() for line in required_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
index = {}
for pattern in ("*.flac", "*.wav"):
    for path in source_dir.rglob(pattern):
        index.setdefault(path.name, path)

output_dir.mkdir(parents=True, exist_ok=True)
staged = []
missing = []
for name in required:
    src = index.get(name)
    if src is None:
        missing.append(name)
        continue
    dst = output_dir / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)
    staged.append(name)

(output_dir / "required_audio_filenames.txt").write_text("\\n".join(required) + "\\n", encoding="utf-8")
(output_dir / "staged_audio_filenames.txt").write_text("\\n".join(staged) + ("\\n" if staged else ""), encoding="utf-8")
(output_dir / "missing_audio_filenames.txt").write_text("\\n".join(missing) + ("\\n" if missing else ""), encoding="utf-8")
summary = {
    "source": "source_audio_dir",
    "source_audio_dir": str(source_dir),
    "required_count": len(required),
    "staged_count": len(staged),
    "missing_count": len(missing),
    "missing_filenames": missing,
    "output_dir": str(output_dir),
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if missing:
    raise SystemExit(f"{len(missing)} required audio files missing from {source_dir}")
PY
else
  python -u scripts/data/part2/extract_required_audio_from_archive.py \\
    --archive-path "$ARCHIVE_PATH" \\
    --available-filenames-txt "$AVAILABLE_FILENAMES" \\
    --required-filenames-txt "$PREP_DIR/required_audio_filenames.txt" \\
    --output-dir "$RAW_DIR" \\
    --allow-missing
fi

python - "$PREP_DIR/selected_source_clips.txt" "$RAW_DIR" <<'PY'
import sys
from pathlib import Path

selected = Path(sys.argv[1])
raw_dir = Path(sys.argv[2])
missing = []
for line in selected.read_text(encoding="utf-8").splitlines():
    name = line.strip()
    if name and not (raw_dir / name).exists():
        missing.append(name)
if missing:
    print("Missing selected source clips after extraction:")
    for name in missing[:30]:
        print(name)
    raise SystemExit(f"{len(missing)} selected source clips missing")
print("All selected source clips are present.")
PY

window_cmd=(
  python -u scripts/data/part2/prepare_trainstyle_windows.py
  --calls-csv "$PREP_DIR/selected_calls.csv"
  --audio-dir "$RAW_DIR"
  --dataset-doc "$DATASET_DOC"
  --out-dir "$MAT_DIR"
  --spec-backend "$SPEC_BACKEND"
  --window-s "$WINDOW_S"
  --edge-context-s "$EDGE_CONTEXT_S"
)
if [[ "$SAVE_IMAGES" == "true" ]]; then
  window_cmd+=(--save-images)
fi
"\${window_cmd[@]}"

python -u scripts/data/multilabel/build_call_mat_manifest.py \\
  --annotations-csv "$ANNOTATIONS_CSV" \\
  --mat-dir "$MAT_DIR" \\
  --output-dir "$MANIFEST_DIR" \\
  --dataset-name "final2025_part2_multispecies_tiny_trainstyle" \\
  --match-tolerance-s 0.25 \\
  --vocab-min-count "$VOCAB_MIN_COUNT"

python -u scripts/data/multilabel/build_candidate_splits.py \\
  --manifest-csv "$MANIFEST_DIR/call_multilabel_manifest.csv" \\
  --output-dir "$SPLIT_DIR" \\
  --strategy "$SPLIT_STRATEGY" \\
  --seed "$SPLIT_SEED"

python - "$OUT_DIR" "$RUN_ID" "$RUN_NAME" "$PREP_DIR" "$RAW_DIR" "$MAT_DIR" "$MANIFEST_DIR" "$SPLIT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir, run_id, run_name, prep_dir, raw_dir, mat_dir, manifest_dir, split_dir = map(Path, sys.argv[1:])
summary = {
    "run_id": str(run_id),
    "run_name": str(run_name),
    "source_audio_dir": "$SOURCE_AUDIO_DIR",
    "prep_dir": str(prep_dir),
    "raw_dir": str(raw_dir),
    "mat_dir": str(mat_dir),
    "manifest_dir": str(manifest_dir),
    "split_dir": str(split_dir),
    "mat_count": len(list(mat_dir.glob("*.mat"))),
    "raw_audio_count": len(list(raw_dir.glob("*.flac"))) + len(list(raw_dir.glob("*.wav"))),
}
for name, path in {
    "prep_summary": prep_dir / "prep_summary.json",
    "extraction_summary": raw_dir / "summary.json",
    "call_manifest_summary": manifest_dir / "call_manifest_summary.json",
    "split_summary": split_dir / "split_summary.json",
}.items():
    if path.exists():
        summary[name] = json.loads(path.read_text(encoding="utf-8"))
out_dir.joinpath("run_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "Finished at \$(date -Is)"
EOF

chmod +x "$JOB_SCRIPT"
echo "Job script: $JOB_SCRIPT"
echo "Output dir: $OUT_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN: not submitting"
  exit 0
fi

sbatch_out="$(sbatch "$JOB_SCRIPT")"
echo "$sbatch_out"
job_id="$(echo "$sbatch_out" | awk '{print $NF}')"
echo -e "job_id\trun_id\tout_dir\tjob_script" > "$OUT_DIR/submitted_jobs.tsv"
echo -e "${job_id}\t${RUN_ID}\t${OUT_DIR}\t${JOB_SCRIPT}" >> "$OUT_DIR/submitted_jobs.tsv"
echo "$job_id"
