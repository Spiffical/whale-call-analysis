#!/bin/bash
# Download/stage a bounded BioDCASE Task 2 smoke prep bundle on Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
EXP_ROOT="$FINAL2025_ROOT/multispecies_calltype_experiments"
DATASET_DOC="$FINAL2025_ROOT/historical/training_dataset/dataset_documentation.json"

RUN_NAME="biodcase_task2_prep_smoke"
ZENODO_URL="https://zenodo.org/records/18832958/files/2026_BioDCASE_development_set.zip?download=1"
DATA_ROOT="$EXP_ROOT/external_data/biodcase2026_task2"
ZIP_PATH="$DATA_ROOT/downloads/2026_BioDCASE_development_set.zip"
EXTRACT_DIR="$DATA_ROOT/extracted"
DEV_DIR="$EXTRACT_DIR/2026_BioDCASE_development_set"

MAX_PER_LABEL="50"
MAX_BACKGROUND="100"
WINDOW_S="40"
EDGE_CONTEXT_S="10.5"
SPEC_BACKEND="torch"
SPLIT_STRATEGY="label_balanced"
SPLIT_SEED="2026"
SAVE_IMAGES="true"

SBATCH_PARTITION=""
SBATCH_TIME="04:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="32G"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_biodcase_task2_prep_smoke.sh [options]

This CPU-only job:
  1. Downloads the 2026 BioDCASE Task 2 development ZIP if missing.
  2. Extracts it if missing.
  3. Builds a bounded train-set manifest from BioDCASE annotations.
  4. Stages selected audio into unique dataset-prefixed filenames.
  5. Generates train-style 40s MAT windows with existing prep code.
  6. Builds grouped splits from the expected multi-label manifest.

Key options:
  --run-name NAME
  --exp-root PATH
  --data-root PATH
  --zip-path PATH
  --zenodo-url URL
  --dataset-doc PATH
  --max-per-label N       Default: 50
  --max-background N      Default: 100
  --window-s SECONDS      Default: 40
  --edge-context-s SEC    Default: 10.5
  --spec-backend BACKEND  Default: torch
  --split-strategy temporal|label_balanced
  --split-seed N
  --no-images

SBATCH:
  --partition NAME
  --time HH:MM:SS         Default: 04:00:00
  --cpus-per-task N       Default: 4
  --mem SIZE              Default: 32G
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; DATA_ROOT="$EXP_ROOT/external_data/biodcase2026_task2"; ZIP_PATH="$DATA_ROOT/downloads/2026_BioDCASE_development_set.zip"; EXTRACT_DIR="$DATA_ROOT/extracted"; DEV_DIR="$EXTRACT_DIR/2026_BioDCASE_development_set"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; ZIP_PATH="$DATA_ROOT/downloads/2026_BioDCASE_development_set.zip"; EXTRACT_DIR="$DATA_ROOT/extracted"; DEV_DIR="$EXTRACT_DIR/2026_BioDCASE_development_set"; shift 2 ;;
    --zip-path) ZIP_PATH="$2"; shift 2 ;;
    --zenodo-url) ZENODO_URL="$2"; shift 2 ;;
    --dataset-doc) DATASET_DOC="$2"; shift 2 ;;
    --max-per-label) MAX_PER_LABEL="$2"; shift 2 ;;
    --max-background) MAX_BACKGROUND="$2"; shift 2 ;;
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

[[ -e "$DATASET_DOC" ]] || { echo "Missing dataset doc: $DATASET_DOC" >&2; exit 1; }

RUN_ID="${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$EXP_ROOT/prep_runs/$RUN_ID"
PREP_DIR="$OUT_DIR/prep_manifest"
RAW_DIR="$OUT_DIR/raw_audio"
MAT_DIR="$OUT_DIR/mat_files"
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

export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_CACHE_DIR" "\$PIP_CACHE_DIR"

mkdir -p "$DATA_ROOT/downloads" "$EXTRACT_DIR" "$PREP_DIR" "$RAW_DIR" "$MAT_DIR" "$SPLIT_DIR"

zip_ok=false
if [[ -s "$ZIP_PATH" ]]; then
  echo "Checking existing ZIP: $ZIP_PATH"
  if unzip -tq "$ZIP_PATH" >/dev/null 2>&1; then
    zip_ok=true
    echo "Existing ZIP passed unzip integrity check."
  else
    echo "Existing ZIP is missing/incomplete/corrupt; resuming download."
  fi
fi

if [[ "$zip_ok" != "true" ]]; then
  echo "Downloading/resuming BioDCASE development ZIP to $ZIP_PATH"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$ZIP_PATH" "$ZENODO_URL"
  else
    curl -L --continue-at - --output "$ZIP_PATH" "$ZENODO_URL"
  fi
  unzip -tq "$ZIP_PATH" >/dev/null
fi

EXTRACT_MARKER="$EXTRACT_DIR/.biodcase2026_development_extract_complete"
if [[ ! -f "$EXTRACT_MARKER" || ! -d "$DEV_DIR/train/annotations" || ! -d "$DEV_DIR/train/audio" ]]; then
  echo "Extracting $ZIP_PATH to $EXTRACT_DIR"
  unzip -q -o "$ZIP_PATH" -d "$EXTRACT_DIR"
  touch "$EXTRACT_MARKER"
else
  echo "Using existing extracted dataset: $DEV_DIR"
fi

mapfile -t annotations < <(find "$DEV_DIR/train/annotations" -maxdepth 1 -type f -name '*.csv' | sort)
if [[ "\${#annotations[@]}" -eq 0 ]]; then
  echo "No train annotations found under $DEV_DIR/train/annotations" >&2
  exit 2
fi

manifest_cmd=(
  python -u scripts/data/multilabel/build_biodcase_task2_manifest.py
  --output-dir "$PREP_DIR"
  --dataset-name "biodcase2026_task2_train"
  --audio-root "$DEV_DIR/train/audio"
  --require-existing-audio
  --max-per-label "$MAX_PER_LABEL"
  --max-background "$MAX_BACKGROUND"
  --clip-name-mode dataset_prefix
)
for ann in "\${annotations[@]}"; do
  manifest_cmd+=(--annotations-csv "\$ann")
done
"\${manifest_cmd[@]}"

python - "$PREP_DIR/selected_calls.csv" "$DEV_DIR/train/audio" "$RAW_DIR" <<'PY'
import csv
import json
import shutil
import sys
from pathlib import Path

selected_csv = Path(sys.argv[1])
audio_root = Path(sys.argv[2])
raw_dir = Path(sys.argv[3])
raw_dir.mkdir(parents=True, exist_ok=True)

staged = []
missing = []
for row in csv.DictReader(selected_csv.open(newline="", encoding="utf-8")):
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

summary = {
    "selected_count": len(staged) + len(missing),
    "staged_count": len(staged),
    "missing_count": len(missing),
    "missing": missing[:25],
    "raw_dir": str(raw_dir),
}
(raw_dir / "staging_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if missing:
    raise SystemExit(f"{len(missing)} selected audio files were missing")
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

cp "$PREP_DIR/expected_multilabel_manifest.csv" "$OUT_DIR/call_multilabel_manifest.csv"
cp "$PREP_DIR/label_vocabulary.json" "$OUT_DIR/label_vocabulary.json"

python -u scripts/data/multilabel/build_candidate_splits.py \\
  --manifest-csv "$OUT_DIR/call_multilabel_manifest.csv" \\
  --output-dir "$SPLIT_DIR" \\
  --strategy "$SPLIT_STRATEGY" \\
  --seed "$SPLIT_SEED"

python - "$OUT_DIR" "$RUN_ID" "$PREP_DIR" "$RAW_DIR" "$MAT_DIR" "$SPLIT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir, run_id, prep_dir, raw_dir, mat_dir, split_dir = map(Path, sys.argv[1:])
summary = {
    "run_id": str(run_id),
    "out_dir": str(out_dir),
    "prep_dir": str(prep_dir),
    "raw_dir": str(raw_dir),
    "mat_dir": str(mat_dir),
    "split_dir": str(split_dir),
    "mat_count": len(list(mat_dir.glob("*.mat"))),
}
for name, path in {
    "prep_summary": prep_dir / "prep_summary.json",
    "staging_summary": raw_dir / "staging_summary.json",
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
