#!/bin/bash
# Build a bounded DCLDE 2027 killer-whale Oo-repair prep bundle on Nibi.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

FINAL2025_ROOT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423"
DATASET_DOC="$FINAL2025_ROOT/historical/training_dataset/dataset_documentation.json"
WEEKEND_ROOT="${SCRATCH:-/scratch/$USER}/whale-call-analysis/multispecies_weekend_20260502"
AUDIT_ROOT="$WEEKEND_ROOT/audits/dclde_2027_killer_whales"
ANNOTATIONS_CSV="$AUDIT_ROOT/Annotations.csv"
GCS_OBJECT_LIST="$AUDIT_ROOT/gcs_objects.txt"

RUN_NAME="dclde_orca_cap200_prep"
MAX_POSITIVE="200"
MAX_HARD_NEGATIVE="200"
HARD_NEGATIVE_CLASSES="HW,UndBio,AB"
WINDOW_S="40"
EDGE_CONTEXT_S="10.5"
SPEC_BACKEND="torch"
VOCAB_MIN_COUNT="1"
SPLIT_STRATEGY="label_balanced"
SPLIT_SEED="2026"
SAVE_IMAGES="false"

SBATCH_PARTITION=""
SBATCH_TIME="12:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="40G"
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_dclde_orca_prep_smoke.sh [options]

This CPU-only job:
  1. Builds or reuses a GCS object list for the public DCLDE 2027 killer-whale corpus.
  2. Selects a bounded, source-balanced KW positive and HW/UndBio/AB hard-negative subset.
  3. Downloads only selected source audio files from public GCS into scratch.
  4. Generates train-style 40s MAT windows.
  5. Builds grouped, leakage-aware candidate splits.

Key options:
  --run-name NAME                 Default: dclde_orca_cap200_prep
  --weekend-root PATH             Default: $SCRATCH/whale-call-analysis/multispecies_weekend_20260502
  --audit-root PATH               Default: WEEKEND_ROOT/audits/dclde_2027_killer_whales
  --annotations-csv PATH          Default: AUDIT_ROOT/Annotations.csv
  --gcs-object-list PATH          Default: AUDIT_ROOT/gcs_objects.txt
  --dataset-doc PATH
  --max-positive N                Default: 200
  --max-hard-negative N           Default: 200
  --hard-negative-classes CSV     Default: HW,UndBio,AB
  --window-s SECONDS              Default: 40
  --edge-context-s SECONDS        Default: 10.5
  --spec-backend auto|scipy|torch Default: torch
  --split-strategy temporal|label_balanced
  --split-seed N                  Default: 2026
  --save-images                   Save diagnostic spectrogram PNGs

SBATCH:
  --partition NAME
  --time HH:MM:SS                 Default: 12:00:00
  --cpus-per-task N               Default: 4
  --mem SIZE                      Default: 40G
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; AUDIT_ROOT="$WEEKEND_ROOT/audits/dclde_2027_killer_whales"; ANNOTATIONS_CSV="$AUDIT_ROOT/Annotations.csv"; GCS_OBJECT_LIST="$AUDIT_ROOT/gcs_objects.txt"; shift 2 ;;
    --audit-root) AUDIT_ROOT="$2"; ANNOTATIONS_CSV="$AUDIT_ROOT/Annotations.csv"; GCS_OBJECT_LIST="$AUDIT_ROOT/gcs_objects.txt"; shift 2 ;;
    --annotations-csv) ANNOTATIONS_CSV="$2"; shift 2 ;;
    --gcs-object-list) GCS_OBJECT_LIST="$2"; shift 2 ;;
    --dataset-doc) DATASET_DOC="$2"; shift 2 ;;
    --max-positive) MAX_POSITIVE="$2"; shift 2 ;;
    --max-hard-negative) MAX_HARD_NEGATIVE="$2"; shift 2 ;;
    --hard-negative-classes) HARD_NEGATIVE_CLASSES="$2"; shift 2 ;;
    --window-s) WINDOW_S="$2"; shift 2 ;;
    --edge-context-s) EDGE_CONTEXT_S="$2"; shift 2 ;;
    --spec-backend) SPEC_BACKEND="$2"; shift 2 ;;
    --split-strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --split-seed) SPLIT_SEED="$2"; shift 2 ;;
    --save-images) SAVE_IMAGES="true"; shift ;;
    --partition) SBATCH_PARTITION="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

for required in "$ANNOTATIONS_CSV" "$DATASET_DOC"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

RUN_ID="${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$WEEKEND_ROOT/prep_runs/$RUN_ID"
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
echo "SLURM job: \${SLURM_JOB_ID:-none}"
cd "$REPO_ROOT"
source .venv/bin/activate

export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-/scratch/merileo/.cache/wandb}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-/scratch/merileo/.cache/pip}"
export HF_HOME="\${HF_HOME:-/scratch/merileo/.cache/huggingface}"
export TORCH_HOME="\${TORCH_HOME:-/scratch/merileo/.cache/torch}"
export MPLCONFIGDIR="\${MPLCONFIGDIR:-/scratch/merileo/.cache/matplotlib}"
mkdir -p "\$XDG_CACHE_HOME" "\$WANDB_CACHE_DIR" "\$PIP_CACHE_DIR" "\$HF_HOME" "\$TORCH_HOME" "\$MPLCONFIGDIR"

mkdir -p "$AUDIT_ROOT" "$PREP_DIR" "$RAW_DIR" "$MAT_DIR" "$SPLIT_DIR"

if [[ ! -s "$GCS_OBJECT_LIST" ]]; then
  echo "Building DCLDE GCS object list at $GCS_OBJECT_LIST"
  python - "$GCS_OBJECT_LIST" <<'PY'
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

out_path = Path(sys.argv[1])
bucket = "noaa-passive-bioacoustic"
prefix = "dclde/2027/dclde_2027_killer_whales/"
base = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
params = {"prefix": prefix, "fields": "nextPageToken,items(name,size)"}
names = []
page_token = None
while True:
    query = dict(params)
    if page_token:
        query["pageToken"] = page_token
    url = base + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names.extend(item["name"] for item in payload.get("items", []) if item.get("name"))
    page_token = payload.get("nextPageToken")
    if not page_token:
        break
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\\n".join(sorted(names)) + ("\\n" if names else ""), encoding="utf-8")
print(json.dumps({"object_count": len(names), "output": str(out_path)}, indent=2, sort_keys=True))
PY
else
  echo "Using existing GCS object list: $GCS_OBJECT_LIST"
fi

python -u scripts/data/multilabel/build_dclde_killer_whale_manifest.py \\
  --annotations-csv "$ANNOTATIONS_CSV" \\
  --output-dir "$PREP_DIR" \\
  --gcs-object-list "$GCS_OBJECT_LIST" \\
  --require-gcs-audio \\
  --max-positive "$MAX_POSITIVE" \\
  --max-hard-negative "$MAX_HARD_NEGATIVE" \\
  --hard-negative-classes "$HARD_NEGATIVE_CLASSES" \\
  --mat-rel-dir "mat_files" \\
  --vocab-min-count "$VOCAB_MIN_COUNT"

python - "$PREP_DIR/required_audio_sources.csv" "$RAW_DIR" <<'PY'
import csv
import json
import subprocess
import sys
from pathlib import Path

sources_csv = Path(sys.argv[1])
raw_dir = Path(sys.argv[2])
raw_dir.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(sources_csv.open(newline="", encoding="utf-8")))
downloaded = []
failed = []
seen = {}
for row in rows:
    clip = row["clip"]
    if clip in seen:
        continue
    seen[clip] = row
    dst = raw_dir / clip
    url = row["https_url"]
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-delay",
        "10",
        "--connect-timeout",
        "60",
        "--continue-at",
        "-",
        "--output",
        str(dst),
        url,
    ]
    print(f"Staging {clip} from {url}", flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        downloaded.append({"clip": clip, "bytes": dst.stat().st_size, "url": url})
    else:
        failed.append({"clip": clip, "returncode": result.returncode, "url": url})

summary = {
    "selected_row_count": len(rows),
    "unique_audio_count": len(seen),
    "staged_count": len(downloaded),
    "failed_count": len(failed),
    "failed": failed[:25],
    "raw_dir": str(raw_dir),
}
(raw_dir / "staging_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
(raw_dir / "staged_audio_filenames.txt").write_text("\\n".join(item["clip"] for item in downloaded) + ("\\n" if downloaded else ""), encoding="utf-8")
(raw_dir / "failed_audio_filenames.txt").write_text("\\n".join(item["clip"] for item in failed) + ("\\n" if failed else ""), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if failed:
    raise SystemExit(f"{len(failed)} selected DCLDE audio files failed to stage")
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

python - "$OUT_DIR" "$RUN_ID" "$PREP_DIR" "$RAW_DIR" "$MAT_DIR" "$SPLIT_DIR" "$GCS_OBJECT_LIST" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir, run_id, prep_dir, raw_dir, mat_dir, split_dir, gcs_object_list = map(Path, sys.argv[1:])
summary = {
    "run_id": str(run_id),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "out_dir": str(out_dir),
    "prep_dir": str(prep_dir),
    "raw_dir": str(raw_dir),
    "mat_dir": str(mat_dir),
    "split_dir": str(split_dir),
    "gcs_object_list": str(gcs_object_list),
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
