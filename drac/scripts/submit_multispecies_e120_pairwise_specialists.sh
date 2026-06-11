#!/bin/bash
# Build pairwise species manifests and submit small resumable MIG training jobs.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
DATASET_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502/mat_archives/multiband40s_20260514T002301Z/extracted"
SOURCE_MANIFEST="$WEEKEND_ROOT/manifests/e100_onc_only_blocked_nov_validation_20260611T020900Z/E101_stage2_ONConly_blocked_nov20_25_30_val/standardized_manifest.csv"
TRAIN_SCRIPT="$WEEKEND_ROOT/pipeline_runs/e44_multiclass_production_20260610T052500Z/train_multiband_multiclass.py"
FIN_INIT_CKPT="/project/6070467/merileo/data/finwhales/final2025_resnet_20260423/benchmark/benchmark_runs/final2025_resnet_20260423/runs/joint_scratch_seed1337/train/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-pcmedge_mix-seed1337-mf1-joint_scratch_seed1337/best.pt"
PYTHON_BIN="python3"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST_ROOT=""
EPOCHS="45"
LR="0.00003"
DROPOUT="0.3"
BATCH_SIZE="32"
NUM_WORKERS="4"
STOP_AFTER_SECONDS="9000"
SBATCH_TIME="03:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="48G"
SBATCH_GRES="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
DEPENDENCY=""
DRY_RUN="false"
PAIRS=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e120_pairwise_specialists.sh [options]

Build pairwise ONC-only species manifests and submit one initial 3-hour MIG job
plus one afterany continuation for each pair. Intended follow-up to E118 when
blue-vs-fin/humpback discrimination needs direct pairwise specialists.

Options:
  --pair A:B                  Species pair, e.g. Bm:Bp. May be repeated.
                              Default: Bm:Bp and Bm:Mn.
  --weekend-root PATH         Default: /scratch/.../multispecies_weekend_20260502
  --repo-root PATH            Default: $weekend_root/repo_e24_expert_hparam_68be99f
  --dataset-root PATH         Multiband MAT extracted dataset root
  --source-manifest PATH      Source standardized manifest to filter
  --train-script PATH         Multiclass trainer path
  --fin-init-checkpoint PATH  Initialization checkpoint
  --python-bin NAME           Python command for manifest creation and jobs. Default: python3
  --manifest-root PATH        Default: $weekend_root/manifests/e120_pairwise_specialists_$stamp
  --stamp STAMP              Default: current UTC stamp
  --dependency SPEC           Passed to initial sbatch, e.g. afterany:123
  --epochs N                  Default: 45
  --lr LR                     Default: 0.00003
  --dropout P                 Default: 0.3
  --batch-size N              Default: 32
  --num-workers N             Default: 4
  --stop-after-seconds N      Default: 9000
  --time HH:MM:SS             Default: 03:00:00
  --cpus-per-task N           Default: 4
  --mem MEM                   Default: 48G
  --gres GRES                 Default: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
  --dry-run                   Write manifests/scripts but do not submit jobs
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pair) PAIRS+=("$2"); shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --source-manifest) SOURCE_MANIFEST="$2"; shift 2 ;;
    --train-script) TRAIN_SCRIPT="$2"; shift 2 ;;
    --fin-init-checkpoint) FIN_INIT_CKPT="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --stop-after-seconds) STOP_AFTER_SECONDS="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${#PAIRS[@]}" -eq 0 ]]; then
  PAIRS=("Bm:Bp" "Bm:Mn")
fi
if [[ -z "$MANIFEST_ROOT" ]]; then
  MANIFEST_ROOT="$WEEKEND_ROOT/manifests/e120_pairwise_specialists_${STAMP}"
fi
if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "Missing source manifest: $SOURCE_MANIFEST" >&2
  exit 1
fi

mkdir -p "$MANIFEST_ROOT"
SUBMITTED_TSV="$MANIFEST_ROOT/e120_pairwise_submitted.tsv"
echo -e "pair\tinitial_job_id\tcontinuation_job_id\trun_dir\tmanifest\tvocab\tjob_script" > "$SUBMITTED_TSV"

for pair in "${PAIRS[@]}"; do
  IFS=: read -r SPECIES_A SPECIES_B extra <<<"$pair"
  if [[ -n "${extra:-}" || -z "${SPECIES_A:-}" || -z "${SPECIES_B:-}" || "$SPECIES_A" == "$SPECIES_B" ]]; then
    echo "Invalid --pair '$pair'; expected distinct species codes like Bm:Bp" >&2
    exit 1
  fi
  PAIR_TAG="${SPECIES_A}${SPECIES_B}"
  VARIANT_DIR="$MANIFEST_ROOT/E120_pairwise_${PAIR_TAG}_ONConly_blocked_nov20_25_30_val"
  RUN_DIR="$WEEKEND_ROOT/runs/E120_pairwise_${PAIR_TAG}_ONConly_3band_lr3e5_${STAMP}"
  LOG_DIR="$RUN_DIR/logs"
  TRAIN_DIR="$RUN_DIR/train"
  mkdir -p "$VARIANT_DIR" "$LOG_DIR" "$TRAIN_DIR"
  MANIFEST="$VARIANT_DIR/standardized_manifest.csv"
  VOCAB="$VARIANT_DIR/label_vocabulary.json"
  COUNTS="$VARIANT_DIR/manifest_counts.json"

  "$PYTHON_BIN" - <<'PYMAKE' "$SOURCE_MANIFEST" "$MANIFEST" "$VOCAB" "$COUNTS" "$SPECIES_A" "$SPECIES_B"
import csv, json, sys
from collections import Counter
from pathlib import Path

src, out_csv, out_vocab, out_counts = map(Path, sys.argv[1:5])
species_a, species_b = sys.argv[5], sys.argv[6]
display = {
    "Bm": ("Blue whale", "Biophony > Marine mammal > Cetacean > Baleen whale > Blue whale"),
    "Bp": ("Fin whale", "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"),
    "Mn": ("Humpback whale", "Biophony > Marine mammal > Cetacean > Baleen whale > Humpback whale"),
}
keep = {species_a: f"species:{species_a}", species_b: f"species:{species_b}"}
rows = []
with src.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    fields = list(reader.fieldnames or [])
    for row in reader:
        species = (row.get("species") or row.get("species_code") or "").strip()
        if species not in keep:
            continue
        label = keep[species]
        row["species"] = species
        row["species_code"] = species
        row["canonical_species"] = species
        row["label_ids"] = label
        row["canonical_label_ids"] = label
        row["analysis_label_ids"] = label
        row["source_label_ids"] = label
        rows.append(row)
if "species_code" not in fields:
    fields.append("species_code")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
counts = Counter((row.get("split", ""), row.get("species", "")) for row in rows)
summary = {
    "source_manifest": str(src),
    "species_pair": [species_a, species_b],
    "rows": len(rows),
    "split_species_counts": {f"{split}|{species}": count for (split, species), count in sorted(counts.items())},
}
out_counts.write_text(json.dumps(summary, indent=2), encoding="utf-8")
labels = []
for species in (species_a, species_b):
    name, hierarchy = display.get(species, (species, f"Biophony > Marine mammal > Cetacean > {species}"))
    labels.append(
        {
            "id": f"species:{species}",
            "group": "species",
            "code": species,
            "name": name,
            "class_hierarchy": hierarchy,
            "count": sum(1 for row in rows if row.get("species") == species),
        }
    )
out_vocab.write_text(json.dumps({"schema_version": "multilabel-v1", "labels": labels}, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PYMAKE

  JOB_SCRIPT="$LOG_DIR/E120pair${PAIR_TAG}.sbatch"
  cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=E120${PAIR_TAG}
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM
#SBATCH --gres=$SBATCH_GRES

set -euo pipefail
ROOT="$WEEKEND_ROOT"
RUN="$RUN_DIR"
REPO="$REPO_ON_NIBI"
DATA="$DATASET_ROOT"
SCRIPT="$TRAIN_SCRIPT"
FINCKPT="$FIN_INIT_CKPT"
MAN="$MANIFEST"
VOC="$VOCAB"
TARGET_EPOCHS="$EPOCHS"

cd "\$REPO"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"
SUMMARY="\$RUN/train/run_summary.json"
if "$PYTHON_BIN" - "\$SUMMARY" "\$TARGET_EPOCHS" <<'PYCHK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
target = int(sys.argv[2])
if not p.exists():
    raise SystemExit(1)
history = json.loads(p.read_text()).get("history") or []
raise SystemExit(0 if max((int(row.get("epoch", 0)) for row in history), default=0) >= target else 1)
PYCHK
then
  echo "E120 ${PAIR_TAG} complete; skipping"
  exit 0
fi
EXTRA=()
if [[ -f "\$RUN/train/last.pt" ]]; then
  EXTRA=(--resume-checkpoint "\$RUN/train/last.pt")
else
  EXTRA=(--init-all-branches-checkpoint "\$FINCKPT")
fi
"$PYTHON_BIN" -u "\$SCRIPT" \\
  --manifest-csv "\$MAN" \\
  --vocab-json "\$VOC" \\
  --dataset-root "\$DATA" \\
  --exp-dir "\$RUN/train" \\
  --bands low,mid,high \\
  --band-crop-shapes low:391x50,mid:256x100,high:256x312 \\
  --encoder resnet18 \\
  --fusion gated \\
  --dropout "$DROPOUT" \\
  --epochs "$EPOCHS" \\
  --batch-size "$BATCH_SIZE" \\
  --num-workers "$NUM_WORKERS" \\
  --weight-decay 0.0001 \\
  --sampler none \\
  --crop-time-seconds 10 \\
  --context-seconds 40 \\
  --center-bias-sigma-frac 0.25 \\
  --positive-crop-mode centered_gaussian \\
  --band-availability-mode all \\
  --device cuda \\
  --seed 2026 \\
  --stop-after-seconds "$STOP_AFTER_SECONDS" \\
  --lr "$LR" \\
  --loss-mode ce \\
  --class-weight-mode none \\
  "\${EXTRA[@]}"
EOF

  echo "Prepared $pair -> $RUN_DIR"
  initial_job=""
  continuation_job=""
  if [[ "$DRY_RUN" == "false" ]]; then
    submit_args=()
    if [[ -n "$DEPENDENCY" ]]; then
      submit_args+=(--dependency="$DEPENDENCY")
    fi
    initial_job="$(sbatch --parsable "${submit_args[@]}" "$JOB_SCRIPT")"
    continuation_job="$(sbatch --parsable --dependency=afterany:"$initial_job" "$JOB_SCRIPT")"
    echo "Submitted $pair initial=$initial_job continuation=$continuation_job"
  fi
  echo -e "${pair}\t${initial_job}\t${continuation_job}\t${RUN_DIR}\t${MANIFEST}\t${VOCAB}\t${JOB_SCRIPT}" >> "$SUBMITTED_TSV"
done

echo "Submitted/created pairwise specialist plan: $SUBMITTED_TSV"
