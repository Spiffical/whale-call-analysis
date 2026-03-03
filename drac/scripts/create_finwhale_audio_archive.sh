#!/bin/bash
# Create an archive of a prepared Perch2 context dataset directory for DRAC jobs.
#
# Recommended input is the output directory of:
#   scripts/data/train/create_perch2_context_dataset.py
#
# Example:
#   bash drac/scripts/create_finwhale_audio_archive.sh \
#     --dataset-dir /path/to/perch2_context_dataset_YYYYMMDDTHHMMSSZ \
#     --output-path "$PROJECT/whale-call-analysis/data/archives/perch2_context_dataset_YYYYMMDDTHHMMSSZ.tar.zst"

set -euo pipefail

DATASET_DIR=""
OUTPUT_PATH=""
FORMAT="tar.zst"   # tar | tar.gz | tar.zst
ZSTD_LEVEL="3"
GZIP_LEVEL="3"
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}"
OVERWRITE="false"

expand_home_path() {
  local p="$1"
  if [[ "$p" == "~/"* ]]; then
    printf '%s\n' "$HOME/${p#\~/}"
    return
  fi
  printf '%s\n' "$p"
}

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/create_finwhale_audio_archive.sh [options]

Required:
  --dataset-dir PATH
  --output-path PATH

Options:
  --audio-dir PATH        Deprecated alias for --dataset-dir
  --format NAME          tar | tar.gz | tar.zst (default: tar.zst)
  --zstd-level N         zstd compression level (default: 3)
  --gzip-level N         gzip compression level (default: 3)
  --threads N            compression threads (default: nproc)
  --overwrite            overwrite output if it exists
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --audio-dir) DATASET_DIR="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --zstd-level) ZSTD_LEVEL="$2"; shift 2 ;;
    --gzip-level) GZIP_LEVEL="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --overwrite) OVERWRITE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

DATASET_DIR="$(expand_home_path "$DATASET_DIR")"
OUTPUT_PATH="$(expand_home_path "$OUTPUT_PATH")"

if [[ -z "$DATASET_DIR" || -z "$OUTPUT_PATH" ]]; then
  echo "Error: --dataset-dir and --output-path are required"
  usage
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Error: dataset directory does not exist: $DATASET_DIR"
  exit 1
fi
case "$FORMAT" in
  tar|tar.gz|tar.zst) ;;
  *) echo "Error: unsupported --format '$FORMAT'"; exit 1 ;;
esac
if [[ -f "$OUTPUT_PATH" && "$OVERWRITE" != "true" ]]; then
  echo "Error: output already exists: $OUTPUT_PATH (use --overwrite)"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Creating archive..."
echo "  source: $DATASET_DIR"
echo "  output: $OUTPUT_PATH"
echo "  format: $FORMAT"
echo "  threads: $THREADS"

tmp_out="$OUTPUT_PATH.tmp"
rm -f "$tmp_out"

if [[ "$FORMAT" == "tar" ]]; then
  tar -cf "$tmp_out" -C "$DATASET_DIR" .
elif [[ "$FORMAT" == "tar.gz" ]]; then
  if command -v pigz >/dev/null 2>&1; then
    tar -I "pigz -p $THREADS -$GZIP_LEVEL" -cf "$tmp_out" -C "$DATASET_DIR" .
  else
    tar -czf "$tmp_out" -C "$DATASET_DIR" .
  fi
elif [[ "$FORMAT" == "tar.zst" ]]; then
  if command -v zstd >/dev/null 2>&1; then
    tar -I "zstd -T$THREADS -$ZSTD_LEVEL" -cf "$tmp_out" -C "$DATASET_DIR" .
  else
    echo "Error: zstd is required for --format tar.zst"
    exit 1
  fi
fi

mv "$tmp_out" "$OUTPUT_PATH"

if command -v du >/dev/null 2>&1; then
  echo "Archive size: $(du -h "$OUTPUT_PATH" | awk '{print $1}')"
fi
echo "Done: $OUTPUT_PATH"
echo ""
echo "Use in DRAC submit script:"
echo "  --context-dataset-tar $OUTPUT_PATH"
