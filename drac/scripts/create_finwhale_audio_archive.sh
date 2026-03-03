#!/bin/bash
# Create an archive of FinWhale audio files for DRAC jobs.
#
# The archive layout is normalized so .wav files extract directly into one folder,
# which matches submit_finwhale_perch2_embeddings.sh expectations.
#
# Example:
#   bash drac/scripts/create_finwhale_audio_archive.sh \
#     --audio-dir /mnt/z/FinWhalesProject/data/audio \
#     --output-path "$PROJECT/whale-call-analysis/data/archives/finwhale_audio_20260302.tar.zst"

set -euo pipefail

AUDIO_DIR=""
OUTPUT_PATH=""
FORMAT="tar.zst"   # tar | tar.gz | tar.zst
ZSTD_LEVEL="3"
GZIP_LEVEL="3"
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}"
OVERWRITE="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/create_finwhale_audio_archive.sh [options]

Required:
  --audio-dir PATH
  --output-path PATH

Options:
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
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
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

if [[ -z "$AUDIO_DIR" || -z "$OUTPUT_PATH" ]]; then
  echo "Error: --audio-dir and --output-path are required"
  usage
  exit 1
fi
if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "Error: audio directory does not exist: $AUDIO_DIR"
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
echo "  source: $AUDIO_DIR"
echo "  output: $OUTPUT_PATH"
echo "  format: $FORMAT"
echo "  threads: $THREADS"

tmp_out="$OUTPUT_PATH.tmp"
rm -f "$tmp_out"

if [[ "$FORMAT" == "tar" ]]; then
  tar -cf "$tmp_out" -C "$AUDIO_DIR" .
elif [[ "$FORMAT" == "tar.gz" ]]; then
  if command -v pigz >/dev/null 2>&1; then
    tar -I "pigz -p $THREADS -$GZIP_LEVEL" -cf "$tmp_out" -C "$AUDIO_DIR" .
  else
    tar -czf "$tmp_out" -C "$AUDIO_DIR" .
  fi
elif [[ "$FORMAT" == "tar.zst" ]]; then
  if command -v zstd >/dev/null 2>&1; then
    tar -I "zstd -T$THREADS -$ZSTD_LEVEL" -cf "$tmp_out" -C "$AUDIO_DIR" .
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
echo "  --audio-tar-path $OUTPUT_PATH"
