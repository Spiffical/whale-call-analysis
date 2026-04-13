#!/bin/bash
# Submit the held-out Stage B attention evaluation on the 2025 Part 2 split.

set -euo pipefail

SPLIT_NAME="${SPLIT_NAME:-part2_eval_test}"
METHODS="${METHODS:-gradcampp,layercam}"
PROJECT_PATH="${PROJECT_PATH:-$PWD}"
VENV_PATH="${VENV_PATH:-$HOME/whale-call-analysis/.venv}"

BUNDLE_TAR="${BUNDLE_TAR:-/project/rpp-kmoran/merileo/data/finwhales/finwhale_part2_bundle.tar}"
SPLIT_ROOT="${SPLIT_ROOT:-/project/rpp-kmoran/merileo/data/finwhales/finwhale_part2_learning_curve_20260331_coherent_negfix_v1}"
OUT_DIR="${OUT_DIR:-/scratch/merileo/finwhale_attention_experiment}"
ARCHIVE="${ARCHIVE:-/project/rpp-kmoran/merileo/data/finwhales/analysis_archive/20260409/part2_curve_20260331_coherent_negfix_v1_slim_20260409.tar.zst}"

BASELINE_CKPT="${BASELINE_CKPT:-/scratch/merileo/finwhale_sweeps/balanced_ov09_vs_ov095_20260209T204522Z/runs/ov09/r015_dov09_resnet18_lr3em4_balnone_cbs0p25_gap120_s1337/finwhale/finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-seed1337-mf1-r015_dov09_resnet18_lr3em4_balnone_cbs0p25_gap120_s1337/best.pt}"
BALANCED_SPEC="balanced=${ARCHIVE}::part2_curve_20260331_coherent_negfix_v1_slim/month_stratified_clip_calls01000_rep01/train/best.pt"
HIGHPERF_SPEC="highperf=${ARCHIVE}::part2_curve_20260331_coherent_negfix_v1_slim/month_stratified_clip_calls05000_rep02/train/best.pt"

sbatch \
  --export=ALL,PROJECT_PATH="$PROJECT_PATH",VENV_PATH="$VENV_PATH" \
  drac/scripts/submit_finwhale_attention_experiment.sh \
  --bundle-tar "$BUNDLE_TAR" \
  --out-dir "$OUT_DIR" \
  --mode quant \
  --methods "$METHODS" \
  --split-dir "$SPLIT_ROOT/$SPLIT_NAME" \
  --checkpoint-spec "baseline=${BASELINE_CKPT}" \
  --checkpoint-spec "$BALANCED_SPEC" \
  --checkpoint-spec "$HIGHPERF_SPEC" \
  --quant-max-positive 0
