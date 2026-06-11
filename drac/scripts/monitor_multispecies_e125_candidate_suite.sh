#!/bin/bash
# Summarize E125 suite jobs, logs, and report artifacts.

set -euo pipefail

SUITE_DIR=""
TAIL_LINES="80"
SHOW_DISK="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/monitor_multispecies_e125_candidate_suite.sh --suite-dir PATH [options]

Summarize an E125 candidate suite after launch:
  - suite plan and collected Slurm job IDs
  - squeue/sacct state when available
  - recent submit/slurm logs
  - E121/E122 summaries and E124 leaderboard artifacts when complete

Options:
  --suite-dir PATH   E125 suite directory
  --tail N           Lines per log tail. Default: 80
  --show-disk        Also run diskusage_report and df -ih for project/scratch
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --tail) TAIL_LINES="$2"; shift 2 ;;
    --show-disk) SHOW_DISK="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SUITE_DIR" ]]; then
  echo "Missing --suite-dir" >&2
  usage >&2
  exit 1
fi

PLAN_TSV="$SUITE_DIR/e125_suite_plan.tsv"
if [[ ! -f "$PLAN_TSV" ]]; then
  echo "Missing suite plan: $PLAN_TSV" >&2
  exit 1
fi

echo "Suite dir: $SUITE_DIR"
echo "Plan: $PLAN_TSV"
echo
echo "== Plan =="
column -t -s $'\t' "$PLAN_TSV" 2>/dev/null || cat "$PLAN_TSV"

collect_ids() {
  awk -F '\t' '
    NR > 1 {
      n = split($3, parts, /[: ,]+/)
      for (i = 1; i <= n; i++) {
        if (parts[i] ~ /^[0-9]+$/) print parts[i]
      }
    }
  ' "$PLAN_TSV"
  find "$SUITE_DIR" -path '*/e120_pairwise_submitted.tsv' -o -path '*/e122_two_stage_submitted.tsv' 2>/dev/null |
    while read -r tsv; do
      awk -F '\t' 'NR > 1 {for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) print $i}' "$tsv"
    done
  find "$SUITE_DIR/logs" -maxdepth 1 -type f -name '*_submit.log' 2>/dev/null |
    xargs -r awk '
      /Submitted .* job:/ {print $NF}
      /Submitted .* initial=/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^initial=[0-9]+$/) {sub(/^initial=/, "", $i); print $i}
          if ($i ~ /^continuation=[0-9]+$/) {sub(/^continuation=/, "", $i); print $i}
        }
      }
      /Submitted E122 gate initial=/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^initial=[0-9]+$/) {sub(/^initial=/, "", $i); print $i}
          if ($i ~ /^continuation=[0-9]+$/) {sub(/^continuation=/, "", $i); print $i}
        }
      }
    '
}

mapfile -t JOB_IDS < <(collect_ids | awk '!seen[$0]++' | sort -n)
if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
  JOB_LIST="$(IFS=,; echo "${JOB_IDS[*]}")"
  echo
  echo "== Slurm Jobs =="
  echo "$JOB_LIST"
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "$JOB_LIST" -o '%.18i %.9T %.12M %.12L %.35j %.25R' || true
  fi
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$JOB_LIST" --format=JobID,JobName%36,State,ExitCode,Elapsed,NodeList%30 -P || true
  fi
else
  echo
  echo "== Slurm Jobs =="
  echo "No numeric job IDs found yet."
fi

if [[ "$SHOW_DISK" == "true" ]]; then
  echo
  echo "== Disk / Inodes =="
  timeout 180 diskusage_report || true
  df -ih /project/def-kmoran /scratch || true
fi

echo
echo "== Submit Logs =="
find "$SUITE_DIR/logs" -maxdepth 1 -type f -name '*_submit.log' -print 2>/dev/null | sort |
  while read -r log; do
    echo "--- $log ---"
    tail -n "$TAIL_LINES" "$log" || true
  done

echo
echo "== Slurm Logs =="
find "$SUITE_DIR" -path '*/logs/slurm-*.out' -type f -print 2>/dev/null | sort |
  while read -r log; do
    echo "--- $log ---"
    tail -n "$TAIL_LINES" "$log" || true
  done

echo
echo "== Report Artifacts =="
for path in \
  "$SUITE_DIR/e121_multi_pairwise_refinement/e121_summary.json" \
  "$SUITE_DIR/e121_multi_pairwise_refinement/e121_multi_pairwise_refinement_report.md" \
  "$SUITE_DIR/e122_two_stage_gate/e122_summary.json" \
  "$SUITE_DIR/e122_two_stage_gate/e122_two_stage_gate_report.md" \
  "$SUITE_DIR/e124_candidate_leaderboard/e124_candidate_leaderboard.csv" \
  "$SUITE_DIR/e124_candidate_leaderboard/e124_candidate_leaderboard.md" \
  "$SUITE_DIR/e124_candidate_leaderboard/e124_candidate_leaderboard.json"; do
  if [[ -f "$path" ]]; then
    printf 'FOUND\t%s\n' "$path"
  else
    printf 'MISSING\t%s\n' "$path"
  fi
done

LEADERBOARD="$SUITE_DIR/e124_candidate_leaderboard/e124_candidate_leaderboard.csv"
if [[ -f "$LEADERBOARD" ]]; then
  echo
  echo "== Leaderboard Preview =="
  head -n 12 "$LEADERBOARD"
fi
