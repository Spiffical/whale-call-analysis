#!/bin/bash
# Submit the E120/E121/E122/E124 candidate-improvement suite under one stamp.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
SOURCE_MANIFEST="$WEEKEND_ROOT/manifests/e100_onc_only_blocked_nov_validation_20260611T020900Z/E101_stage2_ONConly_blocked_nov20_25_30_val/standardized_manifest.csv"
PYTHON_BIN="python3"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUITE_DIR=""
DRY_RUN="false"
SKIP_E120="false"
SKIP_E121="false"
SKIP_E122="false"
SKIP_E124="false"
INCLUDE_E123="false"
SSL_REPO_ROOT="$WEEKEND_ROOT/selfsupervision_anomalies_onc"
VARIANT_TAG="ONConly"
BASE_RUN_DIRS=()
BASE_RUN_GLOBS=()
PAIRWISE_RUN_DIRS=()
PAIRWISE_RUN_GLOBS=()
E120_PAIRS=()
SOURCE_KINDS=()
EXTRA_E124_SUMMARY_GLOBS=()
EXTRA_E124_SUMMARY_JSONS=()
EXTRA_E124_SUMMARY_CSVS=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e125_candidate_suite.sh --base-run-dir PATH [options]

Submit the main candidate-improvement suite with <=3h GPU jobs:
  E120: Bm:Bp and Bm:Mn pairwise specialists, resumable MIG training
  E121: multi-pairwise production-style refinement report
  E122: binary whale-call gate + two-stage production-style report
  E124: leaderboard across generated and existing candidate reports

Options:
  --base-run-dir PATH       Multiclass base run; may be repeated
  --base-run-glob GLOB      Glob for multiclass base runs; may be repeated
  --pairwise-run-dir PATH   Existing pairwise run to include in E121; may be repeated
  --pairwise-run-glob GLOB  Existing pairwise run glob to include in E121; may be repeated
  --pair A:B                E120 pair to train; may be repeated. Default: Bm:Bp and Bm:Mn
  --source-kind KIND        Keep only source kind for E120/E122; may be repeated.
                            Omit to use every source in --source-manifest.
  --variant-tag TAG         Path/report tag for data variant. Default: ONConly
  --weekend-root PATH       Default: /scratch/.../multispecies_weekend_20260502
  --repo-root PATH          Default: $weekend_root/repo_e24_expert_hparam_68be99f
  --source-manifest PATH    Standardized manifest for pairwise/gate experiments
  --python-bin NAME         Default: python3
  --suite-dir PATH          Default: $weekend_root/pipeline_runs/e125_candidate_suite_$stamp
  --stamp STAMP             Default: current UTC stamp
  --skip-e120               Do not train new pairwise specialists
  --skip-e121               Do not run multi-pairwise report
  --skip-e122               Do not train/evaluate two-stage gate
  --skip-e124               Do not build candidate leaderboard
  --include-e123            Also submit optional SSAMBA SSL launcher using --source-manifest
  --ssl-repo-root PATH      Default: $weekend_root/selfsupervision_anomalies_onc
  --e124-summary-json PATH  Extra summary JSON for leaderboard; may be repeated
  --e124-summary-csv PATH   Extra E27/E28 ensemble rankings CSV; may be repeated
  --e124-summary-glob GLOB  Extra summary glob for leaderboard; may be repeated
  --dry-run                 Create scripts/plans but do not submit jobs
USAGE
}

join_by_colon() {
  local IFS=:
  echo "$*"
}

dependency_afterany() {
  local ids=()
  local id
  for id in "$@"; do
    if [[ -n "$id" ]]; then
      ids+=("$id")
    fi
  done
  if [[ "${#ids[@]}" -gt 0 ]]; then
    echo "afterany:$(join_by_colon "${ids[@]}")"
  fi
}

append_plan() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$PLAN_TSV"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-run-dir) BASE_RUN_DIRS+=("$2"); shift 2 ;;
    --base-run-glob) BASE_RUN_GLOBS+=("$2"); shift 2 ;;
    --pairwise-run-dir) PAIRWISE_RUN_DIRS+=("$2"); shift 2 ;;
    --pairwise-run-glob) PAIRWISE_RUN_GLOBS+=("$2"); shift 2 ;;
    --pair) E120_PAIRS+=("$2"); shift 2 ;;
    --source-kind) SOURCE_KINDS+=("$2"); shift 2 ;;
    --variant-tag) VARIANT_TAG="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --source-manifest) SOURCE_MANIFEST="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --skip-e120) SKIP_E120="true"; shift ;;
    --skip-e121) SKIP_E121="true"; shift ;;
    --skip-e122) SKIP_E122="true"; shift ;;
    --skip-e124) SKIP_E124="true"; shift ;;
    --include-e123) INCLUDE_E123="true"; shift ;;
    --ssl-repo-root) SSL_REPO_ROOT="$2"; shift 2 ;;
    --e124-summary-json) EXTRA_E124_SUMMARY_JSONS+=("$2"); shift 2 ;;
    --e124-summary-csv) EXTRA_E124_SUMMARY_CSVS+=("$2"); shift 2 ;;
    --e124-summary-glob) EXTRA_E124_SUMMARY_GLOBS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SUITE_DIR" ]]; then
  SAFE_VARIANT_TAG="$(printf '%s' "$VARIANT_TAG" | tr -c 'A-Za-z0-9_.-' '_')"
  SUITE_DIR="$WEEKEND_ROOT/pipeline_runs/e125_candidate_suite_${SAFE_VARIANT_TAG}_${STAMP}"
else
  SAFE_VARIANT_TAG="$(printf '%s' "$VARIANT_TAG" | tr -c 'A-Za-z0-9_.-' '_')"
fi
LOG_DIR="$SUITE_DIR/logs"
mkdir -p "$LOG_DIR"
PLAN_TSV="$SUITE_DIR/e125_suite_plan.tsv"
echo -e "component\tstatus\tjob_ids\tpath" > "$PLAN_TSV"

for base_run_glob in "${BASE_RUN_GLOBS[@]}"; do
  mapfile -t matches < <(compgen -G "$base_run_glob" | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No base runs matched glob: $base_run_glob" >&2
    exit 1
  fi
  BASE_RUN_DIRS+=("${matches[@]}")
done
for pairwise_run_glob in "${PAIRWISE_RUN_GLOBS[@]}"; do
  mapfile -t matches < <(compgen -G "$pairwise_run_glob" | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No pairwise runs matched glob: $pairwise_run_glob" >&2
    exit 1
  fi
  PAIRWISE_RUN_DIRS+=("${matches[@]}")
done

if [[ "${#E120_PAIRS[@]}" -eq 0 ]]; then
  E120_PAIRS=("Bm:Bp" "Bm:Mn")
fi

if [[ ! -d "$REPO_ON_NIBI" ]]; then
  echo "Missing repo root: $REPO_ON_NIBI" >&2
  exit 1
fi
if [[ "${#BASE_RUN_DIRS[@]}" -eq 0 && ( "$SKIP_E121" == "false" || "$SKIP_E122" == "false" ) ]]; then
  echo "Provide at least one --base-run-dir or --base-run-glob for E121/E122 reports" >&2
  usage >&2
  exit 1
fi

cd "$REPO_ON_NIBI"
echo "Suite dir: $SUITE_DIR"
echo "Plan: $PLAN_TSV"
echo "Repo: $REPO_ON_NIBI"
echo "Variant tag: $VARIANT_TAG"
if [[ "${#SOURCE_KINDS[@]}" -gt 0 ]]; then
  echo "Source kinds: ${SOURCE_KINDS[*]}"
else
  echo "Source kinds: all sources in manifest"
fi
git rev-parse HEAD || true

dry_arg=()
if [[ "$DRY_RUN" == "true" ]]; then
  dry_arg=(--dry-run)
fi

E120_DONE_JOBS=()
if [[ "$SKIP_E120" == "false" ]]; then
  E120_MANIFEST_ROOT="$WEEKEND_ROOT/manifests/e120_pairwise_specialists_${STAMP}"
  E120_LOG="$LOG_DIR/e120_submit.log"
  e120_cmd=(bash drac/scripts/submit_multispecies_e120_pairwise_specialists.sh
    --weekend-root "$WEEKEND_ROOT"
    --repo-root "$REPO_ON_NIBI"
    --source-manifest "$SOURCE_MANIFEST"
    --manifest-root "$E120_MANIFEST_ROOT"
    --stamp "$STAMP"
    --python-bin "$PYTHON_BIN"
    --variant-tag "$VARIANT_TAG")
  for source_kind in "${SOURCE_KINDS[@]}"; do
    e120_cmd+=(--source-kind "$source_kind")
  done
  for pair in "${E120_PAIRS[@]}"; do
    e120_cmd+=(--pair "$pair")
  done
  e120_cmd+=("${dry_arg[@]}")
  "${e120_cmd[@]}" | tee "$E120_LOG"
  E120_TSV="$E120_MANIFEST_ROOT/e120_pairwise_submitted.tsv"
  if [[ -f "$E120_TSV" ]]; then
    mapfile -t new_pairwise_dirs < <(awk -F '\t' 'NR > 1 && $4 != "" {print $4}' "$E120_TSV")
    PAIRWISE_RUN_DIRS+=("${new_pairwise_dirs[@]}")
    mapfile -t E120_DONE_JOBS < <(awk -F '\t' 'NR > 1 && $3 != "" {print $3}' "$E120_TSV")
  fi
  append_plan "E120" "submitted_or_prepared" "$(join_by_colon "${E120_DONE_JOBS[@]}")" "$E120_TSV"
fi

E121_JOB=""
E121_OUTPUT="$SUITE_DIR/e121_multi_pairwise_refinement"
if [[ "$SKIP_E121" == "false" ]]; then
  if [[ "${#PAIRWISE_RUN_DIRS[@]}" -eq 0 ]]; then
    echo "No pairwise run dirs available for E121; train E120 or pass --pairwise-run-dir" >&2
    exit 1
  fi
  e121_cmd=(bash drac/scripts/submit_multispecies_e121_multi_pairwise_refinement_report.sh
    --weekend-root "$WEEKEND_ROOT"
    --repo-root "$REPO_ON_NIBI"
    --output-dir "$E121_OUTPUT"
    --stamp "$STAMP")
  for base_run_dir in "${BASE_RUN_DIRS[@]}"; do
    e121_cmd+=(--base-run-dir "$base_run_dir")
  done
  for pairwise_run_dir in "${PAIRWISE_RUN_DIRS[@]}"; do
    e121_cmd+=(--pairwise-run-dir "$pairwise_run_dir")
  done
  e120_dep="$(dependency_afterany "${E120_DONE_JOBS[@]}")"
  if [[ -n "$e120_dep" ]]; then
    e121_cmd+=(--dependency "$e120_dep")
  fi
  e121_cmd+=("${dry_arg[@]}")
  E121_LOG="$LOG_DIR/e121_submit.log"
  "${e121_cmd[@]}" | tee "$E121_LOG"
  E121_JOB="$(awk '/Submitted E121 report job:/ {print $NF}' "$E121_LOG" | tail -n 1)"
  append_plan "E121" "submitted_or_prepared" "$E121_JOB" "$E121_OUTPUT"
fi

E122_REPORT_JOB=""
E122_REPORT_DIR="$SUITE_DIR/e122_two_stage_gate"
if [[ "$SKIP_E122" == "false" ]]; then
  E122_MANIFEST_ROOT="$WEEKEND_ROOT/manifests/e122_two_stage_gate_${STAMP}"
  E122_RUN_DIR="$WEEKEND_ROOT/runs/E122_whale_call_gate_ONConly_3band_lr3e4_${STAMP}"
  e122_cmd=(bash drac/scripts/submit_multispecies_e122_two_stage_gate.sh
    --weekend-root "$WEEKEND_ROOT"
    --repo-root "$REPO_ON_NIBI"
    --source-manifest "$SOURCE_MANIFEST"
    --manifest-root "$E122_MANIFEST_ROOT"
    --run-dir "$E122_RUN_DIR"
    --report-dir "$E122_REPORT_DIR"
    --stamp "$STAMP"
    --python-bin "$PYTHON_BIN"
    --variant-tag "$VARIANT_TAG")
  for source_kind in "${SOURCE_KINDS[@]}"; do
    e122_cmd+=(--source-kind "$source_kind")
  done
  for base_run_dir in "${BASE_RUN_DIRS[@]}"; do
    e122_cmd+=(--base-run-dir "$base_run_dir")
  done
  e122_cmd+=("${dry_arg[@]}")
  E122_LOG="$LOG_DIR/e122_submit.log"
  "${e122_cmd[@]}" | tee "$E122_LOG"
  E122_TSV="$E122_MANIFEST_ROOT/e122_two_stage_submitted.tsv"
  if [[ -f "$E122_TSV" ]]; then
    E122_REPORT_JOB="$(awk -F '\t' 'NR == 2 {print $3}' "$E122_TSV")"
  fi
  append_plan "E122" "submitted_or_prepared" "$E122_REPORT_JOB" "$E122_REPORT_DIR"
fi

if [[ "$INCLUDE_E123" == "true" ]]; then
  E123_LOG="$LOG_DIR/e123_submit.log"
  E123_RUN_ROOT="$WEEKEND_ROOT/runs/E123_ssl_ssamba_multispecies_${STAMP}"
  e123_cmd=(bash drac/scripts/submit_multispecies_e123_ssl_ssamba.sh
    --weekend-root "$WEEKEND_ROOT"
    --repo-root "$REPO_ON_NIBI"
    --ssl-repo-root "$SSL_REPO_ROOT"
    --manifest-csv "$SOURCE_MANIFEST"
    --run-root "$E123_RUN_ROOT"
    --stamp "$STAMP"
    --python-bin python)
  e123_cmd+=("${dry_arg[@]}")
  "${e123_cmd[@]}" | tee "$E123_LOG"
  append_plan "E123" "submitted_or_prepared" "" "$E123_RUN_ROOT"
fi

if [[ "$SKIP_E124" == "false" ]]; then
  E124_OUTPUT="$SUITE_DIR/e124_candidate_leaderboard"
  e124_cmd=(bash drac/scripts/submit_multispecies_e124_candidate_leaderboard.sh
    --weekend-root "$WEEKEND_ROOT"
    --repo-root "$REPO_ON_NIBI"
    --output-dir "$E124_OUTPUT"
    --stamp "$STAMP"
    --summary-glob "$WEEKEND_ROOT/pipeline_runs/e119*/**/e119_summary.json"
    --summary-glob "$WEEKEND_ROOT/pipeline_runs/e26*/**/diagnostic_summary.json"
    --summary-glob "$WEEKEND_ROOT/pipeline_runs/e27*/**/e27_ensemble_rankings.csv"
    --summary-glob "$WEEKEND_ROOT/pipeline_runs/e28*/**/e28_ensemble_rankings.csv")
  if [[ "$SKIP_E121" == "false" ]]; then
    e124_cmd+=(--summary-json "$E121_OUTPUT/e121_summary.json")
  fi
  if [[ "$SKIP_E122" == "false" ]]; then
    e124_cmd+=(--summary-json "$E122_REPORT_DIR/e122_summary.json")
  fi
  for path in "${EXTRA_E124_SUMMARY_JSONS[@]}"; do
    e124_cmd+=(--summary-json "$path")
  done
  for path in "${EXTRA_E124_SUMMARY_CSVS[@]}"; do
    e124_cmd+=(--summary-csv "$path")
  done
  for pattern in "${EXTRA_E124_SUMMARY_GLOBS[@]}"; do
    e124_cmd+=(--summary-glob "$pattern")
  done
  e124_dep="$(dependency_afterany "$E121_JOB" "$E122_REPORT_JOB")"
  if [[ -n "$e124_dep" ]]; then
    e124_cmd+=(--dependency "$e124_dep")
  fi
  e124_cmd+=("${dry_arg[@]}")
  E124_LOG="$LOG_DIR/e124_submit.log"
  "${e124_cmd[@]}" | tee "$E124_LOG"
  E124_JOB="$(awk '/Submitted E124 leaderboard job:/ {print $NF}' "$E124_LOG" | tail -n 1)"
  append_plan "E124" "submitted_or_prepared" "$E124_JOB" "$E124_OUTPUT"
fi

echo "E125 suite plan written: $PLAN_TSV"
