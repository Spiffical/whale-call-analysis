#!/bin/bash
# Submit E127 synthetic-H5 variants and chain each into E123 SSAMBA training.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
SSL_REPO_ROOT="$WEEKEND_ROOT/selfsupervision_anomalies_onc"
BASE_H5=""
OUTPUT_ROOT=""
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="python3"
DEPENDENCY=""
DRY_RUN="false"
ALLOW_MISSING_BASE_H5="false"

AUGMENT_TIME="03:00:00"
AUGMENT_CPUS="4"
AUGMENT_MEM="48G"
SYNTHETIC_PER_TARGET="1000"
SYNTHETIC_SPLIT="train"
SEED="1337"
SNR_DB_MIN="-10.0"
SNR_DB_MAX="10.0"
FREQ_SHIFT_MIN_BINS="-12"
FREQ_SHIFT_MAX_BINS="12"
TIME_SHIFT_MIN_BINS="0"
TIME_SHIFT_MAX_BINS="0"
TIME_STRETCH_MIN="0.97"
TIME_STRETCH_MAX="1.03"
NONLINEAR_DISTORTION_STRENGTH_MIN="0.0"
NONLINEAR_DISTORTION_STRENGTH_MAX="0.0"
SPECTRAL_FILTER_STRENGTH_MIN="0.0"
SPECTRAL_FILTER_STRENGTH_MAX="0.0"
TRANSMISSION_LOSS_MIN="0.10"
TRANSMISSION_LOSS_MAX="0.75"
REVERB_SMEAR_STRENGTH_MIN="0.0"
REVERB_SMEAR_STRENGTH_MAX="0.0"
REVERB_SMEAR_DECAY_MIN_BINS="2"
REVERB_SMEAR_DECAY_MAX_BINS="12"
END_TRIM_FRACTION_MIN="0.0"
END_TRIM_FRACTION_MAX="0.0"
GAUSSIAN_NOISE_STD="0.01"

NUM_PRETRAIN_JOBS="2"
NUM_FINETUNE_JOBS="1"
SBATCH_TIME="03:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="48G"
SBATCH_GRES="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
RUNNER_PY=""
VENV_PATH=""
WANDB_PROJECT="multispecies_e127_synthetic_ssl"
WANDB_GROUP_PREFIX="E127_synthetic_ssl"
TRAIN_RATIO="0.8"
PRETRAIN_TASK="pretrain_joint"
FINETUNE_TASK="ft_avgtok"

VARIANTS=("baseline=none" "bm=Bm" "mn=Mn" "bm_mn=Bm,Mn")
VARIANTS_CUSTOM="false"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e127_synthetic_ssl_suite.sh --base-h5 PATH [options]

Build GAVDNet-inspired synthetic H5 variants from an existing E123/E126 H5
dataset, then submit each variant to the E123 SSAMBA pretrain + fine-tune
launcher. Defaults use <=3h jobs and a MIG GPU allocation for training.

Required:
  --base-h5 PATH              Existing E123/E126 SSAMBA-compatible H5 dataset

Options:
  --variant NAME=LABELS       Variant to run; may be repeated. LABELS may be
                              comma-separated (Bm,Mn) or 'none' for baseline.
                              First use replaces defaults.
  --synthetic-per-target N    Default: 1000
  --synthetic-split SPLIT     Default: train
  --weekend-root PATH         Default: /scratch/.../multispecies_weekend_20260502
  --repo-root PATH            Default: $weekend_root/repo_e24_expert_hparam_68be99f
  --ssl-repo-root PATH        Default: $weekend_root/selfsupervision_anomalies_onc
  --output-root PATH          Default: $weekend_root/runs/E127_synthetic_ssl_suite_$stamp
  --stamp STAMP              Default: current UTC stamp
  --python-bin NAME           Default: python3
  --dependency SPEC           Dependency for first jobs, e.g. afterany:123
  --allow-missing-base-h5     Allow --base-h5 to be created/validated by an
                              upstream dependency job. Requires --dependency
                              unless the file already exists.
  --augment-time HH:MM:SS     Default: 03:00:00
  --augment-cpus-per-task N   Default: 4
  --augment-mem MEM           Default: 48G
  --seed N                    Base seed. Variant index is added. Default: 1337
  --snr-db-min X              Default: -10.0
  --snr-db-max X              Default: 10.0
  --freq-shift-min-bins N     Default: -12
  --freq-shift-max-bins N     Default: 12
  --time-shift-min-bins N     Default: 0
  --time-shift-max-bins N     Default: 0
  --time-stretch-min X        Default: 0.97
  --time-stretch-max X        Default: 1.03
  --nonlinear-distortion-strength-min X
                              Default: 0.0
  --nonlinear-distortion-strength-max X
                              Default: 0.0
  --spectral-filter-strength-min X
                              Default: 0.0
  --spectral-filter-strength-max X
                              Default: 0.0
  --transmission-loss-min X   Default: 0.10
  --transmission-loss-max X   Default: 0.75
  --reverb-smear-strength-min X
                              Default: 0.0
  --reverb-smear-strength-max X
                              Default: 0.0
  --reverb-smear-decay-min-bins N
                              Default: 2
  --reverb-smear-decay-max-bins N
                              Default: 12
  --end-trim-fraction-min X   Default: 0.0
  --end-trim-fraction-max X   Default: 0.0
  --gaussian-noise-std X      Default: 0.01
  --num-pretrain-jobs N       Default: 2
  --num-finetune-jobs N       Default: 1
  --time HH:MM:SS             Training job time. Default: 03:00:00
  --cpus-per-task N           Training CPUs. Default: 4
  --mem MEM                   Training memory. Default: 48G
  --gres GRES                 Training GPU request. Default: H100 MIG 1g.10gb
  --runner-py PATH            E123 runner. Default: E123 launcher's
                              split-safe H5 runner under --repo-root
  --venv-path PATH            Forwarded to E123 launcher
  --wandb-project NAME        Default: multispecies_e127_synthetic_ssl
  --wandb-group-prefix NAME   Default: E127_synthetic_ssl
  --dry-run                   Write scripts and run E123 launchers in dry-run mode
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-h5) BASE_H5="$2"; shift 2 ;;
    --variant)
      if [[ "$VARIANTS_CUSTOM" != "true" ]]; then
        VARIANTS=()
        VARIANTS_CUSTOM="true"
      fi
      VARIANTS+=("$2")
      shift 2
      ;;
    --synthetic-per-target) SYNTHETIC_PER_TARGET="$2"; shift 2 ;;
    --synthetic-split) SYNTHETIC_SPLIT="$2"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --ssl-repo-root) SSL_REPO_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --allow-missing-base-h5) ALLOW_MISSING_BASE_H5="true"; shift ;;
    --augment-time) AUGMENT_TIME="$2"; shift 2 ;;
    --augment-cpus-per-task) AUGMENT_CPUS="$2"; shift 2 ;;
    --augment-mem) AUGMENT_MEM="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --snr-db-min) SNR_DB_MIN="$2"; shift 2 ;;
    --snr-db-max) SNR_DB_MAX="$2"; shift 2 ;;
    --freq-shift-min-bins) FREQ_SHIFT_MIN_BINS="$2"; shift 2 ;;
    --freq-shift-max-bins) FREQ_SHIFT_MAX_BINS="$2"; shift 2 ;;
    --time-shift-min-bins) TIME_SHIFT_MIN_BINS="$2"; shift 2 ;;
    --time-shift-max-bins) TIME_SHIFT_MAX_BINS="$2"; shift 2 ;;
    --time-stretch-min) TIME_STRETCH_MIN="$2"; shift 2 ;;
    --time-stretch-max) TIME_STRETCH_MAX="$2"; shift 2 ;;
    --nonlinear-distortion-strength-min) NONLINEAR_DISTORTION_STRENGTH_MIN="$2"; shift 2 ;;
    --nonlinear-distortion-strength-max) NONLINEAR_DISTORTION_STRENGTH_MAX="$2"; shift 2 ;;
    --spectral-filter-strength-min) SPECTRAL_FILTER_STRENGTH_MIN="$2"; shift 2 ;;
    --spectral-filter-strength-max) SPECTRAL_FILTER_STRENGTH_MAX="$2"; shift 2 ;;
    --transmission-loss-min) TRANSMISSION_LOSS_MIN="$2"; shift 2 ;;
    --transmission-loss-max) TRANSMISSION_LOSS_MAX="$2"; shift 2 ;;
    --reverb-smear-strength-min) REVERB_SMEAR_STRENGTH_MIN="$2"; shift 2 ;;
    --reverb-smear-strength-max) REVERB_SMEAR_STRENGTH_MAX="$2"; shift 2 ;;
    --reverb-smear-decay-min-bins) REVERB_SMEAR_DECAY_MIN_BINS="$2"; shift 2 ;;
    --reverb-smear-decay-max-bins) REVERB_SMEAR_DECAY_MAX_BINS="$2"; shift 2 ;;
    --end-trim-fraction-min) END_TRIM_FRACTION_MIN="$2"; shift 2 ;;
    --end-trim-fraction-max) END_TRIM_FRACTION_MAX="$2"; shift 2 ;;
    --gaussian-noise-std) GAUSSIAN_NOISE_STD="$2"; shift 2 ;;
    --num-pretrain-jobs) NUM_PRETRAIN_JOBS="$2"; shift 2 ;;
    --num-finetune-jobs) NUM_FINETUNE_JOBS="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --runner-py) RUNNER_PY="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group-prefix) WANDB_GROUP_PREFIX="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$BASE_H5" ]]; then
  echo "Missing required --base-h5" >&2
  usage
  exit 1
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$WEEKEND_ROOT/runs/E127_synthetic_ssl_suite_${STAMP}"
fi
if [[ ! -d "$REPO_ON_NIBI" ]]; then
  echo "Missing repo root: $REPO_ON_NIBI" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "true" && ! -f "$BASE_H5" && "$ALLOW_MISSING_BASE_H5" != "true" ]]; then
  echo "Missing base H5: $BASE_H5" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "true" && ! -f "$BASE_H5" && "$ALLOW_MISSING_BASE_H5" == "true" && -z "$DEPENDENCY" ]]; then
  echo "Missing base H5 is only allowed with --dependency so downstream jobs wait for the upstream producer/audit" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/datasets" "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/submit_logs"
PLAN_TSV="$OUTPUT_ROOT/e127_synthetic_ssl_suite_plan.tsv"
printf "variant\tlabels\tdataset_h5\tsummary_json\taugment_job\te123_run_root\te123_submit_log\tnote\n" > "$PLAN_TSV"

sanitize_name() {
  printf "%s" "$1" | tr -cs 'A-Za-z0-9_-' '_' | sed 's/^_//; s/_$//'
}

labels_are_none() {
  local labels="$1"
  [[ -z "$labels" || "$labels" == "none" || "$labels" == "NONE" || "$labels" == "baseline" ]]
}

submit_job() {
  local job_path="$1"
  local dep="$2"
  local cmd=(sbatch --parsable)
  if [[ -n "$dep" ]]; then
    cmd+=(--dependency="$dep")
  fi
  cmd+=("$job_path")
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: ${cmd[*]}"
    return 0
  fi
  "${cmd[@]}"
}

submit_e123_variant() {
  local variant="$1"
  local dataset_h5="$2"
  local dependency="$3"
  local allow_missing="$4"
  local run_root="$OUTPUT_ROOT/runs/$variant"
  local submit_log="$OUTPUT_ROOT/submit_logs/e123_${variant}.log"
  local cmd=(bash drac/scripts/submit_multispecies_e123_ssl_ssamba.sh
    --dataset-h5 "$dataset_h5"
    --repo-root "$REPO_ON_NIBI"
    --ssl-repo-root "$SSL_REPO_ROOT"
    --run-root "$run_root"
    --python-bin "$PYTHON_BIN"
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "${WANDB_GROUP_PREFIX}_${STAMP}_${variant}"
    --train-ratio "$TRAIN_RATIO"
    --pretrain-task "$PRETRAIN_TASK"
    --finetune-task "$FINETUNE_TASK"
    --num-pretrain-jobs "$NUM_PRETRAIN_JOBS"
    --num-finetune-jobs "$NUM_FINETUNE_JOBS"
    --time "$SBATCH_TIME"
    --cpus-per-task "$SBATCH_CPUS"
    --mem "$SBATCH_MEM"
    --gres "$SBATCH_GRES")
  if [[ -n "$RUNNER_PY" ]]; then
    cmd+=(--runner-py "$RUNNER_PY")
  fi
  if [[ -n "$VENV_PATH" ]]; then
    cmd+=(--venv-path "$VENV_PATH")
  fi
  if [[ -n "$dependency" ]]; then
    cmd+=(--dependency "$dependency")
  fi
  if [[ "$allow_missing" == "true" ]]; then
    cmd+=(--allow-missing-dataset)
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    cmd+=(--dry-run)
  fi
  (
    cd "$REPO_ON_NIBI"
    "${cmd[@]}"
  ) | tee "$submit_log"
}

variant_index=0
for spec in "${VARIANTS[@]}"; do
  variant_index=$((variant_index + 1))
  if [[ "$spec" != *=* ]]; then
    echo "Invalid --variant spec '$spec' (expected NAME=LABELS)" >&2
    exit 1
  fi
  raw_name="${spec%%=*}"
  labels="${spec#*=}"
  variant="$(sanitize_name "$raw_name")"
  if [[ -z "$variant" ]]; then
    echo "Invalid empty variant name in '$spec'" >&2
    exit 1
  fi

  dataset_h5="$BASE_H5"
  summary_json=""
  augment_job=""
  e123_dependency="$DEPENDENCY"
  allow_missing="$ALLOW_MISSING_BASE_H5"
  note="baseline H5 without synthetic rows"

  if ! labels_are_none "$labels"; then
    dataset_h5="$OUTPUT_ROOT/datasets/e127_${variant}.h5"
    summary_json="$OUTPUT_ROOT/datasets/e127_${variant}.summary.json"
    augment_script="$OUTPUT_ROOT/logs/E127_augment_${variant}.sbatch"
    variant_seed=$((SEED + variant_index))
    IFS=',' read -r -a label_array <<< "$labels"
    target_args=()
    for label in "${label_array[@]}"; do
      label="${label//[[:space:]]/}"
      if [[ -n "$label" ]]; then
        target_args+=(--target-label "$label")
      fi
    done
    if [[ "${#target_args[@]}" -eq 0 ]]; then
      echo "Variant '$variant' has no usable target labels" >&2
      exit 1
    fi
    printf '%s\0' "${target_args[@]}" > "$OUTPUT_ROOT/logs/E127_augment_${variant}_target_args.nul"
    cat > "$augment_script" <<EOF
#!/bin/bash
#SBATCH --job-name=E127aug${variant}
#SBATCH --output=$OUTPUT_ROOT/logs/augment_${variant}_%j.out
#SBATCH --error=$OUTPUT_ROOT/logs/augment_${variant}_%j.err
#SBATCH --time=$AUGMENT_TIME
#SBATCH --cpus-per-task=$AUGMENT_CPUS
#SBATCH --mem=$AUGMENT_MEM

set -euo pipefail
REPO="$REPO_ON_NIBI"
BASE_H5="$BASE_H5"
OUT_H5="$dataset_h5"
SUMMARY_JSON="$summary_json"

cd "\$REPO"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"

TARGET_ARGS=()
mapfile -d '' -t TARGET_ARGS < "$OUTPUT_ROOT/logs/E127_augment_${variant}_target_args.nul"

$PYTHON_BIN -u scripts/data/multilabel/build_e127_synthetic_h5_dataset.py \\
  --input-h5 "\$BASE_H5" \\
  --output-h5 "\$OUT_H5" \\
  --output-summary "\$SUMMARY_JSON" \\
  "\${TARGET_ARGS[@]}" \\
  --synthetic-per-target "$SYNTHETIC_PER_TARGET" \\
  --split "$SYNTHETIC_SPLIT" \\
  --seed "$variant_seed" \\
  --snr-db-min "$SNR_DB_MIN" \\
  --snr-db-max "$SNR_DB_MAX" \\
  --freq-shift-min-bins "$FREQ_SHIFT_MIN_BINS" \\
  --freq-shift-max-bins "$FREQ_SHIFT_MAX_BINS" \\
  --time-shift-min-bins "$TIME_SHIFT_MIN_BINS" \\
  --time-shift-max-bins "$TIME_SHIFT_MAX_BINS" \\
  --time-stretch-min "$TIME_STRETCH_MIN" \\
  --time-stretch-max "$TIME_STRETCH_MAX" \\
  --nonlinear-distortion-strength-min "$NONLINEAR_DISTORTION_STRENGTH_MIN" \\
  --nonlinear-distortion-strength-max "$NONLINEAR_DISTORTION_STRENGTH_MAX" \\
  --spectral-filter-strength-min "$SPECTRAL_FILTER_STRENGTH_MIN" \\
  --spectral-filter-strength-max "$SPECTRAL_FILTER_STRENGTH_MAX" \\
  --transmission-loss-strength-min "$TRANSMISSION_LOSS_MIN" \\
  --transmission-loss-strength-max "$TRANSMISSION_LOSS_MAX" \\
  --reverb-smear-strength-min "$REVERB_SMEAR_STRENGTH_MIN" \\
  --reverb-smear-strength-max "$REVERB_SMEAR_STRENGTH_MAX" \\
  --reverb-smear-decay-min-bins "$REVERB_SMEAR_DECAY_MIN_BINS" \\
  --reverb-smear-decay-max-bins "$REVERB_SMEAR_DECAY_MAX_BINS" \\
  --end-trim-fraction-min "$END_TRIM_FRACTION_MIN" \\
  --end-trim-fraction-max "$END_TRIM_FRACTION_MAX" \\
  --gaussian-noise-std "$GAUSSIAN_NOISE_STD"
EOF
    if [[ "$DRY_RUN" == "true" ]]; then
      submit_job "$augment_script" "$DEPENDENCY" | tee "$OUTPUT_ROOT/submit_logs/augment_${variant}.log"
    else
      augment_job="$(submit_job "$augment_script" "$DEPENDENCY")"
      echo "Submitted E127 augmentation job for $variant: $augment_job" | tee "$OUTPUT_ROOT/submit_logs/augment_${variant}.log"
      e123_dependency="afterok:$augment_job"
    fi
    allow_missing="true"
    note="synthetic train rows appended for labels $labels; validation/test rows remain real"
  fi

  submit_e123_variant "$variant" "$dataset_h5" "$e123_dependency" "$allow_missing"
  e123_run_root="$OUTPUT_ROOT/runs/$variant"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$variant" "$labels" "$dataset_h5" "$summary_json" "$augment_job" "$e123_run_root" \
    "$OUTPUT_ROOT/submit_logs/e123_${variant}.log" "$note" >> "$PLAN_TSV"
done

cat <<EOF
E127 synthetic SSL suite prepared.
Suite root: $OUTPUT_ROOT
Plan TSV: $PLAN_TSV
Base H5: $BASE_H5
Variants: ${VARIANTS[*]}

After the E123 fine-tune/evaluation summaries exist, build an E124 leaderboard
with --ledger-path docs/multispecies_experiment_results.md so the final metrics
are written to the living experiment ledger.
EOF
