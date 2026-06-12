#!/bin/bash
# Submit the E123 SSAMBA self-supervised pretrain + species fine-tune experiment.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO_ON_NIBI="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
SSL_REPO_ROOT="$WEEKEND_ROOT/selfsupervision_anomalies_onc"
DATASET_H5=""
MANIFEST_CSV=""
DATASET_ROOT="/project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502/mat_archives/multiband40s_20260514T002301Z/extracted"
RUN_ROOT=""
RUNNER_PY="src/run_amba_spectrogram.py"
VENV_PATH=""
VENV_PATH_SET="false"
PYTHON_BIN="python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WANDB_PROJECT="multispecies_e123_ssl"
WANDB_GROUP="E123_ssl_multispecies"
WANDB_ENTITY="spencer-bialek"
TRAIN_RATIO="0.8"
PRETRAIN_TASK="pretrain_joint"
FINETUNE_TASK="ft_avgtok"
NUM_PRETRAIN_JOBS="2"
NUM_FINETUNE_JOBS="1"
FINETUNE_MULTICLASS="true"
NUM_CLASSES="4"
SBATCH_TIME="03:00:00"
SBATCH_CPUS="4"
SBATCH_MEM="48G"
SBATCH_GRES="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
BUILD_H5="false"
H5_BAND="low"
H5_BAND_CROP_SHAPE="391x50"
H5_OUTPUT_SHAPE="512x512"
H5_SPLITS="train,val"
H5_NON_TARGET_MODE="skip"
H5_MAX_NORMAL="10000"
H5_MAX_PER_TARGET="0"
H5_NORMAL_CROPS_PER_ROW="1"
H5_TIME="03:00:00"
H5_CPUS="4"
H5_MEM="48G"
DEPENDENCY=""
DRY_RUN="false"
EXCLUDE_LABELS=()

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e123_ssl_ssamba.sh --dataset-h5 PATH [options]

Submit a resumable SSAMBA experiment for the multispecies work:
  1. self-supervised pretrain on normal/background spectrograms
  2. multiclass fine-tune on normal + Bm/Bp/Mn whale-call examples

The script uses <=3h Slurm jobs by default and links each continuation with
afterany so partially completed runs can resume from checkpoints.

Required:
  --dataset-h5 PATH          Existing HDF5 dataset in selfsupervision_anomalies_onc format
      or
  --manifest-csv PATH        Standardized multispecies manifest to export to H5 first

Options:
  --weekend-root PATH        Default: /scratch/.../multispecies_weekend_20260502
  --repo-root PATH           Whale-call repo root for H5 builder
  --ssl-repo-root PATH       Default: $weekend_root/selfsupervision_anomalies_onc
  --dataset-root PATH        Multiband MAT extraction root for --manifest-csv
  --runner-py PATH           Relative or absolute SSAMBA runner. Default: src/run_amba_spectrogram.py
  --venv-path PATH           Default: $ssl_repo_root/myenv
  --run-root PATH            Default: $weekend_root/runs/E123_ssl_ssamba_multispecies_$stamp
  --stamp STAMP              Default: current UTC stamp
  --python-bin NAME          Default: python
  --wandb-project NAME       Default: multispecies_e123_ssl
  --wandb-group NAME         Default: E123_ssl_multispecies
  --wandb-entity NAME        Default: spencer-bialek
  --train-ratio X            Default: 0.8
  --pretrain-task TASK       Default: pretrain_joint
  --finetune-task TASK       Default: ft_avgtok
  --num-pretrain-jobs N      Default: 2
  --num-finetune-jobs N      Default: 1
  --num-classes N            Default: 4 (normal/background + Bm + Bp + Mn)
  --binary-finetune          Disable multiclass fine-tune flags
  --exclude-label LABEL      Forwarded to SSAMBA runner; may be repeated
  --dependency SPEC          Add dependency to the first submitted job
  --time HH:MM:SS            Default: 03:00:00
  --cpus-per-task N          Default: 4
  --mem MEM                  Default: 48G
  --gres GRES                Default: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
  --h5-band BAND             H5 export band. Default: low
  --h5-band-crop-shape FxT   Default: 391x50
  --h5-output-shape FxT      Default: 512x512
  --h5-splits CSV            Manifest splits for H5 export. Default: train,val
  --h5-non-target-mode MODE  skip or normal. Default: skip
  --h5-max-normal N          Default: 10000
  --h5-max-per-target N      Default: 0 (all)
  --h5-normal-crops-per-row N
                              Deterministic crops per normal row. Default: 1
  --h5-time HH:MM:SS         Default: 03:00:00
  --h5-cpus-per-task N       Default: 4
  --h5-mem MEM               Default: 48G
  --dry-run                  Write scripts and print sbatch commands only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-h5) DATASET_H5="$2"; shift 2 ;;
    --manifest-csv) MANIFEST_CSV="$2"; BUILD_H5="true"; shift 2 ;;
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO_ON_NIBI="$2"; shift 2 ;;
    --ssl-repo-root) SSL_REPO_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --runner-py) RUNNER_PY="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; VENV_PATH_SET="true"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --pretrain-task) PRETRAIN_TASK="$2"; shift 2 ;;
    --finetune-task) FINETUNE_TASK="$2"; shift 2 ;;
    --num-pretrain-jobs) NUM_PRETRAIN_JOBS="$2"; shift 2 ;;
    --num-finetune-jobs) NUM_FINETUNE_JOBS="$2"; shift 2 ;;
    --num-classes) NUM_CLASSES="$2"; shift 2 ;;
    --binary-finetune) FINETUNE_MULTICLASS="false"; shift ;;
    --exclude-label) EXCLUDE_LABELS+=("$2"); shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --time) SBATCH_TIME="$2"; shift 2 ;;
    --cpus-per-task) SBATCH_CPUS="$2"; shift 2 ;;
    --mem) SBATCH_MEM="$2"; shift 2 ;;
    --gres) SBATCH_GRES="$2"; shift 2 ;;
    --h5-band) H5_BAND="$2"; shift 2 ;;
    --h5-band-crop-shape) H5_BAND_CROP_SHAPE="$2"; shift 2 ;;
    --h5-output-shape) H5_OUTPUT_SHAPE="$2"; shift 2 ;;
    --h5-splits) H5_SPLITS="$2"; shift 2 ;;
    --h5-non-target-mode) H5_NON_TARGET_MODE="$2"; shift 2 ;;
    --h5-max-normal) H5_MAX_NORMAL="$2"; shift 2 ;;
    --h5-max-per-target) H5_MAX_PER_TARGET="$2"; shift 2 ;;
    --h5-normal-crops-per-row) H5_NORMAL_CROPS_PER_ROW="$2"; shift 2 ;;
    --h5-time) H5_TIME="$2"; shift 2 ;;
    --h5-cpus-per-task) H5_CPUS="$2"; shift 2 ;;
    --h5-mem) H5_MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$DATASET_H5" && -n "$MANIFEST_CSV" ]]; then
  DATASET_H5="$WEEKEND_ROOT/datasets/e123_ssl_multispecies_${H5_BAND}_${STAMP}.h5"
fi
if [[ -z "$DATASET_H5" ]]; then
  echo "Missing required --dataset-h5 or --manifest-csv" >&2
  usage
  exit 1
fi
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$WEEKEND_ROOT/runs/E123_ssl_ssamba_multispecies_${STAMP}"
fi
if [[ "$VENV_PATH_SET" != "true" ]]; then
  VENV_PATH="$SSL_REPO_ROOT/myenv"
fi

if [[ "$RUNNER_PY" = /* ]]; then
  RUNNER_PY_ABS="$RUNNER_PY"
else
  RUNNER_PY_ABS="$SSL_REPO_ROOT/$RUNNER_PY"
fi

if [[ ! -d "$SSL_REPO_ROOT" ]]; then
  echo "Missing SSL repo: $SSL_REPO_ROOT" >&2
  exit 1
fi
if [[ "$BUILD_H5" == "false" && ! -f "$DATASET_H5" ]]; then
  echo "Missing H5 dataset: $DATASET_H5" >&2
  exit 1
fi
if [[ "$BUILD_H5" == "true" ]]; then
  if [[ ! -d "$REPO_ON_NIBI" ]]; then
    echo "Missing whale-call repo for H5 builder: $REPO_ON_NIBI" >&2
    exit 1
  fi
  if [[ ! -f "$MANIFEST_CSV" ]]; then
    echo "Missing source manifest for H5 builder: $MANIFEST_CSV" >&2
    exit 1
  fi
fi
if [[ ! -f "$SSL_REPO_ROOT/scripts/run_amba_spectrogram.sh" ]]; then
  echo "Missing SSAMBA shell runner: $SSL_REPO_ROOT/scripts/run_amba_spectrogram.sh" >&2
  exit 1
fi
if [[ ! -f "$RUNNER_PY_ABS" ]]; then
  cat >&2 <<EOF
Missing SSAMBA Python runner: $RUNNER_PY_ABS

The current selfsupervision_anomalies_onc checkout may not include the legacy
src/run_amba_spectrogram.py entrypoint. Restore/adapt that runner in the SSL
repo, then re-run this launcher, or pass --runner-py to a valid replacement.
EOF
  exit 1
fi

mkdir -p "$RUN_ROOT/logs"

quote_array() {
  local out=()
  local item
  for item in "$@"; do
    out+=("$(printf "%q" "$item")")
  done
  printf "%s " "${out[@]}"
}

write_job_script() {
  local job_path="$1"
  local phase="$2"
  local task="$3"
  local job_index="$4"

  local multiclass_block=""
  if [[ "$phase" == "finetune" && "$FINETUNE_MULTICLASS" == "true" ]]; then
    multiclass_block='RUN_ARGS+=(--multiclass --num-classes "$NUM_CLASSES")'
  fi

  local exclude_block=""
  if [[ "${#EXCLUDE_LABELS[@]}" -gt 0 ]]; then
    local quoted_labels
    quoted_labels="$(quote_array "${EXCLUDE_LABELS[@]}")"
    exclude_block="for label in $quoted_labels; do RUN_ARGS+=(--exclude-label \"\$label\"); done"
  fi

  cat > "$job_path" <<EOF
#!/bin/bash
#SBATCH --job-name=E123${phase}${job_index}
#SBATCH --output=$RUN_ROOT/logs/${phase}_${job_index}_%j.out
#SBATCH --error=$RUN_ROOT/logs/${phase}_${job_index}_%j.err
#SBATCH --time=$SBATCH_TIME
#SBATCH --cpus-per-task=$SBATCH_CPUS
#SBATCH --mem=$SBATCH_MEM
#SBATCH --gres=$SBATCH_GRES

set -euo pipefail

SSL_REPO_ROOT="$SSL_REPO_ROOT"
DATASET_H5="$DATASET_H5"
RUN_ROOT="$RUN_ROOT"
RUNNER_PY_ABS="$RUNNER_PY_ABS"
VENV_PATH="$VENV_PATH"
PYTHON_BIN="$PYTHON_BIN"
WANDB_PROJECT="$WANDB_PROJECT"
WANDB_GROUP="$WANDB_GROUP"
WANDB_ENTITY="$WANDB_ENTITY"
TRAIN_RATIO="$TRAIN_RATIO"
NUM_CLASSES="$NUM_CLASSES"
PRETRAIN_TASK="$PRETRAIN_TASK"
TASK="$task"

cd "\$SSL_REPO_ROOT"
if type module >/dev/null 2>&1; then
  module load python/3.10 || true
fi
if [[ -f "\$VENV_PATH/bin/activate" ]]; then
  source "\$VENV_PATH/bin/activate"
else
  echo "Missing venv at \$VENV_PATH/bin/activate" >&2
  exit 2
fi
export PYTHONPATH="\$SSL_REPO_ROOT:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"

RUN_ARGS=(
  --python-script "\$RUNNER_PY_ABS"
  --dataset "\$DATASET_H5"
  --wandb-project "\$WANDB_PROJECT"
  --wandb-group "\$WANDB_GROUP"
  --wandb-entity "\$WANDB_ENTITY"
  --train-ratio "\$TRAIN_RATIO"
  --resume true
  --exp-dir "\$RUN_ROOT"
  --task "\$TASK"
  --venv "\$VENV_PATH"
)

$exclude_block

if [[ "$phase" == "finetune" ]]; then
  pretrain_prefix="\${PRETRAIN_TASK//_/-}"
  ckpt=\$(find "\$RUN_ROOT/pretrain" -path "*/models/\${pretrain_prefix}_best_checkpoint.pth" -print 2>/dev/null | sort | tail -n 1 || true)
  if [[ -z "\$ckpt" ]]; then
    ckpt=\$(find "\$RUN_ROOT/pretrain" -path "*/models/*best_checkpoint.pth" -print 2>/dev/null | sort | tail -n 1 || true)
  fi
  if [[ -z "\$ckpt" ]]; then
    echo "No pretrained checkpoint found under \$RUN_ROOT/pretrain" >&2
    exit 3
  fi
  echo "Using pretrained checkpoint: \$ckpt"
  RUN_ARGS+=(--pretrained-path "\$ckpt")
  $multiclass_block
fi

echo "Running E123 $phase job $job_index"
echo "bash scripts/run_amba_spectrogram.sh \${RUN_ARGS[*]}"
bash scripts/run_amba_spectrogram.sh "\${RUN_ARGS[@]}"
EOF
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

previous_job_id=""
previous_dep_mode="afterany"
if [[ "$BUILD_H5" == "true" ]]; then
  h5_script="$RUN_ROOT/logs/E123_build_h5.sbatch"
  mkdir -p "$(dirname "$DATASET_H5")"
  cat > "$h5_script" <<EOF
#!/bin/bash
#SBATCH --job-name=E123h5
#SBATCH --output=$RUN_ROOT/logs/build_h5_%j.out
#SBATCH --error=$RUN_ROOT/logs/build_h5_%j.err
#SBATCH --time=$H5_TIME
#SBATCH --cpus-per-task=$H5_CPUS
#SBATCH --mem=$H5_MEM

set -euo pipefail
REPO="$REPO_ON_NIBI"
MANIFEST="$MANIFEST_CSV"
DATASET_ROOT="$DATASET_ROOT"
OUT_H5="$DATASET_H5"

cd "\$REPO"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"
mkdir -p "\$(dirname "\$OUT_H5")"

python3 -u scripts/data/multilabel/build_e123_ssl_h5_dataset.py \\
  --manifest-csv "\$MANIFEST" \\
  --dataset-root "\$DATASET_ROOT" \\
  --output-h5 "\$OUT_H5" \\
  --band "$H5_BAND" \\
  --band-crop-shape "$H5_BAND_CROP_SHAPE" \\
  --output-shape "$H5_OUTPUT_SHAPE" \\
  --splits "$H5_SPLITS" \\
  --non-target-mode "$H5_NON_TARGET_MODE" \\
  --max-normal "$H5_MAX_NORMAL" \\
  --max-per-target "$H5_MAX_PER_TARGET" \\
  --normal-crops-per-row "$H5_NORMAL_CROPS_PER_ROW"
EOF
  if [[ "$DRY_RUN" == "true" ]]; then
    submit_job "$h5_script" "$DEPENDENCY"
  else
    previous_job_id="$(submit_job "$h5_script" "$DEPENDENCY")"
    previous_dep_mode="afterok"
    echo "Submitted E123 H5 build job: $previous_job_id"
  fi
fi

for ((i = 1; i <= NUM_PRETRAIN_JOBS; i++)); do
  job_script="$RUN_ROOT/logs/E123_pretrain_${i}.sbatch"
  write_job_script "$job_script" "pretrain" "$PRETRAIN_TASK" "$i"
  dep=""
  if [[ -n "$previous_job_id" ]]; then
    dep="$previous_dep_mode:$previous_job_id"
  elif [[ -n "$DEPENDENCY" ]]; then
    dep="$DEPENDENCY"
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    submit_job "$job_script" "$dep"
  else
    job_id="$(submit_job "$job_script" "$dep")"
    echo "Submitted E123 pretrain job $i: $job_id"
    previous_job_id="$job_id"
    previous_dep_mode="afterany"
  fi
done

for ((i = 1; i <= NUM_FINETUNE_JOBS; i++)); do
  job_script="$RUN_ROOT/logs/E123_finetune_${i}.sbatch"
  write_job_script "$job_script" "finetune" "$FINETUNE_TASK" "$i"
  dep=""
  if [[ -n "$previous_job_id" ]]; then
    dep="afterany:$previous_job_id"
  elif [[ -n "$DEPENDENCY" ]]; then
    dep="$DEPENDENCY"
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    submit_job "$job_script" "$dep"
  else
    job_id="$(submit_job "$job_script" "$dep")"
    echo "Submitted E123 finetune job $i: $job_id"
    previous_job_id="$job_id"
    previous_dep_mode="afterany"
  fi
done

cat <<EOF
E123 SSL submission prepared.
Run root: $RUN_ROOT
Logs: $RUN_ROOT/logs
Dataset: $DATASET_H5
Build H5: $BUILD_H5
SSL repo: $SSL_REPO_ROOT
Runner: $RUNNER_PY_ABS
EOF
