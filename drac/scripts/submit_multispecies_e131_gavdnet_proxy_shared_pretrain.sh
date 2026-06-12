#!/bin/bash
# Submit a richer GAVDNet-proxy synthetic Bm/Mn branch using the shared E128 SSL pretrain.

set -euo pipefail

WEEKEND_ROOT="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502"
REPO="$WEEKEND_ROOT/repo_e24_expert_hparam_68be99f"
SSL="$WEEKEND_ROOT/selfsupervision_anomalies_onc"
SSL_VENV="$SSL/myenv"
BASE_H5="$WEEKEND_ROOT/datasets/e126_ssl_e16_low_bgall_target3000_20260612T031656Z.h5"
BROAD_AUDIT="$WEEKEND_ROOT/pipeline_runs/e126_ssl_h5_audit_20260612T031656Z/e126_ssl_h5_audit_summary.json"
EVAL_H5="$WEEKEND_ROOT/datasets/e128_onc_eval_h5_20260612T095926Z/e128_onc_common_eval_low_20260612T095926Z.h5"
EVAL_AUDIT="$WEEKEND_ROOT/pipeline_runs/e128_onc_eval_h5_audit_20260612T095926Z/e126_ssl_h5_audit_summary.json"
SHARED_PRETRAIN_ROOT="$WEEKEND_ROOT/runs/E128_ssl_binary_gate_20260612T102809Z"
E130_ROOT="$WEEKEND_ROOT/pipeline_runs/e130_shared_pretrain_synthetic_multiclass_20260612T105405Z"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT=""
VARIANT="bm_mn_gavdnet_proxy"
DRY_RUN="false"

TRAIN_H5_AUDIT_JOB="15974542"
SHARED_PRETRAIN_FINAL_JOB="15974555"
EVAL_H5_AUDIT_JOB="15974543"
E130_BASELINE_POST_JOB="15974563"
E130_CONSERVATIVE_POST_JOB="15974564"

usage() {
  cat <<'USAGE'
Usage:
  bash drac/scripts/submit_multispecies_e131_gavdnet_proxy_shared_pretrain.sh [options]

Queues a richer E131 synthetic augmentation branch:
  1. build Bm+Mn synthetic H5 with GAVDNet-inspired spectrogram proxies enabled
  2. fine-tune multiclass SSAMBA from the shared E128 pretrain checkpoint
  3. score on the E128 ONC common-row eval H5 using E129
  4. compare E130 baseline, E130 conservative synthetic, and E131 richer synthetic
  5. run H5/readiness audits

Defaults are tied to the 2026-06-12 E128/E130 queue and keep jobs <=3 hours.

Options:
  --weekend-root PATH
  --repo-root PATH
  --ssl-repo-root PATH
  --ssl-venv PATH
  --base-h5 PATH
  --broad-audit-json PATH
  --eval-h5 PATH
  --eval-audit-json PATH
  --shared-pretrain-root PATH
  --e130-root PATH
  --output-root PATH
  --stamp STAMP
  --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weekend-root) WEEKEND_ROOT="$2"; shift 2 ;;
    --repo-root) REPO="$2"; shift 2 ;;
    --ssl-repo-root) SSL="$2"; shift 2 ;;
    --ssl-venv) SSL_VENV="$2"; shift 2 ;;
    --base-h5) BASE_H5="$2"; shift 2 ;;
    --broad-audit-json) BROAD_AUDIT="$2"; shift 2 ;;
    --eval-h5) EVAL_H5="$2"; shift 2 ;;
    --eval-audit-json) EVAL_AUDIT="$2"; shift 2 ;;
    --shared-pretrain-root) SHARED_PRETRAIN_ROOT="$2"; shift 2 ;;
    --e130-root) E130_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$WEEKEND_ROOT/pipeline_runs/e131_gavdnet_proxy_synthetic_multiclass_${STAMP}"
fi

if [[ ! -d "$REPO" ]]; then
  echo "Missing repo root: $REPO" >&2
  exit 1
fi
if [[ ! -d "$SSL" ]]; then
  echo "Missing SSL repo root: $SSL" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/datasets" "$OUTPUT_ROOT/runs/$VARIANT" "$OUTPUT_ROOT/reports/$VARIANT"

AUG_H5="$OUTPUT_ROOT/datasets/e131_${VARIANT}.h5"
AUG_SUMMARY="$OUTPUT_ROOT/datasets/e131_${VARIANT}.summary.json"

submit_job() {
  local job_path="$1"
  local dep="${2:-}"
  local dry_run_id="${3:-DRYRUN_JOB}"
  local cmd=(sbatch --parsable)
  if [[ -n "$dep" ]]; then
    cmd+=(--dependency="$dep")
  fi
  cmd+=("$job_path")
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: ${cmd[*]}" >&2
    echo "$dry_run_id"
    return 0
  fi
  "${cmd[@]}"
}

AUG_SBATCH="$OUTPUT_ROOT/logs/E131_augment_${VARIANT}.sbatch"
cat > "$AUG_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=E131augBmMn
#SBATCH --output=$OUTPUT_ROOT/logs/augment_${VARIANT}_%j.out
#SBATCH --error=$OUTPUT_ROOT/logs/augment_${VARIANT}_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail
cd "$REPO"
if [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
python3 -u scripts/data/multilabel/build_e127_synthetic_h5_dataset.py \\
  --input-h5 "$BASE_H5" \\
  --output-h5 "$AUG_H5" \\
  --output-summary "$AUG_SUMMARY" \\
  --target-label Bm \\
  --target-label Mn \\
  --synthetic-per-target 1000 \\
  --split train \\
  --seed 2741 \\
  --snr-db-min -10.0 \\
  --snr-db-max 10.0 \\
  --freq-shift-min-bins -16 \\
  --freq-shift-max-bins 16 \\
  --time-shift-min-bins -8 \\
  --time-shift-max-bins 8 \\
  --time-stretch-min 0.93 \\
  --time-stretch-max 1.07 \\
  --nonlinear-distortion-strength-min 0.10 \\
  --nonlinear-distortion-strength-max 0.50 \\
  --spectral-filter-strength-min 0.10 \\
  --spectral-filter-strength-max 0.50 \\
  --transmission-loss-strength-min 0.10 \\
  --transmission-loss-strength-max 0.75 \\
  --reverb-smear-strength-min 0.10 \\
  --reverb-smear-strength-max 0.35 \\
  --reverb-smear-decay-min-bins 2 \\
  --reverb-smear-decay-max-bins 12 \\
  --end-trim-fraction-min 0.0 \\
  --end-trim-fraction-max 0.10 \\
  --gaussian-noise-std 0.01
EOF
AUG_JOB="$(submit_job "$AUG_SBATCH" "afterok:$TRAIN_H5_AUDIT_JOB" "DRYRUN_E131_AUG")"
echo "Submitted E131 augment job: $AUG_JOB"

FT_SBATCH="$OUTPUT_ROOT/logs/E131_finetune_${VARIANT}.sbatch"
cat > "$FT_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=E131ftBmMn
#SBATCH --output=$OUTPUT_ROOT/logs/finetune_${VARIANT}_%j.out
#SBATCH --error=$OUTPUT_ROOT/logs/finetune_${VARIANT}_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

set -euo pipefail
REPO="$REPO"
SSL="$SSL"
SSL_VENV="$SSL_VENV"
DATASET_H5="$AUG_H5"
RUN_ROOT="$OUTPUT_ROOT/runs/$VARIANT"
SHARED_PRETRAIN_ROOT="$SHARED_PRETRAIN_ROOT"

cd "\$SSL"
if [[ -f "\$SSL_VENV/bin/activate" ]]; then
  source "\$SSL_VENV/bin/activate"
else
  echo "Missing SSL venv at \$SSL_VENV/bin/activate" >&2
  exit 2
fi
export PYTHONPATH="\$SSL:\$REPO:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/merileo/.cache}"

CKPT=\$(find "\$SHARED_PRETRAIN_ROOT/pretrain" -path '*/models/pretrain-joint_best_checkpoint.pth' -print 2>/dev/null | sort | tail -n 1 || true)
if [[ -z "\$CKPT" ]]; then
  CKPT=\$(find "\$SHARED_PRETRAIN_ROOT/pretrain" -path '*/models/*best_checkpoint.pth' -print 2>/dev/null | sort | tail -n 1 || true)
fi
if [[ -z "\$CKPT" ]]; then
  echo "No shared pretrain checkpoint found under \$SHARED_PRETRAIN_ROOT/pretrain" >&2
  exit 3
fi

RUN_ARGS=(
  --python-script "$REPO/scripts/analysis/e128_run_ssamba_h5.py"
  --dataset "\$DATASET_H5"
  --wandb-project multispecies_e131_synthetic_multiclass
  --wandb-group E131_${VARIANT}_${STAMP}
  --wandb-entity spencer-bialek
  --train-ratio 0.8
  --resume true
  --exp-dir "\$RUN_ROOT"
  --task ft_avgtok
  --venv "\$SSL_VENV"
  --pretrained-path "\$CKPT"
  --multiclass
  --num-classes 4
)

echo "Using shared pretrain checkpoint: \$CKPT"
bash scripts/run_amba_spectrogram.sh "\${RUN_ARGS[@]}"
produced=\$(find "\$RUN_ROOT/finetune" -path '*/models/ft-avgtok_best_checkpoint.pth' -print 2>/dev/null | sort | tail -n 1 || true)
if [[ -z "\$produced" ]]; then
  echo "No finetune checkpoint produced under \$RUN_ROOT/finetune" >&2
  exit 4
fi
echo "Verified checkpoint: \$produced"
EOF
FT_JOB="$(submit_job "$FT_SBATCH" "afterok:$SHARED_PRETRAIN_FINAL_JOB:$AUG_JOB" "DRYRUN_E131_FT")"
echo "Submitted E131 finetune job: $FT_JOB"

POST_SBATCH="$OUTPUT_ROOT/logs/E131_post_${VARIANT}.sbatch"
cat > "$POST_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=E131postBmMn
#SBATCH --output=$OUTPUT_ROOT/logs/post_${VARIANT}_%j.out
#SBATCH --error=$OUTPUT_ROOT/logs/post_${VARIANT}_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail
cd "$REPO"
if [[ -f "$SSL_VENV/bin/activate" ]]; then
  source "$SSL_VENV/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "No usable venv" >&2
  exit 2
fi
export PYTHONPATH="$REPO:$SSL:\${PYTHONPATH:-}"
MODEL_DIR=\$(find "$OUTPUT_ROOT/runs/$VARIANT/finetune" -name args.pkl -printf '%h\n' 2>/dev/null | sort | tail -n 1 || true)
if [[ -z "\$MODEL_DIR" ]]; then
  echo "No model dir with args.pkl under $OUTPUT_ROOT/runs/$VARIANT/finetune" >&2
  exit 3
fi
python scripts/analysis/e129_ssamba_multiclass_production_report.py \\
  --name "E131_SSL_multiclass_${VARIANT}" \\
  --ssl-repo-root "$SSL" \\
  --model-dir "\$MODEL_DIR" \\
  --dataset-h5 "$EVAL_H5" \\
  --output-dir "$OUTPUT_ROOT/reports/$VARIANT" \\
  --task ft_avgtok \\
  --device cpu \\
  --batch-size 32 \\
  --base-decision-mode calibrated \\
  --ledger-path docs/multispecies_experiment_results.md \\
  --ledger-entry-id "e131-ssl-multiclass-${VARIANT}" \\
  --training-set "Shared E128 SSL pretrain on E126 broad H5, then multiclass fine-tune for richer GAVDNet-proxy synthetic variant ${VARIANT}" \\
  --validation-set "ONC common-row validation split from E128 eval H5" \\
  --test-set "ONC common-row held-out test split from E128 eval H5" \\
  --evaluation-note "production-style multiclass species evaluation; richer GAVDNet-proxy Bm/Mn synthetic augmentation; cross-species false positives and background false positives counted"
EOF
POST_JOB="$(submit_job "$POST_SBATCH" "afterok:$FT_JOB:$EVAL_H5_AUDIT_JOB" "DRYRUN_E131_POST")"
echo "Submitted E131 postprocess job: $POST_JOB"

LEADER_SBATCH="$OUTPUT_ROOT/logs/E131_leaderboard.sbatch"
cat > "$LEADER_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=E131leader
#SBATCH --output=$OUTPUT_ROOT/logs/leaderboard_%j.out
#SBATCH --error=$OUTPUT_ROOT/logs/leaderboard_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
fi
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
python scripts/analysis/e124_compare_production_candidates.py \\
  --candidate "baseline=$E130_ROOT/reports/baseline/e129_summary.json" \\
  --candidate "bm_mn_conservative=$E130_ROOT/reports/bm_mn_conservative/e129_summary.json" \\
  --candidate "${VARIANT}=$OUTPUT_ROOT/reports/$VARIANT/e129_summary.json" \\
  --output-dir "$OUTPUT_ROOT/e124_candidate_leaderboard" \\
  --title "E131 Shared-Pretrain Synthetic Multiclass Candidate Leaderboard" \\
  --ledger-path docs/multispecies_experiment_results.md \\
  --ledger-entry-id e131-rich-synthetic-leaderboard \\
  --training-set "Shared E128 SSL pretrain with baseline, conservative Bm/Mn synthetic, and richer GAVDNet-proxy Bm/Mn synthetic fine-tunes" \\
  --validation-set "ONC common-row validation split from E128 eval H5" \\
  --test-set "ONC common-row held-out test split from E128 eval H5" \\
  --evaluation-note "Compare conservative vs richer synthetic augmentation under production-style cross-species accounting"
EOF
LEADER_JOB="$(submit_job "$LEADER_SBATCH" "afterok:$E130_BASELINE_POST_JOB:$E130_CONSERVATIVE_POST_JOB:$POST_JOB" "DRYRUN_E131_LEADER")"
echo "Submitted E131 leaderboard job: $LEADER_JOB"

H5_AUDIT_DIR="$OUTPUT_ROOT/h5_audit_${VARIANT}"
if [[ "$DRY_RUN" == "true" ]]; then
  H5_AUDIT_JOB="DRYRUN_E131_H5_AUDIT"
  echo "DRY RUN: would submit E131 H5 audit afterok:$AUG_JOB"
else
  bash "$REPO/drac/scripts/submit_multispecies_e126_ssl_h5_audit.sh" \\
    --repo-root "$REPO" \\
    --input-h5 "$AUG_H5" \\
    --builder-summary-json "$AUG_SUMMARY" \\
    --output-dir "$H5_AUDIT_DIR" \\
    --dependency "afterok:$AUG_JOB" \\
    --allow-missing-h5 \\
    --min-normal-rows 10000 \\
    --min-normal-train-rows 10000 \\
    --min-normal-months 12 \\
    --min-normal-train-months 12 \\
    --ledger-path docs/multispecies_experiment_results.md \\
    --ledger-entry-id e131-gavdnet-proxy-h5-audit | tee "$OUTPUT_ROOT/logs/e131_h5_audit_submit.log"
  H5_AUDIT_JOB="$(awk '/Submitted E126 H5 audit job:/ {print $NF}' "$OUTPUT_ROOT/logs/e131_h5_audit_submit.log" | tail -n 1)"
  if [[ -z "$H5_AUDIT_JOB" ]]; then
    echo "Failed to parse E131 H5 audit job" >&2
    exit 5
  fi
fi

READY_DIR="$OUTPUT_ROOT/readiness_audit"
mkdir -p "$READY_DIR/logs"
READY_SBATCH="$READY_DIR/logs/E131ready.sbatch"
cat > "$READY_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=E131ready
#SBATCH --output=$READY_DIR/logs/slurm-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f /home/merileo/whale-call-analysis/.venv/bin/activate ]]; then
  source /home/merileo/whale-call-analysis/.venv/bin/activate
fi
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
python scripts/analysis/multispecies_readiness_audit.py \\
  --leaderboard-json "$OUTPUT_ROOT/e124_candidate_leaderboard/e124_candidate_leaderboard.json" \\
  --h5-audit-json "$BROAD_AUDIT" \\
  --h5-audit-json "$EVAL_AUDIT" \\
  --h5-audit-json "$H5_AUDIT_DIR/e126_ssl_h5_audit_summary.json" \\
  --ledger-path docs/multispecies_experiment_results.md \\
  --require-ledger \\
  --output-dir "$READY_DIR" \\
  --title "E131 Rich Synthetic Multiclass Readiness Audit" \\
  --fail-on-incomplete
EOF
READY_DEP="afterok:$LEADER_JOB:$H5_AUDIT_JOB"
READY_JOB="$(submit_job "$READY_SBATCH" "$READY_DEP" "DRYRUN_E131_READY")"
echo "Submitted E131 readiness job: $READY_JOB"

cat > "$OUTPUT_ROOT/e131_submission_metadata.json" <<EOF
{
  "root": "$OUTPUT_ROOT",
  "variant": "$VARIANT",
  "augmentation_job": "$AUG_JOB",
  "finetune_job": "$FT_JOB",
  "postprocess_job": "$POST_JOB",
  "leaderboard_job": "$LEADER_JOB",
  "h5_audit_job": "$H5_AUDIT_JOB",
  "readiness_job": "$READY_JOB",
  "synthetic_h5": "$AUG_H5",
  "synthetic_summary": "$AUG_SUMMARY",
  "leaderboard_dir": "$OUTPUT_ROOT/e124_candidate_leaderboard",
  "readiness_dir": "$READY_DIR",
  "dependencies": {
    "train_h5_audit": "$TRAIN_H5_AUDIT_JOB",
    "shared_pretrain_final_job": "$SHARED_PRETRAIN_FINAL_JOB",
    "eval_h5_audit": "$EVAL_H5_AUDIT_JOB",
    "e130_baseline_postprocess": "$E130_BASELINE_POST_JOB",
    "e130_conservative_postprocess": "$E130_CONSERVATIVE_POST_JOB"
  }
}
EOF

cat "$OUTPUT_ROOT/e131_submission_metadata.json"
if [[ "$DRY_RUN" != "true" ]]; then
  squeue -j "$AUG_JOB,$FT_JOB,$POST_JOB,$LEADER_JOB,$H5_AUDIT_JOB,$READY_JOB" -h -o 'Q|%i|%j|%T|%M|%l|%R' || true
fi
