# E128 Binary Whale-Gate Comparison

E128 compares the simple supervised whale-call gate against the SSL-pretrained
gate path for the narrower question: "does this clip contain a fin, blue, or
humpback whale call?" Species assignment remains a downstream classifier or
expert-in-the-loop task.

## Evaluation Contract

Use the same ONC held-out rows whenever possible and report:

- precision, recall, F1, accuracy
- TP, FP, TN, FN
- background false-positive rate
- per-species gate recall for `species:Bp`, `species:Bm`, and `species:Mn`
- true positives, false positives, false negatives, and true negatives in the
  examples CSV

For collapsed binary-gate training, the E122 manifest preserves
`original_label_ids` and `gate_positive_source_labels`. The E126 report uses
those audit columns so a `task:whale_call` model can still report per-species
gate recall.

## Supervised Gate

Launch the supervised binary gate with <=3 hour MIG jobs:

```bash
bash drac/scripts/submit_multispecies_e122_two_stage_gate.sh \
  --source-manifest /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/manifests/e100_onc_only_blocked_nov_validation_20260611T020900Z/E101_stage2_ONConly_blocked_nov20_25_30_val/standardized_manifest.csv \
  --variant-tag E128_ONC_binary_gate \
  --time 03:00:00 \
  --gres gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
```

After the continuation job finishes, score the gate directly:

```bash
python scripts/analysis/e126_binary_gate_report.py \
  --name E128_supervised_ONC_binary_gate \
  --val-predictions RUN_DIR/train/validation_predictions.csv \
  --test-predictions RUN_DIR/train/test_predictions.csv \
  --class-ids background,task:whale_call \
  --positive-labels species:Bp,species:Bm,species:Mn \
  --score-label task:whale_call \
  --output-dir OUTPUT_DIR/e128_supervised_ONC_binary_gate \
  --ledger-path docs/multispecies_experiment_results.md \
  --ledger-entry-id e128-supervised-onc-binary-gate \
  --training-set "E122 supervised binary gate; ONC blocked manifest; Bp/Bm/Mn collapsed to task:whale_call" \
  --validation-set "ONC blocked validation rows from the E122 gate manifest" \
  --test-set "ONC blocked held-out test rows from the E122 gate manifest" \
  --evaluation-note "binary whale-call vs background; per-species gate recall recovered from original_label_ids"
```

## SSL Gate

The SSL comparison should use the audited E126 H5 dataset once it exists. The
first H5 bridge job was still pending during the Nibi maintenance/down window:

```text
15973986 / E126sslH5
/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/datasets/e126_ssl_e16_low_bgall_target3000_20260612T031656Z.h5
```

After the H5 exists and the `selfsupervision_anomalies_onc` repo is present on
Nibi, launch the binary fine-tune path:

```bash
bash drac/scripts/submit_multispecies_e123_ssl_ssamba.sh \
  --dataset-h5 /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/datasets/e126_ssl_e16_low_bgall_target3000_20260612T031656Z.h5 \
  --ssl-repo-root /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/selfsupervision_anomalies_onc \
  --binary-finetune \
  --finetune-task ft_cls \
  --num-pretrain-jobs 2 \
  --num-finetune-jobs 1 \
  --time 03:00:00 \
  --gres gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
```

For production-style metrics, export row-level scores from the fine-tuned SSAMBA
checkpoint to E126-compatible CSV. Prefer an ONC held-out evaluation H5 whose
`splits` dataset contains `val` and `test` rows from the common production
manifest; the exporter reads H5 split metadata directly rather than relying on
the SSL repo's internal random split:

```bash
python scripts/analysis/e128_export_ssamba_binary_gate_predictions.py \
  --ssl-repo-root /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/selfsupervision_anomalies_onc \
  --model-dir SSAMBA_FINETUNE_MODEL_DIR \
  --dataset-h5 ONC_COMMON_EVAL_H5_WITH_VAL_TEST_SPLITS.h5 \
  --output-dir OUTPUT_DIR/e128_ssl_binary_gate_predictions \
  --task ft_cls \
  --score-label task:whale_call
```

Then score and ledger the SSL gate:

```bash
python scripts/analysis/e126_binary_gate_report.py \
  --name E128_SSL_binary_gate \
  --val-predictions OUTPUT_DIR/e128_ssl_binary_gate_predictions/validation_predictions.csv \
  --test-predictions OUTPUT_DIR/e128_ssl_binary_gate_predictions/test_predictions.csv \
  --class-ids background,task:whale_call \
  --positive-labels species:Bp,species:Bm,species:Mn \
  --score-label task:whale_call \
  --output-dir OUTPUT_DIR/e128_ssl_binary_gate_report \
  --ledger-path docs/multispecies_experiment_results.md \
  --ledger-entry-id e128-ssl-binary-gate \
  --training-set "SSAMBA normal/background SSL pretraining plus binary whale-call fine-tuning" \
  --validation-set "ONC common-row validation split exported from H5" \
  --test-set "ONC common-row test split exported from H5" \
  --evaluation-note "binary whale-call vs background; per-species gate recall recovered from H5 label_strings"
```

Important blocker: the SSL repo path has been restored on Nibi, but the H5
bridge job must finish before SSL training can launch. The exporter solves
row-level scoring once a trained checkpoint and ONC evaluation H5 exist.

Before treating either gate report as reviewed, run
`scripts/analysis/multispecies_readiness_audit.py` on the
`e126_binary_gate_summary.json`. The audit now checks that overall metrics,
example rows, background false-positive rate, and per-species gate recall are
present, rather than only checking that a summary file exists.

## Current Nibi State

As of the last check, Nibi login was reachable, the whale-call repo was at
`99b6719`, and the clean SSL repo clone was present at commit `9215af6`. Compute
nodes were still largely maintenance/down:
CPU H5 build job `15973986` remained pending with
`ReqNodeNotAvail,_Reserved_for_maintenance`. Do not launch more jobs until
compute nodes are available and the H5 bridge has completed.
