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
  --num-pretrain-jobs 2 \
  --num-finetune-jobs 1 \
  --time 03:00:00 \
  --gres gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
```

Important blocker: the current local SSAMBA runner records aggregate
`result.csv` metrics but does not yet provide the E126-compatible row-level
prediction CSV needed for production-style false-positive and false-negative
examples. Before treating SSL as reviewed, export row-level validation/test
scores and run `e126_binary_gate_report.py` with `--ledger-path`.

## Current Nibi State

As of the last check, Nibi login was reachable, the whale-call repo on Nibi was
updated to commit `6fac4c3`, but compute nodes were largely maintenance/down:
CPU H5 build job `15973986` remained pending with
`ReqNodeNotAvail,_Reserved_for_maintenance`, and the expected SSL repo path was
not present on scratch. Do not launch more jobs until compute nodes and the SSL
repo path are available.
