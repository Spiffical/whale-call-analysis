# Multispecies Experiment Results Ledger

Last updated: 2026-06-12

This is the living local ledger for multispecies whale-call experiments. Add an
entry after every completed experiment, including:

- experiment/run id and date
- training set and validation set
- test set and whether it is production-style common-row evaluation
- headline metrics: precision, recall, F1, accuracy when available
- per-species metrics when available
- cross-species false positives, background false positives, and species-as-background false negatives when available
- artifact/report paths
- interpretation and caveats

Production-style metrics mean all species are evaluated on the same held-out ONC
rows, so a fin whale row can count as a false positive for blue/humpback if those
heads fire. Expert-only one-vs-rest metrics are useful diagnostics, but they can
be optimistic if each expert is not evaluated against the same cross-species
negative examples.

## Logging Future Experiments

Every completed experiment should update this file before we treat the result as
reviewed. For E126-style binary gate reports, pass the ledger arguments directly
to the report command:

```bash
python scripts/analysis/e126_binary_gate_report.py \
  --name EXPERIMENT_NAME \
  --val-predictions VAL_PREDICTIONS.csv \
  --test-predictions TEST_PREDICTIONS.csv \
  --output-dir OUTPUT_DIR \
  --ledger-path docs/multispecies_experiment_results.md \
  --training-set "brief training-set description" \
  --validation-set "brief validation-set description" \
  --test-set "brief test-set description" \
  --evaluation-note "whether this is common-row production-style evaluation"
```

Existing binary gate summaries can also be appended after the fact:

```bash
python scripts/analysis/multispecies_experiment_ledger.py binary-gate \
  --summary-json OUTPUT_DIR/e126_binary_gate_summary.json \
  --ledger-path docs/multispecies_experiment_results.md \
  --training-set "brief training-set description" \
  --validation-set "brief validation-set description" \
  --test-set "brief test-set description"
```

For production-style model comparisons, build the E124 leaderboard with ledger
arguments:

```bash
python scripts/analysis/e124_compare_production_candidates.py \
  --candidate SSL_BASELINE=PATH_TO_SUMMARY.json \
  --candidate SSL_FINETUNED=PATH_TO_SUMMARY.json \
  --output-dir OUTPUT_DIR \
  --ledger-path docs/multispecies_experiment_results.md \
  --training-set "candidate-specific; see source summaries" \
  --validation-set "candidate-specific; see source summaries" \
  --test-set "shared common-row ONC test set" \
  --evaluation-note "production-style common-row comparison with cross-species false positives counted"
```

Synthetic augmentation suite details live in
`docs/e127_synthetic_ssl_suite.md`. The E127 suite is not considered complete
until its variants have real common-row ONC test metrics and an E124 leaderboard
entry in this ledger.

## Dataset And Evaluation References

| Reference | Description | Notes |
| --- | --- | --- |
| E24 expert hparam | Historical one-vs-rest fin/blue/humpback expert training. | Strong diagnostic metrics, but not common-row production comparable. |
| E58 two-stage | Binary whale gate plus species stage. | Useful for "whale vs background" and expert-in-the-loop triage, but species assignment remains the bottleneck. |
| E99/E115/E116 reports | Production-style common-row ONC evaluation. | Best current source for cross-species confusion metrics. |
| E16 broad manifest | Broad ONC/BioDCASE/DCLDE standardized manifest for SSL H5 bridge. | Good normal/background coverage for SSL pretraining. |
| E100/E101 ONC blocked manifest | ONC-only blocked validation manifest. | Too few background rows for SSL normal pretraining by itself. |

## Experiment Ledger

### E24: One-Vs-Rest Expert Hyperparameter Run

Status: completed historical run.

Training set: mixed multispecies expert setup, one expert per target species
(`Bp`, `Bm`, `Mn`) using the E24 training pipeline.

Validation/test set: expert-specific one-vs-rest splits. These were not the
later common-row production evaluation where every species head is scored on the
same ONC test rows.

Headline metrics previously reported:

| Metric | Value |
| --- | ---: |
| Macro F1 | ~0.944 |
| Micro F1 | ~0.969 |
| Precision | ~0.959 |
| Recall | ~0.979 |
| Hard FP rate | ~0.185 |

Interpretation: E24 showed the expert architecture could learn strong
species-specific detectors, especially when not evaluated as a production
ensemble with cross-species competition. Treat these as optimistic diagnostic
metrics, not as production species-discrimination metrics.

Artifacts: historical E24 run on Nibi under
`/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/`.

### E58/E126: Supervised Binary Whale Gate Report

Status: completed report generated from saved E58 predictions.

Training set: E58 supervised two-stage setup with a binary whale-call gate and a
species stage.

Validation set: E58 validation predictions, 1,728 rows.

Test set: E58 saved test predictions, 8,350 rows. Test support is 8,220 whale
rows and 130 background rows.

Report artifacts:

`/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/pipeline_runs/e126_binary_gate_reports_20260612T0335Z/e58_supervised_gate/`

Files include `e126_binary_gate_report.md`,
`e126_binary_gate_summary.json`, `e126_binary_gate_metrics.csv`,
`e126_binary_gate_breakdown.csv`, and example rows.

Validation metrics at tuned threshold 0.08:

| Metric | Value |
| --- | ---: |
| Precision | 0.9948 |
| Recall | 1.0000 |
| F1 | 0.9974 |
| Accuracy | 0.9948 |
| TP / FP / TN / FN | 1711 / 9 / 8 / 0 |

Test metrics at threshold 0.08:

| Metric | Value |
| --- | ---: |
| Precision | 0.9853 |
| Recall | 0.9931 |
| F1 | 0.9892 |
| Accuracy | 0.9786 |
| TP / FP / TN / FN | 8163 / 122 / 8 / 57 |

Test breakdown:

| Group | Support | Detected | Missed/TN | Rate |
| --- | ---: | ---: | ---: | ---: |
| Background | 130 | 122 | 8 | 0.9385 FP rate |
| Fin whale (`Bp`) | 7,827 | 7,772 | 55 | 0.9930 recall |
| Blue whale (`Bm`) | 23 | 23 | 0 | 1.0000 recall |
| Humpback (`Mn`) | 370 | 368 | 2 | 0.9946 recall |

Interpretation: the binary gate is very sensitive for whale calls, which is good
for expert-in-the-loop triage. However, on this small background test subset it
fires on most background rows, so it is not yet a strong background rejector for
production without more background-negative evaluation and/or thresholding work.

### E58: Two-Stage Species Assignment

Status: completed historical report summary.

Training set: E58 two-stage supervised setup.

Test set: same 8,350-row E58 held-out test predictions, with cross-species
confusion counted in the species stage.

`best_val_macro` test metrics:

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.4357 |
| Micro F1 | 0.8153 |
| Precision | 0.8123 |
| Recall | 0.8182 |
| Cross-species FP | 1,432 |
| Background FP | 122 |
| Species-as-background FN | 62 |

Per-species F1 for `best_val_macro`:

| Species | F1 |
| --- | ---: |
| Fin whale (`Bp`) | 0.8911 |
| Blue whale (`Bm`) | 0.1500 |
| Humpback (`Mn`) | 0.2661 |

`best_val_micro` test metrics:

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.4806 |
| Micro F1 | 0.9147 |
| Precision | 0.9114 |
| Recall | 0.9180 |
| Cross-species FP | 612 |
| Background FP | 122 |
| Species-as-background FN | 62 |

Per-species F1 for `best_val_micro`:

| Species | F1 |
| --- | ---: |
| Fin whale (`Bp`) | 0.9500 |
| Blue whale (`Bm`) | 0.1818 |
| Humpback (`Mn`) | 0.3100 |

Interpretation: the two-stage system is promising as a high-recall whale gate,
but production species discrimination is still limited, especially for blue whale
and humpback calls.

### E99/E115/E116: Production-Style Common-Row Candidate Models

Status: completed production-style report summaries.

Training set: candidate multiclass/two-stage multispecies runs using ONC and/or
full mixed-source data, depending on run.

Test set: common-row ONC held-out set with 8,350 rows. Cross-species false
positives are counted against the predicted species and the true species.

Best current common-row candidate:

`E96_E97_plus_E98_ensemble best_val_macro`

Artifact:

`/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/pipeline_runs/e99_blocked_val_production_report_20260611T015600Z/e99_model_metrics.csv`

Overall test metrics:

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.5003 |
| Micro F1 | 0.8896 |
| Precision | 0.8831 |
| Recall | 0.8961 |
| Cross-species FP | 851 |
| Background FP | 124 |
| Species-as-background FN | 3 |

Per-species metrics:

| Species | Precision | Recall | F1 | TP | FP | FN | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fin whale (`Bp`) | 0.9591 | 0.9166 | 0.9373 | 7,174 | 306 | 653 | 7,827 |
| Blue whale (`Bm`) | 0.2500 | 0.2609 | 0.2553 | 6 | 18 | 17 | 23 |
| Humpback (`Mn`) | 0.2222 | 0.5027 | 0.3082 | 186 | 651 | 184 | 370 |

Other close candidates:

| Run | Macro F1 | Micro F1 | Precision | Recall | Cross FP | Background FP | Species-bg FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E96_E97_two_stage best_val_micro | 0.4925 | 0.8699 | 0.8642 | 0.8757 | 1,007 | 124 | 15 |
| E98_full_multiclass best_val_micro | 0.4809 | 0.8938 | 0.8876 | 0.9001 | 812 | 125 | 9 |

Interpretation: common-row production evaluation drops the headline macro F1
substantially compared with E24 expert-only metrics. Fin whale remains strong,
but blue whale is data-limited and humpback has substantial cross-species
confusion.

### E126: SSL H5 Bridge For Normal Spectrogram Pretraining

Status: implementation complete; Nibi H5-build job pending as of last check.

Purpose: build an H5 dataset for SSAMBA-style self-supervised pretraining on
normal/background spectrograms, then fine-tune on whale-call labels.

Source manifest selected: E16 broad manifest
`manifests/E16large_onc_biod_dclde_multiband40s_species/standardized_manifest.csv`.

Normal/background audit:

| Manifest | Background rows | Train background | Val background | Test background | Month coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| E16 broad manifest | 16,376 | 11,491 | 2,442 | 2,443 | 94 months |
| E100/E101 ONC blocked manifest | 176 | 0 | 46 | 130 | 2025-11 to 2025-12 only |

Interpretation: E16 satisfies the "at least 10,000 normal spectrograms spread
throughout the year" requirement much better than the ONC blocked manifest. The
blocked ONC manifest should remain a production evaluation source, not the SSL
normal-pretraining source.

Nibi job:

| Job | State at last check | Reason |
| --- | --- | --- |
| 15973986 / E126sslH5 | PENDING | `ReqNodeNotAvail, Reserved for maintenance` |

Expected output:

`/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/datasets/e126_ssl_e16_low_bgall_target3000_20260612T031656Z.h5`

Interpretation: no SSL whale-vs-background metrics are available yet because the
bridge job has not produced the H5 dataset.

### E127: Synthetic Spectrogram Augmentation Tool

Status: implementation complete; no trained model metrics yet.

Purpose: test whether synthetic blue whale and humpback examples improve scarce
classes without contaminating the real held-out test set.

Implementation:

`scripts/data/multilabel/build_e127_synthetic_h5_dataset.py`

Training set: future variants will append synthetic examples only to the
training split of an existing H5 dataset.

Validation/test set: validation and test rows remain real examples only.

Planned variants:

| Variant | Synthetic target classes | Evaluation |
| --- | --- | --- |
| E127a | none | Real common-row ONC test baseline |
| E127b | `Bm` only | Real common-row ONC test |
| E127c | `Mn` only | Real common-row ONC test |
| E127d | `Bm` + `Mn` | Real common-row ONC test |

Interpretation: this is a spectrogram-space approximation inspired by the Nature
paper/GAVDNet-style synthetic augmentation idea. It still needs training and
production-style common-row evaluation before we can say whether it helps blue
whale or humpback detection.

## Immediate Next Entries To Add

Add rows here when the next jobs complete:

- E126 H5 bridge completion: record final normal/call counts, split counts,
  month coverage, and H5 path.
- E126 supervised/SSL binary gate: record whale-vs-background precision, recall,
  F1, accuracy, confusion matrix, and background FP rate.
- E127 synthetic augmentation variants: record full common-row per-species
  metrics and cross-species false positives for each synthetic-data variant.
