# E123 SSL Multispecies Experiment Plan

E123 tests whether SSAMBA-style self-supervised pretraining on normal/background spectrograms helps the scarce blue whale (`Bm`) and humpback (`Mn`) classes before supervised fin/blue/humpback fine-tuning.

## Question

The production question is not just "is this a whale call?" It is whether the final system can avoid species cross-talk: fin calls should not become blue/humpback detections, and blue/humpback calls should survive the lower-data regime. E123 should therefore be evaluated with the same production-style metrics used in E26/E121/E122: one common ONC test set, full confusion matrix, per-species precision/recall/F1, and false-positive examples for each species head.

## Proposed Runs

1. `E123a`: SSAMBA pretrain on normal/background ONC spectrograms only, then multiclass fine-tune on ONC `normal + Bm + Bp + Mn`.
2. `E123b`: same pretrain, but fine-tune with full training data if we have a species-labelled H5 that includes BioDCASE/DCLDE in the same label vocabulary.
3. `E123c`: optional low-data stress test, fine-tuning with reduced blue/humpback positives to see whether SSL improves the classes most likely to be data-limited.

Each run should use 3-hour Nibi MIG jobs chained with `afterany` and `--resume true`. The launcher is `drac/scripts/submit_multispecies_e123_ssl_ssamba.sh`.

## Required Inputs

The SSAMBA code expects an H5 file with at least:

- `spectrograms`: float array shaped like `(n, freq, time, 1)`
- `labels`: binary label matrix
- `sources`: source filename/id strings
- `label_strings`: strings such as `normal`, `Bm`, `Bp`, `Mn`

The current `/home/sbialek/ONC/selfsupervision_anomalies_onc` checkout is useful, but its documented DRAC runner expects `src/run_amba_spectrogram.py`, which is not present in the current checkout. The launcher checks this before submitting. Restore/adapt that runner in the SSL repo, or pass `--runner-py` to a valid replacement.

The H5 bridge is `scripts/data/multilabel/build_e123_ssl_h5_dataset.py`. It exports one band from a standardized multiband manifest into the SSAMBA H5 schema. By default it:

- uses manifest splits `train,val`, leaving the held-out test split untouched for external evaluation
- maps `species:Bm/Bp/Mn` to `Bm/Bp/Mn`
- maps unlabeled rows to `normal`
- skips labeled non-target rows, so the self-supervised normal pretrain is not accidentally polluted with killer whale or other biological calls
- caps normal rows at 10,000 unless overridden
- can export multiple deterministic crops per normal/background row with `--normal-crops-per-row`; this is useful when the broad existing cache has fewer than 10,000 unique normal 40 s contexts but enough temporally broad context to produce 10 s SSL crops

Important audit note: the blocked-validation ONC-only manifest used for E99/E101
has only a small number of normal/background rows and should not be used as the
SSL-normal source by itself. Prefer a broad manifest such as E13/E16, then audit
the resulting H5 month distribution before training.

## Launch Sketch

One-command launch with H5 build, pretrain, and fine-tune chained:

```bash
bash drac/scripts/submit_multispecies_e123_ssl_ssamba.sh \
  --ssl-repo-root /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/selfsupervision_anomalies_onc \
  --manifest-csv /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/manifests/e100_onc_only_blocked_nov_validation_20260611T020900Z/E101_stage2_ONConly_blocked_nov20_25_30_val/standardized_manifest.csv \
  --dataset-root /project/def-kmoran/merileo/whale-call-analysis/multispecies_weekend_20260502/mat_archives/multiband40s_20260514T002301Z/extracted \
  --num-pretrain-jobs 2 \
  --num-finetune-jobs 1 \
  --time 03:00:00 \
  --gres gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
```

If the H5 already exists, pass `--dataset-h5` instead of `--manifest-csv`.

## H5 Coverage Audit

Before treating an H5 as ready for SSL pretraining, audit the actual H5 contents
and append the result to the living experiment ledger:

```bash
bash drac/scripts/submit_multispecies_e126_ssl_h5_audit.sh \
  --input-h5 /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/datasets/e126_ssl_e16_low_bgall_target3000_20260612T031656Z.h5 \
  --builder-summary-json /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/datasets/e126_ssl_e16_low_bgall_target3000_20260612T031656Z.summary.json \
  --output-dir /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/pipeline_runs/e126_ssl_h5_audit_20260612T031656Z \
  --ledger-path docs/multispecies_experiment_results.md \
  --min-normal-rows 10000 \
  --min-normal-months 12
```

If the audit is queued before the H5 job has finished, add
`--dependency afterok:H5_JOB_ID --allow-missing-h5`.

## Evaluation

After fine-tuning, score the model on the same ONC holdout used by E26 where possible. The report should include:

- macro and weighted F1, per-species precision/recall/F1, and support
- full confusion matrix including background/normal
- cross-species false positives, especially `Bp -> Bm/Mn` and `Bm/Mn -> Bp`
- threshold or confidence analysis if the model exposes calibrated probabilities
- examples/contact sheets for true positives, false positives, false negatives, and cross-species errors
