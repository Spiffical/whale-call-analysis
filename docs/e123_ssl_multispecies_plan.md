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

## Launch Sketch

```bash
bash drac/scripts/submit_multispecies_e123_ssl_ssamba.sh \
  --ssl-repo-root /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/selfsupervision_anomalies_onc \
  --dataset-h5 /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/datasets/e123_multispecies_ssl_onc.h5 \
  --num-pretrain-jobs 2 \
  --num-finetune-jobs 1 \
  --time 03:00:00 \
  --gres gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
```

## Evaluation

After fine-tuning, score the model on the same ONC holdout used by E26 where possible. The report should include:

- macro and weighted F1, per-species precision/recall/F1, and support
- full confusion matrix including background/normal
- cross-species false positives, especially `Bp -> Bm/Mn` and `Bm/Mn -> Bp`
- threshold or confidence analysis if the model exposes calibrated probabilities
- examples/contact sheets for true positives, false positives, false negatives, and cross-species errors
