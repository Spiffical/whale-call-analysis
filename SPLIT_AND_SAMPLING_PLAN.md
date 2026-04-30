# Split And Sampling Plan

## Split Policy

Do not use random clip-level splits. Build splits from event/source groups first, then add temporal and device/site holdouts as the data volume allows.

V1 grouping key:

- explicit `event_group` when present
- otherwise `source_audio`
- otherwise `filename` or `item_id`

All adjacent windows from the same event/source group must stay in one split. Later full-data split generation should also cluster events from the same source file within a configurable time gap.

## Candidate Local Split

Generated from `output/multispecies_data_audit_local/candidate_multilabel_manifest.csv` into:

`output/multispecies_data_audit_local/splits/`

Candidate split summary:

- Train: 156 rows, 151 groups
- Validation: 33 rows, 32 groups
- Test: 33 rows, 33 groups
- Group leakage: 0 groups shared across splits
- Background rows: 0 in all splits

This split is for smoke testing only. It is not a scientific evaluation split because it is small, duplicated across local bundles in places, and lacks background negatives.

## Rare Labels

Preserve rare labels in validation/test where possible. If a rare label cannot appear in validation/test without creating leakage or unstable metrics, mark it train-only and exclude it from macro aggregates.

For full training, set a minimum clean-positive threshold after deduplication. Labels below threshold should remain in the manifest and review workflow but not be first-class trainable outputs.

## Temporal And Device Holdouts

Preferred full split progression:

1. Event/source-group split with no leakage.
2. Temporal holdout by day/week/month.
3. Device/site holdout if there are enough examples outside the dominant devices.
4. Combined stress test by holding out a later month and reporting device/source stratified metrics.

The current audit shows strong device/time confounding:

- historical 2018/2019 rows are mostly `ICLISTENHF1353`
- 2025 rows are mostly `ICLISTENHF6016`

Do not interpret device-holdout results as purely device generalization until source/year effects are separated.

## Sampling

Training batches should use imbalance-aware sampling after the canonical manifest is deduplicated:

- cap background/negative rows to a documented ratio, starting around 1:1 to 3:1 versus positive windows
- oversample rare trainable labels with replacement
- avoid letting `Bp` + `20Hz` dominate every batch
- keep duplicated or adjacent event windows grouped so oversampling does not leak into validation/test

Background is represented as "no positive trainable labels present," not as a mutually exclusive softmax class.

## Experiment Sequence

Run in this order:

1. Local audit and schema checks.
2. Candidate grouped split build.
3. Tiny local smoke training on a few MATs.
4. Nibi smoke job only after these docs are reviewed.
5. Full fine-tuning only after deduplication, label thresholding, and split design are approved.

Suggested Nibi job names for later:

- `multispecies_data_audit`
- `multispecies_split_build`
- `multispecies_resnet_finetune_smoke`
- `multispecies_resnet_finetune_full`
- `multispecies_inference_eval`

Before any Nibi launch, check:

```bash
squeue -u merileo
sacct -u merileo --starttime now-7days
```

No expensive Nibi job was launched in this phase.
