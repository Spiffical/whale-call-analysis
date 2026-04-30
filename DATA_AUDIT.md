# Data Audit: Multi-Species + Multi-Call-Type

## Scope

This first audit is local and read-only. It covers the merged final2025-style manifests, local Part 2 smoke/holdout bundles, and a separate check of `tmp/audio_audit_run/manifests`. It does not claim a fully deduplicated canonical dataset yet; the local bundles overlap the final manifests and are useful mostly because they include MATs for smoke tests.

Generated audit artifacts are under:

- `output/multispecies_data_audit_local/`
- `output/multispecies_audio_audit_run_only/`

The preferred reviewed experiment root for later Nibi work remains:

`/project/6070467/merileo/data/finwhales/final2025_resnet_20260423/multispecies_calltype_experiments/`

## Current Pipeline Inventory

- Final2025 ResNet training lives in `scripts/train/train_cnn.py`, `src/training/mat_dataset.py`, and `src/models/fin_models.py`.
- Nibi launch orchestration lives in `drac/scripts/launch_finwhale_final2025_resnet_benchmark.sh`.
- Annotation and Part 2 manifest parsing lives in `src/dataset/part2_annotations.py`, `src/dataset/part2_finetune.py`, and `scripts/data/part2/*`.
- Inference and review export live in `scripts/inference/run_inference.py`, `scripts/inference/postprocess_predictions.py`, `scripts/inference/evaluate_part2_predictions.py`, and `scripts/inference/transform_predictions_to_o3.py`.
- Current training is binary fin-whale detection: `num_classes=2`, softmax/argmax metrics, `CrossEntropyLoss`, positive/negative MAT directories, and optional `WeightedRandomSampler`.
- Current annotation manifests already contain multi-label fields such as pipe-delimited `species_codes` and `fin_call_type_stds`, but the training target collapses that to fin-positive vs negative.

## Primary Local Audit

Sources:

- `/home/sbialek/ONC/whale-call-analysis/tmp/final2025_manifest_check/unified_annotations.csv`
- `/home/sbialek/ONC/whale-call-analysis/tmp/final2025_manifest_check/clip_manifest.csv`
- `/home/sbialek/ONC/whale-call-analysis/data/finwhale_part2_smoke_bundle`
- `/home/sbialek/ONC/whale-call-analysis/data/cleanholdout/bundle`

Source-row totals:

- Annotation rows: 154,259
- Clip manifest rows: 6,690
- Multi-species clips: 845
- Multi-call-type clips: 1,536
- Candidate MAT smoke rows: 222
- Candidate background rows with MATs: 0
- Missing MAT records in the local bundles: 0
- Duplicate annotation keys: 617 keys, 1,234 rows involved
- Review status: all 154,259 rows are currently represented as `unreviewed`

The source-row inventory is intentionally not deduplicated. The local bundle annotation CSVs are copied/subset forms of the broader manifest lineage, and `duplicate_annotation_keys.csv` flags exact repeated `(filename, species, call_type, begin_time, end_time)` rows.

## Labels Observed

Top species/source labels:

- `Bp` fin whale: 150,095
- `OD` odontocete: 1,858
- `Mn` humpback whale: 1,739
- `Bm` blue whale: 157
- `INSTRUMENT`: 96
- `Oo` killer whale: 88
- `UN` unknown cetacean: 82
- `CE` cetacean: 34
- `MA`: 25
- `Bb` sei whale: 22
- `EQ`: 14
- `SONAR`: 14
- Rare labels under or near threshold: `P`, `BA`, `UNKNOWN`, `Pm`, `Lo`

Top call types:

- `20Hz`: 124,387
- `40Hz`: 12,494
- `other_fin`: 7,051
- `30Hz`: 4,595
- `song`: 2,704
- `CK`: 1,447
- `<blank>`: 733
- `BP`: 370
- `B`: 140
- `W`: 120
- `hydrophone_thud`: 96
- Rare or non-v1 call labels include `BZ`, `downsweep`, `A`, `earthquake`, `upsweep`, `unknown`, `D`, `tone`, and `BZ, CK`.

Top species-call-type pairs:

- `Bp::20Hz`: 124,386
- `Bp::40Hz`: 12,493
- `Bp::other_fin`: 7,051
- `Bp::30Hz`: 4,595
- `Bp::song`: 1,570
- `OD::CK`: 1,353
- `Mn::song`: 1,134
- `Mn::<blank>`: 605
- `OD::BP`: 365
- `Bm::B`: 140

## Time, Device, And Source Distribution

Years:

- 2018: 43,349 rows
- 2019: 65,547 rows
- 2025: 45,362 rows
- 4019: 1 malformed timestamp row requiring cleanup

Devices:

- `ICLISTENHF1353`: 108,892 rows
- `ICLISTENHF6016`: 45,362 rows
- Five malformed or partial device-like strings occur once each.

Source datasets:

- `historical_2018_2019`: 108,897 rows
- `clayoquot_2025_final_annotations`: 34,612 rows
- local cleanholdout bundle annotations: 10,206 rows
- local smoke bundle annotations: 544 rows

This creates a major device/time confound: historical rows are overwhelmingly `ICLISTENHF1353`, while 2025 rows are `ICLISTENHF6016`.

## Separate Audio Audit Manifest Check

`tmp/audio_audit_run/manifests` is overlapping but not identical to `tmp/final2025_manifest_check`.

- Annotation rows: 136,800
- Clip manifest rows: 6,135
- Multi-species clips: 510
- Multi-call-type clips: 1,411
- Duplicate annotation keys: 454 keys, 908 rows involved
- Top species: `Bp`, `OD`, `Mn`, `Bm`, `Bb`, `Pm`
- Top call types: `20Hz`, `40Hz`, `other_fin`, `30Hz`, `song`, `CK`

This source should be treated as an audit comparison input, not simply concatenated into training data until a canonical deduplication pass decides which manifest version wins.

## Media And Windowing

- The current ResNet training pipeline loads precomputed `.mat` spectrograms through `src/training/mat_dataset.py`.
- MAT loading supports power or dB-like arrays, frequency/time axes, optional frequency cropping, time cropping, and positive-window crop augmentation.
- Local candidate smoke rows point to 300 second MAT windows from the smoke and cleanholdout bundles.
- The primary local bundle media check found no missing MATs among the 222 candidate rows.
- Full final2025 manifest audio/MAT availability on whalestor/Nibi remains a later large-file audit step.

## Main Risks

- Multi-label pressure is real: 845 clips have multiple species and 1,536 have multiple fin call types in the primary local audit.
- Absence of a label is not always proof of absence for all species/call types, especially in manifests created for fin-whale work.
- `ICLISTENHF1353` vs `ICLISTENHF6016` is entangled with year/source dataset.
- Some labels have too few examples for trainable v1 outputs.
- Exact duplicate annotations and duplicated local bundle rows must be removed or grouped before full training.
- A malformed `4019-02` month should be corrected or excluded before split generation.
- Candidate smoke MATs currently have no background rows, so smoke training validates mechanics but not background calibration.
