# Fin-Whale BBox Pipeline V1

## Scope

This pipeline is the first dedicated bounding-box path for fin-whale localization in spectrograms.
It replaces the earlier CAM-localization experiment as the main localization track.

V1 decisions:

- Joint training over historical `2018/2019` plus `2025`
- Canonical 2025 row-level bbox source: `Clayoquot_2025_SpeciesTemporalAnalysis.xlsx`
- Pure-negative 2025 listened/analyzed inventory: `Clayoquot_2025_Analysis_Mar26_Final.xlsx`
- `Clayoquot_2025_annotations_Mar18.xlsx` is used only as a guardrail to avoid mining false pure negatives
- `Clayoquot_2025_analysis_Mar9.xlsx` is not used
- First detector is single-class `fin_call` with YOLO26
- Unified manifest keeps all species so the same source can later support multispecies detection

## Unified Annotation Schema

The long-lived source of truth is `unified_annotations.csv`.

One row equals one annotation event with:

- provenance: source dataset, workbook, sheet, row index
- clip identity: filename, device code, clip start, recording day
- species and fin subtype: `species_code`, `call_type_raw`, `call_type_std`
- box coordinates in clip space: begin/end time and low/high frequency
- QC fields: `timestamp_fix`, comments, tags, vessel flag, annotator

Fin subtype normalization is intentionally conservative:

- `20 Hz`, `20Hz`, `20 HZ` -> `20Hz`
- `30 Hz`, `30Hz`, clear 30 Hz variants -> `30Hz`
- `40 Hz`, `40Hz` -> `40Hz`
- `S`, `song` -> `song`
- all other fin labels -> `other_fin`

## Timestamp Repair

Historical timestamp artifacts occasionally appear in `[300, 600)` seconds.

Repair rules:

- if both begin/end are in `[300, 600)`, subtract `300`
- if begin `< 300` and end is in `(300, 600]`, clip end to `300`

This issue is rare and localized:

- `47` rows repaired with `minus_300s`
- `1` row repaired with `clip_end_to_300s`
- `108875` rows already clean in the audited workbook

## Split Strategy

Splits are clip-level and leakage-safe by day.

Grouping key:

- `source_dataset + recording_day_utc`

Assignments:

- historical annotated days: `80/10/10` into `train`, `val_hist`, `test_hist`
- 2025 annotated days: `70/15/15` into `train`, `val_2025`, `test_2025`
- pure-negative 2025 clips from Mar26 inherit the annotated split when they share a day with annotated 2025 data
- pure-negative-only 2025 days are assigned separately with the same `70/15/15` logic by clip count

Model-selection views:

- primary: `val_2025`
- retention guardrail: `val_hist`
- final reporting: `test_2025`, `test_hist`

## Export Strategy

V1 export is fin-focused rather than multispecies-ready.

Parameters:

- spectrogram frequency band: `1-200 Hz`
- saved contexts: `40 s`
- detector crops: `10 s`
- train-time decentering controlled by `center_bias_sigma_frac=0.25`

Negative sources:

- gap negatives from annotated clips with a `2 s` exclusion margin
- annotated non-fin contexts centered on other species events, exported as background-only for V1 COCO
- pure-zero 2025 negatives from Mar26 `verified=1` rows with no species flags and no overlap with SpeciesTemporal or Mar18 filenames

COCO V1 output:

- one class: `fin_call`
- other-species events remain available in the unified manifest and context metadata, but are not emitted as COCO labels yet

## Joint Training Recommendation

V1 should start with one joint YOLO26 run over historical and 2025 data, not historical pretrain followed by 2025 finetuning.

Reasons:

- 2025 is large enough to contribute directly rather than acting like a tiny adaptation set
- staged finetuning adds immediate forgetting risk unless we also add rehearsal or mixed replay
- the split design already provides a clean 2025 validation/test view plus historical retention monitoring
- Ultralytics' current published COCO metrics put YOLO26 ahead of the RT-DETR weights they document, so YOLO26 is the stronger default starting point for the first detector pass

RT-DETR remains implemented in this repo as the main baseline comparator.
If the first YOLO26 run underperforms on our spectrogram boxes, the first comparison should be that retained RT-DETR path.
If YOLO26 has good recall but consistently loose boxes, the next comparison should be D-FINE.
Segmentation remains out of scope until mask-like supervision exists.

## Output Paths

Default durable paths:

- manifests: `output/finwhale_bbox/manifests/joint_v1/`
- splits: `output/finwhale_bbox/splits/joint_v1/`
- exported detector-ready COCO data: `output/finwhale_bbox/exports/fin_1to200_detector_v1/`
- exported YOLO26 dataset: `output/finwhale_bbox/exports/fin_1to200_yolo26_v1/`
- DRAC/Nibi training runs: `${SCRATCH:-/scratch/$USER}/whale-call-analysis/finwhale_bbox_runs/`

The cluster job stages export and training under `$SLURM_TMPDIR` and copies back only:

- manifests and split metadata
- export manifests, COCO JSON, and small QC images
- model checkpoints
- train/eval summaries and prediction JSON
