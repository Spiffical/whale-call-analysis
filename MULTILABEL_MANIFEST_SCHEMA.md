# Multi-Label Manifest Schema

This is the v1 experiment manifest used by `src/dataset/multilabel.py`. It is intentionally CSV-friendly for training and audit scripts, while preserving label structure in `labels_json`.

## Window Rows

One row represents one audio/spectrogram window.

Required for training:

- `item_id`: stable window id
- `mat_path` or `spectrogram_path`: MAT spectrogram path
- `label_ids`: pipe-delimited trainable labels, for example `species:Bp|call:20Hz`

Recommended:

- `source_audio`: source audio filename or path
- `device`: hydrophone/device code
- `start_time`: ISO-8601 window start
- `end_time`: ISO-8601 window end
- `window_start_s`: window offset in source audio
- `duration_s`: window duration
- `source_dataset`: manifest/workbook/dataset provenance
- `review_status`: `reviewed` or `unreviewed`
- `event_group`: grouping key for leakage-safe splits
- `is_background`: `1` when no trainable positive label is present
- `split`: `train`, `val`, or `test` after split generation

## Labels

`labels_json` stores the full per-label records. Example:

```json
[
  {
    "species_code": "Bp",
    "species": "Fin whale",
    "call_type": null,
    "source": "manifest",
    "review_status": "unreviewed",
    "trainable": true
  },
  {
    "species_code": null,
    "species": null,
    "call_type": "20Hz",
    "call_type_name": "20 Hz pulse",
    "source": "manifest",
    "review_status": "unreviewed",
    "trainable": true
  }
]
```

Optional future fields:

- `confidence`
- `annotator`
- `annotation_extent`
- `notes`

## Vocabulary

`label_vocabulary.json` contains one object per trainable output:

- `id`: stable output id, for example `species:Bp`
- `group`: `species` or `call_type`
- `code`: compact label code
- `name`: display name
- `class_hierarchy`: OCEANS3-compatible taxonomy path
- `count`: source manifest count used to build the vocabulary

## OCEANS3 Mapping

For model predictions, export one `items[].model_outputs[]` record per vocabulary label:

```json
{
  "label_id": "species:Bp",
  "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
  "score": 0.91,
  "threshold": 0.5
}
```

For strict OCEANS3 ingestion, human labels should map to `items[].verifications[].label_decisions[]`. The smoke trainer writes `validation_predictions.o3_compatible.json` as a minimal score-export check; full review-package export should continue through the existing OCEANS3 transformation path once multi-label inference is promoted.
