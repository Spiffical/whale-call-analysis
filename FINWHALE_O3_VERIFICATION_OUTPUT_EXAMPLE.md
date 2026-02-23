# Fin Whale Predictions: O3.0 + Verification App Output Example

## Goal
This document shows a concrete JSON output pattern that is:
- compatible with the labeling verification app
- suitable for O3.0 ingestion/querying (find calls, source files, and approximate call time in 5-minute audio)

It uses the unified predictions format (`schema_version: "2.1"`), with event-level items created by postprocessing.

---

## Recommended Pipeline Output
Use postprocessing with:
- merged event media
- one item per event
- optional cross-5min grouping

Example inference/postprocess flags:

```bash
python scripts/inference/run_inference.py \
  --mat-dir "$OUT_DIR/2018-07-01/ICLISTENHF1353/full_spectrograms" \
  --checkpoint "$CKPT" \
  --dataset-metadata "$OUT_DIR/metadata.json" \
  --output-json "$OUT_DIR/2018-07-01/ICLISTENHF1353/predictions.json" \
  --sliding-window \
  --window-step 24 \
  --export-crops \
  --export-threshold 0.70 \
  --raw-audio-dir "$OUT_DIR/raw_audio" \
  --device cuda \
  --postprocess \
  --postprocess-low-threshold 0.70 \
  --postprocess-high-threshold 0.82 \
  --postprocess-min-members 2 \
  --postprocess-max-gap-seconds 15 \
  --postprocess-merge-event-media \
  --postprocess-replace-items-with-events \
  --postprocess-merge-min-score 0.80 \
  --postprocess-merge-across-source-audio
```

---

## Example (Pre-Verification Event Item)

This is the recommended event-level item shape for expert review:

```json
{
  "schema_version": "2.1",
  "task_type": "whale_detection",
  "model": {
    "model_id": "sha256-f2d26d93fac1",
    "architecture": "resnet18",
    "output_classes": [
      "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"
    ]
  },
  "data_sources": [
    {
      "data_source_id": "ICLISTENHF1353",
      "device_code": "ICLISTENHF1353",
      "date_from": "2018-07-01T00:00:00Z",
      "date_to": "2018-07-02T00:00:00Z"
    }
  ],
  "items": [
    {
      "item_id": "evt_000001",
      "data_source_id": "ICLISTENHF1353",
      "audio_start_time": "2018-07-01T00:05:46+00:00",
      "audio_end_time": "2018-07-01T00:06:08.500000+00:00",
      "model_outputs": [
        {
          "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
          "score": 0.91,
          "aggregation_method": "event_max",
          "metadata": {
            "event_mean_score": 0.86,
            "event_n_members": 3,
            "event_n_high": 2,
            "parent_source_audio_files": [
              "ICLISTENHF1353_20180701T000558.726Z.wav",
              "ICLISTENHF1353_20180701T001058.726Z.wav"
            ],
            "windows": [
              {
                "window_id": 0,
                "source_item_id": "ICLISTENHF1353_20180701T000558.726Z_win2885",
                "source_audio": "ICLISTENHF1353_20180701T000558.726Z.wav",
                "time_start_sec": 288.0,
                "time_end_sec": 298.5,
                "score": 0.81,
                "window_indices": [2885, 2981],
                "audio_path": "audio/ICLISTENHF1353_20180701T000558.726Z_win2885.wav",
                "spectrogram_mat_path": "spectrograms/ICLISTENHF1353_20180701T000558.726Z_win2885.mat"
              },
              {
                "window_id": 1,
                "source_item_id": "ICLISTENHF1353_20180701T001058.726Z_win0",
                "source_audio": "ICLISTENHF1353_20180701T001058.726Z.wav",
                "time_start_sec": 0.0,
                "time_end_sec": 10.5,
                "score": 0.91,
                "window_indices": [0, 96],
                "audio_path": "audio/ICLISTENHF1353_20180701T001058.726Z_win0.wav",
                "spectrogram_mat_path": "spectrograms/ICLISTENHF1353_20180701T001058.726Z_win0.mat"
              }
            ]
          }
        }
      ],
      "verifications": [],
      "paths": {
        "spectrogram_mat_path": "predictions_postprocessed_events_media/spectrograms/evt_000001.mat",
        "audio_path": "predictions_postprocessed_events_media/audio/evt_000001.wav"
      }
    }
  ]
}
```

---

## Example (After Expert Verification)

The verification app should append to `items[].verifications[]`:

```json
{
  "item_id": "evt_000001",
  "verifications": [
    {
      "verified_at": "2026-02-12T21:45:00Z",
      "verified_by": "expert@onc.ca",
      "verification_round": 1,
      "verification_status": "verified",
      "label_decisions": [
        {
          "label": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
          "decision": "accepted",
          "threshold_used": 0.7
        },
        {
          "label": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale > 20Hz pulse",
          "decision": "added",
          "threshold_used": null
        }
      ],
      "label_source": "expert",
      "notes": "Strong fin whale pulse train across file boundary."
    }
  ]
}
```

This is the key requirement for O3.0 downstream curation: model output + expert decisions are preserved in one record.

---

## O3.0 Query Mapping

For O3.0 queries like:
- "show fin whale calls"
- "which files contain calls"
- "where in the 5-minute file is the call"

Use:
- call presence/type:
  - `items[].model_outputs[].class_hierarchy`
  - `items[].verifications[].label_decisions[]`
- confidence:
  - `items[].model_outputs[].score`
- source hydrophone/date:
  - `items[].data_source_id` -> `data_sources[]`
- event absolute time bounds:
  - `items[].audio_start_time`, `items[].audio_end_time`
- source 5-minute file(s) and approximate offsets:
  - `items[].model_outputs[].metadata.parent_source_audio_files`
  - `items[].model_outputs[].metadata.windows[].source_audio`
  - `items[].model_outputs[].metadata.windows[].time_start_sec`
  - `items[].model_outputs[].metadata.windows[].time_end_sec`

---

## Strict Ingestion Profile (If Needed)

Current postprocessed output may include extra root keys (for diagnostics), e.g. `events`, `postprocessing`.

If O3.0 ingestion requires a strict subset, create an ingest-specific file:

```bash
jq '{
  schema_version,
  created_at,
  updated_at,
  task_type,
  model,
  data_sources,
  spectrogram_config,
  pipeline,
  items: [.items[] | {
    item_id,
    data_source_id,
    audio_start_time,
    audio_end_time,
    segment_index,
    model_outputs,
    verifications,
    paths
  }]
}' predictions_postprocessed.json > predictions_o3_ingest.json
```

This preserves event-level whale-call information and expert verifications while dropping non-core root diagnostics.

---

## Minimal Schema Edits (Only If Required)

If the O3.0 validator is strict and rejects the above metadata, minimal useful extensions are:
- allow `items[].model_outputs[].aggregation_method` (string)
- allow `items[].model_outputs[].metadata` (object)
- optionally allow root `postprocessing` and `events` as optional diagnostic blocks

These extensions are enough to support event-level merged outputs and source-window traceability without changing core ingestion logic.
