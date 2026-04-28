# Finwhale Output Contract: Verification App + O3.0

This document shows a practical output pattern that stays compatible with:

1. `../labeling-verification-app` (for expert review)
2. O3.0 unified schema ingestion (`schema_version: "2.1"`)

## Recommended Artifacts

Generate two JSON artifacts from one inference run:

1. `predictions_postprocessed.app.json`
2. `predictions_postprocessed.o3.json`
3. `predictions.json` as a copy of the strict O3 JSON

Use the first file when you need rich review/event lineage. Use the second file,
or canonical `predictions.json`, for strict O3.0 ingest.

Reason: the app can use richer event metadata (`events`, merged media lineage), while strict O3.0 schema has `additionalProperties: false` and rejects extra root/item/model-output fields.

## Example Directory Layout

```text
2018-07-01/
  ICLISTENHF1353/
    full_spectrograms/
    predictions.json                 # strict O3-compatible copy
    predictions_postprocessed.app.json
    predictions_postprocessed.o3.json
    predictions_postprocessed_events_media/
      spectrograms/
        evt_000001.mat
      audio/
        evt_000001.wav
```

## 1) App-Facing JSON (Extended)

This supports concatenated/clustered event review and keeps lineage back to original sliding windows.

```json
{
  "schema_version": "2.1",
  "created_at": "2026-02-12T16:12:15.190407+00:00",
  "updated_at": "2026-02-12T16:12:15.190407+00:00",
  "task_type": "whale_detection",
  "model": {
    "model_id": "sha256-f2d26d93fac1",
    "architecture": "resnet18"
  },
  "data_sources": [
    {
      "data_source_id": "ICLISTENHF1353_2018-07-01",
      "device_code": "ICLISTENHF1353"
    }
  ],
  "items": [
    {
      "item_id": "fw-ICLISTENHF1353-20180701T002740000Z-20180701T002757700Z-gb9f63f34",
      "data_source_id": "ICLISTENHF1353_2018-07-01",
      "audio_start_time": "2018-07-01T00:27:40+00:00",
      "audio_end_time": "2018-07-01T00:27:57.700000+00:00",
      "model_outputs": [
        {
          "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
          "score": 0.9983,
          "aggregation_method": "event_max",
          "metadata": {
            "event_mean_score": 0.9493,
            "event_n_members": 4,
            "event_n_high": 4,
            "parent_source_audio_files": [
              "ICLISTENHF1353_20180701T002558.726Z.wav"
            ]
          }
        }
      ],
      "paths": {
        "spectrogram_mat_path": "predictions_postprocessed_events_media/spectrograms/evt_000001.mat",
        "audio_path": "predictions_postprocessed_events_media/audio/evt_000001.wav"
      },
      "verifications": []
    }
  ],
  "events": [
    {
      "event_id": "fw-ICLISTENHF1353-20180701T002740000Z-20180701T002757700Z-gb9f63f34",
      "group": "ICLISTENHF1353_20180701T002558.726Z.wav",
      "start_sec": 102.0,
      "end_sec": 119.7,
      "duration_sec": 17.7,
      "max_score": 0.9983,
      "mean_score": 0.9493,
      "n_members": 4,
      "n_high": 4,
      "member_item_ids": [
        "ICLISTENHF1353_20180701T002558.726Z_win1025",
        "ICLISTENHF1353_20180701T002558.726Z_win1049"
      ],
      "parent_source_audio_files": [
        "ICLISTENHF1353_20180701T002558.726Z.wav"
      ],
      "paths": {
        "spectrogram_mat_path": "predictions_postprocessed_events_media/spectrograms/evt_000001.mat",
        "audio_path": "predictions_postprocessed_events_media/audio/evt_000001.wav"
      }
    }
  ],
  "postprocessing": {
    "method": "temporal_cluster_hysteresis_v1",
    "low_threshold": 0.70,
    "high_threshold": 0.82,
    "min_members": 2,
    "max_gap_seconds": 15.0
  }
}
```

## 2) O3.0 Ingest JSON (Strict)

This version should only contain fields allowed by `OCEANS3_JSON_SCHEMA.md`.
If an app-facing event spans multiple raw source files, split it into one strict
O3 item per source file. Keep the app event id as the prefix and suffix the
strict `item_id` with source information. The strict items may point to the same
merged review media in `paths`, but each item should have its own
`source_audio.file_name`, `audio_start_time`, `audio_end_time`, and per-source
score.

```json
{
  "schema_version": "2.1",
  "created_at": "2026-02-12T16:12:15.190407+00:00",
  "updated_at": "2026-02-12T18:20:00.000000+00:00",
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
      "data_source_id": "ICLISTENHF1353_2018-07-01",
      "device_code": "ICLISTENHF1353"
    }
  ],
  "items": [
    {
      "item_id": "fw-ICLISTENHF1353-20180701T002740000Z-20180701T002757700Z-gb9f63f34__source_01_ICLISTENHF1353-20180701T002500-000Z",
      "data_source_id": "ICLISTENHF1353_2018-07-01",
      "audio_start_time": "2018-07-01T00:27:40+00:00",
      "audio_end_time": "2018-07-01T00:27:57.700000+00:00",
      "model_outputs": [
        {
          "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
          "score": 0.9983
        }
      ],
      "source_audio": {
        "file_name": "ICLISTENHF1353_20180701T002500.000Z.wav",
        "format": "wav",
        "recording_start_time": "2018-07-01T00:25:00+00:00",
        "recording_end_time": "2018-07-01T00:30:00+00:00"
      },
      "verifications": [
        {
          "verified_at": "2026-02-13T02:10:00Z",
          "verified_by": "expert@onc.ca",
          "verification_round": 1,
          "verification_status": "verified",
          "label_decisions": [
            {
              "label": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
              "decision": "accepted",
              "threshold_used": 0.82
            }
          ],
          "confidence": "high",
          "label_source": "expert",
          "notes": "Clear event across overlapping windows"
        }
      ],
      "paths": {
        "spectrogram_mat_path": "predictions_postprocessed_events_media/spectrograms/evt_000001.mat",
        "audio_path": "predictions_postprocessed_events_media/audio/evt_000001.wav"
      }
    }
  ]
}
```

## How This Meets O3 Query Needs

For each predicted call/event, O3 can query:

1. Which file/time region contains the call
   - `data_sources[].device_code`
   - `items[].audio_start_time`
   - `items[].audio_end_time`
2. What call type was predicted/verified
   - `items[].model_outputs[].class_hierarchy`
   - `items[].verifications[].label_decisions[]`
3. Where to open media for inspection
   - `items[].paths.spectrogram_mat_path`
   - `items[].paths.audio_path`

## Verification App Compatibility

The app accepts unified v2 JSON with relative paths and writes verifications back into `items[].verifications[]`.

Keep these stable during the review cycle:

1. `item_id`
2. `paths.*`
3. `audio_start_time`/`audio_end_time`

## If You Need Event Lineage Inside O3 JSON

Do not put event lineage inside strict O3 JSON unless the schema is formally
changed. Current strict schema rejects custom fields (`additionalProperties:
false`). Keep lineage in `predictions_postprocessed.app.json`, especially:

1. root `events[]`
2. root `postprocessing`
3. item `source_segments`
4. item `parent_source_audio_files`
5. model output `metadata`

## Optional Transform: Extended -> Strict O3

If your pipeline emits extended fields, strip them before ingest:

```bash
jq '
  del(.events, .postprocessing)
  | .items |= map(
      . as $i
      | {
          item_id: $i.item_id,
          data_source_id: $i.data_source_id,
          audio_start_time: $i.audio_start_time,
          audio_end_time: $i.audio_end_time,
          model_outputs: (($i.model_outputs // []) | map({class_hierarchy, class_id, score})),
          verifications: ($i.verifications // []),
          source_audio: $i.source_audio,
          paths: ($i.paths // {})
        }
      | with_entries(select(.value != null))
    )
' predictions_postprocessed.app.json > predictions_postprocessed.o3.json
```

That gives you:

1. Rich review JSON for experts
2. Strict O3.0 JSON for ingestion
3. Consistent `item_id` across both artifacts for traceability
