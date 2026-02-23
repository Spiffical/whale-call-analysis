# Finwhale Output Contract: Verification App + O3.0

This document shows a practical output pattern that stays compatible with:

1. `../labeling-verification-app` (for expert review)
2. O3.0 unified schema ingestion (`schema_version: "2.1"`)

## Recommended Artifacts

Generate two JSON artifacts from one inference run:

1. `predictions_postprocessed.app.json`
2. `predictions_postprocessed.o3.json`

Use the first file in the verification app. Use the second file for strict O3.0 ingest.

Reason: the app can use richer event metadata (`events`, merged media lineage), while strict O3.0 schema has `additionalProperties: false` and rejects extra root/item/model-output fields.

## Example Directory Layout

```text
2018-07-01/
  ICLISTENHF1353/
    full_spectrograms/
    predictions.json
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
      "item_id": "fw-ICLISTENHF1353-20180701T002740000Z-20180701T002757700Z-gb9f63f34",
      "data_source_id": "ICLISTENHF1353_2018-07-01",
      "audio_start_time": "2018-07-01T00:27:40+00:00",
      "audio_end_time": "2018-07-01T00:27:57.700000+00:00",
      "model_outputs": [
        {
          "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
          "score": 0.9983
        }
      ],
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

Strict schema currently rejects custom fields (`additionalProperties: false`).

Minimal schema change that preserves strictness:

1. Add optional root `extensions` object (`additionalProperties: true`)
2. Add optional item `extensions` object (`additionalProperties: true`)
3. Add optional model_output `extensions` object (`additionalProperties: true`)

Then store event lineage under:

```json
"extensions": {
  "finwhale": {
    "event_id": "fw-ICLISTENHF1353-20180701T002740000Z-20180701T002757700Z-gb9f63f34",
    "parent_source_audio_files": ["ICLISTENHF1353_20180701T002558.726Z.wav"],
    "source_segments": ["...window ids..."]
  }
}
```

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
          segment_index: $i.segment_index,
          model_outputs: (($i.model_outputs // []) | map({class_hierarchy, class_id, score})),
          verifications: ($i.verifications // []),
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
