import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference.transform_predictions_to_o3 import transform_to_o3


def test_transform_splits_multi_source_event_items():
    payload = {
        "schema_version": "2.1",
        "created_at": "2026-04-28T00:00:00+00:00",
        "task_type": "whale_detection",
        "data_sources": [
            {
                "data_source_id": "ICLISTENHF6016_2025-04",
                "device_code": "ICLISTENHF6016",
            }
        ],
        "items": [
            {
                "item_id": "event-001",
                "data_source_id": "ICLISTENHF6016_2025-04",
                "audio_start_time": "2025-04-01T02:24:51+00:00",
                "audio_end_time": "2025-04-01T02:25:09+00:00",
                "model_outputs": [
                    {
                        "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
                        "score": 0.97,
                        "aggregation_method": "event_max",
                        "metadata": {"not_allowed": True},
                    }
                ],
                "source_segments": [
                    {
                        "source_audio": "ICLISTENHF6016_20250401T022000.000Z.flac",
                        "time_start_sec": 1743474291.4,
                        "time_end_sec": 1743474301.0,
                        "score": 0.97,
                    },
                    {
                        "source_audio": "ICLISTENHF6016_20250401T022500.000Z.flac",
                        "time_start_sec": 1743474300.0,
                        "time_end_sec": 1743474309.6,
                        "score": 0.83,
                    },
                ],
                "paths": {
                    "spectrogram_mat_path": "predictions_postprocessed_events_media/spectrograms/event-001.mat",
                    "audio_path": "predictions_postprocessed_events_media/audio/event-001.wav",
                },
            }
        ],
    }

    transformed = transform_to_o3(payload)

    assert len(transformed["items"]) == 2
    assert transformed["items"][0]["item_id"].startswith("event-001__source_01")
    assert transformed["items"][1]["item_id"].startswith("event-001__source_02")
    assert transformed["items"][0]["source_audio"]["file_name"] == "ICLISTENHF6016_20250401T022000.000Z.flac"
    assert transformed["items"][1]["source_audio"]["file_name"] == "ICLISTENHF6016_20250401T022500.000Z.flac"
    assert transformed["items"][0]["model_outputs"][0] == {
        "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
        "score": 0.97,
    }
    assert transformed["items"][1]["model_outputs"][0]["score"] == 0.83
    assert "aggregation_method" not in transformed["items"][0]["model_outputs"][0]
    assert "metadata" not in transformed["items"][0]["model_outputs"][0]
    assert transformed["items"][0]["paths"] == transformed["items"][1]["paths"]


def test_transform_adds_single_source_audio_without_splitting():
    payload = {
        "schema_version": "2.1",
        "created_at": "2026-04-28T00:00:00+00:00",
        "task_type": "whale_detection",
        "items": [
            {
                "item_id": "event-002",
                "model_outputs": [{"class_hierarchy": "Fin whale", "score": 0.9}],
                "source_segments": [
                    {
                        "source_audio": "ICLISTENHF6016_20250401T022000.000Z.flac",
                        "time_start_sec": 12.0,
                        "time_end_sec": 20.0,
                        "score": 0.9,
                    }
                ],
            }
        ],
    }

    transformed = transform_to_o3(payload)

    assert len(transformed["items"]) == 1
    item = transformed["items"][0]
    assert item["item_id"] == "event-002"
    assert item["source_audio"]["file_name"] == "ICLISTENHF6016_20250401T022000.000Z.flac"
    assert item["audio_start_time"] == "2025-04-01T02:20:12+00:00"
    assert item["audio_end_time"] == "2025-04-01T02:20:20+00:00"
