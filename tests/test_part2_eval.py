import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.part2_eval import (
    build_clip_confusion,
    load_annotations_csv,
    load_clip_manifest_csv,
    load_prediction_segments,
    match_predictions_to_annotations,
)


def _write_csv(path: Path, fieldnames, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestPart2Eval(unittest.TestCase):
    def test_matching_and_clip_confusion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            annotations_csv = tmp / "fin_annotations.csv"
            clip_manifest_csv = tmp / "clip_manifest.csv"
            predictions_json = tmp / "predictions_postprocessed.json"

            _write_csv(
                annotations_csv,
                [
                    "filename",
                    "begin_time_s",
                    "end_time_s",
                    "species",
                    "call_type_bucket",
                    "call_type_raw",
                    "comments",
                    "context_tags",
                ],
                [
                    {
                        "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                        "begin_time_s": "10.0",
                        "end_time_s": "11.0",
                        "species": "Bp",
                        "call_type_bucket": "20Hz",
                        "call_type_raw": "20 Hz",
                        "comments": "",
                        "context_tags": "vessel_or_masking",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250205T000000.000Z.flac",
                        "begin_time_s": "20.0",
                        "end_time_s": "21.0",
                        "species": "Bp",
                        "call_type_bucket": "40Hz",
                        "call_type_raw": "40 Hz",
                        "comments": "",
                        "context_tags": "unknown_other",
                    },
                ],
            )

            _write_csv(
                clip_manifest_csv,
                [
                    "filename",
                    "is_fin_positive",
                    "is_annotated_non_fin",
                    "species_codes",
                    "fin_call_type_buckets",
                    "context_tags",
                ],
                [
                    {
                        "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                        "is_fin_positive": "1",
                        "is_annotated_non_fin": "0",
                        "species_codes": "Bp",
                        "fin_call_type_buckets": "20Hz",
                        "context_tags": "vessel_or_masking",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250205T000000.000Z.flac",
                        "is_fin_positive": "1",
                        "is_annotated_non_fin": "0",
                        "species_codes": "Bp",
                        "fin_call_type_buckets": "40Hz",
                        "context_tags": "unknown_other",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250305T000000.000Z.flac",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "1",
                        "species_codes": "OD",
                        "fin_call_type_buckets": "",
                        "context_tags": "click_overlap",
                    },
                ],
            )

            payload = {
                "schema_version": "2.1",
                "items": [
                    {
                        "item_id": "evt_001",
                        "model_outputs": [{"class_hierarchy": "Fin whale", "score": 0.91}],
                        "source_segments": [
                            {
                                "source_audio": "ICLISTENHF6016_20250105T000000.000Z.flac",
                                "time_start_sec": 9.5,
                                "time_end_sec": 11.2,
                                "score": 0.91,
                            }
                        ],
                    },
                    {
                        "item_id": "evt_002",
                        "model_outputs": [{"class_hierarchy": "Fin whale", "score": 0.83}],
                        "source_segments": [
                            {
                                "source_audio": "ICLISTENHF6016_20250305T000000.000Z.flac",
                                "time_start_sec": 30.0,
                                "time_end_sec": 31.0,
                                "score": 0.83,
                            }
                        ],
                    },
                ],
            }
            predictions_json.write_text(json.dumps(payload), encoding="utf-8")

            annotations = load_annotations_csv(annotations_csv)
            clip_manifest = load_clip_manifest_csv(clip_manifest_csv)
            _, predictions = load_prediction_segments(predictions_json)

            matches, unmatched_predictions, unmatched_annotations = match_predictions_to_annotations(
                predictions,
                annotations,
                collar_s=1.0,
            )

            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(unmatched_predictions), 1)
            self.assertEqual(len(unmatched_annotations), 1)

            clip_confusion = build_clip_confusion(clip_manifest, predictions)
            self.assertEqual(clip_confusion["tp"], 1)
            self.assertEqual(clip_confusion["fp"], 1)
            self.assertEqual(clip_confusion["fn"], 1)
            self.assertEqual(clip_confusion["tn"], 0)

    def test_absolute_source_segment_times_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            predictions_json = tmp / "predictions_postprocessed.json"
            payload = {
                "schema_version": "2.1",
                "items": [
                    {
                        "item_id": "evt_abs",
                        "model_outputs": [{"class_hierarchy": "Fin whale", "score": 0.91}],
                        "source_segments": [
                            {
                                "source_audio": "ICLISTENHF6016_20250105T000000.000Z.flac",
                                "time_start_sec": 1736035210.5,
                                "time_end_sec": 1736035211.7,
                                "score": 0.91,
                            }
                        ],
                    }
                ],
            }
            predictions_json.write_text(json.dumps(payload), encoding="utf-8")
            _, predictions = load_prediction_segments(predictions_json)
            self.assertEqual(len(predictions), 1)
            self.assertAlmostEqual(predictions[0].start_time_s, 10.5, places=4)
            self.assertAlmostEqual(predictions[0].end_time_s, 11.7, places=4)


if __name__ == "__main__":
    unittest.main()
