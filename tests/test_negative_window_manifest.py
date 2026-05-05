import tempfile
import unittest
from pathlib import Path

from scripts.data.multilabel.build_negative_window_manifest import (
    build_negative_manifest,
    leaked_groups_by_split,
    negative_bucket_from_row,
    primary_adjacent_gap_rows,
)
from src.dataset.multilabel import label_balanced_grouped_split, write_csv_rows


class NegativeWindowManifestTest(unittest.TestCase):
    def test_assigns_no_primary_negative_buckets(self):
        self.assertEqual(
            negative_bucket_from_row({"label_ids": "", "review_status": "reviewed_background", "is_background": "1"}),
            "reviewed_background",
        )
        self.assertEqual(
            negative_bucket_from_row({"source_label_ids": "species:OD", "analysis_label_ids": "group:odontocete_unknown"}),
            "nonprimary_biological_signal",
        )
        self.assertEqual(
            negative_bucket_from_row({"source_class_species": "AB", "source_dataset": "dclde_2027_uaf"}),
            "nonbiological_signal",
        )
        self.assertEqual(
            negative_bucket_from_row({"label_ids": "", "source_dataset": "biodcase_task2_train"}),
            "external_source_gap",
        )
        self.assertEqual(
            negative_bucket_from_row(
                {"label_ids": "", "review_status": "reviewed_background", "source_dataset": "biodcase_task2_train"}
            ),
            "external_source_gap",
        )
        self.assertEqual(
            negative_bucket_from_row(
                {
                    "label_ids": "",
                    "review_status": "reviewed_background",
                    "source_dataset": "ballenyislands2015",
                    "mat_path": "/tmp/biodcase_task2_prep/mat_files/background.mat",
                }
            ),
            "external_source_gap",
        )
        self.assertEqual(negative_bucket_from_row({"label_ids": "species:Oo"}), "")

    def test_primary_adjacent_gap_windows_respect_exclusion_buffer(self):
        rows = [
            {
                "filename": "clip-a.wav",
                "begin_s": "20",
                "end_s": "25",
                "label_ids": "species:Bp",
            },
            {
                "filename": "clip-a.wav",
                "begin_s": "60",
                "end_s": "70",
                "label_ids": "species:Mn",
            },
        ]

        gaps = primary_adjacent_gap_rows(
            annotation_rows=rows,
            clip_durations={"clip-a.wav": 100.0},
            window_s=10.0,
            exclusion_buffer_s=5.0,
            step_s=10.0,
        )

        self.assertTrue(gaps)
        for row in gaps:
            start = float(row["begin_s"])
            end = float(row["end_s"])
            self.assertFalse(start < 30 and end > 15)
            self.assertFalse(start < 75 and end > 55)
            self.assertEqual(row["negative_bucket"], "primary_adjacent_gap")

    def test_primary_adjacent_gap_windows_accept_standardized_window_fields(self):
        rows = [
            {
                "source_audio": "/tmp/audio/clip-a.wav",
                "window_start_s": "20",
                "duration_s": "5",
                "label_ids": "species:Bp",
            }
        ]

        gaps = primary_adjacent_gap_rows(
            annotation_rows=rows,
            clip_durations={"/tmp/audio/clip-a.wav": 60.0},
            window_s=10.0,
            exclusion_buffer_s=5.0,
            step_s=10.0,
        )

        self.assertTrue(gaps)
        for row in gaps:
            start = float(row["begin_s"])
            end = float(row["end_s"])
            self.assertFalse(start < 30 and end > 15)

    def test_builder_writes_buckets_and_leak_free_grouped_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            durations = root / "durations.csv"
            out = root / "negative_manifest.csv"
            write_csv_rows(
                annotations,
                [
                    {
                        "item_id": "pos-a",
                        "filename": "clip-a.wav",
                        "begin_s": "20",
                        "end_s": "25",
                        "label_ids": "species:Bp",
                        "event_group": "clip-a.wav",
                    },
                    {
                        "item_id": "od-a",
                        "filename": "clip-b.wav",
                        "begin_s": "5",
                        "end_s": "8",
                        "source_label_ids": "species:OD",
                        "analysis_label_ids": "group:odontocete_unknown",
                        "event_group": "clip-b.wav",
                    },
                    {
                        "item_id": "ab-a",
                        "filename": "clip-c.wav",
                        "begin_s": "5",
                        "end_s": "8",
                        "source_class_species": "AB",
                        "event_group": "clip-c.wav",
                    },
                ],
            )
            write_csv_rows(durations, [{"filename": "clip-a.wav", "duration_s": "60"}])

            summary = build_negative_manifest(
                annotations_csv=annotations,
                output_csv=out,
                clip_duration_csv=durations,
                window_s=10.0,
                exclusion_buffer_s=2.0,
                step_s=10.0,
                max_windows_per_clip=2,
                split=True,
            )

            self.assertEqual(summary["negative_bucket_counts"]["nonprimary_biological_signal"], 1)
            self.assertEqual(summary["negative_bucket_counts"]["nonbiological_signal"], 1)
            self.assertEqual(summary["negative_bucket_counts"]["primary_adjacent_gap"], 2)
            self.assertEqual(summary["leaked_group_count"], 0)

            split_rows = label_balanced_grouped_split(
                [
                    {"item_id": "a1", "event_group": "group-a", "label_ids": ""},
                    {"item_id": "a2", "event_group": "group-a", "label_ids": ""},
                    {"item_id": "b1", "event_group": "group-b", "label_ids": "species:Oo"},
                    {"item_id": "c1", "event_group": "group-c", "label_ids": ""},
                    {"item_id": "d1", "event_group": "group-d", "label_ids": "species:Mn"},
                ],
                train_ratio=0.5,
                val_ratio=0.25,
                seed=9,
            )
            flat = [row for split_items in split_rows.values() for row in split_items]
            self.assertEqual(leaked_groups_by_split(flat), {})


if __name__ == "__main__":
    unittest.main()
