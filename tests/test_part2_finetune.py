import csv
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.part2_finetune import (
    assign_time_pools,
    build_learning_curve_plan,
    inventory_rows_from_dataset,
    load_finetune_clip_records,
    order_train_pool,
    select_budget_clips,
    split_inventory_rows,
)
from src.training.mat_utils import parse_mat_filename


def _write_csv(path: Path, fieldnames, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestPart2FineTunePlanning(unittest.TestCase):
    def test_parse_negative_mat_filename_supports_flac(self):
        src, start, dur = parse_mat_filename("ICLISTENHF6016_20250105T000000.000Z.flac_neg_4.mat")
        self.assertEqual(src, "ICLISTENHF6016_20250105T000000.000Z.flac")
        self.assertIsNone(start)
        self.assertIsNone(dur)

    def test_assign_time_pools_uses_fin_positive_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fin_annotations = tmp / "fin_annotations.csv"
            clip_manifest = tmp / "clip_manifest.csv"

            _write_csv(
                fin_annotations,
                ["filename", "species"],
                [
                    {"filename": f"ICLISTENHF6016_2025010{i}T000000.000Z.flac", "species": "Bp"}
                    for i in range(1, 6)
                ],
            )
            _write_csv(
                clip_manifest,
                ["filename", "fin_call_type_buckets", "context_tags", "is_fin_positive", "is_annotated_non_fin"],
                [
                    {
                        "filename": f"ICLISTENHF6016_2025010{i}T000000.000Z.flac",
                        "fin_call_type_buckets": "20Hz",
                        "context_tags": "vessel_or_masking",
                        "is_fin_positive": "1" if i <= 5 else "0",
                        "is_annotated_non_fin": "0",
                    }
                    for i in range(1, 6)
                ]
                + [
                    {
                        "filename": "ICLISTENHF6016_20250106T000000.000Z.flac",
                        "fin_call_type_buckets": "",
                        "context_tags": "click_overlap",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "1",
                    }
                ],
            )

            records = load_finetune_clip_records(
                fin_annotations_csv=fin_annotations,
                clip_manifest_csv=clip_manifest,
            )
            split_map = assign_time_pools(records, train_ratio=0.6, val_ratio=0.2)
            self.assertEqual([r.filename for r in split_map["train"][:3]], [
                "ICLISTENHF6016_20250101T000000.000Z.flac",
                "ICLISTENHF6016_20250102T000000.000Z.flac",
                "ICLISTENHF6016_20250103T000000.000Z.flac",
            ])
            self.assertEqual(
                [r.filename for r in split_map["val"]],
                ["ICLISTENHF6016_20250104T000000.000Z.flac"],
            )
            self.assertEqual(
                [r.filename for r in split_map["test"]],
                [
                    "ICLISTENHF6016_20250105T000000.000Z.flac",
                    "ICLISTENHF6016_20250106T000000.000Z.flac",
                ],
            )

    def test_budget_selection_is_clip_based_and_month_stratified_mode_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fin_annotations = tmp / "fin_annotations.csv"
            clip_manifest = tmp / "clip_manifest.csv"

            annotation_rows = []
            manifest_rows = []
            clip_specs = [
                ("ICLISTENHF6016_20250101T000000.000Z.flac", 5, "20Hz"),
                ("ICLISTENHF6016_20250102T000000.000Z.flac", 7, "20Hz"),
                ("ICLISTENHF6016_20250201T000000.000Z.flac", 6, "40Hz"),
                ("ICLISTENHF6016_20250202T000000.000Z.flac", 4, "other_fin"),
            ]
            for filename, count, bucket in clip_specs:
                for _ in range(count):
                    annotation_rows.append({"filename": filename, "species": "Bp"})
                manifest_rows.append(
                    {
                        "filename": filename,
                        "fin_call_type_buckets": bucket,
                        "context_tags": "vessel_or_masking",
                        "is_fin_positive": "1",
                        "is_annotated_non_fin": "0",
                    }
                )
            _write_csv(fin_annotations, ["filename", "species"], annotation_rows)
            _write_csv(
                clip_manifest,
                ["filename", "fin_call_type_buckets", "context_tags", "is_fin_positive", "is_annotated_non_fin"],
                manifest_rows,
            )

            records = load_finetune_clip_records(
                fin_annotations_csv=fin_annotations,
                clip_manifest_csv=clip_manifest,
            )
            train_pool = [record for record in records if record.is_fin_positive]
            selected = select_budget_clips(train_pool, budget_calls=8, sampling_mode="chronological", seed=1337)
            self.assertEqual(len(selected), 2)
            self.assertEqual(sum(record.fin_call_count for record in selected), 12)

            month_stratified = order_train_pool(train_pool, sampling_mode="month_stratified_clip", seed=1337)
            self.assertEqual(month_stratified[0].timestamp.strftime("%Y%m"), "202501")
            self.assertEqual(month_stratified[1].timestamp.strftime("%Y%m"), "202502")

    def test_learning_curve_plan_and_split_inventory_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fin_annotations = tmp / "fin_annotations.csv"
            clip_manifest = tmp / "clip_manifest.csv"
            sample_inventory = tmp / "sample_inventory.csv"

            annotation_rows = []
            manifest_rows = []
            for idx in range(1, 9):
                filename = f"ICLISTENHF6016_2025010{idx}T000000.000Z.flac"
                for _ in range(3):
                    annotation_rows.append({"filename": filename, "species": "Bp"})
                manifest_rows.append(
                    {
                        "filename": filename,
                        "fin_call_type_buckets": "20Hz",
                        "context_tags": "vessel_or_masking",
                        "is_fin_positive": "1",
                        "is_annotated_non_fin": "0",
                    }
                )
            manifest_rows.extend(
                [
                    {
                        "filename": "ICLISTENHF6016_20241231T120000.000Z.flac",
                        "fin_call_type_buckets": "",
                        "context_tags": "click_overlap",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "1",
                    },
                    {
                        "filename": "ICLISTENHF6016_20241231T180000.000Z.flac",
                        "fin_call_type_buckets": "",
                        "context_tags": "click_overlap",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "1",
                    },
                ]
            )

            _write_csv(fin_annotations, ["filename", "species"], annotation_rows)
            _write_csv(
                clip_manifest,
                ["filename", "fin_call_type_buckets", "context_tags", "is_fin_positive", "is_annotated_non_fin"],
                manifest_rows,
            )
            _write_csv(
                sample_inventory,
                ["relative_path", "label", "source_audio"],
                [
                    {"relative_path": "mat_files/ICLISTENHF6016_20250101T000000.000Z.flac_0.0s_40.0s.mat", "label": "1", "source_audio": "ICLISTENHF6016_20250101T000000.000Z.flac"},
                    {"relative_path": "neg_mat_files/ICLISTENHF6016_20241231T120000.000Z.flac_neg_0.mat", "label": "0", "source_audio": "ICLISTENHF6016_20241231T120000.000Z.flac"},
                    {"relative_path": "mat_files/ICLISTENHF6016_20250107T000000.000Z.flac_0.0s_40.0s.mat", "label": "1", "source_audio": "ICLISTENHF6016_20250107T000000.000Z.flac"},
                    {"relative_path": "neg_mat_files/ICLISTENHF6016_20241231T180000.000Z.flac_neg_0.mat", "label": "0", "source_audio": "ICLISTENHF6016_20241231T180000.000Z.flac"},
                    {"relative_path": "mat_files/ICLISTENHF6016_20250108T000000.000Z.flac_0.0s_40.0s.mat", "label": "1", "source_audio": "ICLISTENHF6016_20250108T000000.000Z.flac"},
                ],
            )

            records = load_finetune_clip_records(
                fin_annotations_csv=fin_annotations,
                clip_manifest_csv=clip_manifest,
            )
            plan_rows, split_map = build_learning_curve_plan(
                records=records,
                budgets=[6],
                sampling_modes=["chronological"],
                repeats=1,
                train_ratio=0.7,
                val_ratio=0.1,
                base_seed=1337,
            )
            self.assertEqual(len(plan_rows), 1)
            run = plan_rows[0]
            self.assertEqual(run["sampling_mode"], "chronological")
            self.assertGreaterEqual(int(run["actual_budget_calls"]), 6)
            self.assertIn("ICLISTENHF6016_20250101T000000.000Z.flac", run["train_fin_clip_names"])
            self.assertEqual(int(run["train_nonfin_clip_count"]), 2)
            self.assertIn("ICLISTENHF6016_20241231T120000.000Z.flac", run["train_nonfin_clip_names"])
            self.assertIn("ICLISTENHF6016_20241231T180000.000Z.flac", run["train_nonfin_clip_names"])

            inventory_rows = inventory_rows_from_dataset(sample_inventory_csv=sample_inventory)
            split_rows = split_inventory_rows(
                inventory_rows,
                train_clips=set(run["train_fin_clip_names"].split("|")) | set(run["train_nonfin_clip_names"].split("|")),
                val_clips={record.filename for record in split_map["val"] if record.is_fin_positive or record.is_annotated_non_fin},
                test_clips={record.filename for record in split_map["test"] if record.is_fin_positive or record.is_annotated_non_fin},
            )
            self.assertTrue(any(row["label"] == "1" for row in split_rows["train"]))
            self.assertTrue(any(row["label"] == "0" for row in split_rows["train"]))

    def test_small_budget_keeps_full_training_nonfin_pool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fin_annotations = tmp / "fin_annotations.csv"
            clip_manifest = tmp / "clip_manifest.csv"

            _write_csv(
                fin_annotations,
                ["filename", "species"],
                [{"filename": "ICLISTENHF6016_20250101T000000.000Z.flac", "species": "Bp"} for _ in range(5)]
                + [{"filename": "ICLISTENHF6016_20250102T000000.000Z.flac", "species": "Bp"} for _ in range(5)],
            )
            _write_csv(
                clip_manifest,
                ["filename", "fin_call_type_buckets", "context_tags", "is_fin_positive", "is_annotated_non_fin"],
                [
                    {
                        "filename": "ICLISTENHF6016_20250101T000000.000Z.flac",
                        "fin_call_type_buckets": "20Hz",
                        "context_tags": "vessel_or_masking",
                        "is_fin_positive": "1",
                        "is_annotated_non_fin": "0",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250102T000000.000Z.flac",
                        "fin_call_type_buckets": "20Hz",
                        "context_tags": "vessel_or_masking",
                        "is_fin_positive": "1",
                        "is_annotated_non_fin": "0",
                    },
                    {
                        "filename": "ICLISTENHF6016_20241231T120000.000Z.flac",
                        "fin_call_type_buckets": "",
                        "context_tags": "click_overlap",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "1",
                    },
                    {
                        "filename": "ICLISTENHF6016_20241231T180000.000Z.flac",
                        "fin_call_type_buckets": "",
                        "context_tags": "click_overlap",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "1",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250103T120000.000Z.flac",
                        "fin_call_type_buckets": "",
                        "context_tags": "click_overlap",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "1",
                    },
                ],
            )

            records = load_finetune_clip_records(
                fin_annotations_csv=fin_annotations,
                clip_manifest_csv=clip_manifest,
            )
            plan_rows, _ = build_learning_curve_plan(
                records=records,
                budgets=[5],
                sampling_modes=["chronological"],
                repeats=1,
                train_ratio=0.7,
                val_ratio=0.1,
                base_seed=1337,
            )
            run = plan_rows[0]
            self.assertEqual(int(run["train_fin_clip_count"]), 1)
            # All pure non-fin clips assigned to the training partition should stay available
            # even for the smallest fin-call budget.
            self.assertEqual(int(run["train_nonfin_clip_count"]), 2)


if __name__ == "__main__":
    unittest.main()
