import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis import e126_binary_gate_report as e126


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestE126BinaryGateReport(unittest.TestCase):
    def test_tunes_threshold_and_reports_species_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "val.csv"
            test = root / "test.csv"
            write_csv(
                val,
                [
                    {"item_id": "bp_val", "true_class": "species:Bp", "stage1_prob_call": "0.90"},
                    {"item_id": "bm_val", "true_class": "species:Bm", "stage1_prob_call": "0.80"},
                    {"item_id": "mn_val", "true_class": "species:Mn", "stage1_prob_call": "0.70"},
                    {"item_id": "bg_val", "true_class": "background", "stage1_prob_call": "0.30"},
                ],
            )
            write_csv(
                test,
                [
                    {"item_id": "bp", "true_class": "species:Bp", "stage1_prob_call": "0.95"},
                    {"item_id": "bm", "true_class": "species:Bm", "stage1_prob_call": "0.20"},
                    {"item_id": "mn", "true_class": "species:Mn", "stage1_prob_call": "0.75"},
                    {"item_id": "bg_fp", "true_class": "background", "stage1_prob_call": "0.65"},
                    {"item_id": "bg_tn", "true_class": "background", "stage1_prob_call": "0.05"},
                ],
            )

            out = root / "out"
            summary = e126.run_report(
                name="unit",
                val_predictions=val,
                test_predictions=test,
                output_dir=out,
                class_ids=["background", "species:Bp", "species:Bm", "species:Mn"],
                positive_labels=["species:Bp", "species:Bm", "species:Mn"],
                score_label="task:whale_call",
                score_field=None,
                thresholds=[0.0, 0.5, 0.7],
            )

            self.assertEqual(summary["threshold"], 0.7)
            test_metric = [row for row in summary["metrics"] if row["split"] == "test"][0]
            self.assertEqual(test_metric["tp"], 2)
            self.assertEqual(test_metric["fp"], 0)
            self.assertEqual(test_metric["tn"], 2)
            self.assertEqual(test_metric["fn"], 1)

            breakdown = {row["true_bucket"]: row for row in e126.species_breakdown(
                e126.load_gate_rows(
                    test,
                    class_ids=["background", "species:Bp", "species:Bm", "species:Mn"],
                    positive_labels=["species:Bp", "species:Bm", "species:Mn"],
                    score_field=None,
                    score_label="task:whale_call",
                ),
                threshold=0.7,
                positive_labels=["species:Bp", "species:Bm", "species:Mn"],
            )}
            self.assertEqual(breakdown["species:Bm"]["missed"], 1)
            self.assertTrue((out / "e126_binary_gate_report.md").is_file())
            self.assertTrue((out / "e126_binary_gate_examples.csv").is_file())
            payload = json.loads((out / "e126_binary_gate_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["threshold"], 0.7)

    def test_reads_explicit_score_field_and_missing_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "val.csv"
            test = root / "test.csv"
            rows = [
                {"item_id": "pos", "label_ids": "species:Bp", "custom_score": "0.9"},
                {"item_id": "bg", "label_ids": "", "custom_score": "0.1"},
                {"item_id": "missing", "label_ids": "species:Mn", "custom_score": ""},
            ]
            write_csv(val, rows)
            write_csv(test, rows)

            summary = e126.run_report(
                name="custom",
                val_predictions=val,
                test_predictions=test,
                output_dir=root / "out",
                class_ids=["background", "species:Bp", "species:Bm", "species:Mn"],
                positive_labels=["species:Bp", "species:Bm", "species:Mn"],
                score_label="task:whale_call",
                score_field="custom_score",
                thresholds=[0.5],
            )
            metric = [row for row in summary["metrics"] if row["split"] == "test"][0]
            self.assertEqual(metric["missing_score"], 1)
            self.assertEqual(metric["tp"], 1)
            self.assertEqual(metric["fn"], 1)

    def test_recovers_species_breakdown_from_collapsed_gate_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "val.csv"
            test = root / "test.csv"
            rows = [
                {
                    "item_id": "bp",
                    "true_class_index": "1",
                    "target_label_ids": "task:whale_call",
                    "original_label_ids": "species:Bp",
                    "gate_positive_source_labels": "species:Bp",
                    "score__task:whale_call": "0.95",
                },
                {
                    "item_id": "bm",
                    "true_class_index": "1",
                    "target_label_ids": "task:whale_call",
                    "original_label_ids": "species:Bm",
                    "gate_positive_source_labels": "species:Bm",
                    "score__task:whale_call": "0.20",
                },
                {
                    "item_id": "bg",
                    "true_class_index": "0",
                    "target_label_ids": "",
                    "original_label_ids": "",
                    "score__task:whale_call": "0.05",
                },
            ]
            write_csv(val, rows)
            write_csv(test, rows)

            loaded = e126.load_gate_rows(
                test,
                class_ids=["background", "task:whale_call"],
                positive_labels=["species:Bp", "species:Bm", "species:Mn"],
                score_field=None,
                score_label="task:whale_call",
            )
            metrics = e126.binary_metrics(loaded, threshold=0.5)
            self.assertEqual(metrics["tp"], 1)
            self.assertEqual(metrics["fn"], 1)
            breakdown = {
                row["true_bucket"]: row
                for row in e126.species_breakdown(
                    loaded,
                    threshold=0.5,
                    positive_labels=["species:Bp", "species:Bm", "species:Mn"],
                )
            }
            self.assertEqual(breakdown["species:Bp"]["detected"], 1)
            self.assertEqual(breakdown["species:Bm"]["missed"], 1)
            self.assertEqual(breakdown["background"]["missed"], 1)

    def test_can_append_living_ledger_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "val.csv"
            test = root / "test.csv"
            ledger_path = root / "ledger.md"
            write_csv(
                val,
                [
                    {"item_id": "pos", "true_class": "species:Bp", "stage1_prob_call": "0.9"},
                    {"item_id": "bg", "true_class": "background", "stage1_prob_call": "0.1"},
                ],
            )
            write_csv(
                test,
                [
                    {"item_id": "pos", "true_class": "species:Bp", "stage1_prob_call": "0.9"},
                    {"item_id": "bg", "true_class": "background", "stage1_prob_call": "0.1"},
                ],
            )

            summary = e126.run_report(
                name="ledger_unit",
                val_predictions=val,
                test_predictions=test,
                output_dir=root / "out",
                class_ids=["background", "species:Bp", "species:Bm", "species:Mn"],
                positive_labels=["species:Bp", "species:Bm", "species:Mn"],
                score_label="task:whale_call",
                score_field=None,
                thresholds=[0.5],
                ledger_path=ledger_path,
                training_set="unit train set",
                validation_set="unit val set",
                test_set="unit test set",
                evaluation_note="unit production-style gate check",
            )

            self.assertEqual(summary["outputs"]["ledger"], str(ledger_path))
            ledger_text = ledger_path.read_text(encoding="utf-8")
            self.assertIn("ledger_unit: Binary Whale Gate", ledger_text)
            self.assertIn("Training set: unit train set.", ledger_text)
            self.assertIn("unit production-style gate check", ledger_text)
            saved_summary = json.loads((root / "out" / "e126_binary_gate_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["outputs"]["ledger"], str(ledger_path))


if __name__ == "__main__":
    unittest.main()
