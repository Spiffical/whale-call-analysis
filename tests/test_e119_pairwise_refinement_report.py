import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis import e119_pairwise_refinement_report as e119  # noqa: E402


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestE119PairwiseRefinementReport(unittest.TestCase):
    def test_run_dir_discovery_and_pairwise_refinement_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            pairwise = root / "pairwise"
            (base / "train").mkdir(parents=True)
            (pairwise / "train").mkdir(parents=True)
            (base / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Bm", "species:Mn"]}),
                encoding="utf-8",
            )
            (pairwise / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Mn"]}),
                encoding="utf-8",
            )

            write_csv(
                base / "train" / "val_predictions_best_val_rule.csv",
                [
                    {"item_id": "v1", "true_class_index": "1", "pred_class_index": "3"},
                    {"item_id": "v2", "true_class_index": "3", "pred_class_index": "3"},
                    {"item_id": "v3", "true_class_index": "2", "pred_class_index": "2"},
                    {"item_id": "v4", "true_class_index": "0", "pred_class_index": "0"},
                ],
            )
            write_csv(
                base / "train" / "test_predictions_best_val_rule.csv",
                [
                    {"item_id": "t1", "true_class_index": "1", "pred_class_index": "3"},
                    {"item_id": "t2", "true_class_index": "3", "pred_class_index": "1"},
                    {"item_id": "t3", "true_class_index": "2", "pred_class_index": "2"},
                    {"item_id": "t4", "true_class_index": "0", "pred_class_index": "0"},
                ],
            )
            write_csv(
                pairwise / "train" / "val_predictions_argmax.csv",
                [
                    {
                        "item_id": "v1",
                        "true_class_index": "1",
                        "pred_class_index": "1",
                        "prob__species:Bp": "0.90",
                        "prob__species:Mn": "0.10",
                    },
                    {
                        "item_id": "v2",
                        "true_class_index": "2",
                        "pred_class_index": "2",
                        "prob__species:Bp": "0.10",
                        "prob__species:Mn": "0.90",
                    },
                ],
            )
            write_csv(
                pairwise / "train" / "test_predictions_argmax.csv",
                [
                    {
                        "item_id": "t1",
                        "true_class_index": "1",
                        "pred_class_index": "1",
                        "prob__species:Bp": "0.90",
                        "prob__species:Mn": "0.10",
                    },
                    {
                        "item_id": "t2",
                        "true_class_index": "2",
                        "pred_class_index": "2",
                        "prob__species:Bp": "0.20",
                        "prob__species:Mn": "0.80",
                    },
                ],
            )

            output = root / "out"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "e119_pairwise_refinement_report.py",
                    "--name",
                    "unit",
                    "--base-run-dir",
                    str(base),
                    "--pairwise-run-dir",
                    str(pairwise),
                    "--output-dir",
                    str(output),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(e119.main(), 0)
            finally:
                sys.argv = old_argv

            summary = json.loads((output / "e119_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["inputs"]["base_test_predictions"], str(base / "train" / "test_predictions_best_val_rule.csv"))
            self.assertEqual(summary["inputs"]["pairwise_test_predictions"], str(pairwise / "train" / "test_predictions_argmax.csv"))
            test_base = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "pred"][0]
            test_refined = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "refined"][0]
            self.assertGreater(test_refined["macro_f1"], test_base["macro_f1"])
            self.assertLess(test_refined["cross_species_fp"], test_base["cross_species_fp"])
            self.assertTrue((output / "e119_examples.csv").is_file())

    def test_calibrated_base_decision_mode_uses_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            pairwise = root / "pairwise"
            (base / "train").mkdir(parents=True)
            (pairwise / "train").mkdir(parents=True)
            (base / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Bm", "species:Mn"]}),
                encoding="utf-8",
            )
            (pairwise / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Mn"]}),
                encoding="utf-8",
            )
            base_rows = [
                {
                    "item_id": "bp",
                    "true_class_index": "1",
                    "pred_class_index": "0",
                    "prob__background": "0.10",
                    "prob__species:Bp": "0.90",
                    "prob__species:Bm": "0.01",
                    "prob__species:Mn": "0.10",
                },
                {
                    "item_id": "bm",
                    "true_class_index": "2",
                    "pred_class_index": "0",
                    "prob__background": "0.10",
                    "prob__species:Bp": "0.01",
                    "prob__species:Bm": "0.90",
                    "prob__species:Mn": "0.01",
                },
                {
                    "item_id": "mn",
                    "true_class_index": "3",
                    "pred_class_index": "0",
                    "prob__background": "0.10",
                    "prob__species:Bp": "0.10",
                    "prob__species:Bm": "0.01",
                    "prob__species:Mn": "0.90",
                },
                {
                    "item_id": "bg",
                    "true_class_index": "0",
                    "pred_class_index": "0",
                    "prob__background": "0.90",
                    "prob__species:Bp": "0.05",
                    "prob__species:Bm": "0.05",
                    "prob__species:Mn": "0.05",
                },
            ]
            write_csv(base / "train" / "val_predictions_best_val_rule.csv", base_rows)
            write_csv(base / "train" / "test_predictions_best_val_rule.csv", base_rows)
            pair_rows = [
                {
                    "item_id": "bp",
                    "true_class_index": "1",
                    "pred_class_index": "1",
                    "prob__species:Bp": "0.90",
                    "prob__species:Mn": "0.10",
                },
                {
                    "item_id": "mn",
                    "true_class_index": "2",
                    "pred_class_index": "2",
                    "prob__species:Bp": "0.10",
                    "prob__species:Mn": "0.90",
                },
            ]
            write_csv(pairwise / "train" / "val_predictions_argmax.csv", pair_rows)
            write_csv(pairwise / "train" / "test_predictions_argmax.csv", pair_rows)

            output = root / "out"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "e119_pairwise_refinement_report.py",
                    "--name",
                    "unit-calibrated",
                    "--base-run-dir",
                    str(base),
                    "--pairwise-run-dir",
                    str(pairwise),
                    "--base-decision-mode",
                    "calibrated",
                    "--base-calibration-threshold-grid",
                    "0.50",
                    "--base-calibration-margin-grid=-0.25,0.0",
                    "--base-calibration-bias-grid",
                    "0.0",
                    "--output-dir",
                    str(output),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(e119.main(), 0)
            finally:
                sys.argv = old_argv

            summary = json.loads((output / "e119_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["base_decision_mode"], "calibrated")
            self.assertEqual(summary["base_rule"]["threshold"], 0.5)
            test_base = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "pred"][0]
            self.assertEqual(test_base["macro_f1"], 1.0)
            self.assertTrue((output / "e119_base_calibration_sweep.csv").is_file())

    def test_pairwise_labels_are_inferred_for_blue_specialist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            pairwise = root / "pairwise"
            (base / "train").mkdir(parents=True)
            (pairwise / "train").mkdir(parents=True)
            (base / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Bm", "species:Mn"]}),
                encoding="utf-8",
            )
            (pairwise / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bm", "species:Bp"]}),
                encoding="utf-8",
            )

            base_rows = [
                {"item_id": "bm", "true_class_index": "2", "pred_class_index": "1"},
                {"item_id": "bp", "true_class_index": "1", "pred_class_index": "1"},
                {"item_id": "mn", "true_class_index": "3", "pred_class_index": "3"},
                {"item_id": "bg", "true_class_index": "0", "pred_class_index": "0"},
            ]
            pair_rows = [
                {
                    "item_id": "bm",
                    "true_class_index": "1",
                    "pred_class_index": "1",
                    "prob__species:Bm": "0.95",
                    "prob__species:Bp": "0.05",
                },
                {
                    "item_id": "bp",
                    "true_class_index": "2",
                    "pred_class_index": "2",
                    "prob__species:Bm": "0.05",
                    "prob__species:Bp": "0.95",
                },
            ]
            write_csv(base / "train" / "val_predictions_best_val_rule.csv", base_rows)
            write_csv(base / "train" / "test_predictions_best_val_rule.csv", base_rows)
            write_csv(pairwise / "train" / "val_predictions_argmax.csv", pair_rows)
            write_csv(pairwise / "train" / "test_predictions_argmax.csv", pair_rows)

            output = root / "out"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "e119_pairwise_refinement_report.py",
                    "--name",
                    "unit-blue-pairwise",
                    "--base-run-dir",
                    str(base),
                    "--pairwise-run-dir",
                    str(pairwise),
                    "--output-dir",
                    str(output),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(e119.main(), 0)
            finally:
                sys.argv = old_argv

            summary = json.loads((output / "e119_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["pairwise_labels"], ["species:Bm", "species:Bp"])
            test_base = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "pred"][0]
            test_refined = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "refined"][0]
            self.assertGreater(test_refined["macro_f1"], test_base["macro_f1"])

            with (output / "e119_examples.csv").open(newline="", encoding="utf-8") as handle:
                examples = list(csv.DictReader(handle))
            self.assertEqual(examples[0]["pairwise_prob__species:Bm"], "0.95")
            report = (output / "e119_pairwise_refinement_report.md").read_text(encoding="utf-8")
            self.assertIn("species:Bm vs species:Bp", report)

    def test_multiple_base_run_dirs_write_ranked_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairwise = root / "pairwise"
            (pairwise / "train").mkdir(parents=True)
            (pairwise / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Mn"]}),
                encoding="utf-8",
            )

            pair_rows = [
                {
                    "item_id": "bp",
                    "true_class_index": "1",
                    "pred_class_index": "1",
                    "prob__species:Bp": "0.90",
                    "prob__species:Mn": "0.10",
                },
                {
                    "item_id": "mn",
                    "true_class_index": "2",
                    "pred_class_index": "2",
                    "prob__species:Bp": "0.10",
                    "prob__species:Mn": "0.90",
                },
            ]
            write_csv(pairwise / "train" / "val_predictions_argmax.csv", pair_rows)
            write_csv(pairwise / "train" / "test_predictions_argmax.csv", pair_rows)

            def make_base(name, pred_bp, pred_mn):
                base = root / name
                (base / "train").mkdir(parents=True)
                (base / "train" / "run_summary.json").write_text(
                    json.dumps({"class_ids": ["background", "species:Bp", "species:Bm", "species:Mn"]}),
                    encoding="utf-8",
                )
                rows = [
                    {"item_id": "bp", "true_class_index": "1", "pred_class_index": pred_bp},
                    {"item_id": "mn", "true_class_index": "3", "pred_class_index": pred_mn},
                    {"item_id": "bm", "true_class_index": "2", "pred_class_index": "2"},
                ]
                write_csv(base / "train" / "val_predictions_best_val_rule.csv", rows)
                write_csv(base / "train" / "test_predictions_best_val_rule.csv", rows)
                return base

            better_base = make_base("base_better", "3", "3")
            worse_base = make_base("base_worse", "0", "0")

            output = root / "out"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "e119_pairwise_refinement_report.py",
                    "--name",
                    "multi",
                    "--base-run-dir",
                    str(worse_base),
                    "--base-run-dir",
                    str(better_base),
                    "--pairwise-run-dir",
                    str(pairwise),
                    "--output-dir",
                    str(output),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(e119.main(), 0)
            finally:
                sys.argv = old_argv

            with (output / "e119_comparison_rankings.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["base_name"], "base_better")
            self.assertGreater(float(rows[0]["refined_test_macro_f1"]), float(rows[1]["refined_test_macro_f1"]))
            self.assertTrue((output / "e119_comparison_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
