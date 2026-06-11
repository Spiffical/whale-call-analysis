import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis import e121_multi_pairwise_refinement_report as e121  # noqa: E402


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestE121MultiPairwiseRefinementReport(unittest.TestCase):
    def test_multiple_pairwise_specialists_refine_base_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            bm_bp = root / "pair_bm_bp"
            bp_mn = root / "pair_bp_mn"
            for run_dir in (base, bm_bp, bp_mn):
                (run_dir / "train").mkdir(parents=True)

            (base / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Bm", "species:Mn"]}),
                encoding="utf-8",
            )
            (bm_bp / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bm", "species:Bp"]}),
                encoding="utf-8",
            )
            (bp_mn / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Mn"]}),
                encoding="utf-8",
            )

            base_rows = [
                {"item_id": "bm", "true_class_index": "2", "pred_class_index": "1"},
                {"item_id": "mn", "true_class_index": "3", "pred_class_index": "1"},
                {"item_id": "bp", "true_class_index": "1", "pred_class_index": "1"},
                {"item_id": "bg", "true_class_index": "0", "pred_class_index": "0"},
            ]
            write_csv(base / "train" / "val_predictions_best_val_rule.csv", base_rows)
            write_csv(base / "train" / "test_predictions_best_val_rule.csv", base_rows)

            bm_bp_rows = [
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
            bp_mn_rows = [
                {
                    "item_id": "mn",
                    "true_class_index": "2",
                    "pred_class_index": "2",
                    "prob__species:Bp": "0.10",
                    "prob__species:Mn": "0.90",
                },
                {
                    "item_id": "bp",
                    "true_class_index": "1",
                    "pred_class_index": "1",
                    "prob__species:Bp": "0.90",
                    "prob__species:Mn": "0.10",
                },
            ]
            write_csv(bm_bp / "train" / "val_predictions_argmax.csv", bm_bp_rows)
            write_csv(bm_bp / "train" / "test_predictions_argmax.csv", bm_bp_rows)
            write_csv(bp_mn / "train" / "val_predictions_argmax.csv", bp_mn_rows)
            write_csv(bp_mn / "train" / "test_predictions_argmax.csv", bp_mn_rows)

            output = root / "out"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "e121_multi_pairwise_refinement_report.py",
                    "--name",
                    "unit",
                    "--base-run-dir",
                    str(base),
                    "--pairwise-run-dir",
                    str(bm_bp),
                    "--pairwise-run-dir",
                    str(bp_mn),
                    "--base-decision-mode",
                    "existing",
                    "--output-dir",
                    str(output),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(e121.main(), 0)
            finally:
                sys.argv = old_argv

            summary = json.loads((output / "e121_summary.json").read_text(encoding="utf-8"))
            test_base = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "pred"][0]
            test_refined = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "refined"][0]
            self.assertGreater(test_refined["macro_f1"], test_base["macro_f1"])
            self.assertEqual(test_refined["cross_species_fp"], 0)
            self.assertEqual({tuple(model["labels"]) for model in summary["pairwise_models"]}, {("species:Bm", "species:Bp"), ("species:Bp", "species:Mn")})

            with (output / "e121_pairwise_coverage.csv").open(newline="", encoding="utf-8") as handle:
                coverage = list(csv.DictReader(handle))
            test_coverage = [row for row in coverage if row["split"] == "test"]
            self.assertEqual({row["pairwise_labels"] for row in test_coverage}, {"species:Bm|species:Bp", "species:Bp|species:Mn"})
            self.assertTrue((output / "e121_multi_pairwise_refinement_report.md").is_file())

    def test_multiple_base_run_dirs_write_ranked_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bm_bp = root / "pair_bm_bp"
            (bm_bp / "train").mkdir(parents=True)
            (bm_bp / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bm", "species:Bp"]}),
                encoding="utf-8",
            )
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
            write_csv(bm_bp / "train" / "val_predictions_argmax.csv", pair_rows)
            write_csv(bm_bp / "train" / "test_predictions_argmax.csv", pair_rows)

            def make_base(name, bm_pred):
                base = root / name
                (base / "train").mkdir(parents=True)
                (base / "train" / "run_summary.json").write_text(
                    json.dumps({"class_ids": ["background", "species:Bp", "species:Bm", "species:Mn"]}),
                    encoding="utf-8",
                )
                rows = [
                    {"item_id": "bm", "true_class_index": "2", "pred_class_index": bm_pred},
                    {"item_id": "bp", "true_class_index": "1", "pred_class_index": "1"},
                    {"item_id": "mn", "true_class_index": "3", "pred_class_index": "3"},
                ]
                write_csv(base / "train" / "val_predictions_best_val_rule.csv", rows)
                write_csv(base / "train" / "test_predictions_best_val_rule.csv", rows)
                return base

            worse = make_base("base_worse", "1")
            better = make_base("base_better", "2")

            output = root / "out"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "e121_multi_pairwise_refinement_report.py",
                    "--name",
                    "multi",
                    "--base-run-dir",
                    str(worse),
                    "--base-run-dir",
                    str(better),
                    "--pairwise-run-dir",
                    str(bm_bp),
                    "--base-decision-mode",
                    "existing",
                    "--output-dir",
                    str(output),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(e121.main(), 0)
            finally:
                sys.argv = old_argv

            with (output / "e121_comparison_rankings.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["base_name"], "base_better")
            self.assertTrue((output / "e121_comparison_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
