import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis import e124_compare_production_candidates as e124  # noqa: E402


class TestE124CompareProductionCandidates(unittest.TestCase):
    def write_summary(self, path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def read_csv(self, path: Path):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def write_csv(self, path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_ranks_by_f1_then_production_false_positive_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e121_summary = root / "e121" / "e121_summary.json"
            e122_summary = root / "e122" / "e122_summary.json"
            base_metric = {
                "model": "base",
                "split": "test",
                "prediction": "pred",
                "rows": 100,
                "macro_f1": 0.60,
                "micro_f1": 0.62,
                "micro_precision": 0.60,
                "micro_recall": 0.64,
                "cross_species_fp": 5,
                "background_fp": 3,
                "species_as_background_fn": 8,
            }
            self.write_summary(
                e121_summary,
                {
                    "name": "multi_pairwise",
                    "pairwise_models": [{"name": "BmBp"}, {"name": "BpMn"}],
                    "metric_labels": ["species:Bp", "species:Bm", "species:Mn"],
                    "model_metrics": [
                        base_metric,
                        {
                            **base_metric,
                            "prediction": "refined",
                            "macro_f1": 0.80,
                            "micro_f1": 0.81,
                            "cross_species_fp": 4,
                            "background_fp": 2,
                        },
                    ],
                    "outputs": {"report": str(root / "e121" / "report.md"), "examples": str(root / "e121" / "examples.csv")},
                },
            )
            self.write_summary(
                e122_summary,
                {
                    "name": "two_stage",
                    "gate_threshold": 0.4,
                    "metric_labels": ["species:Bp", "species:Bm", "species:Mn"],
                    "model_metrics": [
                        base_metric,
                        {
                            **base_metric,
                            "prediction": "two_stage",
                            "macro_f1": 0.80,
                            "micro_f1": 0.81,
                            "cross_species_fp": 1,
                            "background_fp": 0,
                        },
                    ],
                    "outputs": {"report": str(root / "e122" / "report.md"), "examples": str(root / "e122" / "examples.csv")},
                },
            )

            out = root / "out"
            result = e124.build_leaderboard([("", e121_summary), ("", e122_summary)], out, "Unit Leaderboard")
            rows = self.read_csv(out / "e124_candidate_leaderboard.csv")
            examples = self.read_csv(out / "e124_candidate_examples.csv")

            self.assertEqual(rows[0]["candidate"], "two_stage")
            self.assertEqual(rows[0]["experiment"], "E122")
            self.assertEqual(rows[0]["selected_prediction"], "two_stage")
            self.assertEqual(rows[0]["cross_species_fp"], "1")
            self.assertEqual(rows[0]["delta_macro_f1"], "0.20000000000000007")
            self.assertEqual(examples[0]["candidate"], "two_stage")
            self.assertEqual(examples[0]["example_status"], "missing_examples_path")
            self.assertTrue(Path(result["report"]).is_file())
            self.assertIn("same common ONC test rows", Path(result["report"]).read_text(encoding="utf-8"))
            self.assertIn("candidate_examples_csv", result)

    def test_supports_e26_diagnostic_summary_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "e26" / "diagnostic_summary.json"
            self.write_summary(
                summary,
                {
                    "original_summary": {
                        "samples": 50,
                        "macro_f1": 0.55,
                        "micro_f1": 0.56,
                        "precision": 0.57,
                        "recall": 0.58,
                        "fp": 9,
                        "fn": 11,
                        "hard_fp": 4,
                    },
                    "common_summary": {
                        "samples": 50,
                        "macro_f1": 0.70,
                        "micro_f1": 0.71,
                        "precision": 0.72,
                        "recall": 0.73,
                        "fp": 6,
                        "fn": 7,
                        "hard_fp": 2,
                    },
                },
            )
            out = root / "out"
            e124.build_leaderboard([("common_test", summary)], out, "Unit Leaderboard")
            rows = self.read_csv(out / "e124_candidate_leaderboard.csv")

            self.assertEqual(rows[0]["candidate"], "common_test")
            self.assertEqual(rows[0]["experiment"], "E26")
            self.assertEqual(rows[0]["selected_prediction"], "common_thresholds")
            self.assertEqual(rows[0]["cross_species_fp"], "4")
            self.assertEqual(rows[0]["background_fp"], "2")
            self.assertEqual(rows[0]["species_as_background_fn"], "7")
            payload = json.loads((out / "e124_candidate_leaderboard.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["candidates"][0]["baseline_prediction"], "original_thresholds")

    def test_supports_e129_calibrated_ssamba_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "e129" / "e129_summary.json"
            examples = root / "e129" / "e129_examples.csv"
            self.write_csv(examples, [{"bucket": "cross_species_error", "item_id": "x1"}])
            base_metric = {
                "model": "ssamba",
                "split": "test",
                "prediction": "pred",
                "rows": 100,
                "macro_f1": 0.41,
                "micro_f1": 0.62,
                "micro_precision": 0.61,
                "micro_recall": 0.63,
                "cross_species_fp": 22,
                "background_fp": 8,
                "species_as_background_fn": 12,
            }
            self.write_summary(
                summary,
                {
                    "name": "E129_ssl_multiclass",
                    "model_dir": str(root / "model"),
                    "checkpoint": str(root / "model" / "models" / "ft-avgtok_best_checkpoint.pth"),
                    "dataset_h5": str(root / "eval.h5"),
                    "metric_labels": ["species:Bm", "species:Bp", "species:Mn"],
                    "model_metrics": [
                        base_metric,
                        {
                            **base_metric,
                            "prediction": "calibrated",
                            "macro_f1": 0.56,
                            "micro_f1": 0.70,
                            "cross_species_fp": 12,
                            "background_fp": 4,
                            "species_as_background_fn": 10,
                        },
                    ],
                    "outputs": {
                        "report": str(root / "e129" / "e129_ssamba_multiclass_production_report.md"),
                        "examples": str(examples),
                    },
                },
            )

            out = root / "out"
            e124.build_leaderboard([("ssamba", summary)], out, "Unit Leaderboard")
            rows = self.read_csv(out / "e124_candidate_leaderboard.csv")

            self.assertEqual(rows[0]["candidate"], "ssamba")
            self.assertEqual(rows[0]["experiment"], "E129")
            self.assertEqual(rows[0]["selected_prediction"], "calibrated")
            self.assertEqual(rows[0]["baseline_prediction"], "pred")
            self.assertEqual(rows[0]["cross_species_fp"], "12")
            self.assertEqual(rows[0]["background_fp"], "4")
            self.assertEqual(rows[0]["species_as_background_fn"], "10")
            self.assertAlmostEqual(float(rows[0]["delta_macro_f1"]), 0.15)
            examples_rows = self.read_csv(out / "e124_candidate_examples.csv")
            self.assertEqual(examples_rows[0]["example_status"], "examples_csv")
            self.assertEqual(examples_rows[0]["bucket"], "cross_species_error")

    def test_supports_e27_ensemble_ranking_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensemble_dir = root / "e27" / "ensembles" / "ensemble_0002"
            ensemble_dir.mkdir(parents=True)
            self.write_csv(
                ensemble_dir / "selected_examples.csv",
                [
                    {"example_group": "cross_species_false_positive", "item_id": "x1"},
                    {"example_group": "false_negative", "item_id": "x2"},
                ],
            )
            rankings = root / "e27" / "e27_ensemble_rankings.csv"
            self.write_csv(
                rankings,
                [
                    {
                        "ensemble": "ensemble_0001",
                        "macro_f1": 0.72,
                        "micro_f1": 0.76,
                        "precision": 0.77,
                        "recall": 0.75,
                        "tp": 70,
                        "fp": 10,
                        "fn": 8,
                        "hard_fp": 4,
                        "hard_total": 20,
                        "hard_fp_rate": 0.20,
                        "ensemble_dir": str(root / "e27" / "ensembles" / "ensemble_0001"),
                    },
                    {
                        "ensemble": "ensemble_0002",
                        "macro_f1": 0.78,
                        "micro_f1": 0.79,
                        "precision": 0.80,
                        "recall": 0.78,
                        "tp": 76,
                        "fp": 7,
                        "fn": 5,
                        "hard_fp": 2,
                        "hard_total": 20,
                        "hard_fp_rate": 0.10,
                        "ensemble_dir": str(ensemble_dir),
                    },
                ],
            )

            out = root / "out"
            e124.build_leaderboard([("ovr", rankings)], out, "Unit Leaderboard")
            rows = self.read_csv(out / "e124_candidate_leaderboard.csv")

            self.assertEqual(rows[0]["candidate"], "ovr")
            self.assertEqual(rows[0]["experiment"], "E27")
            self.assertEqual(rows[0]["selected_prediction"], "ensemble_0002")
            self.assertEqual(rows[0]["rows"], "81")
            self.assertEqual(rows[0]["cross_species_fp"], "5")
            self.assertEqual(rows[0]["background_fp"], "2")
            self.assertEqual(rows[0]["species_as_background_fn"], "5")
            self.assertEqual(rows[0]["background_fp_rate"], "0.1")
            self.assertEqual(rows[0]["report"], str(root / "e27" / "e27_one_vs_rest_report.md"))
            self.assertEqual(rows[0]["per_species_csv"], str(root / "e27" / "e27_individual_metrics.csv"))
            self.assertEqual(rows[0]["examples_csv"], str(ensemble_dir))
            examples = self.read_csv(out / "e124_candidate_examples.csv")
            self.assertEqual(examples[0]["example_status"], "examples_dir")
            self.assertEqual(examples[0]["candidate"], "ovr")
            self.assertEqual(examples[0]["source_examples_csv"], str(ensemble_dir / "selected_examples.csv"))

            args = e124.build_parser().parse_args(["--summary-csv", str(rankings), "--output-dir", str(out / "cli")])
            self.assertEqual(e124.collect_summary_paths(args), [("", rankings.resolve())])

    def test_aggregates_candidate_examples_from_csv_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples_csv = root / "e122" / "examples.csv"
            self.write_csv(
                examples_csv,
                [
                    {"bucket": "cross_species_error", "item_id": "cross_1"},
                    {"bucket": "cross_species_error", "item_id": "cross_2"},
                    {"bucket": "residual_background_fp", "item_id": "bg_1"},
                    {"bucket": "correct_species", "item_id": "tp_1"},
                ],
            )
            summary = root / "e122" / "e122_summary.json"
            self.write_summary(
                summary,
                {
                    "name": "two_stage",
                    "gate_threshold": 0.4,
                    "metric_labels": ["species:Bp", "species:Bm", "species:Mn"],
                    "model_metrics": [
                        {
                            "model": "base",
                            "split": "test",
                            "prediction": "two_stage",
                            "rows": 10,
                            "macro_f1": 0.5,
                            "micro_f1": 0.8,
                            "micro_precision": 0.75,
                            "micro_recall": 0.85,
                            "cross_species_fp": 2,
                            "background_fp": 1,
                            "species_as_background_fn": 1,
                        },
                    ],
                    "outputs": {"examples": str(examples_csv)},
                },
            )

            out = root / "out"
            result = e124.build_leaderboard(
                [("", summary)],
                out,
                "Unit Leaderboard",
                max_examples_per_candidate=3,
            )
            examples = self.read_csv(out / "e124_candidate_examples.csv")

            self.assertEqual(result["candidate_examples_csv"], str(out / "e124_candidate_examples.csv"))
            self.assertEqual(len(examples), 3)
            self.assertEqual({row["example_status"] for row in examples}, {"examples_csv"})
            self.assertEqual({row["candidate"] for row in examples}, {"two_stage"})
            self.assertIn("source_example_row", examples[0])

    def test_can_append_living_ledger_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "e122" / "e122_summary.json"
            ledger_path = root / "ledger.md"
            self.write_summary(
                summary,
                {
                    "name": "two_stage_candidate",
                    "gate_threshold": 0.4,
                    "metric_labels": ["species:Bp", "species:Bm", "species:Mn"],
                    "model_metrics": [
                        {
                            "model": "base",
                            "split": "test",
                            "prediction": "two_stage",
                            "rows": 10,
                            "macro_f1": 0.5,
                            "micro_f1": 0.8,
                            "micro_precision": 0.75,
                            "micro_recall": 0.85,
                            "cross_species_fp": 2,
                            "background_fp": 1,
                            "species_as_background_fn": 1,
                        },
                    ],
                    "outputs": {"report": str(root / "e122" / "report.md")},
                },
            )

            result = e124.build_leaderboard(
                [("", summary)],
                root / "out",
                "Unit Ledger Leaderboard",
                ledger_path=ledger_path,
                training_set="unit candidate train sets",
                validation_set="unit validation sets",
                test_set="unit common ONC test",
                evaluation_note="unit production comparison",
            )

            self.assertEqual(result["ledger"], str(ledger_path))
            payload = json.loads((root / "out" / "e124_candidate_leaderboard.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["ledger"], str(ledger_path))
            ledger_text = ledger_path.read_text(encoding="utf-8")
            self.assertIn("Unit Ledger Leaderboard", ledger_text)
            self.assertIn("Training set: unit candidate train sets.", ledger_text)
            self.assertIn("two_stage_candidate", ledger_text)


if __name__ == "__main__":
    unittest.main()
