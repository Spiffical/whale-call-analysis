import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis import multispecies_readiness_audit as audit  # noqa: E402


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestMultispeciesReadinessAudit(unittest.TestCase):
    def test_audits_complete_leaderboard_binary_gate_and_h5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = root / "leaderboard" / "examples.csv"
            write_csv(
                examples,
                [
                    {
                        "candidate_rank": "1",
                        "candidate": "winner",
                        "example_status": "examples_csv",
                        "item_id": "row1",
                    }
                ],
            )
            leaderboard = root / "leaderboard" / "leaderboard.json"
            leaderboard.write_text(
                json.dumps(
                    {
                        "title": "Unit Leaderboard",
                        "leaderboard_json": str(leaderboard),
                        "report": str(root / "leaderboard" / "report.md"),
                        "candidate_examples_csv": str(examples),
                        "candidates": [
                            {
                                "rank": 1,
                                "candidate": "winner",
                                "macro_f1": 0.5,
                                "micro_f1": 0.9,
                                "precision": 0.88,
                                "recall": 0.91,
                                "cross_species_fp": 2,
                                "background_fp": 1,
                                "species_as_background_fn": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            gate_examples = root / "gate" / "examples.csv"
            write_csv(gate_examples, [{"bucket": "true_positive", "item_id": "gate1"}])
            gate = root / "gate" / "summary.json"
            gate.write_text(
                json.dumps(
                    {
                        "name": "gate",
                        "metrics": [
                            {"split": "val", "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
                            {"split": "test", "precision": 0.9, "recall": 0.8, "f1": 0.85, "accuracy": 0.7},
                        ],
                        "positive_labels": ["species:Bp", "species:Bm", "species:Mn"],
                        "test_background_false_positive_rate": 0.1,
                        "test_per_species_gate_recall": {
                            "species:Bp": 1.0,
                            "species:Bm": 0.8,
                            "species:Mn": 0.7,
                        },
                        "outputs": {"examples": str(gate_examples), "report": str(root / "gate" / "report.md")},
                    }
                ),
                encoding="utf-8",
            )
            h5 = root / "h5" / "audit.json"
            h5.parent.mkdir(parents=True, exist_ok=True)
            h5.write_text(
                json.dumps(
                    {
                        "input_h5": "/tmp/unit.h5",
                        "summary": {"normal_train_rows": 12000, "normal_train_months": 18, "normal_months": 18},
                        "quality_checks": [{"check": "normal_train_rows", "passed": True}],
                        "outputs": {"report": str(root / "h5" / "report.md")},
                    }
                ),
                encoding="utf-8",
            )
            ledger = root / "ledger.md"
            ledger.write_text(
                "\n".join([str(leaderboard), "Unit Leaderboard", "gate", "/tmp/unit.h5"]),
                encoding="utf-8",
            )

            result = audit.run_audit(
                output_dir=root / "out",
                ledger_path=ledger,
                require_ledger=True,
                leaderboard_jsons=[leaderboard],
                binary_gate_summary_jsons=[gate],
                h5_audit_jsons=[h5],
                min_normal_train=10000,
                min_normal_months=12,
                title="Unit Audit",
            )

            self.assertEqual(result["status_counts"].get("FAIL", 0), 0)
            self.assertTrue((root / "out" / "multispecies_readiness_audit.md").is_file())
            self.assertTrue((root / "out" / "multispecies_readiness_audit_checks.csv").is_file())

    def test_flags_missing_examples_and_weak_h5_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = root / "leaderboard" / "examples.csv"
            write_csv(
                examples,
                [
                    {
                        "candidate_rank": "1",
                        "candidate": "winner",
                        "example_status": "missing_examples_path",
                    }
                ],
            )
            leaderboard = root / "leaderboard" / "leaderboard.json"
            leaderboard.write_text(
                json.dumps(
                    {
                        "title": "Unit Leaderboard",
                        "candidate_examples_csv": str(examples),
                        "candidates": [
                            {
                                "rank": 1,
                                "candidate": "winner",
                                "macro_f1": 0.5,
                                "micro_f1": 0.9,
                                "precision": 0.88,
                                "recall": 0.91,
                                "cross_species_fp": 2,
                                "background_fp": 1,
                                "species_as_background_fn": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            h5 = root / "h5" / "audit.json"
            h5.parent.mkdir(parents=True, exist_ok=True)
            h5.write_text(
                json.dumps(
                    {
                        "summary": {"normal_train_rows": 8, "normal_train_months": 2, "normal_months": 4},
                        "quality_checks": [{"check": "normal_train_rows", "passed": False}],
                        "outputs": {},
                    }
                ),
                encoding="utf-8",
            )

            result = audit.run_audit(
                output_dir=root / "out",
                ledger_path=None,
                require_ledger=False,
                leaderboard_jsons=[leaderboard],
                binary_gate_summary_jsons=[],
                h5_audit_jsons=[h5],
                min_normal_train=10000,
                min_normal_months=12,
                title="Unit Audit",
            )
            failed = {(row["artifact_type"], row["check"]) for row in result["checks"] if row["status"] == "FAIL"}

            self.assertIn(("leaderboard", "top_candidate_examples_are_row_level"), failed)
            self.assertIn(("h5_audit", "normal_train_rows"), failed)
            self.assertIn(("h5_audit", "normal_train_months"), failed)
            self.assertIn(("h5_audit", "quality_checks_passed"), failed)

    def test_flags_binary_gate_missing_gate_specific_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate_examples = root / "gate" / "examples.csv"
            write_csv(gate_examples, [{"bucket": "true_positive", "item_id": "gate1"}])
            gate = root / "gate" / "summary.json"
            gate.parent.mkdir(parents=True, exist_ok=True)
            gate.write_text(
                json.dumps(
                    {
                        "name": "weak_gate_summary",
                        "metrics": [
                            {"split": "val", "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
                            {"split": "test", "precision": 0.9, "recall": 0.8, "f1": 0.85, "accuracy": 0.7},
                        ],
                        "positive_labels": ["species:Bp", "species:Bm"],
                        "outputs": {"examples": str(gate_examples)},
                    }
                ),
                encoding="utf-8",
            )

            result = audit.run_audit(
                output_dir=root / "out",
                ledger_path=None,
                require_ledger=False,
                leaderboard_jsons=[],
                binary_gate_summary_jsons=[gate],
                h5_audit_jsons=[],
                min_normal_train=10000,
                min_normal_months=12,
                title="Unit Audit",
            )
            failed = {(row["artifact_type"], row["check"]) for row in result["checks"] if row["status"] == "FAIL"}

            self.assertIn(("binary_gate", "test_background_false_positive_rate"), failed)
            self.assertIn(("binary_gate", "test_per_species_gate_recall_present"), failed)


if __name__ == "__main__":
    unittest.main()
