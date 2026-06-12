import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis import multispecies_experiment_ledger as ledger


class TestMultispeciesExperimentLedger(unittest.TestCase):
    def test_appends_and_replaces_marked_binary_gate_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.md"
            ledger_path.write_text(
                "# Multispecies Experiment Results Ledger\n\n"
                "## Experiment Ledger\n\n"
                "## Immediate Next Entries To Add\n\n"
                "- placeholder\n",
                encoding="utf-8",
            )
            summary = {
                "name": "unit_gate",
                "threshold": 0.5,
                "inputs": {"val_predictions": "val.csv", "test_predictions": "test.csv"},
                "metrics": [
                    {
                        "split": "val",
                        "rows": 4,
                        "precision": 1.0,
                        "recall": 0.75,
                        "f1": 0.857142857,
                        "accuracy": 0.75,
                        "tp": 3,
                        "fp": 0,
                        "tn": 0,
                        "fn": 1,
                    },
                    {
                        "split": "test",
                        "rows": 5,
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 0.666666667,
                        "accuracy": 0.6,
                        "tp": 2,
                        "fp": 2,
                        "tn": 1,
                        "fn": 0,
                    },
                ],
                "test_breakdown": [
                    {
                        "true_bucket": "background",
                        "support": 3,
                        "detected": 2,
                        "missed": 1,
                        "detection_rate": 0.666666667,
                    }
                ],
                "outputs": {
                    "report": "report.md",
                    "metrics": "metrics.csv",
                    "examples": "examples.csv",
                },
            }

            ledger.append_binary_gate_summary(
                summary=summary,
                summary_path=root / "summary.json",
                ledger_path=ledger_path,
                training_set="unit train",
                validation_set="unit val",
                test_set="unit test",
                evaluation_note="unit evaluation",
                entry_id="unit-gate",
                entry_date="2026-06-12",
            )
            first = ledger_path.read_text(encoding="utf-8")
            self.assertIn("unit_gate: Binary Whale Gate (2026-06-12)", first)
            self.assertIn("Training set: unit train.", first)
            self.assertIn("| test | 5 | 0.5000 | 1.0000 | 0.6667 | 0.6000 | 2 | 2 | 1 | 0 |", first)
            self.assertLess(first.index("unit_gate: Binary Whale Gate"), first.index("## Immediate Next Entries To Add"))

            summary["metrics"][1]["fp"] = 1
            ledger.append_binary_gate_summary(
                summary=summary,
                summary_path=root / "summary.json",
                ledger_path=ledger_path,
                entry_id="unit-gate",
                entry_date="2026-06-13",
            )
            second = ledger_path.read_text(encoding="utf-8")
            self.assertEqual(second.count("BEGIN experiment-ledger-entry:unit-gate"), 1)
            self.assertIn("unit_gate: Binary Whale Gate (2026-06-13)", second)
            self.assertIn("| test | 5 | 0.5000 | 1.0000 | 0.6667 | 0.6000 | 2 | 1 | 1 | 0 |", second)

    def test_cli_binary_gate_appends_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.json"
            ledger_path = root / "ledger.md"
            summary_path.write_text(
                json.dumps(
                    {
                        "name": "cli_gate",
                        "threshold": 0.25,
                        "metrics": [
                            {
                                "split": "test",
                                "rows": 1,
                                "precision": 1.0,
                                "recall": 1.0,
                                "f1": 1.0,
                                "accuracy": 1.0,
                                "tp": 1,
                                "fp": 0,
                                "tn": 0,
                                "fn": 0,
                            }
                        ],
                        "outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            rc = ledger.main(
                [
                    "binary-gate",
                    "--summary-json",
                    str(summary_path),
                    "--ledger-path",
                    str(ledger_path),
                    "--entry-date",
                    "2026-06-12",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertIn("cli_gate: Binary Whale Gate", ledger_path.read_text(encoding="utf-8"))

    def test_appends_leaderboard_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.md"
            leaderboard = {
                "title": "Unit Candidate Leaderboard",
                "report": "leaderboard.md",
                "leaderboard_csv": "leaderboard.csv",
                "leaderboard_json": "leaderboard.json",
                "candidates": [
                    {
                        "rank": 1,
                        "candidate": "candidate_a",
                        "experiment": "E124",
                        "selected_prediction": "common",
                        "macro_f1": 0.5,
                        "micro_f1": 0.9,
                        "precision": 0.88,
                        "recall": 0.91,
                        "cross_species_fp": 12,
                        "background_fp": 3,
                        "species_as_background_fn": 1,
                    }
                ],
            }
            ledger.append_leaderboard_summary(
                leaderboard=leaderboard,
                ledger_path=ledger_path,
                training_set="unit train variants",
                validation_set="unit val variants",
                test_set="unit common ONC test",
                evaluation_note="unit production comparison",
                entry_id="unit-leaderboard",
                entry_date="2026-06-12",
            )
            text = ledger_path.read_text(encoding="utf-8")
            self.assertIn("Unit Candidate Leaderboard (2026-06-12)", text)
            self.assertIn("Training set: unit train variants.", text)
            self.assertIn("| 1 | candidate_a | E124 | common | 0.5000 | 0.9000 | 0.8800 | 0.9100 | 12 | 3 | 1 |", text)

    def test_cli_h5_audit_appends_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_path = root / "audit.json"
            ledger_path = root / "ledger.md"
            audit_path.write_text(
                json.dumps(
                    {
                        "input_h5": "/tmp/e126.h5",
                        "builder_summary_json": "/tmp/e126.summary.json",
                        "summary": {
                            "rows": 12,
                            "normal_rows": 10,
                            "normal_train_rows": 8,
                            "normal_months": 4,
                            "months": 6,
                            "unknown_month_rows": 0,
                            "target_label_counts": {"Bm": 1, "Bp": 1, "Mn": 1},
                            "label_counts": {"normal": 10, "Bm": 1, "Bp": 1, "Mn": 1},
                        },
                        "quality_checks": [
                            {"check": "normal_rows", "value": 10, "threshold": 10, "passed": True},
                            {"check": "normal_months", "value": 4, "threshold": 12, "passed": False},
                        ],
                        "outputs": {"report": "/tmp/report.md", "summary": "/tmp/audit.json"},
                    }
                ),
                encoding="utf-8",
            )
            rc = ledger.main(
                [
                    "h5-audit",
                    "--audit-json",
                    str(audit_path),
                    "--ledger-path",
                    str(ledger_path),
                    "--entry-date",
                    "2026-06-12",
                ]
            )
            self.assertEqual(rc, 0)
            text = ledger_path.read_text(encoding="utf-8")
            self.assertIn("E126 SSL H5 Coverage Audit (2026-06-12)", text)
            self.assertIn("| normal rows | 10 |", text)
            self.assertIn("| normal_months | 4 | 12 | no |", text)

    def test_cli_note_appends_manual_experiment_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.md"
            rc = ledger.main(
                [
                    "note",
                    "--name",
                    "E999 Manual Smoke",
                    "--ledger-path",
                    str(ledger_path),
                    "--training-set",
                    "unit training split",
                    "--validation-set",
                    "unit validation split",
                    "--test-set",
                    "unit common-row test split",
                    "--evaluation-note",
                    "production-style common-row evaluation",
                    "--metric",
                    "Macro F1: 0.5000",
                    "--metric",
                    "Cross-species FP: 12",
                    "--artifact",
                    "Report=/tmp/report.md",
                    "--artifact",
                    "/tmp/examples.csv",
                    "--interpretation",
                    "manual entries can capture partial diagnostics",
                    "--entry-id",
                    "e999-manual-smoke",
                    "--entry-date",
                    "2026-06-12",
                ]
            )
            self.assertEqual(rc, 0)
            text = ledger_path.read_text(encoding="utf-8")
            self.assertIn("E999 Manual Smoke (2026-06-12)", text)
            self.assertIn("Training set: unit training split.", text)
            self.assertIn("- Macro F1: 0.5000", text)
            self.assertIn("- Cross-species FP: 12", text)
            self.assertIn("- Report: `/tmp/report.md`", text)
            self.assertIn("- Artifact: `/tmp/examples.csv`", text)
            self.assertIn("Interpretation: manual entries can capture partial diagnostics.", text)

            rc = ledger.main(
                [
                    "note",
                    "--name",
                    "E999 Manual Smoke",
                    "--ledger-path",
                    str(ledger_path),
                    "--training-set",
                    "unit training split",
                    "--validation-set",
                    "unit validation split",
                    "--test-set",
                    "unit common-row test split",
                    "--evaluation-note",
                    "production-style common-row evaluation",
                    "--interpretation",
                    "already punctuated.",
                    "--entry-id",
                    "e999-manual-smoke",
                    "--entry-date",
                    "2026-06-13",
                ]
            )
            self.assertEqual(rc, 0)
            text = ledger_path.read_text(encoding="utf-8")
            self.assertIn("Interpretation: already punctuated.", text)
            self.assertNotIn("already punctuated..", text)


if __name__ == "__main__":
    unittest.main()
