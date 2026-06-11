import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis import e122_two_stage_gate_report as e122_report  # noqa: E402
from scripts.data.multilabel import build_e122_two_stage_gate_manifest as e122_manifest  # noqa: E402


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


class TestE122TwoStageGate(unittest.TestCase):
    def test_gate_manifest_rewrites_target_species_to_binary_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.csv"
            write_csv(
                src,
                [
                    {"item_id": "bp", "split": "train", "source_kind": "ONC", "label_ids": "species:Bp"},
                    {"item_id": "bm", "split": "val", "source_kind": "ONC", "species": "Bm", "label_ids": ""},
                    {"item_id": "oo", "split": "test", "source_kind": "ONC", "label_ids": "species:Oo"},
                    {"item_id": "bg", "split": "test", "source_kind": "ONC", "label_ids": ""},
                    {"item_id": "external", "split": "test", "source_kind": "DCLDE", "label_ids": "species:Mn"},
                ],
            )
            out = root / "gate"
            summary = e122_manifest.build_gate_manifest(
                input_manifest=src,
                output_csv=out / "standardized_manifest.csv",
                output_vocab=out / "label_vocabulary.json",
                output_summary=out / "manifest_counts.json",
                positive_labels=["species:Bp", "species:Bm", "species:Mn"],
                gate_label="task:whale_call",
                source_kinds=["ONC"],
            )

            self.assertEqual(summary["rows"], 4)
            self.assertEqual(summary["positive_rows"], 2)
            with (out / "standardized_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = {row["item_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["bp"]["label_ids"], "task:whale_call")
            self.assertEqual(rows["bm"]["label_ids"], "task:whale_call")
            self.assertEqual(rows["oo"]["label_ids"], "")
            self.assertEqual(rows["bg"]["is_background"], "1")
            vocab = json.loads((out / "label_vocabulary.json").read_text(encoding="utf-8"))
            self.assertEqual(vocab["labels"][0]["id"], "task:whale_call")

    def test_two_stage_report_tunes_gate_and_improves_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            gate = root / "gate"
            (base / "train").mkdir(parents=True)
            (gate / "train").mkdir(parents=True)
            (base / "train" / "run_summary.json").write_text(
                json.dumps({"class_ids": ["background", "species:Bp", "species:Bm", "species:Mn"]}),
                encoding="utf-8",
            )

            base_rows = [
                {
                    "item_id": "bp",
                    "true_class_index": "1",
                    "pred_class_index": "1",
                    "prob__species:Bp": "0.90",
                    "prob__species:Bm": "0.05",
                    "prob__species:Mn": "0.05",
                },
                {
                    "item_id": "bm",
                    "true_class_index": "2",
                    "pred_class_index": "0",
                    "prob__species:Bp": "0.10",
                    "prob__species:Bm": "0.80",
                    "prob__species:Mn": "0.10",
                },
                {
                    "item_id": "mn",
                    "true_class_index": "3",
                    "pred_class_index": "1",
                    "prob__species:Bp": "0.10",
                    "prob__species:Bm": "0.10",
                    "prob__species:Mn": "0.80",
                },
                {
                    "item_id": "bg",
                    "true_class_index": "0",
                    "pred_class_index": "1",
                    "prob__species:Bp": "0.90",
                    "prob__species:Bm": "0.05",
                    "prob__species:Mn": "0.05",
                },
            ]
            gate_rows = [
                {"item_id": "bp", "target_label_ids": "task:whale_call", "score__task:whale_call": "0.90"},
                {"item_id": "bm", "target_label_ids": "task:whale_call", "score__task:whale_call": "0.90"},
                {"item_id": "mn", "target_label_ids": "task:whale_call", "score__task:whale_call": "0.90"},
                {"item_id": "bg", "target_label_ids": "", "score__task:whale_call": "0.10"},
            ]
            write_csv(base / "train" / "validation_predictions_best_val_rule.csv", base_rows)
            write_csv(base / "train" / "test_predictions_best_val_rule.csv", base_rows)
            write_csv(gate / "train" / "validation_predictions.csv", gate_rows)
            write_csv(gate / "train" / "test_predictions.csv", gate_rows)

            output = root / "out"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "e122_two_stage_gate_report.py",
                    "--name",
                    "unit",
                    "--base-run-dir",
                    str(base),
                    "--gate-run-dir",
                    str(gate),
                    "--base-decision-mode",
                    "existing",
                    "--species-stage-mode",
                    "force_species_argmax",
                    "--output-dir",
                    str(output),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(e122_report.main(), 0)
            finally:
                sys.argv = old_argv

            summary = json.loads((output / "e122_summary.json").read_text(encoding="utf-8"))
            base_metric = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "pred"][0]
            two_metric = [row for row in summary["model_metrics"] if row["split"] == "test" and row["prediction"] == "two_stage"][0]
            self.assertGreater(two_metric["macro_f1"], base_metric["macro_f1"])
            self.assertEqual(two_metric["background_fp"], 0)
            self.assertEqual(two_metric["cross_species_fp"], 0)
            self.assertTrue((output / "e122_examples.csv").is_file())
            report = (output / "e122_two_stage_gate_report.md").read_text(encoding="utf-8")
            self.assertIn("binary whale-call detector", report)


if __name__ == "__main__":
    unittest.main()
