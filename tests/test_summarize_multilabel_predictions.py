import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis.summarize_multilabel_predictions import summarize  # noqa: E402
from src.dataset.multilabel import write_csv_rows  # noqa: E402


class TestSummarizeMultilabelPredictions(unittest.TestCase):
    def test_onc_calibrated_summary_uses_onc_thresholds_and_hard_negative_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "validation_predictions.csv"
            test = root / "test_predictions.csv"
            field_rows = [
                {
                    "item_id": "onc-pos",
                    "source_kind": "ONC",
                    "source_dataset": "final2025_onc",
                    "target_label_ids": "species:Oo",
                    "negative_bucket": "",
                    "score__species:Oo": "0.80",
                    "score__species:Mn": "0.10",
                },
                {
                    "item_id": "onc-neg",
                    "source_kind": "ONC",
                    "source_dataset": "final2025_onc",
                    "target_label_ids": "",
                    "negative_bucket": "primary_adjacent_gap",
                    "score__species:Oo": "0.20",
                    "score__species:Mn": "0.10",
                },
                {
                    "item_id": "dclde-pos",
                    "source_kind": "DCLDE",
                    "source_dataset": "dclde",
                    "target_label_ids": "species:Oo",
                    "negative_bucket": "",
                    "score__species:Oo": "0.90",
                    "score__species:Mn": "0.10",
                },
            ]
            write_csv_rows(val, field_rows)
            write_csv_rows(
                test,
                [
                    {
                        "item_id": "onc-test-pos",
                        "source_kind": "ONC",
                        "source_dataset": "final2025_onc",
                        "target_label_ids": "species:Oo",
                        "negative_bucket": "",
                        "score__species:Oo": "0.75",
                        "score__species:Mn": "0.10",
                    },
                    {
                        "item_id": "onc-test-neg",
                        "source_kind": "ONC",
                        "source_dataset": "final2025_onc",
                        "target_label_ids": "",
                        "negative_bucket": "primary_adjacent_gap",
                        "score__species:Oo": "0.85",
                        "score__species:Mn": "0.10",
                    },
                ],
            )

            summary = summarize(
                validation_csv=val,
                test_csv=test,
                output_dir=root / "out",
                calibration_source_kind="ONC",
                eval_source_kind="ONC",
                label_ids=("species:Oo", "species:Mn"),
            )

            self.assertEqual(summary["onc_validation_thresholds"]["species:Oo"]["support"], 1)
            self.assertEqual(summary["onc_test_metrics"]["samples"], 2)
            self.assertTrue((root / "out/onc_calibrated_test_hard_negative_fp.csv").exists())
            payload = json.loads((root / "out/onc_calibrated_metrics_summary.json").read_text())
            self.assertEqual(payload["eval_source_kind"], "ONC")
            with (root / "out/onc_calibrated_test_hard_negative_fp.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["negative_bucket"], "primary_adjacent_gap")


if __name__ == "__main__":
    unittest.main()
