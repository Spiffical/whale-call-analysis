import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis import e24_collect_expert_hparam_report as e24  # noqa: E402


class TestE24CollectExpertHparamReport(unittest.TestCase):
    def test_selected_ensemble_examples_use_calibrated_thresholds(self):
        summary = {
            "onc_validation_thresholds": {
                "species:Bp": {"threshold": 0.5},
                "species:Bm": {"threshold": 0.4},
                "species:Mn": {"threshold": 0.6},
            }
        }
        rows = [
            {
                "item_id": "fin_tp",
                "source_kind": "ONC",
                "source_audio": "fin.wav",
                "target_label_ids": "species:Bp",
                "score__species:Bp": "0.90",
                "score__species:Bm": "0.10",
                "score__species:Mn": "0.20",
            },
            {
                "item_id": "fin_as_blue",
                "source_kind": "ONC",
                "source_audio": "cross.wav",
                "target_label_ids": "species:Bp",
                "score__species:Bp": "0.10",
                "score__species:Bm": "0.80",
                "score__species:Mn": "0.20",
            },
            {
                "item_id": "bg_as_humpback",
                "source_kind": "ONC",
                "source_audio": "bg.wav",
                "target_label_ids": "",
                "negative_bucket": "ship_noise",
                "score__species:Bp": "0.10",
                "score__species:Bm": "0.10",
                "score__species:Mn": "0.70",
            },
        ]

        examples = e24.selected_ensemble_examples(rows, summary=summary, max_per_group=5)
        by_case = {(row["label_id"], row["case_type"], row["item_id"]) for row in examples}

        self.assertIn(("species:Bp", "true_positive", "fin_tp"), by_case)
        self.assertIn(("species:Bp", "false_negative", "fin_as_blue"), by_case)
        self.assertIn(("species:Bm", "cross_species_false_positive", "fin_as_blue"), by_case)
        self.assertIn(("species:Mn", "background_false_positive", "bg_as_humpback"), by_case)
        cross = next(row for row in examples if row["item_id"] == "fin_as_blue" and row["label_id"] == "species:Bm")
        self.assertEqual(cross["pred_label_ids"], "species:Bm")
        self.assertAlmostEqual(float(cross["threshold"]), 0.4)
        self.assertAlmostEqual(float(cross["margin"]), 0.4)

    def test_selected_ensemble_examples_respects_per_group_limit(self):
        summary = {"onc_validation_thresholds": {"species:Bp": {"threshold": 0.5}}}
        rows = [
            {
                "item_id": f"row_{idx}",
                "target_label_ids": "species:Bp",
                "score__species:Bp": str(0.9 - idx * 0.01),
            }
            for idx in range(5)
        ]

        examples = e24.selected_ensemble_examples(
            rows,
            summary=summary,
            label_ids=("species:Bp",),
            max_per_group=2,
        )

        self.assertEqual([row["item_id"] for row in examples], ["row_0", "row_1"])


if __name__ == "__main__":
    unittest.main()
