import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference.evaluate_part2_predictions import _select_operating_points


class TestPart2OperatingPoints(unittest.TestCase):
    def test_selector_prefers_coverage_for_primary_rows(self):
        rows = [
            {
                "tag": "strict_best",
                "f1": 0.30,
                "precision": 0.80,
                "recall": 0.20,
                "merged_region_f1": 0.60,
                "merged_region_precision": 0.75,
                "merged_region_recall": 0.55,
                "raw_window_precision": 0.70,
                "raw_window_recall": 0.65,
                "prediction_count": 2000,
            },
            {
                "tag": "coverage_best",
                "f1": 0.22,
                "precision": 0.68,
                "recall": 0.14,
                "merged_region_f1": 0.81,
                "merged_region_precision": 0.73,
                "merged_region_recall": 0.91,
                "raw_window_precision": 0.63,
                "raw_window_recall": 0.88,
                "prediction_count": 5000,
            },
            {
                "tag": "window_best",
                "f1": 0.15,
                "precision": 0.61,
                "recall": 0.08,
                "merged_region_f1": 0.75,
                "merged_region_precision": 0.65,
                "merged_region_recall": 0.86,
                "raw_window_precision": 0.74,
                "raw_window_recall": 0.95,
                "prediction_count": 9000,
            },
        ]

        selected = _select_operating_points(rows)

        self.assertEqual(selected["best_f1"]["tag"], "coverage_best")
        self.assertEqual(selected["high_recall"]["tag"], "coverage_best")
        self.assertEqual(selected["high_precision"]["tag"], "strict_best")
        self.assertEqual(selected["best_strict_f1"]["tag"], "strict_best")
        self.assertEqual(selected["best_window_recall"]["tag"], "window_best")


if __name__ == "__main__":
    unittest.main()
