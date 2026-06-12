import math
import tempfile
import unittest
from pathlib import Path

from scripts.analysis import e128_export_ssamba_binary_gate_predictions as e128


class TestE128ExportSSAMBABinaryGatePredictions(unittest.TestCase):
    def test_species_labels_from_short_and_full_h5_label_strings(self):
        positives = ["species:Bp", "species:Bm", "species:Mn"]
        self.assertEqual(e128.species_labels_from_h5_label_string("Bp", positives), ["species:Bp"])
        self.assertEqual(e128.species_labels_from_h5_label_string("species:Bm", positives), ["species:Bm"])
        self.assertEqual(e128.species_labels_from_h5_label_string("normal", positives), [])
        self.assertEqual(
            e128.species_labels_from_h5_label_string("Bm;Mn;normal", positives),
            ["species:Bm", "species:Mn"],
        )

    def test_probability_from_binary_and_two_class_logits(self):
        self.assertAlmostEqual(e128.probability_from_logits([0.0]), 0.5)
        self.assertAlmostEqual(e128.probability_from_logits([2.0]), 1.0 / (1.0 + math.exp(-2.0)))
        self.assertAlmostEqual(e128.probability_from_logits([0.0, 0.0]), 0.5)
        self.assertGreater(e128.probability_from_logits([-2.0, 2.0]), 0.98)

    def test_output_name_for_split(self):
        self.assertEqual(e128.output_name_for_split("val"), "validation_predictions.csv")
        self.assertEqual(e128.output_name_for_split("test"), "test_predictions.csv")
        self.assertEqual(e128.output_name_for_split("holdout"), "holdout_predictions.csv")

    def test_write_csv_handles_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            e128.write_csv(path, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
