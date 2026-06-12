import unittest

from scripts.analysis import e119_pairwise_refinement_report as e119
from scripts.analysis import e129_ssamba_multiclass_production_report as e129


class TestE129SsambaMulticlassProductionReport(unittest.TestCase):
    def test_softmax_is_normalized(self):
        probs = e129.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[2], probs[1])
        self.assertGreater(probs[1], probs[0])

    def test_true_label_from_h5_label_string(self):
        class_ids = ["background", "species:Bm", "species:Bp", "species:Mn"]
        self.assertEqual(e129.true_label_from_h5_label_string("normal", class_ids), "background")
        self.assertEqual(e129.true_label_from_h5_label_string("Bm", class_ids), "species:Bm")
        self.assertEqual(e129.true_label_from_h5_label_string("species:Bp", class_ids), "species:Bp")
        self.assertEqual(e129.true_label_from_h5_label_string("unknown", class_ids), "background")

    def test_examples_include_cross_species_and_background_fp(self):
        class_ids = ["background", "species:Bm", "species:Bp", "species:Mn"]
        rows = [
            {
                "_true": "species:Bm",
                "_pred": "species:Bp",
                "item_id": "cross",
                "prob__species:Bp": "0.80",
            },
            {
                "_true": "background",
                "_pred": "species:Mn",
                "item_id": "bgfp",
                "prob__species:Mn": "0.70",
            },
            {
                "_true": "species:Mn",
                "_pred": "background",
                "item_id": "miss",
                "prob__background": "0.90",
            },
        ]
        examples = e129.example_rows(rows, "_pred", class_ids, limit_per_bucket=5)
        buckets = {row["item_id"]: row["bucket"] for row in examples}
        self.assertEqual(buckets["cross"], "cross_species_error")
        self.assertEqual(buckets["bgfp"], "background_false_positive")
        self.assertEqual(buckets["miss"], "species_as_background_false_negative")

    def test_metrics_match_e119_production_accounting(self):
        labels = ["species:Bm", "species:Bp", "species:Mn"]
        rows = [
            {"_true": "species:Bm", "_pred": "species:Bm"},
            {"_true": "species:Bm", "_pred": "species:Bp"},
            {"_true": "background", "_pred": "species:Mn"},
            {"_true": "species:Mn", "_pred": "background"},
        ]
        metrics = e119.species_metrics(rows, "_pred", labels)
        self.assertEqual(metrics["cross_species_fp"], 1)
        self.assertEqual(metrics["background_fp"], 1)
        self.assertEqual(metrics["species_as_background_fn"], 1)


if __name__ == "__main__":
    unittest.main()
