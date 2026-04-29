import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.mat_dataset import (
    _choose_start_idx,
    _sample_positive_crop_fraction,
    _start_from_fraction,
)


class TestPositiveCropModes(unittest.TestCase):
    def test_edge_mode_samples_only_edge_bands(self):
        rng = np.random.default_rng(1337)
        samples = [
            _sample_positive_crop_fraction(rng, center_bias_sigma_frac=0.25, positive_crop_mode="edge")
            for _ in range(256)
        ]
        self.assertTrue(all(sample <= 0.25 or sample >= 0.75 for sample in samples))
        self.assertTrue(any(sample < 0.10 for sample in samples))
        self.assertTrue(any(sample > 0.90 for sample in samples))

    def test_edge_mix_samples_both_edges_and_center(self):
        rng = np.random.default_rng(2025)
        samples = [
            _sample_positive_crop_fraction(rng, center_bias_sigma_frac=0.25, positive_crop_mode="edge_mix")
            for _ in range(512)
        ]
        edge_count = sum(sample <= 0.25 or sample >= 0.75 for sample in samples)
        center_count = sum(0.35 <= sample <= 0.65 for sample in samples)
        self.assertGreater(edge_count, 0)
        self.assertGreater(center_count, 0)
        self.assertGreater(edge_count, center_count)

    def test_eval_without_augment_stays_centered_even_in_edge_mode(self):
        rng = np.random.default_rng(42)
        start = _choose_start_idx(
            T=391,
            crop=96,
            split="val",
            is_positive=True,
            center_bias_sigma_frac=0.25,
            positive_crop_mode="edge",
            rng=rng,
            augment_eval=False,
        )
        self.assertEqual(start, _start_from_fraction(391, 96, 0.5))

    def test_invalid_positive_crop_mode_raises(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(ValueError):
            _sample_positive_crop_fraction(
                rng,
                center_bias_sigma_frac=0.25,
                positive_crop_mode="not_a_mode",
            )


if __name__ == "__main__":
    unittest.main()
