import unittest

import torch

from src.models.multiband import create_multiband_model


class MultiBandModelTest(unittest.TestCase):
    def test_per_species_head_outputs_one_logit_per_label(self):
        model = create_multiband_model(
            encoder="deepcnn:w8:d2",
            num_classes=3,
            bands=("low", "mid"),
            fusion="gated",
            head_type="per_species",
        )
        out = model(
            {
                "low": torch.randn(2, 1, 16, 12),
                "mid": torch.randn(2, 1, 16, 12),
            }
        )
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_shared_head_still_supports_mean_logits(self):
        model = create_multiband_model(
            encoder="deepcnn:w8:d2",
            num_classes=2,
            bands=("low", "mid"),
            fusion="mean_logits",
            head_type="shared",
        )
        out = model(
            {
                "low": torch.randn(4, 1, 16, 12),
                "mid": torch.randn(4, 1, 16, 12),
            }
        )
        self.assertEqual(tuple(out.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
