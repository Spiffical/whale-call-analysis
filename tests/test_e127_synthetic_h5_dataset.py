import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    import h5py
except Exception:
    h5py = None

try:
    from scripts.data.multilabel import build_e127_synthetic_h5_dataset as e127
except Exception:
    e127 = None


@unittest.skipIf(np is None or e127 is None, "numpy is required for E127 augmentation tests")
class TestE127SyntheticH5Dataset(unittest.TestCase):
    def test_frequency_shift_does_not_wrap(self):
        spec = np.zeros((5, 3), dtype=np.float32)
        spec[1, :] = 1.0
        shifted = e127.frequency_shift(spec, 2, fill_value=-1.0)
        self.assertTrue(np.allclose(shifted[3, :], 1.0))
        self.assertTrue(np.allclose(shifted[:2, :], -1.0))

    def test_time_stretch_preserves_shape_and_finiteness(self):
        spec = np.arange(24, dtype=np.float32).reshape(4, 6)
        stretched = e127.time_stretch_to_length(spec, 1.4)
        compressed = e127.time_stretch_to_length(spec, 0.6)
        self.assertEqual(stretched.shape, spec.shape)
        self.assertEqual(compressed.shape, spec.shape)
        self.assertTrue(np.isfinite(stretched).all())
        self.assertTrue(np.isfinite(compressed).all())

    def test_synthesize_spectrogram_returns_params(self):
        rng = np.random.default_rng(7)
        signal = np.ones((8, 10), dtype=np.float32)
        signal[3:5, 4:7] = 4.0
        background = rng.normal(0.0, 0.1, size=(8, 10)).astype(np.float32)
        config = e127.AugmentConfig(seed=7)
        synthetic, params = e127.synthesize_spectrogram(signal, background, rng, config)
        self.assertEqual(synthetic.shape, signal.shape)
        self.assertTrue(np.isfinite(synthetic).all())
        self.assertIn("snr_db", params)
        self.assertIn("freq_shift_bins", params)

    @unittest.skipIf(h5py is None, "h5py is required for H5 round-trip test")
    def test_build_synthetic_h5_appends_training_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "in.h5"
            out = root / "out.h5"
            summary = root / "summary.json"
            string_dtype = h5py.string_dtype(encoding="utf-8")
            specs = np.stack(
                [
                    np.full((6, 6, 1), 0.1, dtype=np.float32),
                    np.full((6, 6, 1), 0.5, dtype=np.float32),
                    np.full((6, 6, 1), 0.9, dtype=np.float32),
                    np.full((6, 6, 1), 0.2, dtype=np.float32),
                ]
            )
            labels = np.asarray(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [0, 0],
                ],
                dtype=np.int8,
            )
            with h5py.File(src, "w") as h5:
                h5.create_dataset("spectrograms", data=specs)
                h5.create_dataset("labels", data=labels)
                h5.create_dataset("label_strings", data=np.asarray(["normal", "Bm", "Mn", "normal"], dtype=object), dtype=string_dtype)
                h5.create_dataset("item_ids", data=np.asarray(["n0", "bm0", "mn0", "n1"], dtype=object), dtype=string_dtype)
                h5.create_dataset("splits", data=np.asarray(["train", "train", "val", "train"], dtype=object), dtype=string_dtype)
                h5.create_dataset("sources", data=np.asarray(["s"] * 4, dtype=object), dtype=string_dtype)
                h5.create_dataset("source_kinds", data=np.asarray(["ONC"] * 4, dtype=object), dtype=string_dtype)
                h5.create_dataset("anomaly_label_names", data=np.asarray(["Bm", "Mn"], dtype=object), dtype=string_dtype)

            result = e127.build_synthetic_h5(
                input_h5=src,
                output_h5=out,
                output_summary=summary,
                target_labels=["Bm"],
                synthetic_per_target=2,
                split="train",
                config=e127.AugmentConfig(seed=11),
                compression="gzip",
            )
            self.assertEqual(result["synthetic_rows"], 2)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["target_pool_rows"]["Bm"], 1)
            with h5py.File(out, "r") as h5:
                self.assertEqual(h5["spectrograms"].shape[0], 6)
                labels_out = [x.decode("utf-8") for x in h5["label_strings"][:]]
                splits_out = [x.decode("utf-8") for x in h5["splits"][:]]
                self.assertEqual(labels_out[-2:], ["Bm", "Bm"])
                self.assertEqual(splits_out[-2:], ["train", "train"])


if __name__ == "__main__":
    unittest.main()
