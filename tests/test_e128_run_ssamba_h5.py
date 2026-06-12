import unittest
import tempfile
from pathlib import Path

try:
    import h5py
except Exception:
    h5py = None

try:
    import numpy as np
except Exception:
    np = None

from scripts.analysis import e128_run_ssamba_h5 as runner


class TestE128RunSSAMBAH5(unittest.TestCase):
    def parse(self, *extra):
        args = runner.build_parser().parse_args(
            [
                "--data-train",
                "/tmp/data.h5",
                "--exp-dir",
                "/tmp/run",
                *extra,
            ]
        )
        return runner.normalize_args(args)

    def test_parse_bool(self):
        self.assertTrue(runner.parse_bool("true"))
        self.assertTrue(runner.parse_bool("1"))
        self.assertFalse(runner.parse_bool("false"))
        self.assertFalse(runner.parse_bool("0"))

    def test_binary_ft_cls_uses_single_output_head(self):
        args = self.parse("--task", "ft_cls")
        self.assertFalse(args.multiclass)
        self.assertEqual(args.n_class, 2)
        self.assertEqual(args.num_classes, 1)
        self.assertEqual(args.main_metric, "auc")

    def test_multiclass_preserves_requested_classes(self):
        args = self.parse("--task", "ft_avgtok", "--multiclass", "--num_classes", "4")
        self.assertTrue(args.multiclass)
        self.assertEqual(args.n_class, 4)
        self.assertEqual(args.num_classes, 4)

    def test_pretrain_uses_ssl_binary_shape(self):
        args = self.parse("--task", "pretrain_joint")
        self.assertEqual(args.n_class, 2)
        self.assertEqual(args.num_classes, 2)
        self.assertEqual(args.main_metric, "acc")

    @unittest.skipIf(h5py is None or np is None, "h5py/numpy are required for H5 dataset test")
    def test_h5_split_dataset_honors_splits_and_normal_only_ssl(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "dataset.h5"
            string_dtype = h5py.string_dtype(encoding="utf-8")
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("spectrograms", data=np.ones((5, 4, 4, 1), dtype=np.float32))
                h5.create_dataset("label_strings", data=np.asarray(["normal", "Bm", "normal", "Mn", "normal"], dtype=object), dtype=string_dtype)
                h5.create_dataset("splits", data=np.asarray(["train", "train", "val", "val", "test"], dtype=object), dtype=string_dtype)
                h5.create_dataset("sources", data=np.asarray(["a", "b", "c", "d", "e"], dtype=object), dtype=string_dtype)
                h5.create_dataset("anomaly_label_names", data=np.asarray(["Bm", "Bp", "Mn"], dtype=object), dtype=string_dtype)

            ssl_train = runner.H5SplitSpectrogramDataset(
                h5_path=h5_path,
                split="train",
                supervised=False,
                multiclass=False,
                num_classes=2,
                dataset_mean=0.0,
                dataset_std=1.0,
                amount=1.0,
                mixup=0.0,
                balance=False,
                seed=42,
            )
            self.assertEqual([sample["index"] for sample in ssl_train.sample_info], [0])

            supervised_val = runner.H5SplitSpectrogramDataset(
                h5_path=h5_path,
                split="val",
                supervised=True,
                multiclass=True,
                num_classes=4,
                dataset_mean=0.0,
                dataset_std=1.0,
                amount=1.0,
                mixup=0.0,
                balance=False,
                seed=42,
            )
            self.assertEqual([sample["index"] for sample in supervised_val.sample_info], [2, 3])
            self.assertEqual(supervised_val.label_to_index, {"normal": 0, "Bm": 1, "Bp": 2, "Mn": 3})
            _tensor, label, source = supervised_val[1]
            self.assertEqual(int(label), 3)
            self.assertEqual(source, "d")


if __name__ == "__main__":
    unittest.main()
