import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None
try:
    import scipy.io as sio
except Exception:
    sio = None
try:
    import h5py
except Exception:
    h5py = None

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

if np is not None and sio is not None:
    from scripts.data.multilabel import build_e123_ssl_h5_dataset as e123_h5  # noqa: E402
else:
    e123_h5 = None


def write_csv(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_mat(path: Path, value: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = np.full((12, 20), value, dtype=np.float32)
    sio.savemat(
        path,
        {
            "PdB_norm": spec,
            "F": np.linspace(10, 120, 12, dtype=np.float32),
            "T": np.linspace(0, 40, 20, dtype=np.float32),
        },
    )


class TestE123SslH5Dataset(unittest.TestCase):
    @unittest.skipIf(e123_h5 is None or h5py is None, "numpy/scipy/h5py are required for E123 H5 export tests")
    def test_exports_target_and_background_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in {"bm": -10, "bp": -20, "mn": -30, "bg": -80, "oo": -40}.items():
                write_mat(root / "mats" / f"{name}.mat", value)
            manifest = root / "manifest.csv"
            write_csv(
                manifest,
                [
                    {"item_id": "bm", "split": "train", "source_kind": "ONC", "label_ids": "species:Bm", "low_mat_path": "mats/bm.mat"},
                    {"item_id": "bp", "split": "val", "source_kind": "ONC", "label_ids": "species:Bp", "low_mat_path": "mats/bp.mat"},
                    {"item_id": "mn", "split": "test", "source_kind": "ONC", "label_ids": "species:Mn", "low_mat_path": "mats/mn.mat"},
                    {"item_id": "bg", "split": "train", "source_kind": "ONC", "label_ids": "", "low_mat_path": "mats/bg.mat"},
                    {"item_id": "oo", "split": "train", "source_kind": "ONC", "label_ids": "species:Oo", "low_mat_path": "mats/oo.mat"},
                ],
            )
            output = root / "e123.h5"
            summary = e123_h5.build_e123_h5(
                manifest_csv=manifest,
                output_h5=output,
                output_summary=root / "summary.json",
                dataset_root=root,
                band="low",
                band_crop_shape=(12, 10),
                output_shape=(16, 16),
                target_label_map={"species:Bm": "Bm", "species:Bp": "Bp", "species:Mn": "Mn"},
                splits={"train", "val"},
                source_kinds=None,
                non_target_mode="skip",
                ambiguous_mode="skip",
                max_normal=100,
                max_per_target=0,
                normal_crops_per_row=1,
                context_seconds=40,
                crop_time_seconds=10,
                seed=1,
                compression="lzf",
            )
            self.assertEqual(summary["rows_written"], 3)
            self.assertEqual(summary["label_counts"], {"Bm": 1, "Bp": 1, "normal": 1})
            self.assertEqual(summary["skip_reasons"]["split_filter"], 1)
            self.assertEqual(summary["skip_reasons"]["non_target_labeled"], 1)
            with h5py.File(output, "r") as h5:
                self.assertEqual(tuple(h5["spectrograms"].shape), (3, 16, 16, 1))
                self.assertEqual(tuple(h5["labels"].shape), (3, 3))
                labels = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in h5["label_strings"][:]]
                self.assertEqual(Counter(labels), {"Bm": 1, "Bp": 1, "normal": 1})
                names = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in h5["anomaly_label_names"][:]]
                self.assertEqual(names, ["Bm", "Bp", "Mn"])
            saved_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["band"], "low")

    @unittest.skipIf(e123_h5 is None or h5py is None, "numpy/scipy/h5py are required for E123 H5 export tests")
    def test_non_target_rows_can_be_exported_as_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mat(root / "mats" / "oo.mat", -40)
            manifest = root / "manifest.csv"
            write_csv(
                manifest,
                [
                    {"item_id": "oo", "split": "train", "source_kind": "ONC", "label_ids": "species:Oo", "low_mat_path": "mats/oo.mat"},
                ],
            )
            output = root / "e123.h5"
            summary = e123_h5.build_e123_h5(
                manifest_csv=manifest,
                output_h5=output,
                output_summary=root / "summary.json",
                dataset_root=root,
                band="low",
                band_crop_shape=(12, 10),
                output_shape=(16, 16),
                target_label_map={"species:Bm": "Bm", "species:Bp": "Bp", "species:Mn": "Mn"},
                splits={"train"},
                source_kinds=None,
                non_target_mode="normal",
                ambiguous_mode="skip",
                max_normal=100,
                max_per_target=0,
                normal_crops_per_row=1,
                context_seconds=40,
                crop_time_seconds=10,
                seed=1,
                compression="lzf",
            )
            self.assertEqual(summary["label_counts"], {"normal": 1})

    @unittest.skipIf(e123_h5 is None or h5py is None, "numpy/scipy/h5py are required for E123 H5 export tests")
    def test_can_export_multiple_normal_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mat(root / "mats" / "bg.mat", -80)
            write_mat(root / "mats" / "bp.mat", -20)
            manifest = root / "manifest.csv"
            write_csv(
                manifest,
                [
                    {"item_id": "bg", "split": "train", "source_kind": "ONC", "label_ids": "", "low_mat_path": "mats/bg.mat"},
                    {"item_id": "bp", "split": "train", "source_kind": "ONC", "label_ids": "species:Bp", "low_mat_path": "mats/bp.mat"},
                ],
            )
            output = root / "e123.h5"
            summary = e123_h5.build_e123_h5(
                manifest_csv=manifest,
                output_h5=output,
                output_summary=root / "summary.json",
                dataset_root=root,
                band="low",
                band_crop_shape=(12, 10),
                output_shape=(16, 16),
                target_label_map={"species:Bm": "Bm", "species:Bp": "Bp", "species:Mn": "Mn"},
                splits={"train"},
                source_kinds=None,
                non_target_mode="skip",
                ambiguous_mode="skip",
                max_normal=100,
                max_per_target=0,
                normal_crops_per_row=3,
                context_seconds=40,
                crop_time_seconds=10,
                seed=1,
                compression="lzf",
            )
            self.assertEqual(summary["rows_written"], 4)
            self.assertEqual(summary["label_counts"], {"Bp": 1, "normal": 3})
            self.assertEqual(summary["normal_crops_per_row"], 3)
            with h5py.File(output, "r") as h5:
                item_ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in h5["item_ids"][:]]
                self.assertEqual(Counter(item_ids), {"bg::crop0": 1, "bg::crop1": 1, "bg::crop2": 1, "bp": 1})


if __name__ == "__main__":
    unittest.main()
