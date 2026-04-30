import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.multilabel import (
    LabelVocabulary,
    MultiLabelMatDataset,
    build_vocabulary_from_rows,
    group_key_for_split,
    label_ids_from_row,
    normalize_call_type,
    normalize_species_code,
    temporal_grouped_split,
    write_csv_rows,
)
from scripts.train.train_multilabel_resnet_smoke import write_validation_exports
from scripts.data.multilabel.build_call_mat_manifest import build_call_manifest


class TestMultiLabelHelpers(unittest.TestCase):
    def test_label_normalization_and_target_vector(self):
        rows = [
            {
                "species_codes": "Bp|Mn|INSTRUMENT",
                "fin_call_type_stds": "20Hz|S",
            },
            {
                "species_codes": "OD",
                "call_type_stds": "CK",
            },
        ]
        self.assertEqual(normalize_species_code("fin whale"), "Bp")
        self.assertEqual(normalize_call_type("20 Hz"), "20Hz")
        self.assertEqual(label_ids_from_row(rows[0]), ["call:20Hz", "call:song", "species:Bp", "species:Mn"])

        vocab = build_vocabulary_from_rows(rows)
        vector = vocab.vectorize(label_ids_from_row(rows[0]))
        ids = vocab.label_ids
        self.assertEqual(float(vector[ids.index("species:Bp")]), 1.0)
        self.assertEqual(float(vector[ids.index("call:20Hz")]), 1.0)
        self.assertNotIn("species:INSTRUMENT", ids)
        self.assertNotIn("call:CK", ids)

    def test_temporal_grouped_split_keeps_groups_together(self):
        rows = []
        for idx in range(6):
            group = f"clip-{idx // 2}"
            rows.append(
                {
                    "item_id": f"item-{idx}",
                    "event_group": group,
                    "start_time": f"2025-01-0{idx + 1}T00:00:00+00:00",
                    "label_ids": "species:Bp",
                }
            )
        split_rows = temporal_grouped_split(rows, train_ratio=0.5, val_ratio=0.25)
        seen = {}
        for split, split_items in split_rows.items():
            for row in split_items:
                group = group_key_for_split(row)
                self.assertNotIn(group, seen) if group not in seen else self.assertEqual(seen[group], split)
                seen[group] = split
        self.assertEqual(sum(len(items) for items in split_rows.values()), len(rows))

    def test_multilabel_mat_dataset_loads_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mat_path = root / "ICLISTENHF6016_20250101T000000.000Z.flac_0.0s_10.0s_window.mat"
            sio.savemat(
                mat_path,
                {
                    "P": np.abs(np.random.default_rng(7).normal(size=(48, 80))).astype(np.float32),
                    "frequencies": np.linspace(0, 120, 48).astype(np.float32),
                    "times": np.linspace(0, 10, 80).astype(np.float32),
                },
            )
            manifest = root / "manifest.csv"
            write_csv_rows(
                manifest,
                [
                    {
                        "item_id": "item-1",
                        "mat_path": str(mat_path),
                        "split": "train",
                        "label_ids": "species:Bp|call:20Hz",
                    }
                ],
            )
            vocab = LabelVocabulary(
                labels=(
                    {"id": "species:Bp", "group": "species", "code": "Bp", "name": "Fin whale"},
                    {"id": "call:20Hz", "group": "call_type", "code": "20Hz", "name": "20 Hz pulse"},
                )
            )
            ds = MultiLabelMatDataset(manifest, vocab, split="train", crop_size=32, return_meta=True)
            x, y, meta = ds[0]
            self.assertEqual(tuple(x.shape), (1, 32, 32))
            self.assertEqual(y.tolist(), [1.0, 1.0])
            self.assertEqual(meta["item_id"], "item-1")

    def test_validation_export_has_o3_model_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = LabelVocabulary(
                labels=(
                    {
                        "id": "species:Bp",
                        "group": "species",
                        "code": "Bp",
                        "name": "Fin whale",
                        "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
                    },
                    {
                        "id": "call:20Hz",
                        "group": "call_type",
                        "code": "20Hz",
                        "name": "20 Hz pulse",
                        "class_hierarchy": "Bioacoustic call type > 20 Hz pulse",
                    },
                )
            )
            write_validation_exports(
                root,
                vocab,
                {
                    "scores": np.asarray([[0.9, 0.2]], dtype=np.float32),
                    "targets": np.asarray([[1.0, 0.0]], dtype=np.float32),
                    "metas": [
                        {
                            "item_id": "clip-1",
                            "source_audio": "ICLISTENHF6016_20250101T000000.000Z.flac",
                            "mat_path": "/tmp/clip-1.mat",
                        }
                    ],
                },
                threshold=0.5,
            )

            payload = json.loads((root / "validation_predictions.o3_compatible.json").read_text())
            self.assertEqual(payload["schema_version"], "multilabel-smoke-o3-compatible-v1")
            self.assertEqual(payload["items"][0]["item_id"], "clip-1")
            self.assertEqual(len(payload["items"][0]["model_outputs"]), 2)
            self.assertEqual(payload["items"][0]["model_outputs"][0]["label_id"], "species:Bp")
            self.assertIn("class_hierarchy", payload["items"][0]["model_outputs"][0])

    def test_build_call_mat_manifest_matches_annotation_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mat_dir = root / "mat_files"
            mat_dir.mkdir()
            mat_path = mat_dir / "ICLISTENHF6016_20250101T000000.000Z.flac_12.0s_13.0s.mat"
            sio.savemat(mat_path, {"P": np.ones((8, 8), dtype=np.float32)})
            annotations = root / "annotations_all.csv"
            write_csv_rows(
                annotations,
                [
                    {
                        "filename": "ICLISTENHF6016_20250101T000000.000Z.flac",
                        "begin_time_s": "12.0",
                        "end_time_s": "13.0",
                        "species_code": "Bp",
                        "call_type_std": "20Hz",
                        "source_dataset": "unit",
                    }
                ],
            )
            rows, unmatched, summary = build_call_manifest(
                annotations_csv=annotations,
                mat_dir=mat_dir,
                dataset_name="unit",
                tolerance_s=0.1,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(unmatched, [])
            self.assertEqual(rows[0]["label_ids"], "call:20Hz|species:Bp")
            self.assertEqual(summary["label_counts"]["species:Bp"], 1)


if __name__ == "__main__":
    unittest.main()
