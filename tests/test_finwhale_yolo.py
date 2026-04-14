import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.finwhale_yolo import build_yolo_dataset_from_coco


class TestFinwhaleYolo(unittest.TestCase):
    def test_build_yolo_dataset_from_coco(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coco_root = tmpdir_path / "coco_export"
            train_dir = coco_root / "train" / "images"
            val_dir = coco_root / "val_2025" / "images"
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

            train_img = train_dir / "train_a.png"
            train_img.write_bytes(b"pngdata")
            val_img = val_dir / "val_a.png"
            val_img.write_bytes(b"pngdata")

            (coco_root / "train" / "annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "train/images/train_a.png", "width": 100, "height": 200}
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}
                        ],
                        "categories": [{"id": 1, "name": "fin_call"}],
                    }
                ),
                encoding="utf-8",
            )
            (coco_root / "val_2025" / "annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 2, "file_name": "val_2025/images/val_a.png", "width": 50, "height": 50}
                        ],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "fin_call"}],
                    }
                ),
                encoding="utf-8",
            )

            result = build_yolo_dataset_from_coco(
                coco_export_dir=coco_root,
                output_dir=tmpdir_path / "yolo_export",
                link_mode="copy",
            )

            yolo_root = tmpdir_path / "yolo_export"
            train_label = yolo_root / "labels" / "train" / "train_a.txt"
            val_label = yolo_root / "labels" / "val_2025" / "val_a.txt"
            self.assertTrue(train_label.exists())
            self.assertTrue(val_label.exists())
            self.assertEqual(val_label.read_text(encoding="utf-8"), "")
            self.assertEqual(
                train_label.read_text(encoding="utf-8").strip(),
                "0 0.25000000 0.20000000 0.30000000 0.20000000",
            )

            train_yaml = Path(result["train_yaml"])
            self.assertTrue(train_yaml.exists())
            summary = json.loads((yolo_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["splits"]["train"]["box_count"], 1)
            self.assertEqual(summary["splits"]["val_2025"]["negative_image_count"], 1)

    def test_build_yolo_dataset_skips_empty_eval_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            coco_root = tmpdir_path / "coco_export"
            empty_split_dir = coco_root / "val_hist"
            empty_split_dir.mkdir(parents=True, exist_ok=True)
            (empty_split_dir / "annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "fin_call"}],
                    }
                ),
                encoding="utf-8",
            )

            result = build_yolo_dataset_from_coco(
                coco_export_dir=coco_root,
                output_dir=tmpdir_path / "yolo_export",
                link_mode="copy",
            )

            yolo_root = tmpdir_path / "yolo_export"
            summary = json.loads((yolo_root / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("val_hist", summary["skipped_empty_splits"])
            self.assertNotIn("val_hist", result["eval_yamls"])
            self.assertFalse((yolo_root / "yamls" / "data_eval_val_hist.yaml").exists())


if __name__ == "__main__":
    unittest.main()
