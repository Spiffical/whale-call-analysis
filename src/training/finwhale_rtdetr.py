"""Reusable RT-DETR helpers for fin-whale bbox training and evaluation."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, get_linear_schedule_with_warmup

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover - optional runtime dependency
    COCO = None
    COCOeval = None


FIN_CLASS_NAME = "fin_call"
EXTERNAL_CATEGORY_ID = 1
INTERNAL_CATEGORY_ID = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def build_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=1.0,
            min_visibility=0.0,
            clip=True,
        ),
    )


class FinwhaleCocoDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_dir: Path | str,
        split_name: str,
        processor: RTDetrImageProcessor,
        transforms: Optional[A.Compose] = None,
        max_images: int = 0,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split_name = str(split_name)
        self.processor = processor
        self.transforms = transforms

        ann_path = self.dataset_dir / self.split_name / "annotations.coco.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing COCO annotations for split {self.split_name}: {ann_path}")
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in payload.get("annotations", []):
            image_id = int(ann["image_id"])
            annotations_by_image[image_id].append(dict(ann))

        records: List[Dict[str, Any]] = []
        for image_row in payload.get("images", []):
            image_id = int(image_row["id"])
            records.append(
                {
                    "image_id": image_id,
                    "file_name": str(image_row["file_name"]),
                    "width": int(image_row["width"]),
                    "height": int(image_row["height"]),
                    "annotations": annotations_by_image.get(image_id, []),
                }
            )
        records = sorted(records, key=lambda row: (row["file_name"], row["image_id"]))
        if max_images > 0:
            records = records[: int(max_images)]
        self.records = records
        self.annotation_path = ann_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image_path = self.dataset_dir / record["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image)

        bboxes = [list(ann["bbox"]) for ann in record["annotations"]]
        category_ids = [INTERNAL_CATEGORY_ID for _ in record["annotations"]]

        if self.transforms is not None:
            transformed = self.transforms(image=image_array, bboxes=bboxes, category_ids=category_ids)
            image_array = np.asarray(transformed["image"])
            bboxes = [list(box) for box in transformed["bboxes"]]
            category_ids = [int(cat_id) for cat_id in transformed["category_ids"]]

        annotations = []
        for bbox, category_id in zip(bboxes, category_ids):
            width = max(0.0, float(bbox[2]))
            height = max(0.0, float(bbox[3]))
            if width <= 0.0 or height <= 0.0:
                continue
            annotations.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "category_id": int(category_id),
                    "area": float(width * height),
                    "iscrowd": 0,
                }
            )

        encoded = self.processor(
            images=Image.fromarray(image_array),
            annotations={"image_id": int(record["image_id"]), "annotations": annotations},
            return_tensors="pt",
        )
        labels = encoded["labels"][0]
        return {
            "pixel_values": encoded["pixel_values"][0],
            "labels": labels,
            "image_id": int(record["image_id"]),
            "file_name": str(record["file_name"]),
        }


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": [item["labels"] for item in batch],
        "image_ids": [int(item["image_id"]) for item in batch],
        "file_names": [str(item["file_name"]) for item in batch],
    }


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def load_model_and_processor(model_name_or_path: str) -> Tuple[RTDetrForObjectDetection, RTDetrImageProcessor]:
    processor = RTDetrImageProcessor.from_pretrained(model_name_or_path, do_resize=False, do_pad=False)
    model = RTDetrForObjectDetection.from_pretrained(
        model_name_or_path,
        num_labels=1,
        ignore_mismatched_sizes=True,
        id2label={INTERNAL_CATEGORY_ID: FIN_CLASS_NAME},
        label2id={FIN_CLASS_NAME: INTERNAL_CATEGORY_ID},
    )
    return model, processor


def save_model_bundle(
    *,
    output_dir: Path | str,
    model: RTDetrForObjectDetection,
    processor: RTDetrImageProcessor,
    metadata: Dict[str, Any],
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return out_dir


def move_labels_to_device(labels: Sequence[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    moved: List[Dict[str, Any]] = []
    for item in labels:
        moved_item = {}
        for key, value in item.items():
            if torch.is_tensor(value):
                moved_item[key] = value.to(device)
            else:
                moved_item[key] = value
        moved.append(moved_item)
    return moved


def training_step(
    *,
    model: RTDetrForObjectDetection,
    batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    labels = move_labels_to_device(batch["labels"], device)
    outputs = model(pixel_values=pixel_values, labels=labels)
    return outputs.loss


def evaluate_split(
    *,
    model: RTDetrForObjectDetection,
    processor: RTDetrImageProcessor,
    dataset: FinwhaleCocoDataset,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.001,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels_cpu = batch["labels"]
            labels = move_labels_to_device(labels_cpu, device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            if outputs.loss is not None:
                total_loss += float(outputs.loss.item())
                total_batches += 1

            target_sizes = torch.stack([label["orig_size"].cpu() for label in labels_cpu], dim=0)
            processed = processor.post_process_object_detection(
                outputs,
                threshold=float(score_threshold),
                target_sizes=target_sizes,
            )
            for image_id, result in zip(batch["image_ids"], processed):
                for score, label, bbox in zip(result["scores"], result["labels"], result["boxes"]):
                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": EXTERNAL_CATEGORY_ID,
                            "bbox": [
                                round(float(bbox[0]), 4),
                                round(float(bbox[1]), 4),
                                round(float(bbox[2] - bbox[0]), 4),
                                round(float(bbox[3] - bbox[1]), 4),
                            ],
                            "score": round(float(score), 6),
                            "model_label": int(label),
                        }
                    )

    metrics: Dict[str, Any] = {
        "loss": (total_loss / total_batches) if total_batches > 0 else None,
        "prediction_count": int(len(predictions)),
        "image_count": int(len(dataset)),
    }

    if COCO is not None and COCOeval is not None:
        coco_gt = COCO(str(dataset.annotation_path))
        if predictions:
            with tempfile.TemporaryDirectory() as tmpdir:
                pred_path = Path(tmpdir) / f"{dataset.split_name}_predictions.json"
                pred_path.write_text(json.dumps(predictions), encoding="utf-8")
                coco_dt = coco_gt.loadRes(str(pred_path))
                coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = [float(value) for value in coco_eval.stats.tolist()]
        else:
            stats = [0.0] * 12

        metrics.update(
            {
                "coco/bbox_mAP": stats[0],
                "coco/bbox_mAP_50": stats[1],
                "coco/bbox_mAP_75": stats[2],
                "coco/bbox_mAR_1": stats[6],
                "coco/bbox_mAR_10": stats[7],
                "coco/bbox_mAR_100": stats[8],
            }
        )

    return metrics, predictions


def best_metric_from_eval(metrics: Dict[str, Any], *, fallback_key: str = "loss") -> float:
    map50 = metrics.get("coco/bbox_mAP_50")
    if map50 is not None:
        return float(map50)
    loss = metrics.get(fallback_key)
    if loss is None:
        return float("-inf")
    return -float(loss)


def count_train_steps(dataloader: DataLoader, epochs: int, grad_accum_steps: int) -> int:
    updates_per_epoch = max(1, (len(dataloader) + max(1, grad_accum_steps) - 1) // max(1, grad_accum_steps))
    return int(max(1, epochs) * updates_per_epoch)


def build_optimizer_and_scheduler(
    *,
    model: RTDetrForObjectDetection,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    grad_accum_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    total_steps = count_train_steps(dataloader, epochs=int(epochs), grad_accum_steps=int(grad_accum_steps))
    warmup_steps = int(round(float(total_steps) * float(max(0.0, warmup_ratio))))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=int(total_steps),
    )
    return optimizer, scheduler
