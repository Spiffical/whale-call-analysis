"""YOLO26 dataset preparation and metric helpers for fin-whale bbox detection."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


FIN_CLASS_NAME = "fin_call"
YOLO_CLASS_ID = 0
DEFAULT_TRAIN_YAML = "data_train_val2025.yaml"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _available_coco_splits(coco_export_dir: Path) -> List[str]:
    splits: List[str] = []
    for child in sorted(coco_export_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "annotations.coco.json").exists():
            splits.append(child.name)
    return splits


def _link_or_copy_file(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    mode = str(mode).lower().strip()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _bbox_coco_to_yolo(
    bbox_xywh: Sequence[float],
    *,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    xc = (x + 0.5 * w) / max(1.0, float(image_width))
    yc = (y + 0.5 * h) / max(1.0, float(image_height))
    wn = w / max(1.0, float(image_width))
    hn = h / max(1.0, float(image_height))
    return xc, yc, wn, hn


def _write_dataset_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_yolo_dataset_from_coco(
    *,
    coco_export_dir: Path | str,
    output_dir: Path | str,
    split_names: Optional[Sequence[str]] = None,
    link_mode: str = "symlink",
) -> Dict[str, Any]:
    coco_root = Path(coco_export_dir).resolve()
    yolo_root = Path(output_dir).resolve()
    yolo_root.mkdir(parents=True, exist_ok=True)

    selected_splits = list(split_names) if split_names else _available_coco_splits(coco_root)
    split_summary: Dict[str, Any] = {}

    for split_name in selected_splits:
        ann_path = coco_root / split_name / "annotations.coco.json"
        if not ann_path.exists():
            continue
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in payload.get("annotations", []):
            annotations_by_image[int(ann["image_id"])].append(dict(ann))

        image_count = 0
        box_count = 0
        negative_image_count = 0
        image_dst_dir = yolo_root / "images" / split_name
        label_dst_dir = yolo_root / "labels" / split_name
        image_dst_dir.mkdir(parents=True, exist_ok=True)
        label_dst_dir.mkdir(parents=True, exist_ok=True)

        for image_row in payload.get("images", []):
            image_id = int(image_row["id"])
            src_image = coco_root / str(image_row["file_name"])
            dst_image = image_dst_dir / src_image.name
            _link_or_copy_file(src_image, dst_image, mode=link_mode)

            label_path = label_dst_dir / f"{dst_image.stem}.txt"
            lines: List[str] = []
            for ann in annotations_by_image.get(image_id, []):
                xc, yc, wn, hn = _bbox_coco_to_yolo(
                    ann["bbox"],
                    image_width=int(image_row["width"]),
                    image_height=int(image_row["height"]),
                )
                lines.append(
                    f"{YOLO_CLASS_ID} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}"
                )
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            image_count += 1
            box_count += len(lines)
            if not lines:
                negative_image_count += 1

        split_summary[split_name] = {
            "image_count": int(image_count),
            "box_count": int(box_count),
            "negative_image_count": int(negative_image_count),
        }

    yaml_dir = yolo_root / "yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    train_yaml_payload = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val_2025",
        "test": "images/test_2025" if (yolo_root / "images" / "test_2025").exists() else None,
        "names": {YOLO_CLASS_ID: FIN_CLASS_NAME},
    }
    if train_yaml_payload["test"] is None:
        train_yaml_payload.pop("test")
    train_yaml_path = yaml_dir / DEFAULT_TRAIN_YAML
    _write_dataset_yaml(train_yaml_path, train_yaml_payload)

    eval_yaml_paths: Dict[str, str] = {}
    for split_name in selected_splits:
        split_img_dir = yolo_root / "images" / split_name
        if not split_img_dir.exists():
            continue
        payload = {
            "path": str(yolo_root),
            "val": f"images/{split_name}",
            "names": {YOLO_CLASS_ID: FIN_CLASS_NAME},
        }
        yaml_path = yaml_dir / f"data_eval_{split_name}.yaml"
        _write_dataset_yaml(yaml_path, payload)
        eval_yaml_paths[split_name] = str(yaml_path)

    summary = {
        "coco_export_dir": str(coco_root),
        "output_dir": str(yolo_root),
        "link_mode": str(link_mode),
        "splits": split_summary,
        "train_yaml": str(train_yaml_path),
        "eval_yamls": eval_yaml_paths,
    }
    summary_path = yolo_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "train_yaml": train_yaml_path,
        "eval_yamls": eval_yaml_paths,
    }


def ultralytics_metrics_to_dict(metrics: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                out[str(key)] = float(value)
            else:
                out[str(key)] = value

    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None:
        for attr_name, key in (
            ("map", "box/map50-95"),
            ("map50", "box/map50"),
            ("map75", "box/map75"),
            ("mp", "box/precision"),
            ("mr", "box/recall"),
        ):
            value = getattr(box_metrics, attr_name, None)
            if value is not None:
                out[key] = float(value)

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        out["speed"] = {str(k): float(v) for k, v in speed.items() if isinstance(v, (int, float))}

    fitness = getattr(metrics, "fitness", None)
    if fitness is not None:
        out["fitness"] = float(fitness)

    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        out["save_dir"] = str(save_dir)

    return out
