"""YOLO26 dataset preparation, visual QC, and metric helpers."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PIL import Image, ImageDraw


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

    test_split_has_images = int(split_summary.get("test_2025", {}).get("image_count", 0)) > 0
    train_yaml_payload = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val_2025",
        "test": "images/test_2025" if test_split_has_images else None,
        "names": {YOLO_CLASS_ID: FIN_CLASS_NAME},
    }
    if train_yaml_payload["test"] is None:
        train_yaml_payload.pop("test")
    train_yaml_path = yaml_dir / DEFAULT_TRAIN_YAML
    _write_dataset_yaml(train_yaml_path, train_yaml_payload)

    eval_yaml_paths: Dict[str, str] = {}
    skipped_empty_splits: List[str] = []
    for split_name in selected_splits:
        if int(split_summary.get(split_name, {}).get("image_count", 0)) <= 0:
            skipped_empty_splits.append(str(split_name))
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
        "skipped_empty_splits": skipped_empty_splits,
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


def parse_wandb_tags(raw_tags: str | None) -> List[str]:
    if not raw_tags:
        return []
    return [part.strip() for part in str(raw_tags).split(",") if part.strip()]


def load_yolo_yaml(yaml_path: Path | str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YOLO YAML payload: {yaml_path}")
    return payload


def resolve_yolo_split_dirs(yaml_path: Path | str) -> Tuple[Path, Path]:
    yaml_path = Path(yaml_path).resolve()
    payload = load_yolo_yaml(yaml_path)
    root = Path(str(payload["path"])).resolve()
    rel_image_dir = str(payload["val"]).strip()
    image_dir = (root / rel_image_dir).resolve()
    if not rel_image_dir.startswith("images/"):
        raise ValueError(f"Expected eval yaml val=images/<split>, got {rel_image_dir}")
    label_dir = (root / rel_image_dir.replace("images/", "labels/", 1)).resolve()
    return image_dir, label_dir


def load_yolo_boxes_from_label(
    label_path: Path | str,
    *,
    image_width: int,
    image_height: int,
) -> List[Dict[str, float]]:
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    boxes: List[Dict[str, float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, wn, hn = parts
        xc_f = _safe_float(xc)
        yc_f = _safe_float(yc)
        wn_f = _safe_float(wn)
        hn_f = _safe_float(hn)
        box_w = wn_f * float(image_width)
        box_h = hn_f * float(image_height)
        x0 = (xc_f * float(image_width)) - 0.5 * box_w
        y0 = (yc_f * float(image_height)) - 0.5 * box_h
        boxes.append(
            {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x0 + box_w),
                "y1": float(y0 + box_h),
            }
        )
    return boxes


def draw_detection_overlay(
    *,
    image_path: Path | str,
    gt_boxes: Sequence[Dict[str, float]],
    pred_boxes: Sequence[Dict[str, float]],
    output_path: Path | str,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in gt_boxes:
        draw.rectangle(
            [float(box["x0"]), float(box["y0"]), float(box["x1"]), float(box["y1"])],
            outline=(36, 196, 96),
            width=3,
        )

    for box in pred_boxes:
        draw.rectangle(
            [float(box["x0"]), float(box["y0"]), float(box["x1"]), float(box["y1"])],
            outline=(230, 76, 60),
            width=2,
        )
        conf = box.get("conf")
        if conf is not None:
            draw.text(
                (float(box["x0"]) + 4.0, max(0.0, float(box["y0"]) - 14.0)),
                f"{float(conf):.2f}",
                fill=(230, 76, 60),
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def build_prediction_gallery(
    *,
    model: Any,
    eval_yaml_path: Path | str,
    split_name: str,
    output_dir: Path | str,
    imgsz: int,
    device: str,
    visual_limit: int = 16,
    conf_threshold: float = 0.001,
    max_det: int = 20,
) -> Dict[str, Any]:
    image_dir, label_dir = resolve_yolo_split_dirs(eval_yaml_path)
    image_paths = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda path: path.name,
    )
    if not image_paths:
        return {"split_name": split_name, "image_count": 0, "gallery_count": 0, "examples": []}

    ranked: List[Tuple[int, Path]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = load_yolo_boxes_from_label(label_dir / f"{image_path.stem}.txt", image_width=width, image_height=height)
        ranked.append((len(gt_boxes), image_path))
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    selected = [path for _, path in ranked[: max(0, int(visual_limit))]]

    predictions = model.predict(
        source=[str(path) for path in selected],
        imgsz=int(imgsz),
        device=str(device),
        verbose=False,
        save=False,
        conf=float(conf_threshold),
        max_det=int(max_det),
    )

    gallery_dir = Path(output_dir).resolve() / str(split_name)
    examples: List[Dict[str, Any]] = []
    for image_path, pred in zip(selected, predictions):
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = load_yolo_boxes_from_label(label_dir / f"{image_path.stem}.txt", image_width=width, image_height=height)
        pred_boxes: List[Dict[str, float]] = []
        boxes = getattr(pred, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [None] * len(xyxy)
            for coords, conf in zip(xyxy, confs):
                pred_boxes.append(
                    {
                        "x0": float(coords[0]),
                        "y0": float(coords[1]),
                        "x1": float(coords[2]),
                        "y1": float(coords[3]),
                        "conf": None if conf is None else float(conf),
                    }
                )
        overlay_path = draw_detection_overlay(
            image_path=image_path,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            output_path=gallery_dir / image_path.name,
        )
        examples.append(
            {
                "image_path": str(image_path),
                "overlay_path": str(overlay_path),
                "gt_box_count": int(len(gt_boxes)),
                "pred_box_count": int(len(pred_boxes)),
            }
        )

    return {
        "split_name": str(split_name),
        "image_count": int(len(image_paths)),
        "gallery_count": int(len(examples)),
        "gallery_dir": str(gallery_dir),
        "examples": examples,
    }
