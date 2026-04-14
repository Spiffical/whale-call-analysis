#!/usr/bin/env python3
"""Train a YOLO26 detector on the exported fin-whale bbox dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.finwhale_yolo import ultralytics_metrics_to_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO26 on fin-whale bbox exports")
    ap.add_argument("--data-yaml", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-name", type=str, default="yolo26m.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--project-name", type=str, default="train")
    args = ap.parse_args()

    from ultralytics import YOLO

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_dir
    run_name = str(args.project_name)

    model = YOLO(args.model_name)
    model.train(
        data=str(Path(args.data_yaml).resolve()),
        epochs=int(args.epochs),
        batch=int(args.batch_size),
        imgsz=int(args.imgsz),
        device=str(args.device),
        workers=int(args.workers),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=int(args.seed),
        deterministic=True,
        patience=int(args.patience),
        optimizer="auto",
        rect=False,
        close_mosaic=0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        save_json=True,
        plots=True,
        val=True,
        single_cls=True,
    )

    train_dir = project_dir / run_name
    weights_dir = train_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    summary = {
        "data_yaml": str(Path(args.data_yaml).resolve()),
        "output_dir": str(output_dir),
        "train_dir": str(train_dir),
        "model_name": str(args.model_name),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "best_weights": str(best_path),
        "last_weights": str(last_path),
    }

    if best_path.exists():
        best_model = YOLO(str(best_path))
        val_metrics = best_model.val(
            data=str(Path(args.data_yaml).resolve()),
            split="val",
            device=str(args.device),
            batch=int(args.batch_size),
            imgsz=int(args.imgsz),
            project=str(output_dir),
            name="best_val_2025",
            exist_ok=True,
            save_json=True,
            plots=False,
        )
        summary["best_val_metrics"] = ultralytics_metrics_to_dict(val_metrics)

        shutil.copy2(best_path, output_dir / "best.pt")
    if last_path.exists():
        shutil.copy2(last_path, output_dir / "last.pt")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
