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

from src.training.finwhale_yolo import parse_wandb_tags, ultralytics_metrics_to_dict


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
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="finwhale-bbox")
    ap.add_argument("--wandb-entity", type=str, default="")
    ap.add_argument("--wandb-group", type=str, default="finwhale-yolo26")
    ap.add_argument("--wandb-name", type=str, default="")
    ap.add_argument("--wandb-tags", type=str, default="bbox,yolo26,finwhale")
    args = ap.parse_args()

    from ultralytics import YOLO

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_dir
    run_name = str(args.project_name)
    wandb_run = None

    if args.use_wandb:
        import wandb
        from wandb.integration.ultralytics import add_wandb_callback

        wandb_run = wandb.init(
            project=str(args.wandb_project),
            entity=(str(args.wandb_entity).strip() or None),
            group=str(args.wandb_group),
            name=(str(args.wandb_name).strip() or None),
            tags=parse_wandb_tags(args.wandb_tags),
            job_type="train",
            config={
                "data_yaml": str(Path(args.data_yaml).resolve()),
                "model_name": str(args.model_name),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "imgsz": int(args.imgsz),
                "device": str(args.device),
                "workers": int(args.workers),
                "seed": int(args.seed),
                "patience": int(args.patience),
            },
        )

    model = YOLO(args.model_name)
    if wandb_run is not None:
        add_wandb_callback(model, enable_model_checkpointing=True)
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
        "wandb_enabled": bool(wandb_run is not None),
        "wandb_run_id": None if wandb_run is None else str(wandb_run.id),
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
    if wandb_run is not None:
        import wandb

        wandb_run.summary.update(summary)
        artifact = wandb.Artifact(f"{wandb_run.name or wandb_run.id}-weights", type="model")
        if (output_dir / "best.pt").exists():
            artifact.add_file(str(output_dir / "best.pt"))
        if (output_dir / "last.pt").exists():
            artifact.add_file(str(output_dir / "last.pt"))
        artifact.add_file(str(output_dir / "summary.json"))
        wandb_run.log_artifact(artifact)
        wandb.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
