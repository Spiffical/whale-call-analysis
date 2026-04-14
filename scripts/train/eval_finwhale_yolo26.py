#!/usr/bin/env python3
"""Evaluate a YOLO26 fin-whale detector across one or more prepared eval YAMLs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.finwhale_yolo import (
    build_prediction_gallery,
    parse_wandb_tags,
    ultralytics_metrics_to_dict,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate YOLO26 fin-whale detector")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument(
        "--eval-yamls",
        type=str,
        required=True,
        help="Comma-separated list of split=yaml_path entries",
    )
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--visual-limit", type=int, default=16)
    ap.add_argument("--conf-threshold", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="finwhale-bbox")
    ap.add_argument("--wandb-entity", type=str, default="")
    ap.add_argument("--wandb-group", type=str, default="finwhale-yolo26")
    ap.add_argument("--wandb-name", type=str, default="")
    ap.add_argument("--wandb-tags", type=str, default="bbox,yolo26,finwhale,eval")
    args = ap.parse_args()

    from ultralytics import YOLO

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(Path(args.weights).resolve()))
    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=str(args.wandb_project),
            entity=(str(args.wandb_entity).strip() or None),
            group=str(args.wandb_group),
            name=(str(args.wandb_name).strip() or None),
            tags=parse_wandb_tags(args.wandb_tags),
            job_type="eval",
            config={
                "weights": str(Path(args.weights).resolve()),
                "imgsz": int(args.imgsz),
                "batch_size": int(args.batch_size),
                "device": str(args.device),
                "visual_limit": int(args.visual_limit),
            },
        )

    summary: dict[str, object] = {
        "weights": str(Path(args.weights).resolve()),
        "output_dir": str(output_dir),
        "splits": {},
        "galleries": {},
    }

    pairs = [item.strip() for item in args.eval_yamls.split(",") if item.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Invalid --eval-yamls entry: {pair}")
        split_name, yaml_path = pair.split("=", 1)
        metrics = model.val(
            data=str(Path(yaml_path).resolve()),
            split="val",
            device=str(args.device),
            batch=int(args.batch_size),
            imgsz=int(args.imgsz),
            project=str(output_dir),
            name=f"eval_{split_name}",
            exist_ok=True,
            save_json=True,
            plots=False,
        )
        summary["splits"][split_name] = ultralytics_metrics_to_dict(metrics)
        gallery = build_prediction_gallery(
            model=model,
            eval_yaml_path=yaml_path,
            split_name=split_name,
            output_dir=output_dir / "visuals",
            imgsz=int(args.imgsz),
            device=str(args.device),
            visual_limit=int(args.visual_limit),
            conf_threshold=float(args.conf_threshold),
            max_det=int(args.max_det),
        )
        summary["galleries"][split_name] = gallery
        if wandb_run is not None:
            import wandb

            table = wandb.Table(columns=["split", "image_path", "gt_box_count", "pred_box_count", "overlay"])
            for example in gallery["examples"]:
                table.add_data(
                    split_name,
                    example["image_path"],
                    int(example["gt_box_count"]),
                    int(example["pred_box_count"]),
                    wandb.Image(example["overlay_path"], caption=f"{split_name}: {Path(example['image_path']).name}"),
                )
            wandb_run.log(
                {
                    f"{split_name}/metrics": summary["splits"][split_name],
                    f"{split_name}/examples": table,
                }
            )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if wandb_run is not None:
        import wandb

        wandb_run.summary.update(summary["splits"])
        artifact = wandb.Artifact(f"{wandb_run.name or wandb_run.id}-eval", type="evaluation")
        artifact.add_file(str(output_dir / "summary.json"))
        visuals_dir = output_dir / "visuals"
        if visuals_dir.exists():
            artifact.add_dir(str(visuals_dir))
        wandb_run.log_artifact(artifact)
        wandb.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
