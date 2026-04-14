#!/usr/bin/env python3
"""Evaluate a trained RT-DETR checkpoint on one or more exported bbox splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.finwhale_rtdetr import (
    FinwhaleCocoDataset,
    build_dataloader,
    evaluate_split,
    load_model_and_processor,
    resolve_device,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained RT-DETR fin-whale detector")
    ap.add_argument("--dataset-dir", type=str, required=True)
    ap.add_argument("--checkpoint-dir", type=str, required=True)
    ap.add_argument("--splits", type=str, default="val_2025,test_2025,val_hist,test_hist")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--score-threshold", type=float, default=0.001)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(str(checkpoint_dir))
    device = resolve_device(args.device)
    model.to(device)

    summary: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "splits": {},
    }

    for split_name in [part.strip() for part in str(args.splits).split(",") if part.strip()]:
        ann_path = dataset_dir / split_name / "annotations.coco.json"
        if not ann_path.exists():
            continue
        dataset = FinwhaleCocoDataset(
            dataset_dir=dataset_dir,
            split_name=split_name,
            processor=processor,
            transforms=None,
            max_images=int(args.max_images),
        )
        dataloader = build_dataloader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
        metrics, predictions = evaluate_split(
            model=model,
            processor=processor,
            dataset=dataset,
            dataloader=dataloader,
            device=device,
            score_threshold=float(args.score_threshold),
        )
        summary["splits"][split_name] = metrics
        with open(output_dir / f"{split_name}_predictions.json", "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
