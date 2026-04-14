#!/usr/bin/env python3
"""Train an RT-DETR detector on the exported fin-whale bbox dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.training.finwhale_rtdetr import (
    FIN_CLASS_NAME,
    FinwhaleCocoDataset,
    best_metric_from_eval,
    build_dataloader,
    build_optimizer_and_scheduler,
    build_train_transforms,
    evaluate_split,
    load_model_and_processor,
    resolve_device,
    save_model_bundle,
    seed_everything,
    training_step,
)


def _existing_split_names(dataset_dir: Path, requested: list[str]) -> list[str]:
    names: list[str] = []
    for split_name in requested:
        ann_path = dataset_dir / split_name / "annotations.coco.json"
        if ann_path.exists():
            names.append(split_name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RT-DETR on fin-whale bbox exports")
    ap.add_argument("--dataset-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-name", type=str, default="PekingU/rtdetr_r50vd")
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--primary-eval-split", type=str, default="val_2025")
    ap.add_argument("--secondary-eval-split", type=str, default="val_hist")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-batch-size", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--score-threshold", type=float, default=0.001)
    ap.add_argument("--max-train-images", type=int, default=0)
    ap.add_argument("--max-eval-images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    model, processor = load_model_and_processor(args.model_name)
    model.to(device)

    train_dataset = FinwhaleCocoDataset(
        dataset_dir=dataset_dir,
        split_name=args.train_split,
        processor=processor,
        transforms=build_train_transforms(),
        max_images=int(args.max_train_images),
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(args.train_batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    eval_splits = _existing_split_names(
        dataset_dir,
        [args.primary_eval_split, args.secondary_eval_split],
    )
    eval_datasets = {
        split_name: FinwhaleCocoDataset(
            dataset_dir=dataset_dir,
            split_name=split_name,
            processor=processor,
            transforms=None,
            max_images=int(args.max_eval_images),
        )
        for split_name in eval_splits
    }
    eval_loaders = {
        split_name: build_dataloader(
            dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
        for split_name, dataset in eval_datasets.items()
    }

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        dataloader=train_loader,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        grad_accum_steps=int(args.gradient_accumulation_steps),
    )
    grad_accum_steps = max(1, int(args.gradient_accumulation_steps))

    history: list[dict[str, object]] = []
    best_epoch = 0
    best_score = float("-inf")
    best_metrics: dict[str, object] = {}

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        step_count = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            loss = training_step(model=model, batch=batch, device=device)
            running_loss += float(loss.item())
            step_count += 1
            (loss / grad_accum_steps).backward()

            if step_idx % grad_accum_steps == 0 or step_idx == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, step_count)
        epoch_record: dict[str, object] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }

        current_primary_metrics: dict[str, object] = {"loss": None}
        current_predictions: dict[str, list[dict[str, object]]] = {}
        for split_name in eval_splits:
            metrics, predictions = evaluate_split(
                model=model,
                processor=processor,
                dataset=eval_datasets[split_name],
                dataloader=eval_loaders[split_name],
                device=device,
                score_threshold=float(args.score_threshold),
            )
            epoch_record[f"{split_name}_metrics"] = metrics
            current_predictions[split_name] = predictions
            if split_name == args.primary_eval_split:
                current_primary_metrics = metrics

        history.append(epoch_record)
        with open(output_dir / "metrics_history.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")

        current_score = best_metric_from_eval(current_primary_metrics)
        if current_score > best_score:
            best_score = current_score
            best_epoch = int(epoch)
            best_metrics = epoch_record
            metadata = {
                "epoch": int(epoch),
                "model_name": str(args.model_name),
                "train_split": str(args.train_split),
                "primary_eval_split": str(args.primary_eval_split),
                "secondary_eval_split": str(args.secondary_eval_split),
                "fin_class_name": FIN_CLASS_NAME,
                "best_score": float(best_score),
            }
            save_model_bundle(
                output_dir=output_dir / "best",
                model=model,
                processor=processor,
                metadata=metadata,
            )
            for split_name, predictions in current_predictions.items():
                pred_path = output_dir / f"best_{split_name}_predictions.json"
                with open(pred_path, "w", encoding="utf-8") as handle:
                    json.dump(predictions, handle, indent=2)

    save_model_bundle(
        output_dir=output_dir / "last",
        model=model,
        processor=processor,
        metadata={
            "model_name": str(args.model_name),
            "train_split": str(args.train_split),
            "primary_eval_split": str(args.primary_eval_split),
            "secondary_eval_split": str(args.secondary_eval_split),
            "epochs": int(args.epochs),
            "fin_class_name": FIN_CLASS_NAME,
        },
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "model_name": str(args.model_name),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "history_length": int(len(history)),
        "best_metrics": best_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
