#!/usr/bin/env python3
"""Tiny multi-label CNN/ResNet smoke trainer.

This is intentionally a small first-pass path for validating data loading,
multi-label targets, BCE-with-logits optimization, and validation export. It is
not meant to replace the mature binary training pipeline yet.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
except Exception:
    plt = None

try:
    import wandb
except Exception:
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    LabelVocabulary,
    MultiLabelMatDataset,
    clean_text,
    label_ids_from_row,
    multilabel_metrics,
)
from src.models.fin_models import create_model  # noqa: E402
from src.utils.model_utils import create_checkpoint_metadata  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def maybe_subset(ds: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None or int(max_samples) <= 0 or len(ds) <= int(max_samples):
        return ds
    return Subset(ds, list(range(int(max_samples))))


def _dataset_targets(ds: Dataset) -> np.ndarray:
    base = ds.dataset if isinstance(ds, Subset) else ds
    indices: Iterable[int] = ds.indices if isinstance(ds, Subset) else range(len(base))  # type: ignore[attr-defined]
    targets = [base.files[int(idx)][1] for idx in indices]  # type: ignore[attr-defined]
    if not targets:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(targets).astype(np.float32)


def compute_pos_weight(ds: Dataset) -> torch.Tensor:
    targets = _dataset_targets(ds)
    if targets.size == 0:
        return torch.ones(1, dtype=torch.float32)
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives
    weights = np.ones_like(positives, dtype=np.float32)
    present = positives > 0
    weights[present] = negatives[present] / np.maximum(positives[present], 1.0)
    return torch.from_numpy(np.clip(weights, 1.0, 100.0).astype(np.float32))


def load_partial_checkpoint(model: nn.Module, checkpoint_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
        if state is None:
            state = {key: value for key, value in ckpt.items() if isinstance(value, torch.Tensor)}
    else:
        state = None
    if not isinstance(state, dict):
        raise ValueError(f"Could not find a model state dict in {checkpoint_path}")

    clean_state = {str(key).removeprefix("module."): value for key, value in state.items() if isinstance(value, torch.Tensor)}
    current = model.state_dict()
    matched = {key: value for key, value in clean_state.items() if key in current and tuple(current[key].shape) == tuple(value.shape)}
    skipped = sorted(key for key in clean_state if key not in matched)
    current.update(matched)
    model.load_state_dict(current)
    return {
        "path": str(checkpoint_path),
        "matched_tensor_count": len(matched),
        "skipped_tensor_count": len(skipped),
        "skipped_tensor_examples": skipped[:20],
        "source_model_id": ckpt.get("model_id") if isinstance(ckpt, dict) else None,
    }


def collate_batch(batch: Sequence[Sequence[Any]]) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    metas = [item[2] for item in batch if len(item) == 3]
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), metas if metas else None


def _unpack_batch(batch: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
    if len(batch) == 3:
        x, y, meta = batch
        return x, y, meta
    x, y = batch
    return x, y, None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    scores: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    start = time.time()
    for batch in loader:
        x, y, _ = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        scores.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_score = np.concatenate(scores, axis=0) if scores else np.zeros((0, 0), dtype=np.float32)
    y_true = np.concatenate(targets, axis=0) if targets else np.zeros_like(y_score)
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": total_samples,
        "seconds": time.time() - start,
        "scores": y_score,
        "targets": y_true,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    threshold: float,
    max_example_images: int = 64,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    scores: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    images: List[np.ndarray] = []
    for batch in loader:
        x, y, meta = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        scores.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        if meta is not None:
            metas.extend(meta)
        if len(images) < int(max_example_images):
            room = int(max_example_images) - len(images)
            for arr in x.detach().cpu().numpy()[:room, 0]:
                images.append(arr.astype(np.float32))

    y_score = np.concatenate(scores, axis=0) if scores else np.zeros((0, 0), dtype=np.float32)
    y_true = np.concatenate(targets, axis=0) if targets else np.zeros_like(y_score)
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": total_samples,
        "scores": y_score,
        "targets": y_true,
        "metrics": multilabel_metrics(y_true, y_score, threshold=threshold) if total_samples else {},
        "metas": metas,
        "images": images,
    }


def _meta_at(meta: Dict[str, Any], key: str) -> str:
    value = meta.get(key)
    if isinstance(value, torch.Tensor):
        return str(value.item())
    return clean_text(value)


def write_prediction_exports(
    output_dir: Path,
    vocab: LabelVocabulary,
    eval_result: Dict[str, Any],
    *,
    threshold: float,
    prefix: str = "validation",
) -> None:
    labels = list(vocab.labels)
    scores = eval_result["scores"]
    targets = eval_result["targets"]
    metas = eval_result.get("metas") or [{} for _ in range(scores.shape[0])]

    csv_rows: List[Dict[str, Any]] = []
    json_items: List[Dict[str, Any]] = []
    for row_idx in range(scores.shape[0]):
        meta = metas[row_idx] if row_idx < len(metas) else {}
        target_ids = [labels[idx]["id"] for idx in range(len(labels)) if targets[row_idx, idx] >= 0.5]
        pred_ids = [labels[idx]["id"] for idx in range(len(labels)) if scores[row_idx, idx] >= threshold]
        csv_row: Dict[str, Any] = {
            "item_id": _meta_at(meta, "item_id"),
            "source_dataset": _meta_at(meta, "source_dataset"),
            "source_kind": _meta_at(meta, "source_kind"),
            "source_audio": _meta_at(meta, "source_audio"),
            "mat_path": _meta_at(meta, "mat_path"),
            "low_mat_path": _meta_at(meta, "low_mat_path"),
            "mid_mat_path": _meta_at(meta, "mid_mat_path"),
            "high_mat_path": _meta_at(meta, "high_mat_path"),
            "source_label_ids": _meta_at(meta, "source_label_ids"),
            "canonical_label_ids": _meta_at(meta, "canonical_label_ids"),
            "analysis_label_ids": _meta_at(meta, "analysis_label_ids"),
            "original_label_ids": _meta_at(meta, "original_label_ids"),
            "original_target_label_ids": _meta_at(meta, "original_target_label_ids"),
            "gate_positive_source_labels": _meta_at(meta, "gate_positive_source_labels"),
            "negative_bucket": _meta_at(meta, "negative_bucket"),
            "split": _meta_at(meta, "split"),
            "is_background": _meta_at(meta, "is_background"),
            "review_status": _meta_at(meta, "review_status"),
            "context_tags": _meta_at(meta, "context_tags"),
            "begin_s": _meta_at(meta, "begin_s"),
            "end_s": _meta_at(meta, "end_s"),
            "event_group": _meta_at(meta, "event_group"),
            "target_label_ids": "|".join(target_ids),
            "pred_label_ids": "|".join(pred_ids),
        }
        for idx, label in enumerate(labels):
            csv_row[f"score__{label['id']}"] = f"{float(scores[row_idx, idx]):.8f}"
        csv_rows.append(csv_row)

        json_items.append(
            {
                "item_id": csv_row["item_id"],
                "source_dataset": csv_row["source_dataset"],
                "audio_path": csv_row["source_audio"],
                "spectrogram_path": csv_row["mat_path"],
                "labels": [{"label_id": label_id} for label_id in target_ids],
                "model_outputs": [
                    {
                        "label_id": label["id"],
                        "class_hierarchy": label.get("class_hierarchy", label["id"]),
                        "score": float(scores[row_idx, idx]),
                        "threshold": float(threshold),
                    }
                    for idx, label in enumerate(labels)
                ],
            }
        )

    from src.dataset.multilabel import write_csv_rows

    write_csv_rows(output_dir / f"{prefix}_predictions.csv", csv_rows)
    with open(output_dir / f"{prefix}_predictions.o3_compatible.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "multilabel-smoke-o3-compatible-v1",
                "description": "One model_outputs entry per species/call-type class path.",
                "items": json_items,
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def write_validation_exports(
    output_dir: Path,
    vocab: LabelVocabulary,
    eval_result: Dict[str, Any],
    *,
    threshold: float,
) -> None:
    write_prediction_exports(output_dir, vocab, eval_result, threshold=threshold, prefix="validation")


def write_source_stratified_metrics(
    output_dir: Path,
    vocab: LabelVocabulary,
    eval_result: Dict[str, Any],
    *,
    threshold: float,
    group_field: str = "source_dataset",
    prefix: str = "source_metrics",
) -> List[Path]:
    scores = np.asarray(eval_result.get("scores", []), dtype=np.float32)
    targets = np.asarray(eval_result.get("targets", []), dtype=np.float32)
    metas = eval_result.get("metas") or []
    if scores.size == 0 or targets.size == 0 or scores.shape != targets.shape:
        return []
    source_to_indices: Dict[str, List[int]] = {}
    for idx in range(scores.shape[0]):
        meta = metas[idx] if idx < len(metas) else {}
        source = _meta_at(meta, group_field) or "<unknown>"
        source_to_indices.setdefault(source, []).append(idx)

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    labels = list(vocab.labels)
    for source, indices in sorted(source_to_indices.items()):
        source_scores = scores[indices]
        source_targets = targets[indices]
        metrics = multilabel_metrics(source_targets, source_scores, threshold=threshold)
        summary[source] = {
            "samples": len(indices),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "micro_f1": float(metrics.get("micro_f1", 0.0)),
            "micro_precision": float(metrics.get("micro_precision", 0.0)),
            "micro_recall": float(metrics.get("micro_recall", 0.0)),
        }
        for item in metrics.get("per_class", []):
            label_idx = int(item["index"])
            label = labels[label_idx] if label_idx < len(labels) else {"id": f"label_{label_idx}", "group": ""}
            rows.append(
                {
                    "source_dataset": source,
                    "source_field": group_field,
                    "label_id": label.get("id", f"label_{label_idx}"),
                    "group": label.get("group", ""),
                    "samples": len(indices),
                    "support": int(item.get("support", 0)),
                    "precision": float(item.get("precision", 0.0)),
                    "recall": float(item.get("recall", 0.0)),
                    "f1": float(item.get("f1", 0.0)),
                    "tp": int(item.get("tp", 0)),
                    "fp": int(item.get("fp", 0)),
                    "fn": int(item.get("fn", 0)),
                }
            )
    paths: List[Path] = []
    if rows:
        path = output_dir / f"{prefix}.csv"
        _write_csv_dicts(path, rows)
        paths.append(path)
    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    paths.append(summary_path)
    return paths


def _per_class_rows(vocab: LabelVocabulary, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    labels = list(vocab.labels)
    rows: List[Dict[str, Any]] = []
    for item in metrics.get("per_class", []):
        idx = int(item["index"])
        label = labels[idx] if idx < len(labels) else {"id": f"label_{idx}", "group": "", "name": ""}
        rows.append(
            {
                "label_id": label.get("id", f"label_{idx}"),
                "group": label.get("group", ""),
                "name": label.get("name", ""),
                "support": int(item.get("support", 0)),
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1", 0.0)),
                "tp": int(item.get("tp", 0)),
                "fp": int(item.get("fp", 0)),
                "fn": int(item.get("fn", 0)),
            }
        )
    return rows


def _write_csv_dicts(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    from src.dataset.multilabel import write_csv_rows

    write_csv_rows(path, rows)


def write_threshold_sweep(exp_dir: Path, vocab: LabelVocabulary, eval_result: Dict[str, Any]) -> Dict[str, Any]:
    scores = np.asarray(eval_result.get("scores", []), dtype=np.float32)
    targets = np.asarray(eval_result.get("targets", []), dtype=np.float32)
    labels = list(vocab.labels)
    if scores.size == 0 or targets.size == 0 or scores.shape != targets.shape:
        return {"paths": [], "global_best_macro": {}, "per_label_threshold_aggregate": {}}

    thresholds = [round(float(value), 2) for value in np.linspace(0.05, 0.95, 91)]
    per_label_rows: List[Dict[str, Any]] = []
    global_rows: List[Dict[str, Any]] = []
    best_by_label: Dict[int, Dict[str, Any]] = {}

    for threshold in thresholds:
        metrics = multilabel_metrics(targets, scores, threshold=float(threshold))
        global_rows.append(
            {
                "threshold": threshold,
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "micro_f1": float(metrics.get("micro_f1", 0.0)),
                "micro_precision": float(metrics.get("micro_precision", 0.0)),
                "micro_recall": float(metrics.get("micro_recall", 0.0)),
            }
        )
        for row in _per_class_rows(vocab, metrics):
            sweep_row = {"threshold": threshold, **row}
            per_label_rows.append(sweep_row)
            idx = len(per_label_rows)  # deterministic tie-breaker fallback
            label_idx = next((i for i, label in enumerate(labels) if label.get("id") == row["label_id"]), -1)
            current = best_by_label.get(label_idx)
            candidate_key = (float(row["f1"]), float(row["precision"]), float(threshold), -idx)
            current_key = (
                float(current.get("f1", 0.0)),
                float(current.get("precision", 0.0)),
                float(current.get("threshold", 0.0)),
                0.0,
            ) if current else None
            if current is None or candidate_key > current_key:
                best_by_label[label_idx] = sweep_row

    best_indices = [idx for idx in sorted(best_by_label) if idx >= 0]
    best_rows = [best_by_label[idx] for idx in best_indices]
    per_label_pred = np.zeros_like(targets, dtype=bool)
    for idx in best_indices:
        per_label_pred[:, idx] = scores[:, idx] >= float(best_by_label[idx]["threshold"])
    true = targets >= 0.5
    tp = np.logical_and(per_label_pred, true).sum(axis=0)
    fp = np.logical_and(per_label_pred, ~true).sum(axis=0)
    fn = np.logical_and(~per_label_pred, true).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float32), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float32), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    micro_tp = int(tp.sum())
    micro_fp = int(fp.sum())
    micro_fn = int(fn.sum())
    micro_precision = micro_tp / max(micro_tp + micro_fp, 1)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)

    global_best_macro = max(global_rows, key=lambda row: (float(row["macro_f1"]), float(row["threshold"])))
    global_best_micro = max(global_rows, key=lambda row: (float(row["micro_f1"]), float(row["threshold"])))
    per_label_summary = {
        "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "thresholds": {row["label_id"]: float(row["threshold"]) for row in best_rows},
        "per_class_f1": {labels[idx].get("id", f"label_{idx}"): float(f1[idx]) for idx in range(min(len(labels), len(f1)))},
    }

    _write_csv_dicts(exp_dir / "threshold_sweep.csv", per_label_rows)
    _write_csv_dicts(exp_dir / "threshold_sweep_global.csv", global_rows)
    _write_csv_dicts(exp_dir / "threshold_sweep_best.csv", best_rows)
    summary = {
        "global_best_macro": global_best_macro,
        "global_best_micro": global_best_micro,
        "per_label_threshold_aggregate": per_label_summary,
    }
    with open(exp_dir / "threshold_sweep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    paths: List[Path] = [
        exp_dir / "threshold_sweep.csv",
        exp_dir / "threshold_sweep_global.csv",
        exp_dir / "threshold_sweep_best.csv",
        exp_dir / "threshold_sweep_summary.json",
    ]
    if plt is not None:
        plot_dir = exp_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([row["threshold"] for row in global_rows], [row["macro_f1"] for row in global_rows], label="macro F1")
        ax.plot([row["threshold"] for row in global_rows], [row["micro_f1"] for row in global_rows], label="micro F1")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Validation F1 by Global Threshold")
        ax.grid(alpha=0.25)
        ax.legend()
        path = plot_dir / "threshold_global_f1.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(max(8, len(best_rows) * 0.75), 4))
        x = np.arange(len(best_rows))
        ax.bar(x, [float(row["f1"]) for row in best_rows], width=0.5, label="best F1")
        ax.plot(x, [float(row["threshold"]) for row in best_rows], marker="o", color="black", label="threshold")
        ax.set_xticks(x)
        ax.set_xticklabels([row["label_id"] for row in best_rows], rotation=45, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Best Per-Class Thresholds")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        path = plot_dir / "threshold_best_per_class.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return {**summary, "paths": paths}


def write_training_plots(exp_dir: Path, history: Sequence[Dict[str, Any]]) -> List[Path]:
    if plt is None or not history:
        return []
    plot_dir = exp_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label="train")
    ax.plot(epochs, [float(row["val_loss"]) for row in history], marker="o", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    path = plot_dir / "loss_by_epoch.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(row["train_metrics"].get("micro_f1", 0.0)) for row in history], marker="o", label="train micro F1")
    ax.plot(epochs, [float(row["val_metrics"].get("micro_f1", 0.0)) for row in history], marker="o", label="val micro F1")
    ax.plot(epochs, [float(row["train_metrics"].get("macro_f1", 0.0)) for row in history], marker="o", label="train macro F1")
    ax.plot(epochs, [float(row["val_metrics"].get("macro_f1", 0.0)) for row in history], marker="o", label="val macro F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Multi-Label F1 by Epoch")
    ax.grid(alpha=0.25)
    ax.legend()
    path = plot_dir / "f1_by_epoch.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_validation_plots(exp_dir: Path, vocab: LabelVocabulary, eval_result: Dict[str, Any]) -> List[Path]:
    if plt is None or not eval_result.get("metrics"):
        return []
    plot_dir = exp_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = _per_class_rows(vocab, eval_result["metrics"])
    _write_csv_dicts(exp_dir / "per_class_metrics.csv", rows)
    if not rows:
        return []
    paths: List[Path] = []
    labels = [row["label_id"] for row in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4))
    ax.bar(x - 0.25, [row["precision"] for row in rows], width=0.25, label="precision")
    ax.bar(x, [row["recall"] for row in rows], width=0.25, label="recall")
    ax.bar(x + 0.25, [row["f1"] for row in rows], width=0.25, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-Class Validation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = plot_dir / "per_class_precision_recall_f1.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    matrix = np.asarray([[row["tp"], row["fp"], row["fn"]] for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, max(4, len(rows) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["TP", "FP", "FN"])
    ax.set_title("Per-Class Error Counts")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    path = plot_dir / "per_class_error_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_example_images(
    exp_dir: Path,
    vocab: LabelVocabulary,
    eval_result: Dict[str, Any],
    *,
    threshold: float,
    max_per_kind: int = 12,
) -> List[Path]:
    if plt is None:
        return []
    scores = np.asarray(eval_result.get("scores", []))
    targets = np.asarray(eval_result.get("targets", []))
    images = eval_result.get("images") or []
    metas = eval_result.get("metas") or []
    if scores.size == 0 or not images:
        return []
    labels = list(vocab.labels)
    pred = scores >= float(threshold)
    true = targets >= 0.5
    examples: List[Tuple[str, int, int]] = []
    for kind, mask in (("false_positive", pred & ~true), ("false_negative", ~pred & true)):
        found = np.argwhere(mask)
        for row_idx, label_idx in found[: int(max_per_kind)]:
            if int(row_idx) < len(images):
                examples.append((kind, int(row_idx), int(label_idx)))
    out_dir = exp_dir / "example_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for n, (kind, row_idx, label_idx) in enumerate(examples, start=1):
        label = labels[label_idx] if label_idx < len(labels) else {"id": f"label_{label_idx}"}
        meta = metas[row_idx] if row_idx < len(metas) else {}
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(images[row_idx], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_axis_off()
        title = f"{kind}: {label['id']} score={scores[row_idx, label_idx]:.3f}"
        ax.set_title(title, fontsize=8)
        safe_label = str(label["id"]).replace(":", "_").replace("/", "_")
        item = clean_text(meta.get("item_id"))[:80] or f"row{row_idx}"
        path = out_dir / f"{n:03d}_{kind}_{safe_label}_{item}.png"
        fig.tight_layout(pad=0.2)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def init_wandb_if_requested(args: argparse.Namespace, vocab: LabelVocabulary):
    if not getattr(args, "use_wandb", False):
        return None
    if wandb is None:
        print("Warning: wandb is not importable; continuing without W&B logging")
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=args.wandb_name or Path(args.exp_dir).name,
            tags=[token.strip() for token in str(args.wandb_tags or "").split(",") if token.strip()],
            dir=str(Path(args.exp_dir).resolve()),
            job_type="multilabel-training",
            config={
                **vars(args),
                "num_labels": vocab.size,
                "label_ids": list(vocab.label_ids),
                "loss": "BCEWithLogitsLoss",
            },
        )
    except Exception as exc:
        print(f"Warning: could not initialize wandb: {exc}")
        return None


def wandb_log_artifacts(paths: Sequence[Path], *, prefix: str = "artifacts") -> None:
    if wandb is None or wandb.run is None:
        return
    log_payload = {}
    for path in paths:
        if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            log_payload[f"{prefix}/{path.stem}"] = wandb.Image(str(path))
        elif path.exists():
            try:
                wandb.save(str(path))
            except Exception as exc:
                print(f"Warning: could not wandb.save {path}: {exc}")
    if log_payload:
        wandb.log(log_payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny multi-label BCE smoke training job")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--crop-time-seconds", type=float, default=None)
    parser.add_argument("--freq-min-hz", type=float, default=None)
    parser.add_argument("--freq-max-hz", type=float, default=None)
    parser.add_argument("--center-bias-sigma-frac", type=float, default=0.25)
    parser.add_argument("--positive-crop-mode", default="edge_mix")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-example-images", type=int, default=64)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="whale-multispecies-calltype")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default="multilabel,resnet,species")
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = get_device(args.device)
    exp_dir = Path(args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    vocab = LabelVocabulary.load(args.vocab_json)
    if vocab.size == 0:
        raise SystemExit("Vocabulary is empty; run the audit utility with trainable labels first")
    wandb_run = init_wandb_if_requested(args, vocab)
    crop_freq_range = None
    if args.freq_min_hz is not None and args.freq_max_hz is not None:
        crop_freq_range = (float(args.freq_min_hz), float(args.freq_max_hz))

    train_ds_full = MultiLabelMatDataset(
        args.manifest_csv,
        vocab,
        split="train",
        dataset_root=args.dataset_root,
        crop_size=int(args.crop_size),
        crop_time_seconds=args.crop_time_seconds,
        crop_freq_range_hz=crop_freq_range,
        center_bias_sigma_frac=float(args.center_bias_sigma_frac),
        positive_crop_mode=str(args.positive_crop_mode),
        seed=int(args.seed),
        return_meta=True,
    )
    val_ds_full = MultiLabelMatDataset(
        args.manifest_csv,
        vocab,
        split="val",
        dataset_root=args.dataset_root,
        crop_size=int(args.crop_size),
        crop_time_seconds=args.crop_time_seconds,
        crop_freq_range_hz=crop_freq_range,
        center_bias_sigma_frac=float(args.center_bias_sigma_frac),
        positive_crop_mode=str(args.positive_crop_mode),
        seed=int(args.seed) + 1,
        return_meta=True,
    )
    test_ds_full = MultiLabelMatDataset(
        args.manifest_csv,
        vocab,
        split="test",
        dataset_root=args.dataset_root,
        crop_size=int(args.crop_size),
        crop_time_seconds=args.crop_time_seconds,
        crop_freq_range_hz=crop_freq_range,
        center_bias_sigma_frac=float(args.center_bias_sigma_frac),
        positive_crop_mode=str(args.positive_crop_mode),
        seed=int(args.seed) + 2,
        return_meta=True,
    )
    train_ds = maybe_subset(train_ds_full, args.max_train_samples)
    val_ds = maybe_subset(val_ds_full, args.max_val_samples)
    test_ds = maybe_subset(test_ds_full, args.max_test_samples)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Need non-empty train and val splits, got train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
    ) if len(test_ds) else None

    model = create_model(args.model, num_classes=vocab.size, in_ch=1).to(device)
    init_info = None
    if args.init_checkpoint:
        init_info = load_partial_checkpoint(model, Path(args.init_checkpoint).resolve())
    pos_weight = compute_pos_weight(train_ds).to(device) if args.use_pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_path = exp_dir / "best.pt"
    best_eval: Optional[Dict[str, Any]] = None
    for epoch in range(1, int(args.epochs) + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        train_metrics = multilabel_metrics(train_result["targets"], train_result["scores"], threshold=float(args.threshold))
        val_result = evaluate(model, val_loader, device, loss_fn, float(args.threshold), max_example_images=int(args.max_example_images))
        val_metric = float(val_result["metrics"].get("macro_f1", 0.0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "train_metrics": train_metrics,
                "val_metrics": val_result["metrics"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} val_macro_f1={val_metric:.4f}"
        )
        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_result["loss"],
                    "val/loss": val_result["loss"],
                    "train/micro_f1": train_metrics.get("micro_f1", 0.0),
                    "train/macro_f1": train_metrics.get("macro_f1", 0.0),
                    "train/micro_precision": train_metrics.get("micro_precision", 0.0),
                    "train/micro_recall": train_metrics.get("micro_recall", 0.0),
                    "val/micro_f1": val_result["metrics"].get("micro_f1", 0.0),
                    "val/macro_f1": val_result["metrics"].get("macro_f1", 0.0),
                    "val/micro_precision": val_result["metrics"].get("micro_precision", 0.0),
                    "val/micro_recall": val_result["metrics"].get("micro_recall", 0.0),
                }
            )
        if val_metric >= best_metric:
            best_metric = val_metric
            best_eval = val_result
            checkpoint = create_checkpoint_metadata(model, args, wandb_run_id=None)
            checkpoint.update(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "architecture": args.model,
                    "num_labels": vocab.size,
                    "label_vocabulary": vocab.to_dict(),
                    "best_metric": best_metric,
                    "init_checkpoint": init_info,
                    "loss": "BCEWithLogitsLoss",
                    "pos_weight": pos_weight.detach().cpu().tolist() if pos_weight is not None else None,
                }
            )
            torch.save(checkpoint, best_path)

    if best_eval is None:
        best_eval = evaluate(model, val_loader, device, loss_fn, float(args.threshold), max_example_images=int(args.max_example_images))
    write_validation_exports(exp_dir, vocab, best_eval, threshold=float(args.threshold))
    plot_paths = []
    plot_paths.extend(write_training_plots(exp_dir, history))
    plot_paths.extend(write_validation_plots(exp_dir, vocab, best_eval))
    plot_paths.extend(write_source_stratified_metrics(exp_dir, vocab, best_eval, threshold=float(args.threshold)))
    plot_paths.extend(
        write_source_stratified_metrics(
            exp_dir,
            vocab,
            best_eval,
            threshold=float(args.threshold),
            group_field="source_kind",
            prefix="source_kind_metrics",
        )
    )
    threshold_sweep = write_threshold_sweep(exp_dir, vocab, best_eval)
    example_paths = write_example_images(exp_dir, vocab, best_eval, threshold=float(args.threshold))
    plot_paths.extend(example_paths)
    plot_paths.extend([Path(path) for path in threshold_sweep.get("paths", [])])

    test_metrics: Optional[Dict[str, Any]] = None
    if test_loader is not None:
        checkpoint = torch.load(best_path, map_location=device)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state"), dict):
            model.load_state_dict(checkpoint["model_state"])
        test_result = evaluate(
            model,
            test_loader,
            device,
            loss_fn,
            float(args.threshold),
            max_example_images=0,
        )
        test_metrics = test_result.get("metrics", {})
        write_prediction_exports(exp_dir, vocab, test_result, threshold=float(args.threshold), prefix="test")
        _write_csv_dicts(exp_dir / "test_per_class_metrics.csv", _per_class_rows(vocab, test_metrics))
        plot_paths.extend(
            write_source_stratified_metrics(
                exp_dir,
                vocab,
                test_result,
                threshold=float(args.threshold),
                prefix="test_source_metrics",
            )
        )
        plot_paths.extend(
            write_source_stratified_metrics(
                exp_dir,
                vocab,
                test_result,
                threshold=float(args.threshold),
                group_field="source_kind",
                prefix="test_source_kind_metrics",
            )
        )

    summary = {
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
        "vocab_json": str(Path(args.vocab_json).resolve()),
        "exp_dir": str(exp_dir),
        "model": args.model,
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "num_labels": vocab.size,
        "threshold": float(args.threshold),
        "best_metric": best_metric,
        "best_checkpoint": str(best_path),
        "init_checkpoint": init_info,
        "threshold_sweep": {key: value for key, value in threshold_sweep.items() if key != "paths"},
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(exp_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if wandb is not None and wandb.run is not None:
        rows = _per_class_rows(vocab, best_eval.get("metrics", {}))
        if rows:
            table = wandb.Table(columns=list(rows[0].keys()), data=[[row[key] for key in rows[0].keys()] for row in rows])
            wandb.log({"val/per_class_metrics": table})
        wandb.run.summary["best_macro_f1"] = best_metric
        wandb.run.summary["best_checkpoint"] = str(best_path)
        wandb_log_artifacts(
            [
                exp_dir / "run_summary.json",
                exp_dir / "validation_predictions.csv",
                exp_dir / "validation_predictions.o3_compatible.json",
                exp_dir / "test_predictions.csv",
                exp_dir / "test_predictions.o3_compatible.json",
                exp_dir / "per_class_metrics.csv",
                exp_dir / "source_metrics.csv",
                exp_dir / "source_metrics_summary.json",
                exp_dir / "source_kind_metrics.csv",
                exp_dir / "source_kind_metrics_summary.json",
                exp_dir / "test_per_class_metrics.csv",
                exp_dir / "test_source_metrics.csv",
                exp_dir / "test_source_metrics_summary.json",
                exp_dir / "test_source_kind_metrics.csv",
                exp_dir / "test_source_kind_metrics_summary.json",
                *plot_paths,
            ],
            prefix="multilabel",
        )
        wandb.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
