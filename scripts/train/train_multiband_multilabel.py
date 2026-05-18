#!/usr/bin/env python3
"""Train a multi-label model on aligned low/mid/high MAT spectrograms."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except Exception:
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.train_multilabel_resnet_smoke import (  # noqa: E402
    _per_class_rows,
    _write_csv_dicts,
    compute_pos_weight,
    get_device,
    init_wandb_if_requested,
    maybe_subset,
    set_seed,
    wandb_log_artifacts,
    write_example_images,
    write_prediction_exports,
    write_source_stratified_metrics,
    write_threshold_sweep,
    write_training_plots,
    write_validation_plots,
)
from src.dataset.multiband import MultiBandMatDataset, parse_band_crop_shapes  # noqa: E402
from src.dataset.multilabel import LabelVocabulary, multilabel_metrics  # noqa: E402
from src.models.multiband import create_multiband_model, load_resnet_encoder_checkpoint  # noqa: E402
from src.utils.model_utils import create_checkpoint_metadata  # noqa: E402


class BalancedBCEWithLogitsLoss(nn.Module):
    """Batch-balanced BCE that gives positive and negative examples equal label weight."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if pos_weight is None:
            self.register_buffer("pos_weight", None, persistent=False)
        else:
            self.register_buffer("pos_weight", pos_weight.detach().clone(), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        pos_mask = targets >= 0.5
        neg_mask = ~pos_mask
        pos_count = pos_mask.sum(dim=0).clamp_min(1)
        neg_count = neg_mask.sum(dim=0).clamp_min(1)
        pos_loss = (loss * pos_mask.to(loss.dtype)).sum(dim=0) / pos_count
        neg_loss = (loss * neg_mask.to(loss.dtype)).sum(dim=0) / neg_count
        has_pos = pos_mask.any(dim=0)
        has_neg = neg_mask.any(dim=0)
        both = has_pos & has_neg
        per_label = torch.where(
            both,
            0.5 * (pos_loss + neg_loss),
            torch.where(has_pos, pos_loss, neg_loss),
        )
        return per_label.mean()


def build_label_band_mask(
    *,
    label_ids: Sequence[str],
    bands: Sequence[str],
    mode: str,
) -> torch.Tensor:
    mode = str(mode or "none").strip().lower()
    band_index = {band: idx for idx, band in enumerate(bands)}
    mask = torch.ones(len(bands), len(label_ids), dtype=torch.float32)
    if mode in {"", "none", "all"}:
        return mask

    def apply(label_id: str, allowed: Sequence[str]) -> None:
        if label_id not in label_ids:
            return
        col = list(label_ids).index(label_id)
        mask[:, col] = 0.0
        for band in allowed:
            idx = band_index.get(band)
            if idx is not None:
                mask[idx, col] = 1.0
        if float(mask[:, col].sum()) <= 0.0:
            mask[:, col] = 1.0

    if mode == "audit_v1":
        apply("species:Bm", ("low",))
        apply("species:Bp", ("low",))
        apply("species:Mn", ("low", "mid"))
        apply("species:Oo", ("mid", "high"))
        return mask
    if mode == "audit_v2":
        apply("species:Bm", ("low", "mid"))
        apply("species:Bp", ("low", "mid"))
        apply("species:Mn", ("low", "mid"))
        apply("species:Oo", ("mid", "high"))
        return mask
    if mode == "odont_high":
        apply("species:Bm", ("low", "mid"))
        apply("species:Bp", ("low", "mid"))
        apply("species:Mn", ("low", "mid", "high"))
        apply("species:Oo", ("mid", "high"))
        return mask
    raise ValueError(
        "--class-band-mask-mode must be one of: none, audit_v1, audit_v2, odont_high"
    )


def collate_batch(batch: Sequence[Sequence[Any]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[List[Dict[str, Any]]]]:
    bands = list(batch[0][0].keys())
    xs = {band: torch.stack([item[0][band] for item in batch], dim=0) for band in bands}
    ys = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch if len(item) == 3]
    return xs, ys, metas if metas else None


def _move_inputs(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {band: tensor.to(device, non_blocking=True) for band, tensor in inputs.items()}


def _unpack_batch(batch: Sequence[Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[List[Dict[str, Any]]]]:
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
        x = _move_inputs(x, device)
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
    image_band: str = "high",
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
        x_dev = _move_inputs(x, device)
        y = y.to(device, non_blocking=True)
        logits = model(x_dev)
        loss = loss_fn(logits, y)

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        scores.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        if meta is not None:
            metas.extend(meta)
        if len(images) < int(max_example_images):
            band = image_band if image_band in x else next(iter(x.keys()))
            room = int(max_example_images) - len(images)
            for arr in x[band].detach().cpu().numpy()[:room, 0]:
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


def _load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
        if isinstance(state, dict):
            return {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}
        return {str(key): value for key, value in ckpt.items() if isinstance(value, torch.Tensor)}
    raise ValueError(f"Could not read checkpoint state from {path}")


def _freeze_branch_encoders(model: nn.Module) -> Dict[str, int]:
    branches = getattr(model, "branches", None)
    if branches is None:
        return {"frozen_parameters": 0, "frozen_tensors": 0}
    frozen_parameters = 0
    frozen_tensors = 0
    for param in branches.parameters():
        if param.requires_grad:
            frozen_parameters += int(param.numel())
            frozen_tensors += 1
        param.requires_grad_(False)
    return {"frozen_parameters": frozen_parameters, "frozen_tensors": frozen_tensors}


def _trainable_parameter_count(model: nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--bands", default="low,mid,high")
    parser.add_argument("--band-crop-shapes", default="low:391x50,mid:256x100,high:256x312")
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--fusion", default="gated", choices=["gated", "concat", "mean_logits", "mean"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--init-low-checkpoint", default=None)
    parser.add_argument("--init-all-branches-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-time-seconds", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=40.0)
    parser.add_argument("--center-bias-sigma-frac", type=float, default=0.25)
    parser.add_argument("--positive-crop-mode", default="edge_mix")
    parser.add_argument("--band-availability-mode", default="all", choices=["all", "metadata", "source", "source_or_metadata", "audit_v1"])
    parser.add_argument("--class-band-mask-mode", default="none", choices=["none", "audit_v1", "audit_v2", "odont_high"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--loss-mode", default="bce", choices=["bce", "balanced_bce"])
    parser.add_argument("--freeze-branches", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-example-images", type=int, default=64)
    parser.add_argument("--example-image-band", default="high")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="whale-multispecies-calltype")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default="multilabel,multiband,species")
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = get_device(args.device)
    exp_dir = Path(args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    bands = [token.strip() for token in str(args.bands).split(",") if token.strip()]
    band_shapes = parse_band_crop_shapes(args.band_crop_shapes)
    vocab = LabelVocabulary.load(args.vocab_json)
    if vocab.size == 0:
        raise SystemExit("Vocabulary is empty")
    wandb_run = init_wandb_if_requested(args, vocab)

    ds_kwargs = dict(
        dataset_root=args.dataset_root,
        bands=bands,
        band_crop_shapes=band_shapes,
        crop_time_seconds=float(args.crop_time_seconds),
        context_seconds=float(args.context_seconds),
        center_bias_sigma_frac=float(args.center_bias_sigma_frac),
        positive_crop_mode=str(args.positive_crop_mode),
        band_availability_mode=str(args.band_availability_mode),
        return_meta=True,
    )
    train_ds_full = MultiBandMatDataset(args.manifest_csv, vocab, split="train", seed=int(args.seed), **ds_kwargs)
    val_ds_full = MultiBandMatDataset(args.manifest_csv, vocab, split="val", seed=int(args.seed) + 1, **ds_kwargs)
    test_ds_full = MultiBandMatDataset(args.manifest_csv, vocab, split="test", seed=int(args.seed) + 2, **ds_kwargs)
    train_ds: Dataset = maybe_subset(train_ds_full, args.max_train_samples)
    val_ds: Dataset = maybe_subset(val_ds_full, args.max_val_samples)
    test_ds: Dataset = maybe_subset(test_ds_full, args.max_test_samples)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Need non-empty train and val splits, got train={len(train_ds)} val={len(val_ds)}")

    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "collate_fn": collate_batch,
        "pin_memory": str(device).startswith("cuda"),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs) if len(test_ds) else None

    model = create_multiband_model(
        encoder=str(args.encoder),
        num_classes=vocab.size,
        bands=bands,
        fusion=str(args.fusion),
        dropout=float(args.dropout),
        in_ch=1,
        label_band_mask=build_label_band_mask(
            label_ids=vocab.label_ids,
            bands=bands,
            mode=str(args.class_band_mask_mode),
        ),
    ).to(device)
    init_info: Dict[str, Any] = {}
    if args.init_all_branches_checkpoint:
        state = _load_checkpoint_state(Path(args.init_all_branches_checkpoint).resolve())
        init_info["all_branches"] = load_resnet_encoder_checkpoint(model, state, bands=bands)
    if args.init_low_checkpoint:
        state = _load_checkpoint_state(Path(args.init_low_checkpoint).resolve())
        init_info["low"] = load_resnet_encoder_checkpoint(model, state, bands=["low"])

    freeze_info: Dict[str, Any] = {"freeze_branches": bool(args.freeze_branches)}
    if args.freeze_branches:
        freeze_info.update(_freeze_branch_encoders(model))
    freeze_info["trainable_parameters"] = _trainable_parameter_count(model)

    pos_weight = compute_pos_weight(train_ds).to(device) if args.use_pos_weight else None
    if str(args.loss_mode) == "balanced_bce":
        loss_fn = BalancedBCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise SystemExit("No trainable parameters remain after applying freeze options")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_path = exp_dir / "best.pt"
    best_eval: Optional[Dict[str, Any]] = None
    for epoch in range(1, int(args.epochs) + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        train_metrics = multilabel_metrics(train_result["targets"], train_result["scores"], threshold=float(args.threshold))
        val_result = evaluate(
            model,
            val_loader,
            device,
            loss_fn,
            float(args.threshold),
            max_example_images=int(args.max_example_images),
            image_band=str(args.example_image_band),
        )
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
            f"val_loss={val_result['loss']:.4f} val_macro_f1={val_metric:.4f}",
            flush=True,
        )
        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_result["loss"],
                    "val/loss": val_result["loss"],
                    "train/micro_f1": train_metrics.get("micro_f1", 0.0),
                    "train/macro_f1": train_metrics.get("macro_f1", 0.0),
                    "val/micro_f1": val_result["metrics"].get("micro_f1", 0.0),
                    "val/macro_f1": val_result["metrics"].get("macro_f1", 0.0),
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
                    "architecture": f"multiband-{args.encoder}-{args.fusion}",
                    "num_labels": vocab.size,
                    "label_vocabulary": vocab.to_dict(),
                    "best_metric": best_metric,
                    "init_checkpoint": init_info,
                    "loss": str(args.loss_mode),
                    "pos_weight": pos_weight.detach().cpu().tolist() if pos_weight is not None else None,
                    "freeze_info": freeze_info,
                    "bands": bands,
                    "band_crop_shapes": band_shapes,
                    "band_availability_mode": args.band_availability_mode,
                    "class_band_mask_mode": args.class_band_mask_mode,
                }
            )
            torch.save(checkpoint, best_path)

    if best_eval is None:
        best_eval = evaluate(model, val_loader, device, loss_fn, float(args.threshold), max_example_images=int(args.max_example_images))
    write_prediction_exports(exp_dir, vocab, best_eval, threshold=float(args.threshold), prefix="validation")
    plot_paths: List[Path] = []
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
    plot_paths.extend(write_example_images(exp_dir, vocab, best_eval, threshold=float(args.threshold)))
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
            image_band=str(args.example_image_band),
        )
        test_metrics = test_result.get("metrics", {})
        write_prediction_exports(exp_dir, vocab, test_result, threshold=float(args.threshold), prefix="test")
        _write_csv_dicts(exp_dir / "test_per_class_metrics.csv", _per_class_rows(vocab, test_metrics))
        plot_paths.extend(write_source_stratified_metrics(exp_dir, vocab, test_result, threshold=float(args.threshold), prefix="test_source_metrics"))
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
        "encoder": args.encoder,
        "fusion": args.fusion,
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "num_labels": vocab.size,
        "threshold": float(args.threshold),
        "best_metric": best_metric,
        "best_checkpoint": str(best_path),
        "init_checkpoint": init_info,
        "loss": str(args.loss_mode),
        "pos_weight": pos_weight.detach().cpu().tolist() if pos_weight is not None else None,
        "freeze_info": freeze_info,
        "bands": bands,
        "band_crop_shapes": band_shapes,
        "band_availability_mode": args.band_availability_mode,
        "class_band_mask_mode": args.class_band_mask_mode,
        "threshold_sweep": {key: value for key, value in threshold_sweep.items() if key != "paths"},
        "test_metrics": test_metrics,
        "history": history,
    }
    (exp_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if wandb is not None and wandb.run is not None:
        wandb.run.summary["best_macro_f1"] = best_metric
        wandb.run.summary["best_checkpoint"] = str(best_path)
        wandb_log_artifacts(
            [
                exp_dir / "run_summary.json",
                exp_dir / "validation_predictions.csv",
                exp_dir / "test_predictions.csv",
                exp_dir / "per_class_metrics.csv",
                exp_dir / "source_metrics.csv",
                exp_dir / "source_kind_metrics.csv",
                exp_dir / "test_per_class_metrics.csv",
                exp_dir / "test_source_metrics.csv",
                exp_dir / "test_source_kind_metrics.csv",
                *plot_paths,
            ],
            prefix="multiband",
        )
        wandb.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
