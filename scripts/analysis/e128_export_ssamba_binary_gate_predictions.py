#!/usr/bin/env python3
"""Export SSAMBA binary gate predictions in E126-compatible CSV format.

This bridges the selfsupervision_anomalies_onc H5/model stack to the
multispecies production reporting stack. It reads an E123/E126-style H5 file,
scores requested H5 splits with a fine-tuned SSAMBA binary model, and writes
CSV files that can be passed directly to e126_binary_gate_report.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_POSITIVE_LABELS = ("species:Bp", "species:Bm", "species:Mn")
DEFAULT_SCORE_LABEL = "task:whale_call"
DEFAULT_SPLITS = ("val", "test")
SHORT_TO_SPECIES = {
    "Bm": "species:Bm",
    "Bp": "species:Bp",
    "Mn": "species:Mn",
}


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def decode_h5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.bytes_):
            return value.decode("utf-8")
    except Exception:
        pass
    return clean(value)


def split_tokens(value: Any) -> List[str]:
    return [part.strip() for part in clean(value).replace(",", ";").replace("|", ";").split(";") if part.strip()]


def normalize_species_label(token: str) -> str:
    token = clean(token)
    if not token or token.lower() == "normal" or token == "background":
        return ""
    if token in SHORT_TO_SPECIES:
        return SHORT_TO_SPECIES[token]
    if token.startswith("species:"):
        return token
    return f"species:{token}"


def species_labels_from_h5_label_string(label_string: Any, positive_labels: Sequence[str]) -> List[str]:
    positives = set(positive_labels)
    labels: List[str] = []
    for token in split_tokens(decode_h5_text(label_string)):
        label = normalize_species_label(token)
        if label and label in positives and label not in labels:
            labels.append(label)
    return labels


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def probability_from_logits(logits: Sequence[float], positive_class_index: int = 1) -> float:
    values = [float(value) for value in logits]
    if not values:
        return 0.0
    if len(values) == 1:
        return sigmoid(values[0])
    if positive_class_index < 0 or positive_class_index >= len(values):
        raise ValueError(f"positive class index {positive_class_index} out of range for {len(values)} logits")
    max_logit = max(values)
    exp_values = [math.exp(value - max_logit) for value in values]
    denom = sum(exp_values)
    return exp_values[positive_class_index] / denom if denom else 0.0


def output_name_for_split(split: str) -> str:
    if split == "val":
        return "validation_predictions.csv"
    return f"{split}_predictions.csv"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def add_ssl_repo_to_path(ssl_repo_root: Optional[Path]) -> None:
    if ssl_repo_root is None:
        return
    root = ssl_repo_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SSL repo root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def resolve_checkpoint(model_dir: Path, checkpoint_path: Optional[Path]) -> Path:
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path
    models_dir = model_dir / "models"
    candidates = [
        models_dir / "ft-avgtok_best_checkpoint.pth",
        models_dir / "ft-cls_best_checkpoint.pth",
        models_dir / "best_checkpoint.pth",
        model_dir / "ft-avgtok_best_checkpoint.pth",
        model_dir / "ft-cls_best_checkpoint.pth",
        model_dir / "best_checkpoint.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    best = list(models_dir.glob("*best_checkpoint.pth")) if models_dir.is_dir() else []
    best.extend(model_dir.glob("*best_checkpoint.pth"))
    if best:
        best.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return best[0]
    raise FileNotFoundError(f"No SSAMBA checkpoint found under {model_dir}")


def load_model(model_dir: Path, checkpoint_path: Optional[Path], task: str, device: str):
    import torch  # type: ignore
    from onc_ssamba.utilities.checkpoint_utils import load_checkpoint  # type: ignore
    from onc_ssamba.utilities.training_utils import create_model  # type: ignore

    args_path = model_dir / "args.pkl"
    if not args_path.is_file():
        raise FileNotFoundError(f"args.pkl does not exist in model dir: {model_dir}")
    with args_path.open("rb") as handle:
        model_args = pickle.load(handle)
    if task:
        model_args.task = task
    model_args.exp_dir = str(model_dir)
    if not hasattr(model_args, "multiclass"):
        model_args.multiclass = False
    if not hasattr(model_args, "num_classes"):
        model_args.num_classes = 2

    checkpoint = resolve_checkpoint(model_dir, checkpoint_path)
    model = create_model(model_args).to(device)
    payload = load_checkpoint(str(checkpoint), torch.device(device))
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        state_dict = payload.get("model_state") if isinstance(payload, dict) else payload
    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:] if str(key).startswith("module.") else str(key): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing[:10]}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {unexpected[:10]}", file=sys.stderr)
    model.eval()
    return model, model_args, checkpoint


def h5_strings(h5: Any, name: str, length: int, default: str = "") -> List[str]:
    if name not in h5:
        return [default for _ in range(length)]
    values = h5[name][:]
    return [decode_h5_text(value) for value in values]


def normalize_batch(batch: Any, *, dataset_mean: Optional[float], dataset_std: Optional[float], amount: float):
    import numpy as np  # type: ignore

    arr = np.asarray(batch, dtype=np.float32)
    if dataset_mean is not None and dataset_std is not None:
        arr = (arr - float(dataset_mean)) / (float(dataset_std) * 2.0)
    else:
        flat = arr.reshape((arr.shape[0], -1))
        lo = np.percentile(flat, amount, axis=1)
        hi = np.percentile(flat, 100.0 - amount, axis=1)
        arr = np.clip(arr, lo[:, None, None, None], hi[:, None, None, None])
        arr = np.log(np.maximum(arr, 1e-12))
        flat = arr.reshape((arr.shape[0], -1))
        mn = flat.min(axis=1)
        mx = flat.max(axis=1)
        denom = np.maximum(mx - mn, 1e-12)
        arr = (arr - mn[:, None, None, None]) / denom[:, None, None, None]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def score_h5_split(
    *,
    h5_path: Path,
    split: str,
    model: Any,
    model_args: Any,
    task: str,
    device: str,
    batch_size: int,
    positive_labels: Sequence[str],
    score_label: str,
    positive_class_index: int,
) -> List[Dict[str, Any]]:
    import h5py  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore

    with h5py.File(h5_path, "r") as h5:
        n = int(h5["spectrograms"].shape[0])
        splits = h5_strings(h5, "splits", n, default="")
        label_strings = h5_strings(h5, "label_strings", n, default="normal")
        item_ids = h5_strings(h5, "item_ids", n, default="")
        sources = h5_strings(h5, "sources", n, default="")
        source_kinds = h5_strings(h5, "source_kinds", n, default="")
        indices = [idx for idx, value in enumerate(splits) if value == split]
        rows: List[Dict[str, Any]] = []
        if not indices:
            return rows
        dataset_mean = getattr(model_args, "dataset_mean", None)
        dataset_std = getattr(model_args, "dataset_std", None)
        amount = float(getattr(model_args, "amount", 1.0) or 1.0)
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            spectrograms = h5["spectrograms"][batch_indices]
            spectrograms = normalize_batch(spectrograms, dataset_mean=dataset_mean, dataset_std=dataset_std, amount=amount)
            tensor = torch.from_numpy(np.asarray(spectrograms)).permute(0, 3, 1, 2).float().to(device)
            with torch.no_grad():
                logits = model(tensor, task=task)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits_np = logits.detach().cpu().numpy()
            for local_idx, row_idx in enumerate(batch_indices):
                raw_logits = np.asarray(logits_np[local_idx]).reshape(-1).tolist()
                score = probability_from_logits(raw_logits, positive_class_index=positive_class_index)
                species_labels = species_labels_from_h5_label_string(label_strings[row_idx], positive_labels)
                target = "|".join(species_labels)
                rows.append(
                    {
                        "item_id": item_ids[row_idx] or str(row_idx),
                        "source_audio": sources[row_idx],
                        "source_dataset": "E123_E126_H5",
                        "source_kind": source_kinds[row_idx],
                        "split": split,
                        "h5_index": row_idx,
                        "h5_path": str(h5_path),
                        "h5_label_string": label_strings[row_idx],
                        "target_label_ids": target,
                        "original_label_ids": target,
                        "gate_positive_source_labels": target,
                        "true_class": species_labels[0] if species_labels else "background",
                        "score__" + score_label: f"{score:.8f}",
                    }
                )
        return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssl-repo-root", type=Path, default=None)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--dataset-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", action="append", default=None, help="H5 split to export; default val and test")
    parser.add_argument("--positive-labels", default=",".join(DEFAULT_POSITIVE_LABELS))
    parser.add_argument("--score-label", default=DEFAULT_SCORE_LABEL)
    parser.add_argument("--task", default="ft_avgtok")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--positive-class-index", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    add_ssl_repo_to_path(args.ssl_repo_root)
    if not args.dataset_h5.is_file():
        raise FileNotFoundError(f"H5 dataset does not exist: {args.dataset_h5}")
    import torch  # type: ignore

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args, checkpoint = load_model(args.model_dir, args.checkpoint_path, args.task, device)
    positive_labels = [label for label in split_tokens(args.positive_labels)]
    outputs: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.split or list(DEFAULT_SPLITS):
        rows = score_h5_split(
            h5_path=args.dataset_h5,
            split=split,
            model=model,
            model_args=model_args,
            task=args.task,
            device=device,
            batch_size=args.batch_size,
            positive_labels=positive_labels,
            score_label=args.score_label,
            positive_class_index=args.positive_class_index,
        )
        output_path = args.output_dir / output_name_for_split(split)
        write_csv(output_path, rows)
        outputs[split] = str(output_path)
        counts[split] = len(rows)
    summary = {
        "model_dir": str(args.model_dir),
        "checkpoint": str(checkpoint),
        "dataset_h5": str(args.dataset_h5),
        "task": args.task,
        "score_label": args.score_label,
        "positive_labels": positive_labels,
        "outputs": outputs,
        "rows_by_split": counts,
    }
    summary_path = args.output_dir / "e128_ssamba_binary_prediction_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "outputs": outputs, "rows_by_split": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
