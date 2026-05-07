#!/usr/bin/env python3
"""Summarize multi-label validation/test predictions with source calibration.

The trainer exports one score column per label. This utility chooses per-label
thresholds on a calibration subset (ONC validation by default), applies them to
an evaluation subset (ONC test by default), and separately reports any-primary
false positives for no-primary/hard-negative buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PRIMARY_LABELS = ("species:Bm", "species:Bp", "species:Mn", "species:Oo")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _split_pipe(value: Any) -> List[str]:
    return [token.strip() for token in str(value or "").split("|") if token.strip()]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_labels(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    labels: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("score__"):
                label = key.removeprefix("score__")
                if label not in labels:
                    labels.append(label)
    return labels


def _arrays(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((len(rows), len(labels)), dtype=np.float32)
    targets = np.zeros_like(scores)
    for row_idx, row in enumerate(rows):
        target_ids = set(_split_pipe(row.get("target_label_ids")))
        for label_idx, label in enumerate(labels):
            scores[row_idx, label_idx] = float(row.get(f"score__{label}") or 0.0)
            targets[row_idx, label_idx] = 1.0 if label in target_ids else 0.0
    return scores, targets


def _filter_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_kind: str = "",
    source_dataset_contains: str = "",
) -> List[int]:
    out: List[int] = []
    source_kind = source_kind.strip()
    source_dataset_contains = source_dataset_contains.strip()
    for idx, row in enumerate(rows):
        if source_kind and _clean(row.get("source_kind")) != source_kind:
            continue
        if source_dataset_contains and source_dataset_contains not in _clean(row.get("source_dataset")):
            continue
        out.append(idx)
    return out


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _choose_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    indices: Sequence[int],
    *,
    label_ids: Sequence[str],
    fallback_thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    thresholds = [round(float(value), 2) for value in np.linspace(0.05, 0.95, 91)]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    out: Dict[str, Dict[str, Any]] = {}
    subset = list(indices)
    for label in label_ids:
        label_idx = label_to_idx.get(label)
        if label_idx is None or not subset:
            out[label] = {
                "threshold": float((fallback_thresholds or {}).get(label, 0.5)),
                "support": 0,
                "calibrated": False,
                "reason": "missing_label_or_empty_subset",
            }
            continue
        y_true = (targets[subset, label_idx] >= 0.5).astype(np.int64)
        support = int(y_true.sum())
        if support == 0:
            out[label] = {
                "threshold": float((fallback_thresholds or {}).get(label, 0.5)),
                "support": 0,
                "calibrated": False,
                "reason": "zero_calibration_support",
            }
            continue
        best: Dict[str, Any] = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
        for threshold in thresholds:
            y_pred = (scores[subset, label_idx] >= threshold).astype(np.int64)
            counts = _binary_counts(y_true, y_pred)
            key = (float(counts["f1"]), float(counts["precision"]), float(threshold))
            best_key = (float(best["f1"]), float(best["precision"]), float(best["threshold"]))
            if key > best_key:
                best = {"threshold": threshold, "support": support, "calibrated": True, **counts}
        out[label] = best
    return out


def _evaluate_with_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    indices: Sequence[int],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    label_ids: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    rows: List[Dict[str, Any]] = []
    tp_total = fp_total = fn_total = 0
    f1_values: List[float] = []
    subset = list(indices)
    for label in label_ids:
        label_idx = label_to_idx.get(label)
        threshold = float(thresholds.get(label, {}).get("threshold", 0.5))
        if label_idx is None or not subset:
            row = {"label_id": label, "threshold": threshold, "support": 0, "tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            y_true = (targets[subset, label_idx] >= 0.5).astype(np.int64)
            y_pred = (scores[subset, label_idx] >= threshold).astype(np.int64)
            counts = _binary_counts(y_true, y_pred)
            row = {"label_id": label, "threshold": threshold, "support": int(y_true.sum()), **counts}
        rows.append(row)
        tp_total += int(row["tp"])
        fp_total += int(row["fp"])
        fn_total += int(row["fn"])
        if int(row["support"]) > 0:
            f1_values.append(float(row["f1"]))
    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    micro_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    summary = {
        "samples": len(subset),
        "macro_f1_supported": float(np.mean(f1_values)) if f1_values else 0.0,
        "micro_f1": float(micro_f1),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
    }
    return summary, rows


def _hard_negative_fp_rows(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    labels: Sequence[str],
    indices: Sequence[int],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    label_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    primary_indices = [label_to_idx[label] for label in label_ids if label in label_to_idx]
    threshold_vector = np.array([float(thresholds.get(label, {}).get("threshold", 0.5)) for label in label_ids if label in label_to_idx], dtype=np.float32)
    grouped: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    primary_set = set(label_ids)
    for idx in indices:
        row = rows[idx]
        target_primary = primary_set.intersection(_split_pipe(row.get("target_label_ids")))
        if target_primary:
            continue
        bucket = _clean(row.get("negative_bucket")) or "<blank>"
        source_kind = _clean(row.get("source_kind")) or "<blank>"
        source_dataset = _clean(row.get("source_dataset")) or "<blank>"
        key = (source_kind, source_dataset, bucket)
        grouped[key]["rows"] += 1
        if primary_indices:
            pred_any = bool(np.any(scores[idx, primary_indices] >= threshold_vector))
        else:
            pred_any = False
        if pred_any:
            grouped[key]["any_primary_fp"] += 1
    out: List[Dict[str, Any]] = []
    for (source_kind, source_dataset, bucket), counts in sorted(grouped.items()):
        rows_n = int(counts["rows"])
        fps = int(counts["any_primary_fp"])
        out.append(
            {
                "source_kind": source_kind,
                "source_dataset": source_dataset,
                "negative_bucket": bucket,
                "rows": rows_n,
                "any_primary_fp": fps,
                "any_primary_fp_rate": fps / max(rows_n, 1),
            }
        )
    return out


def summarize(
    *,
    validation_csv: Path,
    test_csv: Path,
    output_dir: Path,
    calibration_source_kind: str,
    eval_source_kind: str,
    label_ids: Sequence[str],
) -> Dict[str, Any]:
    val_rows = _read_csv(validation_csv)
    test_rows = _read_csv(test_csv) if test_csv.exists() else []
    labels = _score_labels(val_rows or test_rows)
    val_scores, val_targets = _arrays(val_rows, labels)
    test_scores, test_targets = _arrays(test_rows, labels)
    val_cal_idx = _filter_indices(val_rows, source_kind=calibration_source_kind)
    test_eval_idx = _filter_indices(test_rows, source_kind=eval_source_kind) if test_rows else []
    global_thresholds = _choose_thresholds(val_scores, val_targets, labels, range(len(val_rows)), label_ids=label_ids)
    onc_thresholds = _choose_thresholds(
        val_scores,
        val_targets,
        labels,
        val_cal_idx,
        label_ids=label_ids,
        fallback_thresholds={label: float(global_thresholds[label]["threshold"]) for label in global_thresholds},
    )
    val_summary, val_per_label = _evaluate_with_thresholds(
        val_scores,
        val_targets,
        labels,
        val_cal_idx,
        onc_thresholds,
        label_ids=label_ids,
    )
    test_summary, test_per_label = _evaluate_with_thresholds(
        test_scores,
        test_targets,
        labels,
        test_eval_idx,
        onc_thresholds,
        label_ids=label_ids,
    )
    hard_negative_rows = _hard_negative_fp_rows(
        test_rows,
        test_scores,
        labels,
        test_eval_idx,
        onc_thresholds,
        label_ids=label_ids,
    ) if test_rows else []

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "onc_calibrated_val_per_label.csv", val_per_label)
    _write_csv(output_dir / "onc_calibrated_test_per_label.csv", test_per_label)
    _write_csv(output_dir / "onc_calibrated_test_hard_negative_fp.csv", hard_negative_rows)
    summary = {
        "validation_csv": str(validation_csv),
        "test_csv": str(test_csv) if test_csv.exists() else "",
        "calibration_source_kind": calibration_source_kind,
        "eval_source_kind": eval_source_kind,
        "label_ids": list(label_ids),
        "score_labels": labels,
        "global_validation_thresholds": global_thresholds,
        "onc_validation_thresholds": onc_thresholds,
        "onc_validation_metrics": val_summary,
        "onc_test_metrics": test_summary,
        "onc_test_hard_negative_fp_rows": hard_negative_rows,
    }
    (output_dir / "onc_calibrated_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-source-kind", default="ONC")
    parser.add_argument("--eval-source-kind", default="ONC")
    parser.add_argument("--label-ids", default=",".join(PRIMARY_LABELS))
    args = parser.parse_args()
    summary = summarize(
        validation_csv=Path(args.validation_csv),
        test_csv=Path(args.test_csv),
        output_dir=Path(args.output_dir),
        calibration_source_kind=args.calibration_source_kind,
        eval_source_kind=args.eval_source_kind,
        label_ids=[token.strip() for token in str(args.label_ids).split(",") if token.strip()],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
