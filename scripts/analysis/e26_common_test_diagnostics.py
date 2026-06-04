#!/usr/bin/env python3
"""Diagnose E26 experts on a shared ONC validation/test set.

The standard E26 report combines filtered single-species experts. Each expert's
native test export only contains that expert's positives plus background rows.
This diagnostic reruns the trained E26 experts on a common ONC split so that
cross-species false positives are measured directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.e24_collect_expert_hparam_report import (  # noqa: E402
    LABEL_NAMES,
    THREE_SPECIES,
    base_key,
    clean,
    read_tsv,
    split_labels,
    write_csv,
)
from scripts.train.train_multiband_multilabel import (  # noqa: E402
    build_label_band_mask,
    collate_batch,
)
from src.dataset.multiband import MultiBandMatDataset, parse_band_crop_shapes  # noqa: E402
from src.dataset.multilabel import LabelVocabulary, label_ids_from_row, read_csv_rows  # noqa: E402
from src.models.multiband import create_multiband_model  # noqa: E402


LABEL_ORDER = tuple(THREE_SPECIES)


def _label_text(labels: Iterable[str]) -> str:
    return "|".join(label for label in LABEL_ORDER if label in set(labels))


def _species_name(label_id: str) -> str:
    return LABEL_NAMES.get(label_id, label_id)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _first_present(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean(row.get(key))
        if value:
            return value
    return ""


def _merge_label_text(existing: str, new: Iterable[str]) -> str:
    labels: List[str] = []
    for label in [*split_labels(existing), *new]:
        if label in LABEL_ORDER and label not in labels:
            labels.append(label)
    return _label_text(labels)


def _common_rows(plan_rows: Sequence[Mapping[str, str]], *, split: str, source_kind: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    merged: Dict[str, Dict[str, str]] = {}
    provenance: Counter[str] = Counter()
    skipped: List[Dict[str, str]] = []
    for plan in plan_rows:
        manifest_text = _first_present(plan, "manifest_csv", "manifest")
        manifest = Path(manifest_text)
        variant = clean(plan.get("variant"))
        if not variant or not clean(plan.get("experiment")) or not manifest.is_file():
            skipped.append(
                {
                    "experiment": clean(plan.get("experiment")),
                    "variant": variant,
                    "manifest_csv": manifest_text,
                    "reason": "missing_or_invalid_manifest",
                }
            )
            continue
        for row in _read_csv(manifest):
            if clean(row.get("split")) != split:
                continue
            if source_kind and clean(row.get("source_kind")) != source_kind:
                continue
            key = base_key(row)
            labels = [label for label in label_ids_from_row(row) if label in LABEL_ORDER]
            if key not in merged:
                out = dict(row)
                out["item_id"] = clean(row.get("item_id")) or key
                out["split"] = split
                for field in ("label_ids", "canonical_label_ids", "target_label_ids", "analysis_label_ids"):
                    out[field] = _label_text(labels)
                out["is_background"] = "0" if labels else "1"
                out["source_variants"] = variant
                merged[key] = out
            else:
                out = merged[key]
                for field in ("label_ids", "canonical_label_ids", "target_label_ids", "analysis_label_ids"):
                    out[field] = _merge_label_text(out.get(field, ""), labels)
                out["is_background"] = "0" if split_labels(out.get("target_label_ids")) else "1"
                existing_variants = split_labels(out.get("source_variants", ""))
                if variant not in existing_variants:
                    out["source_variants"] = "|".join([*existing_variants, variant])
            provenance[variant] += 1
    rows = sorted(merged.values(), key=lambda row: (clean(row.get("source_audio")), clean(row.get("begin_s")), clean(row.get("end_s")), clean(row.get("item_id"))))
    support = Counter()
    for row in rows:
        labs = set(split_labels(row.get("target_label_ids")))
        if not labs:
            support["<background>"] += 1
        for label in LABEL_ORDER:
            if label in labs:
                support[label] += 1
    return rows, {
        "split": split,
        "source_kind": source_kind,
        "rows": len(rows),
        "input_rows_by_variant": dict(provenance),
        "skipped_plan_rows": skipped,
        "support": dict(support),
    }


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(path, rows, fieldnames=fieldnames)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@torch.no_grad()
def _score_split(
    *,
    common_manifest: Path,
    split: str,
    run_meta: Mapping[str, Any],
    label_id: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    vocab = LabelVocabulary.load(run_meta["vocab_json"])
    if vocab.size != 1 or vocab.label_ids[0] != label_id:
        raise ValueError(f"Expected one-label vocab for {label_id}, got {vocab.label_ids}")
    bands = [token.strip() for token in str(run_meta["bands"]).split(",") if token.strip()]
    band_shapes = parse_band_crop_shapes(run_meta.get("band_crop_shapes"))
    dataset = MultiBandMatDataset(
        common_manifest,
        vocab,
        split=split,
        dataset_root=run_meta.get("dataset_root"),
        bands=bands,
        band_crop_shapes=band_shapes,
        crop_time_seconds=float(run_meta.get("crop_time_seconds", 10.0)),
        context_seconds=40.0,
        center_bias_sigma_frac=0.25,
        positive_crop_mode=str(run_meta.get("positive_crop_mode", "centered_gaussian")),
        band_availability_mode=str(run_meta.get("band_availability_mode", "all")),
        seed=2026 + (1 if split == "val" else 2),
        return_meta=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_batch,
        pin_memory=str(device).startswith("cuda"),
    )
    model = create_multiband_model(
        encoder=str(run_meta.get("encoder", "resnet18")),
        num_classes=vocab.size,
        bands=bands,
        fusion=str(run_meta.get("fusion", "gated")),
        head_type=str(run_meta.get("head_type", "shared")),
        dropout=float(run_meta.get("dropout", 0.3)),
        in_ch=1,
        label_band_mask=build_label_band_mask(
            label_ids=vocab.label_ids,
            bands=bands,
            mode=str(run_meta.get("class_band_mask_mode", "none")),
        ),
    ).to(device)
    ckpt_path = Path(run_meta["train_dir"]) / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"Missing model_state in {ckpt_path}")
    model.load_state_dict(state)
    model.eval()

    scores: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    for batch in loader:
        x, _, meta = batch
        x = {band: tensor.to(device, non_blocking=True) for band, tensor in x.items()}
        logits = model(x)
        scores.append(torch.sigmoid(logits).detach().cpu().numpy()[:, 0])
        metas.extend(meta or [])
    return metas, np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def _choose_common_thresholds(val_rows: Sequence[Mapping[str, Any]], val_scores: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    thresholds = [round(float(value), 2) for value in np.linspace(0.05, 0.95, 91)]
    out: Dict[str, Dict[str, Any]] = {}
    for label in LABEL_ORDER:
        y_true = np.array([1 if label in split_labels(row.get("target_label_ids")) else 0 for row in val_rows], dtype=np.int64)
        scores = np.asarray(val_scores[label], dtype=np.float32)
        best: Dict[str, Any] = {"threshold": 0.5, "support": int(y_true.sum()), "f1": -1.0, "precision": 0.0, "recall": 0.0}
        for threshold in thresholds:
            counts = _binary_counts(y_true, (scores >= threshold).astype(np.int64))
            key = (float(counts["f1"]), float(counts["precision"]), float(threshold))
            best_key = (float(best["f1"]), float(best["precision"]), float(best["threshold"]))
            if key > best_key:
                best = {"threshold": threshold, "support": int(y_true.sum()), **counts}
        out[label] = best
    return out


def _load_original_thresholds(run_metas: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for label, meta in run_metas.items():
        metrics_path = Path(meta["train_dir"]) / "onc_calibrated_eval" / "onc_calibrated_metrics_summary.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        thresholds[label] = float(payload["onc_validation_thresholds"][label]["threshold"])
    return thresholds


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    *,
    threshold_name: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        true = set(split_labels(row.get("target_label_ids")))
        pred = {label for label in LABEL_ORDER if float(scores[label][idx]) >= float(thresholds[label])}
        record: Dict[str, Any] = {
            "row_index": idx,
            "item_id": clean(row.get("item_id")),
            "source_audio": clean(row.get("source_audio")),
            "source_dataset": clean(row.get("source_dataset")),
            "source_kind": clean(row.get("source_kind")),
            "begin_s": clean(row.get("begin_s")),
            "end_s": clean(row.get("end_s")),
            "event_group": clean(row.get("event_group")),
            "negative_bucket": clean(row.get("negative_bucket")),
            "true_label_ids": _label_text(true),
            "pred_label_ids": _label_text(pred),
            "threshold_set": threshold_name,
            "exact_match": int(true == pred),
            "any_cross_species_fp": int(any((label in pred and label not in true and true) for label in LABEL_ORDER)),
        }
        for label in LABEL_ORDER:
            record[f"score__{label}"] = float(scores[label][idx])
            record[f"threshold__{label}"] = float(thresholds[label])
            record[f"true__{label}"] = int(label in true)
            record[f"pred__{label}"] = int(label in pred)
        out.append(record)
    return out


def _metrics_table(prediction_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_label: List[Dict[str, Any]] = []
    tp_total = fp_total = fn_total = 0
    f1_values: List[float] = []
    for label in LABEL_ORDER:
        y_true = np.array([int(row.get(f"true__{label}") or 0) for row in prediction_rows], dtype=np.int64)
        y_pred = np.array([int(row.get(f"pred__{label}") or 0) for row in prediction_rows], dtype=np.int64)
        counts = _binary_counts(y_true, y_pred)
        fp_other = 0
        fp_background = 0
        fp_by_truth = Counter()
        for row in prediction_rows:
            if not int(row.get(f"pred__{label}") or 0) or int(row.get(f"true__{label}") or 0):
                continue
            true_labels = split_labels(row.get("true_label_ids"))
            if true_labels:
                fp_other += 1
                for true_label in true_labels:
                    fp_by_truth[true_label] += 1
            else:
                fp_background += 1
        row = {
            "label_id": label,
            "label_name": _species_name(label),
            "support": int(y_true.sum()),
            **counts,
            "fp_other_species": fp_other,
            "fp_background": fp_background,
            **{f"fp_when_true_{true_label}": int(fp_by_truth[true_label]) for true_label in LABEL_ORDER},
        }
        per_label.append(row)
        tp_total += int(counts["tp"])
        fp_total += int(counts["fp"])
        fn_total += int(counts["fn"])
        if int(y_true.sum()) > 0:
            f1_values.append(float(counts["f1"]))
    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    micro_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    hard_rows = [row for row in prediction_rows if not split_labels(row.get("true_label_ids"))]
    hard_fp = sum(1 for row in hard_rows if split_labels(row.get("pred_label_ids")))
    exact_match = sum(int(row.get("exact_match") or 0) for row in prediction_rows)
    summary = {
        "samples": len(prediction_rows),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "hard_fp": hard_fp,
        "hard_total": len(hard_rows),
        "hard_fp_rate": hard_fp / max(len(hard_rows), 1),
        "exact_match": exact_match,
        "exact_match_rate": exact_match / max(len(prediction_rows), 1),
    }
    return per_label, summary


def _truth_bucket(labels: Sequence[str]) -> str:
    return _label_text(labels) or "<background>"


def _confusion_rows(prediction_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str]] = Counter()
    sample_counts: Counter[Tuple[str, str]] = Counter()
    for row in prediction_rows:
        true = split_labels(row.get("true_label_ids"))
        pred = split_labels(row.get("pred_label_ids"))
        true_bucket = _truth_bucket(true)
        pred_bucket = _truth_bucket(pred)
        sample_counts[(true_bucket, pred_bucket)] += 1
        for label in pred:
            if label not in true:
                counts[(true_bucket, label)] += 1
    rows = [
        {
            "table": "false_positive_label_by_truth_bucket",
            "true_bucket": true_bucket,
            "predicted_label": pred_label,
            "count": count,
        }
        for (true_bucket, pred_label), count in sorted(counts.items())
    ]
    rows.extend(
        {
            "table": "sample_true_pred_bucket",
            "true_bucket": true_bucket,
            "predicted_label": pred_bucket,
            "count": count,
        }
        for (true_bucket, pred_bucket), count in sorted(sample_counts.items())
    )
    return rows


def _example_rows(prediction_rows: Sequence[Mapping[str, Any]], *, max_per_kind: int) -> Dict[str, List[Dict[str, Any]]]:
    examples: Dict[str, List[Dict[str, Any]]] = {}
    keep_fields = [
        "row_index",
        "item_id",
        "source_audio",
        "begin_s",
        "end_s",
        "event_group",
        "negative_bucket",
        "true_label_ids",
        "pred_label_ids",
    ]
    for label in LABEL_ORDER:
        score_key = f"score__{label}"
        threshold_key = f"threshold__{label}"
        cases = {
            "true_positive": [
                row for row in prediction_rows
                if int(row.get(f"true__{label}") or 0) and int(row.get(f"pred__{label}") or 0)
            ],
            "false_positive": [
                row for row in prediction_rows
                if not int(row.get(f"true__{label}") or 0) and int(row.get(f"pred__{label}") or 0)
            ],
            "false_negative": [
                row for row in prediction_rows
                if int(row.get(f"true__{label}") or 0) and not int(row.get(f"pred__{label}") or 0)
            ],
        }
        for kind, rows in cases.items():
            if kind == "false_negative":
                ordered = sorted(rows, key=lambda row: float(row.get(score_key) or 0.0))
            else:
                ordered = sorted(rows, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)
            selected: List[Dict[str, Any]] = []
            for row in ordered[:max_per_kind]:
                item = {field: row.get(field, "") for field in keep_fields}
                item.update(
                    {
                        "label_id": label,
                        "label_name": _species_name(label),
                        "case_type": kind,
                        "score": row.get(score_key),
                        "threshold": row.get(threshold_key),
                        "margin": float(row.get(score_key) or 0.0) - float(row.get(threshold_key) or 0.0),
                        "score_fin": row.get("score__species:Bp"),
                        "score_blue": row.get("score__species:Bm"),
                        "score_humpback": row.get("score__species:Mn"),
                    }
                )
                selected.append(item)
            examples[f"{label}:{kind}"] = selected
    cross = [
        row for row in prediction_rows
        if int(row.get("any_cross_species_fp") or 0)
    ]
    cross_ordered = sorted(
        cross,
        key=lambda row: max(
            [
                float(row.get(f"score__{label}") or 0.0) - float(row.get(f"threshold__{label}") or 0.0)
                for label in LABEL_ORDER
                if int(row.get(f"pred__{label}") or 0) and not int(row.get(f"true__{label}") or 0)
            ] or [-math.inf]
        ),
        reverse=True,
    )
    examples["cross_species_false_positive"] = [
        {
            field: row.get(field, "")
            for field in [
                "row_index",
                "item_id",
                "source_audio",
                "begin_s",
                "end_s",
                "event_group",
                "true_label_ids",
                "pred_label_ids",
                "score__species:Bp",
                "score__species:Bm",
                "score__species:Mn",
            ]
        }
        for row in cross_ordered[:max_per_kind]
    ]
    return examples


def _flatten_examples(examples: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, items in examples.items():
        for item in items:
            out = {"example_group": group}
            out.update(item)
            rows.append(out)
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def _write_report(
    *,
    output_dir: Path,
    split_summaries: Sequence[Mapping[str, Any]],
    original_summary: Mapping[str, Any],
    original_per_label: Sequence[Mapping[str, Any]],
    common_summary: Mapping[str, Any],
    common_per_label: Sequence[Mapping[str, Any]],
    original_thresholds: Mapping[str, float],
    common_thresholds: Mapping[str, Mapping[str, Any]],
    confusion_rows: Sequence[Mapping[str, Any]],
    examples: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    lines: List[str] = [
        "# E26 Common ONC Test Diagnostics",
        "",
        "This report reruns the three E26 single-species experts on the same ONC validation/test rows. It is intended to check cross-species false positives directly. The original E26 ensemble report merged filtered expert exports; rows not seen by a non-target expert were filled with a zero score, so that report should not be used as the only cross-species confusion check.",
        "",
        "## Common Split Support",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["split", "rows", "fin support", "blue support", "humpback support", "background"],
            [
                [
                    item["split"],
                    item["rows"],
                    item.get("support", {}).get("species:Bp", 0),
                    item.get("support", {}).get("species:Bm", 0),
                    item.get("support", {}).get("species:Mn", 0),
                    item.get("support", {}).get("<background>", 0),
                ]
                for item in split_summaries
            ],
        )
    )
    lines.extend(["", "## Ensemble Metrics On Common Test Rows", ""])
    lines.extend(
        _markdown_table(
            ["thresholds", "macro F1", "micro F1", "precision", "recall", "TP", "FP", "FN", "hard FP rate", "exact match"],
            [
                [
                    "original E26 per-expert validation thresholds",
                    _fmt(original_summary["macro_f1"]),
                    _fmt(original_summary["micro_f1"]),
                    _fmt(original_summary["precision"]),
                    _fmt(original_summary["recall"]),
                    original_summary["tp"],
                    original_summary["fp"],
                    original_summary["fn"],
                    f"{original_summary['hard_fp']}/{original_summary['hard_total']} ({_fmt(original_summary['hard_fp_rate'])})",
                    _fmt(original_summary["exact_match_rate"]),
                ],
                [
                    "recalibrated on common ONC validation rows",
                    _fmt(common_summary["macro_f1"]),
                    _fmt(common_summary["micro_f1"]),
                    _fmt(common_summary["precision"]),
                    _fmt(common_summary["recall"]),
                    common_summary["tp"],
                    common_summary["fp"],
                    common_summary["fn"],
                    f"{common_summary['hard_fp']}/{common_summary['hard_total']} ({_fmt(common_summary['hard_fp_rate'])})",
                    _fmt(common_summary["exact_match_rate"]),
                ],
            ],
        )
    )
    lines.extend(["", "## Per-Species Metrics", ""])
    for title, rows in [("Original thresholds", original_per_label), ("Common-validation thresholds", common_per_label)]:
        lines.extend([f"### {title}", ""])
        lines.extend(
            _markdown_table(
                ["species", "F1", "precision", "recall", "TP", "FP", "FN", "FP on other species", "FP on background"],
                [
                    [
                        row["label_name"],
                        _fmt(row["f1"]),
                        _fmt(row["precision"]),
                        _fmt(row["recall"]),
                        row["tp"],
                        row["fp"],
                        row["fn"],
                        row["fp_other_species"],
                        row["fp_background"],
                    ]
                    for row in rows
                ],
            )
        )
        lines.append("")
    lines.extend(["## Thresholds", ""])
    lines.extend(
        _markdown_table(
            ["species", "original threshold", "common-val threshold", "common-val F1", "common-val precision", "common-val recall"],
            [
                [
                    _species_name(label),
                    _fmt(original_thresholds[label], 2),
                    _fmt(common_thresholds[label]["threshold"], 2),
                    _fmt(common_thresholds[label]["f1"]),
                    _fmt(common_thresholds[label]["precision"]),
                    _fmt(common_thresholds[label]["recall"]),
                ]
                for label in LABEL_ORDER
            ],
        )
    )
    fp_rows = [row for row in confusion_rows if row.get("table") == "false_positive_label_by_truth_bucket"]
    lines.extend(["", "## Cross-Species False Positives", ""])
    lines.extend(
        _markdown_table(
            ["true label bucket", "false predicted label", "count"],
            [
                [row["true_bucket"], _species_name(str(row["predicted_label"])), row["count"]]
                for row in fp_rows
            ] or [["<none>", "", 0]],
        )
    )
    lines.extend(
        [
            "",
            "## Example Files",
            "",
            f"- all common-test predictions with original thresholds: `{output_dir / 'common_test_predictions_original_thresholds.csv'}`",
            f"- all common-test predictions with common-validation thresholds: `{output_dir / 'common_test_predictions_common_thresholds.csv'}`",
            f"- selected examples with true positives, false positives, false negatives, and cross-species false positives: `{output_dir / 'selected_examples_original_thresholds.csv'}`",
            f"- confusion/count tables: `{output_dir / 'confusion_counts_original_thresholds.csv'}`",
            "",
            "## Selected Cross-Species FP Examples",
            "",
        ]
    )
    cross_examples = examples.get("cross_species_false_positive", [])[:12]
    lines.extend(
        _markdown_table(
            ["row", "true", "pred", "audio", "begin", "end", "score fin", "score blue", "score hump"],
            [
                [
                    row.get("row_index", ""),
                    row.get("true_label_ids", ""),
                    row.get("pred_label_ids", ""),
                    Path(str(row.get("source_audio", ""))).name,
                    row.get("begin_s", ""),
                    row.get("end_s", ""),
                    _fmt(row.get("score__species:Bp")),
                    _fmt(row.get("score__species:Bm")),
                    _fmt(row.get("score__species:Mn")),
                ]
                for row in cross_examples
            ] or [["", "<none>", "", "", "", "", "", "", ""]],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-kind", default="ONC")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples-per-kind", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = read_tsv(args.pipeline_dir / "e26_training_plan.tsv")
    if not plan_rows:
        raise SystemExit(f"No E26 plan rows found in {args.pipeline_dir}")
    run_metas: Dict[str, Mapping[str, Any]] = {}
    for plan in plan_rows:
        if not clean(plan.get("run_dir")) or not clean(plan.get("experiment")):
            continue
        labels = split_labels(plan.get("eval_label_ids"))
        if len(labels) != 1 or labels[0] not in LABEL_ORDER:
            continue
        meta_path = Path(clean(plan["run_dir"])) / "run_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_metas[labels[0]] = meta
    missing = [label for label in LABEL_ORDER if label not in run_metas]
    if missing:
        raise SystemExit(f"Missing E26 run metadata for labels: {missing}")

    split_summaries: List[Mapping[str, Any]] = []
    common_paths: Dict[str, Path] = {}
    common_rows_by_split: Dict[str, List[Dict[str, str]]] = {}
    for split in ("val", "test"):
        rows, summary = _common_rows(plan_rows, split=split, source_kind=str(args.source_kind))
        split_summaries.append(summary)
        path = args.output_dir / f"common_{split}_manifest.csv"
        _write_manifest(path, rows)
        common_paths[split] = path
        common_rows_by_split[split] = rows

    device = _device(str(args.device))
    all_scores: Dict[str, Dict[str, np.ndarray]] = {"val": {}, "test": {}}
    for label in LABEL_ORDER:
        for split in ("val", "test"):
            metas, scores = _score_split(
                common_manifest=common_paths[split],
                split=split,
                run_meta=run_metas[label],
                label_id=label,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
            )
            if len(metas) != len(common_rows_by_split[split]):
                raise RuntimeError(f"{label} {split} metadata length mismatch: {len(metas)} vs {len(common_rows_by_split[split])}")
            all_scores[split][label] = scores

    original_thresholds = _load_original_thresholds(run_metas)
    common_thresholds = _choose_common_thresholds(common_rows_by_split["val"], all_scores["val"])
    common_threshold_values = {label: float(common_thresholds[label]["threshold"]) for label in LABEL_ORDER}

    original_predictions = _prediction_rows(
        common_rows_by_split["test"],
        all_scores["test"],
        original_thresholds,
        threshold_name="original_e26",
    )
    common_predictions = _prediction_rows(
        common_rows_by_split["test"],
        all_scores["test"],
        common_threshold_values,
        threshold_name="common_val",
    )
    original_per_label, original_summary = _metrics_table(original_predictions)
    common_per_label, common_summary = _metrics_table(common_predictions)
    confusion = _confusion_rows(original_predictions)
    examples = _example_rows(original_predictions, max_per_kind=int(args.max_examples_per_kind))

    write_csv(args.output_dir / "common_test_predictions_original_thresholds.csv", original_predictions)
    write_csv(args.output_dir / "common_test_predictions_common_thresholds.csv", common_predictions)
    write_csv(args.output_dir / "per_species_metrics_original_thresholds.csv", original_per_label)
    write_csv(args.output_dir / "per_species_metrics_common_thresholds.csv", common_per_label)
    write_csv(args.output_dir / "confusion_counts_original_thresholds.csv", confusion)
    write_csv(args.output_dir / "selected_examples_original_thresholds.csv", _flatten_examples(examples))
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(
            {
                "split_summaries": split_summaries,
                "original_thresholds": original_thresholds,
                "common_thresholds": common_thresholds,
                "original_summary": original_summary,
                "common_summary": common_summary,
                "run_metas": {label: dict(run_metas[label]) for label in LABEL_ORDER},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = _write_report(
        output_dir=args.output_dir,
        split_summaries=split_summaries,
        original_summary=original_summary,
        original_per_label=original_per_label,
        common_summary=common_summary,
        common_per_label=common_per_label,
        original_thresholds=original_thresholds,
        common_thresholds=common_thresholds,
        confusion_rows=confusion,
        examples=examples,
    )
    report_path = args.output_dir / "e26_common_onc_test_diagnostics.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "original_summary": original_summary, "common_summary": common_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
