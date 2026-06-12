#!/usr/bin/env python3
"""Evaluate a binary whale-call gate on common validation/test rows.

This report is intentionally narrower than the E122 two-stage report: it asks
only whether stage 1 says "any target whale call" vs background. It keeps
species labels in the output so gate misses can be inspected by species before a
downstream classifier or expert-in-the-loop app handles species identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.analysis import e119_pairwise_refinement_report as e119


DEFAULT_CLASS_IDS = ("background", "species:Bp", "species:Bm", "species:Mn")
DEFAULT_POSITIVE_LABELS = ("species:Bp", "species:Bm", "species:Mn")
DEFAULT_SCORE_LABEL = "task:whale_call"


def parse_labels(value: str) -> List[str]:
    labels = [part.strip() for part in str(value or "").replace(",", "|").split("|") if part.strip()]
    if not labels:
        raise ValueError("label list cannot be empty")
    return labels


def parse_thresholds(value: str) -> List[float]:
    text = str(value or "").strip()
    if text == "centile":
        return [round(i / 100, 2) for i in range(101)]
    thresholds = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("threshold grid cannot be empty")
    return sorted(dict.fromkeys(thresholds))


def gate_score(row: Mapping[str, Any], *, score_field: Optional[str], score_label: str) -> Optional[float]:
    if score_field:
        value = e119.as_float(row.get(score_field))
        if value is not None:
            return value
    for field in ("stage1_prob_call", "gate_score", "prob_call", "score_call", "whale_call_score", "score"):
        value = e119.as_float(row.get(field))
        if value is not None:
            return value
    value = e119.probability(row, score_label)
    if value is not None:
        return value
    value = e119.probability(row, score_label.replace(":", "_"))
    if value is not None:
        return value
    return None


def load_gate_rows(
    path: Path,
    *,
    class_ids: Sequence[str],
    positive_labels: Sequence[str],
    score_field: Optional[str],
    score_label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    positives = set(positive_labels)
    for row in e119.load_predictions(path, class_ids):
        score = gate_score(row, score_field=score_field, score_label=score_label)
        out = dict(row)
        out["gate_score"] = "" if score is None else float(score)
        out["_binary_true"] = "whale" if e119.clean(out.get("_true")) in positives else "background"
        rows.append(out)
    return rows


def binary_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    tp = fp = tn = fn = missing_score = 0
    for row in rows:
        score = e119.as_float(row.get("gate_score"))
        if score is None:
            missing_score += 1
            score = 0.0
        true_positive = e119.clean(row.get("_binary_true")) == "whale"
        pred_positive = float(score) >= float(threshold)
        if true_positive and pred_positive:
            tp += 1
        elif true_positive:
            fn += 1
        elif pred_positive:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "rows": len(rows),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "missing_score": missing_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def tune_threshold(rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> Tuple[float, List[Dict[str, Any]]]:
    sweep = [binary_metrics(rows, threshold) for threshold in thresholds]
    best = max(
        sweep,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            -int(row["fp"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), sweep


def true_bucket(row: Mapping[str, Any]) -> str:
    true = e119.clean(row.get("_true"))
    return true if true and true != "background" else "background"


def species_breakdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    positive_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    buckets = ["background", *positive_labels]
    out: List[Dict[str, Any]] = []
    for bucket in buckets:
        subset = [row for row in rows if true_bucket(row) == bucket]
        detected = 0
        missed = 0
        for row in subset:
            score = e119.as_float(row.get("gate_score")) or 0.0
            if score >= threshold:
                detected += 1
            else:
                missed += 1
        out.append(
            {
                "true_bucket": bucket,
                "support": len(subset),
                "detected": detected,
                "missed": missed,
                "detection_rate": detected / len(subset) if subset else 0.0,
            }
        )
    return out


def example_rows(rows: Sequence[Mapping[str, Any]], *, threshold: float, limit_per_bucket: int = 50) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {
        "true_positive": [],
        "false_positive_background": [],
        "false_negative_whale": [],
        "true_negative": [],
    }
    for row in rows:
        score = e119.as_float(row.get("gate_score")) or 0.0
        pred_positive = score >= threshold
        true_positive = e119.clean(row.get("_binary_true")) == "whale"
        if true_positive and pred_positive:
            buckets["true_positive"].append(row)
        elif true_positive:
            buckets["false_negative_whale"].append(row)
        elif pred_positive:
            buckets["false_positive_background"].append(row)
        else:
            buckets["true_negative"].append(row)

    examples: List[Dict[str, Any]] = []
    for bucket, bucket_rows in buckets.items():
        reverse = bucket != "false_negative_whale"
        ordered = sorted(bucket_rows, key=lambda row: float(e119.as_float(row.get("gate_score")) or 0.0), reverse=reverse)
        for row in ordered[:limit_per_bucket]:
            examples.append(
                {
                    "bucket": bucket,
                    "item_id": e119.clean(row.get("item_id")) or e119.clean(row.get("_key")),
                    "true_class": e119.clean(row.get("_true")),
                    "binary_true": e119.clean(row.get("_binary_true")),
                    "gate_score": row.get("gate_score", ""),
                    "threshold": threshold,
                    "clip": e119.first_present(row, ("clip", "filename", "source_audio", "source_soundfile")),
                    "begin_s": e119.first_present(row, ("begin_s", "begin_time_s", "window_start_s")),
                    "end_s": e119.first_present(row, ("end_s", "end_time_s")),
                    "mat_path": e119.first_present(row, ("mat_path", "low_mat_path")),
                }
            )
    return examples


def metric_row(name: str, split: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {"name": name, "split": split, **metrics}


def markdown_report(
    *,
    name: str,
    output_dir: Path,
    score_label: str,
    threshold: float,
    metrics: Sequence[Mapping[str, Any]],
    breakdown: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# E126 Binary Whale-Call Gate Report: {name}",
        "",
        f"Validation-tuned threshold: `{threshold:.2f}`.",
        f"Score label/field fallback: `{score_label}`.",
        "",
        "## Overall Metrics",
        "",
        "| split | rows | threshold | F1 | precision | recall | accuracy | TP | FP | TN | FN | missing scores |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {split} | {rows} | {threshold:.2f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {accuracy:.4f} | {tp} | {fp} | {tn} | {fn} | {missing_score} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Test Breakdown By True Class",
            "",
            "| true bucket | support | detected | missed | detection rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in breakdown:
        lines.append(
            "| {true_bucket} | {support} | {detected} | {missed} | {detection_rate:.4f} |".format(**row)
        )
    bucket_counts: Dict[str, int] = {}
    for row in examples:
        bucket = e119.clean(row.get("bucket"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    lines.extend(["", "## Example Buckets", "", "| bucket | count |", "| --- | ---: |"])
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"| {bucket} | {count} |")
    lines.extend(
        [
            "",
            f"Metrics CSV: `{output_dir / 'e126_binary_gate_metrics.csv'}`",
            f"Threshold sweep CSV: `{output_dir / 'e126_binary_gate_threshold_sweep.csv'}`",
            f"Breakdown CSV: `{output_dir / 'e126_binary_gate_breakdown.csv'}`",
            f"Examples CSV: `{output_dir / 'e126_binary_gate_examples.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report(
    *,
    name: str,
    val_predictions: Path,
    test_predictions: Path,
    output_dir: Path,
    class_ids: Sequence[str],
    positive_labels: Sequence[str],
    score_label: str,
    score_field: Optional[str],
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    val_rows = load_gate_rows(
        val_predictions,
        class_ids=class_ids,
        positive_labels=positive_labels,
        score_field=score_field,
        score_label=score_label,
    )
    test_rows = load_gate_rows(
        test_predictions,
        class_ids=class_ids,
        positive_labels=positive_labels,
        score_field=score_field,
        score_label=score_label,
    )
    threshold, sweep = tune_threshold(val_rows, thresholds)
    metrics = [
        metric_row(name, "val", binary_metrics(val_rows, threshold)),
        metric_row(name, "test", binary_metrics(test_rows, threshold)),
    ]
    breakdown = species_breakdown(test_rows, threshold=threshold, positive_labels=positive_labels)
    examples = example_rows(test_rows, threshold=threshold)

    e119.write_csv(output_dir / "e126_binary_gate_metrics.csv", metrics)
    e119.write_csv(output_dir / "e126_binary_gate_threshold_sweep.csv", sweep)
    e119.write_csv(output_dir / "e126_binary_gate_breakdown.csv", breakdown)
    e119.write_csv(output_dir / "e126_binary_gate_examples.csv", examples)
    report_path = output_dir / "e126_binary_gate_report.md"
    report_path.write_text(
        markdown_report(
            name=name,
            output_dir=output_dir,
            score_label=score_field or score_label,
            threshold=threshold,
            metrics=metrics,
            breakdown=breakdown,
            examples=examples,
        ),
        encoding="utf-8",
    )
    summary = {
        "name": name,
        "threshold": threshold,
        "class_ids": list(class_ids),
        "positive_labels": list(positive_labels),
        "score_label": score_label,
        "score_field": score_field or "",
        "inputs": {"val_predictions": str(val_predictions), "test_predictions": str(test_predictions)},
        "metrics": metrics,
        "outputs": {
            "report": str(report_path),
            "metrics": str(output_dir / "e126_binary_gate_metrics.csv"),
            "threshold_sweep": str(output_dir / "e126_binary_gate_threshold_sweep.csv"),
            "breakdown": str(output_dir / "e126_binary_gate_breakdown.csv"),
            "examples": str(output_dir / "e126_binary_gate_examples.csv"),
        },
    }
    (output_dir / "e126_binary_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--val-predictions", required=True, type=Path)
    parser.add_argument("--test-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-ids", default=",".join(DEFAULT_CLASS_IDS))
    parser.add_argument("--positive-labels", default=",".join(DEFAULT_POSITIVE_LABELS))
    parser.add_argument("--score-label", default=DEFAULT_SCORE_LABEL)
    parser.add_argument("--score-field", default=None)
    parser.add_argument("--thresholds", default="centile")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_report(
        name=args.name,
        val_predictions=args.val_predictions,
        test_predictions=args.test_predictions,
        output_dir=args.output_dir,
        class_ids=parse_labels(args.class_ids),
        positive_labels=parse_labels(args.positive_labels),
        score_label=args.score_label,
        score_field=args.score_field,
        thresholds=parse_thresholds(args.thresholds),
    )
    print(json.dumps({"report": summary["outputs"]["report"], "summary": summary["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
