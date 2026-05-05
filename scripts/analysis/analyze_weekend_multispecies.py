#!/usr/bin/env python3
"""Analyze weekend multi-source whale classifier experiments.

This script is intentionally filesystem-oriented so it can run on Nibi next to
the Slurm artifacts and write a small, git-friendly report under experiments/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import evaluation_bucket_from_row  # noqa: E402


PRIMARY_SPECIES = ("species:Bm", "species:Bp", "species:Mn", "species:Oo")

DEFAULT_RUNS = {
    "E01 ONC control": "runs/E01_onc_bal100_control_20260502T074301Z",
    "E04 ONC+BioDCASE species+call": "runs/E04_onc_biod_train50_species_call_20260502T074323Z",
    "E06 ONC+BioDCASE+DCLDE": "runs/E06_onc_biod_dclde_oo_repair_species_call_20260503T022802Z",
    "E08 ONC+DCLDE species-only": "runs/E08_onc_dclde_species_20260504T173524Z",
    "E09 ONC+BioDCASE+DCLDE species-only": "runs/E09_onc_biod_dclde_species_20260504T173543Z",
}

DEFAULT_MANIFESTS = {
    "E01 ONC control": "manifests/E01_onc_bal100_control/standardized_manifest.csv",
    "E04 ONC+BioDCASE species+call": "manifests/E04_onc_biod_train50_species_call/standardized_manifest.csv",
    "E06 ONC+BioDCASE+DCLDE": "manifests/E06_onc_biod_dclde_oo_repair_species_call/standardized_manifest.csv",
    "E08 ONC+DCLDE species-only": "manifests/E08_onc_dclde_species/standardized_manifest.csv",
    "E09 ONC+BioDCASE+DCLDE species-only": "manifests/E09_onc_biod_dclde_species/standardized_manifest.csv",
}


def split_pipe(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return [token.strip() for token in text.split("|") if token.strip()]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def source_group(source_dataset: str) -> str:
    text = (source_dataset or "").lower()
    if "final2025" in text or text.startswith("onc"):
        return "ONC"
    if text.startswith("dclde") or "killer_whales" in text:
        return "DCLDE"
    if text:
        return "BioDCASE"
    return "unknown"


def thresholds_from_summary(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("per_label_threshold_aggregate"), dict):
            return {k: float(v) for k, v in payload["per_label_threshold_aggregate"].get("thresholds", {}).items()}
        if isinstance(payload.get("thresholds"), dict):
            return {k: float(v) for k, v in payload["thresholds"].items()}
    raise ValueError(f"Could not find per-label thresholds in {path}")


def score(row: Mapping[str, str], label: str) -> float:
    try:
        return float(row.get(f"score__{label}", "") or 0.0)
    except ValueError:
        return 0.0


def label_set(row: Mapping[str, str], field: str = "target_label_ids") -> set[str]:
    return set(split_pipe(row.get(field)))


def row_bucket(row: Mapping[str, str]) -> str:
    return evaluation_bucket_from_row(dict(row), primary_species_label_ids=PRIMARY_SPECIES)


def any_primary_prediction(row: Mapping[str, str], thresholds: Mapping[str, float]) -> bool:
    return any(score(row, label) >= float(thresholds.get(label, 0.5)) for label in PRIMARY_SPECIES)


def compute_metrics(
    rows: Sequence[Mapping[str, str]],
    thresholds: Mapping[str, float],
    labels: Sequence[str] = PRIMARY_SPECIES,
) -> Dict[str, Any]:
    per_label: Dict[str, Dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = fp = fn = tn = 0
        threshold = float(thresholds.get(label, 0.5))
        for row in rows:
            actual = label in label_set(row)
            pred = score(row, label) >= threshold
            if pred and actual:
                tp += 1
            elif pred and not actual:
                fp += 1
            elif not pred and actual:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    macro_f1 = sum(item["f1"] for item in per_label.values()) / len(labels)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    no_primary_rows = [row for row in rows if not (label_set(row) & set(labels))]
    no_primary_fp = sum(any_primary_prediction(row, thresholds) for row in no_primary_rows)
    bucket_counts: Counter[str] = Counter(row_bucket(row) for row in rows)
    bucket_fp_counts: Counter[str] = Counter()
    bucket_row_counts: Counter[str] = Counter()
    for row in rows:
        bucket = row_bucket(row)
        bucket_row_counts[bucket] += 1
        if bucket != "primary_species_positive" and any_primary_prediction(row, thresholds):
            bucket_fp_counts[bucket] += 1
    reviewed_background_rows = [row for row in rows if row_bucket(row) == "reviewed_background"]
    reviewed_background_fp = sum(any_primary_prediction(row, thresholds) for row in reviewed_background_rows)
    return {
        "row_count": len(rows),
        "background_row_count": len(reviewed_background_rows),
        "background_any_primary_fp": reviewed_background_fp,
        "background_any_primary_fp_rate": reviewed_background_fp / len(reviewed_background_rows)
        if reviewed_background_rows
        else None,
        "no_primary_row_count": len(no_primary_rows),
        "no_primary_any_primary_fp": no_primary_fp,
        "no_primary_any_primary_fp_rate": no_primary_fp / len(no_primary_rows) if no_primary_rows else None,
        "evaluation_bucket_counts": dict(bucket_counts.most_common()),
        "evaluation_bucket_any_primary_fp_counts": dict(bucket_fp_counts.most_common()),
        "evaluation_bucket_any_primary_fp_rates": {
            bucket: bucket_fp_counts[bucket] / count if count else None for bucket, count in bucket_row_counts.items()
        },
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "per_label": per_label,
    }


def summarize_manifest(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    label_counts: Counter[str] = Counter()
    call_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    for row in rows:
        split = row.get("split") or "unsplit"
        group = source_group(row.get("source_dataset", ""))
        labels = split_pipe(row.get("label_ids") or row.get("target_label_ids"))
        split_counts[split] += 1
        source_counts[group] += 1
        source_split_counts[group][split] += 1
        label_counts.update(labels or ["<background>"])
        species = [label for label in labels if label.startswith("species:")]
        calls = [label for label in labels if label.startswith("call:")]
        primary_counts.update([label for label in species if label in PRIMARY_SPECIES] or ["<background>"])
        call_counts.update(calls or ["<no-call-label>"])
    return {
        "row_count": len(rows),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "source_split_counts": {key: dict(value) for key, value in source_split_counts.items()},
        "label_counts": dict(label_counts.most_common()),
        "primary_species_counts": dict(primary_counts.most_common()),
        "call_counts": dict(call_counts.most_common()),
    }


def max_primary(row: Mapping[str, str]) -> Tuple[str, float]:
    values = [(label, score(row, label)) for label in PRIMARY_SPECIES]
    return max(values, key=lambda item: item[1])


def false_positive_rows(
    rows: Sequence[Mapping[str, str]],
    thresholds: Mapping[str, float],
    *,
    source: str = "ONC",
    evaluation_bucket: Optional[str] = None,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if source_group(row.get("source_dataset", "")) != source:
            continue
        true = label_set(row)
        true_primary = true & set(PRIMARY_SPECIES)
        predictions = [
            label for label in PRIMARY_SPECIES if score(row, label) >= float(thresholds.get(label, 0.5))
        ]
        if true_primary or not predictions:
            continue
        bucket = row_bucket(row)
        if evaluation_bucket is not None and bucket != evaluation_bucket:
            continue
        top_label, top_score = max_primary(row)
        out.append(
            {
                "item_id": row.get("item_id", ""),
                "evaluation_bucket": bucket,
                "source_dataset": row.get("source_dataset", ""),
                "source_audio": row.get("source_audio", ""),
                "mat_path": row.get("mat_path", ""),
                "source_label_ids": row.get("source_label_ids", ""),
                "analysis_label_ids": row.get("analysis_label_ids", ""),
                "is_background": row.get("is_background", ""),
                "context_tags": row.get("context_tags", ""),
                "predicted_primary": "|".join(predictions),
                "top_primary": top_label,
                "top_primary_score": f"{top_score:.6f}",
                "target_label_ids": row.get("target_label_ids", ""),
            }
        )
    return sorted(out, key=lambda row: float(row["top_primary_score"]), reverse=True)[:limit]


def per_label_error_rows(
    rows: Sequence[Mapping[str, str]],
    thresholds: Mapping[str, float],
    label: str,
    *,
    source: str = "ONC",
    limit: int = 40,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    false_pos: List[Dict[str, Any]] = []
    false_neg: List[Dict[str, Any]] = []
    threshold = float(thresholds.get(label, 0.5))
    for row in rows:
        if source_group(row.get("source_dataset", "")) != source:
            continue
        actual = label in label_set(row)
        value = score(row, label)
        pred = value >= threshold
        payload = {
            "item_id": row.get("item_id", ""),
            "source_dataset": row.get("source_dataset", ""),
            "source_audio": row.get("source_audio", ""),
            "mat_path": row.get("mat_path", ""),
            "target_label_ids": row.get("target_label_ids", ""),
            "pred_label_ids": row.get("pred_label_ids", ""),
            "score": f"{value:.6f}",
            "threshold": f"{threshold:.6f}",
        }
        if pred and not actual:
            false_pos.append(payload)
        elif actual and not pred:
            false_neg.append(payload)
    return (
        sorted(false_pos, key=lambda row: float(row["score"]), reverse=True)[:limit],
        sorted(false_neg, key=lambda row: float(row["score"]))[:limit],
    )


def _manifest_lookup_rows(path: Path) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return lookup
    for row in read_csv_rows(path):
        for key in (row.get("mat_path"), row.get("item_id"), row.get("expected_mat_name")):
            text = str(key or "").strip()
            if text:
                lookup[text] = row
                lookup[Path(text).name] = row
                lookup[Path(text).stem] = row
    return lookup


def _enrich_prediction_rows(
    rows: Sequence[Dict[str, str]],
    *,
    manifest_csv: Optional[Path] = None,
) -> List[Dict[str, str]]:
    if manifest_csv is None:
        return [dict(row) for row in rows]
    lookup = _manifest_lookup_rows(manifest_csv)
    fields = (
        "source_label_ids",
        "canonical_label_ids",
        "analysis_label_ids",
        "is_background",
        "review_status",
        "context_tags",
        "begin_s",
        "end_s",
        "event_group",
        "label_ids",
    )
    enriched: List[Dict[str, str]] = []
    for row in rows:
        out = dict(row)
        keys = [
            out.get("mat_path", ""),
            Path(out.get("mat_path", "")).name,
            Path(out.get("mat_path", "")).stem,
            out.get("item_id", ""),
        ]
        match = next((lookup[key] for key in keys if key in lookup), None)
        if match:
            for field in fields:
                if not out.get(field):
                    out[field] = str(match.get(field, ""))
        out["evaluation_bucket"] = row_bucket(out)
        enriched.append(out)
    return enriched


def run_metrics(run_dir: Path, manifest_csv: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    train_dir = run_dir / "train"
    predictions = train_dir / "validation_predictions.csv"
    thresholds_path = train_dir / "threshold_sweep_summary.json"
    if not predictions.exists() or not thresholds_path.exists():
        return None
    rows = _enrich_prediction_rows(read_csv_rows(predictions), manifest_csv=manifest_csv)
    thresholds = thresholds_from_summary(thresholds_path)
    by_group: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[source_group(row.get("source_dataset", ""))].append(row)
    metrics = {
        "thresholds": thresholds,
        "all": compute_metrics(rows, thresholds),
        "by_source": {source: compute_metrics(source_rows, thresholds) for source, source_rows in sorted(by_group.items())},
        "prediction_rows": rows,
    }
    return metrics


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_model_metric_plot(metrics_by_run: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    plt = ensure_matplotlib()
    labels = list(PRIMARY_SPECIES)
    run_names = list(metrics_by_run)
    width = 0.8 / max(1, len(run_names))
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, run_name in enumerate(run_names):
        source_metrics = metrics_by_run[run_name]["by_source"].get("ONC", {})
        values = [source_metrics.get("per_label", {}).get(label, {}).get("f1", 0.0) for label in labels]
        offsets = [value + (idx - (len(run_names) - 1) / 2) * width for value in x]
        ax.bar(offsets, values, width=width, label=run_name)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("species:", "") for label in labels])
    ax.set_ylim(0, 1)
    ax.set_ylabel("ONC calibrated F1")
    ax.set_title("ONC primary-species F1 by model")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_background_score_plot(metrics_by_run: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = [idx / 20 for idx in range(21)]
    for run_name, metrics in metrics_by_run.items():
        rows = [
            row
            for row in metrics["prediction_rows"]
            if source_group(row.get("source_dataset", "")) == "ONC"
            and not (label_set(row) & set(PRIMARY_SPECIES))
        ]
        values = [max_primary(row)[1] for row in rows]
        ax.hist(values, bins=bins, alpha=0.45, label=f"{run_name} (n={len(values)})")
    ax.set_xlabel("Max primary-species score on ONC background rows")
    ax.set_ylabel("Row count")
    ax.set_title("External-data runs raise whale scores on ONC background")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_manifest_composition_plot(manifest_summaries: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    plt = ensure_matplotlib()
    run_names = list(manifest_summaries)
    sources = sorted({source for summary in manifest_summaries.values() for source in summary["source_counts"]})
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0] * len(run_names)
    x = list(range(len(run_names)))
    for source in sources:
        values = [manifest_summaries[name]["source_counts"].get(source, 0) for name in run_names]
        ax.bar(x, values, bottom=bottoms, label=source)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=20, ha="right")
    ax.set_ylabel("Rows in standardized manifest")
    ax.set_title("Training/validation manifest composition by source")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_fp_bar_plot(metrics_by_run: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    plt = ensure_matplotlib()
    labels = list(PRIMARY_SPECIES)
    run_names = list(metrics_by_run)
    width = 0.8 / max(1, len(run_names))
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, run_name in enumerate(run_names):
        source_metrics = metrics_by_run[run_name]["by_source"].get("ONC", {})
        values = [source_metrics.get("per_label", {}).get(label, {}).get("fp", 0) for label in labels]
        offsets = [value + (idx - (len(run_names) - 1) / 2) * width for value in x]
        ax.bar(offsets, values, width=width, label=run_name)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("species:", "") for label in labels])
    ax.set_ylabel("ONC false-positive count")
    ax.set_title("ONC primary-species false positives by model")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_onc_background_top_label_plot(metrics_by_run: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    plt = ensure_matplotlib()
    run_names = list(metrics_by_run)
    labels = list(PRIMARY_SPECIES)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0] * len(run_names)
    x = list(range(len(run_names)))
    for label in labels:
        values: List[int] = []
        for run_name in run_names:
            metrics = metrics_by_run[run_name]
            thresholds = metrics["thresholds"]
            count = 0
            for row in metrics["prediction_rows"]:
                if source_group(row.get("source_dataset", "")) != "ONC":
                    continue
                if label_set(row) & set(PRIMARY_SPECIES):
                    continue
                predictions = [
                    species_label
                    for species_label in labels
                    if score(row, species_label) >= float(thresholds.get(species_label, 0.5))
                ]
                if predictions and max_primary(row)[0] == label:
                    count += 1
            values.append(count)
        ax.bar(x, values, bottom=bottoms, label=label.replace("species:", ""))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=22, ha="right")
    ax.set_ylabel("ONC background rows crossing a primary threshold")
    ax.set_title("Which species scores dominate ONC background false positives")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_source_background_score_plot(metrics_by_run: Mapping[str, Dict[str, Any]], out_path: Path) -> None:
    plt = ensure_matplotlib()
    selected = {
        name: metrics
        for name, metrics in metrics_by_run.items()
        if "DCLDE" in name or "BioDCASE" in name or name.startswith("E01")
    }
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, max(4, len(selected) * 2.2)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    bins = [idx / 20 for idx in range(21)]
    for ax, (run_name, metrics) in zip(axes, selected.items()):
        for source in ("ONC", "BioDCASE", "DCLDE"):
            rows = [
                row
                for row in metrics["prediction_rows"]
                if source_group(row.get("source_dataset", "")) == source
                and not (label_set(row) & set(PRIMARY_SPECIES))
            ]
            if not rows:
                continue
            ax.hist([max_primary(row)[1] for row in rows], bins=bins, alpha=0.45, label=f"{source} bg (n={len(rows)})")
        ax.set_ylabel("rows")
        ax.set_title(run_name, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Max primary-species score on background/no-primary rows")
    fig.suptitle("Background score distributions by source", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def metric_table_rows(metrics_by_run: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name, metrics in metrics_by_run.items():
        for source, source_metrics in metrics["by_source"].items():
            row: Dict[str, Any] = {
                "run": run_name,
                "source": source,
                "row_count": source_metrics["row_count"],
                "reviewed_background_row_count": source_metrics["background_row_count"],
                "reviewed_background_any_primary_fp": source_metrics["background_any_primary_fp"],
                "reviewed_background_any_primary_fp_rate": format_float(source_metrics["background_any_primary_fp_rate"]),
                "no_primary_row_count": source_metrics.get("no_primary_row_count", ""),
                "no_primary_any_primary_fp": source_metrics.get("no_primary_any_primary_fp", ""),
                "no_primary_any_primary_fp_rate": format_float(source_metrics.get("no_primary_any_primary_fp_rate")),
                "macro_f1": format_float(source_metrics["macro_f1"]),
                "micro_f1": format_float(source_metrics["micro_f1"]),
                "micro_precision": format_float(source_metrics["micro_precision"]),
                "micro_recall": format_float(source_metrics["micro_recall"]),
            }
            for label in PRIMARY_SPECIES:
                label_key = label.replace("species:", "")
                label_metrics = source_metrics["per_label"][label]
                row[f"{label_key}_f1"] = format_float(label_metrics["f1"])
                row[f"{label_key}_precision"] = format_float(label_metrics["precision"])
                row[f"{label_key}_recall"] = format_float(label_metrics["recall"])
                row[f"{label_key}_fp"] = label_metrics["fp"]
                row[f"{label_key}_fn"] = label_metrics["fn"]
            rows.append(row)
    return rows


def score_quantile_rows(metrics_by_run: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    def quantiles(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"n": 0, "p50": "", "p75": "", "p90": "", "p95": "", "max": ""}
        sorted_values = sorted(values)
        def pick(q: float) -> float:
            idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
            return sorted_values[idx]
        return {
            "n": len(sorted_values),
            "p50": f"{pick(0.50):.6f}",
            "p75": f"{pick(0.75):.6f}",
            "p90": f"{pick(0.90):.6f}",
            "p95": f"{pick(0.95):.6f}",
            "max": f"{sorted_values[-1]:.6f}",
        }

    rows: List[Dict[str, Any]] = []
    for run_name, metrics in metrics_by_run.items():
        for source in sorted({source_group(row.get("source_dataset", "")) for row in metrics["prediction_rows"]}):
            source_rows = [row for row in metrics["prediction_rows"] if source_group(row.get("source_dataset", "")) == source]
            subsets = {
                "all_rows": source_rows,
                "background_rows": [row for row in source_rows if not (label_set(row) & set(PRIMARY_SPECIES))],
            }
            for subset_name, subset_rows in subsets.items():
                max_values = [max_primary(row)[1] for row in subset_rows]
                summary = quantiles(max_values)
                rows.append(
                    {
                        "run": run_name,
                        "source": source,
                        "subset": subset_name,
                        "score": "max_primary",
                        **summary,
                    }
                )
                for label in PRIMARY_SPECIES:
                    label_summary = quantiles([score(row, label) for row in subset_rows])
                    rows.append(
                        {
                            "run": run_name,
                            "source": source,
                            "subset": subset_name,
                            "score": label,
                            **label_summary,
                        }
                    )
    return rows


def onc_background_top_label_rows(metrics_by_run: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name, metrics in metrics_by_run.items():
        counts: Counter[str] = Counter()
        thresholds = metrics["thresholds"]
        total_background = 0
        any_fp = 0
        for row in metrics["prediction_rows"]:
            if source_group(row.get("source_dataset", "")) != "ONC":
                continue
            if label_set(row) & set(PRIMARY_SPECIES):
                continue
            total_background += 1
            predictions = [
                label for label in PRIMARY_SPECIES if score(row, label) >= float(thresholds.get(label, 0.5))
            ]
            if not predictions:
                counts["<below_threshold>"] += 1
                continue
            any_fp += 1
            counts[max_primary(row)[0]] += 1
        for top_label, count in counts.most_common():
            rows.append(
                {
                    "run": run_name,
                    "top_primary_label": top_label,
                    "count": count,
                    "background_rows": total_background,
                    "any_primary_fp": any_fp,
                    "share_of_background": format_float(count / total_background if total_background else None),
                }
            )
    return rows


def evaluation_bucket_rows(metrics_by_run: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name, metrics in metrics_by_run.items():
        for source, source_metrics in metrics["by_source"].items():
            counts = source_metrics.get("evaluation_bucket_counts", {})
            fp_counts = source_metrics.get("evaluation_bucket_any_primary_fp_counts", {})
            fp_rates = source_metrics.get("evaluation_bucket_any_primary_fp_rates", {})
            for bucket, count in counts.items():
                rows.append(
                    {
                        "run": run_name,
                        "source": source,
                        "evaluation_bucket": bucket,
                        "row_count": count,
                        "any_primary_fp": fp_counts.get(bucket, 0),
                        "any_primary_fp_rate": format_float(fp_rates.get(bucket)),
                    }
                )
    return rows


def make_contact_sheet(image_paths: Sequence[Path], out_path: Path, *, title: str, max_images: int = 12) -> bool:
    if not image_paths:
        return False
    plt = ensure_matplotlib()
    selected = list(image_paths)[:max_images]
    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.7))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]
    for ax, path in zip(axes_list, selected):
        try:
            image = plt.imread(path)
            ax.imshow(image)
            ax.set_title(path.stem[:34], fontsize=7)
        except Exception:
            ax.text(0.5, 0.5, path.name, ha="center", va="center")
        ax.axis("off")
    for ax in axes_list[len(selected) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    metrics_by_run: Mapping[str, Dict[str, Any]],
    manifest_summaries: Mapping[str, Dict[str, Any]],
    generated_figures: Sequence[Path],
) -> None:
    rows = []
    for run_name, metrics in metrics_by_run.items():
        onc = metrics["by_source"].get("ONC")
        if not onc:
            continue
        rows.append(
            {
                "run": run_name,
                "ONC macro F1": format_float(onc["macro_f1"]),
                "ONC micro F1": format_float(onc["micro_f1"]),
                "ONC reviewed bg FP": format_float(onc["background_any_primary_fp_rate"]),
                "ONC no-primary FP": format_float(onc.get("no_primary_any_primary_fp_rate")),
                "Bm F1": format_float(onc["per_label"]["species:Bm"]["f1"]),
                "Bp F1": format_float(onc["per_label"]["species:Bp"]["f1"]),
                "Mn F1": format_float(onc["per_label"]["species:Mn"]["f1"]),
                "Oo F1": format_float(onc["per_label"]["species:Oo"]["f1"]),
            }
        )

    report = [
        "# Weekend Multi-Species Dataset Analysis",
        "",
        "## Short Diagnosis",
        "",
        "The external datasets are not failing because they lack signal. They are failing because the model is learning source-specific decision boundaries that do not transfer cleanly back to ONC deployment audio. BioDCASE and DCLDE add real whale examples, but they also shift the model toward higher primary-species scores on ONC background. That shows up as much higher background false positives and weaker ONC Oo/Mn precision.",
        "",
        "The species-only E08/E09 retries show that call-type complexity was not the only issue. Removing call labels did not recover deployability: E08 lost macro F1 and still raised background false positives, while E09 roughly matched macro F1 only by accepting a much higher ONC background false-positive rate. E01 remains the best deployable baseline.",
        "",
        "## ONC-Gated Metrics",
        "",
        markdown_table(
            rows,
            [
                "run",
                "ONC macro F1",
                "ONC micro F1",
                "ONC reviewed bg FP",
                "ONC no-primary FP",
                "Bm F1",
                "Bp F1",
                "Mn F1",
                "Oo F1",
            ],
        ),
        "",
        "## What Looks Wrong",
        "",
        "- E06 added DCLDE killer-whale positives and hard negatives, but ONC Oo precision dropped instead of improving. That means the DCLDE examples are not teaching the model an ONC-compatible killer-whale boundary.",
        "- The biggest deployment problem is calibration on rows without primary species labels. Some of these are true reviewed background, but others are demoted OD or other known signal and should not be interpreted as silent background.",
        "- BioDCASE appears to add useful Bm/Bp signal and raises species recall, but it also makes ONC background look whale-like to the model. That is why its macro F1 can look acceptable while deployment risk increases.",
        "- DCLDE cap200 did not repair ONC Oo. The model learned extra Oo sensitivity, but the DCLDE Oo boundary does not transfer cleanly to ONC Oo versus ONC background.",
        "- Mn and Oo are the fragile labels. Their recall can look acceptable, but precision collapses because the model starts assigning these labels to ONC background or other-species rows.",
        "- Call labels probably made the initial external-data problem harder, but the deeper issue is source/domain calibration. Species-only training alone is not enough.",
        "",
        "## Dataset/Training Hypotheses",
        "",
        "- Source mismatch: BioDCASE and DCLDE have different hydrophones, annotation styles, event durations, frequency ranges, and background scenes than the ONC held-out target.",
        "- Background definition mismatch: DCLDE hard negatives are selected confounders, while ONC background includes local noise and ambiguous low-frequency events. A negative from one source is not automatically a good negative for another.",
        "- Label granularity mismatch: ONC OD was demoted correctly, but DCLDE adds explicit Oo. That helps ontology, yet it changes the class boundary unless ONC-like Oo/background examples anchor it.",
        "- Threshold transfer: thresholds optimized on the mixed validation set do not necessarily produce good ONC deployment thresholds.",
        "- Pos-weight and source imbalance likely encourage sensitivity over specificity, which worsens background false positives. The next ResNet ablation should only happen if it directly tests ONC-specific calibration or source balancing.",
        "- External validation rows are too easy for the external sources relative to ONC background. Mixed validation can pick thresholds that look good globally while failing the ONC deployment distribution.",
        "",
        "## Figures",
        "",
    ]
    for figure in generated_figures:
        report.append(f"- ![{figure.stem}]({figure.relative_to(out_dir).as_posix()})")
    report.extend(
        [
            "",
            "## Recommended Next Experiments",
            "",
            "1. Stop broad ResNet scaling for now. E08/E09 show that the issue is not solved by species-only training.",
            "2. Run ONC-calibrated post-hoc analysis first: per-source thresholds, source-normalized score calibration, and ONC-background hard-negative mining.",
            "3. If we run one more ResNet job, make it narrow: species-only ONC+DCLDE or ONC+BioDCASE+DCLDE with source-balanced batches and an ONC-background-heavy validation/calibration split. Do not reintroduce call types yet.",
            "4. Add explicit ONC-like hard negatives for Oo/Mn/background before scaling DCLDE.",
            "5. Prioritize the embedding branch: extract Perch/other foundation embeddings for ONC/BioDCASE/DCLDE caps, train linear/MLP probes, and compare source-separable clusters. If embeddings separate source more strongly than label, that confirms domain shift and suggests adaptation/calibration work before more ResNet training.",
            "",
            "## Manifest Composition",
            "",
        ]
    )
    for run_name, summary in manifest_summaries.items():
        report.append(f"### {run_name}")
        report.append("")
        report.append(f"- Rows: `{summary['row_count']}`")
        report.append(f"- Source counts: `{json.dumps(summary['source_counts'], sort_keys=True)}`")
        report.append(f"- Primary species counts: `{json.dumps(summary['primary_species_counts'], sort_keys=True)}`")
        report.append(f"- Top call counts: `{json.dumps(dict(list(summary['call_counts'].items())[:10]), sort_keys=True)}`")
        report.append("")
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekend-root", default="/scratch/merileo/whale-call-analysis/multispecies_weekend_20260502")
    parser.add_argument("--output-dir", default="experiments/weekend_20260502_analysis")
    args = parser.parse_args()

    weekend_root = Path(args.weekend_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    metrics_by_run: Dict[str, Dict[str, Any]] = {}
    manifest_summaries: Dict[str, Dict[str, Any]] = {}

    for run_name, rel_path in DEFAULT_RUNS.items():
        manifest_rel = DEFAULT_MANIFESTS.get(run_name)
        manifest_path = weekend_root / manifest_rel if manifest_rel else None
        metrics = run_metrics(weekend_root / rel_path, manifest_csv=manifest_path)
        if metrics is not None:
            metrics_by_run[run_name] = metrics

    for run_name, rel_path in DEFAULT_MANIFESTS.items():
        path = weekend_root / rel_path
        if path.exists():
            rows = read_csv_rows(path)
            manifest_summaries[run_name] = summarize_manifest(rows)

    if not metrics_by_run:
        raise SystemExit(f"No run metrics found under {weekend_root}")

    figures: List[Path] = []
    metric_plot = out_dir / "figures" / "onc_primary_f1_by_model.png"
    save_model_metric_plot(metrics_by_run, metric_plot)
    figures.append(metric_plot)
    bg_plot = out_dir / "figures" / "onc_background_max_primary_scores.png"
    save_background_score_plot(metrics_by_run, bg_plot)
    figures.append(bg_plot)
    fp_plot = out_dir / "figures" / "onc_primary_false_positives.png"
    save_fp_bar_plot(metrics_by_run, fp_plot)
    figures.append(fp_plot)
    fp_top_plot = out_dir / "figures" / "onc_background_false_positive_top_labels.png"
    save_onc_background_top_label_plot(metrics_by_run, fp_top_plot)
    figures.append(fp_top_plot)
    source_bg_plot = out_dir / "figures" / "source_background_score_distributions.png"
    save_source_background_score_plot(metrics_by_run, source_bg_plot)
    figures.append(source_bg_plot)
    if manifest_summaries:
        comp_plot = out_dir / "figures" / "manifest_source_composition.png"
        save_manifest_composition_plot(manifest_summaries, comp_plot)
        figures.append(comp_plot)

    write_csv_rows(out_dir / "tables" / "source_domain_metrics.csv", metric_table_rows(metrics_by_run))
    write_csv_rows(out_dir / "tables" / "evaluation_bucket_metrics.csv", evaluation_bucket_rows(metrics_by_run))
    write_csv_rows(out_dir / "tables" / "score_quantiles_by_source.csv", score_quantile_rows(metrics_by_run))
    write_csv_rows(
        out_dir / "tables" / "onc_background_top_primary_label_counts.csv",
        onc_background_top_label_rows(metrics_by_run),
    )

    for run_name, metrics in metrics_by_run.items():
        safe = run_name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
        write_csv_rows(
            out_dir / "tables" / f"{safe}_onc_background_false_positives.csv",
            false_positive_rows(metrics["prediction_rows"], metrics["thresholds"]),
        )
        write_csv_rows(
            out_dir / "tables" / f"{safe}_onc_reviewed_background_false_positives.csv",
            false_positive_rows(
                metrics["prediction_rows"],
                metrics["thresholds"],
                evaluation_bucket="reviewed_background",
            ),
        )
        write_csv_rows(
            out_dir / "tables" / f"{safe}_onc_demoted_nonprimary_signal_false_positives.csv",
            false_positive_rows(
                metrics["prediction_rows"],
                metrics["thresholds"],
                evaluation_bucket="demoted_nonprimary_signal",
            ),
        )
        for label in ("species:Mn", "species:Oo"):
            fp_rows, fn_rows = per_label_error_rows(metrics["prediction_rows"], metrics["thresholds"], label)
            label_safe = label.replace(":", "_")
            write_csv_rows(out_dir / "tables" / f"{safe}_{label_safe}_false_positives.csv", fp_rows)
            write_csv_rows(out_dir / "tables" / f"{safe}_{label_safe}_false_negatives.csv", fn_rows)

    contact_sheet_figures: List[Path] = []
    for run_name, rel_path in DEFAULT_RUNS.items():
        image_dir = weekend_root / rel_path / "train" / "example_images"
        if not image_dir.exists():
            continue
        images = sorted(image_dir.glob("*.png"))
        if not images:
            continue
        safe = run_name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
        out_path = out_dir / "figures" / f"{safe}_example_images_contact_sheet.png"
        if make_contact_sheet(images, out_path, title=f"{run_name} example images"):
            contact_sheet_figures.append(out_path)
    figures.extend(contact_sheet_figures)

    serializable_metrics = {
        run_name: {
            "all": {key: value for key, value in metrics["all"].items() if key != "per_label"},
            "all_per_label": metrics["all"]["per_label"],
            "by_source": metrics["by_source"],
        }
        for run_name, metrics in metrics_by_run.items()
    }
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(serializable_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(manifest_summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(out_dir, metrics_by_run, manifest_summaries, figures)
    print(json.dumps({"output_dir": str(out_dir), "runs": list(metrics_by_run), "figures": [str(p) for p in figures]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
