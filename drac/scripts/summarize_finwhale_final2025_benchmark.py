#!/usr/bin/env python3
"""Summarize the focused final-2025 benchmark into one CSV and Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _format_metric(value: Optional[float], ndigits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def _find_latest_child(root: Path, pattern: str) -> Optional[Path]:
    candidates = sorted(root.glob(pattern))
    return candidates[-1] if candidates else None


def _part2_eval_summary(eval_dir: Path) -> Dict[str, Any]:
    latest_run = _find_latest_child(eval_dir, "finwhale_part2_*")
    if latest_run is None:
        return {}

    best_row: Dict[str, Any] = {}
    best_score = float("-inf")
    for metrics_path in sorted(latest_run.glob("window_step_*/evaluation/metrics.json")):
        payload = _read_json(metrics_path)
        merged = payload.get("merged_region_metrics") or {}
        raw = payload.get("raw_window_metrics") or {}
        score = _to_float(merged.get("f1"))
        if score is None:
            score = float("-inf")
        if score > best_score:
            best_score = score
            best_row = {
                "run_dir": str(latest_run),
                "window_step": metrics_path.parents[1].name.replace("window_step_", ""),
                "metrics_path": str(metrics_path),
                "merged_region_precision": _to_float(merged.get("precision")),
                "merged_region_recall": _to_float(merged.get("recall")),
                "merged_region_f1": _to_float(merged.get("f1")),
                "merged_region_review_minutes": _to_float(merged.get("total_review_minutes")),
                "raw_window_precision": _to_float(raw.get("precision")),
                "raw_window_recall": _to_float(raw.get("recall")),
                "raw_window_f1": _to_float(raw.get("f1")),
                "raw_window_review_minutes": _to_float(raw.get("total_review_minutes")),
            }
    return best_row


def _historical_eval_summary(eval_dir: Path) -> Dict[str, Any]:
    latest_run = _find_latest_child(eval_dir, "finwhale_part2_*")
    if latest_run is None:
        return {}
    metrics_candidates = sorted(latest_run.glob("historical_baseline/**/metrics.json"))
    if not metrics_candidates:
        return {}
    payload = _read_json(metrics_candidates[-1])
    return {
        "metrics_path": str(metrics_candidates[-1]),
        "acc": _to_float(payload.get("acc")),
        "precision": _to_float(payload.get("precision")),
        "recall": _to_float(payload.get("recall")),
        "f1": _to_float(payload.get("f1")),
        "auc": _to_float(payload.get("auc")),
    }


def _training_summary(train_dir: Path) -> Dict[str, Any]:
    summary_path = train_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    best = payload.get("best") or {}
    val_metrics = best.get("val_metrics") or {}
    dataset_counts = payload.get("dataset_counts") or {}
    return {
        "summary_path": str(summary_path),
        "checkpoint_path": str(best.get("checkpoint_path") or ""),
        "best_epoch": best.get("epoch"),
        "best_main_metric": _to_float(best.get("value")),
        "val_f1": _to_float(val_metrics.get("f1")),
        "val_auc": _to_float(val_metrics.get("auc")),
        "val_precision": _to_float(val_metrics.get("precision")),
        "val_recall": _to_float(val_metrics.get("recall")),
        "joint_train_total": (dataset_counts.get("train") or {}).get("total"),
        "joint_train_pos": (dataset_counts.get("train") or {}).get("pos"),
        "joint_train_neg": (dataset_counts.get("train") or {}).get("neg"),
        "joint_val_total": (dataset_counts.get("val") or {}).get("total"),
        "joint_val_pos": (dataset_counts.get("val") or {}).get("pos"),
        "joint_val_neg": (dataset_counts.get("val") or {}).get("neg"),
    }


def _baseline_rows(benchmark_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for phase, split_name, path_name in [
        ("baseline_val", "val", "baseline_val"),
        ("baseline_test", "test", "baseline_test"),
    ]:
        eval_dir = benchmark_dir / path_name
        if not eval_dir.exists():
            continue
        part2 = _part2_eval_summary(eval_dir)
        hist = _historical_eval_summary(eval_dir)
        rows.append(
            {
                "recipe": "baseline",
                "seed": "base",
                "phase": phase,
                "historical_split": split_name,
                "train_best_main_metric": None,
                "train_val_f1": None,
                "train_val_auc": None,
                "joint_train_total": None,
                "joint_val_total": None,
                f"historical_{split_name}_f1": hist.get("f1"),
                f"historical_{split_name}_auc": hist.get("auc"),
                f"part2_{split_name}_window_step": part2.get("window_step"),
                f"part2_{split_name}_merged_region_f1": part2.get("merged_region_f1"),
                f"part2_{split_name}_merged_region_recall": part2.get("merged_region_recall"),
                f"part2_{split_name}_raw_window_f1": part2.get("raw_window_f1"),
                f"part2_{split_name}_raw_window_recall": part2.get("raw_window_recall"),
                f"part2_{split_name}_review_minutes": part2.get("merged_region_review_minutes"),
            }
        )
    return rows


def _run_rows(benchmark_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runs_dir = benchmark_dir / "runs"
    if not runs_dir.exists():
        return rows
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        recipe = run_dir.name
        seed = ""
        if "_seed" in recipe:
            recipe, seed = recipe.rsplit("_seed", 1)
        train = _training_summary(run_dir / "train")
        val_part2 = _part2_eval_summary(run_dir / "val_eval")
        val_hist = _historical_eval_summary(run_dir / "val_eval")
        test_part2 = _part2_eval_summary(run_dir / "test_eval")
        test_hist = _historical_eval_summary(run_dir / "test_eval")
        rows.append(
            {
                "recipe": recipe,
                "seed": seed,
                "phase": "run",
                "historical_split": "val",
                "train_best_main_metric": train.get("best_main_metric"),
                "train_best_epoch": train.get("best_epoch"),
                "train_val_f1": train.get("val_f1"),
                "train_val_auc": train.get("val_auc"),
                "train_val_precision": train.get("val_precision"),
                "train_val_recall": train.get("val_recall"),
                "joint_train_total": train.get("joint_train_total"),
                "joint_train_pos": train.get("joint_train_pos"),
                "joint_train_neg": train.get("joint_train_neg"),
                "joint_val_total": train.get("joint_val_total"),
                "joint_val_pos": train.get("joint_val_pos"),
                "joint_val_neg": train.get("joint_val_neg"),
                "checkpoint_path": train.get("checkpoint_path"),
                "historical_val_f1": val_hist.get("f1"),
                "historical_val_auc": val_hist.get("auc"),
                "part2_val_window_step": val_part2.get("window_step"),
                "part2_val_merged_region_f1": val_part2.get("merged_region_f1"),
                "part2_val_merged_region_recall": val_part2.get("merged_region_recall"),
                "part2_val_raw_window_f1": val_part2.get("raw_window_f1"),
                "part2_val_raw_window_recall": val_part2.get("raw_window_recall"),
                "part2_val_review_minutes": val_part2.get("merged_region_review_minutes"),
                "historical_test_f1": test_hist.get("f1"),
                "historical_test_auc": test_hist.get("auc"),
                "part2_test_window_step": test_part2.get("window_step"),
                "part2_test_merged_region_f1": test_part2.get("merged_region_f1"),
                "part2_test_merged_region_recall": test_part2.get("merged_region_recall"),
                "part2_test_raw_window_f1": test_part2.get("raw_window_f1"),
                "part2_test_raw_window_recall": test_part2.get("raw_window_recall"),
                "part2_test_review_minutes": test_part2.get("merged_region_review_minutes"),
            }
        )
    return rows


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_rows = [row for row in rows if row.get("phase") == "run"]
    ranked = sorted(
        run_rows,
        key=lambda row: _to_float(row.get("part2_val_merged_region_f1")) if _to_float(row.get("part2_val_merged_region_f1")) is not None else float("-inf"),
        reverse=True,
    )

    lines: List[str] = []
    lines.append("# Final 2025 ResNet Benchmark Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Total benchmark rows: {len(rows)}")
    lines.append(f"- Benchmark runs: {len(run_rows)}")
    lines.append("")
    lines.append("## Ranked Validation Runs")
    lines.append("")
    lines.append("| rank | recipe | seed | 2025 merged F1 | 2025 merged recall | 2025 raw F1 | hist val F1 | train val F1 | review min | ws |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("recipe", "")),
                    str(row.get("seed", "")),
                    _format_metric(_to_float(row.get("part2_val_merged_region_f1"))),
                    _format_metric(_to_float(row.get("part2_val_merged_region_recall"))),
                    _format_metric(_to_float(row.get("part2_val_raw_window_f1"))),
                    _format_metric(_to_float(row.get("historical_val_f1"))),
                    _format_metric(_to_float(row.get("train_val_f1"))),
                    _format_metric(_to_float(row.get("part2_val_review_minutes")), ndigits=1),
                    str(row.get("part2_val_window_step", "")),
                ]
            )
            + " |"
        )
    lines.append("")

    baseline_rows = [row for row in rows if row.get("recipe") == "baseline"]
    if baseline_rows:
        lines.append("## Baselines")
        lines.append("")
        for row in baseline_rows:
            split_name = str(row.get("historical_split", ""))
            lines.append(
                f"- `{split_name}`: 2025 merged F1 "
                f"`{_format_metric(_to_float(row.get(f'part2_{split_name}_merged_region_f1')))}`"
                f" and historical {split_name} F1 "
                f"`{_format_metric(_to_float(row.get(f'historical_{split_name}_f1')))}`"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the final-2025 ResNet benchmark")
    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-md", type=str, default=None)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).resolve()
    rows = _baseline_rows(benchmark_dir) + _run_rows(benchmark_dir)
    out_csv = Path(args.out_csv).resolve() if args.out_csv else benchmark_dir / "results" / "benchmark_summary.csv"
    out_md = Path(args.out_md).resolve() if args.out_md else benchmark_dir / "results" / "benchmark_summary.md"
    _write_csv(rows, out_csv)
    _write_markdown(rows, out_md)
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
