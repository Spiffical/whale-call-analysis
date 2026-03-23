#!/usr/bin/env python3
"""
Summarize a DRAC FinWhale training sweep into one CSV + Markdown report.

Expects outputs from:
  drac/scripts/launch_finwhale_training_sweep.sh
and per-run artifacts:
  run_summary.json, metrics_history.csv
written by scripts/train/train_cnn.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _find_run_summary(run_exp_dir: Path) -> Optional[Path]:
    candidates = sorted(run_exp_dir.glob("**/run_summary.json"))
    if not candidates:
        return None
    return candidates[-1]


def _best_val_loss_from_history(summary_dir: Path, best_epoch: Optional[int]) -> Optional[float]:
    if best_epoch is None:
        return None
    history_path = summary_dir / "metrics_history.csv"
    if not history_path.exists():
        return None
    try:
        with open(history_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("epoch", -1)) == int(best_epoch):
                    return _to_float(row.get("val_loss"))
    except Exception:
        return None
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_submitted_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def _format_metric(v: Optional[float], ndigits: int = 4) -> str:
    if v is None:
        return ""
    return f"{v:.{ndigits}f}"


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "run_slug",
        "dataset_tag",
        "dataset_source",
        "status",
        "exp_dir",
        "model",
        "split_strategy",
        "min_gap_seconds",
        "center_bias_sigma_frac",
        "balance",
        "lr",
        "seed",
        "main_metric",
        "best_main_metric",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "best_val_precision",
        "best_val_recall",
        "best_val_f1",
        "best_val_auc",
        "test_loss",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
        "train_total",
        "train_pos",
        "train_neg",
        "val_total",
        "val_pos",
        "val_neg",
        "test_total",
        "test_pos",
        "test_neg",
        "checkpoint_path",
        "wandb_run_id",
        "wandb_run_url",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_markdown(rows: List[Dict[str, Any]], out_path: Path, top_n: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [r for r in rows if r.get("status") == "completed"]
    pending = [r for r in rows if r.get("status") != "completed"]

    metric_key = "best_main_metric"
    completed_sorted = sorted(
        completed,
        key=lambda r: _to_float(r.get(metric_key)) if _to_float(r.get(metric_key)) is not None else float("-inf"),
        reverse=True,
    )

    lines: List[str] = []
    lines.append("# FinWhale Sweep Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Total runs in manifest: {len(rows)}")
    lines.append(f"- Completed: {len(completed)}")
    lines.append(f"- Pending/failed/missing: {len(pending)}")
    lines.append("")

    dataset_tags = sorted({(r.get("dataset_tag") or "dataset") for r in rows})
    lines.append(f"- Datasets: {', '.join(dataset_tags)}")
    lines.append("")

    lines.append("## Top Runs")
    lines.append("")
    lines.append("| rank | dataset | model | run_slug | best(main) | best_epoch | val_f1 | val_auc | test_f1 | test_auc | balance | cbs | gap | seed | wandb |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|")

    for idx, row in enumerate(completed_sorted[:top_n], start=1):
        wandb_url = row.get("wandb_run_url") or ""
        wandb_cell = f"[link]({wandb_url})" if wandb_url else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("dataset_tag", "")),
                    str(row.get("model", "")),
                    str(row.get("run_slug", "")),
                    _format_metric(_to_float(row.get("best_main_metric"))),
                    str(row.get("best_epoch", "")),
                    _format_metric(_to_float(row.get("best_val_f1"))),
                    _format_metric(_to_float(row.get("best_val_auc"))),
                    _format_metric(_to_float(row.get("test_f1"))),
                    _format_metric(_to_float(row.get("test_auc"))),
                    str(row.get("balance", "")),
                    str(row.get("center_bias_sigma_frac", "")),
                    str(row.get("min_gap_seconds", "")),
                    str(row.get("seed", "")),
                    wandb_cell,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Best Per Dataset")
    lines.append("")
    lines.append("| dataset | model | run_slug | best(main) | val_f1 | val_auc | test_f1 | test_auc | balance | cbs | gap | seed |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in completed:
        dtag = str(row.get("dataset_tag") or "dataset")
        by_dataset.setdefault(dtag, []).append(row)
    for dtag in sorted(by_dataset):
        ranked = sorted(
            by_dataset[dtag],
            key=lambda r: _to_float(r.get(metric_key)) if _to_float(r.get(metric_key)) is not None else float("-inf"),
            reverse=True,
        )
        best = ranked[0] if ranked else None
        if best is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    dtag,
                    str(best.get("model", "")),
                    str(best.get("run_slug", "")),
                    _format_metric(_to_float(best.get("best_main_metric"))),
                    _format_metric(_to_float(best.get("best_val_f1"))),
                    _format_metric(_to_float(best.get("best_val_auc"))),
                    _format_metric(_to_float(best.get("test_f1"))),
                    _format_metric(_to_float(best.get("test_auc"))),
                    str(best.get("balance", "")),
                    str(best.get("center_bias_sigma_frac", "")),
                    str(best.get("min_gap_seconds", "")),
                    str(best.get("seed", "")),
                ]
            )
            + " |"
        )

    lines.append("")
    if pending:
        lines.append("## Pending Or Missing Runs")
        lines.append("")
        for row in pending:
            lines.append(
                f"- {row.get('run_slug', '')} "
                f"(dataset={row.get('dataset_tag', '')}): status={row.get('status', '')}, job_id={row.get('job_id', '')}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FinWhale DRAC sweep outputs")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Sweep directory created by launch script")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N rows in markdown table")
    parser.add_argument("--out-csv", type=str, default=None, help="Output CSV path (default: <sweep-dir>/results/sweep_results.csv)")
    parser.add_argument("--out-md", type=str, default=None, help="Output markdown path (default: <sweep-dir>/results/sweep_results.md)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    submitted_tsv = sweep_dir / "submitted_jobs.tsv"
    if not submitted_tsv.exists():
        raise SystemExit(f"Missing submitted_jobs.tsv: {submitted_tsv}")

    out_csv = Path(args.out_csv) if args.out_csv else sweep_dir / "results" / "sweep_results.csv"
    out_md = Path(args.out_md) if args.out_md else sweep_dir / "results" / "sweep_results.md"

    submitted_rows = _load_submitted_rows(submitted_tsv)
    rows: List[Dict[str, Any]] = []

    for row in submitted_rows:
        exp_dir = Path(row["exp_dir"])
        run_slug = row.get("run_slug", "")
        status = "missing"

        summary_path = _find_run_summary(exp_dir)
        out: Dict[str, Any] = {
            "job_id": row.get("job_id", ""),
            "run_slug": run_slug,
            "dataset_tag": row.get("dataset_tag", ""),
            "dataset_source": row.get("dataset_source", ""),
            "status": status,
            "exp_dir": str(exp_dir),
            "model": row.get("model", ""),
            "split_strategy": "",
            "min_gap_seconds": row.get("min_gap_seconds", ""),
            "center_bias_sigma_frac": row.get("center_bias_sigma_frac", ""),
            "balance": row.get("balance", ""),
            "lr": row.get("lr", ""),
            "seed": row.get("seed", ""),
            "main_metric": "",
            "best_main_metric": "",
            "best_epoch": "",
            "best_val_loss": "",
            "best_val_acc": "",
            "best_val_precision": "",
            "best_val_recall": "",
            "best_val_f1": "",
            "best_val_auc": "",
            "test_loss": "",
            "test_acc": "",
            "test_precision": "",
            "test_recall": "",
            "test_f1": "",
            "test_auc": "",
            "train_total": "",
            "train_pos": "",
            "train_neg": "",
            "val_total": "",
            "val_pos": "",
            "val_neg": "",
            "test_total": "",
            "test_pos": "",
            "test_neg": "",
            "checkpoint_path": "",
            "wandb_run_id": "",
            "wandb_run_url": "",
        }

        if summary_path is None:
            out["status"] = "pending_or_failed"
            rows.append(out)
            continue

        try:
            summary = _read_json(summary_path)
        except Exception:
            out["status"] = "summary_parse_error"
            rows.append(out)
            continue

        out["status"] = "completed"

        args_dict = summary.get("args", {})
        best = summary.get("best", {})
        valm = best.get("val_metrics") or {}
        test = summary.get("final_test", {})
        testm = test.get("metrics") or {}
        counts = summary.get("dataset_counts") or {}
        wandb = summary.get("wandb") or {}

        best_epoch = best.get("epoch")

        out.update(
            {
                "model": args_dict.get("model", out["model"]),
                "split_strategy": args_dict.get("split_strategy", ""),
                "main_metric": best.get("main_metric", args_dict.get("main_metric", "")),
                "best_main_metric": best.get("value", ""),
                "best_epoch": best_epoch if best_epoch is not None else "",
                "best_val_loss": _best_val_loss_from_history(summary_path.parent, best_epoch),
                "best_val_acc": valm.get("acc", ""),
                "best_val_precision": valm.get("precision", ""),
                "best_val_recall": valm.get("recall", ""),
                "best_val_f1": valm.get("f1", ""),
                "best_val_auc": valm.get("auc", ""),
                "test_loss": test.get("loss", ""),
                "test_acc": testm.get("acc", ""),
                "test_precision": testm.get("precision", ""),
                "test_recall": testm.get("recall", ""),
                "test_f1": testm.get("f1", ""),
                "test_auc": testm.get("auc", ""),
                "train_total": ((counts.get("train") or {}).get("total", "")),
                "train_pos": ((counts.get("train") or {}).get("pos", "")),
                "train_neg": ((counts.get("train") or {}).get("neg", "")),
                "val_total": ((counts.get("val") or {}).get("total", "")),
                "val_pos": ((counts.get("val") or {}).get("pos", "")),
                "val_neg": ((counts.get("val") or {}).get("neg", "")),
                "test_total": ((counts.get("test") or {}).get("total", "")),
                "test_pos": ((counts.get("test") or {}).get("pos", "")),
                "test_neg": ((counts.get("test") or {}).get("neg", "")),
                "checkpoint_path": best.get("checkpoint_path", ""),
                "wandb_run_id": wandb.get("run_id", ""),
                "wandb_run_url": wandb.get("run_url", ""),
            }
        )

        rows.append(out)

    _write_csv(rows, out_csv)
    _write_markdown(rows, out_md, top_n=max(1, int(args.top_n)))

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote Markdown: {out_md}")
    print(f"Completed runs: {sum(1 for r in rows if r.get('status') == 'completed')}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
