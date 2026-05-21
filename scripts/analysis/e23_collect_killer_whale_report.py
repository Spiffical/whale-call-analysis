#!/usr/bin/env python3
"""Collect E23 killer whale split diagnostics into a compact report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def clean(value: Any) -> str:
    return str(value or "").strip()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_per_label(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def hard_fp(payload: Mapping[str, Any]) -> tuple[int, int, float | str]:
    rows = payload.get("onc_test_hard_negative_fp_rows", []) or []
    fp = sum(int(row.get("any_primary_fp") or 0) for row in rows)
    total = sum(int(row.get("rows") or 0) for row in rows)
    return fp, total, fp / total if total else ""


def metrics_rows(submitted_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for submitted in submitted_rows:
        experiment = clean(submitted.get("experiment"))
        run_dir = Path(clean(submitted.get("run_dir")))
        if not experiment or not run_dir or clean(submitted.get("job_id")) == "DRY_RUN":
            continue
        metrics_path = run_dir / "train" / "onc_calibrated_eval" / "onc_calibrated_metrics_summary.json"
        payload = load_json(metrics_path)
        if not payload:
            rows.append(
                {
                    "job_id": clean(submitted.get("job_id")),
                    "experiment": experiment,
                    "status": "missing_metrics",
                    "run_dir": str(run_dir),
                    "metrics_path": str(metrics_path),
                }
            )
            continue
        test = payload.get("onc_test_metrics", {}) or {}
        per_label = read_per_label(run_dir / "train" / "onc_calibrated_eval" / "onc_calibrated_test_per_label.csv")
        label_row = per_label[0] if per_label else {}
        fp, total, rate = hard_fp(payload)
        rows.append(
            {
                "job_id": clean(submitted.get("job_id")),
                "experiment": experiment,
                "status": "complete",
                "macro_f1": test.get("macro_f1_supported", 0.0),
                "micro_f1": test.get("micro_f1", 0.0),
                "precision": test.get("micro_precision", 0.0),
                "recall": test.get("micro_recall", 0.0),
                "tp": test.get("tp", 0),
                "fp": test.get("fp", 0),
                "fn": test.get("fn", 0),
                "killer_whale_threshold": label_row.get("threshold", ""),
                "killer_whale_support": label_row.get("support", ""),
                "hard_negative_fp": fp,
                "hard_negative_total": total,
                "hard_negative_fp_rate": rate,
                "label_ids": ",".join(payload.get("label_ids", [])),
                "calibration_source_kind": payload.get("calibration_source_kind", ""),
                "eval_source_kind": payload.get("eval_source_kind", ""),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def variant_rows(variant_root: Optional[Path]) -> List[Dict[str, Any]]:
    if variant_root is None:
        return []
    index_path = variant_root / "variant_index.json"
    payload = load_json(index_path)
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        split_counts = item.get("split_counts", {}) or {}
        split_summary = item.get("split_grouping_summary", {}) or {}
        oversample = item.get("oversample_summary", {}) or {}
        rows.append(
            {
                "variant": item.get("variant_name", ""),
                "rows": item.get("row_count", 0),
                "train_rows": split_counts.get("train", 0),
                "val_rows": split_counts.get("val", 0),
                "test_rows": split_counts.get("test", 0),
                "bands": ",".join(item.get("bands", [])),
                "sources": ",".join(item.get("sources", [])),
                "split_grouping": split_summary.get("mode", ""),
                "split_groups": split_summary.get("group_count", ""),
                "positive_group_counts": json.dumps(split_summary.get("positive_group_counts", {}), sort_keys=True),
                "dropped_oversampled_rows": oversample.get("dropped_oversampled_rows", 0),
                "manifest": item.get("manifest_csv", ""),
            }
        )
    return rows


def markdown_report(
    *,
    metric_rows: Sequence[Mapping[str, Any]],
    variant_rows_: Sequence[Mapping[str, Any]],
    audit_summary: Mapping[str, Any],
    audit_dir: Optional[Path],
    output_dir: Path,
) -> str:
    lines = [
        "# E23 Killer Whale Localization Diagnostics",
        "",
        "E23 tests whether the weak ONC killer whale result is caused by insufficient or leaky ONC support. It drops synthetic oversampling, preserves raw ONC time/frequency metadata, and compares clip-held-out versus date-held-out splits.",
        "",
        "## Metrics",
        "",
        "| experiment | macro F1 | precision | recall | TP | FP | FN | hard FP | threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        if row.get("status") != "complete":
            lines.append(f"| {row.get('experiment')} | missing |  |  |  |  |  |  |  |")
            continue
        hard = row.get("hard_negative_fp_rate")
        hard_text = "" if hard == "" else f"{float(hard):.4f}"
        lines.append(
            "| {experiment} | {macro:.4f} | {precision:.4f} | {recall:.4f} | {tp} | {fp} | {fn} | {hard} | {threshold} |".format(
                experiment=row.get("experiment", ""),
                macro=float(row.get("macro_f1") or 0.0),
                precision=float(row.get("precision") or 0.0),
                recall=float(row.get("recall") or 0.0),
                tp=row.get("tp", 0),
                fp=row.get("fp", 0),
                fn=row.get("fn", 0),
                hard=hard_text,
                threshold=row.get("killer_whale_threshold", ""),
            )
        )

    lines.extend(["", "## Variant Checks", ""])
    if variant_rows_:
        lines.extend(
            [
                "| variant | rows | train | val | test | grouping | positive groups | dropped oversampled |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for row in variant_rows_:
            lines.append(
                f"| {row.get('variant')} | {row.get('rows')} | {row.get('train_rows')} | {row.get('val_rows')} | {row.get('test_rows')} | {row.get('split_grouping')} | {row.get('positive_group_counts')} | {row.get('dropped_oversampled_rows')} |"
            )
    else:
        lines.append("Variant summaries were not available.")

    lines.extend(["", "## Annotation Audit", ""])
    if audit_summary:
        lines.append(f"- Raw ONC killer whale annotations reviewed: {audit_summary.get('annotation_count', 0)}")
        lines.append(f"- Included in E16/E22 manifest: {audit_summary.get('included_in_e16_manifest_count', 0)}")
        lines.append(f"- Fully inside high band 500-32000 Hz: {audit_summary.get('fully_inside_high_500_32000', 0)}")
        lines.append(f"- Extends above 32000 Hz: {audit_summary.get('extends_above_32000', 0)}")
        lines.append(f"- Review CSV: `{audit_summary.get('review_csv', '')}`")
        contact_sheets = (audit_summary.get("image_summary", {}) or {}).get("contact_sheets", [])
        if contact_sheets:
            lines.append("- Contact sheets:")
            lines.extend(f"  - `{path}`" for path in contact_sheets)
    elif audit_dir:
        lines.append(f"Audit summary was not available yet under `{audit_dir}`.")
    else:
        lines.append("No audit directory was supplied.")

    lines.extend(
        [
            "",
            "## Gate Read",
            "",
            "Killer whale should remain diagnostic unless one of the ONC-calibrated split-safe runs materially improves recall/F1 without a false-positive blowup. If date-held-out performance collapses relative to clip-held-out, the apparent ONC signal is probably too date/site-specific for deployment.",
            "",
            f"Metrics CSV: `{output_dir / 'e23_metrics_summary.csv'}`",
            f"Variant CSV: `{output_dir / 'e23_variant_summary.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submitted-tsv", required=True, type=Path)
    parser.add_argument("--variant-root", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submitted_rows = read_tsv(args.submitted_tsv)
    metrics = metrics_rows(submitted_rows)
    variants = variant_rows(args.variant_root)
    audit_summary = load_json(args.audit_dir / "onc_killer_whale_annotation_audit_summary.json") if args.audit_dir else {}

    write_csv(args.output_dir / "e23_metrics_summary.csv", metrics)
    write_csv(args.output_dir / "e23_variant_summary.csv", variants)
    report = markdown_report(
        metric_rows=metrics,
        variant_rows_=variants,
        audit_summary=audit_summary,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
    )
    report_path = args.output_dir / "e23_killer_whale_diagnostics_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics_rows": len(metrics), "variant_rows": len(variants)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
