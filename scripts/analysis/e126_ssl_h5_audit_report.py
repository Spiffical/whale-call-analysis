#!/usr/bin/env python3
"""Audit an E123/E126 SSAMBA H5 dataset for SSL coverage and split balance."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import multispecies_experiment_ledger as experiment_ledger

try:
    import h5py
except Exception:
    h5py = None


DEFAULT_TARGET_LABELS = ("Bm", "Bp", "Mn")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def label_tokens(label_string: str) -> List[str]:
    labels = [
        clean(part)
        for part in re.split(r"[;,|]", clean(label_string))
        if clean(part)
    ]
    return labels or ["unlabeled"]


def extract_month(*values: Any) -> str:
    """Extract YYYY-MM from common ONC/manifest/file timestamp encodings."""
    text = " ".join(clean(value) for value in values if clean(value))
    if not text:
        return "unknown"
    patterns = [
        re.compile(r"(?P<year>20\d{2})[-_/](?P<month>0[1-9]|1[0-2])[-_/]\d{2}"),
        re.compile(r"(?P<year>20\d{2})(?P<month>0[1-9]|1[0-2])\d{2}T\d{6}"),
        re.compile(r"(?P<year>20\d{2})(?P<month>0[1-9]|1[0-2])\d{2}"),
        re.compile(r"(?P<year>20\d{2})[-_/](?P<month>0[1-9]|1[0-2])"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return f"{match.group('year')}-{match.group('month')}"
    return "unknown"


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
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def counter_rows(counter: Counter, *, key_name: str, value_name: str = "rows") -> List[Dict[str, Any]]:
    return [
        {key_name: key, value_name: value}
        for key, value in sorted(counter.items(), key=lambda item: (str(item[0]), item[1]))
    ]


def read_strings(handle: Any, dataset_name: str, default: str = "") -> List[str]:
    if dataset_name not in handle:
        rows = int(handle["spectrograms"].shape[0]) if "spectrograms" in handle else 0
        return [default for _ in range(rows)]
    return [clean(value) for value in handle[dataset_name][:]]


def summarize_rows(
    *,
    label_strings: Sequence[str],
    splits: Sequence[str],
    source_kinds: Sequence[str],
    item_ids: Sequence[str],
    sources: Sequence[str],
    target_labels: Sequence[str] = DEFAULT_TARGET_LABELS,
    spectrogram_shape: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    row_count = len(label_strings)
    if not all(len(values) == row_count for values in (splits, source_kinds, item_ids, sources)):
        raise ValueError("label_strings, splits, source_kinds, item_ids, and sources must have the same length")

    label_counts: Counter = Counter()
    split_counts: Counter = Counter(clean(split) or "unknown" for split in splits)
    source_kind_counts: Counter = Counter(clean(kind) or "unknown" for kind in source_kinds)
    split_label_counts: Counter = Counter()
    month_counts: Counter = Counter()
    normal_month_counts: Counter = Counter()
    normal_train_month_counts: Counter = Counter()
    split_normal_counts: Counter = Counter()
    target_counts: Counter = Counter()
    unknown_month_rows = 0

    targets = set(target_labels)
    for label_string, split, source_kind, item_id, source in zip(
        label_strings, splits, source_kinds, item_ids, sources
    ):
        labels = set(label_tokens(label_string))
        split_key = clean(split) or "unknown"
        source_key = clean(source_kind) or "unknown"
        month = extract_month(item_id, source)
        month_counts[month] += 1
        if month == "unknown":
            unknown_month_rows += 1
        for label in sorted(labels):
            label_counts[label] += 1
            split_label_counts[(split_key, label)] += 1
            if label in targets:
                target_counts[label] += 1
        if "normal" in labels:
            normal_month_counts[month] += 1
            split_normal_counts[split_key] += 1
            if split_key == "train":
                normal_train_month_counts[month] += 1

    label_count_rows = counter_rows(label_counts, key_name="label")
    split_count_rows = counter_rows(split_counts, key_name="split")
    source_kind_count_rows = counter_rows(source_kind_counts, key_name="source_kind")
    month_count_rows = counter_rows(month_counts, key_name="month")
    normal_month_count_rows = counter_rows(normal_month_counts, key_name="month")
    normal_train_month_count_rows = counter_rows(normal_train_month_counts, key_name="month")
    split_label_rows = [
        {"split": split, "label": label, "rows": rows}
        for (split, label), rows in sorted(split_label_counts.items())
    ]
    target_label_rows = [
        {"label": label, "rows": int(target_counts.get(label, 0))}
        for label in target_labels
    ]
    normal_rows = int(label_counts.get("normal", 0))
    normal_months = sorted(month for month, rows in normal_month_counts.items() if month != "unknown" and rows > 0)
    normal_train_months = sorted(month for month, rows in normal_train_month_counts.items() if month != "unknown" and rows > 0)
    months = sorted(month for month, rows in month_counts.items() if month != "unknown" and rows > 0)
    return {
        "rows": row_count,
        "spectrogram_shape": list(spectrogram_shape or []),
        "normal_rows": normal_rows,
        "normal_train_rows": int(split_normal_counts.get("train", 0)),
        "normal_months": len(normal_months),
        "normal_month_range": [normal_months[0], normal_months[-1]] if normal_months else [],
        "normal_train_months": len(normal_train_months),
        "normal_train_month_range": [normal_train_months[0], normal_train_months[-1]] if normal_train_months else [],
        "months": len(months),
        "month_range": [months[0], months[-1]] if months else [],
        "unknown_month_rows": unknown_month_rows,
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "target_label_counts": {label: int(target_counts.get(label, 0)) for label in target_labels},
        "label_count_rows": label_count_rows,
        "split_count_rows": split_count_rows,
        "source_kind_count_rows": source_kind_count_rows,
        "month_count_rows": month_count_rows,
        "normal_month_count_rows": normal_month_count_rows,
        "normal_train_month_count_rows": normal_train_month_count_rows,
        "split_label_rows": split_label_rows,
        "target_label_rows": target_label_rows,
    }


def quality_checks(
    summary: Mapping[str, Any],
    *,
    min_normal_rows: int,
    min_normal_train_rows: int,
    min_normal_months: int,
    min_normal_train_months: int,
    target_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    target_counts = summary.get("target_label_counts") or {}
    checks = [
        {
            "check": "normal_rows",
            "value": int(summary.get("normal_rows", 0)),
            "threshold": int(min_normal_rows),
            "passed": int(summary.get("normal_rows", 0)) >= int(min_normal_rows),
        },
        {
            "check": "normal_train_rows",
            "value": int(summary.get("normal_train_rows", 0)),
            "threshold": int(min_normal_train_rows),
            "passed": int(summary.get("normal_train_rows", 0)) >= int(min_normal_train_rows),
        },
        {
            "check": "normal_months",
            "value": int(summary.get("normal_months", 0)),
            "threshold": int(min_normal_months),
            "passed": int(summary.get("normal_months", 0)) >= int(min_normal_months),
        },
        {
            "check": "normal_train_months",
            "value": int(summary.get("normal_train_months", 0)),
            "threshold": int(min_normal_train_months),
            "passed": int(summary.get("normal_train_months", 0)) >= int(min_normal_train_months),
        },
    ]
    for label in target_labels:
        value = int(target_counts.get(label, 0))
        checks.append({"check": f"target_rows:{label}", "value": value, "threshold": 1, "passed": value > 0})
    return checks


def load_h5_summary(input_h5: Path, *, target_labels: Sequence[str]) -> Dict[str, Any]:
    if h5py is None:
        raise RuntimeError("h5py is required to audit an E123/E126 H5 dataset")
    with h5py.File(input_h5, "r") as handle:
        if "label_strings" not in handle:
            raise KeyError("H5 missing required dataset 'label_strings'")
        label_strings = read_strings(handle, "label_strings")
        splits = read_strings(handle, "splits", default="unknown")
        source_kinds = read_strings(handle, "source_kinds", default="unknown")
        item_ids = read_strings(handle, "item_ids")
        sources = read_strings(handle, "sources")
        spectrogram_shape = list(handle["spectrograms"].shape) if "spectrograms" in handle else []
    return summarize_rows(
        label_strings=label_strings,
        splits=splits,
        source_kinds=source_kinds,
        item_ids=item_ids,
        sources=sources,
        target_labels=target_labels,
        spectrogram_shape=spectrogram_shape,
    )


def markdown_report(
    *,
    input_h5: Path,
    output_dir: Path,
    summary: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    builder_summary: Optional[Mapping[str, Any]],
) -> str:
    lines = [
        "# E126 SSL H5 Audit Report",
        "",
        f"H5 dataset: `{input_h5}`",
        "",
        "## Coverage Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| rows | {summary.get('rows', 0)} |",
        f"| normal rows | {summary.get('normal_rows', 0)} |",
        f"| normal train rows | {summary.get('normal_train_rows', 0)} |",
        f"| normal months | {summary.get('normal_months', 0)} |",
        f"| normal train months | {summary.get('normal_train_months', 0)} |",
        f"| all months | {summary.get('months', 0)} |",
        f"| unknown-month rows | {summary.get('unknown_month_rows', 0)} |",
    ]
    shape = summary.get("spectrogram_shape") or []
    if shape:
        lines.append(f"| spectrogram dataset shape | {' x '.join(str(part) for part in shape)} |")
    lines.extend(["", "## Quality Checks", "", "| check | value | threshold | passed |", "| --- | ---: | ---: | --- |"])
    for row in checks:
        lines.append(
            "| {check} | {value} | {threshold} | {passed} |".format(
                check=row.get("check"),
                value=row.get("value"),
                threshold=row.get("threshold"),
                passed="yes" if row.get("passed") else "no",
            )
        )
    lines.extend(["", "## Label Counts", "", "| label | rows |", "| --- | ---: |"])
    for row in summary.get("label_count_rows", []):
        lines.append(f"| {row.get('label')} | {row.get('rows')} |")
    lines.extend(["", "## Split Counts", "", "| split | rows |", "| --- | ---: |"])
    for row in summary.get("split_count_rows", []):
        lines.append(f"| {row.get('split')} | {row.get('rows')} |")
    lines.extend(["", "## Normal Month Counts", "", "| month | normal rows |", "| --- | ---: |"])
    normal_month_rows = list(summary.get("normal_month_count_rows", []))
    for row in normal_month_rows[:80]:
        lines.append(f"| {row.get('month')} | {row.get('rows')} |")
    if len(normal_month_rows) > 80:
        lines.append(f"| ... | {len(normal_month_rows) - 80} more months omitted |")
    lines.extend(["", "## Normal Train Month Counts", "", "| month | normal train rows |", "| --- | ---: |"])
    normal_train_month_rows = list(summary.get("normal_train_month_count_rows", []))
    for row in normal_train_month_rows[:80]:
        lines.append(f"| {row.get('month')} | {row.get('rows')} |")
    if len(normal_train_month_rows) > 80:
        lines.append(f"| ... | {len(normal_train_month_rows) - 80} more months omitted |")
    if builder_summary:
        lines.extend(
            [
                "",
                "## Builder Summary Cross-Check",
                "",
                f"- Rows written by builder summary: `{builder_summary.get('rows_written', '')}`",
                f"- Builder label counts: `{json.dumps(builder_summary.get('label_counts', {}), sort_keys=True)}`",
                f"- Builder split counts: `{json.dumps(builder_summary.get('split_counts', {}), sort_keys=True)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"Summary JSON: `{output_dir / 'e126_ssl_h5_audit_summary.json'}`",
            f"Label counts CSV: `{output_dir / 'e126_ssl_h5_label_counts.csv'}`",
            f"Split-label counts CSV: `{output_dir / 'e126_ssl_h5_split_label_counts.csv'}`",
            f"Normal month counts CSV: `{output_dir / 'e126_ssl_h5_normal_month_counts.csv'}`",
            f"Normal train month counts CSV: `{output_dir / 'e126_ssl_h5_normal_train_month_counts.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(
    *,
    input_h5: Path,
    output_dir: Path,
    builder_summary_json: Optional[Path],
    target_labels: Sequence[str],
    min_normal_rows: int,
    min_normal_train_rows: int,
    min_normal_months: int,
    min_normal_train_months: int,
    ledger_path: Optional[Path] = None,
    ledger_entry_id: str = "",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_h5_summary(input_h5, target_labels=target_labels)
    checks = quality_checks(
        summary,
        min_normal_rows=min_normal_rows,
        min_normal_train_rows=min_normal_train_rows,
        min_normal_months=min_normal_months,
        min_normal_train_months=min_normal_train_months,
        target_labels=target_labels,
    )
    builder_summary = None
    if builder_summary_json and builder_summary_json.is_file():
        builder_summary = json.loads(builder_summary_json.read_text(encoding="utf-8"))

    write_csv(output_dir / "e126_ssl_h5_label_counts.csv", summary["label_count_rows"])
    write_csv(output_dir / "e126_ssl_h5_split_counts.csv", summary["split_count_rows"])
    write_csv(output_dir / "e126_ssl_h5_source_kind_counts.csv", summary["source_kind_count_rows"])
    write_csv(output_dir / "e126_ssl_h5_month_counts.csv", summary["month_count_rows"])
    write_csv(output_dir / "e126_ssl_h5_normal_month_counts.csv", summary["normal_month_count_rows"])
    write_csv(output_dir / "e126_ssl_h5_normal_train_month_counts.csv", summary["normal_train_month_count_rows"])
    write_csv(output_dir / "e126_ssl_h5_split_label_counts.csv", summary["split_label_rows"])
    write_csv(output_dir / "e126_ssl_h5_quality_checks.csv", checks)

    report_path = output_dir / "e126_ssl_h5_audit_report.md"
    report_path.write_text(
        markdown_report(
            input_h5=input_h5,
            output_dir=output_dir,
            summary=summary,
            checks=checks,
            builder_summary=builder_summary,
        ),
        encoding="utf-8",
    )
    payload = {
        "input_h5": str(input_h5),
        "builder_summary_json": str(builder_summary_json) if builder_summary_json else "",
        "target_labels": list(target_labels),
        "min_normal_rows": int(min_normal_rows),
        "min_normal_train_rows": int(min_normal_train_rows),
        "min_normal_months": int(min_normal_months),
        "min_normal_train_months": int(min_normal_train_months),
        "summary": summary,
        "quality_checks": list(checks),
        "outputs": {
            "report": str(report_path),
            "summary": str(output_dir / "e126_ssl_h5_audit_summary.json"),
            "label_counts": str(output_dir / "e126_ssl_h5_label_counts.csv"),
            "split_counts": str(output_dir / "e126_ssl_h5_split_counts.csv"),
            "source_kind_counts": str(output_dir / "e126_ssl_h5_source_kind_counts.csv"),
            "month_counts": str(output_dir / "e126_ssl_h5_month_counts.csv"),
            "normal_month_counts": str(output_dir / "e126_ssl_h5_normal_month_counts.csv"),
            "normal_train_month_counts": str(output_dir / "e126_ssl_h5_normal_train_month_counts.csv"),
            "split_label_counts": str(output_dir / "e126_ssl_h5_split_label_counts.csv"),
            "quality_checks": str(output_dir / "e126_ssl_h5_quality_checks.csv"),
        },
    }
    if ledger_path is not None:
        ledger_written = experiment_ledger.append_h5_audit_summary(
            audit=payload,
            ledger_path=ledger_path,
            entry_id=ledger_entry_id,
        )
        payload["outputs"]["ledger"] = str(ledger_written)
    (output_dir / "e126_ssl_h5_audit_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_labels(value: str) -> List[str]:
    labels = [clean(part) for part in value.split(",") if clean(part)]
    if not labels:
        raise ValueError("target label list cannot be empty")
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--builder-summary-json", default=None, type=Path)
    parser.add_argument("--target-labels", default=",".join(DEFAULT_TARGET_LABELS))
    parser.add_argument("--min-normal-rows", type=int, default=10000)
    parser.add_argument("--min-normal-train-rows", type=int, default=10000)
    parser.add_argument("--min-normal-months", type=int, default=12)
    parser.add_argument("--min-normal-train-months", type=int, default=12)
    parser.add_argument("--ledger-path", default=None, type=Path)
    parser.add_argument("--ledger-entry-id", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_audit(
        input_h5=args.input_h5,
        output_dir=args.output_dir,
        builder_summary_json=args.builder_summary_json,
        target_labels=parse_labels(args.target_labels),
        min_normal_rows=args.min_normal_rows,
        min_normal_train_rows=args.min_normal_train_rows,
        min_normal_months=args.min_normal_months,
        min_normal_train_months=args.min_normal_train_months,
        ledger_path=args.ledger_path,
        ledger_entry_id=args.ledger_entry_id,
    )
    print(json.dumps({"report": payload["outputs"]["report"], "summary": payload["outputs"]["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
