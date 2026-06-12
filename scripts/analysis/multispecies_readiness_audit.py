#!/usr/bin/env python3
"""Audit multispecies experiment artifacts before treating results as reviewed.

This is a lightweight guardrail around the living experiment workflow. It checks
whether completed artifacts include the pieces we need for production decisions:
common-row metrics, row-level examples, H5 SSL coverage, and a ledger entry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


BAD_EXAMPLE_STATUSES = {
    "missing_examples_path",
    "directory_without_example_csv",
    "no_examples_path",
}


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        text = clean(value)
        return default if text == "" else float(text)
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def artifact_exists(path_text: Any, *, base_dir: Optional[Path] = None) -> bool:
    text = clean(path_text)
    if not text:
        return False
    path = Path(text)
    if path.is_absolute():
        return path.exists()
    return ((base_dir or Path.cwd()) / path).exists()


def load_csv_rows(path_text: Any, *, base_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    text = clean(path_text)
    if not text:
        return []
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    if not path.is_file():
        return []
    return read_csv(path)


def add_check(
    checks: List[Dict[str, Any]],
    *,
    artifact_type: str,
    artifact: str,
    check: str,
    status: str,
    detail: str = "",
    value: Any = "",
    threshold: Any = "",
) -> None:
    checks.append(
        {
            "artifact_type": artifact_type,
            "artifact": artifact,
            "check": check,
            "status": status,
            "value": value,
            "threshold": threshold,
            "detail": detail,
        }
    )


def ledger_contains(ledger_text: str, tokens: Sequence[Any]) -> bool:
    return any(clean(token) and clean(token) in ledger_text for token in tokens)


def audit_leaderboard(
    path: Path,
    *,
    ledger_text: str,
    require_ledger: bool,
    checks: List[Dict[str, Any]],
) -> None:
    payload = read_json(path)
    artifact = str(path)
    candidates = list(payload.get("candidates", []) or [])
    add_check(
        checks,
        artifact_type="leaderboard",
        artifact=artifact,
        check="has_candidates",
        status="PASS" if candidates else "FAIL",
        value=len(candidates),
        threshold=">0",
    )
    top = candidates[0] if candidates else {}
    for key in ("macro_f1", "micro_f1", "precision", "recall"):
        value = as_float(top.get(key))
        add_check(
            checks,
            artifact_type="leaderboard",
            artifact=artifact,
            check=f"top_candidate_{key}",
            status="PASS" if value is not None else "FAIL",
            value="" if value is None else value,
            detail=clean(top.get("candidate")),
        )
    for key in ("cross_species_fp", "background_fp", "species_as_background_fn"):
        value = as_float(top.get(key))
        add_check(
            checks,
            artifact_type="leaderboard",
            artifact=artifact,
            check=f"top_candidate_{key}",
            status="PASS" if value is not None else "FAIL",
            value="" if value is None else value,
            detail=clean(top.get("candidate")),
        )

    examples_path = payload.get("candidate_examples_csv")
    examples_exist = artifact_exists(examples_path, base_dir=path.parent)
    examples = load_csv_rows(examples_path, base_dir=path.parent)
    add_check(
        checks,
        artifact_type="leaderboard",
        artifact=artifact,
        check="candidate_examples_csv_exists",
        status="PASS" if examples_exist else "FAIL",
        value=clean(examples_path),
    )
    add_check(
        checks,
        artifact_type="leaderboard",
        artifact=artifact,
        check="candidate_examples_rows",
        status="PASS" if examples else "FAIL",
        value=len(examples),
        threshold=">0",
    )
    top_rank = clean(top.get("rank")) or "1"
    top_bad = [
        row for row in examples
        if clean(row.get("candidate_rank")) == top_rank and clean(row.get("example_status")) in BAD_EXAMPLE_STATUSES
    ]
    add_check(
        checks,
        artifact_type="leaderboard",
        artifact=artifact,
        check="top_candidate_examples_are_row_level",
        status="PASS" if examples and not top_bad else "FAIL",
        value=len(top_bad),
        threshold="0 bad statuses",
        detail=";".join(sorted({clean(row.get("example_status")) for row in top_bad if clean(row.get("example_status"))})),
    )
    if require_ledger:
        in_ledger = ledger_contains(
            ledger_text,
            [path, payload.get("leaderboard_json"), payload.get("report"), payload.get("title")],
        )
        add_check(
            checks,
            artifact_type="leaderboard",
            artifact=artifact,
            check="ledger_entry_present",
            status="PASS" if in_ledger else "FAIL",
            detail="looked for leaderboard path/report/title",
        )


def metric_by_split(summary: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    for row in summary.get("metrics", []) or []:
        if clean(row.get("split")) == split:
            return row
    return {}


def audit_binary_gate(
    path: Path,
    *,
    ledger_text: str,
    require_ledger: bool,
    checks: List[Dict[str, Any]],
) -> None:
    summary = read_json(path)
    artifact = str(path)
    for split in ("val", "test"):
        row = metric_by_split(summary, split)
        add_check(
            checks,
            artifact_type="binary_gate",
            artifact=artifact,
            check=f"{split}_metrics_present",
            status="PASS" if row else "FAIL",
        )
        for key in ("precision", "recall", "f1", "accuracy"):
            value = as_float(row.get(key)) if row else None
            add_check(
                checks,
                artifact_type="binary_gate",
                artifact=artifact,
                check=f"{split}_{key}",
                status="PASS" if value is not None else "FAIL",
                value="" if value is None else value,
            )
    outputs = summary.get("outputs", {}) or {}
    examples_path = outputs.get("examples")
    examples_exist = artifact_exists(examples_path, base_dir=path.parent)
    examples = load_csv_rows(examples_path, base_dir=path.parent)
    add_check(
        checks,
        artifact_type="binary_gate",
        artifact=artifact,
        check="examples_csv_exists",
        status="PASS" if examples_exist else "FAIL",
        value=clean(examples_path),
    )
    add_check(
        checks,
        artifact_type="binary_gate",
        artifact=artifact,
        check="examples_rows",
        status="PASS" if examples else "FAIL",
        value=len(examples),
        threshold=">0",
    )
    if require_ledger:
        in_ledger = ledger_contains(ledger_text, [path, summary.get("name"), outputs.get("report")])
        add_check(
            checks,
            artifact_type="binary_gate",
            artifact=artifact,
            check="ledger_entry_present",
            status="PASS" if in_ledger else "FAIL",
        )


def audit_h5(
    path: Path,
    *,
    ledger_text: str,
    require_ledger: bool,
    checks: List[Dict[str, Any]],
    min_normal_train: int,
    min_normal_months: int,
) -> None:
    audit = read_json(path)
    artifact = str(path)
    summary = audit.get("summary", {}) or {}
    quality_checks = list(audit.get("quality_checks", []) or [])
    normal_train = int(summary.get("normal_train_rows") or 0)
    normal_months = int(summary.get("normal_train_months") or summary.get("normal_months") or 0)
    add_check(
        checks,
        artifact_type="h5_audit",
        artifact=artifact,
        check="normal_train_rows",
        status="PASS" if normal_train >= int(min_normal_train) else "FAIL",
        value=normal_train,
        threshold=min_normal_train,
    )
    add_check(
        checks,
        artifact_type="h5_audit",
        artifact=artifact,
        check="normal_train_months",
        status="PASS" if normal_months >= int(min_normal_months) else "FAIL",
        value=normal_months,
        threshold=min_normal_months,
    )
    failed = [row for row in quality_checks if not bool(row.get("passed"))]
    add_check(
        checks,
        artifact_type="h5_audit",
        artifact=artifact,
        check="quality_checks_passed",
        status="PASS" if quality_checks and not failed else "FAIL",
        value=len(failed),
        threshold="0 failed checks",
    )
    outputs = audit.get("outputs", {}) or {}
    if require_ledger:
        in_ledger = ledger_contains(ledger_text, [path, audit.get("input_h5"), outputs.get("report")])
        add_check(
            checks,
            artifact_type="h5_audit",
            artifact=artifact,
            check="ledger_entry_present",
            status="PASS" if in_ledger else "FAIL",
        )


def status_counts(checks: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for row in checks:
        status = clean(row.get("status")) or "FAIL"
        counts[status] = counts.get(status, 0) + 1
    return counts


def markdown_report(checks: Sequence[Mapping[str, Any]], *, title: str) -> str:
    counts = status_counts(checks)
    lines = [
        f"# {title}",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        f"| PASS | {counts.get('PASS', 0)} |",
        f"| WARN | {counts.get('WARN', 0)} |",
        f"| FAIL | {counts.get('FAIL', 0)} |",
        "",
        "## Checks",
        "",
        "| status | artifact type | check | value | threshold | detail |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in checks:
        lines.append(
            "| {status} | {artifact_type} | {check} | {value} | {threshold} | {detail} |".format(
                status=clean(row.get("status")),
                artifact_type=clean(row.get("artifact_type")),
                check=clean(row.get("check")),
                value=clean(row.get("value")),
                threshold=clean(row.get("threshold")),
                detail=clean(row.get("detail")).replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def run_audit(
    *,
    output_dir: Path,
    ledger_path: Optional[Path],
    require_ledger: bool,
    leaderboard_jsons: Sequence[Path],
    binary_gate_summary_jsons: Sequence[Path],
    h5_audit_jsons: Sequence[Path],
    min_normal_train: int,
    min_normal_months: int,
    title: str,
) -> Dict[str, Any]:
    ledger_text = ledger_path.read_text(encoding="utf-8") if ledger_path and ledger_path.exists() else ""
    checks: List[Dict[str, Any]] = []
    for path in leaderboard_jsons:
        audit_leaderboard(path, ledger_text=ledger_text, require_ledger=require_ledger, checks=checks)
    for path in binary_gate_summary_jsons:
        audit_binary_gate(path, ledger_text=ledger_text, require_ledger=require_ledger, checks=checks)
    for path in h5_audit_jsons:
        audit_h5(
            path,
            ledger_text=ledger_text,
            require_ledger=require_ledger,
            checks=checks,
            min_normal_train=min_normal_train,
            min_normal_months=min_normal_months,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "multispecies_readiness_audit.md"
    checks_csv = output_dir / "multispecies_readiness_audit_checks.csv"
    summary_path = output_dir / "multispecies_readiness_audit_summary.json"
    write_csv(checks_csv, checks)
    payload = {
        "title": title,
        "status_counts": status_counts(checks),
        "checks": checks,
        "outputs": {
            "report": str(report_path),
            "checks_csv": str(checks_csv),
            "summary": str(summary_path),
        },
    }
    report_path.write_text(markdown_report(checks, title=title), encoding="utf-8")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ledger-path", default=None, type=Path)
    parser.add_argument("--require-ledger", action="store_true")
    parser.add_argument("--leaderboard-json", action="append", type=Path, default=[])
    parser.add_argument("--binary-gate-summary-json", action="append", type=Path, default=[])
    parser.add_argument("--h5-audit-json", action="append", type=Path, default=[])
    parser.add_argument("--min-normal-train", type=int, default=10000)
    parser.add_argument("--min-normal-months", type=int, default=12)
    parser.add_argument("--title", default="Multispecies Experiment Readiness Audit")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_audit(
        output_dir=args.output_dir,
        ledger_path=args.ledger_path,
        require_ledger=bool(args.require_ledger),
        leaderboard_jsons=args.leaderboard_json,
        binary_gate_summary_jsons=args.binary_gate_summary_json,
        h5_audit_jsons=args.h5_audit_json,
        min_normal_train=int(args.min_normal_train),
        min_normal_months=int(args.min_normal_months),
        title=args.title,
    )
    print(json.dumps({"summary": payload["outputs"]["summary"], "report": payload["outputs"]["report"]}, indent=2))
    if args.fail_on_incomplete and int(payload["status_counts"].get("FAIL", 0)) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
