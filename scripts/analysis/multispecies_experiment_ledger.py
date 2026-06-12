#!/usr/bin/env python3
"""Append concise experiment results to the multispecies living ledger."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = REPO_ROOT / "docs" / "multispecies_experiment_results.md"
INSERT_BEFORE_HEADING = "## Immediate Next Entries To Add"


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = clean(value)
        return default if text == "" else float(text)
    except (TypeError, ValueError):
        return default


def format_number(value: Any, *, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return clean(value)
    return f"{numeric:.{digits}f}"


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", clean(value)).strip("-")
    return text or "experiment"


def current_date_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def metric_by_split(summary: Mapping[str, Any], split: str) -> Dict[str, Any]:
    for row in summary.get("metrics", []):
        if clean(row.get("split")) == split:
            return dict(row)
    return {}


def append_or_replace_block(
    *,
    ledger_path: Path,
    entry_id: str,
    body: str,
    insert_before_heading: str = INSERT_BEFORE_HEADING,
) -> Path:
    """Append an entry to the ledger, replacing an existing block with the same id."""
    marker = slugify(entry_id)
    begin = f"<!-- BEGIN experiment-ledger-entry:{marker} -->"
    end = f"<!-- END experiment-ledger-entry:{marker} -->"
    block = f"{begin}\n{body.rstrip()}\n{end}\n"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        text = ledger_path.read_text(encoding="utf-8")
    else:
        text = "# Multispecies Experiment Results Ledger\n\n"

    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(block, text)
    elif insert_before_heading and insert_before_heading in text:
        text = text.replace(insert_before_heading, block + "\n" + insert_before_heading, 1)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    ledger_path.write_text(text, encoding="utf-8")
    return ledger_path


def binary_gate_entry_markdown(
    *,
    summary: Mapping[str, Any],
    summary_path: Optional[Path],
    training_set: str,
    validation_set: str,
    test_set: str,
    evaluation_note: str,
    status: str,
    entry_date: str,
) -> str:
    name = clean(summary.get("name")) or "binary gate"
    threshold = summary.get("threshold", "")
    val = metric_by_split(summary, "val")
    test = metric_by_split(summary, "test")
    outputs = summary.get("outputs", {}) or {}
    breakdown = summary.get("test_breakdown") or summary.get("breakdown") or []
    if not breakdown and outputs.get("breakdown"):
        breakdown = []

    lines = [
        f"### {name}: Binary Whale Gate ({entry_date})",
        "",
        f"Status: {status}.",
        "",
        f"Training set: {training_set or 'not specified'}.",
        "",
        f"Validation set: {validation_set or clean(summary.get('inputs', {}).get('val_predictions')) or 'not specified'}.",
        "",
        f"Test set: {test_set or clean(summary.get('inputs', {}).get('test_predictions')) or 'not specified'}.",
        "",
        f"Evaluation: {evaluation_note or 'binary whale-call vs background gate evaluation'}.",
        "",
        f"Validation-tuned threshold: `{format_number(threshold, digits=2)}`.",
        "",
        "| Split | Rows | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in (("val", val), ("test", test)):
        if row:
            lines.append(
                "| {split} | {rows} | {precision} | {recall} | {f1} | {accuracy} | {tp} | {fp} | {tn} | {fn} |".format(
                    split=split,
                    rows=clean(row.get("rows")),
                    precision=format_number(row.get("precision")),
                    recall=format_number(row.get("recall")),
                    f1=format_number(row.get("f1")),
                    accuracy=format_number(row.get("accuracy")),
                    tp=clean(row.get("tp")),
                    fp=clean(row.get("fp")),
                    tn=clean(row.get("tn")),
                    fn=clean(row.get("fn")),
                )
            )
    if breakdown:
        lines.extend(
            [
                "",
                "Test breakdown by true class:",
                "",
                "| Group | Support | Detected | Missed/TN | Rate |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in breakdown:
            lines.append(
                "| {group} | {support} | {detected} | {missed} | {rate} |".format(
                    group=clean(row.get("true_bucket")),
                    support=clean(row.get("support")),
                    detected=clean(row.get("detected")),
                    missed=clean(row.get("missed")),
                    rate=format_number(row.get("detection_rate")),
                )
            )
    lines.extend(["", "Artifacts:"])
    if summary_path:
        lines.append(f"- Summary JSON: `{summary_path}`")
    for label, key in (
        ("Report", "report"),
        ("Metrics CSV", "metrics"),
        ("Threshold sweep CSV", "threshold_sweep"),
        ("Breakdown CSV", "breakdown"),
        ("Examples CSV", "examples"),
    ):
        value = clean(outputs.get(key))
        if value:
            lines.append(f"- {label}: `{value}`")
    lines.extend(
        [
            "",
            "Interpretation: update after review if this experiment changes model selection or production readiness.",
        ]
    )
    return "\n".join(lines) + "\n"


def append_binary_gate_summary(
    *,
    summary: Mapping[str, Any],
    summary_path: Optional[Path],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    training_set: str = "",
    validation_set: str = "",
    test_set: str = "",
    evaluation_note: str = "",
    status: str = "completed",
    entry_id: str = "",
    entry_date: str = "",
) -> Path:
    name = clean(summary.get("name")) or "binary-gate"
    summary_key = clean(summary_path) if summary_path else clean(summary.get("outputs", {}).get("summary"))
    marker = entry_id or f"binary-gate:{name}:{summary_key}"
    body = binary_gate_entry_markdown(
        summary=summary,
        summary_path=summary_path,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        evaluation_note=evaluation_note,
        status=status,
        entry_date=entry_date or current_date_utc(),
    )
    return append_or_replace_block(ledger_path=ledger_path, entry_id=marker, body=body)


def leaderboard_entry_markdown(
    *,
    leaderboard: Mapping[str, Any],
    training_set: str,
    validation_set: str,
    test_set: str,
    evaluation_note: str,
    status: str,
    entry_date: str,
    max_rows: int = 5,
) -> str:
    candidates = list(leaderboard.get("candidates", []) or [])
    title = clean(leaderboard.get("title")) or "Production Candidate Leaderboard"
    lines = [
        f"### {title} ({entry_date})",
        "",
        f"Status: {status}.",
        "",
        f"Training set: {training_set or 'see candidate summaries'}.",
        "",
        f"Validation set: {validation_set or 'see candidate summaries'}.",
        "",
        f"Test set: {test_set or 'common-row ONC test set; verify candidate comparability in source summaries'}.",
        "",
        f"Evaluation: {evaluation_note or 'production-style common-row candidate comparison'}.",
        "",
        "| Rank | Candidate | Experiment | Prediction | Macro F1 | Micro F1 | Precision | Recall | Cross FP | Background FP | Species-bg FN |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidates[: max(0, int(max_rows))]:
        lines.append(
            "| {rank} | {candidate} | {experiment} | {prediction} | {macro} | {micro} | {precision} | {recall} | {cross} | {bgfp} | {bgfn} |".format(
                rank=clean(row.get("rank")),
                candidate=clean(row.get("candidate")),
                experiment=clean(row.get("experiment")),
                prediction=clean(row.get("selected_prediction")),
                macro=format_number(row.get("macro_f1")),
                micro=format_number(row.get("micro_f1")),
                precision=format_number(row.get("precision")),
                recall=format_number(row.get("recall")),
                cross=clean(row.get("cross_species_fp")),
                bgfp=clean(row.get("background_fp")),
                bgfn=clean(row.get("species_as_background_fn")),
            )
        )
    lines.extend(["", "Artifacts:"])
    for label, key in (
        ("Report", "report"),
        ("Leaderboard CSV", "leaderboard_csv"),
        ("Leaderboard JSON", "leaderboard_json"),
        ("Candidate examples CSV", "candidate_examples_csv"),
    ):
        value = clean(leaderboard.get(key))
        if value:
            lines.append(f"- {label}: `{value}`")
    lines.extend(
        [
            "",
            "Interpretation: top-ranked rows are only directly comparable when they use the same held-out common ONC rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def append_leaderboard_summary(
    *,
    leaderboard: Mapping[str, Any],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    training_set: str = "",
    validation_set: str = "",
    test_set: str = "",
    evaluation_note: str = "",
    status: str = "completed",
    entry_id: str = "",
    entry_date: str = "",
    max_rows: int = 5,
) -> Path:
    marker = entry_id or f"leaderboard:{clean(leaderboard.get('leaderboard_json')) or clean(leaderboard.get('report'))}"
    body = leaderboard_entry_markdown(
        leaderboard=leaderboard,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        evaluation_note=evaluation_note,
        status=status,
        entry_date=entry_date or current_date_utc(),
        max_rows=max_rows,
    )
    return append_or_replace_block(ledger_path=ledger_path, entry_id=marker, body=body)


def h5_audit_entry_markdown(
    *,
    audit: Mapping[str, Any],
    status: str,
    entry_date: str,
) -> str:
    summary = audit.get("summary", {}) or {}
    outputs = audit.get("outputs", {}) or {}
    checks = list(audit.get("quality_checks", []) or [])
    target_counts = summary.get("target_label_counts", {}) or {}
    label_counts = summary.get("label_counts", {}) or {}
    lines = [
        f"### E126 SSL H5 Coverage Audit ({entry_date})",
        "",
        f"Status: {status}.",
        "",
        f"H5 dataset: `{clean(audit.get('input_h5'))}`",
        "",
        f"Builder summary: `{clean(audit.get('builder_summary_json'))}`" if clean(audit.get("builder_summary_json")) else "Builder summary: not provided.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| rows | {clean(summary.get('rows'))} |",
        f"| normal rows | {clean(summary.get('normal_rows'))} |",
        f"| normal train rows | {clean(summary.get('normal_train_rows'))} |",
        f"| normal months | {clean(summary.get('normal_months'))} |",
        f"| normal train months | {clean(summary.get('normal_train_months'))} |",
        f"| all months | {clean(summary.get('months'))} |",
        f"| unknown-month rows | {clean(summary.get('unknown_month_rows'))} |",
    ]
    if checks:
        lines.extend(["", "Quality checks:", "", "| Check | Value | Threshold | Passed |", "| --- | ---: | ---: | --- |"])
        for row in checks:
            lines.append(
                "| {check} | {value} | {threshold} | {passed} |".format(
                    check=clean(row.get("check")),
                    value=clean(row.get("value")),
                    threshold=clean(row.get("threshold")),
                    passed="yes" if row.get("passed") else "no",
                )
            )
    if target_counts:
        lines.extend(["", "Target rows:", "", "| Label | Rows |", "| --- | ---: |"])
        for label, value in sorted(target_counts.items()):
            lines.append(f"| {label} | {value} |")
    if label_counts:
        lines.extend(["", f"Label counts: `{json.dumps(label_counts, sort_keys=True)}`"])
    lines.extend(["", "Artifacts:"])
    for label, key in (
        ("Report", "report"),
        ("Summary JSON", "summary"),
        ("Quality checks CSV", "quality_checks"),
        ("Label counts CSV", "label_counts"),
        ("Split-label counts CSV", "split_label_counts"),
        ("Normal month counts CSV", "normal_month_counts"),
        ("Normal train month counts CSV", "normal_train_month_counts"),
    ):
        value = clean(outputs.get(key))
        if value:
            lines.append(f"- {label}: `{value}`")
    lines.extend(
        [
            "",
            "Interpretation: use this audit to verify the SSL normal/background phase has enough real temporal coverage before training or comparing SSL variants.",
        ]
    )
    return "\n".join(lines) + "\n"


def append_h5_audit_summary(
    *,
    audit: Mapping[str, Any],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    status: str = "completed",
    entry_id: str = "",
    entry_date: str = "",
) -> Path:
    marker = entry_id or f"h5-audit:{clean(audit.get('input_h5'))}:{clean(audit.get('outputs', {}).get('summary'))}"
    body = h5_audit_entry_markdown(
        audit=audit,
        status=status,
        entry_date=entry_date or current_date_utc(),
    )
    return append_or_replace_block(ledger_path=ledger_path, entry_id=marker, body=body)


def parse_artifact(value: str) -> Tuple[str, str]:
    text = clean(value)
    if "=" in text:
        label, path = text.split("=", 1)
        return clean(label) or "Artifact", clean(path)
    return "Artifact", text


def generic_note_entry_markdown(
    *,
    name: str,
    status: str,
    entry_date: str,
    training_set: str,
    validation_set: str,
    test_set: str,
    evaluation_note: str,
    metrics: Sequence[str],
    artifacts: Sequence[Tuple[str, str]],
    interpretation: str,
) -> str:
    interpretation_text = interpretation or "pending review"
    if interpretation_text and interpretation_text[-1] not in ".!?":
        interpretation_text += "."
    lines = [
        f"### {name} ({entry_date})",
        "",
        f"Status: {status}.",
        "",
        f"Training set: {training_set or 'not specified'}.",
        "",
        f"Validation set: {validation_set or 'not specified'}.",
        "",
        f"Test set: {test_set or 'not specified'}.",
        "",
        f"Evaluation: {evaluation_note or 'not specified'}.",
    ]
    if metrics:
        lines.extend(["", "Metrics:"])
        for metric in metrics:
            lines.append(f"- {metric}")
    else:
        lines.extend(["", "Metrics: not available."])
    if artifacts:
        lines.extend(["", "Artifacts:"])
        for label, path in artifacts:
            if path:
                lines.append(f"- {label}: `{path}`")
    else:
        lines.extend(["", "Artifacts: not specified."])
    lines.extend(["", f"Interpretation: {interpretation_text}"])
    return "\n".join(lines) + "\n"


def append_generic_note(
    *,
    name: str,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    training_set: str = "",
    validation_set: str = "",
    test_set: str = "",
    evaluation_note: str = "",
    metrics: Sequence[str] = (),
    artifacts: Sequence[Tuple[str, str]] = (),
    interpretation: str = "",
    status: str = "completed",
    entry_id: str = "",
    entry_date: str = "",
) -> Path:
    marker = entry_id or f"note:{name}:{entry_date or current_date_utc()}"
    body = generic_note_entry_markdown(
        name=name,
        status=status,
        entry_date=entry_date or current_date_utc(),
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        evaluation_note=evaluation_note,
        metrics=metrics,
        artifacts=artifacts,
        interpretation=interpretation,
    )
    return append_or_replace_block(ledger_path=ledger_path, entry_id=marker, body=body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("binary-gate", help="append an E126-style binary gate summary")
    gate.add_argument("--summary-json", required=True, type=Path)
    gate.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH, type=Path)
    gate.add_argument("--training-set", default="")
    gate.add_argument("--validation-set", default="")
    gate.add_argument("--test-set", default="")
    gate.add_argument("--evaluation-note", default="")
    gate.add_argument("--status", default="completed")
    gate.add_argument("--entry-id", default="")
    gate.add_argument("--entry-date", default="")

    leaderboard = subparsers.add_parser("leaderboard", help="append an E124-style candidate leaderboard")
    leaderboard.add_argument("--leaderboard-json", required=True, type=Path)
    leaderboard.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH, type=Path)
    leaderboard.add_argument("--training-set", default="")
    leaderboard.add_argument("--validation-set", default="")
    leaderboard.add_argument("--test-set", default="")
    leaderboard.add_argument("--evaluation-note", default="")
    leaderboard.add_argument("--status", default="completed")
    leaderboard.add_argument("--entry-id", default="")
    leaderboard.add_argument("--entry-date", default="")
    leaderboard.add_argument("--max-rows", type=int, default=5)

    h5_audit = subparsers.add_parser("h5-audit", help="append an E126 SSL H5 audit summary")
    h5_audit.add_argument("--audit-json", required=True, type=Path)
    h5_audit.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH, type=Path)
    h5_audit.add_argument("--status", default="completed")
    h5_audit.add_argument("--entry-id", default="")
    h5_audit.add_argument("--entry-date", default="")

    note = subparsers.add_parser("note", help="append a manual experiment note")
    note.add_argument("--name", required=True)
    note.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH, type=Path)
    note.add_argument("--training-set", required=True)
    note.add_argument("--validation-set", required=True)
    note.add_argument("--test-set", required=True)
    note.add_argument("--evaluation-note", required=True)
    note.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric line to include; repeat for precision/recall/F1/etc.",
    )
    note.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact path, optionally LABEL=PATH; repeat for reports/csvs/checkpoints.",
    )
    note.add_argument("--interpretation", default="")
    note.add_argument("--status", default="completed")
    note.add_argument("--entry-id", default="")
    note.add_argument("--entry-date", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "binary-gate":
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        ledger_path = append_binary_gate_summary(
            summary=summary,
            summary_path=args.summary_json,
            ledger_path=args.ledger_path,
            training_set=args.training_set,
            validation_set=args.validation_set,
            test_set=args.test_set,
            evaluation_note=args.evaluation_note,
            status=args.status,
            entry_id=args.entry_id,
            entry_date=args.entry_date,
        )
        print(json.dumps({"ledger": str(ledger_path)}, indent=2))
        return 0
    if args.command == "leaderboard":
        leaderboard = json.loads(args.leaderboard_json.read_text(encoding="utf-8"))
        leaderboard.setdefault("leaderboard_json", str(args.leaderboard_json))
        ledger_path = append_leaderboard_summary(
            leaderboard=leaderboard,
            ledger_path=args.ledger_path,
            training_set=args.training_set,
            validation_set=args.validation_set,
            test_set=args.test_set,
            evaluation_note=args.evaluation_note,
            status=args.status,
            entry_id=args.entry_id,
            entry_date=args.entry_date,
            max_rows=args.max_rows,
        )
        print(json.dumps({"ledger": str(ledger_path)}, indent=2))
        return 0
    if args.command == "h5-audit":
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        ledger_path = append_h5_audit_summary(
            audit=audit,
            ledger_path=args.ledger_path,
            status=args.status,
            entry_id=args.entry_id,
            entry_date=args.entry_date,
        )
        print(json.dumps({"ledger": str(ledger_path)}, indent=2))
        return 0
    if args.command == "note":
        artifacts: List[Tuple[str, str]] = [parse_artifact(value) for value in args.artifact]
        ledger_path = append_generic_note(
            name=args.name,
            ledger_path=args.ledger_path,
            training_set=args.training_set,
            validation_set=args.validation_set,
            test_set=args.test_set,
            evaluation_note=args.evaluation_note,
            metrics=args.metric,
            artifacts=artifacts,
            interpretation=args.interpretation,
            status=args.status,
            entry_id=args.entry_id,
            entry_date=args.entry_date,
        )
        print(json.dumps({"ledger": str(ledger_path)}, indent=2))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
