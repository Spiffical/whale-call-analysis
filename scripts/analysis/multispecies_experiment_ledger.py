#!/usr/bin/env python3
"""Append concise experiment results to the multispecies living ledger."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


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
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
