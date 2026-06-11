#!/usr/bin/env python3
"""Build a production-style leaderboard across multispecies candidate reports.

E119/E121/E122 reports already use the production-style species accounting we
want: cross-species mistakes are false positives for the predicted species and
false negatives for the true species. This helper collects those summaries, plus
E26 common-test diagnostics, into one compact ranking table so the next model
selection step is driven by the same metrics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PREFERRED_PREDICTIONS = {
    "E119": ("refined", "pred"),
    "E121": ("refined", "pred"),
    "E122": ("two_stage", "pred"),
    "E26": ("common_thresholds", "original_thresholds"),
    "unknown": ("refined", "two_stage", "common_thresholds", "pred", "original_thresholds"),
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = clean(value)
        return default if text == "" else float(text)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        text = clean(value)
        return default if text == "" else int(float(text))
    except (TypeError, ValueError):
        return default


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


def detect_experiment(summary_path: Path, payload: Mapping[str, Any]) -> str:
    name = summary_path.name.lower()
    if "original_summary" in payload and "common_summary" in payload:
        return "E26"
    if name == "diagnostic_summary.json":
        return "E26"
    if "gate_threshold" in payload:
        return "E122"
    if "pairwise_models" in payload:
        return "E121"
    if "pairwise_run_dir" in payload or "pairwise_labels" in payload:
        return "E119"
    return "unknown"


def infer_report_path(summary_path: Path, experiment: str, outputs: Mapping[str, Any]) -> str:
    explicit = clean(outputs.get("report"))
    if explicit:
        return explicit
    names = {
        "E119": "e119_pairwise_refinement_report.md",
        "E121": "e121_multi_pairwise_refinement_report.md",
        "E122": "e122_two_stage_gate_report.md",
        "E26": "e26_common_onc_test_diagnostics.md",
    }
    name = names.get(experiment)
    if name:
        return str(summary_path.parent / name)
    return ""


def normalize_metric(metric: Mapping[str, Any], *, e26: bool = False) -> Dict[str, Any]:
    background_fp = as_int(metric.get("background_fp", metric.get("hard_fp", 0)))
    if e26:
        total_fp = as_int(metric.get("fp", 0))
        cross_species_fp = max(total_fp - background_fp, 0)
        species_as_background_fn = as_int(metric.get("fn", 0))
        precision = as_float(metric.get("precision"))
        recall = as_float(metric.get("recall"))
        rows = as_int(metric.get("samples"))
        exact_accuracy = as_float(metric.get("exact_match_rate"))
    else:
        cross_species_fp = as_int(metric.get("cross_species_fp", 0))
        species_as_background_fn = as_int(metric.get("species_as_background_fn", 0))
        precision = as_float(metric.get("micro_precision", metric.get("precision")))
        recall = as_float(metric.get("micro_recall", metric.get("recall")))
        rows = as_int(metric.get("rows", metric.get("samples", 0)))
        exact_accuracy = as_float(metric.get("species_exact_accuracy", metric.get("exact_match_rate")))
    return {
        "rows": rows,
        "macro_f1": as_float(metric.get("macro_f1")),
        "micro_f1": as_float(metric.get("micro_f1")),
        "precision": precision,
        "recall": recall,
        "exact_accuracy": exact_accuracy,
        "cross_species_fp": cross_species_fp,
        "background_fp": background_fp,
        "species_as_background_fn": species_as_background_fn,
    }


def select_metric_row(
    rows: Sequence[Mapping[str, Any]],
    preferred: Sequence[str],
) -> Tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]:
    test_rows = [row for row in rows if clean(row.get("split")) in ("", "test")]
    if not test_rows:
        raise ValueError("summary has no test model_metrics rows")
    by_prediction = {clean(row.get("prediction")): row for row in test_rows}
    selected: Optional[Mapping[str, Any]] = None
    for prediction in preferred:
        if prediction in by_prediction:
            selected = by_prediction[prediction]
            break
    if selected is None:
        selected = max(
            test_rows,
            key=lambda row: (
                as_float(row.get("macro_f1")),
                as_float(row.get("micro_f1")),
                -as_int(row.get("cross_species_fp")),
                -as_int(row.get("background_fp")),
            ),
        )
    baseline = by_prediction.get("pred")
    if baseline is selected:
        baseline = None
    return selected, baseline


def e26_metric_rows(payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Optional[Mapping[str, Any]], str, str]:
    original = dict(payload.get("original_summary") or {})
    common = dict(payload.get("common_summary") or {})
    original["prediction"] = "original_thresholds"
    common["prediction"] = "common_thresholds"
    if common:
        return common, original or None, "common_thresholds", "original_thresholds"
    if original:
        return original, None, "original_thresholds", ""
    raise ValueError("E26 diagnostic summary lacks original_summary/common_summary")


def candidate_from_summary(summary_path: Path, *, alias: str = "") -> Dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment = detect_experiment(summary_path, payload)
    outputs = dict(payload.get("outputs") or {})
    candidate_name = alias or clean(payload.get("name")) or summary_path.parent.name

    if experiment == "E26":
        selected, baseline, selected_prediction, baseline_prediction = e26_metric_rows(payload)
        selected_metrics = normalize_metric(selected, e26=True)
        baseline_metrics = normalize_metric(baseline, e26=True) if baseline else None
        outputs = {
            "report": str(summary_path.parent / "e26_common_onc_test_diagnostics.md"),
            "metrics": "",
            "per_species": str(summary_path.parent / "per_species_metrics_common_thresholds.csv"),
            "examples": str(summary_path.parent / "selected_examples_original_thresholds.csv"),
            "threshold_sweep": "",
            "confusion": str(summary_path.parent / "confusion_counts_original_thresholds.csv"),
        }
    else:
        model_metrics = payload.get("model_metrics")
        if not isinstance(model_metrics, list):
            raise ValueError(f"{summary_path} is not an E119/E121/E122/E26 summary")
        selected, baseline = select_metric_row(model_metrics, PREFERRED_PREDICTIONS.get(experiment, PREFERRED_PREDICTIONS["unknown"]))
        selected_prediction = clean(selected.get("prediction"))
        baseline_prediction = clean(baseline.get("prediction")) if baseline else ""
        selected_metrics = normalize_metric(selected)
        baseline_metrics = normalize_metric(baseline) if baseline else None

    baseline_macro = "" if baseline_metrics is None else baseline_metrics["macro_f1"]
    baseline_micro = "" if baseline_metrics is None else baseline_metrics["micro_f1"]
    delta_macro = "" if baseline_metrics is None else selected_metrics["macro_f1"] - float(baseline_metrics["macro_f1"])
    delta_micro = "" if baseline_metrics is None else selected_metrics["micro_f1"] - float(baseline_metrics["micro_f1"])
    report_path = infer_report_path(summary_path, experiment, outputs)
    metric_labels = payload.get("metric_labels") or payload.get("base_class_ids") or []
    if not isinstance(metric_labels, list):
        metric_labels = []

    return {
        "rank": "",
        "candidate": candidate_name,
        "experiment": experiment,
        "summary_json": str(summary_path),
        "selected_prediction": selected_prediction,
        "baseline_prediction": baseline_prediction,
        "rows": selected_metrics["rows"],
        "metric_labels": "|".join(clean(label) for label in metric_labels if clean(label)),
        "macro_f1": selected_metrics["macro_f1"],
        "micro_f1": selected_metrics["micro_f1"],
        "precision": selected_metrics["precision"],
        "recall": selected_metrics["recall"],
        "exact_accuracy": selected_metrics["exact_accuracy"],
        "cross_species_fp": selected_metrics["cross_species_fp"],
        "background_fp": selected_metrics["background_fp"],
        "species_as_background_fn": selected_metrics["species_as_background_fn"],
        "baseline_macro_f1": baseline_macro,
        "delta_macro_f1": delta_macro,
        "baseline_micro_f1": baseline_micro,
        "delta_micro_f1": delta_micro,
        "threshold": payload.get("threshold", payload.get("gate_threshold", "")),
        "base_decision_mode": clean(payload.get("base_decision_mode")),
        "report": report_path,
        "metrics_csv": clean(outputs.get("metrics")),
        "per_species_csv": clean(outputs.get("per_species")),
        "examples_csv": clean(outputs.get("examples")),
        "threshold_sweep_csv": clean(outputs.get("threshold_sweep")),
        "confusion_csv": clean(outputs.get("confusion")),
        "comparability_note": "verify same common ONC test rows before comparing absolute ranks",
    }


def ranking_key(row: Mapping[str, Any]) -> Tuple[float, float, float, float, int, int, int]:
    return (
        as_float(row.get("macro_f1")),
        as_float(row.get("micro_f1")),
        as_float(row.get("precision")),
        as_float(row.get("recall")),
        -as_int(row.get("cross_species_fp")),
        -as_int(row.get("background_fp")),
        -as_int(row.get("species_as_background_fn")),
    )


def load_candidates(summary_paths: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for alias, path in summary_paths:
        try:
            rows.append(candidate_from_summary(path, alias=alias))
        except Exception as exc:  # noqa: BLE001 - collect all bad summaries for one useful error.
            errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError("failed to load summary JSON(s):\n" + "\n".join(errors))
    rows.sort(key=ranking_key, reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def markdown_report(rows: Sequence[Mapping[str, Any]], output_dir: Path, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "This ranks candidate multispecies systems using production-style test metrics.",
        "Cross-species predictions count against both the predicted species and the true species; background false positives are tracked separately.",
        "",
        "> Compare absolute ranks only when all source summaries were generated on the same common ONC test rows.",
        "",
        "## Leaderboard",
        "",
        "| rank | candidate | experiment | prediction | macro F1 | micro F1 | precision | recall | cross-species FP | background FP | species-as-background FN | delta macro F1 | report | examples |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        delta = clean(row.get("delta_macro_f1"))
        delta_text = "" if delta == "" else f"{as_float(delta):+.4f}"
        report = clean(row.get("report"))
        examples = clean(row.get("examples_csv"))
        lines.append(
            "| {rank} | {candidate} | {experiment} | {prediction} | {macro:.4f} | {micro:.4f} | {precision:.4f} | {recall:.4f} | {cross} | {bgfp} | {bgfn} | {delta} | {report} | {examples} |".format(
                rank=row["rank"],
                candidate=row["candidate"],
                experiment=row["experiment"],
                prediction=row["selected_prediction"],
                macro=as_float(row["macro_f1"]),
                micro=as_float(row["micro_f1"]),
                precision=as_float(row["precision"]),
                recall=as_float(row["recall"]),
                cross=row["cross_species_fp"],
                bgfp=row["background_fp"],
                bgfn=row["species_as_background_fn"],
                delta=delta_text,
                report=f"`{report}`" if report else "",
                examples=f"`{examples}`" if examples else "",
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"CSV: `{output_dir / 'e124_candidate_leaderboard.csv'}`",
            f"JSON: `{output_dir / 'e124_candidate_leaderboard.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def unique_paths(items: Iterable[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
    seen: set[Path] = set()
    out: List[Tuple[str, Path]] = []
    for alias, path in items:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((alias, resolved))
    return out


def collect_summary_paths(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for value in args.candidate or []:
        if "=" not in value:
            raise ValueError("--candidate must be NAME=PATH")
        alias, path_text = value.split("=", 1)
        items.append((alias.strip(), Path(path_text)))
    for path in args.summary_json or []:
        items.append(("", path))
    for pattern in args.summary_glob or []:
        for match in sorted(glob.glob(pattern, recursive=True)):
            items.append(("", Path(match)))
    items = unique_paths(items)
    missing = [str(path) for _, path in items if not path.is_file()]
    if missing:
        raise ValueError("summary JSON not found:\n" + "\n".join(missing))
    if not items:
        raise ValueError("provide at least one --summary-json, --summary-glob, or --candidate NAME=PATH")
    return items


def build_leaderboard(summary_paths: Sequence[Tuple[str, Path]], output_dir: Path, title: str) -> Dict[str, Any]:
    rows = load_candidates(summary_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "e124_candidate_leaderboard.csv", rows)
    (output_dir / "e124_candidate_leaderboard.json").write_text(json.dumps({"candidates": rows}, indent=2), encoding="utf-8")
    report = markdown_report(rows, output_dir, title)
    report_path = output_dir / "e124_candidate_leaderboard.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "leaderboard_csv": str(output_dir / "e124_candidate_leaderboard.csv"),
        "leaderboard_json": str(output_dir / "e124_candidate_leaderboard.json"),
        "candidates": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", action="append", type=Path, default=[], help="Path to an E119/E121/E122/E26 summary JSON")
    parser.add_argument("--summary-glob", action="append", default=[], help="Glob for summary JSONs; supports ** with recursive=True")
    parser.add_argument("--candidate", action="append", default=[], help="Named candidate in NAME=PATH form")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="E124 Production Candidate Leaderboard")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary_paths = collect_summary_paths(args)
        result = build_leaderboard(summary_paths, args.output_dir, args.title)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
