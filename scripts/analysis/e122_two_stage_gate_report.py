#!/usr/bin/env python3
"""Evaluate a binary whale-call gate followed by a species classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.analysis import e119_pairwise_refinement_report as e119


DEFAULT_GATE_LABEL = "task:whale_call"


def read_csv(path: Path) -> List[Dict[str, str]]:
    return e119.read_csv(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    e119.write_csv(path, rows)


def load_gate_scores(path: Path, gate_label: str) -> Dict[str, Dict[str, Any]]:
    scores: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(read_csv(path)):
        score = e119.probability(row, gate_label)
        if score is None:
            score = e119.probability(row, gate_label.replace(":", "_"))
        if score is None:
            continue
        key = e119.row_key(row, index)
        scores[key] = {"key": key, "score": float(score), "row": row}
    return scores


def species_stage_prediction(row: Mapping[str, Any], labels: Sequence[str], mode: str) -> str:
    base_pred = e119.clean(row.get("_pred")) or "background"
    if mode == "base_pred":
        return base_pred if base_pred in labels else "background"
    if mode != "force_species_argmax":
        raise ValueError(f"unknown species stage mode: {mode}")
    scored: List[Tuple[float, str]] = []
    for label in labels:
        score = e119.probability(row, label)
        if score is not None:
            scored.append((float(score), label))
    if scored:
        return max(scored, key=lambda item: item[0])[1]
    return base_pred if base_pred in labels else "background"


def apply_two_stage(
    rows: Sequence[Mapping[str, Any]],
    gate_scores: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    *,
    gate_threshold: float,
    species_stage_mode: str,
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        gate = gate_scores.get(e119.clean(row.get("_key")))
        gate_score = float(gate.get("score")) if gate is not None else 0.0
        gate_detected = gate is not None and gate_score >= float(gate_threshold)
        species_pred = species_stage_prediction(row, labels, species_stage_mode)
        out["gate_score"] = gate_score
        out["gate_threshold"] = float(gate_threshold)
        out["gate_detected"] = "1" if gate_detected else "0"
        out["species_stage_pred"] = species_pred
        out["_two_stage"] = species_pred if gate_detected and species_pred in labels else "background"
        if gate is None:
            out["two_stage_action"] = "missing_gate_score"
        elif not gate_detected:
            out["two_stage_action"] = "gate_rejected"
        elif out["_two_stage"] == "background":
            out["two_stage_action"] = "gate_detected_no_species"
        else:
            out["two_stage_action"] = "gate_detected_species"
        out_rows.append(out)
    return out_rows


def tune_gate_threshold(
    val_rows: Sequence[Mapping[str, Any]],
    gate_scores: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    *,
    species_stage_mode: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for threshold in [round(i / 100, 2) for i in range(0, 101, 5)]:
        two_stage = apply_two_stage(
            val_rows,
            gate_scores,
            labels,
            gate_threshold=threshold,
            species_stage_mode=species_stage_mode,
        )
        metrics = e119.species_metrics(two_stage, "_two_stage", labels)
        rows.append(
            {
                "gate_threshold": threshold,
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "cross_species_fp": metrics["cross_species_fp"],
                "background_fp": metrics["background_fp"],
                "species_as_background_fn": metrics["species_as_background_fn"],
                "gate_detected": sum(1 for row in two_stage if row.get("gate_detected") == "1"),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["micro_f1"]),
            -int(row["background_fp"]),
            -int(row["cross_species_fp"]),
            float(row["gate_threshold"]),
        ),
    )
    return float(best["gate_threshold"]), rows


def example_rows(rows: Sequence[Mapping[str, Any]], limit_per_bucket: int = 50) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {
        "gate_suppressed_background_fp": [],
        "residual_background_fp": [],
        "gate_missed_species": [],
        "cross_species_error": [],
        "correct_species": [],
    }
    for row in rows:
        true = e119.clean(row.get("_true"))
        base = e119.clean(row.get("_pred"))
        two = e119.clean(row.get("_two_stage"))
        if true == "background" and base != "background" and two == "background":
            buckets["gate_suppressed_background_fp"].append(row)
        elif true == "background" and two != "background":
            buckets["residual_background_fp"].append(row)
        elif true != "background" and two == "background":
            buckets["gate_missed_species"].append(row)
        elif true != "background" and two != true:
            buckets["cross_species_error"].append(row)
        elif true != "background" and two == true:
            buckets["correct_species"].append(row)
    examples: List[Dict[str, Any]] = []
    for bucket, bucket_rows in buckets.items():
        ordered = sorted(bucket_rows, key=lambda row: float(row.get("gate_score") or 0.0), reverse=True)
        for row in ordered[:limit_per_bucket]:
            examples.append(
                {
                    "bucket": bucket,
                    "item_id": e119.clean(row.get("item_id")) or e119.clean(row.get("_key")),
                    "true": e119.clean(row.get("_true")),
                    "base_pred": e119.clean(row.get("_pred")),
                    "two_stage_pred": e119.clean(row.get("_two_stage")),
                    "species_stage_pred": e119.clean(row.get("species_stage_pred")),
                    "gate_score": row.get("gate_score", ""),
                    "gate_threshold": row.get("gate_threshold", ""),
                    "gate_detected": row.get("gate_detected", ""),
                    "clip": e119.clean(row.get("clip")) or e119.clean(row.get("filename")) or e119.clean(row.get("source_audio")),
                    "begin_s": e119.clean(row.get("begin_s")) or e119.clean(row.get("begin_time_s")),
                    "end_s": e119.clean(row.get("end_s")) or e119.clean(row.get("end_time_s")),
                    "mat_path": e119.clean(row.get("mat_path")) or e119.clean(row.get("low_mat_path")),
                }
            )
    return examples


def metric_row(name: str, split: str, pred_field: str, rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    return e119.metric_row(name, split, pred_field, rows, labels)


def markdown_report(
    *,
    name: str,
    output_dir: Path,
    gate_threshold: float,
    species_stage_mode: str,
    model_metrics: Sequence[Mapping[str, Any]],
    per_species_rows: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# E122 Two-Stage Gate Report: {name}",
        "",
        "This evaluates a binary whale-call detector as stage 1 and a species classifier as stage 2.",
        "",
        f"Validation-tuned gate threshold: `{gate_threshold:.2f}`.",
        f"Species stage mode: `{species_stage_mode}`.",
        "",
        "## Overall Metrics",
        "",
        "| split | prediction | macro F1 | micro F1 | precision | recall | cross-species FP | background FP | species-as-background FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model_metrics:
        lines.append(
            "| {split} | {pred} | {macro:.4f} | {micro:.4f} | {precision:.4f} | {recall:.4f} | {cross} | {bgfp} | {bgfn} |".format(
                split=row["split"],
                pred=row["prediction"],
                macro=float(row["macro_f1"]),
                micro=float(row["micro_f1"]),
                precision=float(row["micro_precision"]),
                recall=float(row["micro_recall"]),
                cross=row["cross_species_fp"],
                bgfp=row["background_fp"],
                bgfn=row["species_as_background_fn"],
            )
        )
    lines.extend(
        [
            "",
            "## Test Per-Species Metrics",
            "",
            "| prediction | species | support | predicted | precision | recall | F1 | FP | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_species_rows:
        lines.append(
            "| {prediction} | {name} | {support} | {predicted} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {fp} | {fn} |".format(
                prediction=row["prediction"],
                name=row["name"],
                support=row["support"],
                predicted=row["predicted"],
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                fp=row["fp"],
                fn=row["fn"],
            )
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
            f"Metrics CSV: `{output_dir / 'e122_model_metrics.csv'}`",
            f"Per-species CSV: `{output_dir / 'e122_per_species_metrics.csv'}`",
            f"Examples CSV: `{output_dir / 'e122_examples.csv'}`",
            f"Threshold sweep CSV: `{output_dir / 'e122_gate_threshold_sweep.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report(
    *,
    parser: argparse.ArgumentParser,
    name: str,
    output_dir: Path,
    base_run_dir: Optional[Path],
    gate_run_dir: Optional[Path],
    base_val_predictions: Optional[Path],
    base_test_predictions: Optional[Path],
    gate_val_predictions: Optional[Path],
    gate_test_predictions: Optional[Path],
    base_summary_json: Optional[Path],
    gate_label: str,
    species_stage_mode: str,
    base_decision_mode: str,
    base_calibration_threshold_grid: str,
    base_calibration_margin_grid: str,
    base_calibration_bias_grid: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_summary_json = base_summary_json or e119.discover_summary(base_run_dir)
    base_val_predictions = base_val_predictions or (
        e119.discover_predictions(base_run_dir, "val", prefer_rule=True) if base_run_dir else None
    )
    base_test_predictions = base_test_predictions or (
        e119.discover_predictions(base_run_dir, "test", prefer_rule=True) if base_run_dir else None
    )
    gate_val_predictions = gate_val_predictions or (
        e119.discover_predictions(gate_run_dir, "val", prefer_rule=False) if gate_run_dir else None
    )
    gate_test_predictions = gate_test_predictions or (
        e119.discover_predictions(gate_run_dir, "test", prefer_rule=False) if gate_run_dir else None
    )
    base_val_predictions = e119.require_path(parser, base_val_predictions, "base validation predictions")
    base_test_predictions = e119.require_path(parser, base_test_predictions, "base test predictions")
    gate_val_predictions = e119.require_path(parser, gate_val_predictions, "gate validation predictions")
    gate_test_predictions = e119.require_path(parser, gate_test_predictions, "gate test predictions")

    base_class_ids = e119.load_class_ids(base_summary_json, ("background", "species:Bp", "species:Bm", "species:Mn"))
    metric_labels = e119.species_class_ids(base_class_ids)
    base_val = e119.load_predictions(base_val_predictions, base_class_ids)
    base_test = e119.load_predictions(base_test_predictions, base_class_ids)
    gate_val = load_gate_scores(gate_val_predictions, gate_label)
    gate_test = load_gate_scores(gate_test_predictions, gate_label)

    base_rule: Optional[Dict[str, Any]] = None
    base_rule_sweep: List[Dict[str, Any]] = []
    if base_decision_mode == "calibrated":
        base_rule, base_rule_sweep = e119.tune_base_rule(
            base_val,
            metric_labels,
            thresholds=e119.parse_float_grid(base_calibration_threshold_grid),
            margins=e119.parse_float_grid(base_calibration_margin_grid),
            biases=e119.parse_float_grid(base_calibration_bias_grid),
        )
        base_val = e119.apply_base_rule(base_val, metric_labels, base_rule)
        base_test = e119.apply_base_rule(base_test, metric_labels, base_rule)

    gate_threshold, threshold_rows = tune_gate_threshold(
        base_val,
        gate_val,
        metric_labels,
        species_stage_mode=species_stage_mode,
    )
    two_stage_val = apply_two_stage(
        base_val,
        gate_val,
        metric_labels,
        gate_threshold=gate_threshold,
        species_stage_mode=species_stage_mode,
    )
    two_stage_test = apply_two_stage(
        base_test,
        gate_test,
        metric_labels,
        gate_threshold=gate_threshold,
        species_stage_mode=species_stage_mode,
    )
    model_metrics = [
        metric_row(name, "val", "_pred", base_val, metric_labels),
        metric_row(name, "val", "_two_stage", two_stage_val, metric_labels),
        metric_row(name, "test", "_pred", base_test, metric_labels),
        metric_row(name, "test", "_two_stage", two_stage_test, metric_labels),
    ]
    per_species_rows: List[Dict[str, Any]] = []
    for prediction, rows, field in (("base", base_test, "_pred"), ("two_stage", two_stage_test, "_two_stage")):
        metrics = e119.species_metrics(rows, field, metric_labels)
        for per_class in metrics["per_class"]:
            per_species_rows.append({"prediction": prediction, **per_class})
    examples = example_rows(two_stage_test)
    confusion = e119.confusion_rows(two_stage_test, "_two_stage", base_class_ids)

    write_csv(output_dir / "e122_model_metrics.csv", model_metrics)
    write_csv(output_dir / "e122_per_species_metrics.csv", per_species_rows)
    write_csv(output_dir / "e122_examples.csv", examples)
    write_csv(output_dir / "e122_gate_threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "e122_two_stage_confusion.csv", confusion)
    if base_rule_sweep:
        write_csv(output_dir / "e122_base_calibration_sweep.csv", base_rule_sweep)
    report_path = output_dir / "e122_two_stage_gate_report.md"
    report_path.write_text(
        markdown_report(
            name=name,
            output_dir=output_dir,
            gate_threshold=gate_threshold,
            species_stage_mode=species_stage_mode,
            model_metrics=model_metrics,
            per_species_rows=per_species_rows,
            examples=examples,
        ),
        encoding="utf-8",
    )
    summary = {
        "name": name,
        "gate_label": gate_label,
        "gate_threshold": gate_threshold,
        "species_stage_mode": species_stage_mode,
        "base_decision_mode": base_decision_mode,
        "base_rule": base_rule or {},
        "base_run_dir": "" if base_run_dir is None else str(base_run_dir),
        "gate_run_dir": "" if gate_run_dir is None else str(gate_run_dir),
        "base_class_ids": base_class_ids,
        "metric_labels": metric_labels,
        "inputs": {
            "base_val_predictions": str(base_val_predictions),
            "base_test_predictions": str(base_test_predictions),
            "gate_val_predictions": str(gate_val_predictions),
            "gate_test_predictions": str(gate_test_predictions),
            "base_summary_json": "" if base_summary_json is None else str(base_summary_json),
        },
        "model_metrics": model_metrics,
        "outputs": {
            "report": str(report_path),
            "metrics": str(output_dir / "e122_model_metrics.csv"),
            "per_species": str(output_dir / "e122_per_species_metrics.csv"),
            "examples": str(output_dir / "e122_examples.csv"),
            "threshold_sweep": str(output_dir / "e122_gate_threshold_sweep.csv"),
            "confusion": str(output_dir / "e122_two_stage_confusion.csv"),
            "base_calibration_sweep": "" if not base_rule_sweep else str(output_dir / "e122_base_calibration_sweep.csv"),
        },
    }
    (output_dir / "e122_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**summary, "report": str(report_path)}


def comparison_report(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# E122 Multi-Base Two-Stage Comparison",
        "",
        "| rank | base | two-stage macro F1 | base macro F1 | delta | two-stage micro F1 | background FP | report |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {base} | {two_macro:.4f} | {base_macro:.4f} | {delta:.4f} | {two_micro:.4f} | {bgfp} | {report} |".format(
                rank=rank,
                base=row["base_name"],
                two_macro=float(row["two_stage_test_macro_f1"]),
                base_macro=float(row["base_test_macro_f1"]),
                delta=float(row["delta_test_macro_f1"]),
                two_micro=float(row["two_stage_test_micro_f1"]),
                bgfp=row["two_stage_test_background_fp"],
                report=row["report"],
            )
        )
    lines.extend(["", f"Rankings CSV: `{output_dir / 'e122_comparison_rankings.csv'}`"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-run-dir", type=Path, action="append", default=None)
    parser.add_argument("--gate-run-dir", type=Path, default=None)
    parser.add_argument("--base-val-predictions", type=Path, default=None)
    parser.add_argument("--base-test-predictions", type=Path, default=None)
    parser.add_argument("--gate-val-predictions", type=Path, default=None)
    parser.add_argument("--gate-test-predictions", type=Path, default=None)
    parser.add_argument("--base-summary-json", type=Path, default=None)
    parser.add_argument("--gate-label", default=DEFAULT_GATE_LABEL)
    parser.add_argument("--species-stage-mode", default="force_species_argmax", choices=["base_pred", "force_species_argmax"])
    parser.add_argument("--base-decision-mode", default="calibrated", choices=["existing", "calibrated"])
    parser.add_argument(
        "--base-calibration-threshold-grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--base-calibration-margin-grid", default="-0.25,0.0,0.25")
    parser.add_argument("--base-calibration-bias-grid", default="-0.30,-0.15,0.0,0.15,0.30")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    base_run_dirs = list(args.base_run_dir or [])
    if len(base_run_dirs) > 1 and (args.base_val_predictions or args.base_test_predictions or args.base_summary_json):
        parser.error("explicit base prediction/summary paths cannot be combined with multiple --base-run-dir values")
    if len(base_run_dirs) > 1:
        comparison_rows: List[Dict[str, Any]] = []
        for base_run_dir in base_run_dirs:
            base_name = e119.safe_stem(base_run_dir)
            summary = run_report(
                parser=parser,
                name=f"{args.name}__{base_name}",
                output_dir=args.output_dir / base_name,
                base_run_dir=base_run_dir,
                gate_run_dir=args.gate_run_dir,
                base_val_predictions=None,
                base_test_predictions=None,
                gate_val_predictions=args.gate_val_predictions,
                gate_test_predictions=args.gate_test_predictions,
                base_summary_json=None,
                gate_label=args.gate_label,
                species_stage_mode=args.species_stage_mode,
                base_decision_mode=args.base_decision_mode,
                base_calibration_threshold_grid=args.base_calibration_threshold_grid,
                base_calibration_margin_grid=args.base_calibration_margin_grid,
                base_calibration_bias_grid=args.base_calibration_bias_grid,
            )
            base_metric = e119.select_metric(summary["model_metrics"], "test", "pred")
            two_metric = e119.select_metric(summary["model_metrics"], "test", "two_stage")
            comparison_rows.append(
                {
                    "base_name": base_name,
                    "base_run_dir": str(base_run_dir),
                    "report": summary["report"],
                    "base_test_macro_f1": base_metric["macro_f1"],
                    "base_test_micro_f1": base_metric["micro_f1"],
                    "base_test_background_fp": base_metric["background_fp"],
                    "two_stage_test_macro_f1": two_metric["macro_f1"],
                    "two_stage_test_micro_f1": two_metric["micro_f1"],
                    "two_stage_test_background_fp": two_metric["background_fp"],
                    "delta_test_macro_f1": float(two_metric["macro_f1"]) - float(base_metric["macro_f1"]),
                    "gate_threshold": summary["gate_threshold"],
                    "base_rule": json.dumps(summary.get("base_rule", {}), sort_keys=True),
                }
            )
        comparison_rows = sorted(
            comparison_rows,
            key=lambda row: (
                float(row["two_stage_test_macro_f1"]),
                float(row["two_stage_test_micro_f1"]),
                -int(row["two_stage_test_background_fp"]),
                float(row["base_test_macro_f1"]),
            ),
            reverse=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "e122_comparison_rankings.csv", comparison_rows)
        report_path = args.output_dir / "e122_comparison_report.md"
        report_path.write_text(comparison_report(args.output_dir, comparison_rows), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "rankings": str(args.output_dir / "e122_comparison_rankings.csv"), "rows": comparison_rows}, indent=2))
        return 0

    summary = run_report(
        parser=parser,
        name=args.name,
        output_dir=args.output_dir,
        base_run_dir=base_run_dirs[0] if base_run_dirs else None,
        gate_run_dir=args.gate_run_dir,
        base_val_predictions=args.base_val_predictions,
        base_test_predictions=args.base_test_predictions,
        gate_val_predictions=args.gate_val_predictions,
        gate_test_predictions=args.gate_test_predictions,
        base_summary_json=args.base_summary_json,
        gate_label=args.gate_label,
        species_stage_mode=args.species_stage_mode,
        base_decision_mode=args.base_decision_mode,
        base_calibration_threshold_grid=args.base_calibration_threshold_grid,
        base_calibration_margin_grid=args.base_calibration_margin_grid,
        base_calibration_bias_grid=args.base_calibration_bias_grid,
    )
    print(json.dumps({"report": summary["report"], "summary": summary["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
