#!/usr/bin/env python3
"""Evaluate multiple pairwise species specialists as one refinement layer.

This combines E118/E120-style one-vs-one specialists on top of a multiclass
base model. A validation sweep tunes one shared "flip only when confident"
margin threshold; test metrics then use the same production-style accounting as
E119, where cross-species mistakes count as a false positive for the predicted
species and a false negative for the true species.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.analysis import e119_pairwise_refinement_report as e119


def safe_stem(path: Path) -> str:
    return e119.safe_stem(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    e119.write_csv(path, rows)


def select_metric(rows: Sequence[Mapping[str, Any]], split: str, prediction: str) -> Mapping[str, Any]:
    return e119.select_metric(rows, split, prediction)


def load_pairwise_models(
    parser: argparse.ArgumentParser,
    pairwise_run_dirs: Sequence[Path],
) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str]] = set()
    for run_dir in pairwise_run_dirs:
        summary_json = e119.discover_summary(run_dir)
        val_predictions = e119.discover_predictions(run_dir, "val", prefer_rule=False)
        test_predictions = e119.discover_predictions(run_dir, "test", prefer_rule=False)
        val_predictions = e119.require_path(parser, val_predictions, f"validation predictions for {run_dir}")
        test_predictions = e119.require_path(parser, test_predictions, f"test predictions for {run_dir}")
        class_ids = e119.load_class_ids(summary_json, ("background", "species:Bp", "species:Mn"))
        labels = e119.require_pairwise_labels(parser, class_ids)
        pair_key = tuple(sorted(labels))
        if pair_key in seen_pairs:
            parser.error(f"duplicate pairwise specialist for labels {pair_key}")
        seen_pairs.add(pair_key)
        models.append(
            {
                "name": safe_stem(run_dir),
                "run_dir": str(run_dir),
                "summary_json": "" if summary_json is None else str(summary_json),
                "val_predictions": str(val_predictions),
                "test_predictions": str(test_predictions),
                "class_ids": class_ids,
                "labels": labels,
                "val": e119.load_pairwise(val_predictions, class_ids, labels),
                "test": e119.load_pairwise(test_predictions, class_ids, labels),
            }
        )
    if not models:
        parser.error("provide at least one --pairwise-run-dir")
    return models


def candidate_for_row(row: Mapping[str, Any], models: Sequence[Mapping[str, Any]], split: str) -> Optional[Dict[str, Any]]:
    base_pred = e119.clean(row.get("_pred")) or "background"
    key = e119.clean(row.get("_key"))
    best: Optional[Dict[str, Any]] = None
    for model in models:
        labels = tuple(model["labels"])
        if base_pred not in labels:
            continue
        pair = model[split].get(key)
        if pair is None:
            continue
        pair_pred = e119.clean(pair.get("pairwise_pred"))
        margin = float(pair.get("pairwise_margin") or 0.0)
        candidate = {
            "model_name": model["name"],
            "labels": labels,
            "pair": pair,
            "pairwise_pred": pair_pred,
            "margin": margin,
            "is_flip": pair_pred in labels and pair_pred != base_pred,
        }
        if best is None or margin > float(best["margin"]):
            best = candidate
    return best


def apply_multi_refinement(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    *,
    split: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    refined: List[Dict[str, Any]] = []
    species_labels = {label for model in models for label in model["labels"]}
    for row in rows:
        out = dict(row)
        base_pred = e119.clean(row.get("_pred")) or "background"
        out["_refined"] = base_pred
        out["refinement_action"] = "not_species_candidate" if base_pred not in species_labels else "no_pairwise_prediction"
        candidate = candidate_for_row(row, models, split)
        if candidate is not None:
            pair = candidate["pair"]
            labels = tuple(candidate["labels"])
            out["pairwise_model"] = candidate["model_name"]
            out["pairwise_labels"] = "|".join(labels)
            out["pairwise_pred"] = candidate["pairwise_pred"]
            out["pairwise_margin"] = candidate["margin"]
            for label in labels:
                out[f"pairwise_{e119.pairwise_prob_field(label)}"] = pair.get(e119.pairwise_prob_field(label))
            if bool(candidate["is_flip"]) and float(candidate["margin"]) >= threshold:
                out["_refined"] = candidate["pairwise_pred"]
                out["refinement_action"] = "flipped"
            else:
                out["refinement_action"] = "kept_candidate"
        refined.append(out)
    return refined


def tune_threshold(
    val_rows: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for threshold in [round(i / 100, 2) for i in range(0, 101, 5)]:
        refined = apply_multi_refinement(val_rows, models, split="val", threshold=threshold)
        metrics = e119.species_metrics(refined, "_refined", labels)
        rows.append(
            {
                "threshold": threshold,
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "cross_species_fp": metrics["cross_species_fp"],
                "background_fp": metrics["background_fp"],
                "species_as_background_fn": metrics["species_as_background_fn"],
                "flipped": sum(1 for row in refined if row.get("refinement_action") == "flipped"),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["micro_f1"]),
            -int(row["cross_species_fp"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), rows


def coverage_rows(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model in models:
        labels = tuple(model["labels"])
        label_set = set(labels)
        pairwise = model[split]
        with_prediction = candidates = disagreements = 0
        for row in rows:
            key = e119.clean(row.get("_key"))
            base_pred = e119.clean(row.get("_pred")) or "background"
            pair = pairwise.get(key)
            if pair is None:
                continue
            with_prediction += 1
            if base_pred not in label_set:
                continue
            candidates += 1
            pair_pred = e119.clean(pair.get("pairwise_pred"))
            if pair_pred in label_set and pair_pred != base_pred:
                disagreements += 1
        out.append(
            {
                "split": split,
                "pairwise_model": model["name"],
                "pairwise_labels": "|".join(labels),
                "rows": len(rows),
                "rows_with_pairwise_prediction": with_prediction,
                "base_candidate_rows": candidates,
                "pairwise_disagreements": disagreements,
            }
        )
    return out


def metric_row(name: str, split: str, pred_field: str, rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    return e119.metric_row(name, split, pred_field, rows, labels)


def markdown_report(
    *,
    name: str,
    output_dir: Path,
    threshold: float,
    pairwise_models: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    per_species_rows: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> str:
    model_text = ", ".join(f"{model['name']} ({' vs '.join(model['labels'])})" for model in pairwise_models)
    lines = [
        f"# E121 Multi-Pairwise Refinement Report: {name}",
        "",
        "This evaluates multiple pairwise specialists as a single conservative refinement layer on top of a multiclass base model.",
        "",
        f"Pairwise specialists: {model_text}.",
        f"Validation-tuned flip threshold: `{threshold:.2f}`.",
        "",
        "## Overall Metrics",
        "",
        "| split | prediction | macro F1 | micro F1 | precision | recall | cross-species FP | background FP | species-as-background FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
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
    lines.extend(
        [
            "",
            "## Pairwise Coverage",
            "",
            "| split | specialist | labels | rows with pairwise prediction | candidate rows | disagreements |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in coverage:
        lines.append(
            "| {split} | {model} | {labels} | {seen} | {candidates} | {disagree} |".format(
                split=row["split"],
                model=row["pairwise_model"],
                labels=row["pairwise_labels"],
                seen=row["rows_with_pairwise_prediction"],
                candidates=row["base_candidate_rows"],
                disagree=row["pairwise_disagreements"],
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
            f"Metrics CSV: `{output_dir / 'e121_model_metrics.csv'}`",
            f"Per-species CSV: `{output_dir / 'e121_per_species_metrics.csv'}`",
            f"Coverage CSV: `{output_dir / 'e121_pairwise_coverage.csv'}`",
            f"Examples CSV: `{output_dir / 'e121_examples.csv'}`",
            f"Threshold sweep CSV: `{output_dir / 'e121_threshold_sweep.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report(
    *,
    parser: argparse.ArgumentParser,
    name: str,
    output_dir: Path,
    base_run_dir: Optional[Path],
    pairwise_run_dirs: Sequence[Path],
    base_val_predictions: Optional[Path],
    base_test_predictions: Optional[Path],
    base_summary_json: Optional[Path],
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
    base_val_predictions = e119.require_path(parser, base_val_predictions, "base validation predictions")
    base_test_predictions = e119.require_path(parser, base_test_predictions, "base test predictions")

    base_class_ids = e119.load_class_ids(base_summary_json, ("background", "species:Bp", "species:Bm", "species:Mn"))
    metric_labels = e119.species_class_ids(base_class_ids)
    base_val = e119.load_predictions(base_val_predictions, base_class_ids)
    base_test = e119.load_predictions(base_test_predictions, base_class_ids)
    pairwise_models = load_pairwise_models(parser, pairwise_run_dirs)

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

    threshold, threshold_rows = tune_threshold(base_val, pairwise_models, metric_labels)
    refined_val = apply_multi_refinement(base_val, pairwise_models, split="val", threshold=threshold)
    refined_test = apply_multi_refinement(base_test, pairwise_models, split="test", threshold=threshold)

    model_metrics = [
        metric_row(name, "val", "_pred", base_val, metric_labels),
        metric_row(name, "val", "_refined", refined_val, metric_labels),
        metric_row(name, "test", "_pred", base_test, metric_labels),
        metric_row(name, "test", "_refined", refined_test, metric_labels),
    ]
    per_species_rows: List[Dict[str, Any]] = []
    for prediction, rows, field in (("base", base_test, "_pred"), ("refined", refined_test, "_refined")):
        metrics = e119.species_metrics(rows, field, metric_labels)
        for per_class in metrics["per_class"]:
            per_species_rows.append({"prediction": prediction, **per_class})
    coverage = [
        *coverage_rows(base_val, pairwise_models, split="val"),
        *coverage_rows(base_test, pairwise_models, split="test"),
    ]
    examples = e119.example_rows(refined_test)
    confusion = e119.confusion_rows(refined_test, "_refined", base_class_ids)

    write_csv(output_dir / "e121_model_metrics.csv", model_metrics)
    write_csv(output_dir / "e121_per_species_metrics.csv", per_species_rows)
    write_csv(output_dir / "e121_pairwise_coverage.csv", coverage)
    write_csv(output_dir / "e121_examples.csv", examples)
    write_csv(output_dir / "e121_threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "e121_refined_confusion.csv", confusion)
    if base_rule_sweep:
        write_csv(output_dir / "e121_base_calibration_sweep.csv", base_rule_sweep)
    report_path = output_dir / "e121_multi_pairwise_refinement_report.md"
    report = markdown_report(
        name=name,
        output_dir=output_dir,
        threshold=threshold,
        pairwise_models=pairwise_models,
        metric_rows=model_metrics,
        per_species_rows=per_species_rows,
        coverage=coverage,
        examples=examples,
    )
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "name": name,
        "threshold": threshold,
        "base_decision_mode": base_decision_mode,
        "base_rule": base_rule or {},
        "base_run_dir": "" if base_run_dir is None else str(base_run_dir),
        "base_class_ids": base_class_ids,
        "metric_labels": metric_labels,
        "pairwise_models": [
            {
                "name": model["name"],
                "run_dir": model["run_dir"],
                "labels": list(model["labels"]),
                "summary_json": model["summary_json"],
                "val_predictions": model["val_predictions"],
                "test_predictions": model["test_predictions"],
            }
            for model in pairwise_models
        ],
        "inputs": {
            "base_val_predictions": str(base_val_predictions),
            "base_test_predictions": str(base_test_predictions),
            "base_summary_json": "" if base_summary_json is None else str(base_summary_json),
        },
        "model_metrics": model_metrics,
        "outputs": {
            "report": str(report_path),
            "metrics": str(output_dir / "e121_model_metrics.csv"),
            "per_species": str(output_dir / "e121_per_species_metrics.csv"),
            "coverage": str(output_dir / "e121_pairwise_coverage.csv"),
            "examples": str(output_dir / "e121_examples.csv"),
            "threshold_sweep": str(output_dir / "e121_threshold_sweep.csv"),
            "confusion": str(output_dir / "e121_refined_confusion.csv"),
            "base_calibration_sweep": "" if not base_rule_sweep else str(output_dir / "e121_base_calibration_sweep.csv"),
        },
    }
    (output_dir / "e121_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**summary, "report": str(report_path)}


def comparison_report(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# E121 Multi-Base Multi-Pairwise Refinement Comparison",
        "",
        "| rank | base | refined macro F1 | base macro F1 | delta | refined micro F1 | cross-species FP | report |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {base} | {ref_macro:.4f} | {base_macro:.4f} | {delta:.4f} | {ref_micro:.4f} | {cross} | {report} |".format(
                rank=rank,
                base=row["base_name"],
                ref_macro=float(row["refined_test_macro_f1"]),
                base_macro=float(row["base_test_macro_f1"]),
                delta=float(row["delta_test_macro_f1"]),
                ref_micro=float(row["refined_test_micro_f1"]),
                cross=row["refined_test_cross_species_fp"],
                report=row["report"],
            )
        )
    lines.extend(["", f"Rankings CSV: `{output_dir / 'e121_comparison_rankings.csv'}`"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-run-dir", type=Path, action="append", default=None)
    parser.add_argument("--pairwise-run-dir", type=Path, action="append", default=None)
    parser.add_argument("--base-val-predictions", type=Path, default=None)
    parser.add_argument("--base-test-predictions", type=Path, default=None)
    parser.add_argument("--base-summary-json", type=Path, default=None)
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
    pairwise_run_dirs = list(args.pairwise_run_dir or [])
    if not pairwise_run_dirs:
        parser.error("provide at least one --pairwise-run-dir")
    if len(base_run_dirs) > 1 and (args.base_val_predictions or args.base_test_predictions or args.base_summary_json):
        parser.error("explicit base prediction/summary paths cannot be combined with multiple --base-run-dir values")

    if len(base_run_dirs) > 1:
        comparison_rows: List[Dict[str, Any]] = []
        for base_run_dir in base_run_dirs:
            base_name = safe_stem(base_run_dir)
            summary = run_report(
                parser=parser,
                name=f"{args.name}__{base_name}",
                output_dir=args.output_dir / base_name,
                base_run_dir=base_run_dir,
                pairwise_run_dirs=pairwise_run_dirs,
                base_val_predictions=None,
                base_test_predictions=None,
                base_summary_json=None,
                base_decision_mode=args.base_decision_mode,
                base_calibration_threshold_grid=args.base_calibration_threshold_grid,
                base_calibration_margin_grid=args.base_calibration_margin_grid,
                base_calibration_bias_grid=args.base_calibration_bias_grid,
            )
            base_metric = select_metric(summary["model_metrics"], "test", "pred")
            refined_metric = select_metric(summary["model_metrics"], "test", "refined")
            comparison_rows.append(
                {
                    "base_name": base_name,
                    "base_run_dir": str(base_run_dir),
                    "report": summary["report"],
                    "base_test_macro_f1": base_metric["macro_f1"],
                    "base_test_micro_f1": base_metric["micro_f1"],
                    "base_test_cross_species_fp": base_metric["cross_species_fp"],
                    "refined_test_macro_f1": refined_metric["macro_f1"],
                    "refined_test_micro_f1": refined_metric["micro_f1"],
                    "refined_test_cross_species_fp": refined_metric["cross_species_fp"],
                    "delta_test_macro_f1": float(refined_metric["macro_f1"]) - float(base_metric["macro_f1"]),
                    "threshold": summary["threshold"],
                    "base_rule": json.dumps(summary.get("base_rule", {}), sort_keys=True),
                }
            )
        comparison_rows = sorted(
            comparison_rows,
            key=lambda row: (
                float(row["refined_test_macro_f1"]),
                float(row["refined_test_micro_f1"]),
                float(row["base_test_macro_f1"]),
                -int(row["refined_test_cross_species_fp"]),
            ),
            reverse=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "e121_comparison_rankings.csv", comparison_rows)
        report_path = args.output_dir / "e121_comparison_report.md"
        report_path.write_text(comparison_report(args.output_dir, comparison_rows), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "rankings": str(args.output_dir / "e121_comparison_rankings.csv"), "rows": comparison_rows}, indent=2))
        return 0

    summary = run_report(
        parser=parser,
        name=args.name,
        output_dir=args.output_dir,
        base_run_dir=base_run_dirs[0] if base_run_dirs else None,
        pairwise_run_dirs=pairwise_run_dirs,
        base_val_predictions=args.base_val_predictions,
        base_test_predictions=args.base_test_predictions,
        base_summary_json=args.base_summary_json,
        base_decision_mode=args.base_decision_mode,
        base_calibration_threshold_grid=args.base_calibration_threshold_grid,
        base_calibration_margin_grid=args.base_calibration_margin_grid,
        base_calibration_bias_grid=args.base_calibration_bias_grid,
    )
    print(json.dumps({"report": summary["report"], "summary": summary["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
