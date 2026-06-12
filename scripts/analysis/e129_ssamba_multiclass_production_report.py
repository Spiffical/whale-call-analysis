#!/usr/bin/env python3
"""Score a SSAMBA multiclass checkpoint on common-row ONC H5 splits.

The E123/E127 SSL stack trains inside the selfsupervision_anomalies_onc model
code, while the multispecies project compares candidates with production-style
common-row accounting. This bridge exports row-level multiclass predictions from
an H5 dataset and writes an E124-compatible summary: cross-species mistakes
count as false positives for the predicted species and false negatives for the
true species; background false positives are tracked separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import e119_pairwise_refinement_report as e119
from scripts.analysis import e128_export_ssamba_binary_gate_predictions as e128
from scripts.analysis import multispecies_experiment_ledger as experiment_ledger


DEFAULT_CLASS_IDS = ("background", "species:Bm", "species:Bp", "species:Mn")
DEFAULT_SPLITS = ("val", "test")


def parse_csv(value: str) -> List[str]:
    items = [part.strip() for part in str(value or "").replace("|", ",").split(",") if part.strip()]
    if not items:
        raise ValueError("list cannot be empty")
    return items


def softmax(values: Sequence[float]) -> List[float]:
    logits = [float(value) for value in values]
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(value - m) for value in logits]
    denom = sum(exps)
    return [value / denom for value in exps] if denom else [0.0 for _ in logits]


def true_label_from_h5_label_string(label_string: Any, class_ids: Sequence[str]) -> str:
    class_set = set(class_ids)
    for token in e128.split_tokens(e128.decode_h5_text(label_string)):
        label = e128.normalize_species_label(token)
        if label in class_set and label != "background":
            return label
    return "background"


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


def score_h5_split(
    *,
    h5_path: Path,
    split: str,
    model: Any,
    model_args: Any,
    task: str,
    device: str,
    batch_size: int,
    class_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    import h5py  # type: ignore
    import torch  # type: ignore

    with h5py.File(h5_path, "r") as h5:
        n = int(h5["spectrograms"].shape[0])
        splits = e128.h5_strings(h5, "splits", n, default="")
        label_strings = e128.h5_strings(h5, "label_strings", n, default="normal")
        item_ids = e128.h5_strings(h5, "item_ids", n, default="")
        sources = e128.h5_strings(h5, "sources", n, default="")
        source_kinds = e128.h5_strings(h5, "source_kinds", n, default="")
        indices = [idx for idx, value in enumerate(splits) if value == split]
        rows: List[Dict[str, Any]] = []
        dataset_mean = getattr(model_args, "dataset_mean", None)
        dataset_std = getattr(model_args, "dataset_std", None)
        amount = float(getattr(model_args, "amount", 1.0) or 1.0)
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            spectrograms = h5["spectrograms"][batch_indices]
            spectrograms = e128.normalize_batch(
                spectrograms,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                amount=amount,
            )
            tensor = torch.from_numpy(np.asarray(spectrograms)).permute(0, 3, 1, 2).float().to(device)
            with torch.no_grad():
                logits = model(tensor, task=task)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits_np = logits.detach().cpu().numpy()
            for local_idx, row_idx in enumerate(batch_indices):
                raw_logits = np.asarray(logits_np[local_idx]).reshape(-1).tolist()
                probs = softmax(raw_logits)
                if len(probs) < len(class_ids):
                    raise ValueError(
                        f"model produced {len(probs)} class probabilities, but {len(class_ids)} class ids were requested"
                    )
                probs = probs[: len(class_ids)]
                pred_index = int(np.argmax(np.asarray(probs, dtype=np.float32)))
                pred_label = class_ids[pred_index]
                true_label = true_label_from_h5_label_string(label_strings[row_idx], class_ids)
                out: Dict[str, Any] = {
                    "item_id": item_ids[row_idx] or str(row_idx),
                    "source_audio": sources[row_idx],
                    "source_dataset": "E123_E126_H5",
                    "source_kind": source_kinds[row_idx],
                    "split": split,
                    "h5_index": row_idx,
                    "h5_path": str(h5_path),
                    "h5_label_string": label_strings[row_idx],
                    "true_class": true_label,
                    "target_label_ids": "" if true_label == "background" else true_label,
                    "pred_label": pred_label,
                    "predicted_class": pred_label,
                }
                for class_id, prob in zip(class_ids, probs):
                    out[f"prob__{class_id}"] = f"{prob:.8f}"
                    out[f"score__{class_id}"] = f"{prob:.8f}"
                rows.append(out)
        return rows


def load_scored_predictions(path: Path, class_ids: Sequence[str]) -> List[Dict[str, Any]]:
    return e119.load_predictions(path, class_ids)


def add_prediction_field(rows: Sequence[Mapping[str, Any]], field: str, values: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row, value_row in zip(rows, values):
        out = dict(row)
        out[field] = e119.clean(value_row.get("_pred"))
        out_rows.append(out)
    return out_rows


def per_species_rows(rows: Sequence[Mapping[str, Any]], pred_field: str, labels: Sequence[str], prediction_name: str) -> List[Dict[str, Any]]:
    metrics = e119.species_metrics(rows, pred_field, labels)
    return [{"prediction": prediction_name, **row} for row in metrics["per_class"]]


def confusion_rows(rows: Sequence[Mapping[str, Any]], pred_field: str, class_ids: Sequence[str]) -> List[Dict[str, Any]]:
    return e119.confusion_rows(rows, pred_field, class_ids)


def example_rows(
    rows: Sequence[Mapping[str, Any]],
    pred_field: str,
    class_ids: Sequence[str],
    limit_per_bucket: int = 50,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {
        "correct_species": [],
        "background_false_positive": [],
        "cross_species_error": [],
        "species_as_background_false_negative": [],
        "true_background": [],
    }
    for row in rows:
        true = e119.clean(row.get("_true"))
        pred = e119.clean(row.get(pred_field))
        if true != "background" and pred == true:
            buckets["correct_species"].append(row)
        elif true == "background" and pred != "background":
            buckets["background_false_positive"].append(row)
        elif true != "background" and pred == "background":
            buckets["species_as_background_false_negative"].append(row)
        elif true != "background" and pred != true:
            buckets["cross_species_error"].append(row)
        else:
            buckets["true_background"].append(row)

    examples: List[Dict[str, Any]] = []
    for bucket, bucket_rows in buckets.items():
        ordered = sorted(
            bucket_rows,
            key=lambda row: max(float(e119.probability(row, label) or 0.0) for label in class_ids),
            reverse=True,
        )
        for row in ordered[:limit_per_bucket]:
            example = {
                "bucket": bucket,
                "item_id": e119.clean(row.get("item_id")) or e119.clean(row.get("_key")),
                "true": e119.clean(row.get("_true")),
                "pred": e119.clean(row.get(pred_field)),
                "source_audio": e119.clean(row.get("source_audio")),
                "h5_index": e119.clean(row.get("h5_index")),
                "h5_label_string": e119.clean(row.get("h5_label_string")),
            }
            for label in class_ids:
                value = e119.probability(row, label)
                if value is not None:
                    example[f"prob__{label}"] = f"{float(value):.8f}"
            examples.append(example)
    return examples


def markdown_report(
    *,
    name: str,
    output_dir: Path,
    model_metrics: Sequence[Mapping[str, Any]],
    per_class_rows: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    base_decision_mode: str,
    base_rule: Mapping[str, Any],
) -> str:
    lines = [
        f"# E129 SSAMBA Multiclass Production Report: {name}",
        "",
        f"Decision mode: `{base_decision_mode}`.",
        f"Validation-calibrated rule: `{json.dumps(dict(base_rule), sort_keys=True) if base_rule else '{}'}`.",
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
    for row in per_class_rows:
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
    counts: Dict[str, int] = {}
    for row in examples:
        bucket = e119.clean(row.get("bucket"))
        counts[bucket] = counts.get(bucket, 0) + 1
    lines.extend(["", "## Example Buckets", "", "| bucket | count |", "| --- | ---: |"])
    for bucket, count in sorted(counts.items()):
        lines.append(f"| {bucket} | {count} |")
    lines.extend(
        [
            "",
            f"Metrics CSV: `{output_dir / 'e129_model_metrics.csv'}`",
            f"Per-species CSV: `{output_dir / 'e129_per_species_metrics.csv'}`",
            f"Examples CSV: `{output_dir / 'e129_examples.csv'}`",
            f"Confusion CSV: `{output_dir / 'e129_confusion.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report(
    *,
    name: str,
    ssl_repo_root: Optional[Path],
    model_dir: Path,
    checkpoint_path: Optional[Path],
    dataset_h5: Path,
    output_dir: Path,
    class_ids: Sequence[str],
    task: str,
    device: str,
    batch_size: int,
    base_decision_mode: str,
    calibration_threshold_grid: str,
    calibration_margin_grid: str,
    calibration_bias_grid: str,
    ledger_path: Optional[Path] = None,
    ledger_entry_id: str = "",
    training_set: str = "",
    validation_set: str = "",
    test_set: str = "",
    evaluation_note: str = "",
) -> Dict[str, Any]:
    import torch  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    e128.add_ssl_repo_to_path(ssl_repo_root)
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args, checkpoint = e128.load_model(model_dir, checkpoint_path, task, resolved_device)

    prediction_paths: Dict[str, str] = {}
    rows_by_split: Dict[str, int] = {}
    predictions_dir = output_dir / "predictions"
    for split in DEFAULT_SPLITS:
        rows = score_h5_split(
            h5_path=dataset_h5,
            split=split,
            model=model,
            model_args=model_args,
            task=task,
            device=resolved_device,
            batch_size=batch_size,
            class_ids=class_ids,
        )
        out_path = predictions_dir / ("validation_predictions.csv" if split == "val" else f"{split}_predictions.csv")
        write_csv(out_path, rows)
        prediction_paths[split] = str(out_path)
        rows_by_split[split] = len(rows)

    labels = e119.species_class_ids(class_ids)
    val_rows = load_scored_predictions(Path(prediction_paths["val"]), class_ids)
    test_rows = load_scored_predictions(Path(prediction_paths["test"]), class_ids)
    base_rule: Dict[str, Any] = {}
    base_rule_sweep: List[Dict[str, Any]] = []
    selected_prediction = "pred"
    selected_pred_field = "_pred"
    selected_val = val_rows
    selected_test = test_rows
    if base_decision_mode == "calibrated":
        base_rule, base_rule_sweep = e119.tune_base_rule(
            val_rows,
            labels,
            thresholds=e119.parse_float_grid(calibration_threshold_grid),
            margins=e119.parse_float_grid(calibration_margin_grid),
            biases=e119.parse_float_grid(calibration_bias_grid),
        )
        calibrated_val = e119.apply_base_rule(val_rows, labels, base_rule)
        calibrated_test = e119.apply_base_rule(test_rows, labels, base_rule)
        selected_prediction = "calibrated"
        selected_pred_field = "_pred"
        selected_val = calibrated_val
        selected_test = calibrated_test
    elif base_decision_mode != "argmax":
        raise ValueError(f"unknown base decision mode: {base_decision_mode}")

    model_metrics = [
        e119.metric_row(name, "val", "_pred", val_rows, labels),
        e119.metric_row(name, "test", "_pred", test_rows, labels),
    ]
    if base_decision_mode == "calibrated":
        model_metrics.extend(
            [
                {**e119.metric_row(name, "val", selected_pred_field, selected_val, labels), "prediction": selected_prediction},
                {**e119.metric_row(name, "test", selected_pred_field, selected_test, labels), "prediction": selected_prediction},
            ]
        )
    per_class = per_species_rows(selected_test, selected_pred_field, labels, selected_prediction)
    examples = example_rows(selected_test, selected_pred_field, class_ids)
    confusion = confusion_rows(selected_test, selected_pred_field, class_ids)

    write_csv(output_dir / "e129_model_metrics.csv", model_metrics)
    write_csv(output_dir / "e129_per_species_metrics.csv", per_class)
    write_csv(output_dir / "e129_examples.csv", examples)
    write_csv(output_dir / "e129_confusion.csv", confusion)
    if base_rule_sweep:
        write_csv(output_dir / "e129_base_calibration_sweep.csv", base_rule_sweep)
    report_path = output_dir / "e129_ssamba_multiclass_production_report.md"
    report_path.write_text(
        markdown_report(
            name=name,
            output_dir=output_dir,
            model_metrics=model_metrics,
            per_class_rows=per_class,
            examples=examples,
            base_decision_mode=base_decision_mode,
            base_rule=base_rule,
        ),
        encoding="utf-8",
    )
    summary = {
        "name": name,
        "class_ids": list(class_ids),
        "metric_labels": labels,
        "base_decision_mode": base_decision_mode,
        "base_rule": base_rule,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "dataset_h5": str(dataset_h5),
        "task": task,
        "device": resolved_device,
        "rows_by_split": rows_by_split,
        "selected_prediction": selected_prediction,
        "model_metrics": model_metrics,
        "outputs": {
            "report": str(report_path),
            "metrics": str(output_dir / "e129_model_metrics.csv"),
            "per_species": str(output_dir / "e129_per_species_metrics.csv"),
            "examples": str(output_dir / "e129_examples.csv"),
            "confusion": str(output_dir / "e129_confusion.csv"),
            "threshold_sweep": "" if not base_rule_sweep else str(output_dir / "e129_base_calibration_sweep.csv"),
            "predictions_dir": str(predictions_dir),
            "val_predictions": prediction_paths["val"],
            "test_predictions": prediction_paths["test"],
        },
    }
    summary_path = output_dir / "e129_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if ledger_path is not None:
        test_metric = next(
            row for row in model_metrics if row["split"] == "test" and row["prediction"] == selected_prediction
        )
        experiment_ledger.append_generic_note(
            name=f"{name}: SSAMBA Multiclass Production Report",
            ledger_path=ledger_path,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            evaluation_note=evaluation_note or "production-style common-row multiclass species evaluation",
            metrics=[
                f"Macro F1: {float(test_metric['macro_f1']):.4f}",
                f"Micro F1: {float(test_metric['micro_f1']):.4f}",
                f"Precision: {float(test_metric['micro_precision']):.4f}",
                f"Recall: {float(test_metric['micro_recall']):.4f}",
                f"Cross-species FP: {test_metric['cross_species_fp']}",
                f"Background FP: {test_metric['background_fp']}",
                f"Species-as-background FN: {test_metric['species_as_background_fn']}",
            ],
            artifacts=[
                ("Report", str(report_path)),
                ("Summary JSON", str(summary_path)),
                ("Examples CSV", str(output_dir / "e129_examples.csv")),
            ],
            interpretation="Review examples and leaderboard before using this as a model-selection decision",
            entry_id=ledger_entry_id,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ssl-repo-root", type=Path, default=None)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--dataset-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-ids", default=",".join(DEFAULT_CLASS_IDS))
    parser.add_argument("--task", default="ft_avgtok")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-decision-mode", default="argmax", choices=["argmax", "calibrated"])
    parser.add_argument(
        "--base-calibration-threshold-grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--base-calibration-margin-grid", default="-0.25,0.0,0.25")
    parser.add_argument("--base-calibration-bias-grid", default="-0.30,-0.15,0.0,0.15,0.30")
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--ledger-entry-id", default="")
    parser.add_argument("--training-set", default="")
    parser.add_argument("--validation-set", default="")
    parser.add_argument("--test-set", default="")
    parser.add_argument("--evaluation-note", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_report(
        name=args.name,
        ssl_repo_root=args.ssl_repo_root,
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint_path,
        dataset_h5=args.dataset_h5,
        output_dir=args.output_dir,
        class_ids=parse_csv(args.class_ids),
        task=args.task,
        device=args.device,
        batch_size=args.batch_size,
        base_decision_mode=args.base_decision_mode,
        calibration_threshold_grid=args.base_calibration_threshold_grid,
        calibration_margin_grid=args.base_calibration_margin_grid,
        calibration_bias_grid=args.base_calibration_bias_grid,
        ledger_path=args.ledger_path,
        ledger_entry_id=args.ledger_entry_id,
        training_set=args.training_set,
        validation_set=args.validation_set,
        test_set=args.test_set,
        evaluation_note=args.evaluation_note,
    )
    print(json.dumps({"summary": str(args.output_dir / "e129_summary.json"), "outputs": summary["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
