#!/usr/bin/env python3
"""Evaluate a pairwise species refinement on multiclass predictions.

The report tunes a conservative "flip only when confident" rule on validation
rows, then applies it to test rows. Metrics keep the production-style accounting:
wrong species predictions count as a false positive for the predicted species
and a false negative for the true species.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPECIES_NAMES = {
    "species:Bm": "blue whale",
    "species:Bp": "fin whale",
    "species:Mn": "humpback whale",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def split_labels(text: str) -> List[str]:
    labels: List[str] = []
    for part in clean(text).replace(",", "|").split("|"):
        label = part.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def load_class_ids(summary_path: Optional[Path], fallback: Sequence[str]) -> List[str]:
    if summary_path and summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        class_ids = payload.get("class_ids")
        if isinstance(class_ids, list) and class_ids:
            return [clean(value) for value in class_ids]
    return list(fallback)


def species_class_ids(class_ids: Sequence[str]) -> List[str]:
    return [label for label in class_ids if label != "background" and label.startswith("species:")]


def require_pairwise_labels(parser: argparse.ArgumentParser, class_ids: Sequence[str]) -> Tuple[str, str]:
    labels = species_class_ids(class_ids)
    if len(labels) != 2:
        parser.error(
            "pairwise summary must define exactly two species class_ids; "
            f"got {labels or list(class_ids)}"
        )
    return labels[0], labels[1]


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def discover_summary(run_dir: Optional[Path]) -> Optional[Path]:
    if run_dir is None:
        return None
    return first_existing((run_dir / "train" / "run_summary.json", run_dir / "run_summary.json"))


def discover_predictions(run_dir: Path, split: str, *, prefer_rule: bool) -> Optional[Path]:
    aliases = [split]
    if split == "val":
        aliases.append("validation")
    elif split == "test":
        aliases.append("testing")
    roots = [run_dir / "train", run_dir]
    exact_suffixes = (
        ("predictions_best_val_rule.csv", "predictions_argmax.csv", "predictions.csv")
        if prefer_rule
        else ("predictions_argmax.csv", "predictions.csv", "predictions_best_val_rule.csv")
    )
    exact_paths: List[Path] = []
    for root in roots:
        for alias in aliases:
            for suffix in exact_suffixes:
                exact_paths.append(root / f"{alias}_{suffix}")
                exact_paths.append(root / f"{alias}_prediction_{suffix}")
    found = first_existing(exact_paths)
    if found is not None:
        return found

    wildcard_suffixes = (
        ("*best*rule*.csv", "*argmax*.csv", "*.csv")
        if prefer_rule
        else ("*argmax*.csv", "*.csv", "*best*rule*.csv")
    )
    candidates: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for alias in aliases:
            for suffix in wildcard_suffixes:
                candidates.extend(sorted(root.glob(f"*{alias}*pred*{suffix}")))
    return candidates[0] if candidates else None


def require_path(parser: argparse.ArgumentParser, path: Optional[Path], label: str) -> Path:
    if path is None:
        parser.error(f"missing {label}; pass it explicitly or provide the corresponding run directory")
    if not path.is_file():
        parser.error(f"{label} does not exist: {path}")
    return path


def normalize_label(value: Any, class_ids: Sequence[str]) -> str:
    text = clean(value)
    if text == "":
        return ""
    if text in class_ids or text.startswith("species:") or text == "background":
        return text
    try:
        idx = int(float(text))
    except ValueError:
        return text
    if 0 <= idx < len(class_ids):
        return class_ids[idx]
    return text


def first_present(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = clean(row.get(key))
        if value:
            return value
    return ""


def row_key(row: Mapping[str, Any], index: int) -> str:
    direct = first_present(
        row,
        (
            "item_id",
            "meta__item_id",
            "metadata_item_id",
            "expected_mat_name",
            "mat_path",
            "low_mat_path",
        ),
    )
    if direct:
        return direct
    clip = first_present(row, ("clip", "filename", "source_audio", "source_soundfile"))
    begin = first_present(row, ("begin_s", "begin_time_s", "window_start_s"))
    end = first_present(row, ("end_s", "end_time_s"))
    if clip and begin:
        return f"{clip}|{begin}|{end}"
    return f"__row_index__:{index}"


def true_label(row: Mapping[str, Any], class_ids: Sequence[str]) -> str:
    explicit = first_present(
        row,
        (
            "true_label",
            "target_label",
            "target_class_id",
            "true_class_id",
            "target_class_index",
            "true_class_index",
            "y_true_label",
            "target_class",
            "true_class",
            "target",
            "y_true",
            "label",
        ),
    )
    label = normalize_label(explicit, class_ids)
    if label:
        return label
    for field in ("target_label_ids", "canonical_label_ids", "analysis_label_ids", "label_ids"):
        for label_id in split_labels(clean(row.get(field))):
            if label_id in class_ids and label_id != "background":
                return label_id
    species = first_present(row, ("species", "species_code", "canonical_species"))
    if species in {"Bm", "Bp", "Mn"}:
        return f"species:{species}"
    return "background"


def pred_label(row: Mapping[str, Any], class_ids: Sequence[str]) -> str:
    explicit = first_present(
        row,
        (
            "pred_label",
            "predicted_label",
            "pred_class_id",
            "predicted_class_id",
            "pred_class_index",
            "predicted_class_index",
            "y_pred_label",
            "pred_argmax",
            "argmax_pred",
            "pred_best_val_rule",
            "best_val_rule_pred",
            "pred_class",
            "predicted_class",
            "prediction",
            "pred",
            "y_pred",
        ),
    )
    label = normalize_label(explicit, class_ids)
    return label or "background"


def as_float(value: Any) -> Optional[float]:
    text = clean(value)
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def probability(row: Mapping[str, Any], label: str) -> Optional[float]:
    compact = label.replace(":", "_")
    candidates = (
        f"prob__{label}",
        f"score__{label}",
        f"p__{label}",
        f"prob_{label}",
        f"score_{label}",
        f"prob__{compact}",
        f"score__{compact}",
        f"p__{compact}",
        f"prob_{compact}",
        f"score_{compact}",
        f"probability__{label}",
        f"probability_{label}",
        f"probability__{compact}",
        f"probability_{compact}",
        label,
    )
    for key in candidates:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def parse_float_grid(text: str) -> List[float]:
    values: List[float] = []
    for part in clean(text).split(","):
        if not part.strip():
            continue
        values.append(float(part))
    if not values:
        raise ValueError("grid cannot be empty")
    return values


def load_predictions(path: Path, class_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(read_csv(path)):
        out: Dict[str, Any] = dict(row)
        out["_row_index"] = index
        out["_key"] = row_key(row, index)
        out["_true"] = true_label(row, class_ids)
        out["_pred"] = pred_label(row, class_ids)
        rows.append(out)
    return rows


def predict_base_rule_for_row(row: Mapping[str, Any], labels: Sequence[str], rule: Mapping[str, Any]) -> str:
    scores: List[Tuple[float, str]] = []
    for label in labels:
        value = probability(row, label)
        if value is None:
            continue
        bias = float(rule.get(f"bias__{label}", 0.0))
        scores.append((float(value) + bias, label))
    if not scores:
        return clean(row.get("_pred")) or "background"
    best_score, best_label = max(scores, key=lambda item: item[0])
    bg = probability(row, "background")
    bg_score = float(bg) + float(rule.get("bias__background", 0.0)) if bg is not None else 0.0
    if best_score < float(rule.get("threshold", 0.0)):
        return "background"
    if (best_score - bg_score) < float(rule.get("margin", -1.0)):
        return "background"
    return best_label


def apply_base_rule(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    rule: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["_pred_existing"] = clean(row.get("_pred"))
        out["_pred"] = predict_base_rule_for_row(row, labels, rule)
        out_rows.append(out)
    return out_rows


def tune_base_rule(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    thresholds: Sequence[float],
    margins: Sequence[float],
    biases: Sequence[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sweep_rows: List[Dict[str, Any]] = []
    bias_names = [f"bias__{label}" for label in labels]
    for threshold, margin, bias_values in itertools.product(thresholds, margins, itertools.product(biases, repeat=len(labels))):
        rule: Dict[str, Any] = {"threshold": threshold, "margin": margin}
        rule.update(dict(zip(bias_names, bias_values)))
        predicted_rows = apply_base_rule(rows, labels, rule)
        metrics = species_metrics(predicted_rows, "_pred", labels)
        sweep_rows.append(
            {
                **rule,
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "cross_species_fp": metrics["cross_species_fp"],
                "background_fp": metrics["background_fp"],
                "species_as_background_fn": metrics["species_as_background_fn"],
            }
        )
    best = max(
        sweep_rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["micro_f1"]),
            -int(row["cross_species_fp"]),
            -int(row["background_fp"]),
        ),
    )
    rule = {key: best[key] for key in best if key == "threshold" or key == "margin" or key.startswith("bias__")}
    return rule, sweep_rows


def pairwise_prob_field(label: str) -> str:
    return f"prob__{label}"


def load_pairwise(
    path: Path,
    class_ids: Sequence[str],
    pairwise_labels: Tuple[str, str],
) -> Dict[str, Dict[str, Any]]:
    pairwise: Dict[str, Dict[str, Any]] = {}
    left_label, right_label = pairwise_labels
    for index, row in enumerate(read_csv(path)):
        key = row_key(row, index)
        left_prob = probability(row, left_label)
        right_prob = probability(row, right_label)
        if left_prob is None or right_prob is None:
            pred = pred_label(row, class_ids)
            if pred == left_label:
                left_prob, right_prob = 1.0, 0.0
            elif pred == right_label:
                left_prob, right_prob = 0.0, 1.0
            else:
                continue
        left_prob = float(left_prob)
        right_prob = float(right_prob)
        pairwise[key] = {
            "key": key,
            pairwise_prob_field(left_label): left_prob,
            pairwise_prob_field(right_label): right_prob,
            "pairwise_pred": left_label if left_prob >= right_prob else right_label,
            "pairwise_margin": abs(left_prob - right_prob),
            "pairwise_true": true_label(row, class_ids),
        }
    return pairwise


def species_metrics(rows: Sequence[Mapping[str, Any]], pred_field: str, labels: Sequence[str]) -> Dict[str, Any]:
    per_class: List[Dict[str, Any]] = []
    tp_total = fp_total = fn_total = 0
    for label in labels:
        tp = fp = fn = support = predicted = 0
        for row in rows:
            true = clean(row.get("_true"))
            pred = clean(row.get(pred_field))
            if true == label:
                support += 1
            if pred == label:
                predicted += 1
            if true == label and pred == label:
                tp += 1
            elif true != label and pred == label:
                fp += 1
            elif true == label and pred != label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append(
            {
                "class_id": label,
                "name": SPECIES_NAMES.get(label, label),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
    macro_f1 = sum(float(row["f1"]) for row in per_class) / max(len(per_class), 1)
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    species_rows = [row for row in rows if clean(row.get("_true")) != "background"]
    species_exact = (
        sum(1 for row in species_rows if clean(row.get("_true")) == clean(row.get(pred_field))) / len(species_rows)
        if species_rows
        else 0.0
    )
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "species_exact_accuracy": species_exact,
        "cross_species_fp": sum(
            1
            for row in rows
            if clean(row.get("_true")) != "background"
            and clean(row.get(pred_field)) != "background"
            and clean(row.get("_true")) != clean(row.get(pred_field))
        ),
        "background_fp": sum(
            1
            for row in rows
            if clean(row.get("_true")) == "background" and clean(row.get(pred_field)) != "background"
        ),
        "species_as_background_fn": sum(
            1
            for row in rows
            if clean(row.get("_true")) != "background" and clean(row.get(pred_field)) == "background"
        ),
        "rows": len(rows),
        "species_rows": len(species_rows),
        "per_class": per_class,
    }


def apply_refinement(
    rows: Sequence[Mapping[str, Any]],
    pairwise: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float,
    pairwise_labels: Tuple[str, str],
) -> List[Dict[str, Any]]:
    refined: List[Dict[str, Any]] = []
    candidate_labels = set(pairwise_labels)
    for row in rows:
        out = dict(row)
        base_pred = clean(row.get("_pred")) or "background"
        out["_refined"] = base_pred
        out["refinement_action"] = "not_candidate"
        pair = pairwise.get(clean(row.get("_key")))
        if base_pred in candidate_labels and pair is not None:
            pair_pred = clean(pair.get("pairwise_pred"))
            margin = float(pair.get("pairwise_margin") or 0.0)
            out["pairwise_pred"] = pair_pred
            out["pairwise_margin"] = margin
            for label in pairwise_labels:
                out[f"pairwise_{pairwise_prob_field(label)}"] = pair.get(pairwise_prob_field(label))
            if pair_pred in candidate_labels and pair_pred != base_pred and margin >= threshold:
                out["_refined"] = pair_pred
                out["refinement_action"] = "flipped"
            else:
                out["refinement_action"] = "kept_candidate"
        refined.append(out)
    return refined


def tune_threshold(
    val_rows: Sequence[Mapping[str, Any]],
    pairwise: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    pairwise_labels: Tuple[str, str],
) -> Tuple[float, List[Dict[str, Any]]]:
    candidates = [round(i / 100, 2) for i in range(0, 101, 5)]
    rows: List[Dict[str, Any]] = []
    for threshold in candidates:
        refined = apply_refinement(
            val_rows,
            pairwise,
            threshold=threshold,
            pairwise_labels=pairwise_labels,
        )
        metrics = species_metrics(refined, "_refined", labels)
        rows.append(
            {
                "threshold": threshold,
                "pairwise_labels": "|".join(pairwise_labels),
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


def confusion_rows(rows: Sequence[Mapping[str, Any]], pred_field: str, class_ids: Sequence[str]) -> List[Dict[str, Any]]:
    labels = list(class_ids)
    matrix: Dict[Tuple[str, str], int] = {(true, pred): 0 for true in labels for pred in labels}
    for row in rows:
        true = clean(row.get("_true")) or "background"
        pred = clean(row.get(pred_field)) or "background"
        if true not in labels:
            labels.append(true)
        if pred not in labels:
            labels.append(pred)
        matrix[(true, pred)] = matrix.get((true, pred), 0) + 1
    return [
        {"true": true, "pred": pred, "count": matrix.get((true, pred), 0)}
        for true in labels
        for pred in labels
        if matrix.get((true, pred), 0)
    ]


def example_rows(rows: Sequence[Mapping[str, Any]], limit_per_bucket: int = 50) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {
        "corrected_by_pairwise": [],
        "damaged_by_pairwise": [],
        "remaining_error": [],
        "kept_correct_candidate": [],
    }
    for row in rows:
        true = clean(row.get("_true"))
        base = clean(row.get("_pred"))
        refined = clean(row.get("_refined"))
        if row.get("refinement_action") == "flipped" and base != true and refined == true:
            buckets["corrected_by_pairwise"].append(row)
        elif row.get("refinement_action") == "flipped" and base == true and refined != true:
            buckets["damaged_by_pairwise"].append(row)
        elif refined != true:
            buckets["remaining_error"].append(row)
        elif row.get("refinement_action") == "kept_candidate" and refined == true:
            buckets["kept_correct_candidate"].append(row)
    examples: List[Dict[str, Any]] = []
    for bucket, bucket_rows in buckets.items():
        ordered = sorted(bucket_rows, key=lambda row: float(row.get("pairwise_margin") or 0.0), reverse=True)
        for row in ordered[:limit_per_bucket]:
            example = {
                key: row.get(key, "")
                for key in row
                if key.startswith("pairwise_prob__")
            }
            example.update(
                {
                    "bucket": bucket,
                    "item_id": clean(row.get("item_id")) or clean(row.get("_key")),
                    "true": clean(row.get("_true")),
                    "base_pred": clean(row.get("_pred")),
                    "refined_pred": clean(row.get("_refined")),
                    "pairwise_pred": clean(row.get("pairwise_pred")),
                    "pairwise_margin": row.get("pairwise_margin", ""),
                    "clip": clean(row.get("clip")) or clean(row.get("filename")),
                    "begin_s": clean(row.get("begin_s")) or clean(row.get("begin_time_s")),
                    "end_s": clean(row.get("end_s")) or clean(row.get("end_time_s")),
                    "mat_path": clean(row.get("mat_path")) or clean(row.get("low_mat_path")),
                }
            )
            examples.append(example)
    return examples


def metric_row(name: str, split: str, pred_field: str, rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    metrics = species_metrics(rows, pred_field, labels)
    return {
        "model": name,
        "split": split,
        "prediction": pred_field.lstrip("_"),
        "rows": metrics["rows"],
        "species_rows": metrics["species_rows"],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "micro_precision": metrics["micro_precision"],
        "micro_recall": metrics["micro_recall"],
        "species_exact_accuracy": metrics["species_exact_accuracy"],
        "cross_species_fp": metrics["cross_species_fp"],
        "background_fp": metrics["background_fp"],
        "species_as_background_fn": metrics["species_as_background_fn"],
    }


def markdown_report(
    *,
    name: str,
    output_dir: Path,
    threshold: float,
    pairwise_labels: Tuple[str, str],
    base_decision_mode: str,
    base_rule: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    per_class_rows: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> str:
    base_rule_text = json.dumps(dict(base_rule), sort_keys=True) if base_rule else "{}"
    pairwise_label_text = " vs ".join(pairwise_labels)
    lines = [
        f"# E119 Pairwise Refinement Report: {name}",
        "",
        "This evaluates a pairwise species specialist as a conservative refinement on top of an existing production-style multiclass model.",
        "",
        f"Pairwise specialist labels: `{pairwise_label_text}`.",
        f"Base decision mode: `{base_decision_mode}`.",
        f"Base calibration rule: `{base_rule_text}`.",
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
    lines.extend(
        [
            "",
            "## Example Buckets",
            "",
            "| bucket | count |",
            "| --- | ---: |",
        ]
    )
    counts: Dict[str, int] = {}
    for row in examples:
        bucket = clean(row.get("bucket"))
        counts[bucket] = counts.get(bucket, 0) + 1
    for bucket, count in sorted(counts.items()):
        lines.append(f"| {bucket} | {count} |")
    lines.extend(
        [
            "",
            f"Metrics CSV: `{output_dir / 'e119_model_metrics.csv'}`",
            f"Per-species CSV: `{output_dir / 'e119_per_species_metrics.csv'}`",
            f"Examples CSV: `{output_dir / 'e119_examples.csv'}`",
            f"Threshold sweep CSV: `{output_dir / 'e119_threshold_sweep.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def safe_stem(path: Path) -> str:
    text = path.name.strip() or "base"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def select_metric(rows: Sequence[Mapping[str, Any]], split: str, prediction: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("split") == split and row.get("prediction") == prediction:
            return row
    raise KeyError(f"missing metric row for split={split!r} prediction={prediction!r}")


def run_refinement_report(
    *,
    parser: argparse.ArgumentParser,
    name: str,
    output_dir: Path,
    base_run_dir: Optional[Path],
    pairwise_run_dir: Optional[Path],
    base_val_predictions: Optional[Path],
    base_test_predictions: Optional[Path],
    pairwise_val_predictions: Optional[Path],
    pairwise_test_predictions: Optional[Path],
    base_summary_json: Optional[Path],
    pairwise_summary_json: Optional[Path],
    base_decision_mode: str,
    base_calibration_threshold_grid: str,
    base_calibration_margin_grid: str,
    base_calibration_bias_grid: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_summary_json = base_summary_json or discover_summary(base_run_dir)
    pairwise_summary_json = pairwise_summary_json or discover_summary(pairwise_run_dir)
    base_val_predictions = base_val_predictions or (
        discover_predictions(base_run_dir, "val", prefer_rule=True) if base_run_dir else None
    )
    base_test_predictions = base_test_predictions or (
        discover_predictions(base_run_dir, "test", prefer_rule=True) if base_run_dir else None
    )
    pairwise_val_predictions = pairwise_val_predictions or (
        discover_predictions(pairwise_run_dir, "val", prefer_rule=False) if pairwise_run_dir else None
    )
    pairwise_test_predictions = pairwise_test_predictions or (
        discover_predictions(pairwise_run_dir, "test", prefer_rule=False) if pairwise_run_dir else None
    )
    base_val_predictions = require_path(parser, base_val_predictions, "base validation predictions")
    base_test_predictions = require_path(parser, base_test_predictions, "base test predictions")
    pairwise_val_predictions = require_path(parser, pairwise_val_predictions, "pairwise validation predictions")
    pairwise_test_predictions = require_path(parser, pairwise_test_predictions, "pairwise test predictions")

    base_class_ids = load_class_ids(base_summary_json, ("background", "species:Bp", "species:Bm", "species:Mn"))
    pair_class_ids = load_class_ids(pairwise_summary_json, ("background", "species:Bp", "species:Mn"))
    pairwise_labels = require_pairwise_labels(parser, pair_class_ids)
    metric_labels = species_class_ids(base_class_ids)

    base_val = load_predictions(base_val_predictions, base_class_ids)
    base_test = load_predictions(base_test_predictions, base_class_ids)
    pair_val = load_pairwise(pairwise_val_predictions, pair_class_ids, pairwise_labels)
    pair_test = load_pairwise(pairwise_test_predictions, pair_class_ids, pairwise_labels)

    base_rule: Optional[Dict[str, Any]] = None
    base_rule_sweep: List[Dict[str, Any]] = []
    if base_decision_mode == "calibrated":
        base_rule, base_rule_sweep = tune_base_rule(
            base_val,
            metric_labels,
            thresholds=parse_float_grid(base_calibration_threshold_grid),
            margins=parse_float_grid(base_calibration_margin_grid),
            biases=parse_float_grid(base_calibration_bias_grid),
        )
        base_val = apply_base_rule(base_val, metric_labels, base_rule)
        base_test = apply_base_rule(base_test, metric_labels, base_rule)

    threshold, threshold_rows = tune_threshold(base_val, pair_val, metric_labels, pairwise_labels)
    refined_val = apply_refinement(base_val, pair_val, threshold=threshold, pairwise_labels=pairwise_labels)
    refined_test = apply_refinement(base_test, pair_test, threshold=threshold, pairwise_labels=pairwise_labels)

    model_metrics = [
        metric_row(name, "val", "_pred", base_val, metric_labels),
        metric_row(name, "val", "_refined", refined_val, metric_labels),
        metric_row(name, "test", "_pred", base_test, metric_labels),
        metric_row(name, "test", "_refined", refined_test, metric_labels),
    ]
    per_species_rows: List[Dict[str, Any]] = []
    for prediction, rows, field in (("base", base_test, "_pred"), ("refined", refined_test, "_refined")):
        metrics = species_metrics(rows, field, metric_labels)
        for per_class in metrics["per_class"]:
            per_species_rows.append({"prediction": prediction, **per_class})
    examples = example_rows(refined_test)
    confusion = confusion_rows(refined_test, "_refined", base_class_ids)

    write_csv(output_dir / "e119_model_metrics.csv", model_metrics)
    write_csv(output_dir / "e119_per_species_metrics.csv", per_species_rows)
    write_csv(output_dir / "e119_examples.csv", examples)
    write_csv(output_dir / "e119_threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "e119_refined_confusion.csv", confusion)
    if base_rule_sweep:
        write_csv(output_dir / "e119_base_calibration_sweep.csv", base_rule_sweep)
    summary = {
        "name": name,
        "threshold": threshold,
        "base_decision_mode": base_decision_mode,
        "base_rule": base_rule or {},
        "base_run_dir": "" if base_run_dir is None else str(base_run_dir),
        "pairwise_run_dir": "" if pairwise_run_dir is None else str(pairwise_run_dir),
        "base_class_ids": base_class_ids,
        "pairwise_class_ids": pair_class_ids,
        "pairwise_labels": list(pairwise_labels),
        "metric_labels": metric_labels,
        "inputs": {
            "base_val_predictions": str(base_val_predictions),
            "base_test_predictions": str(base_test_predictions),
            "pairwise_val_predictions": str(pairwise_val_predictions),
            "pairwise_test_predictions": str(pairwise_test_predictions),
            "base_summary_json": "" if base_summary_json is None else str(base_summary_json),
            "pairwise_summary_json": "" if pairwise_summary_json is None else str(pairwise_summary_json),
        },
        "model_metrics": model_metrics,
        "outputs": {
            "metrics": str(output_dir / "e119_model_metrics.csv"),
            "per_species": str(output_dir / "e119_per_species_metrics.csv"),
            "examples": str(output_dir / "e119_examples.csv"),
            "threshold_sweep": str(output_dir / "e119_threshold_sweep.csv"),
            "confusion": str(output_dir / "e119_refined_confusion.csv"),
            "base_calibration_sweep": "" if not base_rule_sweep else str(output_dir / "e119_base_calibration_sweep.csv"),
        },
    }
    (output_dir / "e119_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = markdown_report(
        name=name,
        output_dir=output_dir,
        threshold=threshold,
        pairwise_labels=pairwise_labels,
        base_decision_mode=base_decision_mode,
        base_rule=base_rule or {},
        metric_rows=model_metrics,
        per_class_rows=per_species_rows,
        examples=examples,
    )
    report_path = output_dir / "e119_pairwise_refinement_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {**summary, "report": str(report_path)}


def comparison_report(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# E119 Multi-Base Pairwise Refinement Comparison",
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
    lines.extend(
        [
            "",
            f"Rankings CSV: `{output_dir / 'e119_comparison_rankings.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-run-dir", type=Path, action="append", default=None)
    parser.add_argument("--pairwise-run-dir", type=Path, default=None)
    parser.add_argument("--base-val-predictions", type=Path, default=None)
    parser.add_argument("--base-test-predictions", type=Path, default=None)
    parser.add_argument("--pairwise-val-predictions", type=Path, default=None)
    parser.add_argument("--pairwise-test-predictions", type=Path, default=None)
    parser.add_argument("--base-summary-json", type=Path, default=None)
    parser.add_argument("--pairwise-summary-json", type=Path, default=None)
    parser.add_argument("--base-decision-mode", default="existing", choices=["existing", "calibrated"])
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

    if len(base_run_dirs) > 1:
        if args.base_val_predictions or args.base_test_predictions or args.base_summary_json:
            parser.error("explicit base prediction/summary paths cannot be combined with multiple --base-run-dir values")
        comparison_rows: List[Dict[str, Any]] = []
        for base_run_dir in base_run_dirs:
            base_name = safe_stem(base_run_dir)
            child_output = args.output_dir / base_name
            summary = run_refinement_report(
                parser=parser,
                name=f"{args.name}__{base_name}",
                output_dir=child_output,
                base_run_dir=base_run_dir,
                pairwise_run_dir=args.pairwise_run_dir,
                base_val_predictions=None,
                base_test_predictions=None,
                pairwise_val_predictions=args.pairwise_val_predictions,
                pairwise_test_predictions=args.pairwise_test_predictions,
                base_summary_json=None,
                pairwise_summary_json=args.pairwise_summary_json,
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
                -int(row["refined_test_cross_species_fp"]),
            ),
            reverse=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "e119_comparison_rankings.csv", comparison_rows)
        report_path = args.output_dir / "e119_comparison_report.md"
        report_path.write_text(comparison_report(args.output_dir, comparison_rows), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "rankings": str(args.output_dir / "e119_comparison_rankings.csv"), "rows": comparison_rows}, indent=2))
        return 0

    summary = run_refinement_report(
        parser=parser,
        name=args.name,
        output_dir=args.output_dir,
        base_run_dir=base_run_dirs[0] if base_run_dirs else None,
        pairwise_run_dir=args.pairwise_run_dir,
        base_val_predictions=args.base_val_predictions,
        base_test_predictions=args.base_test_predictions,
        pairwise_val_predictions=args.pairwise_val_predictions,
        pairwise_test_predictions=args.pairwise_test_predictions,
        base_summary_json=args.base_summary_json,
        pairwise_summary_json=args.pairwise_summary_json,
        base_decision_mode=args.base_decision_mode,
        base_calibration_threshold_grid=args.base_calibration_threshold_grid,
        base_calibration_margin_grid=args.base_calibration_margin_grid,
        base_calibration_bias_grid=args.base_calibration_bias_grid,
    )
    print(json.dumps({"report": summary["report"], "threshold": summary["threshold"], "metrics": summary["model_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
