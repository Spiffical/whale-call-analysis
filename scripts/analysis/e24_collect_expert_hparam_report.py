#!/usr/bin/env python3
"""Collect E24 expert hyperparameter metrics and rank posthoc ensembles."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THREE_SPECIES = ("species:Bp", "species:Bm", "species:Mn")
LABEL_NAMES = {
    "species:Bp": "fin whale",
    "species:Bm": "blue whale",
    "species:Mn": "humpback whale",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def split_labels(value: Any) -> List[str]:
    return [token.strip() for token in clean(value).replace(",", "|").replace(";", "|").split("|") if token.strip()]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def base_key(row: Mapping[str, str]) -> str:
    return clean(row.get("item_id")) or "|".join(
        [
            clean(row.get("source_dataset")),
            clean(row.get("source_audio")),
            clean(row.get("begin_s")),
            clean(row.get("end_s")),
            clean(row.get("split")),
        ]
    )


def merge_labels(existing: str, new: str) -> str:
    labels: List[str] = []
    for label in [*split_labels(existing), *split_labels(new)]:
        if label not in labels:
            labels.append(label)
    return "|".join(labels)


def run_rows(submitted_rows: Sequence[Mapping[str, str]], plan_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    plan_by_experiment = {clean(row.get("experiment")): row for row in plan_rows}
    out: List[Dict[str, Any]] = []
    for row in submitted_rows:
        experiment = clean(row.get("experiment"))
        run_dir = Path(clean(row.get("run_dir")))
        if not experiment or not run_dir or clean(row.get("job_id")) == "DRY_RUN":
            continue
        plan = plan_by_experiment.get(experiment, {})
        metrics_path = run_dir / "train" / "onc_calibrated_eval" / "onc_calibrated_metrics_summary.json"
        payload = load_json(metrics_path)
        label_ids = split_labels(plan.get("eval_label_ids")) or payload.get("label_ids", [])
        label_id = label_ids[0] if len(label_ids) == 1 else ",".join(label_ids)
        test = payload.get("onc_test_metrics", {}) if payload else {}
        hard_rows = payload.get("onc_test_hard_negative_fp_rows", []) if payload else []
        hard_fp = sum(int(item.get("any_primary_fp") or 0) for item in hard_rows)
        hard_total = sum(int(item.get("rows") or 0) for item in hard_rows)
        out.append(
            {
                "job_id": clean(row.get("job_id")),
                "experiment": experiment,
                "status": "complete" if payload else "missing_metrics",
                "label_id": label_id,
                "label_name": LABEL_NAMES.get(label_id, label_id),
                "macro_f1": test.get("macro_f1_supported", 0.0),
                "micro_f1": test.get("micro_f1", 0.0),
                "precision": test.get("micro_precision", 0.0),
                "recall": test.get("micro_recall", 0.0),
                "tp": test.get("tp", 0),
                "fp": test.get("fp", 0),
                "fn": test.get("fn", 0),
                "hard_fp": hard_fp,
                "hard_total": hard_total,
                "hard_fp_rate": hard_fp / max(hard_total, 1) if hard_total else "",
                "variant": clean(plan.get("variant")),
                "encoder": clean(plan.get("encoder")),
                "lr": clean(plan.get("lr")),
                "dropout": clean(plan.get("dropout")),
                "bands": clean(plan.get("bands")),
                "crop_seconds": clean(plan.get("crop_seconds")),
                "band_crop_shapes": clean(plan.get("band_crop_shapes")),
                "loss_mode": clean(plan.get("loss_mode")),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
            }
        )
    return out


def build_ensemble_rows(*, members: Mapping[str, Path], split: str, source_kind: str = "ONC") -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for label_id, run_dir in members.items():
        csv_path = run_dir / "train" / f"{split}_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        for row in read_csv(csv_path):
            if source_kind and clean(row.get("source_kind")) != source_kind:
                continue
            key = base_key(row)
            out = merged.get(key)
            if out is None:
                out = {
                    "item_id": key,
                    "source_dataset": clean(row.get("source_dataset")),
                    "source_kind": clean(row.get("source_kind")),
                    "source_audio": clean(row.get("source_audio")),
                    "mat_path": clean(row.get("mat_path")),
                    "low_mat_path": clean(row.get("low_mat_path")),
                    "mid_mat_path": clean(row.get("mid_mat_path")),
                    "high_mat_path": clean(row.get("high_mat_path")),
                    "source_label_ids": clean(row.get("source_label_ids")),
                    "canonical_label_ids": clean(row.get("canonical_label_ids")),
                    "analysis_label_ids": clean(row.get("analysis_label_ids")),
                    "negative_bucket": clean(row.get("negative_bucket")),
                    "split": "val" if split == "validation" else split,
                    "is_background": clean(row.get("is_background")),
                    "review_status": clean(row.get("review_status")),
                    "context_tags": clean(row.get("context_tags")),
                    "begin_s": clean(row.get("begin_s")),
                    "end_s": clean(row.get("end_s")),
                    "event_group": clean(row.get("event_group")),
                    "target_label_ids": "",
                    "pred_label_ids": "",
                }
                for label in THREE_SPECIES:
                    out[f"score__{label}"] = "0.00000000"
                merged[key] = out
            out["target_label_ids"] = merge_labels(out.get("target_label_ids", ""), row.get("target_label_ids", ""))
            if f"score__{label_id}" in row:
                out[f"score__{label_id}"] = clean(row.get(f"score__{label_id}"))
    return list(merged.values())


def hard_fp(summary: Mapping[str, Any]) -> tuple[int, int, float | str]:
    rows = summary.get("onc_test_hard_negative_fp_rows", []) or []
    fp = sum(int(row.get("any_primary_fp") or 0) for row in rows)
    total = sum(int(row.get("rows") or 0) for row in rows)
    return fp, total, fp / total if total else ""


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = clean(value)
        return default if text == "" else float(text)
    except (TypeError, ValueError):
        return default


def thresholds_from_summary(summary: Mapping[str, Any], label_ids: Sequence[str]) -> Dict[str, float]:
    thresholds = summary.get("onc_validation_thresholds", {}) or {}
    out: Dict[str, float] = {}
    for label in label_ids:
        value = thresholds.get(label, {}) if isinstance(thresholds, Mapping) else {}
        if isinstance(value, Mapping):
            out[label] = as_float(value.get("threshold"), 0.5)
        else:
            out[label] = as_float(value, 0.5)
    return out


def prediction_labels(row: Mapping[str, Any], thresholds: Mapping[str, float], label_ids: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for label in label_ids:
        score = as_float(row.get(f"score__{label}"), 0.0)
        if score >= float(thresholds.get(label, 0.5)):
            labels.append(label)
    return labels


def example_base_row(
    row: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
    label_ids: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "item_id": clean(row.get("item_id")),
        "source_dataset": clean(row.get("source_dataset")),
        "source_kind": clean(row.get("source_kind")),
        "source_audio": clean(row.get("source_audio")),
        "mat_path": clean(row.get("mat_path")),
        "low_mat_path": clean(row.get("low_mat_path")),
        "mid_mat_path": clean(row.get("mid_mat_path")),
        "high_mat_path": clean(row.get("high_mat_path")),
        "negative_bucket": clean(row.get("negative_bucket")),
        "begin_s": clean(row.get("begin_s")),
        "end_s": clean(row.get("end_s")),
        "event_group": clean(row.get("event_group")),
        "target_label_ids": clean(row.get("target_label_ids")),
        "pred_label_ids": "|".join(prediction_labels(row, thresholds, label_ids)),
    }
    for label in label_ids:
        out[f"score__{label}"] = clean(row.get(f"score__{label}"))
        out[f"threshold__{label}"] = thresholds.get(label, 0.5)
    return out


def selected_ensemble_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    label_ids: Sequence[str] = THREE_SPECIES,
    max_per_group: int = 20,
) -> List[Dict[str, Any]]:
    thresholds = thresholds_from_summary(summary, label_ids)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        true_labels = set(split_labels(row.get("target_label_ids"))).intersection(label_ids)
        pred_labels = set(prediction_labels(row, thresholds, label_ids))
        for label in label_ids:
            score = as_float(row.get(f"score__{label}"), 0.0)
            threshold = float(thresholds.get(label, 0.5))
            margin = score - threshold
            label_true = label in true_labels
            label_pred = label in pred_labels
            if label_true and label_pred:
                case_type = "true_positive"
            elif label_true and not label_pred:
                case_type = "false_negative"
            elif label_pred and true_labels:
                case_type = "cross_species_false_positive"
            elif label_pred:
                case_type = "background_false_positive"
            else:
                continue
            group = f"{label}:{case_type}"
            item = example_base_row(row, thresholds=thresholds, label_ids=label_ids)
            item.update(
                {
                    "example_group": group,
                    "label_id": label,
                    "label_name": LABEL_NAMES.get(label, label),
                    "case_type": case_type,
                    "score": score,
                    "threshold": threshold,
                    "margin": margin,
                }
            )
            grouped.setdefault(group, []).append(item)
    examples: List[Dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        reverse = not group.endswith(":false_negative")
        ordered = sorted(items, key=lambda item: as_float(item.get("margin")), reverse=reverse)
        examples.extend(ordered[: max(0, int(max_per_group))])
    return examples


def evaluate_ensembles(run_infos: Sequence[Mapping[str, Any]], output_dir: Path, *, max_rank_outputs: int = 20) -> List[Dict[str, Any]]:
    from scripts.analysis.summarize_multilabel_predictions import summarize

    by_label: Dict[str, List[Mapping[str, Any]]] = {label: [] for label in THREE_SPECIES}
    for row in run_infos:
        if row.get("status") != "complete":
            continue
        label = clean(row.get("label_id"))
        if label in by_label:
            by_label[label].append(row)
    if not all(by_label[label] for label in THREE_SPECIES):
        return []

    ensemble_root = output_dir / "ensembles"
    ensemble_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "source_dataset",
        "source_kind",
        "source_audio",
        "mat_path",
        "low_mat_path",
        "mid_mat_path",
        "high_mat_path",
        "source_label_ids",
        "canonical_label_ids",
        "analysis_label_ids",
        "negative_bucket",
        "split",
        "is_background",
        "review_status",
        "context_tags",
        "begin_s",
        "end_s",
        "event_group",
        "target_label_ids",
        "pred_label_ids",
        *[f"score__{label}" for label in THREE_SPECIES],
    ]
    results: List[Dict[str, Any]] = []
    for combo_idx, combo in enumerate(itertools.product(*(by_label[label] for label in THREE_SPECIES)), start=1):
        members = {clean(row["label_id"]): Path(clean(row["run_dir"])) for row in combo}
        combo_name = f"ensemble_{combo_idx:04d}"
        combo_dir = ensemble_root / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)
        val_rows = build_ensemble_rows(members=members, split="validation")
        test_rows = build_ensemble_rows(members=members, split="test")
        validation_csv = combo_dir / "validation_predictions.csv"
        test_csv = combo_dir / "test_predictions.csv"
        write_csv(validation_csv, val_rows, fieldnames=fieldnames)
        write_csv(test_csv, test_rows, fieldnames=fieldnames)
        summary = summarize(
            validation_csv=validation_csv,
            test_csv=test_csv,
            output_dir=combo_dir / "onc_calibrated_eval",
            calibration_source_kind="ONC",
            eval_source_kind="ONC",
            label_ids=THREE_SPECIES,
        )
        examples_csv = combo_dir / "selected_examples.csv"
        write_csv(examples_csv, selected_ensemble_examples(test_rows, summary=summary))
        test = summary.get("onc_test_metrics", {})
        hard_n, hard_total, hard_rate = hard_fp(summary)
        result = {
            "ensemble": combo_name,
            "macro_f1": test.get("macro_f1_supported", 0.0),
            "micro_f1": test.get("micro_f1", 0.0),
            "precision": test.get("micro_precision", 0.0),
            "recall": test.get("micro_recall", 0.0),
            "tp": test.get("tp", 0),
            "fp": test.get("fp", 0),
            "fn": test.get("fn", 0),
            "hard_fp": hard_n,
            "hard_total": hard_total,
            "hard_fp_rate": hard_rate,
            "fin_whale_experiment": clean(combo[0].get("experiment")),
            "blue_whale_experiment": clean(combo[1].get("experiment")),
            "humpback_whale_experiment": clean(combo[2].get("experiment")),
            "ensemble_dir": str(combo_dir),
            "examples_csv": str(examples_csv),
        }
        results.append(result)

    ranked = sorted(
        results,
        key=lambda row: (
            float(row.get("macro_f1") or 0.0),
            -float(row.get("hard_fp_rate") or 1.0),
            float(row.get("micro_f1") or 0.0),
        ),
        reverse=True,
    )
    keep = {row["ensemble"] for row in ranked[:max_rank_outputs]}
    for row in results:
        if row["ensemble"] not in keep:
            combo_dir = Path(str(row["ensemble_dir"]))
            for child in combo_dir.glob("*.csv"):
                child.unlink(missing_ok=True)
    return ranked


def variant_rows(variant_root: Optional[Path]) -> List[Dict[str, Any]]:
    if variant_root is None:
        return []
    payload = load_json(variant_root / "variant_index.json")
    if not isinstance(payload, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        split_counts = item.get("split_counts", {}) or {}
        rows.append(
            {
                "variant": item.get("variant_name", ""),
                "rows": item.get("row_count", 0),
                "train_rows": split_counts.get("train", 0),
                "val_rows": split_counts.get("val", 0),
                "test_rows": split_counts.get("test", 0),
                "bands": ",".join(item.get("bands", [])),
                "sources": ",".join(item.get("sources", [])),
                "cap_strategy": (item.get("cap_summary", {}) or {}).get("strategy", ""),
                "manifest": item.get("manifest_csv", ""),
            }
        )
    return rows


def markdown_report(
    *,
    individual: Sequence[Mapping[str, Any]],
    ensembles: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    lines = [
        "# E24 Expert-Ensemble Hyperparameter Report",
        "",
        "E24 optimizes the strategy that worked best in E22: independent fin whale, blue whale, and humpback whale experts combined with ONC-validation calibrated posthoc ensembling.",
        "",
        "## Best Ensembles",
        "",
        "| rank | macro F1 | micro F1 | precision | recall | hard FP | fin whale | blue whale | humpback whale |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for rank, row in enumerate(ensembles[:20], start=1):
        hard = row.get("hard_fp_rate")
        hard_text = "" if hard == "" else f"{float(hard):.4f}"
        lines.append(
            "| {rank} | {macro:.4f} | {micro:.4f} | {precision:.4f} | {recall:.4f} | {hard} | {fin} | {blue} | {hump} |".format(
                rank=rank,
                macro=float(row.get("macro_f1") or 0.0),
                micro=float(row.get("micro_f1") or 0.0),
                precision=float(row.get("precision") or 0.0),
                recall=float(row.get("recall") or 0.0),
                hard=hard_text,
                fin=row.get("fin_whale_experiment", ""),
                blue=row.get("blue_whale_experiment", ""),
                hump=row.get("humpback_whale_experiment", ""),
            )
        )

    lines.extend(["", "## Best Individual Experts", ""])
    for label in THREE_SPECIES:
        label_rows = [row for row in individual if clean(row.get("label_id")) == label and row.get("status") == "complete"]
        label_rows = sorted(label_rows, key=lambda row: float(row.get("macro_f1") or 0.0), reverse=True)
        lines.append(f"### {LABEL_NAMES[label].title()}")
        lines.append("")
        lines.append("| rank | F1 | precision | recall | hard FP | experiment |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for rank, row in enumerate(label_rows[:8], start=1):
            hard = row.get("hard_fp_rate")
            hard_text = "" if hard == "" else f"{float(hard):.4f}"
            lines.append(
                f"| {rank} | {float(row.get('macro_f1') or 0.0):.4f} | {float(row.get('precision') or 0.0):.4f} | {float(row.get('recall') or 0.0):.4f} | {hard_text} | {row.get('experiment', '')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Variant Checks",
            "",
            "| variant | rows | train | val | test | cap |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in variants:
        lines.append(
            f"| {row.get('variant')} | {row.get('rows')} | {row.get('train_rows')} | {row.get('val_rows')} | {row.get('test_rows')} | {row.get('cap_strategy')} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison Targets",
            "",
            "- E22 three-species expert ensemble: macro F1 0.9088, hard-negative FP 0.1826.",
            "- Small 40s best: macro F1 about 0.5580.",
            "- E01/E09 ONC baselines: macro F1 about 0.637-0.639.",
            "",
            f"Individual metrics CSV: `{output_dir / 'e24_individual_metrics.csv'}`",
            f"Ensemble rankings CSV: `{output_dir / 'e24_ensemble_rankings.csv'}`",
            f"Selected example CSVs: `{output_dir / 'ensembles'}/<ensemble>/selected_examples.csv` for retained ensembles",
            f"Variant CSV: `{output_dir / 'e24_variant_summary.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submitted-tsv", required=True, type=Path)
    parser.add_argument("--plan-tsv", required=True, type=Path)
    parser.add_argument("--variant-root", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submitted = read_tsv(args.submitted_tsv)
    plan = read_tsv(args.plan_tsv)
    individual = run_rows(submitted, plan)
    variants = variant_rows(args.variant_root)
    ensembles = evaluate_ensembles(individual, args.output_dir)
    write_csv(args.output_dir / "e24_individual_metrics.csv", individual)
    write_csv(args.output_dir / "e24_ensemble_rankings.csv", ensembles)
    write_csv(args.output_dir / "e24_variant_summary.csv", variants)
    report = markdown_report(individual=individual, ensembles=ensembles, variants=variants, output_dir=args.output_dir)
    report_path = args.output_dir / "e24_expert_hparam_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "individual_runs": len(individual), "ensembles": len(ensembles)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
