#!/usr/bin/env python3
"""Evaluate Part 2 fin-whale predictions and build the Markdown report bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.part2_eval import (
    PART2_BUCKETS,
    bucket_event_metrics,
    bucket_coverage_metrics,
    build_clip_confusion,
    context_recall_rows,
    coverage_match_sets,
    coverage_metrics,
    evaluation_report_lines,
    filter_prediction_items_for_rapid_review,
    filter_predictions_by_score,
    hardest_context_rows,
    load_annotations_csv,
    load_clip_manifest_csv,
    load_prediction_segments,
    match_predictions_to_annotations,
    maybe_plot_bucket_recall_comparison,
    maybe_plot_confusion_matrix,
    maybe_plot_sweep_curve,
    maybe_plot_view_summary,
    rapid_review_rows,
    recommendations_from_errors,
    strict_o3_subset,
    summarize_sweep_rows,
    write_csv,
)
from src.dataset.part2_examples import export_part2_example_gallery
from src.utils.wandb_utils import finish_run, init_wandb_test, save_wandb_files, update_wandb_summary


def _parse_float_list(raw: str, *, allow_empty: bool = False) -> List[float]:
    values = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not values and allow_empty:
        return []
    if not values:
        raise ValueError("Expected at least one numeric value")
    return [float(token) for token in values]


def _parse_int_list(raw: str) -> List[int]:
    values = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return [int(token) for token in values]


def _parse_optional_float_list(raw: str) -> List[Optional[float]]:
    tokens = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not tokens:
        return [None]
    out: List[Optional[float]] = []
    for token in tokens:
        if token.lower() in {"auto", "none", "null"}:
            out.append(None)
        else:
            out.append(float(token))
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_baseline_summary(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _match_rows(matches, annotations_by_id, predictions_by_id) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in matches:
        ann = annotations_by_id[match.annotation_id]
        pred = predictions_by_id[match.prediction_id]
        rows.append(
            {
                "annotation_id": ann.annotation_id,
                "prediction_id": pred.prediction_id,
                "item_id": pred.item_id,
                "filename": match.filename,
                "call_type_bucket": ann.call_type_bucket,
                "call_type_raw": ann.call_type_raw,
                "annotation_start_s": f"{ann.begin_time_s:.6f}",
                "annotation_end_s": f"{ann.end_time_s:.6f}",
                "prediction_start_s": f"{pred.start_time_s:.6f}",
                "prediction_end_s": f"{pred.end_time_s:.6f}",
                "score": f"{pred.score:.6f}",
                "overlap_s": f"{match.overlap_s:.6f}",
                "context_tags": "|".join(ann.context_tags),
            }
        )
    return rows


def _unmatched_annotation_rows(annotations) -> List[Dict[str, Any]]:
    return [
        {
            "annotation_id": ann.annotation_id,
            "filename": ann.filename,
            "call_type_bucket": ann.call_type_bucket,
            "call_type_raw": ann.call_type_raw,
            "begin_time_s": f"{ann.begin_time_s:.6f}",
            "end_time_s": f"{ann.end_time_s:.6f}",
            "context_tags": "|".join(ann.context_tags),
            "comments": ann.comments,
        }
        for ann in annotations
    ]


def _unmatched_prediction_rows(predictions, clip_manifest) -> List[Dict[str, Any]]:
    rows = rapid_review_rows(predictions, clip_manifest)
    return rows


def _write_confusion_csv(path: Path, metrics: Dict[str, Any]) -> None:
    rows = [
        {
            "tp": metrics.get("tp", 0),
            "fp": metrics.get("fp", 0),
            "fn": metrics.get("fn", 0),
            "tn": metrics.get("tn", 0),
            "precision": f"{float(metrics.get('precision', 0.0)):.6f}",
            "recall": f"{float(metrics.get('recall', 0.0)):.6f}",
            "f1": f"{float(metrics.get('f1', 0.0)):.6f}",
            "accuracy": f"{float(metrics.get('accuracy', 0.0)):.6f}",
        }
    ]
    write_csv(path, rows)


def _overall_view_rows(
    *,
    strict_event_metrics: Dict[str, Any],
    merged_region_metrics: Dict[str, Any],
    raw_window_metrics: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [
        {
            "view": "merged_region_coverage",
            "view_label": "Merged Clip Coverage",
            "precision": merged_region_metrics.get("precision", 0.0),
            "recall": merged_region_metrics.get("recall", 0.0),
            "f1": merged_region_metrics.get("f1", 0.0),
        },
    ]
    if raw_window_metrics is not None:
        rows.append(
            {
                "view": "raw_window_coverage",
                "view_label": "Raw Window Detection",
                "precision": raw_window_metrics.get("precision", 0.0),
                "recall": raw_window_metrics.get("recall", 0.0),
                "f1": raw_window_metrics.get("f1", 0.0),
            }
        )
    rows.append(
        {
            "view": "strict_event",
            "view_label": "Strict Single-Call Extraction",
            "precision": strict_event_metrics.get("precision", 0.0),
            "recall": strict_event_metrics.get("recall", 0.0),
            "f1": strict_event_metrics.get("f1", 0.0),
        }
    )
    return rows


def _sweep_row(
    *,
    tag: str,
    combo_dir: Path,
    window_step_label: str,
    low_threshold: float,
    high_threshold: float,
    min_members: int,
    max_gap_seconds: Optional[float],
    eval_summary: Dict[str, Any],
    raw_window_threshold: float,
) -> Dict[str, Any]:
    strict_metrics = eval_summary["overall_event_metrics"]
    merged_metrics = eval_summary["merged_region_metrics"]
    raw_metrics = eval_summary.get("raw_window_metrics") or {}
    return {
        "tag": tag,
        "combo_dir": str(combo_dir),
        "window_step": window_step_label,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "min_members": min_members,
        "max_gap_seconds": "" if max_gap_seconds is None else max_gap_seconds,
        **strict_metrics,
        "merged_region_precision": merged_metrics.get("precision", 0.0),
        "merged_region_recall": merged_metrics.get("recall", 0.0),
        "merged_region_f1": merged_metrics.get("f1", 0.0),
        "raw_window_threshold": raw_window_threshold,
        "raw_window_precision": raw_metrics.get("precision", ""),
        "raw_window_recall": raw_metrics.get("recall", ""),
        "raw_window_f1": raw_metrics.get("f1", ""),
        "prediction_count": merged_metrics.get("prediction_count", 0),
        "covered_annotation_count": merged_metrics.get("covered_annotation_count", 0),
        "total_review_minutes": merged_metrics.get("total_review_minutes", 0.0),
    }


def evaluate_single_postprocessed_output(
    *,
    postprocessed_json: Path,
    annotations,
    all_annotations,
    clip_manifest,
    output_dir: Path,
    match_collar_s: float,
    raw_window_predictions: Optional[Sequence[Any]] = None,
    raw_window_payload: Optional[Dict[str, Any]] = None,
    raw_window_json_path: Optional[Path] = None,
    raw_window_threshold: Optional[float] = None,
    baseline_summary: Optional[Dict[str, Any]] = None,
    summary_title: str = "Fin Whale Part 2 Report",
    sweep_summary_rows: Optional[Sequence[Dict[str, Any]]] = None,
    example_mat_dir: Optional[Path] = None,
    export_example_images: bool = False,
    max_examples_per_group: int = 8,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, prediction_segments = load_prediction_segments(postprocessed_json)

    fin_annotations = [ann for ann in annotations if ann.species == "Bp"]
    matches, unmatched_predictions, unmatched_annotations = match_predictions_to_annotations(
        prediction_segments,
        fin_annotations,
        match_collar_s,
    )
    coverage_useful_prediction_ids, coverage_covered_annotation_ids = coverage_match_sets(
        prediction_segments,
        fin_annotations,
        match_collar_s,
    )
    coverage_unmatched_predictions = [
        pred for pred in prediction_segments if pred.prediction_id not in coverage_useful_prediction_ids
    ]
    coverage_unmatched_annotations = [
        ann for ann in fin_annotations if ann.annotation_id not in coverage_covered_annotation_ids
    ]

    overall_event = {
        "tp": len(matches),
        "fp": len(unmatched_predictions),
        "fn": len(unmatched_annotations),
    }
    precision = overall_event["tp"] / (overall_event["tp"] + overall_event["fp"]) if (overall_event["tp"] + overall_event["fp"]) else 0.0
    recall = overall_event["tp"] / (overall_event["tp"] + overall_event["fn"]) if (overall_event["tp"] + overall_event["fn"]) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    overall_event.update({"precision": precision, "recall": recall, "f1": f1})

    merged_region_metrics = coverage_metrics(prediction_segments, fin_annotations, match_collar_s)
    overall_clip = build_clip_confusion(clip_manifest, prediction_segments)
    bucket_events = bucket_event_metrics(prediction_segments, fin_annotations, match_collar_s)
    bucket_merged_region = bucket_coverage_metrics(prediction_segments, fin_annotations, match_collar_s)
    bucket_clips = {
        bucket: build_clip_confusion(clip_manifest, prediction_segments, bucket=bucket)
        for bucket in PART2_BUCKETS
    }

    raw_window_positive_predictions = None
    raw_window_metrics = None
    bucket_raw_window = None
    if raw_window_predictions is not None and raw_window_threshold is not None:
        raw_window_positive_predictions = filter_predictions_by_score(raw_window_predictions, float(raw_window_threshold))
        raw_window_metrics = coverage_metrics(raw_window_positive_predictions, fin_annotations, match_collar_s)
        bucket_raw_window = bucket_coverage_metrics(raw_window_positive_predictions, fin_annotations, match_collar_s)

    annotations_by_id = {ann.annotation_id: ann for ann in fin_annotations}
    predictions_by_id = {pred.prediction_id: pred for pred in prediction_segments}

    write_csv(output_dir / "matches.csv", _match_rows(matches, annotations_by_id, predictions_by_id))
    write_csv(output_dir / "strict_false_negatives.csv", _unmatched_annotation_rows(unmatched_annotations))
    write_csv(output_dir / "strict_false_positives.csv", _unmatched_prediction_rows(unmatched_predictions, clip_manifest))
    write_csv(output_dir / "coverage_missed_annotations.csv", _unmatched_annotation_rows(coverage_unmatched_annotations))
    write_csv(
        output_dir / "coverage_false_positives.csv",
        _unmatched_prediction_rows(coverage_unmatched_predictions, clip_manifest),
    )
    write_csv(output_dir / "false_negatives.csv", _unmatched_annotation_rows(coverage_unmatched_annotations))
    write_csv(
        output_dir / "false_positives.csv",
        _unmatched_prediction_rows(coverage_unmatched_predictions, clip_manifest),
    )
    write_csv(output_dir / "rapid_review.csv", rapid_review_rows(coverage_unmatched_predictions, clip_manifest))

    _write_confusion_csv(output_dir / "overall_clip_confusion.csv", overall_clip)
    maybe_plot_confusion_matrix(output_dir / "overall_clip_confusion.png", overall_clip, "Part 2 overall clip confusion")

    overall_view_rows = _overall_view_rows(
        strict_event_metrics=overall_event,
        merged_region_metrics=merged_region_metrics,
        raw_window_metrics=raw_window_metrics,
    )
    write_csv(output_dir / "overall_view_metrics.csv", overall_view_rows)
    maybe_plot_view_summary(output_dir / "overall_view_metrics.png", overall_view_rows)

    bucket_metric_rows: List[Dict[str, Any]] = []
    for bucket in PART2_BUCKETS:
        clip_metrics = bucket_clips[bucket]
        event_metrics = bucket_events[bucket]
        merged_metrics = bucket_merged_region[bucket]
        raw_metrics = bucket_raw_window[bucket] if bucket_raw_window is not None else None
        _write_confusion_csv(output_dir / f"{bucket}_clip_confusion.csv", clip_metrics)
        maybe_plot_confusion_matrix(
            output_dir / f"{bucket}_clip_confusion.png",
            clip_metrics,
            f"Part 2 {bucket} clip confusion",
        )
        bucket_metric_rows.append(
            {
                "bucket": bucket,
                "event_tp": event_metrics["tp"],
                "event_fp": event_metrics["fp"],
                "event_fn": event_metrics["fn"],
                "event_precision": f"{event_metrics['precision']:.6f}",
                "event_recall": f"{event_metrics['recall']:.6f}",
                "event_f1": f"{event_metrics['f1']:.6f}",
                "merged_region_precision": f"{merged_metrics['precision']:.6f}",
                "merged_region_recall": f"{merged_metrics['recall']:.6f}",
                "merged_region_f1": f"{merged_metrics['f1']:.6f}",
                "merged_region_covered_calls": int(merged_metrics["covered_annotation_count"]),
                "merged_region_review_minutes": f"{merged_metrics['total_review_minutes']:.6f}",
                "raw_window_precision": f"{raw_metrics['precision']:.6f}" if raw_metrics is not None else "",
                "raw_window_recall": f"{raw_metrics['recall']:.6f}" if raw_metrics is not None else "",
                "raw_window_f1": f"{raw_metrics['f1']:.6f}" if raw_metrics is not None else "",
                "raw_window_covered_calls": int(raw_metrics["covered_annotation_count"]) if raw_metrics is not None else "",
                "raw_window_review_minutes": f"{raw_metrics['total_review_minutes']:.6f}" if raw_metrics is not None else "",
                "clip_tp": clip_metrics["tp"],
                "clip_fp": clip_metrics["fp"],
                "clip_fn": clip_metrics["fn"],
                "clip_tn": clip_metrics["tn"],
                "clip_precision": f"{clip_metrics['precision']:.6f}",
                "clip_recall": f"{clip_metrics['recall']:.6f}",
                "clip_f1": f"{clip_metrics['f1']:.6f}",
            }
        )
    write_csv(output_dir / "bucket_metrics.csv", bucket_metric_rows)
    if bucket_raw_window is not None:
        maybe_plot_bucket_recall_comparison(
            output_dir / "bucket_recall_comparison.png",
            strict_bucket_metrics=bucket_events,
            merged_bucket_metrics=bucket_merged_region,
            raw_bucket_metrics=bucket_raw_window,
        )

    recommendations = recommendations_from_errors(
        unmatched_annotations=coverage_unmatched_annotations,
        unmatched_predictions=coverage_unmatched_predictions,
        clip_manifest=clip_manifest,
    )
    with open(output_dir / "recommendations.md", "w", encoding="utf-8") as handle:
        for line in recommendations:
            handle.write(f"- {line}\n")

    rapid_payload = filter_prediction_items_for_rapid_review(payload, coverage_unmatched_predictions)
    rapid_app_json = output_dir / "rapid_review.app.json"
    rapid_o3_json = output_dir / "rapid_review.o3.json"
    _write_json(rapid_app_json, rapid_payload)
    _write_json(rapid_o3_json, strict_o3_subset(rapid_payload))

    context_rows = []
    context_rows.extend(
        context_recall_rows(
            predictions=prediction_segments,
            annotations=fin_annotations,
            collar_s=match_collar_s,
            view_name="strict_event",
        )
    )
    context_rows.extend(
        context_recall_rows(
            predictions=prediction_segments,
            annotations=fin_annotations,
            collar_s=match_collar_s,
            view_name="merged_region_coverage",
        )
    )
    if raw_window_positive_predictions is not None:
        context_rows.extend(
            context_recall_rows(
                predictions=raw_window_positive_predictions,
                annotations=fin_annotations,
                collar_s=match_collar_s,
                view_name="raw_window_coverage",
            )
        )
    write_csv(output_dir / "context_recall_metrics.csv", context_rows)
    hardest_rows = hardest_context_rows(context_rows, max_items=8, min_annotation_count=25)

    examples_summary = None
    if export_example_images and example_mat_dir is not None:
        examples_summary = export_part2_example_gallery(
            output_dir=output_dir / "examples",
            postprocessed_json_path=postprocessed_json,
            postprocessed_payload=payload,
            merged_predictions=prediction_segments,
            merged_useful_prediction_ids=coverage_useful_prediction_ids,
            merged_unmatched_predictions=coverage_unmatched_predictions,
            merged_missed_annotations=coverage_unmatched_annotations,
            raw_window_json_path=raw_window_json_path,
            raw_window_payload=raw_window_payload,
            raw_window_predictions=raw_window_predictions,
            raw_window_threshold=raw_window_threshold,
            annotations=fin_annotations,
            all_annotations=all_annotations,
            clip_manifest=clip_manifest,
            mat_dir=example_mat_dir,
            max_examples_per_group=max_examples_per_group,
            match_collar_s=match_collar_s,
        )

    summary_payload = {
        "postprocessed_json": str(postprocessed_json),
        "match_collar_s": float(match_collar_s),
        "overall_event_metrics": overall_event,
        "merged_region_metrics": merged_region_metrics,
        "raw_window_metrics": raw_window_metrics,
        "overall_clip_metrics": overall_clip,
        "bucket_event_metrics": bucket_events,
        "bucket_merged_region_metrics": bucket_merged_region,
        "bucket_raw_window_metrics": bucket_raw_window,
        "bucket_clip_metrics": bucket_clips,
        "recommendations": recommendations,
        "rapid_review_count": len(coverage_unmatched_predictions),
        "raw_window_threshold": float(raw_window_threshold) if raw_window_threshold is not None else None,
        "coverage_false_positive_count": len(coverage_unmatched_predictions),
        "coverage_missed_annotation_count": len(coverage_unmatched_annotations),
        "examples_summary": examples_summary,
    }
    _write_json(output_dir / "metrics.json", summary_payload)

    report_lines = evaluation_report_lines(
        summary_title=summary_title,
        strict_event_metrics=overall_event,
        merged_region_metrics=merged_region_metrics,
        raw_window_metrics=raw_window_metrics,
        overall_clip_metrics=overall_clip,
        bucket_strict_event_metrics_map=bucket_events,
        bucket_merged_region_metrics_map=bucket_merged_region,
        bucket_raw_window_metrics_map=bucket_raw_window,
        bucket_clip_metrics_map=bucket_clips,
        recommendations=recommendations,
        rapid_review_count=len(coverage_unmatched_predictions),
        baseline_summary=baseline_summary,
        sweep_summary_rows=sweep_summary_rows,
        hardest_context_rows=hardest_rows,
    )
    if examples_summary is not None:
        report_lines.extend(
            [
                "",
                "## Example Spectrogram Gallery",
                "",
                "Representative merged-region and raw-window spectrogram examples are exported under `examples/`.",
                "",
                "- Gallery README: `examples/README.md`",
                "- Combined example manifest: `examples/examples_index.csv`",
                "- Contact sheets live in `examples/contact_sheets/`",
                "",
            ]
        )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "overall_event_metrics": overall_event,
        "merged_region_metrics": merged_region_metrics,
        "raw_window_metrics": raw_window_metrics,
        "overall_clip_metrics": overall_clip,
        "bucket_event_metrics": bucket_events,
        "bucket_merged_region_metrics": bucket_merged_region,
        "bucket_raw_window_metrics": bucket_raw_window,
        "bucket_clip_metrics": bucket_clips,
        "recommendations": recommendations,
        "rapid_review_count": len(coverage_unmatched_predictions),
        "metrics_json": str(output_dir / "metrics.json"),
        "report_md": str(report_path),
        "examples_summary": examples_summary,
    }


def _run_postprocess(
    *,
    input_json: Path,
    output_json: Path,
    low_threshold: float,
    high_threshold: float,
    min_members: int,
    max_gap_seconds: Optional[float],
    class_hierarchy: Optional[str],
    merge_event_media: bool,
    event_media_dir: Optional[Path],
) -> None:
    script = REPO_ROOT / "scripts" / "inference" / "postprocess_predictions.py"
    cmd = [
        sys.executable,
        str(script),
        "--input-json",
        str(input_json),
        "--output-json",
        str(output_json),
        "--low-threshold",
        str(low_threshold),
        "--high-threshold",
        str(high_threshold),
        "--min-members",
        str(min_members),
        "--replace-items-with-events",
    ]
    if class_hierarchy:
        cmd.extend(["--class-hierarchy", class_hierarchy])
    if max_gap_seconds is not None:
        cmd.extend(["--max-gap-seconds", str(max_gap_seconds)])
    if merge_event_media:
        cmd.append("--merge-event-media")
        if event_media_dir is not None:
            cmd.extend(["--event-media-dir", str(event_media_dir)])
    subprocess.run(cmd, check=True)


def _sweep_candidates(
    *,
    window_predictions_json: Path,
    output_dir: Path,
    window_step_label: str,
    low_thresholds: Sequence[float],
    high_thresholds: Sequence[float],
    min_members_values: Sequence[int],
    max_gap_values: Sequence[Optional[float]],
    class_hierarchy: Optional[str],
    annotations,
    all_annotations,
    clip_manifest,
    match_collar_s: float,
    raw_window_predictions: Sequence[Any],
) -> List[Dict[str, Any]]:
    sweep_rows: List[Dict[str, Any]] = []
    sweep_dir = output_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    combo_index = 0
    for low_threshold in low_thresholds:
        for high_threshold in high_thresholds:
            if high_threshold < low_threshold:
                continue
            for min_members in min_members_values:
                for max_gap_seconds in max_gap_values:
                    combo_index += 1
                    tag = (
                        f"low{str(low_threshold).replace('.', 'p')}_"
                        f"high{str(high_threshold).replace('.', 'p')}_"
                        f"min{min_members}_"
                        f"gap{'auto' if max_gap_seconds is None else str(max_gap_seconds).replace('.', 'p')}"
                    )
                    combo_dir = sweep_dir / tag
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    combo_json = combo_dir / "predictions_postprocessed.json"
                    _run_postprocess(
                        input_json=window_predictions_json,
                        output_json=combo_json,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        min_members=min_members,
                        max_gap_seconds=max_gap_seconds,
                        class_hierarchy=class_hierarchy,
                        merge_event_media=False,
                        event_media_dir=None,
                    )
                    eval_summary = evaluate_single_postprocessed_output(
                        postprocessed_json=combo_json,
                        annotations=annotations,
                        all_annotations=all_annotations,
                        clip_manifest=clip_manifest,
                        output_dir=combo_dir / "eval",
                        match_collar_s=match_collar_s,
                        raw_window_predictions=raw_window_predictions,
                        raw_window_payload=None,
                        raw_window_json_path=None,
                        raw_window_threshold=low_threshold,
                        baseline_summary=None,
                        summary_title=f"Part 2 sweep evaluation: {tag}",
                    )
                    row = _sweep_row(
                        tag=tag,
                        combo_dir=combo_dir,
                        window_step_label=window_step_label,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        min_members=min_members,
                        max_gap_seconds=max_gap_seconds,
                        eval_summary=eval_summary,
                        raw_window_threshold=low_threshold,
                    )
                    sweep_rows.append(row)
                    print(
                        "sweep",
                        combo_index,
                        tag,
                        f"precision={row['precision']:.4f}",
                        f"recall={row['recall']:.4f}",
                        f"f1={row['f1']:.4f}",
                        f"coverage_recall={float(row['merged_region_recall']):.4f}",
                        f"window_recall={float(row['raw_window_recall']) if row['raw_window_recall'] != '' else 0.0:.4f}",
                    )
    write_csv(output_dir / "sweep_summary.csv", summarize_sweep_rows(sweep_rows))
    maybe_plot_sweep_curve(
        output_dir / "sweep_precision_recall.png",
        sweep_rows,
        window_step_label or "window predictions",
    )
    return sweep_rows


def _select_operating_points(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not rows:
        raise ValueError("No sweep rows were generated")

    def _best(key: str, secondary: str) -> Dict[str, Any]:
        return max(
            rows,
            key=lambda row: (
                float(row.get(key, 0.0)),
                float(row.get(secondary, 0.0)),
                -float(row.get("prediction_count", row.get("fp", 0.0))),
            ),
        )

    out = {
        "best_f1": _best("merged_region_f1", "merged_region_precision"),
        "high_recall": _best("merged_region_recall", "merged_region_precision"),
        "high_precision": _best("merged_region_precision", "merged_region_recall"),
        "best_strict_f1": _best("f1", "precision"),
        "best_window_recall": _best("raw_window_recall", "raw_window_precision"),
    }

    unique: Dict[str, Dict[str, Any]] = {}
    for label, row in out.items():
        unique.setdefault(row["tag"], row)
    return {
        label: unique[row["tag"]] if row["tag"] in unique else row
        for label, row in out.items()
    }


def _copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _log_part2_wandb(
    *,
    output_dir: Path,
    selected_rows: Sequence[Dict[str, Any]],
    best_metrics_json: Path,
    sweep_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not args.use_wandb:
        return

    config = {
        "annotations_csv": args.annotations_csv,
        "clip_manifest_csv": args.clip_manifest_csv,
        "match_collar_s": float(args.match_collar_s),
        "window_step_label": args.window_step_label,
        "low_thresholds": args.low_thresholds,
        "high_thresholds": args.high_thresholds,
        "min_members_values": args.min_members_values,
        "max_gap_values": args.max_gap_values,
        "class_hierarchy": args.class_hierarchy,
        "mode": "postprocessed" if args.postprocessed_json else "sweep",
    }
    init_wandb_test(
        project_name=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        run_name=args.wandb_name or f"part2_eval_{output_dir.name}",
        config=config,
        out_dir=str(output_dir),
        tags=args.wandb_tags,
        job_type="part2_evaluation",
    )
    try:
        with open(best_metrics_json, "r", encoding="utf-8") as handle:
            metrics_payload = json.load(handle)
        strict_metrics = metrics_payload.get("overall_event_metrics", {})
        merged_metrics = metrics_payload.get("merged_region_metrics", {})
        raw_metrics = metrics_payload.get("raw_window_metrics") or {}
        clip_metrics = metrics_payload.get("overall_clip_metrics", {})
        summary = {
            "best/strict_event_precision": float(strict_metrics.get("precision", 0.0)),
            "best/strict_event_recall": float(strict_metrics.get("recall", 0.0)),
            "best/strict_event_f1": float(strict_metrics.get("f1", 0.0)),
            "best/merged_region_precision": float(merged_metrics.get("precision", 0.0)),
            "best/merged_region_recall": float(merged_metrics.get("recall", 0.0)),
            "best/merged_region_f1": float(merged_metrics.get("f1", 0.0)),
            "best/clip_precision": float(clip_metrics.get("precision", 0.0)),
            "best/clip_recall": float(clip_metrics.get("recall", 0.0)),
            "best/clip_f1": float(clip_metrics.get("f1", 0.0)),
        }
        if raw_metrics:
            summary.update(
                {
                    "best/raw_window_precision": float(raw_metrics.get("precision", 0.0)),
                    "best/raw_window_recall": float(raw_metrics.get("recall", 0.0)),
                    "best/raw_window_f1": float(raw_metrics.get("f1", 0.0)),
                }
            )
        update_wandb_summary(summary)

        try:
            import wandb

            op_table = wandb.Table(
                columns=list(selected_rows[0].keys()) if selected_rows else ["label"],
                data=[[row.get(col) for col in (list(selected_rows[0].keys()) if selected_rows else ["label"])] for row in selected_rows] if selected_rows else [],
            )
            wandb.log({"operating_points": op_table})
            if sweep_rows:
                sweep_columns = list(sweep_rows[0].keys())
                sweep_table = wandb.Table(columns=sweep_columns)
                for row in sweep_rows:
                    sweep_table.add_data(*[row.get(col) for col in sweep_columns])
                wandb.log({"sweep_summary": sweep_table})

            for image_name in (
                "overall_view_metrics.png",
                "overall_clip_confusion.png",
                "bucket_recall_comparison.png",
                "sweep_precision_recall.png",
            ):
                image_path = output_dir / image_name
                if image_path.exists():
                    wandb.log({image_name.replace(".png", ""): wandb.Image(str(image_path))})
            examples_dir = output_dir / "examples" / "contact_sheets"
            if examples_dir.exists():
                for image_path in sorted(examples_dir.glob("*.png")):
                    wandb.log({f"examples/{image_path.stem}": wandb.Image(str(image_path))})
        except Exception as exc:
            print(f"Warning: Could not log rich Part 2 artifacts to wandb: {exc}")

        save_wandb_files(
            [
                output_dir / "report.md",
                output_dir / "metrics.json",
                output_dir / "overall_view_metrics.csv",
                output_dir / "bucket_metrics.csv",
                output_dir / "context_recall_metrics.csv",
                output_dir / "selected_operating_points.csv",
                output_dir / "sweep_summary.csv",
                output_dir / "rapid_review.csv",
                output_dir / "recommendations.md",
                output_dir / "examples" / "README.md",
                output_dir / "examples" / "examples_index.csv",
            ],
            base_path=output_dir,
        )
    finally:
        finish_run()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Part 2 predictions and build the report bundle")
    ap.add_argument("--annotations-csv", type=str, required=True)
    ap.add_argument("--all-annotations-csv", type=str, default=None)
    ap.add_argument("--clip-manifest-csv", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--postprocessed-json", type=str, default=None)
    ap.add_argument("--window-predictions-json", type=str, default=None)
    ap.add_argument("--match-collar-s", type=float, default=1.0)
    ap.add_argument("--class-hierarchy", type=str, default=None)
    ap.add_argument("--baseline-metrics-json", type=str, default=None)
    ap.add_argument("--window-step-label", type=str, default="")
    ap.add_argument("--low-thresholds", type=str, default="0.70,0.75,0.80")
    ap.add_argument("--high-thresholds", type=str, default="0.82,0.85,0.90")
    ap.add_argument("--min-members-values", type=str, default="2,3")
    ap.add_argument("--max-gap-values", type=str, default="auto,10,15")
    ap.add_argument("--merge-event-media", action="store_true")
    ap.add_argument("--example-mat-dir", type=str, default=None)
    ap.add_argument("--export-example-images", action="store_true")
    ap.add_argument("--max-examples-per-group", type=int, default=8)
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="whale-call-analysis")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-group", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument("--wandb-tags", type=str, default=None)
    args = ap.parse_args()

    if bool(args.postprocessed_json) == bool(args.window_predictions_json):
        raise SystemExit("Provide exactly one of --postprocessed-json or --window-predictions-json")

    annotations = load_annotations_csv(args.annotations_csv)
    all_annotations = load_annotations_csv(args.all_annotations_csv) if args.all_annotations_csv else annotations
    clip_manifest = load_clip_manifest_csv(args.clip_manifest_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = _load_baseline_summary(args.baseline_metrics_json)

    raw_window_predictions = None
    raw_window_payload = None
    if args.window_predictions_json:
        raw_window_payload, raw_window_predictions = load_prediction_segments(args.window_predictions_json)

    if args.postprocessed_json:
        evaluate_single_postprocessed_output(
            postprocessed_json=Path(args.postprocessed_json),
            annotations=annotations,
            all_annotations=all_annotations,
            clip_manifest=clip_manifest,
            output_dir=output_dir,
            match_collar_s=float(args.match_collar_s),
            raw_window_predictions=raw_window_predictions,
            raw_window_payload=raw_window_payload,
            raw_window_json_path=Path(args.window_predictions_json) if args.window_predictions_json else None,
            raw_window_threshold=min(_parse_float_list(args.low_thresholds)),
            baseline_summary=baseline_summary,
            example_mat_dir=Path(args.example_mat_dir) if args.example_mat_dir else None,
            export_example_images=bool(args.export_example_images),
            max_examples_per_group=int(args.max_examples_per_group),
        )
        if args.use_wandb:
            _log_part2_wandb(
                output_dir=output_dir,
                selected_rows=[],
                best_metrics_json=output_dir / "metrics.json",
                sweep_rows=[],
                args=args,
            )
        return

    sweep_rows = _sweep_candidates(
        window_predictions_json=Path(args.window_predictions_json),
        output_dir=output_dir,
        window_step_label=args.window_step_label,
        low_thresholds=_parse_float_list(args.low_thresholds),
        high_thresholds=_parse_float_list(args.high_thresholds),
        min_members_values=_parse_int_list(args.min_members_values),
        max_gap_values=_parse_optional_float_list(args.max_gap_values),
        class_hierarchy=args.class_hierarchy,
        annotations=annotations,
        all_annotations=all_annotations,
        clip_manifest=clip_manifest,
        match_collar_s=float(args.match_collar_s),
        raw_window_predictions=raw_window_predictions or [],
    )
    selected = _select_operating_points(sweep_rows)
    selected_rows = [{"label": label, **row} for label, row in selected.items()]
    write_csv(output_dir / "selected_operating_points.csv", selected_rows)

    selected_dirs: Dict[str, Path] = {}
    for label, row in selected.items():
        op_dir = output_dir / label
        op_json = op_dir / "predictions_postprocessed.json"
        _run_postprocess(
            input_json=Path(args.window_predictions_json),
            output_json=op_json,
            low_threshold=float(row["low_threshold"]),
            high_threshold=float(row["high_threshold"]),
            min_members=int(row["min_members"]),
            max_gap_seconds=None if row.get("max_gap_seconds", "") == "" else float(row["max_gap_seconds"]),
            class_hierarchy=args.class_hierarchy,
            merge_event_media=bool(args.merge_event_media),
            event_media_dir=op_dir / "predictions_postprocessed_events_media" if args.merge_event_media else None,
        )
        evaluate_single_postprocessed_output(
            postprocessed_json=op_json,
            annotations=annotations,
            all_annotations=all_annotations,
            clip_manifest=clip_manifest,
            output_dir=op_dir,
            match_collar_s=float(args.match_collar_s),
            raw_window_predictions=raw_window_predictions,
            raw_window_payload=raw_window_payload,
            raw_window_json_path=Path(args.window_predictions_json),
            raw_window_threshold=float(row["low_threshold"]),
            baseline_summary=baseline_summary,
            summary_title=f"Fin Whale Part 2 Report ({label})",
            sweep_summary_rows=summarize_sweep_rows(sweep_rows),
            example_mat_dir=Path(args.example_mat_dir) if args.example_mat_dir else None,
            export_example_images=bool(args.export_example_images),
            max_examples_per_group=int(args.max_examples_per_group),
        )
        selected_dirs[label] = op_dir

    best_dir = selected_dirs["best_f1"]
    for artifact_name in [
        "report.md",
        "metrics.json",
        "rapid_review.csv",
        "rapid_review.app.json",
        "rapid_review.o3.json",
        "recommendations.md",
        "overall_clip_confusion.csv",
        "overall_clip_confusion.png",
        "overall_view_metrics.csv",
        "overall_view_metrics.png",
        "bucket_metrics.csv",
        "bucket_recall_comparison.png",
        "context_recall_metrics.csv",
        "examples",
    ]:
        src = best_dir / artifact_name
        if src.exists():
            target = output_dir / artifact_name
            if src.is_dir():
                _copytree_replace(src, target)
            else:
                shutil.copy2(src, target)

    if args.use_wandb:
        _log_part2_wandb(
            output_dir=output_dir,
            selected_rows=selected_rows,
            best_metrics_json=output_dir / "metrics.json",
            sweep_rows=sweep_rows,
            args=args,
        )


if __name__ == "__main__":
    main()
