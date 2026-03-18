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
    build_clip_confusion,
    evaluation_report_lines,
    filter_prediction_items_for_rapid_review,
    load_annotations_csv,
    load_clip_manifest_csv,
    load_prediction_segments,
    match_predictions_to_annotations,
    maybe_plot_confusion_matrix,
    maybe_plot_sweep_curve,
    rapid_review_rows,
    recommendations_from_errors,
    strict_o3_subset,
    summarize_sweep_rows,
    write_csv,
)


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


def evaluate_single_postprocessed_output(
    *,
    postprocessed_json: Path,
    annotations,
    clip_manifest,
    output_dir: Path,
    match_collar_s: float,
    baseline_summary: Optional[Dict[str, Any]] = None,
    summary_title: str = "Fin Whale Part 2 Report",
    sweep_summary_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, prediction_segments = load_prediction_segments(postprocessed_json)

    fin_annotations = [ann for ann in annotations if ann.species == "Bp"]
    matches, unmatched_predictions, unmatched_annotations = match_predictions_to_annotations(
        prediction_segments,
        fin_annotations,
        match_collar_s,
    )

    overall_event = {
        "tp": len(matches),
        "fp": len(unmatched_predictions),
        "fn": len(unmatched_annotations),
    }
    precision = overall_event["tp"] / (overall_event["tp"] + overall_event["fp"]) if (overall_event["tp"] + overall_event["fp"]) else 0.0
    recall = overall_event["tp"] / (overall_event["tp"] + overall_event["fn"]) if (overall_event["tp"] + overall_event["fn"]) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    overall_event.update({"precision": precision, "recall": recall, "f1": f1})

    overall_clip = build_clip_confusion(clip_manifest, prediction_segments)
    bucket_events = bucket_event_metrics(prediction_segments, fin_annotations, match_collar_s)
    bucket_clips = {
        bucket: build_clip_confusion(clip_manifest, prediction_segments, bucket=bucket)
        for bucket in PART2_BUCKETS
    }

    annotations_by_id = {ann.annotation_id: ann for ann in fin_annotations}
    predictions_by_id = {pred.prediction_id: pred for pred in prediction_segments}

    write_csv(output_dir / "matches.csv", _match_rows(matches, annotations_by_id, predictions_by_id))
    write_csv(output_dir / "false_negatives.csv", _unmatched_annotation_rows(unmatched_annotations))
    write_csv(output_dir / "false_positives.csv", _unmatched_prediction_rows(unmatched_predictions, clip_manifest))
    write_csv(output_dir / "rapid_review.csv", rapid_review_rows(unmatched_predictions, clip_manifest))

    _write_confusion_csv(output_dir / "overall_clip_confusion.csv", overall_clip)
    maybe_plot_confusion_matrix(output_dir / "overall_clip_confusion.png", overall_clip, "Part 2 overall clip confusion")

    bucket_metric_rows: List[Dict[str, Any]] = []
    for bucket in PART2_BUCKETS:
        clip_metrics = bucket_clips[bucket]
        event_metrics = bucket_events[bucket]
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

    recommendations = recommendations_from_errors(
        unmatched_annotations=unmatched_annotations,
        unmatched_predictions=unmatched_predictions,
        clip_manifest=clip_manifest,
    )
    with open(output_dir / "recommendations.md", "w", encoding="utf-8") as handle:
        for line in recommendations:
            handle.write(f"- {line}\n")

    rapid_payload = filter_prediction_items_for_rapid_review(payload, unmatched_predictions)
    rapid_app_json = output_dir / "rapid_review.app.json"
    rapid_o3_json = output_dir / "rapid_review.o3.json"
    _write_json(rapid_app_json, rapid_payload)
    _write_json(rapid_o3_json, strict_o3_subset(rapid_payload))

    summary_payload = {
        "postprocessed_json": str(postprocessed_json),
        "match_collar_s": float(match_collar_s),
        "overall_event_metrics": overall_event,
        "overall_clip_metrics": overall_clip,
        "bucket_event_metrics": bucket_events,
        "bucket_clip_metrics": bucket_clips,
        "recommendations": recommendations,
        "rapid_review_count": len(unmatched_predictions),
    }
    _write_json(output_dir / "metrics.json", summary_payload)

    report_lines = evaluation_report_lines(
        summary_title=summary_title,
        overall_event_metrics=overall_event,
        overall_clip_metrics=overall_clip,
        bucket_event_metrics_map=bucket_events,
        bucket_clip_metrics_map=bucket_clips,
        recommendations=recommendations,
        rapid_review_count=len(unmatched_predictions),
        baseline_summary=baseline_summary,
        sweep_summary_rows=sweep_summary_rows,
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "overall_event_metrics": overall_event,
        "overall_clip_metrics": overall_clip,
        "bucket_event_metrics": bucket_events,
        "bucket_clip_metrics": bucket_clips,
        "recommendations": recommendations,
        "rapid_review_count": len(unmatched_predictions),
        "metrics_json": str(output_dir / "metrics.json"),
        "report_md": str(report_path),
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
    low_thresholds: Sequence[float],
    high_thresholds: Sequence[float],
    min_members_values: Sequence[int],
    max_gap_values: Sequence[Optional[float]],
    class_hierarchy: Optional[str],
    annotations,
    clip_manifest,
    match_collar_s: float,
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
                        clip_manifest=clip_manifest,
                        output_dir=combo_dir / "eval",
                        match_collar_s=match_collar_s,
                        baseline_summary=None,
                        summary_title=f"Part 2 sweep evaluation: {tag}",
                    )
                    row = {
                        "tag": tag,
                        "combo_dir": str(combo_dir),
                        "low_threshold": low_threshold,
                        "high_threshold": high_threshold,
                        "min_members": min_members,
                        "max_gap_seconds": "" if max_gap_seconds is None else max_gap_seconds,
                        **eval_summary["overall_event_metrics"],
                    }
                    sweep_rows.append(row)
                    print(
                        "sweep",
                        combo_index,
                        tag,
                        f"precision={row['precision']:.4f}",
                        f"recall={row['recall']:.4f}",
                        f"f1={row['f1']:.4f}",
                    )
    write_csv(output_dir / "sweep_summary.csv", summarize_sweep_rows(sweep_rows))
    maybe_plot_sweep_curve(output_dir / "sweep_precision_recall.png", sweep_rows, "window predictions")
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
                -float(row.get("fp", 0.0)),
            ),
        )

    out = {
        "best_f1": _best("f1", "precision"),
        "high_recall": _best("recall", "precision"),
        "high_precision": _best("precision", "recall"),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Part 2 predictions and build the report bundle")
    ap.add_argument("--annotations-csv", type=str, required=True)
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
    args = ap.parse_args()

    if bool(args.postprocessed_json) == bool(args.window_predictions_json):
        raise SystemExit("Provide exactly one of --postprocessed-json or --window-predictions-json")

    annotations = load_annotations_csv(args.annotations_csv)
    clip_manifest = load_clip_manifest_csv(args.clip_manifest_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = _load_baseline_summary(args.baseline_metrics_json)

    if args.postprocessed_json:
        evaluate_single_postprocessed_output(
            postprocessed_json=Path(args.postprocessed_json),
            annotations=annotations,
            clip_manifest=clip_manifest,
            output_dir=output_dir,
            match_collar_s=float(args.match_collar_s),
            baseline_summary=baseline_summary,
        )
        return

    sweep_rows = _sweep_candidates(
        window_predictions_json=Path(args.window_predictions_json),
        output_dir=output_dir,
        low_thresholds=_parse_float_list(args.low_thresholds),
        high_thresholds=_parse_float_list(args.high_thresholds),
        min_members_values=_parse_int_list(args.min_members_values),
        max_gap_values=_parse_optional_float_list(args.max_gap_values),
        class_hierarchy=args.class_hierarchy,
        annotations=annotations,
        clip_manifest=clip_manifest,
        match_collar_s=float(args.match_collar_s),
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
            clip_manifest=clip_manifest,
            output_dir=op_dir,
            match_collar_s=float(args.match_collar_s),
            baseline_summary=baseline_summary,
            summary_title=f"Fin Whale Part 2 Report ({label})",
            sweep_summary_rows=summarize_sweep_rows(sweep_rows),
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
        "bucket_metrics.csv",
    ]:
        src = best_dir / artifact_name
        if src.exists():
            target = output_dir / artifact_name
            if src.is_dir():
                _copytree_replace(src, target)
            else:
                shutil.copy2(src, target)


if __name__ == "__main__":
    main()
