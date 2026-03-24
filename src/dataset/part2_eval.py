"""Evaluation helpers for the Part 2 fin-whale report pipeline."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .part2_annotations import (
    FIN_BUCKET_20,
    FIN_BUCKET_40,
    FIN_BUCKET_OTHER,
    FIN_SPECIES_CODE,
    UNKNOWN_CONTEXT,
    parse_filename_timestamp,
)

PART2_BUCKETS = [FIN_BUCKET_20, FIN_BUCKET_40, FIN_BUCKET_OTHER]
STRICT_EVENT_VIEW = "strict_event"
MERGED_REGION_VIEW = "merged_region_coverage"
RAW_WINDOW_VIEW = "raw_window_coverage"
VIEW_LABELS = {
    STRICT_EVENT_VIEW: "Strict Call Extraction",
    MERGED_REGION_VIEW: "Merged Region Coverage",
    RAW_WINDOW_VIEW: "Raw Window Coverage",
}


@dataclass(frozen=True)
class AnnotationEvent:
    annotation_id: str
    filename: str
    begin_time_s: float
    end_time_s: float
    species: str
    call_type_bucket: str
    call_type_raw: str
    comments: str
    context_tags: Tuple[str, ...]


@dataclass(frozen=True)
class ClipManifestRow:
    filename: str
    is_fin_positive: bool
    is_annotated_non_fin: bool
    species_codes: Tuple[str, ...]
    fin_call_type_buckets: Tuple[str, ...]
    context_tags: Tuple[str, ...]


@dataclass(frozen=True)
class PredictedSegment:
    prediction_id: str
    item_id: str
    filename: str
    start_time_s: float
    end_time_s: float
    score: float
    source_index: int


@dataclass(frozen=True)
class Match:
    annotation_id: str
    prediction_id: str
    filename: str
    score: float
    overlap_s: float


def _split_tags(raw: str) -> Tuple[str, ...]:
    tags = [token.strip() for token in str(raw or "").split("|") if token.strip()]
    return tuple(sorted(dict.fromkeys(tags))) if tags else (UNKNOWN_CONTEXT,)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_annotations_csv(path: Path | str) -> List[AnnotationEvent]:
    rows: List[AnnotationEvent] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            rows.append(
                AnnotationEvent(
                    annotation_id=f"ann_{idx:06d}",
                    filename=str(row.get("filename", "")).strip(),
                    begin_time_s=_safe_float(row.get("begin_time_s")),
                    end_time_s=_safe_float(row.get("end_time_s")),
                    species=str(row.get("species", "")).strip(),
                    call_type_bucket=str(row.get("call_type_bucket", "")).strip(),
                    call_type_raw=str(row.get("call_type_raw", "")).strip(),
                    comments=str(row.get("comments", "")).strip(),
                    context_tags=_split_tags(row.get("context_tags", "")),
                )
            )
    return rows


def load_clip_manifest_csv(path: Path | str) -> Dict[str, ClipManifestRow]:
    out: Dict[str, ClipManifestRow] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            out[filename] = ClipManifestRow(
                filename=filename,
                is_fin_positive=str(row.get("is_fin_positive", "0")).strip() == "1",
                is_annotated_non_fin=str(row.get("is_annotated_non_fin", "0")).strip() == "1",
                species_codes=tuple(token for token in str(row.get("species_codes", "")).split("|") if token),
                fin_call_type_buckets=tuple(
                    token for token in str(row.get("fin_call_type_buckets", "")).split("|") if token
                ),
                context_tags=_split_tags(row.get("context_tags", "")),
            )
    return out


def _event_segments_from_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_segments = item.get("source_segments")
    if isinstance(source_segments, list) and source_segments:
        return [segment for segment in source_segments if isinstance(segment, dict)]

    outputs = item.get("model_outputs")
    if isinstance(outputs, list):
        for output in outputs:
            if not isinstance(output, dict):
                continue
            metadata = output.get("metadata")
            if not isinstance(metadata, dict):
                continue
            windows = metadata.get("windows")
            if isinstance(windows, list) and windows:
                return [window for window in windows if isinstance(window, dict)]
    return []


def _source_audio_from_item(item: Dict[str, Any]) -> str:
    if isinstance(item.get("source_audio"), str) and item["source_audio"]:
        return str(item["source_audio"])
    parent_files = item.get("parent_source_audio_files")
    if isinstance(parent_files, list) and parent_files:
        return str(parent_files[0])
    source_audio = item.get("source_audio")
    if isinstance(source_audio, dict):
        name = source_audio.get("file_name")
        if name:
            return str(name)
    return ""


def _relative_seconds_from_iso(filename: str, iso_value: Any) -> Optional[float]:
    clip_ts = parse_filename_timestamp(filename)
    if clip_ts is None or iso_value is None:
        return None
    try:
        from datetime import datetime

        event_ts = datetime.fromisoformat(str(iso_value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (event_ts - clip_ts).total_seconds()


def _normalize_segment_seconds(filename: str, seconds_value: Any) -> Optional[float]:
    seconds = _safe_float(seconds_value, default=float("nan"))
    if seconds != seconds:
        return None
    if abs(seconds) <= 86400.0:
        return seconds
    clip_ts = parse_filename_timestamp(filename)
    if clip_ts is None:
        return seconds
    return seconds - clip_ts.timestamp()


def load_prediction_segments(path: Path | str) -> Tuple[Dict[str, Any], List[PredictedSegment]]:
    json_path = Path(path)
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    segments: List[PredictedSegment] = []
    for item_index, item in enumerate(payload.get("items", [])):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id") or f"item_{item_index:06d}")
        base_score = None
        outputs = item.get("model_outputs")
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            base_score = _safe_float(outputs[0].get("score"), 0.0)

        grouped: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        extracted_segments = _event_segments_from_item(item)
        for seg_index, segment in enumerate(extracted_segments):
            filename = str(segment.get("source_audio") or segment.get("file_name") or "").strip()
            if not filename:
                continue
            start_s = _normalize_segment_seconds(filename, segment.get("time_start_sec"))
            end_s = _normalize_segment_seconds(filename, segment.get("time_end_sec"))
            if start_s is None or end_s is None:
                continue
            score = _safe_float(segment.get("score"), base_score or 0.0)
            grouped[filename].append((start_s, end_s, score))

        if not grouped:
            fallback_filename = _source_audio_from_item(item)
            if fallback_filename:
                start_s = _relative_seconds_from_iso(fallback_filename, item.get("audio_start_time"))
                end_s = _relative_seconds_from_iso(fallback_filename, item.get("audio_end_time"))
                if start_s is not None and end_s is not None:
                    grouped[fallback_filename].append((start_s, end_s, base_score or 0.0))

        for source_index, filename in enumerate(sorted(grouped)):
            parts = grouped[filename]
            starts = [part[0] for part in parts]
            ends = [part[1] for part in parts]
            scores = [part[2] for part in parts]
            segments.append(
                PredictedSegment(
                    prediction_id=f"{item_id}::{filename}::{source_index}",
                    item_id=item_id,
                    filename=filename,
                    start_time_s=min(starts),
                    end_time_s=max(ends),
                    score=max(scores),
                    source_index=source_index,
                )
            )

    return payload, segments


def _overlap_seconds(pred: PredictedSegment, ann: AnnotationEvent, collar_s: float) -> float:
    start = max(pred.start_time_s, ann.begin_time_s - collar_s)
    end = min(pred.end_time_s, ann.end_time_s + collar_s)
    return max(0.0, end - start)


def _group_predictions_by_file(
    predictions: Sequence[PredictedSegment],
) -> Dict[str, List[PredictedSegment]]:
    grouped: Dict[str, List[PredictedSegment]] = defaultdict(list)
    for pred in predictions:
        grouped[pred.filename].append(pred)
    return grouped


def _group_annotations_by_file(
    annotations: Sequence[AnnotationEvent],
) -> Dict[str, List[AnnotationEvent]]:
    grouped: Dict[str, List[AnnotationEvent]] = defaultdict(list)
    for ann in annotations:
        grouped[ann.filename].append(ann)
    return grouped


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cleaned = sorted(
        (
            (min(float(start), float(end)), max(float(start), float(end)))
            for start, end in intervals
        ),
        key=lambda item: (item[0], item[1]),
    )
    merged: List[Tuple[float, float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _prediction_review_summary(predictions: Sequence[PredictedSegment]) -> Dict[str, float]:
    durations = [max(0.0, pred.end_time_s - pred.start_time_s) for pred in predictions]
    intervals_by_file: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for pred in predictions:
        intervals_by_file[pred.filename].append((pred.start_time_s, pred.end_time_s))

    union_seconds = 0.0
    for intervals in intervals_by_file.values():
        union_seconds += sum(max(0.0, end - start) for start, end in _merge_intervals(intervals))

    return {
        "prediction_count": len(predictions),
        "unique_predicted_clip_count": len(intervals_by_file),
        "mean_prediction_duration_s": mean(durations) if durations else 0.0,
        "median_prediction_duration_s": median(durations) if durations else 0.0,
        "max_prediction_duration_s": max(durations) if durations else 0.0,
        "total_review_seconds": union_seconds,
        "total_review_minutes": union_seconds / 60.0,
    }


def filter_predictions_by_score(
    predictions: Sequence[PredictedSegment],
    min_score: float,
) -> List[PredictedSegment]:
    threshold = float(min_score)
    return [pred for pred in predictions if pred.score >= threshold]


def match_predictions_to_annotations(
    predictions: Sequence[PredictedSegment],
    annotations: Sequence[AnnotationEvent],
    collar_s: float,
) -> Tuple[List[Match], List[PredictedSegment], List[AnnotationEvent]]:
    candidates: List[Tuple[float, float, float, str, str, Match]] = []
    ann_by_file = _group_annotations_by_file(annotations)

    for pred in predictions:
        for ann in ann_by_file.get(pred.filename, []):
            overlap_s = _overlap_seconds(pred, ann, collar_s)
            if overlap_s <= 0:
                continue
            pred_center = 0.5 * (pred.start_time_s + pred.end_time_s)
            ann_center = 0.5 * (ann.begin_time_s + ann.end_time_s)
            center_delta = abs(pred_center - ann_center)
            match = Match(
                annotation_id=ann.annotation_id,
                prediction_id=pred.prediction_id,
                filename=pred.filename,
                score=pred.score,
                overlap_s=overlap_s,
            )
            candidates.append(
                (
                    pred.score,
                    overlap_s,
                    -center_delta,
                    pred.prediction_id,
                    ann.annotation_id,
                    match,
                )
            )

    candidates.sort(reverse=True)
    matched_pred_ids = set()
    matched_ann_ids = set()
    matches: List[Match] = []
    for _, _, _, pred_id, ann_id, match in candidates:
        if pred_id in matched_pred_ids or ann_id in matched_ann_ids:
            continue
        matched_pred_ids.add(pred_id)
        matched_ann_ids.add(ann_id)
        matches.append(match)

    unmatched_predictions = [pred for pred in predictions if pred.prediction_id not in matched_pred_ids]
    unmatched_annotations = [ann for ann in annotations if ann.annotation_id not in matched_ann_ids]
    return matches, unmatched_predictions, unmatched_annotations


def coverage_metrics(
    predictions: Sequence[PredictedSegment],
    annotations: Sequence[AnnotationEvent],
    collar_s: float,
) -> Dict[str, float]:
    ann_by_file = _group_annotations_by_file(annotations)
    useful_prediction_ids: set[str] = set()
    covered_annotation_ids: set[str] = set()

    for pred in predictions:
        anns = ann_by_file.get(pred.filename, [])
        if not anns:
            continue
        pred_hit = False
        for ann in anns:
            overlap_s = _overlap_seconds(pred, ann, collar_s)
            if overlap_s <= 0:
                continue
            pred_hit = True
            covered_annotation_ids.add(ann.annotation_id)
        if pred_hit:
            useful_prediction_ids.add(pred.prediction_id)

    summary = _prediction_review_summary(predictions)
    useful_predictions = len(useful_prediction_ids)
    covered_annotations = len(covered_annotation_ids)
    prediction_count = len(predictions)
    annotation_count = len(annotations)
    precision = _safe_div(useful_predictions, prediction_count)
    recall = _safe_div(covered_annotations, annotation_count)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return {
        "prediction_count": prediction_count,
        "useful_prediction_count": useful_predictions,
        "annotation_count": annotation_count,
        "covered_annotation_count": covered_annotations,
        "uncovered_annotation_count": max(0, annotation_count - covered_annotations),
        "unmatched_prediction_count": max(0, prediction_count - useful_predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions_per_covered_call": _safe_div(prediction_count, covered_annotations),
        **summary,
    }


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def confusion_metrics(tp: int, fp: int, fn: int, tn: int = 0) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def build_clip_confusion(
    clip_manifest: Dict[str, ClipManifestRow],
    predictions: Sequence[PredictedSegment],
    *,
    bucket: Optional[str] = None,
) -> Dict[str, float]:
    predicted_positive = {pred.filename for pred in predictions}
    tp = fp = fn = tn = 0
    for filename, row in clip_manifest.items():
        actual_positive = row.is_fin_positive
        if bucket is not None:
            actual_positive = bucket in set(row.fin_call_type_buckets)
        pred_positive = filename in predicted_positive
        if actual_positive and pred_positive:
            tp += 1
        elif actual_positive and not pred_positive:
            fn += 1
        elif not actual_positive and pred_positive:
            fp += 1
        else:
            tn += 1
    return confusion_metrics(tp=tp, fp=fp, fn=fn, tn=tn)


def bucket_event_metrics(
    predictions: Sequence[PredictedSegment],
    annotations: Sequence[AnnotationEvent],
    collar_s: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for bucket in PART2_BUCKETS:
        bucket_annotations = [ann for ann in annotations if ann.call_type_bucket == bucket]
        matches, unmatched_predictions, unmatched_annotations = match_predictions_to_annotations(
            predictions,
            bucket_annotations,
            collar_s,
        )
        out[bucket] = {
            **confusion_metrics(
                tp=len(matches),
                fp=len(unmatched_predictions),
                fn=len(unmatched_annotations),
                tn=0,
            ),
            "annotation_count": len(bucket_annotations),
            "prediction_count": len(predictions),
        }
    return out


def bucket_coverage_metrics(
    predictions: Sequence[PredictedSegment],
    annotations: Sequence[AnnotationEvent],
    collar_s: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for bucket in PART2_BUCKETS:
        bucket_annotations = [ann for ann in annotations if ann.call_type_bucket == bucket]
        out[bucket] = coverage_metrics(predictions, bucket_annotations, collar_s)
    return out


def context_recall_rows(
    *,
    predictions: Sequence[PredictedSegment],
    annotations: Sequence[AnnotationEvent],
    collar_s: float,
    view_name: str,
    min_annotation_count: int = 1,
) -> List[Dict[str, Any]]:
    tags = sorted({tag for ann in annotations for tag in ann.context_tags} or {UNKNOWN_CONTEXT})
    rows: List[Dict[str, Any]] = []
    for bucket in PART2_BUCKETS:
        bucket_annotations = [ann for ann in annotations if ann.call_type_bucket == bucket]
        for tag in tags:
            tagged_annotations = [ann for ann in bucket_annotations if tag in ann.context_tags]
            if len(tagged_annotations) < int(min_annotation_count):
                continue
            if view_name == STRICT_EVENT_VIEW:
                matches, _, unmatched_annotations = match_predictions_to_annotations(
                    predictions,
                    tagged_annotations,
                    collar_s,
                )
                covered_count = len(matches)
                recall = _safe_div(covered_count, len(tagged_annotations))
            else:
                metrics = coverage_metrics(predictions, tagged_annotations, collar_s)
                covered_count = int(metrics["covered_annotation_count"])
                recall = float(metrics["recall"])
            rows.append(
                {
                    "view": view_name,
                    "bucket": bucket,
                    "context_tag": tag,
                    "annotation_count": len(tagged_annotations),
                    "covered_annotation_count": covered_count,
                    "uncovered_annotation_count": max(0, len(tagged_annotations) - covered_count),
                    "recall": recall,
                }
            )
    return rows


def hardest_context_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    max_items: int = 8,
    min_annotation_count: int = 25,
) -> List[Dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if int(row.get("annotation_count", 0)) >= int(min_annotation_count)
    ]
    filtered.sort(
        key=lambda row: (
            _safe_float(row.get("recall"), 1.0),
            -int(row.get("annotation_count", 0)),
            str(row.get("view", "")),
            str(row.get("bucket", "")),
            str(row.get("context_tag", "")),
        )
    )
    return list(filtered[:max_items])


def summarize_sweep_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [
        "window_step",
        "low_threshold",
        "high_threshold",
        "min_members",
        "max_gap_seconds",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "merged_region_precision",
        "merged_region_recall",
        "merged_region_f1",
        "raw_window_threshold",
        "raw_window_precision",
        "raw_window_recall",
        "raw_window_f1",
        "prediction_count",
        "covered_annotation_count",
        "total_review_minutes",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def recommendations_from_errors(
    unmatched_annotations: Sequence[AnnotationEvent],
    unmatched_predictions: Sequence[PredictedSegment],
    clip_manifest: Dict[str, ClipManifestRow],
    *,
    max_items: int = 8,
) -> List[str]:
    fn_counts: Counter[Tuple[str, str]] = Counter()
    for ann in unmatched_annotations:
        tags = ann.context_tags or (UNKNOWN_CONTEXT,)
        for tag in tags:
            fn_counts[(ann.call_type_bucket or FIN_BUCKET_OTHER, tag)] += 1

    fp_counts: Counter[str] = Counter()
    for pred in unmatched_predictions:
        clip_row = clip_manifest.get(pred.filename)
        tags = clip_row.context_tags if clip_row else (UNKNOWN_CONTEXT,)
        for tag in tags:
            fp_counts[tag] += 1

    recommendations: List[str] = []
    for (bucket, tag), count in fn_counts.most_common(max_items):
        recommendations.append(
            f"Need more annotated examples of {bucket} calls in {tag} contexts ({count} false negatives)."
        )
    remaining = max(0, max_items - len(recommendations))
    for tag, count in fp_counts.most_common(remaining):
        recommendations.append(
            f"Add more non-fin review coverage for {tag} clips ({count} unmatched predicted events) to reduce false positives."
        )
    return recommendations[:max_items]


def rapid_review_rows(
    unmatched_predictions: Sequence[PredictedSegment],
    clip_manifest: Dict[str, ClipManifestRow],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pred in sorted(unmatched_predictions, key=lambda item: (-item.score, item.filename, item.start_time_s)):
        clip_row = clip_manifest.get(pred.filename)
        rows.append(
            {
                "filename": pred.filename,
                "prediction_id": pred.prediction_id,
                "item_id": pred.item_id,
                "score": f"{pred.score:.6f}",
                "start_time_s": f"{pred.start_time_s:.6f}",
                "end_time_s": f"{pred.end_time_s:.6f}",
                "duration_s": f"{max(0.0, pred.end_time_s - pred.start_time_s):.6f}",
                "context_tags": "|".join(clip_row.context_tags) if clip_row else UNKNOWN_CONTEXT,
                "species_codes": "|".join(clip_row.species_codes) if clip_row else "",
            }
        )
    return rows


def filter_prediction_items_for_rapid_review(
    payload: Dict[str, Any],
    unmatched_predictions: Sequence[PredictedSegment],
) -> Dict[str, Any]:
    keep_item_ids = {pred.item_id for pred in unmatched_predictions}
    filtered_items = [
        item
        for item in payload.get("items", [])
        if isinstance(item, dict) and str(item.get("item_id", "")) in keep_item_ids
    ]
    out = {
        key: value
        for key, value in payload.items()
        if key
        in {
            "schema_version",
            "created_at",
            "updated_at",
            "task_type",
            "model",
            "data_sources",
            "spectrogram_config",
            "pipeline",
            "items",
        }
    }
    out["items"] = filtered_items
    return out


def strict_o3_subset(payload: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        clean_item = {
            "item_id": item.get("item_id"),
            "data_source_id": item.get("data_source_id"),
            "audio_start_time": item.get("audio_start_time"),
            "audio_end_time": item.get("audio_end_time"),
            "segment_index": item.get("segment_index"),
            "model_outputs": item.get("model_outputs", []),
            "verifications": item.get("verifications", []),
            "paths": item.get("paths", {}),
        }
        items.append({key: value for key, value in clean_item.items() if value not in (None, {}, [])})

    return {
        key: value
        for key, value in {
            "schema_version": payload.get("schema_version"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "task_type": payload.get("task_type"),
            "model": payload.get("model"),
            "data_sources": payload.get("data_sources", []),
            "spectrogram_config": payload.get("spectrogram_config", {}),
            "pipeline": payload.get("pipeline", {}),
            "items": items,
        }.items()
        if value not in (None, {}, [])
    }


def write_csv(path: Path | str, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _configure_plot_style(plt) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def maybe_plot_confusion_matrix(path: Path | str, matrix: Dict[str, float], title: str) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_style(plt)
    image = [
        [matrix.get("tn", 0), matrix.get("fp", 0)],
        [matrix.get("fn", 0), matrix.get("tp", 0)],
    ]

    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=220)
    heatmap = ax.imshow(image, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    vmax = max(1.0, float(np.max(image)))
    for i in range(2):
        for j in range(2):
            value = image[i][j]
            text_color = "white" if float(value) > 0.55 * vmax else "#17324d"
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{value:,}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold",
                color=text_color,
            )
    ax.set_xticks([0, 1], labels=["Predicted non-fin", "Predicted fin"])
    ax.set_yticks([0, 1], labels=["Actual non-fin", "Actual fin"])
    ax.set_xlabel("Model output")
    ax.set_ylabel("Reference label")
    ax.set_title(title, pad=12)
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def maybe_plot_sweep_curve(path: Path | str, rows: Sequence[Dict[str, Any]], window_step_label: str) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_style(plt)

    panels = [
        ("Strict Call Extraction", "precision", "recall", "#0f4c81"),
        ("Merged Region Coverage", "merged_region_precision", "merged_region_recall", "#2a9d8f"),
        ("Raw Window Coverage", "raw_window_precision", "raw_window_recall", "#e76f51"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), dpi=220, sharex=False, sharey=False)
    if not isinstance(axes, (list, tuple)):
        axes = list(axes)
    for ax, (title, p_key, r_key, color) in zip(axes, panels):
        sorted_rows = sorted(
            rows,
            key=lambda row: (_safe_float(row.get(r_key, row.get("recall"))), _safe_float(row.get(p_key, row.get("precision")))),
        )
        recalls = [_safe_float(row.get(r_key, row.get("recall"))) for row in sorted_rows]
        precisions = [_safe_float(row.get(p_key, row.get("precision"))) for row in sorted_rows]
        ax.plot(recalls, precisions, marker="o", markersize=5, linewidth=2.0, color=color)
        if recalls and precisions:
            best_index = max(
                range(len(sorted_rows)),
                key=lambda idx: (
                    _safe_float(sorted_rows[idx].get("f1")),
                    _safe_float(sorted_rows[idx].get(r_key, sorted_rows[idx].get("recall"))),
                ),
            )
            ax.scatter(
                [recalls[best_index]],
                [precisions[best_index]],
                s=64,
                color=color,
                edgecolor="white",
                linewidth=1.2,
                zorder=5,
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title, pad=10)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0, top=1.02)
        ax.grid(True, alpha=0.28)
    fig.suptitle(f"Part 2 Sweep Tradeoffs ({window_step_label})", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def maybe_plot_view_summary(path: Path | str, rows: Sequence[Dict[str, Any]]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_style(plt)

    labels = [row.get("view_label", row.get("view", "")) for row in rows]
    precisions = [_safe_float(row.get("precision")) for row in rows]
    recalls = [_safe_float(row.get("recall")) for row in rows]
    f1s = [_safe_float(row.get("f1")) for row in rows]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=220)
    ax.bar(x - width, precisions, width=width, label="Precision", color="#0f4c81")
    ax.bar(x, recalls, width=width, label="Recall", color="#2a9d8f")
    ax.bar(x + width, f1s, width=width, label="F1", color="#e76f51")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Part 2 Evaluation Views", pad=10)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(True, axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def maybe_plot_bucket_recall_comparison(
    path: Path | str,
    *,
    strict_bucket_metrics: Dict[str, Dict[str, float]],
    merged_bucket_metrics: Dict[str, Dict[str, float]],
    raw_bucket_metrics: Dict[str, Dict[str, float]],
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_style(plt)

    x = np.arange(len(PART2_BUCKETS))
    width = 0.24
    strict_vals = [float(strict_bucket_metrics[bucket]["recall"]) for bucket in PART2_BUCKETS]
    merged_vals = [float(merged_bucket_metrics[bucket]["recall"]) for bucket in PART2_BUCKETS]
    raw_vals = [float(raw_bucket_metrics[bucket]["recall"]) for bucket in PART2_BUCKETS]

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=220)
    ax.bar(x - width, strict_vals, width=width, label="Strict event recall", color="#0f4c81")
    ax.bar(x, merged_vals, width=width, label="Merged-region call coverage", color="#2a9d8f")
    ax.bar(x + width, raw_vals, width=width, label="Raw-window call coverage", color="#e76f51")
    ax.set_xticks(x, labels=PART2_BUCKETS)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Fin-whale call subtype")
    ax.set_title("Subtype Recall by Evaluation View", pad=10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def evaluation_report_lines(
    *,
    summary_title: str,
    strict_event_metrics: Dict[str, float],
    merged_region_metrics: Dict[str, float],
    raw_window_metrics: Optional[Dict[str, float]],
    overall_clip_metrics: Dict[str, float],
    bucket_strict_event_metrics_map: Dict[str, Dict[str, float]],
    bucket_merged_region_metrics_map: Dict[str, Dict[str, float]],
    bucket_raw_window_metrics_map: Optional[Dict[str, Dict[str, float]]],
    bucket_clip_metrics_map: Dict[str, Dict[str, float]],
    recommendations: Sequence[str],
    rapid_review_count: int,
    baseline_summary: Optional[Dict[str, Any]] = None,
    sweep_summary_rows: Optional[Sequence[Dict[str, Any]]] = None,
    hardest_context_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[str]:
    lines = [f"# {summary_title}", ""]
    lines.extend(
        [
            "## Part 2 Summary",
            "",
            "This report shows three complementary views of performance:",
            "",
            "- `Strict Call Extraction`: one predicted event can match only one annotated call. This is the hardest metric and reflects per-call extraction quality.",
            "- `Merged Region Coverage`: one merged predicted region can cover many annotated calls. This reflects rapid-review usefulness.",
            "- `Raw Window Coverage`: pre-merge detector windows above threshold count as positive. This isolates the CNN detector from the event-merging logic.",
            "",
            f"- Strict event precision: `{strict_event_metrics['precision']:.4f}`",
            f"- Strict event recall: `{strict_event_metrics['recall']:.4f}`",
            f"- Strict event F1: `{strict_event_metrics['f1']:.4f}`",
            f"- Strict event counts: `TP={strict_event_metrics['tp']}`, `FP={strict_event_metrics['fp']}`, `FN={strict_event_metrics['fn']}`",
            f"- Merged-region precision: `{merged_region_metrics['precision']:.4f}`",
            f"- Merged-region call coverage recall: `{merged_region_metrics['recall']:.4f}`",
            f"- Merged-region coverage F1: `{merged_region_metrics['f1']:.4f}`",
            f"- Merged-region review burden: `{int(merged_region_metrics['prediction_count'])}` regions, `{merged_region_metrics['total_review_minutes']:.1f}` review minutes",
            f"- Clip-level accuracy: `{overall_clip_metrics['accuracy']:.4f}`",
            f"- Clip-level counts: `TP={overall_clip_metrics['tp']}`, `FP={overall_clip_metrics['fp']}`, `FN={overall_clip_metrics['fn']}`, `TN={overall_clip_metrics['tn']}`",
            f"- Rapid-review queue size: `{rapid_review_count}`",
            "",
        ]
    )
    if raw_window_metrics is not None:
        lines.extend(
            [
                f"- Raw-window precision: `{raw_window_metrics['precision']:.4f}`",
                f"- Raw-window call coverage recall: `{raw_window_metrics['recall']:.4f}`",
                f"- Raw-window coverage F1: `{raw_window_metrics['f1']:.4f}`",
                f"- Raw-window review burden: `{int(raw_window_metrics['prediction_count'])}` positive windows, `{raw_window_metrics['total_review_minutes']:.1f}` review minutes",
                "",
            ]
        )

    if baseline_summary:
        lines.extend(["## Historical Baseline", ""])
        for key, value in baseline_summary.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    if sweep_summary_rows:
        lines.extend(
            [
                "## Sweep Summary",
                "",
                "| window_step | low | high | min_members | max_gap | strict precision | strict recall | strict f1 | merged recall | raw-window recall |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in sweep_summary_rows:
            lines.append(
                "| {window_step} | {low_threshold} | {high_threshold} | {min_members} | {max_gap_seconds} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {merged_region_recall:.4f} | {raw_window_recall} |".format(
                    window_step=row.get("window_step", ""),
                    low_threshold=row.get("low_threshold", ""),
                    high_threshold=row.get("high_threshold", ""),
                    min_members=row.get("min_members", ""),
                    max_gap_seconds=row.get("max_gap_seconds", ""),
                    precision=_safe_float(row.get("precision")),
                    recall=_safe_float(row.get("recall")),
                    f1=_safe_float(row.get("f1")),
                    merged_region_recall=_safe_float(row.get("merged_region_recall")),
                    raw_window_recall=f"{_safe_float(row.get('raw_window_recall')):.4f}" if row.get("raw_window_recall", "") != "" else "n/a",
                )
            )
        lines.append("")

    lines.extend(["## Fin Subtype Metrics", ""])
    lines.append("| bucket | strict recall | merged-region coverage recall | raw-window coverage recall | clip recall | annotations |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for bucket in PART2_BUCKETS:
        event_metrics = bucket_strict_event_metrics_map[bucket]
        merged_metrics = bucket_merged_region_metrics_map[bucket]
        raw_metrics = bucket_raw_window_metrics_map[bucket] if bucket_raw_window_metrics_map is not None else None
        clip_metrics = bucket_clip_metrics_map[bucket]
        lines.append(
            "| {bucket} | {strict_recall:.4f} | {merged_recall:.4f} | {raw_recall} | {clip_recall:.4f} | {annotation_count} |".format(
                bucket=bucket,
                strict_recall=event_metrics["recall"],
                merged_recall=merged_metrics["recall"],
                raw_recall=f"{raw_metrics['recall']:.4f}" if raw_metrics is not None else "n/a",
                clip_recall=clip_metrics["recall"],
                annotation_count=int(event_metrics.get("annotation_count", 0)),
            )
        )
    lines.append("")

    if hardest_context_rows:
        lines.extend(["## Hardest Contexts", ""])
        lines.append("| view | bucket | context | covered / total | recall |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in hardest_context_rows:
            lines.append(
                "| {view} | {bucket} | {context_tag} | {covered_annotation_count} / {annotation_count} | {recall:.4f} |".format(
                    view=VIEW_LABELS.get(str(row.get("view", "")), str(row.get("view", ""))),
                    bucket=row.get("bucket", ""),
                    context_tag=row.get("context_tag", ""),
                    covered_annotation_count=int(row.get("covered_annotation_count", 0)),
                    annotation_count=int(row.get("annotation_count", 0)),
                    recall=_safe_float(row.get("recall")),
                )
            )
        lines.append("")

    lines.extend(["## Annotation Recommendations", ""])
    if recommendations:
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
    else:
        lines.append("- No strong error clusters were detected from the current output set.")
    lines.append("")
    return lines
