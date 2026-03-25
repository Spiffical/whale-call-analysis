"""Example-image export helpers for Part 2 evaluation reports."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .part2_annotations import (
    FIN_BUCKET_20,
    FIN_BUCKET_40,
    FIN_BUCKET_OTHER,
    UNKNOWN_CONTEXT,
)
from .part2_eval import AnnotationEvent, ClipManifestRow, PredictedSegment

_AUDIO_EXTENSIONS = (".flac", ".wav", ".aif", ".aiff", ".mp3")
_PRIORITY_LABELS = (
    FIN_BUCKET_20,
    FIN_BUCKET_40,
    FIN_BUCKET_OTHER,
    "vessel_or_masking",
    "mixed_species",
    "song",
    "faint",
)


@dataclass(frozen=True)
class ExampleCandidate:
    group: str
    example_id: str
    filename: str
    display_start_s: float
    display_end_s: float
    prediction_start_s: Optional[float]
    prediction_end_s: Optional[float]
    score: Optional[float]
    mat_path: Optional[Path]
    bucket_labels: Tuple[str, ...]
    context_tags: Tuple[str, ...]
    species_codes: Tuple[str, ...]
    annotation_spans: Tuple[Tuple[float, float, str], ...]
    category_labels: Tuple[str, ...]
    panel_title: str
    detail_text: str
    sort_key: Tuple[Any, ...]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "item"


def _extract_source_audio_name(path_like: str) -> Optional[str]:
    name = Path(str(path_like or "")).name
    lower = name.lower()
    for ext in _AUDIO_EXTENSIONS:
        idx = lower.find(ext)
        if idx >= 0:
            return name[: idx + len(ext)]
    return None


def _build_mat_lookup(mat_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for mat_path in sorted(mat_dir.glob("*.mat")):
        source_name = _extract_source_audio_name(mat_path.name)
        if source_name:
            lookup.setdefault(source_name, mat_path.resolve())
        lookup.setdefault(mat_path.name, mat_path.resolve())
    return lookup


def _candidate_mat_refs_from_item(item: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []
    paths_obj = item.get("paths")
    if isinstance(paths_obj, Mapping):
        value = paths_obj.get("spectrogram_mat_path")
        if value:
            refs.append(str(value))

    source_segments = item.get("source_segments")
    if isinstance(source_segments, list):
        for segment in source_segments:
            if not isinstance(segment, Mapping):
                continue
            value = segment.get("spectrogram_mat_path")
            if value:
                refs.append(str(value))
    return refs


def _resolve_mat_path(
    *,
    filename: str,
    item: Optional[Mapping[str, Any]],
    mat_lookup: Mapping[str, Path],
    mat_dir: Optional[Path],
    json_base_dir: Optional[Path],
) -> Optional[Path]:
    if item is not None:
        for ref in _candidate_mat_refs_from_item(item):
            ref_path = Path(ref)
            if ref_path.is_absolute() and ref_path.exists():
                return ref_path
            if json_base_dir is not None:
                rel_candidate = (json_base_dir / ref_path).resolve()
                if rel_candidate.exists():
                    return rel_candidate
            name_candidate = Path(ref).name
            if name_candidate in mat_lookup:
                return mat_lookup[name_candidate]
            source_name = _extract_source_audio_name(name_candidate)
            if source_name and source_name in mat_lookup:
                return mat_lookup[source_name]
    if filename in mat_lookup:
        return mat_lookup[filename]
    if mat_dir is not None:
        matches = sorted(mat_dir.glob(f"{filename}_*.mat"))
        if matches:
            return matches[0].resolve()
    return None


def _overlaps_window(
    start_s: float,
    end_s: float,
    ann: AnnotationEvent,
    collar_s: float,
) -> bool:
    start = max(float(start_s), float(ann.begin_time_s) - float(collar_s))
    end = min(float(end_s), float(ann.end_time_s) + float(collar_s))
    return end > start


def _annotations_for_prediction(
    prediction: PredictedSegment,
    annotations_by_file: Mapping[str, Sequence[AnnotationEvent]],
    collar_s: float,
) -> List[AnnotationEvent]:
    out: List[AnnotationEvent] = []
    for ann in annotations_by_file.get(prediction.filename, []):
        if _overlaps_window(prediction.start_time_s, prediction.end_time_s, ann, collar_s):
            out.append(ann)
    return out


def _category_labels(
    *,
    bucket_labels: Iterable[str],
    context_tags: Iterable[str],
    species_codes: Iterable[str],
) -> Tuple[str, ...]:
    labels: List[str] = []
    for token in list(bucket_labels) + list(context_tags):
        token = str(token or "").strip()
        if token and token not in labels:
            labels.append(token)
    for code in species_codes:
        code = str(code or "").strip()
        if code:
            label = f"species:{code}"
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def _annotation_spans_for_rows(rows: Sequence[AnnotationEvent]) -> Tuple[Tuple[float, float, str], ...]:
    spans = []
    for ann in rows:
        spans.append((float(ann.begin_time_s), float(ann.end_time_s), ann.call_type_bucket or FIN_BUCKET_OTHER))
    return tuple(spans)


def _prediction_panel_title(
    *,
    base_label: str,
    buckets: Sequence[str],
    context_tags: Sequence[str],
) -> str:
    parts: List[str] = [base_label]
    if buckets:
        parts.append("/".join(buckets[:2]))
    for tag in context_tags:
        if tag in {"vessel_or_masking", "mixed_species", "song", "faint"}:
            parts.append(tag)
            break
    return " | ".join(parts[:3])


def _prediction_detail_text(
    *,
    score: Optional[float],
    duration_s: float,
    annotation_count: int,
    species_codes: Sequence[str],
    context_tags: Sequence[str],
) -> str:
    bits: List[str] = []
    if score is not None:
        bits.append(f"score={score:.3f}")
    bits.append(f"duration={duration_s:.1f}s")
    if annotation_count:
        bits.append(f"calls={annotation_count}")
    if species_codes:
        bits.append("species=" + ",".join(species_codes[:3]))
    interesting_tags = [tag for tag in context_tags if tag != UNKNOWN_CONTEXT]
    if interesting_tags:
        bits.append("tags=" + ",".join(interesting_tags[:3]))
    return " | ".join(bits)


def _raw_window_duration(raw_window_predictions: Sequence[PredictedSegment]) -> float:
    durations = [
        max(0.0, float(pred.end_time_s) - float(pred.start_time_s))
        for pred in raw_window_predictions
        if pred.end_time_s > pred.start_time_s
    ]
    if not durations:
        return 10.0
    durations.sort()
    return float(durations[len(durations) // 2])


def _select_diverse_examples(candidates: Sequence[ExampleCandidate], max_count: int) -> List[ExampleCandidate]:
    sorted_candidates = sorted(candidates, key=lambda item: item.sort_key)
    selected: List[ExampleCandidate] = []
    used_ids = set()
    used_files = set()

    for label in _PRIORITY_LABELS:
        for candidate in sorted_candidates:
            if len(selected) >= max_count:
                break
            if candidate.example_id in used_ids:
                continue
            if label not in candidate.category_labels:
                continue
            if candidate.filename in used_files and len(selected) < max_count // 2:
                continue
            selected.append(candidate)
            used_ids.add(candidate.example_id)
            used_files.add(candidate.filename)
            break

    for candidate in sorted_candidates:
        if len(selected) >= max_count:
            break
        if candidate.example_id in used_ids:
            continue
        if candidate.filename in used_files and len(selected) < max_count // 2:
            continue
        selected.append(candidate)
        used_ids.add(candidate.example_id)
        used_files.add(candidate.filename)

    for candidate in sorted_candidates:
        if len(selected) >= max_count:
            break
        if candidate.example_id in used_ids:
            continue
        selected.append(candidate)
        used_ids.add(candidate.example_id)

    return selected[:max_count]


def _annotation_bucket_priority(bucket: str) -> int:
    if bucket == FIN_BUCKET_20:
        return 0
    if bucket == FIN_BUCKET_40:
        return 1
    return 2


def _build_example_candidates(
    *,
    merged_predictions: Sequence[PredictedSegment],
    merged_useful_prediction_ids: Sequence[str],
    merged_unmatched_predictions: Sequence[PredictedSegment],
    merged_missed_annotations: Sequence[AnnotationEvent],
    raw_window_predictions: Sequence[PredictedSegment],
    raw_window_useful_prediction_ids: Sequence[str],
    raw_window_threshold: Optional[float],
    annotations: Sequence[AnnotationEvent],
    clip_manifest: Mapping[str, ClipManifestRow],
    postprocessed_item_map: Mapping[str, Mapping[str, Any]],
    raw_window_item_map: Mapping[str, Mapping[str, Any]],
    mat_lookup: Mapping[str, Path],
    mat_dir: Optional[Path],
    postprocessed_json_dir: Optional[Path],
    raw_window_json_dir: Optional[Path],
    collar_s: float,
) -> Dict[str, List[ExampleCandidate]]:
    annotations_by_file: Dict[str, List[AnnotationEvent]] = {}
    for ann in annotations:
        annotations_by_file.setdefault(ann.filename, []).append(ann)

    merged_tp_ids = set(merged_useful_prediction_ids)
    raw_tp_ids = set(raw_window_useful_prediction_ids)
    raw_positive_ids = set()
    raw_negative_predictions: List[PredictedSegment] = []

    for pred in raw_window_predictions:
        if raw_window_threshold is not None and pred.score >= float(raw_window_threshold):
            raw_positive_ids.add(pred.prediction_id)
        else:
            raw_negative_predictions.append(pred)

    out: Dict[str, List[ExampleCandidate]] = {
        "merged_tp": [],
        "merged_fp": [],
        "merged_fn": [],
        "raw_window_tp": [],
        "raw_window_fp": [],
        "raw_window_fn": [],
        "raw_window_tn": [],
    }

    for pred in merged_predictions:
        if pred.prediction_id not in merged_tp_ids:
            continue
        matched_annotations = _annotations_for_prediction(pred, annotations_by_file, collar_s)
        clip_row = clip_manifest.get(pred.filename)
        bucket_labels = tuple(sorted({ann.call_type_bucket or FIN_BUCKET_OTHER for ann in matched_annotations}))
        context_tags = tuple(sorted({tag for ann in matched_annotations for tag in ann.context_tags} or (clip_row.context_tags if clip_row else (UNKNOWN_CONTEXT,))))
        species_codes = clip_row.species_codes if clip_row else ()
        out["merged_tp"].append(
            ExampleCandidate(
                group="merged_tp",
                example_id=pred.prediction_id,
                filename=pred.filename,
                display_start_s=max(pred.start_time_s - 2.0, -12.0),
                display_end_s=pred.end_time_s + 2.0,
                prediction_start_s=pred.start_time_s,
                prediction_end_s=pred.end_time_s,
                score=pred.score,
                mat_path=_resolve_mat_path(
                    filename=pred.filename,
                    item=postprocessed_item_map.get(pred.item_id),
                    mat_lookup=mat_lookup,
                    mat_dir=mat_dir,
                    json_base_dir=postprocessed_json_dir,
                ),
                bucket_labels=bucket_labels,
                context_tags=context_tags,
                species_codes=species_codes,
                annotation_spans=_annotation_spans_for_rows(matched_annotations),
                category_labels=_category_labels(
                    bucket_labels=bucket_labels,
                    context_tags=context_tags,
                    species_codes=species_codes,
                ),
                panel_title=_prediction_panel_title(
                    base_label="Merged TP",
                    buckets=bucket_labels,
                    context_tags=context_tags,
                ),
                detail_text=_prediction_detail_text(
                    score=pred.score,
                    duration_s=max(0.0, pred.end_time_s - pred.start_time_s),
                    annotation_count=len(matched_annotations),
                    species_codes=species_codes,
                    context_tags=context_tags,
                ),
                sort_key=(
                    -len(matched_annotations),
                    -float(pred.score),
                    max(0.0, pred.end_time_s - pred.start_time_s),
                    pred.filename,
                    pred.start_time_s,
                ),
            )
        )

    for pred in merged_unmatched_predictions:
        clip_row = clip_manifest.get(pred.filename)
        context_tags = clip_row.context_tags if clip_row else (UNKNOWN_CONTEXT,)
        species_codes = clip_row.species_codes if clip_row else ()
        out["merged_fp"].append(
            ExampleCandidate(
                group="merged_fp",
                example_id=pred.prediction_id,
                filename=pred.filename,
                display_start_s=max(pred.start_time_s - 2.0, -12.0),
                display_end_s=pred.end_time_s + 2.0,
                prediction_start_s=pred.start_time_s,
                prediction_end_s=pred.end_time_s,
                score=pred.score,
                mat_path=_resolve_mat_path(
                    filename=pred.filename,
                    item=postprocessed_item_map.get(pred.item_id),
                    mat_lookup=mat_lookup,
                    mat_dir=mat_dir,
                    json_base_dir=postprocessed_json_dir,
                ),
                bucket_labels=(),
                context_tags=context_tags,
                species_codes=species_codes,
                annotation_spans=(),
                category_labels=_category_labels(
                    bucket_labels=(),
                    context_tags=context_tags,
                    species_codes=species_codes,
                ),
                panel_title=_prediction_panel_title(
                    base_label="Merged FP",
                    buckets=(),
                    context_tags=context_tags,
                ),
                detail_text=_prediction_detail_text(
                    score=pred.score,
                    duration_s=max(0.0, pred.end_time_s - pred.start_time_s),
                    annotation_count=0,
                    species_codes=species_codes,
                    context_tags=context_tags,
                ),
                sort_key=(
                    -float(pred.score),
                    max(0.0, pred.end_time_s - pred.start_time_s),
                    pred.filename,
                    pred.start_time_s,
                ),
            )
        )

    for ann in merged_missed_annotations:
        clip_row = clip_manifest.get(ann.filename)
        context_tags = ann.context_tags or (clip_row.context_tags if clip_row else (UNKNOWN_CONTEXT,))
        species_codes = clip_row.species_codes if clip_row else ()
        display_pad = max(5.0, min(12.0, 0.5 * max(6.0, ann.end_time_s - ann.begin_time_s)))
        out["merged_fn"].append(
            ExampleCandidate(
                group="merged_fn",
                example_id=ann.annotation_id,
                filename=ann.filename,
                display_start_s=max(ann.begin_time_s - display_pad, -12.0),
                display_end_s=ann.end_time_s + display_pad,
                prediction_start_s=None,
                prediction_end_s=None,
                score=None,
                mat_path=_resolve_mat_path(
                    filename=ann.filename,
                    item=None,
                    mat_lookup=mat_lookup,
                    mat_dir=mat_dir,
                    json_base_dir=postprocessed_json_dir,
                ),
                bucket_labels=(ann.call_type_bucket or FIN_BUCKET_OTHER,),
                context_tags=context_tags,
                species_codes=species_codes,
                annotation_spans=((ann.begin_time_s, ann.end_time_s, ann.call_type_bucket or FIN_BUCKET_OTHER),),
                category_labels=_category_labels(
                    bucket_labels=(ann.call_type_bucket or FIN_BUCKET_OTHER,),
                    context_tags=context_tags,
                    species_codes=species_codes,
                ),
                panel_title=_prediction_panel_title(
                    base_label="Merged FN",
                    buckets=(ann.call_type_bucket or FIN_BUCKET_OTHER,),
                    context_tags=context_tags,
                ),
                detail_text=_prediction_detail_text(
                    score=None,
                    duration_s=max(0.0, ann.end_time_s - ann.begin_time_s),
                    annotation_count=1,
                    species_codes=species_codes,
                    context_tags=context_tags,
                ),
                sort_key=(
                    _annotation_bucket_priority(ann.call_type_bucket or FIN_BUCKET_OTHER),
                    0 if "vessel_or_masking" in context_tags else 1,
                    ann.filename,
                    ann.begin_time_s,
                ),
            )
        )

    raw_window_missed_annotations = [
        ann
        for ann in annotations
        if not any(
            pred.prediction_id in raw_tp_ids
            and pred.filename == ann.filename
            and _overlaps_window(pred.start_time_s, pred.end_time_s, ann, collar_s)
            for pred in raw_window_predictions
        )
    ]
    raw_window_duration_s = _raw_window_duration(raw_window_predictions)

    for pred in raw_window_predictions:
        is_positive = pred.prediction_id in raw_positive_ids
        if not is_positive:
            continue
        matched_annotations = _annotations_for_prediction(pred, annotations_by_file, collar_s)
        clip_row = clip_manifest.get(pred.filename)
        bucket_labels = tuple(sorted({ann.call_type_bucket or FIN_BUCKET_OTHER for ann in matched_annotations}))
        context_tags = tuple(sorted({tag for ann in matched_annotations for tag in ann.context_tags} or (clip_row.context_tags if clip_row else (UNKNOWN_CONTEXT,))))
        species_codes = clip_row.species_codes if clip_row else ()
        target_group = "raw_window_tp" if pred.prediction_id in raw_tp_ids else "raw_window_fp"
        out[target_group].append(
            ExampleCandidate(
                group=target_group,
                example_id=pred.prediction_id,
                filename=pred.filename,
                display_start_s=pred.start_time_s,
                display_end_s=pred.end_time_s,
                prediction_start_s=pred.start_time_s,
                prediction_end_s=pred.end_time_s,
                score=pred.score,
                mat_path=_resolve_mat_path(
                    filename=pred.filename,
                    item=raw_window_item_map.get(pred.item_id),
                    mat_lookup=mat_lookup,
                    mat_dir=mat_dir,
                    json_base_dir=raw_window_json_dir,
                ),
                bucket_labels=bucket_labels,
                context_tags=context_tags,
                species_codes=species_codes,
                annotation_spans=_annotation_spans_for_rows(matched_annotations),
                category_labels=_category_labels(
                    bucket_labels=bucket_labels,
                    context_tags=context_tags,
                    species_codes=species_codes,
                ),
                panel_title=_prediction_panel_title(
                    base_label="Raw TP" if target_group == "raw_window_tp" else "Raw FP",
                    buckets=bucket_labels,
                    context_tags=context_tags,
                ),
                detail_text=_prediction_detail_text(
                    score=pred.score,
                    duration_s=max(0.0, pred.end_time_s - pred.start_time_s),
                    annotation_count=len(matched_annotations),
                    species_codes=species_codes,
                    context_tags=context_tags,
                ),
                sort_key=(
                    -len(matched_annotations),
                    -float(pred.score),
                    pred.filename,
                    pred.start_time_s,
                ),
            )
        )

    for ann in raw_window_missed_annotations:
        clip_row = clip_manifest.get(ann.filename)
        context_tags = ann.context_tags or (clip_row.context_tags if clip_row else (UNKNOWN_CONTEXT,))
        species_codes = clip_row.species_codes if clip_row else ()
        mid = 0.5 * (ann.begin_time_s + ann.end_time_s)
        half = 0.5 * raw_window_duration_s
        out["raw_window_fn"].append(
            ExampleCandidate(
                group="raw_window_fn",
                example_id=f"rawfn::{ann.annotation_id}",
                filename=ann.filename,
                display_start_s=max(mid - half, -12.0),
                display_end_s=mid + half,
                prediction_start_s=None,
                prediction_end_s=None,
                score=None,
                mat_path=_resolve_mat_path(
                    filename=ann.filename,
                    item=None,
                    mat_lookup=mat_lookup,
                    mat_dir=mat_dir,
                    json_base_dir=raw_window_json_dir,
                ),
                bucket_labels=(ann.call_type_bucket or FIN_BUCKET_OTHER,),
                context_tags=context_tags,
                species_codes=species_codes,
                annotation_spans=((ann.begin_time_s, ann.end_time_s, ann.call_type_bucket or FIN_BUCKET_OTHER),),
                category_labels=_category_labels(
                    bucket_labels=(ann.call_type_bucket or FIN_BUCKET_OTHER,),
                    context_tags=context_tags,
                    species_codes=species_codes,
                ),
                panel_title=_prediction_panel_title(
                    base_label="Raw FN",
                    buckets=(ann.call_type_bucket or FIN_BUCKET_OTHER,),
                    context_tags=context_tags,
                ),
                detail_text=_prediction_detail_text(
                    score=None,
                    duration_s=raw_window_duration_s,
                    annotation_count=1,
                    species_codes=species_codes,
                    context_tags=context_tags,
                ),
                sort_key=(
                    _annotation_bucket_priority(ann.call_type_bucket or FIN_BUCKET_OTHER),
                    0 if "vessel_or_masking" in context_tags else 1,
                    ann.filename,
                    ann.begin_time_s,
                ),
            )
        )

    for pred in raw_negative_predictions:
        matched_annotations = _annotations_for_prediction(pred, annotations_by_file, collar_s)
        if matched_annotations:
            continue
        clip_row = clip_manifest.get(pred.filename)
        if clip_row is None:
            continue
        context_tags = clip_row.context_tags or (UNKNOWN_CONTEXT,)
        species_codes = clip_row.species_codes
        tn_priority = 0 if clip_row.is_annotated_non_fin else 1
        out["raw_window_tn"].append(
            ExampleCandidate(
                group="raw_window_tn",
                example_id=f"rawtn::{pred.prediction_id}",
                filename=pred.filename,
                display_start_s=pred.start_time_s,
                display_end_s=pred.end_time_s,
                prediction_start_s=pred.start_time_s,
                prediction_end_s=pred.end_time_s,
                score=pred.score,
                mat_path=_resolve_mat_path(
                    filename=pred.filename,
                    item=raw_window_item_map.get(pred.item_id),
                    mat_lookup=mat_lookup,
                    mat_dir=mat_dir,
                    json_base_dir=raw_window_json_dir,
                ),
                bucket_labels=(),
                context_tags=context_tags,
                species_codes=species_codes,
                annotation_spans=(),
                category_labels=_category_labels(
                    bucket_labels=(),
                    context_tags=context_tags,
                    species_codes=species_codes,
                ),
                panel_title=_prediction_panel_title(
                    base_label="Raw TN",
                    buckets=(),
                    context_tags=context_tags,
                ),
                detail_text=_prediction_detail_text(
                    score=pred.score,
                    duration_s=max(0.0, pred.end_time_s - pred.start_time_s),
                    annotation_count=0,
                    species_codes=species_codes,
                    context_tags=context_tags,
                ),
                sort_key=(
                    tn_priority,
                    -float(pred.score),
                    pred.filename,
                    pred.start_time_s,
                ),
            )
        )

    return out


def _load_mat_crop(
    mat_path: Path,
    display_start_s: float,
    display_end_s: float,
) -> Tuple[Any, Any, Any]:
    from scipy.io import loadmat
    import numpy as np

    data = loadmat(str(mat_path), simplify_cells=True)
    spec = data.get("PdB_norm")
    if spec is None:
        power = data.get("P")
        if power is None:
            raise KeyError(f"Could not find spectrogram data in {mat_path}")
        power = np.abs(np.asarray(power, dtype=np.float32))
        max_power = float(np.max(power)) if power.size else 1.0
        if max_power <= 0:
            max_power = 1.0
        spec = 10.0 * np.log10(np.maximum(power / max_power, 1e-10))
    spec = np.asarray(spec, dtype=np.float32)
    freqs = np.asarray(data.get("F"), dtype=np.float32).reshape(-1)
    times = np.asarray(data.get("T"), dtype=np.float32).reshape(-1)

    if spec.shape[0] != freqs.shape[0] and spec.shape[1] == freqs.shape[0]:
        spec = spec.T
    if spec.shape[1] != times.shape[0] and spec.shape[0] == times.shape[0]:
        spec = spec.T

    if times.size == 0:
        return spec, freqs, times

    start = float(display_start_s)
    end = float(display_end_s)
    if end <= start:
        end = start + 1.0
    mask = (times >= start) & (times <= end)
    if not mask.any():
        start_idx = max(0, min(int(np.searchsorted(times, start, side="left")), len(times) - 1))
        end_idx = max(start_idx + 1, min(int(np.searchsorted(times, end, side="right")), len(times)))
        mask = np.zeros_like(times, dtype=bool)
        mask[start_idx:end_idx] = True
    if int(mask.sum()) == 1 and len(times) > 1:
        index = int(np.flatnonzero(mask)[0])
        start_idx = max(0, index - 1)
        end_idx = min(len(times), index + 2)
        mask = np.zeros_like(times, dtype=bool)
        mask[start_idx:end_idx] = True
    return spec[:, mask], freqs, times[mask]


def _render_candidate_png(candidate: ExampleCandidate, out_path: Path) -> Optional[Path]:
    if candidate.mat_path is None or not candidate.mat_path.exists():
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from textwrap import fill
    except Exception:
        return None

    from .part2_eval import _configure_plot_style

    spec, freqs, times = _load_mat_crop(candidate.mat_path, candidate.display_start_s, candidate.display_end_s)
    if times.size == 0:
        return None

    _configure_plot_style(plt)
    duration = max(1.0, float(times[-1]) - float(times[0]))
    fig_width = min(14.0, max(7.0, 6.0 + 0.045 * duration))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=220)

    import numpy as np

    vmin = float(np.percentile(spec, 2))
    vmax = float(np.percentile(spec, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    im = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    if candidate.prediction_start_s is not None and candidate.prediction_end_s is not None:
        ax.axvspan(
            float(candidate.prediction_start_s),
            float(candidate.prediction_end_s),
            facecolor="#ffb703",
            alpha=0.12,
            linewidth=0,
        )
        ax.axvline(float(candidate.prediction_start_s), color="#fb8500", linewidth=1.5, alpha=0.9)
        ax.axvline(float(candidate.prediction_end_s), color="#fb8500", linewidth=1.5, alpha=0.9)

    for start_s, end_s, label in candidate.annotation_spans:
        ax.axvspan(float(start_s), float(end_s), facecolor="#8ecae6", alpha=0.22, linewidth=0)
        if len(candidate.annotation_spans) <= 8:
            ax.text(
                0.5 * (float(start_s) + float(end_s)),
                float(freqs[-1]) - 1.5,
                label,
                ha="center",
                va="top",
                fontsize=8,
                color="#0b1f2a",
                bbox={"facecolor": "#caf0f8", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            )

    ax.set_xlabel("Time within clip (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(False)
    if len(times) > 1 and float(times[-1]) > float(times[0]):
        ax.set_xlim(float(times[0]), float(times[-1]))
    else:
        center = float(times[0])
        ax.set_xlim(center - 0.5, center + 0.5)
    ax.set_ylim(float(freqs[0]), float(freqs[-1]))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Relative power (dB)", rotation=270, labelpad=14)
    cbar.ax.grid(False)

    subtitle_bits = [Path(candidate.filename).name]
    if candidate.detail_text:
        subtitle_bits.append(candidate.detail_text)
    subtitle_text = fill(" | ".join(bit for bit in subtitle_bits if bit), width=max(88, int(fig_width * 12)))
    fig.suptitle(
        candidate.panel_title,
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.94,
        subtitle_text,
        ha="left",
        va="top",
        fontsize=9,
        color="#4a5568",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _build_contact_sheet(
    *,
    title: str,
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    rows = [row for row in rows if row.get("image_path")]
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    from .part2_eval import _configure_plot_style

    _configure_plot_style(plt)
    cols = 2
    n_items = len(rows)
    n_rows = int(math.ceil(n_items / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(11.5, max(3.8, 3.6 * n_rows)), dpi=220)
    if hasattr(axes, "flat"):
        axes_flat = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        axes_flat = list(axes)
    else:
        axes_flat = [axes]

    for ax, row in zip(axes_flat, rows):
        image = plt.imread(row["image_path"])
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(row.get("short_meta", row.get("group", "")), fontsize=9, loc="left")
        ax.set_title(Path(str(row.get("filename", ""))).name, fontsize=8, loc="right", color="#4a5568")

    for ax in axes_flat[len(rows):]:
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def export_part2_example_gallery(
    *,
    output_dir: Path,
    postprocessed_json_path: Path,
    postprocessed_payload: Mapping[str, Any],
    merged_predictions: Sequence[PredictedSegment],
    merged_useful_prediction_ids: Sequence[str],
    merged_unmatched_predictions: Sequence[PredictedSegment],
    merged_missed_annotations: Sequence[AnnotationEvent],
    raw_window_json_path: Optional[Path],
    raw_window_payload: Optional[Mapping[str, Any]],
    raw_window_predictions: Optional[Sequence[PredictedSegment]],
    raw_window_threshold: Optional[float],
    annotations: Sequence[AnnotationEvent],
    clip_manifest: Mapping[str, ClipManifestRow],
    mat_dir: Optional[Path],
    max_examples_per_group: int = 8,
    match_collar_s: float = 1.0,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mat_lookup = _build_mat_lookup(mat_dir) if mat_dir is not None and mat_dir.exists() else {}
    postprocessed_items = postprocessed_payload.get("items", []) if isinstance(postprocessed_payload, Mapping) else []
    postprocessed_item_map = {
        str(item.get("item_id")): item
        for item in postprocessed_items
        if isinstance(item, Mapping) and item.get("item_id")
    }
    raw_window_items = raw_window_payload.get("items", []) if isinstance(raw_window_payload, Mapping) else []
    raw_window_item_map = {
        str(item.get("item_id")): item
        for item in raw_window_items
        if isinstance(item, Mapping) and item.get("item_id")
    }

    raw_window_preds = list(raw_window_predictions or [])
    raw_window_useful_prediction_ids: Sequence[str] = []
    if raw_window_preds and raw_window_threshold is not None:
        positive_predictions = [pred for pred in raw_window_preds if pred.score >= float(raw_window_threshold)]
        from .part2_eval import coverage_match_sets

        raw_window_useful_prediction_ids, _ = coverage_match_sets(
            positive_predictions,
            annotations,
            match_collar_s,
        )

    candidates_by_group = _build_example_candidates(
        merged_predictions=merged_predictions,
        merged_useful_prediction_ids=list(merged_useful_prediction_ids),
        merged_unmatched_predictions=merged_unmatched_predictions,
        merged_missed_annotations=merged_missed_annotations,
        raw_window_predictions=raw_window_preds,
        raw_window_useful_prediction_ids=list(raw_window_useful_prediction_ids),
        raw_window_threshold=raw_window_threshold,
        annotations=annotations,
        clip_manifest=clip_manifest,
        postprocessed_item_map=postprocessed_item_map,
        raw_window_item_map=raw_window_item_map,
        mat_lookup=mat_lookup,
        mat_dir=mat_dir,
        postprocessed_json_dir=postprocessed_json_path.parent,
        raw_window_json_dir=raw_window_json_path.parent if raw_window_json_path is not None else None,
        collar_s=match_collar_s,
    )

    index_rows: List[Dict[str, Any]] = []
    group_summaries: List[Dict[str, Any]] = []
    readme_lines = [
        "# Part 2 Example Spectrogram Gallery",
        "",
        "Representative examples are grouped by merged-region coverage and raw-window detection outcomes.",
        "",
    ]

    for group_name, title in (
        ("merged_tp", "Merged Clip Coverage: True Positives"),
        ("merged_fp", "Merged Clip Coverage: False Positives"),
        ("merged_fn", "Merged Clip Coverage: Missed Annotated Calls"),
        ("raw_window_tp", "Raw Window Detection: True Positives"),
        ("raw_window_fp", "Raw Window Detection: False Positives"),
        ("raw_window_fn", "Raw Window Detection: Missed Annotated Calls"),
        ("raw_window_tn", "Raw Window Detection: True Negatives"),
    ):
        selected = _select_diverse_examples(candidates_by_group.get(group_name, []), max_examples_per_group)
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        group_rows: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(selected, start=1):
            file_stub = "_".join(
                part
                for part in [
                    f"{idx:02d}",
                    group_name,
                    candidate.bucket_labels[0] if candidate.bucket_labels else "",
                    next((tag for tag in candidate.context_tags if tag != UNKNOWN_CONTEXT), ""),
                    _slugify(candidate.filename),
                ]
                if part
            )
            out_path = group_dir / f"{file_stub}.png"
            rendered = _render_candidate_png(candidate, out_path)
            row = {
                "group": group_name,
                "filename": candidate.filename,
                "image_path": str(rendered) if rendered is not None else "",
                "mat_path": str(candidate.mat_path) if candidate.mat_path is not None else "",
                "display_start_s": f"{candidate.display_start_s:.6f}",
                "display_end_s": f"{candidate.display_end_s:.6f}",
                "prediction_start_s": "" if candidate.prediction_start_s is None else f"{candidate.prediction_start_s:.6f}",
                "prediction_end_s": "" if candidate.prediction_end_s is None else f"{candidate.prediction_end_s:.6f}",
                "score": "" if candidate.score is None else f"{candidate.score:.6f}",
                "bucket_labels": "|".join(candidate.bucket_labels),
                "context_tags": "|".join(candidate.context_tags),
                "species_codes": "|".join(candidate.species_codes),
                "annotation_count": len(candidate.annotation_spans),
                "panel_title": candidate.panel_title,
                "detail_text": candidate.detail_text,
                "short_meta": ",".join(list(candidate.bucket_labels[:1]) + [tag for tag in candidate.context_tags if tag != UNKNOWN_CONTEXT][:1]),
            }
            group_rows.append(row)
            index_rows.append(row)

        group_csv = group_dir / "examples.csv"
        from .part2_eval import write_csv

        write_csv(group_csv, group_rows)
        sheet_path = output_dir / "contact_sheets" / f"{group_name}_contact_sheet.png"
        contact_sheet = _build_contact_sheet(title=title, rows=group_rows, out_path=sheet_path)
        group_summaries.append(
            {
                "group": group_name,
                "title": title,
                "count": len(group_rows),
                "contact_sheet_path": str(contact_sheet) if contact_sheet else "",
                "examples_csv": str(group_csv),
            }
        )
        readme_lines.extend([f"## {title}", ""])
        if contact_sheet is not None:
            readme_lines.append(f"![{title}](contact_sheets/{contact_sheet.name})")
            readme_lines.append("")
        readme_lines.append(f"- Example count: `{len(group_rows)}`")
        readme_lines.append(f"- Manifest: `{group_name}/examples.csv`")
        readme_lines.append("")

    from .part2_eval import write_csv

    write_csv(output_dir / "examples_index.csv", index_rows)
    manifest_payload = {
        "max_examples_per_group": int(max_examples_per_group),
        "groups": group_summaries,
    }
    (output_dir / "examples_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return manifest_payload
