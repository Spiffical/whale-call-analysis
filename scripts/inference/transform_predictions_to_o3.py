#!/usr/bin/env python3
"""Transform predictions JSON to strict O3 unified schema v2.1.

This script removes non-schema fields introduced by postprocessing/event aggregation
and keeps only fields allowed by labeling-verification-app's O3 schema.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prune_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _datetime_to_schema_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _parse_audio_file_start(file_name: Optional[str]) -> Optional[datetime]:
    if not file_name:
        return None
    match = re.search(r"_(\d{8}T\d{6})(?:\.(\d+))?Z", Path(str(file_name)).name)
    if not match:
        return None
    dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    fraction = match.group(2)
    if fraction:
        micro = int((fraction + "000000")[:6])
        dt = dt.replace(microsecond=micro)
    return dt


def _is_epoch_seconds(value: Optional[float]) -> bool:
    return value is not None and value > 946684800.0


def _seconds_to_iso(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return _datetime_to_schema_iso(datetime.fromtimestamp(float(value), tz=timezone.utc))


def _source_format_from_name(file_name: Optional[str]) -> Optional[str]:
    if not file_name:
        return None
    suffix = Path(file_name).suffix.lower().lstrip(".")
    return suffix or None


def _safe_item_id_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "source"


def _clean_model(model: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(model, dict):
        return None
    out = _prune_none(
        {
            "model_id": _as_str(model.get("model_id")),
            "model_version": _as_str(model.get("model_version")),
            "architecture": _as_str(model.get("architecture")),
            "checkpoint_path": _as_str(model.get("checkpoint_path")),
            "checkpoint_url": _as_str(model.get("checkpoint_url")),
            "trained_at": _as_str(model.get("trained_at")),
            "wandb_run_id": _as_str(model.get("wandb_run_id")),
            "training_dataset_id": _as_str(model.get("training_dataset_id")),
            "training_dataset_version": _as_str(model.get("training_dataset_version")),
            "training_dataset_url": _as_str(model.get("training_dataset_url")),
            "training_data_time_range": _as_str(model.get("training_data_time_range")),
            "input_shape": model.get("input_shape") if isinstance(model.get("input_shape"), list) else None,
            "output_classes": model.get("output_classes") if isinstance(model.get("output_classes"), list) else None,
        }
    )
    if not out.get("model_id"):
        return None
    return out


def _clean_data_source(ds: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(ds, dict):
        return None
    out = _prune_none(
        {
            "data_source_id": _as_str(ds.get("data_source_id")),
            "device_code": _as_str(ds.get("device_code")),
            "deployment_id": _as_str(ds.get("deployment_id")),
            "location_name": _as_str(ds.get("location_name")),
            "site_code": _as_str(ds.get("site_code")),
            "latitude": _as_float(ds.get("latitude")),
            "longitude": _as_float(ds.get("longitude")),
            "depth_m": _as_float(ds.get("depth_m")),
            "channel": _as_str(ds.get("channel")),
            "sample_rate": _as_float(ds.get("sample_rate")),
            "is_calibrated": ds.get("is_calibrated") if isinstance(ds.get("is_calibrated"), bool) else None,
            "calibration_reference": _as_str(ds.get("calibration_reference")),
            "date_from": _as_str(ds.get("date_from")),
            "date_to": _as_str(ds.get("date_to")),
        }
    )
    if not out.get("data_source_id") or not out.get("device_code"):
        return None
    return out


def _clean_spectrogram_config(cfg: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(cfg, dict):
        return None

    freq_limits = cfg.get("frequency_limits")
    if isinstance(freq_limits, dict):
        freq_limits = _prune_none(
            {
                "min": _as_float(freq_limits.get("min")),
                "max": _as_float(freq_limits.get("max")),
            }
        )
        if set(freq_limits.keys()) != {"min", "max"}:
            freq_limits = None
    else:
        freq_limits = None

    source = cfg.get("source")
    if isinstance(source, dict):
        source = _prune_none(
            {
                "type": _as_str(source.get("type")),
                "generator": _as_str(source.get("generator")),
                "backend": _as_str(source.get("backend")),
                "onc_data_product_code": _as_str(source.get("onc_data_product_code")),
                "onc_data_product_options": source.get("onc_data_product_options")
                if isinstance(source.get("onc_data_product_options"), dict)
                else None,
            }
        )
        if not source:
            source = None
    else:
        source = None

    audio_source = cfg.get("audio_source")
    if isinstance(audio_source, dict):
        audio_source = _prune_none(
            {
                "type": _as_str(audio_source.get("type")),
                "onc_data_product_code": _as_str(audio_source.get("onc_data_product_code")),
                "format": _as_str(audio_source.get("format")),
            }
        )
        if not audio_source:
            audio_source = None
    else:
        audio_source = None

    out = _prune_none(
        {
            "nfft": int(cfg.get("nfft")) if isinstance(cfg.get("nfft"), (int, float)) else None,
            "window_function": _as_str(cfg.get("window_function")),
            "window_duration_sec": _as_float(cfg.get("window_duration_sec") or cfg.get("window_duration")),
            "hop_length": int(cfg.get("hop_length")) if isinstance(cfg.get("hop_length"), (int, float)) else None,
            "overlap": _as_float(cfg.get("overlap")),
            "frequency_limits": freq_limits,
            "context_duration_sec": _as_float(cfg.get("context_duration_sec") or cfg.get("context_duration")),
            "crop_size": int(cfg.get("crop_size")) if isinstance(cfg.get("crop_size"), (int, float)) else None,
            "source": source,
            "audio_source": audio_source,
        }
    )
    return out if out else None


def _clean_pipeline(p: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(p, dict):
        return None
    out = _prune_none(
        {
            "pipeline_version": _as_str(p.get("pipeline_version")),
            "pipeline_commit": _as_str(p.get("pipeline_commit")),
            "pipeline_repo": _as_str(p.get("pipeline_repo")),
        }
    )
    return out if out else None


def _clean_model_outputs(model_outputs: Any) -> List[Dict[str, Any]]:
    if not isinstance(model_outputs, list):
        return []
    out: List[Dict[str, Any]] = []
    for mo in model_outputs:
        if not isinstance(mo, dict):
            continue
        cls = _as_str(mo.get("class_hierarchy"))
        score = _as_float(mo.get("score"))
        if cls is None or score is None:
            continue
        rec = _prune_none(
            {
                "class_hierarchy": cls,
                "class_id": _as_str(mo.get("class_id")),
                "score": score,
                "annotation_extent": _clean_annotation_extent(mo.get("annotation_extent")),
            }
        )
        out.append(rec)
    return out


def _clean_source_audio(source_audio: Any) -> Optional[Dict[str, Any]]:
    if isinstance(source_audio, dict):
        file_name = _as_str(source_audio.get("file_name"))
        if not file_name:
            return None
        out = _prune_none(
            {
                "file_name": file_name,
                "format": _as_str(source_audio.get("format")),
                "uri": _as_str(source_audio.get("uri")),
                "recording_start_time": _as_str(source_audio.get("recording_start_time")),
                "recording_end_time": _as_str(source_audio.get("recording_end_time")),
                "checksum_sha256": _as_str(source_audio.get("checksum_sha256")),
            }
        )
        return out if out else None
    if isinstance(source_audio, str):
        source_name = source_audio.strip()
        if source_name:
            return {"file_name": Path(source_name).name}
    return None


def _clean_annotation_extent(extent: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(extent, dict):
        return None
    extent_type = _as_str(extent.get("type"))
    if extent_type not in {"clip", "time_range", "freq_range", "time_freq_box"}:
        return None
    out = _prune_none(
        {
            "type": extent_type,
            "time_start_sec": _as_float(extent.get("time_start_sec")),
            "time_end_sec": _as_float(extent.get("time_end_sec")),
            "freq_min_hz": _as_float(extent.get("freq_min_hz")),
            "freq_max_hz": _as_float(extent.get("freq_max_hz")),
        }
    )
    if extent_type == "time_range" and not {"time_start_sec", "time_end_sec"}.issubset(out.keys()):
        return None
    if extent_type == "time_freq_box" and not {"time_start_sec", "time_end_sec", "freq_min_hz", "freq_max_hz"}.issubset(out.keys()):
        return None
    if extent_type == "freq_range" and not {"freq_min_hz", "freq_max_hz"}.issubset(out.keys()):
        return None
    return out


def _clean_label_decisions(label_decisions: Any) -> List[Dict[str, Any]]:
    if not isinstance(label_decisions, list):
        return []
    out: List[Dict[str, Any]] = []
    allowed_decisions = {"accepted", "rejected", "added"}
    for ld in label_decisions:
        if not isinstance(ld, dict):
            continue
        label = _as_str(ld.get("label"))
        decision = _as_str(ld.get("decision"))
        threshold_used = ld.get("threshold_used")
        if label is None or decision not in allowed_decisions:
            continue
        if threshold_used is not None:
            threshold_used = _as_float(threshold_used)
        rec = {
            "label": label,
            "decision": decision,
            "threshold_used": threshold_used,
            "annotation_extent": _clean_annotation_extent(ld.get("annotation_extent")),
        }
        out.append(rec)
    return out


def _clean_verifications(verifications: Any) -> List[Dict[str, Any]]:
    if not isinstance(verifications, list):
        return []
    out: List[Dict[str, Any]] = []
    allowed_status = {"verified", "rejected", "uncertain"}
    allowed_conf = {"high", "medium", "low", None}
    allowed_src = {"expert", "auto", "consensus"}
    for v in verifications:
        if not isinstance(v, dict):
            continue
        decisions = _clean_label_decisions(v.get("label_decisions"))
        verified_at = _as_str(v.get("verified_at"))
        verified_by = _as_str(v.get("verified_by"))
        verification_round = v.get("verification_round")
        if verified_at is None or verified_by is None or not isinstance(verification_round, (int, float)):
            continue
        verification_status = _as_str(v.get("verification_status"))
        if verification_status not in allowed_status:
            verification_status = None
        confidence = v.get("confidence")
        if isinstance(confidence, str):
            confidence = confidence.strip().lower()
        if confidence not in allowed_conf:
            confidence = None
        label_source = _as_str(v.get("label_source"))
        if label_source not in allowed_src:
            label_source = None

        rec = _prune_none(
            {
                "verified_at": verified_at,
                "verified_by": verified_by,
                "reviewer_affiliation": _as_str(v.get("reviewer_affiliation")),
                "verification_round": int(verification_round),
                "verification_status": verification_status,
                "label_decisions": decisions,
                "confidence": confidence,
                "notes": _as_str(v.get("notes")),
                "label_source": label_source,
                "taxonomy_version": _as_str(v.get("taxonomy_version")),
            }
        )
        # label_decisions is required by schema for each verification object.
        if "label_decisions" not in rec:
            rec["label_decisions"] = []
        out.append(rec)
    return out


def _clean_paths(paths: Any, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = paths if isinstance(paths, dict) else {}
    # Legacy fallback support in case paths object is absent.
    spec_mat = _as_str(p.get("spectrogram_mat_path") if isinstance(p, dict) else None) or _as_str(item.get("spectrogram_mat_path") or item.get("mat_path"))
    spec_png = _as_str(p.get("spectrogram_png_path") if isinstance(p, dict) else None) or _as_str(item.get("spectrogram_png_path") or item.get("spectrogram_path"))
    audio = _as_str(p.get("audio_path") if isinstance(p, dict) else None) or _as_str(item.get("audio_path"))
    out = _prune_none(
        {
            "spectrogram_mat_path": spec_mat,
            "spectrogram_png_path": spec_png,
            "audio_path": audio,
        }
    )
    return out if out else None


def _source_audio_from_name(source_name: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    del segments  # Source audio describes the raw file; item times describe the event segment.
    source_start = _parse_audio_file_start(source_name)
    return _prune_none(
        {
            "file_name": Path(source_name).name,
            "format": _source_format_from_name(source_name),
            "recording_start_time": _datetime_to_schema_iso(source_start) if source_start else None,
        }
    )


def _source_segment_time_bounds(
    segments: List[Dict[str, Any]],
    source_name: Optional[str],
    fallback_start: Optional[str],
    fallback_end: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    starts: List[float] = []
    ends: List[float] = []
    for segment in segments:
        start = _as_float(segment.get("time_start_sec"))
        end = _as_float(segment.get("time_end_sec"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)

    if starts and ends:
        start = min(starts)
        end = max(ends)
        if _is_epoch_seconds(start) or _is_epoch_seconds(end):
            return _seconds_to_iso(start), _seconds_to_iso(end)

        source_start = _parse_audio_file_start(source_name)
        if source_start is not None:
            return (
                _datetime_to_schema_iso(source_start + timedelta(seconds=float(start))),
                _datetime_to_schema_iso(source_start + timedelta(seconds=float(end))),
            )

    return fallback_start, fallback_end


def _split_source_segments(item: Dict[str, Any]) -> List[Tuple[Optional[str], List[Dict[str, Any]]]]:
    segments = item.get("source_segments")
    if not isinstance(segments, list):
        return []

    grouped: List[Tuple[Optional[str], List[Dict[str, Any]]]] = []
    index_by_source: Dict[Optional[str], int] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        source = _as_str(segment.get("source_audio"))
        key = source or None
        if key not in index_by_source:
            index_by_source[key] = len(grouped)
            grouped.append((key, []))
        grouped[index_by_source[key]][1].append(segment)
    return grouped


def _score_for_source_segments(segments: List[Dict[str, Any]], fallback: Optional[float]) -> Optional[float]:
    scores = [_as_float(segment.get("score")) for segment in segments]
    scores = [score for score in scores if score is not None]
    return max(scores) if scores else fallback


def _clean_item_base(
    item: Dict[str, Any],
    *,
    item_id: str,
    source_audio: Optional[Dict[str, Any]],
    audio_start_time: Optional[str],
    audio_end_time: Optional[str],
    model_score_override: Optional[float] = None,
) -> Dict[str, Any]:
    model_outputs = _clean_model_outputs(item.get("model_outputs"))
    if model_score_override is not None:
        for output in model_outputs:
            output["score"] = float(model_score_override)

    out = _prune_none(
        {
            "item_id": item_id,
            "data_source_id": _as_str(item.get("data_source_id")),
            "audio_start_time": audio_start_time,
            "audio_end_time": audio_end_time,
            "model_outputs": model_outputs,
            "verifications": _clean_verifications(item.get("verifications")),
            "source_audio": source_audio,
            "paths": _clean_paths(item.get("paths"), item),
        }
    )
    if "model_outputs" not in out:
        out["model_outputs"] = []
    if "verifications" not in out:
        out["verifications"] = []
    return out


def _clean_items(item: Any) -> List[Dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    item_id = _as_str(item.get("item_id"))
    if item_id is None:
        return []

    fallback_start = _as_str(item.get("audio_start_time"))
    fallback_end = _as_str(item.get("audio_end_time"))
    fallback_score = None
    for output in _clean_model_outputs(item.get("model_outputs")):
        fallback_score = _as_float(output.get("score"))
        if fallback_score is not None:
            break

    source_groups = _split_source_segments(item)
    source_groups = [(source, segments) for source, segments in source_groups if source and segments]
    if source_groups:
        out_items: List[Dict[str, Any]] = []
        split_count = len(source_groups)
        for idx, (source_name, segments) in enumerate(source_groups, start=1):
            start_iso, end_iso = _source_segment_time_bounds(
                segments,
                source_name,
                fallback_start if split_count == 1 else None,
                fallback_end if split_count == 1 else None,
            )
            split_item_id = item_id
            if split_count > 1:
                split_item_id = f"{item_id}__source_{idx:02d}_{_safe_item_id_token(Path(source_name).stem)[:48]}"
            out_items.append(
                _clean_item_base(
                    item,
                    item_id=split_item_id,
                    source_audio=_source_audio_from_name(source_name, segments),
                    audio_start_time=start_iso,
                    audio_end_time=end_iso,
                    model_score_override=_score_for_source_segments(segments, fallback_score),
                )
            )
        return out_items

    return [
        _clean_item_base(
            item,
            item_id=item_id,
            source_audio=_clean_source_audio(item.get("source_audio")),
            audio_start_time=fallback_start,
            audio_end_time=fallback_end,
        )
    ]


def transform_to_o3(input_data: Dict[str, Any], *, keep_updated_at: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schema_version": "2.1",
        "created_at": _as_str(input_data.get("created_at")) or _now_iso(),
        "task_type": _as_str(input_data.get("task_type")) or "whale_detection",
    }
    updated_at = _as_str(input_data.get("updated_at"))
    if keep_updated_at:
        out["updated_at"] = updated_at or _now_iso()

    model = _clean_model(input_data.get("model"))
    if model is not None:
        out["model"] = model

    ds_list: List[Dict[str, Any]] = []
    for ds in input_data.get("data_sources", []) if isinstance(input_data.get("data_sources"), list) else []:
        cleaned = _clean_data_source(ds)
        if cleaned is not None:
            ds_list.append(cleaned)
    if ds_list:
        out["data_sources"] = ds_list

    spec_cfg = _clean_spectrogram_config(input_data.get("spectrogram_config"))
    if spec_cfg is not None:
        out["spectrogram_config"] = spec_cfg

    pipeline = _clean_pipeline(input_data.get("pipeline"))
    if pipeline is not None:
        out["pipeline"] = pipeline

    items: List[Dict[str, Any]] = []
    for item in input_data.get("items", []) if isinstance(input_data.get("items"), list) else []:
        items.extend(_clean_items(item))
    out["items"] = items

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Transform predictions JSON to strict O3 schema-compatible JSON")
    ap.add_argument("--input-json", required=True, type=str, help="Input predictions JSON (app/extended or unified)")
    ap.add_argument("--output-json", required=True, type=str, help="Output strict O3 JSON path")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    ap.add_argument(
        "--drop-updated-at",
        action="store_true",
        help="Omit updated_at from output",
    )
    args = ap.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path} (use --overwrite)")

    with open(input_path, "r") as f:
        input_data = json.load(f)

    transformed = transform_to_o3(input_data, keep_updated_at=not args.drop_updated_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transformed, f, indent=2)

    print(f"Input items: {len(input_data.get('items', []) if isinstance(input_data.get('items'), list) else [])}")
    print(f"Output items: {len(transformed.get('items', []))}")
    print(f"Saved strict O3 JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
