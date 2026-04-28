#!/usr/bin/env python3
"""
Post-process inference predictions with temporal clustering + hysteresis filtering.

This script is intended for sliding-window inference outputs (UnifiedPredictionTracker v2 JSON).
It reduces isolated false positives by keeping only event-like clusters:
1) candidate windows must exceed a low threshold
2) each kept cluster must contain at least one high-threshold window
3) each kept cluster must contain at least N windows
4) optional minimum cluster duration
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io
import soundfile as sf


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _safe_epoch_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.timestamp()


def _basename(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    return Path(path_value).name


def _item_path(item: Dict[str, Any], path_key: str) -> Optional[str]:
    """Resolve path from unified `paths` object with legacy fallback."""
    paths = item.get("paths")
    if isinstance(paths, dict):
        value = paths.get(path_key)
        if value:
            return value
    # Legacy flat fallback
    if path_key == "spectrogram_mat_path":
        return item.get("spectrogram_mat_path") or item.get("mat_path")
    if path_key == "spectrogram_png_path":
        return item.get("spectrogram_png_path") or item.get("spectrogram_path")
    if path_key == "audio_path":
        return item.get("audio_path")
    return None


def _base_id_from_item_id(item_id: Optional[str]) -> Optional[str]:
    if not item_id:
        return None
    text = str(item_id)
    if "_win" in text:
        return text.rsplit("_win", 1)[0]
    return text


def _extract_score(item: Dict[str, Any], class_hierarchy: Optional[str]) -> Optional[float]:
    outputs = item.get("model_outputs")
    if not isinstance(outputs, list):
        return None
    for output in outputs:
        if not isinstance(output, dict):
            continue
        if class_hierarchy is not None and output.get("class_hierarchy") != class_hierarchy:
            continue
        score = _safe_float(output.get("score"))
        if score is not None:
            return score
    if class_hierarchy is None:
        for output in outputs:
            if isinstance(output, dict):
                score = _safe_float(output.get("score"))
                if score is not None:
                    return score
    return None


def _extract_time_bounds(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    # Prefer normalized window/segment bounds helper (handles bin->second conversion).
    start_inferred, end_inferred = _infer_window_times(item, 0)
    if start_inferred is not None and end_inferred is not None and end_inferred > start_inferred:
        return start_inferred, end_inferred

    start = (
        _safe_float(item.get("window_time_start"))
        if item.get("window_time_start") is not None
        else _safe_float(item.get("segment_start_sec"))
    )
    end = (
        _safe_float(item.get("window_time_end"))
        if item.get("window_time_end") is not None
        else _safe_float(item.get("segment_end_sec"))
    )
    duration = _safe_float(item.get("duration_sec"))

    if start is not None and end is None and duration is not None:
        end = start + max(duration, 0.0)
    if start is None and end is not None and duration is not None:
        start = end - max(duration, 0.0)

    # Fallback to absolute timestamps when relative offsets are unavailable.
    if start is None:
        start = _safe_epoch_seconds(item.get("audio_start_time") or item.get("audio_timestamp"))
    if end is None:
        end = _safe_epoch_seconds(item.get("audio_end_time"))
        if end is None and start is not None and duration is not None:
            end = start + max(duration, 0.0)
    return start, end


def _source_audio_file_name(item: Dict[str, Any]) -> Optional[str]:
    source_audio = item.get("source_audio")
    if isinstance(source_audio, dict):
        file_name = source_audio.get("file_name")
        if isinstance(file_name, str) and file_name:
            return file_name
    elif isinstance(source_audio, str) and source_audio:
        return source_audio
    return None


def _group_key(item: Dict[str, Any], merge_across_source_audio: bool = False) -> str:
    if merge_across_source_audio:
        ds = item.get("data_source_id")
        if isinstance(ds, str) and ds:
            return ds
        device = item.get("device_code")
        if isinstance(device, str) and device:
            return device
        return "global_group"
    source_audio = _source_audio_file_name(item)
    if source_audio:
        return source_audio
    audio_path_name = _basename(_item_path(item, "audio_path"))
    if audio_path_name:
        return audio_path_name
    base = _base_id_from_item_id(item.get("item_id"))
    if base:
        return base
    return "unknown_group"


def _resolve_media_path(input_json: Path, path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p
    # First try JSON-local relative resolution (standard unified schema behavior).
    candidate = (input_json.parent / p).resolve()
    if candidate.exists():
        return candidate
    # Fallback: some producers store media paths relative to a higher dataset root.
    for base in input_json.parents:
        probe = (base / p).resolve()
        if probe.exists():
            return probe
    return candidate


def _to_output_rel(path: Optional[Path], output_json: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(output_json.parent.resolve()))
    except ValueError:
        return str(path)


def _find_key(data: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for key in keys:
        if key in data:
            return key
    lowered = {k.lower(): k for k in data.keys()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


SPECTRO_KEYS = ("spectrogram", "PdB_norm", "power_db_norm", "PdB", "P_db", "P", "PSD", "psd", "Sxx", "S", "spec")
POWER_KEYS = ("P", "Sxx", "PSD", "psd")
DB_KEYS = ("PdB_norm", "power_db_norm", "PdB", "P_db")
FREQ_KEYS = ("F", "frequencies", "freqs", "freq", "f")
TIME_KEYS = ("T", "times", "time", "t")


def _load_mat_spectrogram(
    mat_path: Path,
) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    # Prefer raw power when present so event-level merge can normalize once
    # globally instead of stitching pre-normalized window dB maps.
    key = _find_key(data, POWER_KEYS)
    spec_kind = "power"
    if key is None:
        key = _find_key(data, DB_KEYS)
        spec_kind = "db"
    if key is None:
        key = _find_key(data, SPECTRO_KEYS)
        spec_kind = "db"
    if key is None:
        raise KeyError(f"No spectrogram key found in {mat_path}")

    spec = np.asarray(data[key])
    if spec.ndim != 2:
        raise ValueError(f"Unexpected spectrogram ndim={spec.ndim} in {mat_path}")

    freq = None
    time = None
    fk = _find_key(data, FREQ_KEYS)
    tk = _find_key(data, TIME_KEYS)
    if fk in data:
        freq = np.asarray(data[fk]).squeeze()
    if tk in data:
        time = np.asarray(data[tk]).squeeze()

    if spec_kind == "power":
        spec = np.abs(spec.astype(np.float32))
    else:
        spec = spec.astype(np.float32)

    if freq is not None and time is not None:
        f_len = int(np.asarray(freq).ravel().shape[0])
        t_len = int(np.asarray(time).ravel().shape[0])
        r, c = spec.shape[:2]
        if (r, c) == (t_len, f_len):
            spec = spec.T

    return (
        spec,
        spec_kind,
        (np.asarray(freq).ravel() if freq is not None else None),
        (np.asarray(time).ravel() if time is not None else None),
    )


def _infer_window_times(item: Dict[str, Any], n_time_bins: int) -> Tuple[Optional[float], Optional[float]]:
    start = _safe_float(item.get("window_time_start"))
    end = _safe_float(item.get("window_time_end"))
    if start is not None and end is not None and end > start:
        # Backward-compatibility: some exports stored window_time_* in bin units.
        tbin = _safe_float(item.get("time_bin_seconds"))
        ws = _safe_float(item.get("window_start"))
        crop_t = _safe_int(item.get("crop_time_bins"))
        if crop_t is None:
            crop_t = _safe_int(item.get("crop_size"))
        if (
            tbin is not None
            and tbin > 0
            and ws is not None
            and abs(start - ws) < 1e-6
            and (crop_t is None or abs((end - start) - float(crop_t)) < 1e-3)
        ):
            start = float(start) * float(tbin)
            end = float(end) * float(tbin)
        return start, end
    start = _safe_float(item.get("segment_start_sec"))
    end = _safe_float(item.get("segment_end_sec"))
    if start is not None and end is not None and end > start:
        return start, end
    duration = _safe_float(item.get("duration_sec"))
    if start is not None and duration is not None and duration > 0:
        return start, start + duration
    if duration is not None and duration > 0:
        return 0.0, duration
    if n_time_bins > 1:
        return 0.0, float(n_time_bins - 1)
    if n_time_bins == 1:
        return 0.0, 1.0
    return None, None


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_absolute_times(member_items: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    starts: List[datetime] = []
    ends: List[datetime] = []
    for item in member_items:
        s = _parse_iso(item.get("audio_start_time") or item.get("audio_timestamp"))
        e = _parse_iso(item.get("audio_end_time"))
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)
    start_iso = min(starts).isoformat() if starts else None
    end_iso = max(ends).isoformat() if ends else None
    return start_iso, end_iso


def _absolute_time_bounds_seconds(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    start = _safe_epoch_seconds(item.get("audio_start_time") or item.get("audio_timestamp"))
    end = _safe_epoch_seconds(item.get("audio_end_time"))
    if start is not None and end is None:
        duration = _safe_float(item.get("duration_sec"))
        if duration is not None and duration >= 0:
            end = start + duration
    return start, end


def _compact_utc_timestamp(value: Optional[str]) -> Optional[str]:
    dt = _parse_iso(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    # Millisecond precision keeps IDs compact while preserving ordering.
    return dt.strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"


def _safe_id_token(value: Optional[str], *, fallback: str, max_len: int = 80) -> str:
    if value is None:
        value = fallback
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in str(value)).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


def _extract_device_token(event: "Event", member_items: Sequence[Dict[str, Any]]) -> str:
    for item in member_items:
        source_audio = _source_audio_file_name(item)
        if source_audio:
            src_name = Path(source_audio).name
            if "_" in src_name:
                return _safe_id_token(src_name.split("_", 1)[0], fallback="unknown-device", max_len=32)
    for item in member_items:
        ds = item.get("data_source_id")
        if isinstance(ds, str) and ds:
            if "_" in ds:
                return _safe_id_token(ds.split("_", 1)[0], fallback="unknown-device", max_len=32)
            return _safe_id_token(ds, fallback="unknown-device", max_len=32)
    for item in member_items:
        dev = item.get("device_code")
        if isinstance(dev, str) and dev:
            return _safe_id_token(dev, fallback="unknown-device", max_len=32)
    group_name = Path(str(event.group)).name
    if "_" in group_name:
        return _safe_id_token(group_name.split("_", 1)[0], fallback="unknown-device", max_len=32)
    return "unknown-device"


def _build_event_id_base(event: "Event", member_items: Sequence[Dict[str, Any]]) -> str:
    abs_start, abs_end = _event_absolute_times(member_items)
    device = _extract_device_token(event, member_items)
    start_token = _compact_utc_timestamp(abs_start)
    end_token = _compact_utc_timestamp(abs_end)
    if start_token and end_token:
        core = f"fw-{device}-{start_token}-{end_token}"
    else:
        start_local = "na" if event.start_sec is None else f"{event.start_sec:.3f}".replace(".", "p")
        end_local = "na" if event.end_sec is None else f"{event.end_sec:.3f}".replace(".", "p")
        core = f"fw-{device}-s{start_local}-e{end_local}"
    group_hash = hashlib.sha1(str(event.group).encode("utf-8")).hexdigest()[:8]
    return _safe_id_token(f"{core}-g{group_hash}", fallback=f"fw-{device}-g{group_hash}", max_len=160)


def _assign_descriptive_event_ids(events: Sequence["Event"], items: Sequence[Dict[str, Any]]) -> None:
    used: Dict[str, int] = {}
    for event in events:
        member_items = [
            items[i]
            for i in event.member_indices
            if 0 <= i < len(items) and isinstance(items[i], dict)
        ]
        base = _build_event_id_base(event, member_items)
        n = used.get(base, 0) + 1
        used[base] = n
        event.event_id = f"{base}-r{n:02d}" if n > 1 else base


def _build_window_metadata(
    item: Dict[str, Any],
    *,
    class_hierarchy: Optional[str],
    window_id: int,
) -> Dict[str, Any]:
    start_sec = _safe_float(item.get("window_time_start"))
    end_sec = _safe_float(item.get("window_time_end"))
    start_inferred, end_inferred = _infer_window_times(item, 0)
    if start_inferred is not None and end_inferred is not None and end_inferred > start_inferred:
        start_sec, end_sec = start_inferred, end_inferred
    if start_sec is None or end_sec is None:
        start_sec, end_sec = _extract_time_bounds(item)
    score = _extract_score(item, class_hierarchy)
    win_start_idx = item.get("window_start")
    crop_size = item.get("crop_size")
    win_end_idx = None
    try:
        if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            win_end_idx = int(win_start_idx) + int(crop_size[1]) if win_start_idx is not None else None
        elif crop_size is not None and win_start_idx is not None:
            win_end_idx = int(win_start_idx) + int(crop_size)
    except (TypeError, ValueError):
        win_end_idx = None

    out: Dict[str, Any] = {
        "window_id": int(window_id),
        "source_item_id": item.get("item_id"),
        "source_audio": _source_audio_file_name(item),
        "time_start_sec": start_sec,
        "time_end_sec": end_sec,
        "score": score,
    }
    if win_start_idx is not None and win_end_idx is not None:
        out["window_indices"] = [int(win_start_idx), int(win_end_idx)]
    audio_rel = _item_path(item, "audio_path")
    mat_rel = _item_path(item, "spectrogram_mat_path")
    if audio_rel is not None:
        out["audio_path"] = audio_rel
    if mat_rel is not None:
        out["spectrogram_mat_path"] = mat_rel
    parent_mat_rel = item.get("parent_spectrogram_mat_path")
    parent_audio_rel = item.get("parent_audio_path")
    if parent_mat_rel is not None:
        out["parent_spectrogram_mat_path"] = str(parent_mat_rel)
    if parent_audio_rel is not None:
        out["parent_audio_path"] = str(parent_audio_rel)
    for key in (
        "parent_freq_bin_start",
        "parent_freq_bin_end",
        "parent_time_bin_start",
        "parent_time_bin_end",
    ):
        value = item.get(key)
        if value is not None:
            out[key] = value
    return out


def _common_parent_path(member_items: Sequence[Dict[str, Any]], key: str) -> Optional[str]:
    values: List[str] = []
    for item in member_items:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    if not values:
        return None
    uniq = sorted(set(values))
    if len(uniq) != 1:
        return None
    return uniq[0]


def _crop_time_bins_from_item(item: Dict[str, Any]) -> Optional[int]:
    crop_size = item.get("crop_size")
    if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
        return _safe_int(crop_size[1])
    if crop_size is not None:
        return _safe_int(crop_size)
    return _safe_int(item.get("crop_time_bins"))


def _crop_freq_bins_from_item(item: Dict[str, Any]) -> Optional[int]:
    crop_size = item.get("crop_size")
    if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 1:
        return _safe_int(crop_size[0])
    if crop_size is not None:
        return _safe_int(crop_size)
    return _safe_int(item.get("crop_freq_bins"))


def _parent_time_bounds(member_items: Sequence[Dict[str, Any]], total_bins: int) -> Optional[Tuple[int, int]]:
    starts: List[int] = []
    ends: List[int] = []
    for item in member_items:
        start = _safe_int(item.get("parent_time_bin_start"))
        end = _safe_int(item.get("parent_time_bin_end"))
        if start is None or end is None:
            win_start = _safe_int(item.get("window_start"))
            crop_t = _crop_time_bins_from_item(item)
            if win_start is not None and crop_t is not None:
                start = win_start
                end = win_start + crop_t
        if start is None or end is None:
            continue
        starts.append(int(start))
        ends.append(int(end))
    if not starts or not ends:
        return None
    t0 = max(0, min(starts))
    t1 = min(int(total_bins), max(ends))
    if t1 <= t0:
        return None
    return t0, t1


def _parent_freq_bounds(member_items: Sequence[Dict[str, Any]], total_bins: int) -> Optional[Tuple[int, int]]:
    starts: List[int] = []
    ends: List[int] = []
    for item in member_items:
        start = _safe_int(item.get("parent_freq_bin_start"))
        end = _safe_int(item.get("parent_freq_bin_end"))
        if start is None or end is None:
            crop_f = _crop_freq_bins_from_item(item)
            if crop_f is not None and crop_f > 0:
                start = max(0, (int(total_bins) - int(crop_f)) // 2)
                end = start + int(crop_f)
        if start is None or end is None:
            continue
        starts.append(int(start))
        ends.append(int(end))
    if starts and ends:
        f0 = max(0, min(starts))
        f1 = min(int(total_bins), max(ends))
        if f1 > f0:
            return f0, f1
    if total_bins <= 0:
        return None
    return 0, int(total_bins)


def _event_time_bounds_seconds(member_items: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    starts: List[float] = []
    ends: List[float] = []
    for item in member_items:
        start, end = _infer_window_times(item, 0)
        if start is not None:
            starts.append(float(start))
        if end is not None:
            ends.append(float(end))
    if not starts or not ends:
        return None, None
    start = float(min(starts))
    end = float(max(ends))
    if end <= start:
        return None, None
    return start, end


def _event_time_bounds_from_parent_mat(
    member_items: Sequence[Dict[str, Any]],
    input_json: Path,
) -> Tuple[Optional[float], Optional[float]]:
    parent_mat_rel = _common_parent_path(member_items, "parent_spectrogram_mat_path")
    if parent_mat_rel is None:
        return None, None
    parent_mat = _resolve_media_path(input_json, parent_mat_rel)
    if parent_mat is None or not parent_mat.exists():
        return None, None
    try:
        data = scipy.io.loadmat(str(parent_mat), simplify_cells=True)
    except Exception:
        return None, None
    t = data.get("T")
    if t is None:
        return None, None
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    if t_arr.size == 0:
        return None, None
    bounds = _parent_time_bounds(member_items, int(t_arr.size))
    if bounds is None:
        return None, None
    t0, t1 = bounds
    t0 = max(0, min(int(t0), int(t_arr.size) - 1))
    t1 = max(t0 + 1, min(int(t1), int(t_arr.size)))
    diffs = np.diff(t_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size:
        dt = float(np.median(diffs))
        start = float(t_arr[t0]) - 0.5 * dt
        end = float(t_arr[t1 - 1]) + 0.5 * dt
    else:
        start = float(t_arr[t0])
        end = float(t_arr[t1 - 1])
    if end <= start:
        return None, None
    return max(0.0, start), max(start, end)


def _extract_event_spectrogram_from_parent(
    event_id: str,
    member_items: Sequence[Dict[str, Any]],
    input_json: Path,
    output_dir: Path,
    output_json: Path,
) -> Optional[str]:
    parent_rel = _common_parent_path(member_items, "parent_spectrogram_mat_path")
    if parent_rel is None:
        return None
    parent_mat = _resolve_media_path(input_json, parent_rel)
    if parent_mat is None or not parent_mat.exists():
        return None
    try:
        parent_spec, spec_kind, freq, times = _load_mat_spectrogram(parent_mat)
    except Exception:
        return None

    f_bins, t_bins = parent_spec.shape
    time_bounds = _parent_time_bounds(member_items, t_bins)
    freq_bounds = _parent_freq_bounds(member_items, f_bins)
    if time_bounds is None or freq_bounds is None:
        return None
    t0, t1 = time_bounds
    f0, f1 = freq_bounds

    spec_slice = parent_spec[f0:f1, t0:t1]
    if spec_slice.ndim != 2 or spec_slice.shape[1] == 0:
        return None

    if freq is not None and np.asarray(freq).size == f_bins:
        freq_slice = np.asarray(freq, dtype=np.float32).ravel()[f0:f1]
    else:
        freq_slice = np.arange(f0, f1, dtype=np.float32)

    if times is not None and np.asarray(times).size == t_bins:
        time_slice = np.asarray(times, dtype=np.float64).ravel()[t0:t1]
    else:
        start_sec, end_sec = _event_time_bounds_seconds(member_items)
        if start_sec is not None and end_sec is not None and t1 > t0:
            dt = (end_sec - start_sec) / float(max(t1 - t0, 1))
            dt = max(dt, 1e-6)
            time_slice = start_sec + np.arange(t1 - t0, dtype=np.float64) * dt
        else:
            time_slice = np.arange(t1 - t0, dtype=np.float64)

    if spec_kind == "power":
        power_slice = np.abs(spec_slice.astype(np.float32))
        max_power = float(np.max(power_slice)) if power_slice.size else 0.0
        if max_power > 0:
            db_slice = 10.0 * np.log10(np.maximum(power_slice / max_power, 1e-10))
        else:
            db_slice = np.full_like(power_slice, -100.0, dtype=np.float32)
    else:
        power_slice = None
        db_slice = np.minimum(spec_slice.astype(np.float32), 0.0)

    spec_dir = output_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    out_path = spec_dir / f"{event_id}.mat"
    payload: Dict[str, Any] = {
        "PdB_norm": db_slice.astype(np.float32),
        "F": np.asarray(freq_slice, dtype=np.float32),
        "T": np.asarray(time_slice, dtype=np.float64),
        "parent_freq_bin_start": np.int32(f0),
        "parent_freq_bin_end": np.int32(f1),
        "parent_time_bin_start": np.int32(t0),
        "parent_time_bin_end": np.int32(t1),
    }
    if power_slice is not None:
        payload["P"] = power_slice.astype(np.float32)
    scipy.io.savemat(str(out_path), payload)
    return _to_output_rel(out_path, output_json)


def _extract_event_audio_from_parent(
    event_id: str,
    member_items: Sequence[Dict[str, Any]],
    input_json: Path,
    output_dir: Path,
    output_json: Path,
) -> Optional[str]:
    parent_rel = _common_parent_path(member_items, "parent_audio_path")
    if parent_rel is None:
        return None
    parent_audio = _resolve_media_path(input_json, parent_rel)
    if parent_audio is None or not parent_audio.exists():
        return None
    # Prefer deriving time bounds from parent spectrogram time bins to avoid
    # unit-mismatch issues when window_time_start/end are stored as bin indices.
    start_sec, end_sec = _event_time_bounds_from_parent_mat(member_items, input_json)
    if start_sec is None or end_sec is None or end_sec <= start_sec:
        start_sec, end_sec = _event_time_bounds_seconds(member_items)
    if start_sec is None or end_sec is None or end_sec <= start_sec:
        return None

    try:
        with sf.SoundFile(str(parent_audio)) as f:
            sr = int(f.samplerate)
            if sr <= 0:
                return None
            start_frame = int(max(0.0, start_sec) * sr)
            end_frame = int(max(start_sec, end_sec) * sr)
            start_frame = max(0, min(start_frame, len(f)))
            end_frame = max(start_frame, min(end_frame, len(f)))
            if end_frame <= start_frame:
                return None
            f.seek(start_frame)
            wav = f.read(end_frame - start_frame)
    except Exception:
        return None

    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"{event_id}.wav"
    sf.write(str(out_path), wav.astype(np.float32), sr, subtype="FLOAT")
    return _to_output_rel(out_path, output_json)


def _merge_event_spectrogram(
    event_id: str,
    member_items: Sequence[Dict[str, Any]],
    input_json: Path,
    output_dir: Path,
    output_json: Path,
) -> Optional[str]:
    rows: List[Dict[str, Any]] = []
    for item in member_items:
        mat_path = _resolve_media_path(input_json, _item_path(item, "spectrogram_mat_path"))
        if mat_path is None or not mat_path.exists():
            continue
        try:
            spec, spec_kind, freq, t = _load_mat_spectrogram(mat_path)
        except Exception:
            continue
        rel_start, rel_end = _infer_window_times(item, spec.shape[1])
        abs_start, abs_end = _absolute_time_bounds_seconds(item)
        start = abs_start if abs_start is not None else rel_start
        end = abs_end if abs_end is not None else rel_end
        uses_absolute_timeline = abs_start is not None
        if start is None:
            continue
        # Prefer actual clipped audio duration when available so merged spectrogram
        # and merged audio stay aligned sample-for-sample at event scale.
        audio_path = _resolve_media_path(input_json, _item_path(item, "audio_path"))
        if audio_path is not None and audio_path.exists():
            try:
                audio_info = sf.info(str(audio_path))
                if audio_info.samplerate > 0 and audio_info.frames >= 0:
                    end = float(start) + (float(audio_info.frames) / float(audio_info.samplerate))
            except Exception:
                pass

        spec = spec.astype(np.float32)
        n_cols = int(spec.shape[1])
        if n_cols <= 0:
            continue

        t_vec: Optional[np.ndarray] = None
        dt: Optional[float] = None

        # Prefer MAT time bins when available. For cross-file stitching, put
        # every crop on an absolute timeline so adjacent source clips sort and
        # overlap-trim correctly.
        if t is not None and np.asarray(t).size == n_cols:
            t_raw = np.asarray(t, dtype=np.float64).ravel()
            if t_raw.size >= 2:
                diffs = np.diff(t_raw)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    dt = float(np.median(diffs))
            if dt is not None and dt > 0 and np.all(np.isfinite(t_raw)):
                if uses_absolute_timeline:
                    t_vec = float(start) + (t_raw - float(t_raw[0]))
                else:
                    t_vec = t_raw

        # Fallback when MAT has no usable time axis.
        if t_vec is None and end is not None and end > start:
            if n_cols > 1:
                dt = float(end - start) / float(n_cols)
            else:
                dt = float(end - start)
            dt = max(float(dt), 1e-6)
            t_vec = float(start) + (0.5 * dt) + np.arange(n_cols, dtype=np.float64) * dt

        if dt is None or dt <= 0:
            dt = 1.0
        if t_vec is None or t_vec.size != n_cols:
            t_vec = float(start) + np.arange(n_cols, dtype=np.float64) * float(dt)

        win_duration = None
        if end is not None and end > start and n_cols >= 1:
            win_duration = float(end - start) - float(max(n_cols - 1, 0)) * float(dt)
            if win_duration <= 0:
                win_duration = None

        rows.append(
            {
                "spec": spec,
                "kind": spec_kind,
                "freq": freq,
                "times": t_vec.astype(np.float64),
                "start": float(start),
                "dt": float(dt),
                "win_duration": float(win_duration) if win_duration is not None else None,
                "uses_absolute_timeline": bool(uses_absolute_timeline),
            }
        )
    if not rows:
        return None

    # Keep only consistent frequency-bin members.
    freq_bins = rows[0]["spec"].shape[0]
    rows = [r for r in rows if r["spec"].shape[0] == freq_bins]
    if not rows:
        return None

    rows.sort(key=lambda r: (float(r["start"]), float(r["times"][0])))
    dts = [float(r["dt"]) for r in rows if r.get("dt") is not None and float(r["dt"]) > 0]
    dt_ref = float(median(dts)) if dts else 1.0
    dt_ref = max(dt_ref, 1e-6)

    kinds = {str(r.get("kind", "db")) for r in rows}
    merge_kind = "power" if kinds == {"power"} else "db"
    if merge_kind == "db":
        # Mixed inputs: convert any power member to dB independently.
        for row in rows:
            if str(row.get("kind")) == "power":
                spec_row = np.abs(np.asarray(row["spec"], dtype=np.float32))
                mx = float(np.max(spec_row)) if spec_row.size else 0.0
                if mx > 0:
                    row["spec"] = 10.0 * np.log10(np.maximum(spec_row / mx, 1e-10))
                else:
                    row["spec"] = np.full_like(spec_row, -100.0, dtype=np.float32)
                row["kind"] = "db"

    merged_spec = rows[0]["spec"].copy()
    merged_times = rows[0]["times"].copy()
    first_dt = float(rows[0]["dt"]) if float(rows[0]["dt"]) > 0 else dt_ref
    current_end = float(merged_times[-1]) + first_dt

    for row in rows[1:]:
        spec = row["spec"]
        times = row["times"].copy()
        dt_row = float(row["dt"]) if float(row["dt"]) > 0 else dt_ref
        if spec.shape[1] == 0 or times.size == 0:
            continue

        gap_sec = float(times[0]) - current_end
        # Remove overlap from the new segment (keep earlier data unchanged).
        if gap_sec < -0.5 * dt_ref:
            trim_cols = int(round((-gap_sec) / max(dt_row, 1e-6)))
            if trim_cols >= spec.shape[1]:
                continue
            spec = spec[:, trim_cols:]
            times = times[trim_cols:]
            if spec.shape[1] == 0 or times.size == 0:
                continue
            gap_sec = float(times[0]) - current_end

        # Pure concatenate strategy: trim overlaps and append; do not synthesize
        # bridge columns for positive gaps.
        if times.size:
            desired_start = current_end
            shift = desired_start - float(times[0])
            times = times + shift

        merged_spec = np.concatenate([merged_spec, spec], axis=1)
        merged_times = np.concatenate([merged_times, times], axis=0)
        current_end = float(merged_times[-1]) + dt_row

    if merge_kind == "power":
        merged_power = np.abs(np.asarray(merged_spec, dtype=np.float32))
        max_power = float(np.max(merged_power)) if merged_power.size else 0.0
        if max_power > 0:
            merged = 10.0 * np.log10(np.maximum(merged_power / max_power, 1e-10))
        else:
            merged = np.full_like(merged_power, -100.0, dtype=np.float32)
    else:
        merged_power = None
        merged = np.minimum(merged_spec, 0.0).astype(np.float32)
    timeline = merged_times.astype(np.float64)
    if any(bool(row.get("uses_absolute_timeline")) for row in rows) and timeline.size:
        timeline = timeline - float(timeline[0])
    freq = rows[0]["freq"]
    if freq is None or np.asarray(freq).size != freq_bins:
        freq = np.arange(freq_bins, dtype=np.float32)

    spec_dir = output_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    out_path = spec_dir / f"{event_id}.mat"
    win_durations = [
        float(r["win_duration"])
        for r in rows
        if r.get("win_duration") is not None and float(r["win_duration"]) > 0
    ]
    win_duration_sec = float(median(win_durations)) if win_durations else None

    payload: Dict[str, Any] = {
        "PdB_norm": merged,
        "F": np.asarray(freq).astype(np.float32),
        "T": timeline.astype(np.float64),
    }
    if merged_power is not None:
        payload["P"] = merged_power.astype(np.float32)
    if win_duration_sec is not None:
        payload["window_duration_sec"] = float(win_duration_sec)

    scipy.io.savemat(
        str(out_path),
        payload,
    )
    return _to_output_rel(out_path, output_json)


def _merge_event_audio(
    event_id: str,
    member_items: Sequence[Dict[str, Any]],
    input_json: Path,
    output_dir: Path,
    output_json: Path,
) -> Optional[str]:
    rows: List[Dict[str, Any]] = []
    for item in member_items:
        audio_path = _resolve_media_path(input_json, _item_path(item, "audio_path"))
        if audio_path is None or not audio_path.exists():
            continue
        rel_start, rel_end = _infer_window_times(item, 0)
        abs_start, abs_end = _absolute_time_bounds_seconds(item)
        start = abs_start if abs_start is not None else rel_start
        end = abs_end if abs_end is not None else rel_end
        if start is None:
            continue
        try:
            wav, sr = sf.read(str(audio_path), always_2d=False)
        except Exception:
            continue
        wav = np.asarray(wav)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        rows.append(
            {
                "wav": wav.astype(np.float32),
                "sr": int(sr),
                "start": float(start),
                "end": float(end) if end is not None else float(start + (len(wav) / max(sr, 1))),
            }
        )
    if not rows:
        return None

    sr_values = {r["sr"] for r in rows}
    if len(sr_values) != 1:
        # Keep dominant sample rate and drop mismatches.
        target_sr = max(sr_values, key=lambda s: sum(1 for r in rows if r["sr"] == s))
        rows = [r for r in rows if r["sr"] == target_sr]
    if not rows:
        return None

    rows.sort(key=lambda r: (float(r["start"]), float(r["end"])))
    sr = int(rows[0]["sr"])
    if sr <= 0:
        return None

    merged = rows[0]["wav"].astype(np.float32).copy()
    merged_start = float(rows[0]["start"])
    current_end = merged_start + (merged.shape[0] / float(sr))

    for row in rows[1:]:
        wav = row["wav"].astype(np.float32)
        if wav.size == 0:
            continue
        start = float(row["start"])
        gap_sec = start - current_end

        # Remove overlap from the new segment (keep earlier audio unchanged).
        if gap_sec < -0.5 / float(sr):
            trim_samples = int(round((-gap_sec) * float(sr)))
            if trim_samples >= wav.shape[0]:
                continue
            wav = wav[trim_samples:]
            if wav.size == 0:
                continue
            gap_sec = start + (trim_samples / float(sr)) - current_end

        # Pure concatenate strategy: trim overlaps and append; do not insert
        # synthetic silence for positive gaps.
        merged = np.concatenate([merged, wav], axis=0)
        current_end = merged_start + (merged.shape[0] / float(sr))
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"{event_id}.wav"
    sf.write(str(out_path), merged.astype(np.float32), sr, subtype="FLOAT")
    return _to_output_rel(out_path, output_json)


def _spectrogram_duration_seconds(mat_path: Path) -> Optional[float]:
    try:
        data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception:
        return None
    t = data.get("T")
    if t is None:
        return None
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    if t_arr.size == 0:
        return None
    win_duration = data.get("window_duration_sec")
    if win_duration is not None:
        try:
            win_duration = float(np.asarray(win_duration).ravel()[0])
        except Exception:
            win_duration = None
    if win_duration is not None and np.isfinite(win_duration) and win_duration > 0:
        if t_arr.size == 1:
            return float(win_duration)
        return float(t_arr[-1] - t_arr[0]) + float(win_duration)
    if t_arr.size == 1:
        return 0.0
    diffs = np.diff(t_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    dt = float(np.median(diffs))
    return float(t_arr[-1] - t_arr[0]) + dt


def _align_audio_to_spectrogram_duration(audio_path: Path, mat_path: Path) -> None:
    spec_dur = _spectrogram_duration_seconds(mat_path)
    if spec_dur is None or spec_dur < 0:
        return
    try:
        info = sf.info(str(audio_path))
    except Exception:
        return
    sr = int(info.samplerate)
    if sr <= 0:
        return
    target_samples = int(round(float(spec_dur) * float(sr)))
    target_samples = max(target_samples, 0)
    current_samples = int(info.frames)
    if current_samples == target_samples:
        return
    try:
        wav, _ = sf.read(str(audio_path), always_2d=False)
    except Exception:
        return
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if wav.shape[0] > target_samples:
        wav = wav[:target_samples]
    elif wav.shape[0] < target_samples:
        wav = np.pad(wav, (0, target_samples - wav.shape[0]))
    sf.write(str(audio_path), wav.astype(np.float32), sr, subtype="FLOAT")


@dataclass
class Candidate:
    idx: int
    group: str
    score: float
    start_sec: Optional[float]
    end_sec: Optional[float]


@dataclass
class Event:
    event_id: str
    group: str
    member_indices: List[int]
    start_sec: Optional[float]
    end_sec: Optional[float]
    duration_sec: Optional[float]
    max_score: float
    mean_score: float
    n_members: int
    n_high: int
    inferred_gap_sec: Optional[float]


def _sort_key(c: Candidate) -> Tuple[int, float]:
    # Unknown times get pushed to the end while preserving deterministic order by idx.
    if c.start_sec is None:
        return (1, float(c.idx))
    return (0, float(c.start_sec))


def _infer_gap_seconds(cands: Sequence[Candidate]) -> Optional[float]:
    starts = [c.start_sec for c in cands if c.start_sec is not None]
    if len(starts) < 2:
        return None
    starts = sorted(starts)
    diffs: List[float] = []
    for i in range(1, len(starts)):
        d = starts[i] - starts[i - 1]
        if d > 0 and math.isfinite(d):
            diffs.append(d)
    if not diffs:
        return None
    return float(median(diffs))


def _cluster_candidates(
    candidates: Sequence[Candidate],
    low_threshold: float,
    high_threshold: float,
    min_members: int,
    min_duration_sec: float,
    max_gap_seconds: Optional[float],
) -> Tuple[List[Event], Dict[str, Any]]:
    by_group: Dict[str, List[Candidate]] = {}
    for c in candidates:
        if c.score >= low_threshold:
            by_group.setdefault(c.group, []).append(c)

    events: List[Event] = []
    event_idx = 0
    debug: Dict[str, Any] = {"groups": {}}

    for group, members in by_group.items():
        members_sorted = sorted(members, key=_sort_key)
        inferred_step = _infer_gap_seconds(members_sorted)
        effective_gap = max_gap_seconds
        if effective_gap is None and inferred_step is not None:
            effective_gap = max(inferred_step * 1.5, inferred_step + 1e-6)
        if effective_gap is None:
            effective_gap = 0.0

        clusters: List[List[Candidate]] = []
        current: List[Candidate] = []
        prev = None
        for c in members_sorted:
            if not current:
                current = [c]
                prev = c
                continue
            contiguous = False
            if prev is not None and prev.end_sec is not None and c.start_sec is not None:
                contiguous = (c.start_sec - prev.end_sec) <= float(effective_gap)
            elif prev is not None and prev.start_sec is not None and c.start_sec is not None:
                contiguous = (c.start_sec - prev.start_sec) <= float(effective_gap)
            else:
                contiguous = (c.idx == prev.idx + 1) if prev is not None else False

            if contiguous:
                current.append(c)
            else:
                clusters.append(current)
                current = [c]
            prev = c
        if current:
            clusters.append(current)

        kept = 0
        dropped = 0
        debug_clusters: List[Dict[str, Any]] = []
        for cluster in clusters:
            scores = [c.score for c in cluster]
            n_high = sum(1 for s in scores if s >= high_threshold)
            n_members = len(cluster)
            starts = [c.start_sec for c in cluster if c.start_sec is not None]
            ends = [c.end_sec for c in cluster if c.end_sec is not None]
            start_sec = min(starts) if starts else None
            end_sec = max(ends) if ends else None
            duration = (end_sec - start_sec) if (start_sec is not None and end_sec is not None) else None
            passes = (
                n_high >= 1
                and n_members >= min_members
                and ((duration is None) or (duration >= min_duration_sec))
            )
            if passes:
                event_idx += 1
                events.append(
                    Event(
                        event_id=f"evt_{event_idx:06d}",
                        group=group,
                        member_indices=[c.idx for c in cluster],
                        start_sec=start_sec,
                        end_sec=end_sec,
                        duration_sec=duration,
                        max_score=max(scores),
                        mean_score=float(sum(scores) / len(scores)),
                        n_members=n_members,
                        n_high=n_high,
                        inferred_gap_sec=float(effective_gap),
                    )
                )
                kept += 1
            else:
                dropped += 1
            debug_clusters.append(
                {
                    "n_members": n_members,
                    "n_high": n_high,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration,
                    "max_score": max(scores) if scores else None,
                    "kept": bool(passes),
                }
            )

        debug["groups"][group] = {
            "n_candidates": len(members_sorted),
            "n_clusters": len(clusters),
            "n_kept_clusters": kept,
            "n_dropped_clusters": dropped,
            "inferred_step_sec": inferred_step,
            "effective_gap_sec": effective_gap,
            "clusters": debug_clusters,
        }

    return events, debug


def _write_events_csv(
    path: Path,
    events: Sequence[Event],
    *,
    idx_to_item_id: Optional[Dict[int, str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "group",
                "start_sec",
                "end_sec",
                "duration_sec",
                "n_members",
                "n_high",
                "max_score",
                "mean_score",
                "member_item_ids",
            ],
        )
        w.writeheader()
        for e in events:
            w.writerow(
                {
                    "event_id": e.event_id,
                    "group": e.group,
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "duration_sec": e.duration_sec,
                    "n_members": e.n_members,
                    "n_high": e.n_high,
                    "max_score": e.max_score,
                    "mean_score": e.mean_score,
                    "member_item_ids": ",".join(
                        idx_to_item_id.get(i, str(i)) if idx_to_item_id is not None else str(i)
                        for i in e.member_indices
                    ),
                }
            )


def _write_summary_md(
    path: Path,
    *,
    input_json: Path,
    output_json: Path,
    low_threshold: float,
    high_threshold: float,
    min_members: int,
    min_duration_sec: float,
    max_gap_seconds: Optional[float],
    total_items: int,
    candidate_items: int,
    kept_items: int,
    n_events: int,
    by_group_counts: Dict[str, int],
) -> None:
    lines: List[str] = [
        "# Prediction Post-processing Summary",
        "",
        f"- input: `{input_json}`",
        f"- output: `{output_json}`",
        f"- generated_at: `{_iso_now()}`",
        f"- low_threshold: `{low_threshold:.4f}`",
        f"- high_threshold: `{high_threshold:.4f}`",
        f"- min_members: `{min_members}`",
        f"- min_duration_sec: `{min_duration_sec:.2f}`",
        f"- max_gap_seconds: `{max_gap_seconds if max_gap_seconds is not None else 'auto'}`",
        f"- total_input_items: `{total_items}`",
        f"- candidate_items(score>=low): `{candidate_items}`",
        f"- kept_items: `{kept_items}`",
        f"- kept_events: `{n_events}`",
        "",
        "## Kept Events Per Group",
        "",
    ]
    if by_group_counts:
        for group, count in sorted(by_group_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{group}`: `{count}`")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Post-process sliding-window predictions via temporal clustering + hysteresis."
    )
    ap.add_argument("--input-json", type=str, required=True, help="Input predictions JSON from run_inference.py")
    ap.add_argument("--output-json", type=str, required=True, help="Filtered predictions JSON output")
    ap.add_argument(
        "--class-hierarchy",
        type=str,
        default=None,
        help="Class hierarchy to score/filter. Default: first model output per item.",
    )
    ap.add_argument(
        "--low-threshold",
        type=float,
        default=0.70,
        help="Low threshold for candidate windows (cluster membership).",
    )
    ap.add_argument(
        "--high-threshold",
        type=float,
        default=0.82,
        help="High threshold required at least once in each kept cluster.",
    )
    ap.add_argument(
        "--min-members",
        type=int,
        default=2,
        help="Minimum number of windows in a kept cluster.",
    )
    ap.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Minimum cluster duration in seconds.",
    )
    ap.add_argument(
        "--max-gap-seconds",
        type=float,
        default=None,
        help="Maximum allowed inter-window gap within a cluster. Default: auto from median step.",
    )
    ap.add_argument(
        "--events-csv",
        type=str,
        default=None,
        help="Optional event summary CSV path (default: <output-json stem>_events.csv).",
    )
    ap.add_argument(
        "--summary-md",
        type=str,
        default=None,
        help="Optional markdown summary path (default: <output-json stem>_summary.md).",
    )
    ap.add_argument(
        "--debug-json",
        type=str,
        default=None,
        help="Optional detailed debug JSON (cluster diagnostics).",
    )
    ap.add_argument(
        "--merge-event-media",
        action="store_true",
        help="Create one merged spectrogram/audio clip per kept event by trimming overlaps and concatenating in time order.",
    )
    ap.add_argument(
        "--merge-min-score",
        type=float,
        default=None,
        help="Optional score floor for members included in event-media merge (default: use all kept event members).",
    )
    ap.add_argument(
        "--event-media-dir",
        type=str,
        default=None,
        help="Directory for merged event media (default: <output-json stem>_events_media).",
    )
    ap.add_argument(
        "--clear-event-media-dir",
        action="store_true",
        help="Delete existing event-media-dir contents before writing merged media.",
    )
    ap.add_argument(
        "--replace-items-with-events",
        action="store_true",
        help="Replace window-level items with one event-level item per kept event.",
    )
    ap.add_argument(
        "--merge-across-source-audio",
        action="store_true",
        help="Allow event clustering across adjacent source audio files (same data source/device).",
    )
    args = ap.parse_args()

    if not (0.0 <= args.low_threshold <= 1.0):
        raise SystemExit("--low-threshold must be in [0,1]")
    if not (0.0 <= args.high_threshold <= 1.0):
        raise SystemExit("--high-threshold must be in [0,1]")
    if args.high_threshold < args.low_threshold:
        raise SystemExit("--high-threshold must be >= --low-threshold")
    if args.min_members < 1:
        raise SystemExit("--min-members must be >= 1")
    if args.min_duration_sec < 0:
        raise SystemExit("--min-duration-sec must be >= 0")
    if args.max_gap_seconds is not None and args.max_gap_seconds < 0:
        raise SystemExit("--max-gap-seconds must be >= 0 when provided")
    if args.merge_min_score is not None and not (0.0 <= args.merge_min_score <= 1.0):
        raise SystemExit("--merge-min-score must be in [0,1]")

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    if not input_json.exists():
        raise SystemExit(f"Input JSON not found: {input_json}")

    with open(input_json, "r") as f:
        data = json.load(f)
    items = data.get("items")
    if not isinstance(items, list):
        raise SystemExit("Invalid predictions JSON: missing list field 'items'")

    candidates: List[Candidate] = []
    total_items = len(items)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        score = _extract_score(item, class_hierarchy=args.class_hierarchy)
        if score is None:
            continue
        start_sec, end_sec = _extract_time_bounds(item)
        if args.merge_across_source_audio:
            abs_start = _safe_epoch_seconds(item.get("audio_start_time") or item.get("audio_timestamp"))
            abs_end = _safe_epoch_seconds(item.get("audio_end_time"))
            if abs_start is not None:
                start_sec = abs_start
            if abs_end is not None:
                end_sec = abs_end
            elif abs_start is not None:
                duration = _safe_float(item.get("duration_sec"))
                if duration is not None and duration >= 0:
                    end_sec = abs_start + duration
        candidates.append(
            Candidate(
                idx=idx,
                group=_group_key(item, merge_across_source_audio=bool(args.merge_across_source_audio)),
                score=float(score),
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    events, debug = _cluster_candidates(
        candidates=candidates,
        low_threshold=float(args.low_threshold),
        high_threshold=float(args.high_threshold),
        min_members=int(args.min_members),
        min_duration_sec=float(args.min_duration_sec),
        max_gap_seconds=args.max_gap_seconds,
    )
    _assign_descriptive_event_ids(events, items)

    keep_idx_to_event: Dict[int, Event] = {}
    for event in events:
        for idx in event.member_indices:
            keep_idx_to_event[idx] = event

    kept_items: List[Dict[str, Any]] = []
    by_group_counts: Dict[str, int] = {}
    for idx, item in enumerate(items):
        event = keep_idx_to_event.get(idx)
        if event is None:
            continue
        new_item = dict(item)
        new_item["postprocess_event_id"] = event.event_id
        new_item["postprocess_group"] = event.group
        new_item["postprocess_event_max_score"] = float(event.max_score)
        new_item["postprocess_event_mean_score"] = float(event.mean_score)
        new_item["postprocess_event_n_members"] = int(event.n_members)
        new_item["postprocess_event_n_high"] = int(event.n_high)
        kept_items.append(new_item)
        by_group_counts[event.group] = by_group_counts.get(event.group, 0) + 1

    event_media_root = (
        Path(args.event_media_dir)
        if args.event_media_dir
        else output_json.with_name(f"{output_json.stem}_events_media")
    )
    if args.merge_event_media and args.clear_event_media_dir and event_media_root.exists():
        shutil.rmtree(event_media_root)
    event_items: List[Dict[str, Any]] = []
    events_payload: List[Dict[str, Any]] = []
    merged_media_count = 0
    merged_without_audio = 0
    merged_without_spectrogram = 0

    total_events = len(events)
    if args.merge_event_media and total_events > 0:
        print(f"Merging event media for {total_events} events...", flush=True)

    for event_idx, event in enumerate(events, start=1):
        member_items = [
            items[i]
            for i in event.member_indices
            if 0 <= i < len(items) and isinstance(items[i], dict)
        ]
        if not member_items:
            continue

        media_members = member_items
        if args.merge_min_score is not None:
            media_members = [
                m
                for m in member_items
                if (_extract_score(m, class_hierarchy=args.class_hierarchy) or -1.0) >= float(args.merge_min_score)
            ]

        merged_mat_rel = None
        merged_audio_rel = None
        if args.merge_event_media and media_members:
            # Prefer direct extraction from parent 5-minute media when available.
            merged_mat_rel = _extract_event_spectrogram_from_parent(
                event_id=event.event_id,
                member_items=media_members,
                input_json=input_json,
                output_dir=event_media_root,
                output_json=output_json,
            )
            merged_audio_rel = _extract_event_audio_from_parent(
                event_id=event.event_id,
                member_items=media_members,
                input_json=input_json,
                output_dir=event_media_root,
                output_json=output_json,
            )
            # Fallback to window stitching when parent references are unavailable.
            if merged_mat_rel is None:
                merged_mat_rel = _merge_event_spectrogram(
                    event_id=event.event_id,
                    member_items=media_members,
                    input_json=input_json,
                    output_dir=event_media_root,
                    output_json=output_json,
                )
            if merged_audio_rel is None:
                merged_audio_rel = _merge_event_audio(
                    event_id=event.event_id,
                    member_items=media_members,
                    input_json=input_json,
                    output_dir=event_media_root,
                    output_json=output_json,
                )
            if merged_mat_rel and merged_audio_rel:
                merged_mat_abs = _resolve_media_path(output_json, merged_mat_rel)
                merged_audio_abs = _resolve_media_path(output_json, merged_audio_rel)
                if (
                    merged_mat_abs is not None
                    and merged_audio_abs is not None
                    and merged_mat_abs.exists()
                    and merged_audio_abs.exists()
                ):
                    _align_audio_to_spectrogram_duration(merged_audio_abs, merged_mat_abs)
            if merged_mat_rel or merged_audio_rel:
                merged_media_count += 1
            if merged_mat_rel is None:
                merged_without_spectrogram += 1
            if merged_audio_rel is None:
                merged_without_audio += 1
        if args.merge_event_media and (
            event_idx == 1 or event_idx % 25 == 0 or event_idx == total_events
        ):
            print(
                f"  Event media progress: {event_idx}/{total_events}",
                flush=True,
            )

        member_ids = [str(m.get("item_id")) for m in member_items if m.get("item_id") is not None]
        parent_source_audio_files = sorted(
            {
                src_name
                for src_name in (_source_audio_file_name(m) for m in member_items)
                if src_name is not None
            }
        )
        source_segments = [
            _build_window_metadata(
                m,
                class_hierarchy=args.class_hierarchy,
                window_id=w_idx,
            )
            for w_idx, m in enumerate(member_items)
        ]
        event_payload: Dict[str, Any] = {
            "event_id": event.event_id,
            "group": event.group,
            "start_sec": event.start_sec,
            "end_sec": event.end_sec,
            "duration_sec": event.duration_sec,
            "max_score": float(event.max_score),
            "mean_score": float(event.mean_score),
            "n_members": int(event.n_members),
            "n_high": int(event.n_high),
            "member_item_ids": member_ids,
            "parent_source_audio_files": parent_source_audio_files,
            "source_segments": source_segments,
        }
        event_paths: Dict[str, str] = {}
        if merged_mat_rel:
            event_paths["spectrogram_mat_path"] = merged_mat_rel
        if merged_audio_rel:
            event_paths["audio_path"] = merged_audio_rel
        if event_paths:
            event_payload["paths"] = event_paths
        events_payload.append(event_payload)

        if args.replace_items_with_events:
            class_hierarchy = args.class_hierarchy
            if class_hierarchy is None:
                for m in member_items:
                    outputs = m.get("model_outputs")
                    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                        class_hierarchy = outputs[0].get("class_hierarchy")
                        if class_hierarchy:
                            break

            first_member = member_items[0]
            data_source_id = first_member.get("data_source_id")
            abs_start, abs_end = _event_absolute_times(member_items)
            event_item: Dict[str, Any] = {
                "item_id": event.event_id,
                "model_outputs": [
                    {
                        "class_hierarchy": class_hierarchy or "unknown",
                        "score": float(event.max_score),
                        "aggregation_method": "event_max",
                        "metadata": {
                            "event_mean_score": float(event.mean_score),
                            "event_n_members": int(event.n_members),
                            "event_n_high": int(event.n_high),
                            "parent_source_audio_files": parent_source_audio_files,
                            "windows": source_segments,
                        },
                    }
                ],
                "verifications": [],
                "postprocess_event_id": event.event_id,
                "postprocess_group": event.group,
                "postprocess_event_max_score": float(event.max_score),
                "postprocess_event_mean_score": float(event.mean_score),
                "postprocess_event_n_members": int(event.n_members),
                "postprocess_event_n_high": int(event.n_high),
                "event_member_item_ids": member_ids,
                "parent_source_audio_files": parent_source_audio_files,
                "source_segments": source_segments,
            }
            if data_source_id:
                event_item["data_source_id"] = data_source_id
            if abs_start:
                event_item["audio_start_time"] = abs_start
            if abs_end:
                event_item["audio_end_time"] = abs_end

            paths: Dict[str, str] = {}
            if merged_mat_rel:
                paths["spectrogram_mat_path"] = merged_mat_rel
            if merged_audio_rel:
                paths["audio_path"] = merged_audio_rel
            if not paths:
                fallback_mat = _item_path(first_member, "spectrogram_mat_path")
                fallback_audio = _item_path(first_member, "audio_path")
                if fallback_mat:
                    paths["spectrogram_mat_path"] = fallback_mat
                if fallback_audio:
                    paths["audio_path"] = fallback_audio
            if paths:
                event_item["paths"] = paths
            event_items.append(event_item)

    output_data = dict(data)
    output_data["items"] = event_items if args.replace_items_with_events else kept_items
    output_data["updated_at"] = _iso_now()
    output_data["events"] = events_payload
    output_data["postprocessing"] = {
        "method": "temporal_cluster_hysteresis_v1",
        "input_json": str(input_json),
        "generated_at": _iso_now(),
        "class_hierarchy": args.class_hierarchy,
        "low_threshold": float(args.low_threshold),
        "high_threshold": float(args.high_threshold),
        "min_members": int(args.min_members),
        "min_duration_sec": float(args.min_duration_sec),
        "max_gap_seconds": float(args.max_gap_seconds) if args.max_gap_seconds is not None else None,
        "total_items_in": total_items,
        "total_items_scored": len(candidates),
        "candidate_items": sum(1 for c in candidates if c.score >= args.low_threshold),
        "events_kept": len(events),
        "items_kept": len(kept_items),
        "merge_event_media": bool(args.merge_event_media),
        "merge_min_score": float(args.merge_min_score) if args.merge_min_score is not None else None,
        "replace_items_with_events": bool(args.replace_items_with_events),
        "merged_event_media_count": int(merged_media_count),
        "output_item_count": len(output_data["items"]),
        "merge_across_source_audio": bool(args.merge_across_source_audio),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    events_csv = Path(args.events_csv) if args.events_csv else output_json.with_name(f"{output_json.stem}_events.csv")
    summary_md = Path(args.summary_md) if args.summary_md else output_json.with_name(f"{output_json.stem}_summary.md")
    idx_to_item_id = {
        idx: str(item.get("item_id", idx))
        for idx, item in enumerate(items)
        if isinstance(item, dict)
    }
    _write_events_csv(events_csv, events, idx_to_item_id=idx_to_item_id)
    _write_summary_md(
        summary_md,
        input_json=input_json,
        output_json=output_json,
        low_threshold=float(args.low_threshold),
        high_threshold=float(args.high_threshold),
        min_members=int(args.min_members),
        min_duration_sec=float(args.min_duration_sec),
        max_gap_seconds=args.max_gap_seconds,
        total_items=total_items,
        candidate_items=sum(1 for c in candidates if c.score >= args.low_threshold),
        kept_items=len(kept_items),
        n_events=len(events),
        by_group_counts=by_group_counts,
    )

    if args.debug_json:
        debug_path = Path(args.debug_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w") as f:
            json.dump(debug, f, indent=2)

    print("Post-processing complete")
    print(f"  Input items: {total_items}")
    print(f"  Output items: {len(output_data['items'])}")
    print(f"  Events kept: {len(events)}")
    if args.merge_event_media:
        print(f"  Merged event media: {merged_media_count}")
        if merged_without_audio > 0:
            print(
                f"  Warning: {merged_without_audio} events have no merged audio path "
                "(check parent/window audio availability)."
            )
        if merged_without_spectrogram > 0:
            print(
                f"  Warning: {merged_without_spectrogram} events have no merged spectrogram path."
            )
    print(f"  Output JSON: {output_json}")
    print(f"  Events CSV: {events_csv}")
    print(f"  Summary MD: {summary_md}")
    if args.debug_json:
        print(f"  Debug JSON: {args.debug_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
