"""Detector-ready export helpers for the fin-whale bbox pipeline."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from PIL import Image, ImageDraw
from scipy import signal

from .audio import stitch_audio_files
from .finwhale_bbox import (
    FIN_SPECIES_CODE,
    PURE_NEGATIVE_DATASET,
    enumerate_gap_negative_contexts,
    load_annotation_manifest,
    load_clip_manifest,
    load_split_assignments,
)


CONTEXT_COLUMNS = [
    "context_id",
    "split_name",
    "source_dataset",
    "filename",
    "recording_day_utc",
    "context_type",
    "label",
    "context_start_s",
    "context_end_s",
    "context_duration_s",
    "event_annotation_id",
    "event_species_code",
    "event_begin_s",
    "event_end_s",
    "event_call_type_std",
    "context_image_relpath",
]

CROP_COLUMNS = [
    "crop_id",
    "context_id",
    "split_name",
    "source_dataset",
    "filename",
    "recording_day_utc",
    "context_type",
    "label",
    "crop_start_s",
    "crop_end_s",
    "crop_duration_s",
    "image_relpath",
    "image_width",
    "image_height",
    "fin_box_count",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sample_call_fraction_in_window(rng: np.random.Generator, center_bias_sigma_frac: float) -> float:
    sigma = max(1e-3, float(center_bias_sigma_frac)) * 0.5
    for _ in range(50):
        frac = float(rng.normal(loc=0.5, scale=sigma))
        if 0.0 <= frac <= 1.0:
            return frac
    return 0.5


def _clip_interval_within_context(
    *,
    desired_start_s: float,
    clip_duration_s: float,
    context_start_s: float,
    context_duration_s: float,
) -> float:
    context_end_s = float(context_start_s) + float(context_duration_s)
    max_start = max(float(context_start_s), context_end_s - float(clip_duration_s))
    return float(np.clip(float(desired_start_s), float(context_start_s), float(max_start)))


def _normalize_image(rgb: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8))
    image = image.resize((int(image_size), int(image_size)), resample=Image.BICUBIC)
    return np.asarray(image)


def _power_to_rgb(power_db: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    from matplotlib import cm

    arr = np.asarray(power_db, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr_min = float(arr.min()) if arr.size else 0.0
    arr_max = float(arr.max()) if arr.size else 1.0
    if arr_max > arr_min:
        arr01 = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr01 = np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(arr01, 2.0)) if arr01.size else 0.0
    hi = float(np.percentile(arr01, 98.0)) if arr01.size else 1.0
    if hi > lo:
        arr01 = np.clip((arr01 - lo) / (hi - lo), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    return (cmap(arr01)[..., :3] * 255.0).astype(np.uint8)


def _load_spec_params(config_path: Path | str, freq_min_hz: float, freq_max_hz: float) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    spec_cfg = cfg.get("custom_spectrograms", {}) or {}
    return {
        "win_dur": float(spec_cfg.get("window_duration", 1.0)),
        "overlap": float(spec_cfg.get("overlap", 0.9)),
        "freq_min_hz": float(freq_min_hz),
        "freq_max_hz": float(freq_max_hz),
    }


def _make_spectrogram_generator(spec_params: Dict[str, Any]) -> Any:
    try:
        from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator

        return SpectrogramGenerator(
            win_dur=float(spec_params["win_dur"]),
            overlap=float(spec_params["overlap"]),
            freq_lims=(float(spec_params["freq_min_hz"]), float(spec_params["freq_max_hz"])),
            log_freq=False,
            clim=(-60, 0),
            colormap="viridis",
        )
    except Exception:
        class _FallbackSpectrogramGenerator:
            def __init__(self, *, win_dur: float, overlap: float) -> None:
                self.win_dur = float(win_dur)
                self.overlap = float(overlap)

            def compute_spectrogram(self, audio_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                win_len = max(16, int(round(float(self.win_dur) * float(sample_rate))))
                noverlap = int(round(float(self.overlap) * float(win_len)))
                noverlap = max(0, min(noverlap, win_len - 1))
                freqs, times, sxx = signal.spectrogram(
                    np.asarray(audio_data, dtype=np.float32),
                    fs=float(sample_rate),
                    nperseg=int(win_len),
                    noverlap=int(noverlap),
                    detrend=False,
                    scaling="density",
                    mode="psd",
                )
                power_db = 10.0 * np.log10(np.maximum(sxx, 1e-12))
                return freqs, times, sxx, power_db

        return _FallbackSpectrogramGenerator(
            win_dur=float(spec_params["win_dur"]),
            overlap=float(spec_params["overlap"]),
        )


def _render_spectrogram_png(
    *,
    audio_data: np.ndarray,
    sample_rate: float,
    spec_gen: Any,
    freq_min_hz: float,
    freq_max_hz: float,
    edge_buffer_s: float,
    target_duration_s: float,
    image_size: int,
    output_path: Path,
) -> Tuple[int, int]:
    freqs, times, _sxx, power_db = spec_gen.compute_spectrogram(audio_data, sample_rate)
    freq_mask = (freqs >= float(freq_min_hz)) & (freqs <= float(freq_max_hz))
    freqs = freqs[freq_mask]
    power_db = power_db[freq_mask, :]

    if edge_buffer_s > 0:
        start_s = float(edge_buffer_s)
        end_s = start_s + float(target_duration_s)
        time_mask = (times >= start_s) & (times <= end_s)
        if not np.any(time_mask):
            t0 = int(np.searchsorted(times, start_s, side="left"))
            t1 = int(np.searchsorted(times, end_s, side="right"))
            t0 = max(0, min(t0, len(times) - 1))
            t1 = max(t0 + 1, min(t1, len(times)))
            time_mask = np.zeros_like(times, dtype=bool)
            time_mask[t0:t1] = True
        power_db = power_db[:, time_mask]

    rgb = _power_to_rgb(power_db)
    rgb = _normalize_image(rgb, image_size=int(image_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)
    return int(rgb.shape[1]), int(rgb.shape[0])


def _audio_duration(audio_path: Path) -> float:
    with sf.SoundFile(audio_path) as handle:
        return float(len(handle) / handle.samplerate)


def _load_audio_window_with_buffer(
    *,
    audio_dir: Path,
    filename: str,
    start_s: float,
    duration_s: float,
    edge_buffer_s: float,
) -> Tuple[np.ndarray, float]:
    audio_path = audio_dir / filename
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    with sf.SoundFile(audio_path) as handle:
        sample_rate = float(handle.samplerate)

    context_total_s = float(duration_s) + 2.0 * float(edge_buffer_s)
    audio = stitch_audio_files(
        onc_token="",
        clip_id=filename,
        device_code=filename.split("_", 1)[0],
        desired_start=float(start_s) - float(edge_buffer_s),
        desired_end=float(start_s) + float(duration_s) + float(edge_buffer_s),
        context_duration=float(context_total_s),
        audio_dir=audio_dir,
        show_onc_warnings=False,
        allow_downloads=False,
    )
    if audio is None:
        raise RuntimeError(f"failed to stitch audio window for {filename}")
    return np.asarray(audio, dtype=np.float32), sample_rate


def _context_window_from_event(begin_s: float, end_s: float, context_duration_s: float) -> Tuple[float, float]:
    center_s = 0.5 * (float(begin_s) + float(end_s))
    max_start = max(0.0, 300.0 - float(context_duration_s))
    start_s = float(np.clip(center_s - 0.5 * float(context_duration_s), 0.0, max_start))
    return start_s, start_s + float(context_duration_s)


def build_context_manifest(
    annotation_df: pd.DataFrame,
    clip_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    *,
    context_duration_s: float = 40.0,
    pure_zero_ratio: float = 0.5,
    negative_margin_s: float = 2.0,
    allowed_filenames: Optional[set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    allowed = set(allowed_filenames or set())
    if allowed:
        annotation_df = annotation_df[annotation_df["filename"].astype(str).isin(allowed)].copy()
        clip_df = clip_df[clip_df["filename"].astype(str).isin(allowed)].copy()
        assignments_df = assignments_df[assignments_df["filename"].astype(str).isin(allowed)].copy()
    assignment_lookup = {
        (str(row["source_dataset"]), str(row["filename"])): str(row["split_name"])
        for row in assignments_df.to_dict("records")
    }

    context_rows: List[Dict[str, Any]] = []

    # Fin positives.
    fin_rows = annotation_df[annotation_df["species_code"] == FIN_SPECIES_CODE].copy()
    for row in fin_rows.to_dict("records"):
        if allowed and str(row["filename"]) not in allowed:
            continue
        split_name = assignment_lookup.get((str(row["source_dataset"]), str(row["filename"])))
        if not split_name:
            continue
        start_s, end_s = _context_window_from_event(
            _safe_float(row["begin_time_s"]),
            _safe_float(row["end_time_s"]),
            float(context_duration_s),
        )
        context_rows.append(
            {
                "context_id": f"ctx_fin_{row['annotation_id']}",
                "split_name": split_name,
                "source_dataset": str(row["source_dataset"]),
                "filename": str(row["filename"]),
                "recording_day_utc": str(row["recording_day_utc"]),
                "context_type": "fin_positive",
                "label": 1,
                "context_start_s": float(start_s),
                "context_end_s": float(end_s),
                "context_duration_s": float(context_duration_s),
                "event_annotation_id": str(row["annotation_id"]),
                "event_species_code": str(row["species_code"]),
                "event_begin_s": float(row["begin_time_s"]),
                "event_end_s": float(row["end_time_s"]),
                "event_call_type_std": str(row["call_type_std"]),
                "context_image_relpath": "",
            }
        )

    # Annotated non-fin backgrounds.
    nonfin_rows = annotation_df[annotation_df["species_code"] != FIN_SPECIES_CODE].copy()
    for row in nonfin_rows.to_dict("records"):
        if allowed and str(row["filename"]) not in allowed:
            continue
        split_name = assignment_lookup.get((str(row["source_dataset"]), str(row["filename"])))
        if not split_name:
            continue
        start_s, end_s = _context_window_from_event(
            _safe_float(row["begin_time_s"]),
            _safe_float(row["end_time_s"]),
            float(context_duration_s),
        )
        context_rows.append(
            {
                "context_id": f"ctx_nonfin_{row['annotation_id']}",
                "split_name": split_name,
                "source_dataset": str(row["source_dataset"]),
                "filename": str(row["filename"]),
                "recording_day_utc": str(row["recording_day_utc"]),
                "context_type": "annotated_nonfin",
                "label": 0,
                "context_start_s": float(start_s),
                "context_end_s": float(end_s),
                "context_duration_s": float(context_duration_s),
                "event_annotation_id": str(row["annotation_id"]),
                "event_species_code": str(row["species_code"]),
                "event_begin_s": float(row["begin_time_s"]),
                "event_end_s": float(row["end_time_s"]),
                "event_call_type_std": str(row["call_type_std"]),
                "context_image_relpath": "",
            }
        )

    # Gap negatives from annotated clips.
    gap_df = enumerate_gap_negative_contexts(
        annotation_df,
        context_duration_s=float(context_duration_s),
        clip_duration_s=300.0,
        negative_margin_s=float(negative_margin_s),
    )
    source_lookup = {
        str(row["filename"]): str(row["source_dataset"])
        for row in clip_df.loc[clip_df["is_pure_negative_candidate"] == 0, ["filename", "source_dataset"]].to_dict("records")
    }
    day_lookup = {
        (str(row["source_dataset"]), str(row["filename"])): str(row["recording_day_utc"])
        for row in clip_df.to_dict("records")
    }
    for row in gap_df.to_dict("records"):
        filename = str(row["filename"])
        if allowed and filename not in allowed:
            continue
        source_dataset = source_lookup.get(filename, "")
        split_name = assignment_lookup.get((source_dataset, filename))
        if not split_name:
            continue
        context_rows.append(
            {
                "context_id": f"ctx_gap_{source_dataset}_{filename.replace('.', '_')}_{int(row['context_index']):03d}",
                "split_name": split_name,
                "source_dataset": source_dataset,
                "filename": filename,
                "recording_day_utc": day_lookup.get((source_dataset, filename), ""),
                "context_type": "gap_negative",
                "label": 0,
                "context_start_s": float(row["context_start_s"]),
                "context_end_s": float(row["context_end_s"]),
                "context_duration_s": float(row["context_duration_s"]),
                "event_annotation_id": "",
                "event_species_code": "",
                "event_begin_s": np.nan,
                "event_end_s": np.nan,
                "event_call_type_std": "",
                "context_image_relpath": "",
            }
        )

    # Pure-zero negatives from Mar26 inventory, sampled after building candidates.
    pure_rows = clip_df[clip_df["source_dataset"] == PURE_NEGATIVE_DATASET].copy()
    pure_candidates_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pure_rows.to_dict("records"):
        filename = str(row["filename"])
        if allowed and filename not in allowed:
            continue
        split_name = assignment_lookup.get((PURE_NEGATIVE_DATASET, filename))
        if not split_name:
            continue
        for idx, start_s in enumerate([0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 260.0]):
            pure_candidates_by_split[split_name].append(
                {
                    "context_id": f"ctx_zero_{filename.replace('.', '_')}_{idx:03d}",
                    "split_name": split_name,
                    "source_dataset": PURE_NEGATIVE_DATASET,
                    "filename": filename,
                    "recording_day_utc": str(row["recording_day_utc"]),
                    "context_type": "pure_zero_negative",
                    "label": 0,
                    "context_start_s": float(start_s),
                    "context_end_s": float(start_s + context_duration_s),
                    "context_duration_s": float(context_duration_s),
                    "event_annotation_id": "",
                    "event_species_code": "",
                    "event_begin_s": np.nan,
                    "event_end_s": np.nan,
                    "event_call_type_std": "",
                    "context_image_relpath": "",
                }
            )

    base_context_df = pd.DataFrame(context_rows, columns=CONTEXT_COLUMNS)
    fin_positive_counts = (
        base_context_df.loc[base_context_df["context_type"] == "fin_positive", "split_name"]
        .value_counts()
        .to_dict()
    )
    for split_name, candidates in pure_candidates_by_split.items():
        target = int(round(float(fin_positive_counts.get(split_name, 0)) * float(pure_zero_ratio)))
        if target <= 0:
            continue
        ordered = sorted(candidates, key=lambda item: (item["recording_day_utc"], item["filename"], item["context_start_s"]))
        by_clip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in ordered:
            by_clip[str(item["filename"])].append(item)
        clip_names = sorted(by_clip)
        chosen: List[Dict[str, Any]] = []
        progress = True
        while progress and len(chosen) < target:
            progress = False
            for clip_name in clip_names:
                if not by_clip[clip_name]:
                    continue
                chosen.append(by_clip[clip_name].pop(0))
                progress = True
                if len(chosen) >= target:
                    break
        context_rows.extend(chosen)

    context_df = pd.DataFrame(context_rows, columns=CONTEXT_COLUMNS)
    context_df = context_df.sort_values(
        ["split_name", "context_type", "filename", "context_start_s", "context_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    summary = {
        "context_count": int(len(context_df)),
        "context_type_counts": dict(Counter(context_df["context_type"].astype(str).tolist())) if not context_df.empty else {},
        "split_counts": dict(Counter(context_df["split_name"].astype(str).tolist())) if not context_df.empty else {},
    }
    return context_df, summary


def _fin_annotations_by_clip(annotation_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    fin_df = annotation_df[annotation_df["species_code"] == FIN_SPECIES_CODE].copy()
    for row in fin_df.to_dict("records"):
        grouped[str(row["filename"])].append(row)
    for filename in grouped:
        grouped[filename] = sorted(grouped[filename], key=lambda item: float(item["begin_time_s"]))
    return grouped


def project_fin_boxes_to_crop(
    *,
    fin_rows: Sequence[Dict[str, Any]],
    crop_start_s: float,
    crop_end_s: float,
    freq_min_hz: float,
    freq_max_hz: float,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    crop_duration_s = float(crop_end_s) - float(crop_start_s)
    freq_span = float(freq_max_hz) - float(freq_min_hz)
    if crop_duration_s <= 0 or freq_span <= 0:
        return boxes
    for row in fin_rows:
        box_start = max(float(row["begin_time_s"]), float(crop_start_s))
        box_end = min(float(row["end_time_s"]), float(crop_end_s))
        box_low = max(float(row["low_freq_hz"]), float(freq_min_hz))
        box_high = min(float(row["high_freq_hz"]), float(freq_max_hz))
        if box_end <= box_start or box_high <= box_low:
            continue
        x = (box_start - float(crop_start_s)) / crop_duration_s * float(image_width)
        y = (box_low - float(freq_min_hz)) / freq_span * float(image_height)
        w = (box_end - box_start) / crop_duration_s * float(image_width)
        h = (box_high - box_low) / freq_span * float(image_height)
        if w <= 0 or h <= 0:
            continue
        boxes.append(
            {
                "annotation_id": str(row["annotation_id"]),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
            }
        )
    return boxes


def build_crop_manifest_and_coco(
    annotation_df: pd.DataFrame,
    context_df: pd.DataFrame,
    *,
    freq_min_hz: float,
    freq_max_hz: float,
    image_size: int,
    train_crop_duration_s: float = 10.0,
    eval_crop_duration_s: float = 10.0,
    center_bias_sigma_frac: float = 0.25,
    seed: int = 1337,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    fin_rows_by_clip = _fin_annotations_by_clip(annotation_df)
    crop_rows: List[Dict[str, Any]] = []
    coco_by_split: Dict[str, Dict[str, Any]] = {}
    dropped_reasons: Counter[str] = Counter()
    next_image_id = 1
    next_annotation_id = 1

    for split_name in sorted(context_df["split_name"].astype(str).unique()):
        coco_by_split[split_name] = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "fin_call"}],
        }

    for row in context_df.to_dict("records"):
        split_name = str(row["split_name"])
        context_type = str(row["context_type"])
        filename = str(row["filename"])
        context_start = float(row["context_start_s"])
        context_duration = float(row["context_duration_s"])
        event_begin = _safe_float(row.get("event_begin_s"), default=np.nan)
        event_end = _safe_float(row.get("event_end_s"), default=np.nan)
        crop_duration = float(train_crop_duration_s if split_name == "train" else eval_crop_duration_s)

        if context_type in {"fin_positive", "annotated_nonfin"}:
            event_center = 0.5 * (event_begin + event_end)
            if split_name == "train":
                frac = _sample_call_fraction_in_window(rng, center_bias_sigma_frac)
            else:
                frac = 0.5
            desired_start = event_center - frac * crop_duration
            crop_start = _clip_interval_within_context(
                desired_start_s=desired_start,
                clip_duration_s=crop_duration,
                context_start_s=context_start,
                context_duration_s=context_duration,
            )
        else:
            if split_name == "train":
                max_offset = max(0.0, context_duration - crop_duration)
                offset = float(rng.uniform(0.0, max_offset)) if max_offset > 0 else 0.0
            else:
                offset = 0.5 * max(0.0, context_duration - crop_duration)
            crop_start = float(context_start + offset)

        crop_end = float(crop_start + crop_duration)
        fin_boxes = project_fin_boxes_to_crop(
            fin_rows=fin_rows_by_clip.get(filename, []),
            crop_start_s=crop_start,
            crop_end_s=crop_end,
            freq_min_hz=float(freq_min_hz),
            freq_max_hz=float(freq_max_hz),
            image_width=int(image_size),
            image_height=int(image_size),
        )

        if context_type == "fin_positive":
            if not fin_boxes:
                dropped_reasons["positive_crop_without_fin_box"] += 1
                continue
        else:
            if fin_boxes:
                dropped_reasons["background_crop_overlaps_fin_box"] += 1
                continue

        crop_id = f"crop_{row['context_id']}"
        image_relpath = f"{split_name}/images/{crop_id}.png"
        crop_row = {
            "crop_id": crop_id,
            "context_id": str(row["context_id"]),
            "split_name": split_name,
            "source_dataset": str(row["source_dataset"]),
            "filename": filename,
            "recording_day_utc": str(row["recording_day_utc"]),
            "context_type": context_type,
            "label": int(row["label"]),
            "crop_start_s": float(crop_start),
            "crop_end_s": float(crop_end),
            "crop_duration_s": float(crop_duration),
            "image_relpath": image_relpath,
            "image_width": int(image_size),
            "image_height": int(image_size),
            "fin_box_count": int(len(fin_boxes)),
        }
        crop_rows.append(crop_row)
        coco_by_split[split_name]["images"].append(
            {
                "id": next_image_id,
                "file_name": image_relpath,
                "width": int(image_size),
                "height": int(image_size),
            }
        )
        for box in fin_boxes:
            coco_by_split[split_name]["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": 1,
                    "bbox": [round(v, 4) for v in box["bbox"]],
                    "area": round(float(box["area"]), 4),
                    "iscrowd": 0,
                    "annotation_id": box["annotation_id"],
                }
            )
            next_annotation_id += 1
        next_image_id += 1

    crop_df = pd.DataFrame(crop_rows, columns=CROP_COLUMNS)
    summary = {
        "crop_count": int(len(crop_df)),
        "split_counts": dict(Counter(crop_df["split_name"].astype(str).tolist())) if not crop_df.empty else {},
        "context_type_counts": dict(Counter(crop_df["context_type"].astype(str).tolist())) if not crop_df.empty else {},
        "dropped_reasons": dict(dropped_reasons),
    }
    return crop_df, coco_by_split, summary


def _overlay_boxes(source_path: Path, dest_path: Path, boxes: Sequence[Sequence[float]]) -> None:
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x, y, w, h = [float(v) for v in box]
        draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=2)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path)


def export_bbox_dataset(
    *,
    annotation_manifest_csv: Path | str,
    clip_manifest_csv: Path | str,
    split_assignments_csv: Path | str,
    audio_dir: Path | str,
    output_dir: Path | str,
    config_path: Path | str,
    allowed_filenames: Optional[set[str]] = None,
    context_duration_s: float = 40.0,
    train_crop_duration_s: float = 10.0,
    eval_crop_duration_s: float = 10.0,
    freq_min_hz: float = 1.0,
    freq_max_hz: float = 200.0,
    edge_buffer_s: float = 2.0,
    image_size: int = 640,
    pure_zero_ratio: float = 0.5,
    negative_margin_s: float = 2.0,
    center_bias_sigma_frac: float = 0.25,
    seed: int = 1337,
    qc_limit: int = 0,
) -> Dict[str, Any]:
    audio_root = Path(audio_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_df = load_annotation_manifest(annotation_manifest_csv)
    clip_df = load_clip_manifest(clip_manifest_csv)
    assignments_df = load_split_assignments(split_assignments_csv)
    context_df, context_summary = build_context_manifest(
        annotation_df,
        clip_df,
        assignments_df,
        context_duration_s=float(context_duration_s),
        pure_zero_ratio=float(pure_zero_ratio),
        negative_margin_s=float(negative_margin_s),
        allowed_filenames=allowed_filenames,
    )
    crop_df, coco_by_split, crop_summary = build_crop_manifest_and_coco(
        annotation_df,
        context_df,
        freq_min_hz=float(freq_min_hz),
        freq_max_hz=float(freq_max_hz),
        image_size=int(image_size),
        train_crop_duration_s=float(train_crop_duration_s),
        eval_crop_duration_s=float(eval_crop_duration_s),
        center_bias_sigma_frac=float(center_bias_sigma_frac),
        seed=int(seed),
    )

    spec_params = _load_spec_params(config_path, float(freq_min_hz), float(freq_max_hz))
    spec_gen = _make_spectrogram_generator(spec_params)

    context_path_lookup: Dict[str, Path] = {}
    missing_context_audio: List[str] = []
    for row in context_df.to_dict("records"):
        filename = str(row["filename"])
        if allowed_filenames and filename not in allowed_filenames:
            continue
        png_relpath = Path("contexts") / row["split_name"] / "images" / f"{row['context_id']}.png"
        png_abspath = out_dir / png_relpath
        try:
            audio_data, sample_rate = _load_audio_window_with_buffer(
                audio_dir=audio_root,
                filename=filename,
                start_s=float(row["context_start_s"]),
                duration_s=float(row["context_duration_s"]),
                edge_buffer_s=float(edge_buffer_s),
            )
            _render_spectrogram_png(
                audio_data=audio_data,
                sample_rate=sample_rate,
                spec_gen=spec_gen,
                freq_min_hz=float(freq_min_hz),
                freq_max_hz=float(freq_max_hz),
                edge_buffer_s=float(edge_buffer_s),
                target_duration_s=float(row["context_duration_s"]),
                image_size=int(image_size),
                output_path=png_abspath,
            )
            context_path_lookup[str(row["context_id"])] = png_abspath
        except Exception:
            missing_context_audio.append(str(row["context_id"]))

    rendered_context_ids = set(context_path_lookup)
    if rendered_context_ids:
        context_df = context_df[context_df["context_id"].astype(str).isin(rendered_context_ids)].copy()
        context_df["context_image_relpath"] = context_df["context_id"].map(
            lambda context_id: str(context_path_lookup[str(context_id)].relative_to(out_dir))
        )
        context_df = context_df[CONTEXT_COLUMNS].copy()
    else:
        context_df = pd.DataFrame(columns=CONTEXT_COLUMNS)

    rendered_crops = 0
    skipped_crops = 0
    qc_written = 0
    rendered_crop_ids: set[str] = set()
    for row in crop_df.to_dict("records"):
        context_id = str(row["context_id"])
        context_row = context_df.loc[context_df["context_id"] == context_id]
        if context_row.empty:
            skipped_crops += 1
            continue
        context_item = context_row.iloc[0].to_dict()
        filename = str(row["filename"])
        image_relpath = Path(str(row["image_relpath"]))
        image_abspath = out_dir / image_relpath
        try:
            audio_data, sample_rate = _load_audio_window_with_buffer(
                audio_dir=audio_root,
                filename=filename,
                start_s=float(row["crop_start_s"]),
                duration_s=float(row["crop_duration_s"]),
                edge_buffer_s=float(edge_buffer_s),
            )
            _render_spectrogram_png(
                audio_data=audio_data,
                sample_rate=sample_rate,
                spec_gen=spec_gen,
                freq_min_hz=float(freq_min_hz),
                freq_max_hz=float(freq_max_hz),
                edge_buffer_s=float(edge_buffer_s),
                target_duration_s=float(row["crop_duration_s"]),
                image_size=int(image_size),
                output_path=image_abspath,
            )
            rendered_crops += 1
            rendered_crop_ids.add(str(row["crop_id"]))

            if qc_limit > 0 and qc_written < qc_limit:
                split_name = str(row["split_name"])
                image_key = str(row["image_relpath"])
                image_id = next(
                    (
                        int(image_row["id"])
                        for image_row in coco_by_split.get(split_name, {}).get("images", [])
                        if str(image_row["file_name"]) == image_key
                    ),
                    None,
                )
                boxes = [
                    ann["bbox"]
                    for ann in coco_by_split.get(split_name, {}).get("annotations", [])
                    if image_id is not None and int(ann["image_id"]) == int(image_id)
                ]
                if boxes or context_item["context_type"] != "fin_positive":
                    qc_path = out_dir / "qc" / split_name / f"{row['crop_id']}.png"
                    _overlay_boxes(image_abspath, qc_path, boxes)
                    qc_written += 1
        except Exception:
            skipped_crops += 1
            continue

    if rendered_crop_ids:
        crop_df = crop_df[crop_df["crop_id"].astype(str).isin(rendered_crop_ids)].copy()
        crop_df = crop_df[CROP_COLUMNS].copy()
    else:
        crop_df = pd.DataFrame(columns=CROP_COLUMNS)

    rendered_image_relpaths = set(crop_df["image_relpath"].astype(str).tolist())
    for split_name, coco in coco_by_split.items():
        kept_images = [
            image_row for image_row in coco["images"] if str(image_row["file_name"]) in rendered_image_relpaths
        ]
        kept_image_ids = {int(image_row["id"]) for image_row in kept_images}
        kept_annotations = [
            ann for ann in coco["annotations"] if int(ann["image_id"]) in kept_image_ids
        ]
        coco["images"] = kept_images
        coco["annotations"] = kept_annotations

    context_manifest_path = out_dir / "context_manifest.csv"
    crop_manifest_path = out_dir / "crop_manifest.csv"
    context_df.to_csv(context_manifest_path, index=False)
    crop_df.to_csv(crop_manifest_path, index=False)

    coco_paths: Dict[str, str] = {}
    for split_name, coco in coco_by_split.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        coco_path = split_dir / "annotations.coco.json"
        with open(coco_path, "w", encoding="utf-8") as handle:
            json.dump(coco, handle, indent=2)
        coco_paths[split_name] = str(coco_path)

    summary = {
        "annotation_manifest_csv": str(Path(annotation_manifest_csv)),
        "clip_manifest_csv": str(Path(clip_manifest_csv)),
        "split_assignments_csv": str(Path(split_assignments_csv)),
        "audio_dir": str(audio_root),
        "context_summary": context_summary,
        "crop_summary": crop_summary,
        "context_rendered_count": int(len(context_path_lookup)),
        "context_missing_audio_count": int(len(missing_context_audio)),
        "crop_rendered_count": int(rendered_crops),
        "crop_skipped_count": int(skipped_crops),
        "coco_paths": coco_paths,
        "export_params": {
            "context_duration_s": float(context_duration_s),
            "train_crop_duration_s": float(train_crop_duration_s),
            "eval_crop_duration_s": float(eval_crop_duration_s),
            "freq_min_hz": float(freq_min_hz),
            "freq_max_hz": float(freq_max_hz),
            "edge_buffer_s": float(edge_buffer_s),
            "image_size": int(image_size),
            "pure_zero_ratio": float(pure_zero_ratio),
            "negative_margin_s": float(negative_margin_s),
            "center_bias_sigma_frac": float(center_bias_sigma_frac),
            "seed": int(seed),
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return {
        "context_manifest": context_manifest_path,
        "crop_manifest": crop_manifest_path,
        "summary": summary_path,
        "coco_paths": coco_paths,
    }
