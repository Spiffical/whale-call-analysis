#!/usr/bin/env python3
"""Run E24-style multiband expert ensemble inference and cache high-confidence hits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.train_multiband_multilabel import build_label_band_mask, collate_batch  # noqa: E402
from scripts.data.multilabel.prepare_multiband_context_windows import (  # noqa: E402
    DEFAULT_BANDS,
    _find_adjacent_file_with_index,
    _find_audio_file_with_index,
    _read_audio,
    build_audio_index,
    compute_band_spectrogram,
    parse_clip_ts,
)
from src.dataset.multiband import MultiBandMatDataset, parse_band_crop_shapes  # noqa: E402
from src.dataset.multilabel import LabelVocabulary, label_metadata, read_csv_rows  # noqa: E402
from src.models.multiband import create_multiband_model  # noqa: E402
from src.dataset.multiband import _crop_freq, _crop_time  # noqa: E402
from src.training.mat_dataset import _normalize_db_to_unit  # noqa: E402


LABEL_NAMES = {
    "species:Bp": "fin whale",
    "species:Bm": "blue whale",
    "species:Mn": "humpback whale",
    "species:Oo": "killer whale",
}


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def split_csv(value: Any) -> List[str]:
    return [part.strip() for part in clean(value).replace("|", ",").split(",") if part.strip()]


def safe_name(value: str, max_len: int = 180) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:max_len] or "item"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    if fieldnames is None:
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def threshold_for_label(run_dir: Path, label_id: str) -> float:
    payload = read_json(run_dir / "train" / "onc_calibrated_eval" / "onc_calibrated_metrics_summary.json")
    for key in ("onc_validation_thresholds", "global_validation_thresholds"):
        block = payload.get(key, {})
        if isinstance(block, Mapping) and isinstance(block.get(label_id), Mapping):
            try:
                return float(block[label_id].get("threshold"))
            except (TypeError, ValueError):
                pass
    return 0.5


def vocabulary_from_checkpoint(checkpoint: Mapping[str, Any]) -> LabelVocabulary:
    payload = checkpoint.get("label_vocabulary")
    if isinstance(payload, Mapping):
        labels = payload.get("labels", [])
        if labels:
            return LabelVocabulary(labels=tuple(dict(label) for label in labels))
    label_ids = checkpoint.get("label_ids") or checkpoint.get("labels") or []
    if isinstance(label_ids, str):
        label_ids = split_csv(label_ids)
    labels = [label_metadata(str(label_id)) for label_id in label_ids]
    return LabelVocabulary(labels=tuple(labels))


@dataclass
class ExpertSpec:
    label_id: str
    run_dir: Path
    checkpoint_path: Path
    threshold: float
    vocab: LabelVocabulary
    bands: List[str]
    encoder: str
    fusion: str
    head_type: str
    dropout: float
    band_crop_shapes: str
    band_availability_mode: str
    class_band_mask_mode: str
    crop_time_seconds: float
    context_seconds: float
    model: torch.nn.Module


def load_expert(raw_spec: str, device: torch.device) -> ExpertSpec:
    if "=" not in raw_spec:
        raise ValueError(f"Expert spec must look like label_id=run_dir, got {raw_spec!r}")
    label_id, run_text = raw_spec.split("=", 1)
    label_id = clean(label_id)
    run_dir = Path(run_text).resolve()
    checkpoint_path = run_dir / "train" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    metadata = read_json(run_dir / "run_metadata.json")
    run_summary = read_json(run_dir / "train" / "run_summary.json")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {checkpoint_path}")
    vocab = vocabulary_from_checkpoint(checkpoint)
    if label_id not in vocab.label_ids:
        raise ValueError(f"{label_id} is not in checkpoint vocabulary {vocab.label_ids} for {run_dir}")
    training_args = checkpoint.get("training_args", {}) if isinstance(checkpoint.get("training_args"), Mapping) else {}

    bands = split_csv(metadata.get("bands") or checkpoint.get("bands") or run_summary.get("bands") or training_args.get("bands"))
    if not bands:
        bands = ["low", "mid", "high"]
    encoder = clean(metadata.get("encoder") or run_summary.get("encoder") or training_args.get("encoder") or "resnet18")
    fusion = clean(metadata.get("fusion") or run_summary.get("fusion") or training_args.get("fusion") or "gated")
    head_type = clean(metadata.get("head_type") or run_summary.get("head_type") or training_args.get("head_type") or "shared")
    dropout = float(metadata.get("dropout") or training_args.get("dropout") or 0.3)
    band_crop_shapes = clean(metadata.get("band_crop_shapes") or run_summary.get("band_crop_shapes") or checkpoint.get("band_crop_shapes") or "")
    band_availability_mode = clean(
        metadata.get("band_availability_mode")
        or run_summary.get("band_availability_mode")
        or checkpoint.get("band_availability_mode")
        or training_args.get("band_availability_mode")
        or "all"
    )
    class_band_mask_mode = clean(
        metadata.get("class_band_mask_mode")
        or run_summary.get("class_band_mask_mode")
        or checkpoint.get("class_band_mask_mode")
        or training_args.get("class_band_mask_mode")
        or "none"
    )
    crop_time_seconds = float(
        metadata.get("crop_time_seconds")
        or training_args.get("crop_time_seconds")
        or run_summary.get("crop_time_seconds")
        or 10.0
    )
    context_seconds = float(training_args.get("context_seconds") or 40.0)
    model = create_multiband_model(
        encoder=encoder,
        num_classes=vocab.size,
        bands=bands,
        fusion=fusion,
        head_type=head_type,
        dropout=dropout,
        in_ch=1,
        label_band_mask=build_label_band_mask(
            label_ids=vocab.label_ids,
            bands=bands,
            mode=class_band_mask_mode,
        ),
    )
    state = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint state is not a mapping: {checkpoint_path}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return ExpertSpec(
        label_id=label_id,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        threshold=threshold_for_label(run_dir, label_id),
        vocab=vocab,
        bands=bands,
        encoder=encoder,
        fusion=fusion,
        head_type=head_type,
        dropout=dropout,
        band_crop_shapes=band_crop_shapes,
        band_availability_mode=band_availability_mode,
        class_band_mask_mode=class_band_mask_mode,
        crop_time_seconds=crop_time_seconds,
        context_seconds=context_seconds,
        model=model,
    )


def infer_expert(
    *,
    spec: ExpertSpec,
    manifest_csv: Path,
    dataset_root: Optional[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataset = MultiBandMatDataset(
        manifest_csv,
        spec.vocab,
        split=None,
        dataset_root=dataset_root,
        bands=spec.bands,
        band_crop_shapes=parse_band_crop_shapes(spec.band_crop_shapes),
        crop_time_seconds=spec.crop_time_seconds,
        context_seconds=spec.context_seconds,
        positive_crop_mode="centered_gaussian",
        band_availability_mode=spec.band_availability_mode,
        seed=0,
        return_meta=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_batch,
        pin_memory=str(device).startswith("cuda"),
    )
    label_idx = spec.vocab.index()[spec.label_id]
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for x, _y, _meta in loader:
            x = {band: tensor.to(device, non_blocking=True) for band, tensor in x.items()}
            logits = spec.model(x)
            scores = torch.sigmoid(logits)[:, label_idx].detach().cpu().numpy()
            chunks.append(scores.astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)


def add_scores(rows: List[Dict[str, str]], specs: Sequence[ExpertSpec], scores_by_label: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    thresholds = {spec.label_id: float(spec.threshold) for spec in specs}
    for idx, row in enumerate(rows):
        out = dict(row)
        pred_labels: List[str] = []
        max_label = ""
        max_score = -math.inf
        for spec in specs:
            score = float(scores_by_label[spec.label_id][idx])
            out[f"score__{spec.label_id}"] = f"{score:.8f}"
            out[f"threshold__{spec.label_id}"] = f"{thresholds[spec.label_id]:.6f}"
            out[f"pred__{spec.label_id}"] = "1" if score >= thresholds[spec.label_id] else "0"
            if score >= thresholds[spec.label_id]:
                pred_labels.append(spec.label_id)
            if score > max_score:
                max_score = score
                max_label = spec.label_id
        out["pred_label_ids"] = "|".join(pred_labels)
        out["max_label_id"] = max_label
        out["max_label_name"] = LABEL_NAMES.get(max_label, max_label)
        out["max_score"] = f"{max_score:.8f}"
        out_rows.append(out)
    return out_rows


def ffloat(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def clip_timestamp(source_audio: str) -> Optional[datetime]:
    match = re.search(r"_(\d{8}T\d{6})", str(source_audio))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


@dataclass
class StitchedAudio:
    source_audio: str
    audio: np.ndarray
    sample_rate_hz: int
    target_offset_s: float
    context_seconds: float
    current_path: Path
    previous_path: Optional[Path]
    next_path: Optional[Path]

    def row_metadata(self) -> Dict[str, str]:
        return {
            "stitch_context_seconds": f"{self.context_seconds:.6f}",
            "stitch_target_offset_s": f"{self.target_offset_s:.6f}",
            "stitch_current_audio_path": str(self.current_path),
            "stitch_previous_audio_path": str(self.previous_path or ""),
            "stitch_next_audio_path": str(self.next_path or ""),
            "stitch_has_previous": "1" if self.previous_path is not None else "0",
            "stitch_has_next": "1" if self.next_path is not None else "0",
        }


def load_stitched_audio(
    *,
    raw_audio_dir: Path,
    source_audio: str,
    clip_seconds: float,
    audio_index: Mapping[str, Path],
    audio_index_by_second: Mapping[str, Path],
) -> StitchedAudio:
    """Load previous/current/next 5-minute files for edge-safe spectrograms."""

    current_path = _find_audio_file_with_index(raw_audio_dir, source_audio, audio_index, audio_index_by_second)
    if current_path is None:
        raise FileNotFoundError(f"Audio file not found for {source_audio} under {raw_audio_dir}")
    current_audio, sample_rate_hz = _read_audio(current_path)
    current_audio = np.asarray(current_audio, dtype=np.float32)

    clip_dt = parse_clip_ts(source_audio)
    device = Path(source_audio).name.split("_")[0]
    previous_path: Optional[Path] = None
    next_path: Optional[Path] = None
    chunks: List[np.ndarray] = []
    target_offset_s = 0.0

    if clip_dt is not None and device:
        candidate = _find_adjacent_file_with_index(
            raw_audio_dir,
            device,
            clip_dt - timedelta(seconds=float(clip_seconds)),
            audio_index_by_second,
        )
        if candidate is not None and candidate.exists():
            previous_audio, previous_sr = _read_audio(candidate)
            if int(previous_sr) == int(sample_rate_hz):
                previous_audio = np.asarray(previous_audio, dtype=np.float32)
                chunks.append(previous_audio)
                target_offset_s = len(previous_audio) / float(sample_rate_hz)
                previous_path = candidate

    chunks.append(current_audio)

    if clip_dt is not None and device:
        candidate = _find_adjacent_file_with_index(
            raw_audio_dir,
            device,
            clip_dt + timedelta(seconds=float(clip_seconds)),
            audio_index_by_second,
        )
        if candidate is not None and candidate.exists():
            next_audio, next_sr = _read_audio(candidate)
            if int(next_sr) == int(sample_rate_hz):
                chunks.append(np.asarray(next_audio, dtype=np.float32))
                next_path = candidate

    stitched = np.concatenate(chunks).astype(np.float32, copy=False)
    return StitchedAudio(
        source_audio=source_audio,
        audio=stitched,
        sample_rate_hz=int(sample_rate_hz),
        target_offset_s=float(target_offset_s),
        context_seconds=len(stitched) / float(sample_rate_hz),
        current_path=current_path,
        previous_path=previous_path,
        next_path=next_path,
    )


def build_audio_rows(
    *,
    audio_list: Path,
    crop_seconds: float,
    step_seconds: float,
    clip_seconds: float,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    names = [line.strip() for line in audio_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    first_center = crop_seconds / 2.0
    last_center = clip_seconds - (crop_seconds / 2.0)
    centers: List[float] = []
    cur = first_center
    while cur <= last_center + 1e-6:
        centers.append(round(cur, 6))
        cur += step_seconds
    for name in names:
        stem = Path(name).stem
        base_ts = clip_timestamp(name)
        for center_s in centers:
            begin_s = max(0.0, center_s - (crop_seconds / 2.0))
            end_s = min(clip_seconds, center_s + (crop_seconds / 2.0))
            row = {
                "item_id": safe_name(f"{stem}__center_{center_s:07.2f}s"),
                "clip": name,
                "filename": name,
                "source_audio": name,
                "raw_audio_path": f"raw_audio/{name}",
                "begin_s": f"{begin_s:.6f}",
                "end_s": f"{end_s:.6f}",
                "window_center_s": f"{center_s:.6f}",
                "window_step_s": f"{step_seconds:.6f}",
                "crop_seconds": f"{crop_seconds:.6f}",
                "clip_seconds": f"{clip_seconds:.6f}",
                "source_kind": "ONC",
                "source_dataset": "ONC_Clayoquot_unreviewed_month",
                "split": "inference",
                "label_ids": "",
                "target_label_ids": "",
                "canonical_label_ids": "",
                "source_label_ids": "",
                "analysis_label_ids": "",
                "is_background": "",
                "review_status": "unreviewed",
                "negative_bucket": "",
                "context_tags": "deployment_unreviewed_sliding_window",
                "event_group": "",
            }
            if base_ts is not None:
                row["absolute_begin_time"] = (base_ts + timedelta(seconds=begin_s)).isoformat()
                row["absolute_end_time"] = (base_ts + timedelta(seconds=end_s)).isoformat()
            rows.append(row)
    return rows


def _band_config_map() -> Dict[str, Any]:
    return {band.name: band for band in DEFAULT_BANDS}


def compute_audio_band_payloads(
    *,
    audio: np.ndarray,
    source_sr: int,
    bands: Iterable[str],
    context_seconds: float,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    configs = _band_config_map()
    for band_name in bands:
        band = configs[band_name]
        freqs, times, db, info = compute_band_spectrogram(
            audio,
            int(source_sr),
            band,
            context_seconds=float(context_seconds),
        )
        out[band_name] = {
            "freqs": freqs,
            "times": times,
            "db": db,
            "info": info,
        }
    return out


def crop_audio_band_tensor(
    *,
    payload: Mapping[str, Any],
    crop_start_s: float,
    shape: Sequence[int],
) -> torch.Tensor:
    spec = np.asarray(payload["db"], dtype=np.float32)
    spec = _crop_freq(spec, int(shape[0]))
    spec, _ = _crop_time(
        spec,
        times=np.asarray(payload["times"], dtype=np.float32),
        crop_start_s=float(crop_start_s),
        target_t=int(shape[1]),
    )
    spec = _normalize_db_to_unit(spec, -80.0, 0.0)
    return torch.from_numpy(spec).unsqueeze(0).float()


def infer_audio_rows(
    *,
    rows: List[Dict[str, str]],
    raw_audio_dir: Path,
    specs: Sequence[ExpertSpec],
    batch_size: int,
    device: torch.device,
    clip_seconds: float,
) -> List[Dict[str, Any]]:
    needed_bands = sorted({band for spec in specs for band in spec.bands})
    scores_by_item: Dict[str, Dict[str, float]] = {row["item_id"]: {} for row in rows}
    by_audio: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_audio.setdefault(clean(row.get("source_audio")), []).append(row)
    audio_index, audio_index_by_second = build_audio_index(raw_audio_dir)
    stitch_meta_by_audio: Dict[str, Dict[str, str]] = {}

    for audio_idx, (source_audio, audio_rows) in enumerate(sorted(by_audio.items()), start=1):
        stitched = load_stitched_audio(
            raw_audio_dir=raw_audio_dir,
            source_audio=source_audio,
            clip_seconds=float(clip_seconds),
            audio_index=audio_index,
            audio_index_by_second=audio_index_by_second,
        )
        stitch_meta_by_audio[source_audio] = stitched.row_metadata()
        payloads = compute_audio_band_payloads(
            audio=stitched.audio,
            source_sr=stitched.sample_rate_hz,
            bands=needed_bands,
            context_seconds=stitched.context_seconds,
        )
        for spec in specs:
            shapes = parse_band_crop_shapes(spec.band_crop_shapes)
            label_idx = spec.vocab.index()[spec.label_id]
            tensors_by_band: Dict[str, List[torch.Tensor]] = {band: [] for band in spec.bands}
            item_ids: List[str] = []
            for row in audio_rows:
                center_s = ffloat(row, "window_center_s", (ffloat(row, "begin_s") + ffloat(row, "end_s")) / 2.0)
                crop_start_s = stitched.target_offset_s + center_s - spec.crop_time_seconds / 2.0
                crop_start_s = max(0.0, min(stitched.context_seconds - spec.crop_time_seconds, crop_start_s))
                for band in spec.bands:
                    tensors_by_band[band].append(
                        crop_audio_band_tensor(
                            payload=payloads[band],
                            crop_start_s=crop_start_s,
                            shape=shapes[band],
                        )
                    )
                item_ids.append(row["item_id"])
            with torch.no_grad():
                for start in range(0, len(item_ids), int(batch_size)):
                    batch_ids = item_ids[start : start + int(batch_size)]
                    inputs = {
                        band: torch.stack(tensors_by_band[band][start : start + int(batch_size)], dim=0).to(device)
                        for band in spec.bands
                    }
                    logits = spec.model(inputs)
                    scores = torch.sigmoid(logits)[:, label_idx].detach().cpu().numpy()
                    for item_id, score in zip(batch_ids, scores):
                        scores_by_item[item_id][spec.label_id] = float(score)
        if audio_idx % 100 == 0:
            print(f"Scored {audio_idx}/{len(by_audio)} audio clips", flush=True)

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out.update(stitch_meta_by_audio.get(clean(row.get("source_audio")), {}))
        pred_labels: List[str] = []
        max_label = ""
        max_score = -math.inf
        item_scores = scores_by_item[row["item_id"]]
        for spec in specs:
            score = float(item_scores.get(spec.label_id, 0.0))
            out[f"score__{spec.label_id}"] = f"{score:.8f}"
            out[f"threshold__{spec.label_id}"] = f"{spec.threshold:.6f}"
            out[f"pred__{spec.label_id}"] = "1" if score >= spec.threshold else "0"
            if score >= spec.threshold:
                pred_labels.append(spec.label_id)
            if score > max_score:
                max_score = score
                max_label = spec.label_id
        out["pred_label_ids"] = "|".join(pred_labels)
        out["max_label_id"] = max_label
        out["max_label_name"] = LABEL_NAMES.get(max_label, max_label)
        out["max_score"] = f"{max_score:.8f}"
        out_rows.append(out)
    return out_rows


def cluster_high_confidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_ids: Sequence[str],
    low_threshold: float,
    high_threshold: float,
    min_members: int,
    max_gap_seconds: float,
    max_events_per_label: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    event_windows: List[Dict[str, Any]] = []
    for label_id in label_ids:
        score_key = f"score__{label_id}"
        by_audio: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            score = ffloat(row, score_key, -math.inf)
            if score < float(low_threshold):
                continue
            by_audio.setdefault(clean(row.get("source_audio")) or clean(row.get("clip")), []).append(row)

        label_events: List[Dict[str, Any]] = []
        for source_audio, audio_rows in by_audio.items():
            ordered = sorted(audio_rows, key=lambda row: (ffloat(row, "begin_s"), ffloat(row, "end_s")))
            clusters: List[List[Mapping[str, Any]]] = []
            current: List[Mapping[str, Any]] = []
            current_end = -math.inf
            for row in ordered:
                begin = ffloat(row, "begin_s")
                end = ffloat(row, "end_s")
                if current and begin - current_end > float(max_gap_seconds):
                    clusters.append(current)
                    current = []
                current.append(row)
                current_end = max(current_end, end)
            if current:
                clusters.append(current)

            for cluster_idx, cluster in enumerate(clusters, start=1):
                scored = [(ffloat(row, score_key, 0.0), row) for row in cluster]
                best_score, best_row = max(scored, key=lambda item: item[0])
                if len(cluster) < int(min_members) or best_score < float(high_threshold):
                    continue
                event_id = safe_name(
                    f"{label_id.replace(':', '_')}_{Path(source_audio).stem}_{cluster_idx:04d}"
                )
                event = {
                    "event_id": event_id,
                    "label_id": label_id,
                    "label_name": LABEL_NAMES.get(label_id, label_id),
                    "source_audio": source_audio,
                    "event_begin_s": f"{min(ffloat(row, 'begin_s') for row in cluster):.6f}",
                    "event_end_s": f"{max(ffloat(row, 'end_s') for row in cluster):.6f}",
                    "member_count": len(cluster),
                    "max_score": f"{best_score:.8f}",
                    "mean_score": f"{float(np.mean([score for score, _ in scored])):.8f}",
                    "best_item_id": clean(best_row.get("item_id")),
                    "best_begin_s": clean(best_row.get("begin_s")),
                    "best_end_s": clean(best_row.get("end_s")),
                    "best_mat_path": clean(best_row.get("mat_path")),
                    "absolute_begin_time": clean(best_row.get("absolute_begin_time")),
                    "absolute_end_time": clean(best_row.get("absolute_end_time")),
                }
                label_events.append(event)
                for score, row in scored:
                    out = dict(row)
                    out["event_id"] = event_id
                    out["event_label_id"] = label_id
                    out["event_score"] = f"{score:.8f}"
                    event_windows.append(out)
        label_events = sorted(label_events, key=lambda row: float(row["max_score"]), reverse=True)
        if max_events_per_label > 0:
            label_events = label_events[: int(max_events_per_label)]
            keep = {row["event_id"] for row in label_events}
            event_windows = [row for row in event_windows if row.get("event_id") in keep]
        events.extend(label_events)
    events = sorted(events, key=lambda row: float(row["max_score"]), reverse=True)
    return events, event_windows


def materialize_cache(
    *,
    events: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    raw_audio_dir: Optional[Path],
    copy_raw_audio: bool,
) -> List[Dict[str, Any]]:
    cache_dir = output_dir / "high_confidence_cache"
    mat_dir = cache_dir / "mat_files"
    audio_dir = cache_dir / "raw_audio"
    mat_dir.mkdir(parents=True, exist_ok=True)
    if copy_raw_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    by_item = {clean(row.get("item_id")): row for row in windows}
    manifest: List[Dict[str, Any]] = []
    copied_audio: set[str] = set()
    for event in events:
        best_item = clean(event.get("best_item_id"))
        row = by_item.get(best_item, {})
        cached_mat = ""
        src_mat_text = clean(event.get("best_mat_path") or row.get("mat_path"))
        if src_mat_text:
            src_mat = Path(src_mat_text)
            if src_mat.exists() and src_mat.is_file():
                dst = mat_dir / f"{clean(event.get('event_id'))}__{safe_name(src_mat.name)}"
                shutil.copy2(src_mat, dst)
                cached_mat = str(dst)
        raw_path = ""
        if raw_audio_dir is not None:
            source_audio = clean(event.get("source_audio"))
            candidate = raw_audio_dir / source_audio
            if candidate.exists():
                raw_path = str(candidate)
                if copy_raw_audio and source_audio not in copied_audio:
                    shutil.copy2(candidate, audio_dir / source_audio)
                    copied_audio.add(source_audio)
        out = dict(event)
        out["cached_mat_path"] = cached_mat
        out["raw_audio_path"] = raw_path
        manifest.append(out)

    write_csv(cache_dir / "cache_manifest.csv", manifest)
    tar_path = output_dir / "high_confidence_cache.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(cache_dir, arcname=cache_dir.name)
    return manifest


def add_audio_crop_cache(
    *,
    cache_manifest: List[Dict[str, Any]],
    raw_audio_dir: Path,
    specs: Sequence[ExpertSpec],
    output_dir: Path,
    clip_seconds: float,
) -> List[Dict[str, Any]]:
    if not cache_manifest:
        return cache_manifest
    cache_dir = output_dir / "high_confidence_cache"
    crop_dir = cache_dir / "npz_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    needed_bands = sorted({band for spec in specs for band in spec.bands})
    shape_by_band: Dict[str, Sequence[int]] = {}
    crop_seconds = 10.0
    for spec in specs:
        shapes = parse_band_crop_shapes(spec.band_crop_shapes)
        crop_seconds = float(spec.crop_time_seconds)
        for band in spec.bands:
            shape_by_band.setdefault(band, shapes[band])

    audio_index, audio_index_by_second = build_audio_index(raw_audio_dir)
    payload_cache: Dict[str, Tuple[StitchedAudio, Dict[str, Dict[str, Any]]]] = {}
    for row in cache_manifest:
        source_audio = clean(row.get("source_audio"))
        if not source_audio:
            continue
        if source_audio not in payload_cache:
            stitched = load_stitched_audio(
                raw_audio_dir=raw_audio_dir,
                source_audio=source_audio,
                clip_seconds=float(clip_seconds),
                audio_index=audio_index,
                audio_index_by_second=audio_index_by_second,
            )
            payload_cache[source_audio] = (
                stitched,
                compute_audio_band_payloads(
                    audio=stitched.audio,
                    source_sr=stitched.sample_rate_hz,
                    bands=needed_bands,
                    context_seconds=stitched.context_seconds,
                ),
            )
        stitched, payloads = payload_cache[source_audio]
        center_s = ffloat(row, "best_begin_s") + ((ffloat(row, "best_end_s") - ffloat(row, "best_begin_s")) / 2.0)
        crop_start_s = stitched.target_offset_s + center_s - crop_seconds / 2.0
        crop_start_s = max(0.0, min(stitched.context_seconds - crop_seconds, crop_start_s))
        arrays: Dict[str, np.ndarray] = {}
        for band in needed_bands:
            tensor = crop_audio_band_tensor(
                payload=payloads[band],
                crop_start_s=crop_start_s,
                shape=shape_by_band[band],
            )
            arrays[band] = tensor.squeeze(0).numpy().astype(np.float32)
        npz_path = crop_dir / f"{clean(row.get('event_id'))}.npz"
        np.savez_compressed(
            npz_path,
            **arrays,
            event_id=clean(row.get("event_id")),
            label_id=clean(row.get("label_id")),
            source_audio=source_audio,
            best_begin_s=clean(row.get("best_begin_s")),
            best_end_s=clean(row.get("best_end_s")),
            max_score=clean(row.get("max_score")),
            stitch_context_seconds=f"{stitched.context_seconds:.6f}",
            stitch_target_offset_s=f"{stitched.target_offset_s:.6f}",
            stitch_previous_audio_path=str(stitched.previous_path or ""),
            stitch_next_audio_path=str(stitched.next_path or ""),
        )
        row["cached_npz_path"] = str(npz_path)
        row.update(stitched.row_metadata())
    write_csv(cache_dir / "cache_manifest.csv", cache_manifest)
    tar_path = output_dir / "high_confidence_cache.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(cache_dir, arcname=cache_dir.name)
    return cache_manifest


def write_summary(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    specs: Sequence[ExpertSpec],
    args: argparse.Namespace,
    cache_manifest: Sequence[Mapping[str, Any]],
) -> None:
    by_label: Dict[str, int] = {spec.label_id: 0 for spec in specs}
    for event in events:
        by_label[clean(event.get("label_id"))] = by_label.get(clean(event.get("label_id")), 0) + 1
    audio_rows = {clean(row.get("source_audio")): row for row in rows if clean(row.get("source_audio"))}
    missing_previous = sum(1 for row in audio_rows.values() if clean(row.get("stitch_has_previous")) == "0")
    missing_next = sum(1 for row in audio_rows.values() if clean(row.get("stitch_has_next")) == "0")
    summary = {
        "manifest_csv": str(args.manifest_csv) if args.manifest_csv else "",
        "audio_list": str(args.audio_list) if getattr(args, "audio_list", None) else "",
        "window_rows": len(rows),
        "audio_clip_count": len(audio_rows),
        "audio_clips_without_previous_context": missing_previous,
        "audio_clips_without_next_context": missing_next,
        "event_count": len(events),
        "events_by_label": by_label,
        "low_threshold": float(args.low_threshold),
        "high_threshold": float(args.high_threshold),
        "min_members": int(args.min_members),
        "max_gap_seconds": float(args.max_gap_seconds),
        "cache_rows": len(cache_manifest),
        "experts": [
            {
                "label_id": spec.label_id,
                "label_name": LABEL_NAMES.get(spec.label_id, spec.label_id),
                "run_dir": str(spec.run_dir),
                "checkpoint": str(spec.checkpoint_path),
                "threshold": spec.threshold,
                "bands": spec.bands,
                "encoder": spec.encoder,
                "crop_time_seconds": spec.crop_time_seconds,
            }
            for spec in specs
        ],
        "outputs": {
            "window_predictions_csv": str(output_dir / "window_predictions.csv"),
            "high_confidence_events_csv": str(output_dir / "high_confidence_events.csv"),
            "high_confidence_windows_csv": str(output_dir / "high_confidence_windows.csv"),
            "cache_manifest_csv": str(output_dir / "high_confidence_cache" / "cache_manifest.csv"),
            "cache_tar": str(output_dir / "high_confidence_cache.tar"),
        },
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Multispecies High-Confidence Prediction Cache",
        "",
        f"- input windows: `{len(rows)}`",
        f"- kept high-confidence events: `{len(events)}`",
        f"- low threshold: `{float(args.low_threshold):.3f}`",
        f"- high threshold: `{float(args.high_threshold):.3f}`",
        f"- min members: `{int(args.min_members)}`",
        "",
        "## Events By Label",
        "",
    ]
    for label_id, count in by_label.items():
        lines.append(f"- {LABEL_NAMES.get(label_id, label_id)} (`{label_id}`): `{count}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    (output_dir / "prediction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--audio-list", type=Path, default=None, help="Text file of audio filenames for direct audio streaming mode")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Raw audio directory for direct audio streaming mode")
    parser.add_argument("--expert", action="append", required=True, help="label_id=run_dir; repeat for each expert")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-audio-dir", type=Path, default=None)
    parser.add_argument("--clip-seconds", type=float, default=300.0)
    parser.add_argument("--crop-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-threshold", type=float, default=0.70)
    parser.add_argument("--high-threshold", type=float, default=0.90)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--max-gap-seconds", type=float, default=15.0)
    parser.add_argument("--max-events-per-label", type=int, default=0)
    parser.add_argument("--copy-raw-audio", action="store_true")
    parser.add_argument("--cache-audio-crops", action="store_true", help="Store compressed low/mid crop arrays for selected events")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    specs = [load_expert(raw, device) for raw in args.expert]
    if args.audio_list is not None:
        if args.audio_dir is None:
            raise SystemExit("--audio-dir is required with --audio-list")
        rows = build_audio_rows(
            audio_list=args.audio_list,
            crop_seconds=float(args.crop_seconds),
            step_seconds=float(args.step_seconds),
            clip_seconds=float(args.clip_seconds),
        )
        scored_rows = infer_audio_rows(
            rows=rows,
            raw_audio_dir=args.audio_dir,
            specs=specs,
            batch_size=int(args.batch_size),
            device=device,
            clip_seconds=float(args.clip_seconds),
        )
        if args.raw_audio_dir is None:
            args.raw_audio_dir = args.audio_dir
    else:
        if args.manifest_csv is None:
            raise SystemExit("Either --manifest-csv or --audio-list is required")
        rows = read_csv_rows(args.manifest_csv)
        if not rows:
            raise SystemExit(f"No rows in {args.manifest_csv}")
        scores_by_label: Dict[str, np.ndarray] = {}
        for spec in specs:
            scores = infer_expert(
                spec=spec,
                manifest_csv=args.manifest_csv,
                dataset_root=args.dataset_root,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
            )
            if scores.shape[0] != len(rows):
                raise RuntimeError(f"{spec.label_id} returned {scores.shape[0]} scores for {len(rows)} rows")
            scores_by_label[spec.label_id] = scores
        scored_rows = add_scores(rows, specs, scores_by_label)
    write_csv(args.output_dir / "window_predictions.csv", scored_rows)
    events, event_windows = cluster_high_confidence(
        scored_rows,
        label_ids=[spec.label_id for spec in specs],
        low_threshold=float(args.low_threshold),
        high_threshold=float(args.high_threshold),
        min_members=int(args.min_members),
        max_gap_seconds=float(args.max_gap_seconds),
        max_events_per_label=int(args.max_events_per_label),
    )
    write_csv(args.output_dir / "high_confidence_events.csv", events)
    write_csv(args.output_dir / "high_confidence_windows.csv", event_windows)
    cache_manifest = materialize_cache(
        events=events,
        windows=event_windows,
        output_dir=args.output_dir,
        raw_audio_dir=args.raw_audio_dir,
        copy_raw_audio=bool(args.copy_raw_audio),
    )
    if args.cache_audio_crops and args.raw_audio_dir is not None:
        cache_manifest = add_audio_crop_cache(
            cache_manifest=cache_manifest,
            raw_audio_dir=args.raw_audio_dir,
            specs=specs,
            output_dir=args.output_dir,
            clip_seconds=float(args.clip_seconds),
        )
    write_summary(
        output_dir=args.output_dir,
        rows=scored_rows,
        events=events,
        specs=specs,
        args=args,
        cache_manifest=cache_manifest,
    )
    print(json.dumps(read_json(args.output_dir / "prediction_summary.json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
