#!/usr/bin/env python3
"""
Train a fin-whale call detector from Perch 2.0 embeddings (no spectrograms).

Pipeline:
1) Load call annotations from one or more Excel files.
2) Build base context windows (default 40s) centered on calls.
3) Split contexts with leakage-safe time separation.
4) Build train/eval subclips (default 10s) from contexts:
   - train positives: decentered jitter
   - train negatives: random offset
   - val/test: deterministic centered clip
5) Extract Perch embeddings from subclips and train/evaluate classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.dataset.call_catalog import load_whale_data
from src.dataset.negative_sampler import sample_negative_windows_for_file
from src.training.splits import split_time_separated, summarise_counts


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sample_call_fraction_in_window(
    rng: np.random.Generator,
    center_bias_sigma_frac: float,
) -> float:
    """Match CNN pipeline decenter rule: Gaussian around 0.5 truncated to [0,1]."""
    sigma = max(1e-3, float(center_bias_sigma_frac)) * 0.5
    for _ in range(10):
        frac = 0.5 + float(rng.normal(0.0, sigma))
        if 0.0 <= frac <= 1.0:
            return float(frac)
    return 0.5


def build_split_indices(
    used_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    min_gap_seconds: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
    """Build leakage-safe split indices over already-extracted windows."""
    entries: List[dict] = []
    for idx, row in used_df.iterrows():
        entries.append(
            {
                "idx": int(idx),
                "src": str(row["src"]),
                "start": _safe_float(row.get("resolved_window_start_s"), default=row.get("window_start_s", 0.0)),
                "dur": _safe_float(row.get("window_duration_s"), default=5.0),
                "label": int(row["label"]),
            }
        )

    split = split_time_separated(
        entries=entries,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        min_gap_seconds=min_gap_seconds,
    )
    split_counts = summarise_counts(split)
    index_by_split: Dict[str, np.ndarray] = {}
    for split_name, split_entries in split.items():
        index_by_split[split_name] = np.asarray([int(e["idx"]) for e in split_entries], dtype=np.int64)
    return index_by_split, split_counts


def _clip_interval_within_context(
    desired_start_s: float,
    clip_duration_s: float,
    context_start_s: float,
    context_duration_s: float,
) -> float:
    context_end_s = float(context_start_s) + float(context_duration_s)
    max_start = max(float(context_start_s), context_end_s - float(clip_duration_s))
    return float(np.clip(float(desired_start_s), float(context_start_s), float(max_start)))


def build_subclip_manifest_from_contexts(
    context_manifest: pd.DataFrame,
    context_index_by_split: Dict[str, np.ndarray],
    train_clip_seconds: float,
    eval_clip_seconds: float,
    center_bias_sigma_frac: float,
    train_pos_augment_copies: int,
    train_neg_augment_copies: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Create subclip manifest from base context windows following train/eval rules."""
    rng = np.random.default_rng(seed)
    records: List[dict] = []

    for split_name in ("train", "val", "test"):
        split_idx = context_index_by_split.get(split_name, np.asarray([], dtype=np.int64))
        if split_idx.size == 0:
            continue
        split_df = context_manifest.iloc[split_idx]

        for row in split_df.to_dict("records"):
            label = int(row.get("label", 0))
            context_start = _safe_float(row["window_start_s"], default=0.0)
            context_duration = _safe_float(row["window_duration_s"], default=40.0)
            context_center = context_start + 0.5 * context_duration
            call_begin = _safe_float(row.get("call_begin_s"))
            call_end = _safe_float(row.get("call_end_s"))
            call_center = 0.5 * (call_begin + call_end) if np.isfinite(call_begin) and np.isfinite(call_end) else context_center

            if split_name == "train":
                clip_seconds = float(train_clip_seconds)
                if label == 1:
                    n_copies = int(train_pos_augment_copies)
                    if n_copies <= 0:
                        n_copies = 1
                    for aug_idx in range(n_copies):
                        if int(train_pos_augment_copies) > 0:
                            frac = _sample_call_fraction_in_window(rng, center_bias_sigma_frac)
                        else:
                            frac = 0.5
                        raw_start = call_center - frac * clip_seconds
                        sub_start = _clip_interval_within_context(
                            desired_start_s=raw_start,
                            clip_duration_s=clip_seconds,
                            context_start_s=context_start,
                            context_duration_s=context_duration,
                        )
                        rec = dict(row)
                        rec["example_id"] = f"{row['example_id']}_train_pos_{aug_idx:02d}"
                        rec["window_start_s"] = sub_start
                        rec["window_end_s"] = sub_start + clip_seconds
                        rec["window_duration_s"] = clip_seconds
                        rec["split"] = split_name
                        rec["augmentation"] = "train_positive_decenter" if int(train_pos_augment_copies) > 0 else "train_positive_center"
                        rec["target_call_fraction_in_window"] = float(frac)
                        rec["context_start_s"] = float(context_start)
                        rec["context_duration_s"] = float(context_duration)
                        records.append(rec)
                else:
                    n_copies = int(train_neg_augment_copies)
                    if n_copies <= 0:
                        n_copies = 1
                    max_offset = max(0.0, context_duration - clip_seconds)
                    for aug_idx in range(n_copies):
                        if int(train_neg_augment_copies) > 0 and max_offset > 0:
                            offset = float(rng.uniform(0.0, max_offset))
                            aug_name = "train_negative_random"
                        else:
                            offset = 0.5 * max_offset
                            aug_name = "train_negative_center"
                        sub_start = float(context_start + offset)
                        rec = dict(row)
                        rec["example_id"] = f"{row['example_id']}_train_neg_{aug_idx:02d}"
                        rec["window_start_s"] = sub_start
                        rec["window_end_s"] = sub_start + clip_seconds
                        rec["window_duration_s"] = clip_seconds
                        rec["split"] = split_name
                        rec["augmentation"] = aug_name
                        rec["target_call_fraction_in_window"] = np.nan
                        rec["context_start_s"] = float(context_start)
                        rec["context_duration_s"] = float(context_duration)
                        records.append(rec)
            else:
                clip_seconds = float(eval_clip_seconds)
                if label == 1:
                    raw_start = call_center - 0.5 * clip_seconds
                    aug_name = "eval_positive_center"
                    frac = 0.5
                else:
                    raw_start = context_center - 0.5 * clip_seconds
                    aug_name = "eval_negative_center"
                    frac = np.nan
                sub_start = _clip_interval_within_context(
                    desired_start_s=raw_start,
                    clip_duration_s=clip_seconds,
                    context_start_s=context_start,
                    context_duration_s=context_duration,
                )
                rec = dict(row)
                rec["example_id"] = f"{row['example_id']}_{split_name}_center"
                rec["window_start_s"] = sub_start
                rec["window_end_s"] = sub_start + clip_seconds
                rec["window_duration_s"] = clip_seconds
                rec["split"] = split_name
                rec["augmentation"] = aug_name
                rec["target_call_fraction_in_window"] = frac
                rec["context_start_s"] = float(context_start)
                rec["context_duration_s"] = float(context_duration)
                records.append(rec)

    out = pd.DataFrame(records).reset_index(drop=True)
    summary = {
        "total_subclips": int(len(out)),
        "train_subclips": int((out["split"] == "train").sum()) if not out.empty else 0,
        "val_subclips": int((out["split"] == "val").sum()) if not out.empty else 0,
        "test_subclips": int((out["split"] == "test").sum()) if not out.empty else 0,
        "positive_subclips": int((out["label"] == 1).sum()) if not out.empty else 0,
        "negative_subclips": int((out["label"] == 0).sum()) if not out.empty else 0,
        "train_clip_seconds": float(train_clip_seconds),
        "eval_clip_seconds": float(eval_clip_seconds),
    }
    return out, summary

def _sample_positives(df: pd.DataFrame, max_positives: int | None, seed: int) -> pd.DataFrame:
    if max_positives is None or max_positives <= 0 or max_positives >= len(df):
        return df.copy()

    rng = np.random.default_rng(seed)
    if "call_type" not in df.columns:
        return df.sample(n=max_positives, random_state=seed).copy()

    parts: List[pd.DataFrame] = []
    groups = list(df.groupby("call_type", dropna=False))
    if not groups:
        return df.sample(n=max_positives, random_state=seed).copy()

    per_group = max(1, max_positives // len(groups))
    for _, group in groups:
        n = min(len(group), per_group)
        parts.append(group.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))

    sampled = pd.concat(parts, ignore_index=False).drop_duplicates()
    if len(sampled) < max_positives:
        remaining = df.drop(sampled.index, errors="ignore")
        if not remaining.empty:
            extra = remaining.sample(
                n=min(max_positives - len(sampled), len(remaining)),
                random_state=seed,
            )
            sampled = pd.concat([sampled, extra], ignore_index=False)

    sampled = sampled.sample(frac=1.0, random_state=seed).head(max_positives)
    return sampled.copy()


def build_window_manifest(
    excel_files: List[str],
    context_duration_s: float,
    negatives_per_positive: int,
    negative_margin_s: float,
    max_positives: int | None,
    max_audio_files: int | None,
    seed: int,
    assumed_clip_duration_s: float = 300.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build a base context-window manifest with positives and sampled negatives."""
    whale_df = load_whale_data(excel_files)
    whale_df = whale_df.copy()

    whale_df["begin time (s)"] = pd.to_numeric(whale_df["begin time (s)"], errors="coerce")
    whale_df["end time (s)"] = pd.to_numeric(whale_df["end time (s)"], errors="coerce")
    whale_df["call_duration_s"] = whale_df["end time (s)"] - whale_df["begin time (s)"]
    whale_df = whale_df[
        whale_df["clip id"].notna()
        & whale_df["begin time (s)"].notna()
        & whale_df["end time (s)"].notna()
        & (whale_df["end time (s)"] > whale_df["begin time (s)"])
    ].copy()

    if whale_df.empty:
        raise RuntimeError("No valid calls were found in the provided Excel files.")

    if max_audio_files is not None and max_audio_files > 0:
        keep_clips = sorted(whale_df["clip id"].astype(str).unique())[:max_audio_files]
        whale_df = whale_df[whale_df["clip id"].astype(str).isin(keep_clips)].copy()
        if whale_df.empty:
            raise RuntimeError("No calls remain after applying --max-audio-files.")

    positives = _sample_positives(whale_df, max_positives=max_positives, seed=seed)
    if positives.empty:
        raise RuntimeError("No positive examples remain after sampling.")

    half_context = context_duration_s / 2.0
    records: List[dict] = []

    for row_idx, row in positives.iterrows():
        clip_id = str(row["clip id"])
        begin_s = _safe_float(row["begin time (s)"])
        end_s = _safe_float(row["end time (s)"])
        center_s = 0.5 * (begin_s + end_s)
        max_start = max(0.0, float(assumed_clip_duration_s) - float(context_duration_s))
        window_start_s = float(np.clip(center_s - half_context, 0.0, max_start))
        window_end_s = window_start_s + context_duration_s
        records.append(
            {
                "example_id": f"pos_{len(records):07d}",
                "label": 1,
                "label_name": "call",
                "clip_id": clip_id,
                "src": clip_id,
                "window_start_s": window_start_s,
                "window_end_s": window_end_s,
                "window_duration_s": context_duration_s,
                "call_begin_s": begin_s,
                "call_end_s": end_s,
                "call_type": str(row.get("call_type", "unknown")),
                "source_file": str(row.get("source_file", "")),
                "source_row_idx": int(row_idx),
            }
        )

    calls_by_file: Dict[str, List[Tuple[float, float]]] = {}
    for clip_id, group in whale_df.groupby("clip id"):
        clip = str(clip_id)
        pairs: List[Tuple[float, float]] = []
        for b, e in zip(group["begin time (s)"], group["end time (s)"]):
            b_s = _safe_float(b)
            e_s = _safe_float(e)
            if np.isfinite(b_s) and np.isfinite(e_s) and e_s > b_s:
                pairs.append((b_s, e_s))
        calls_by_file[clip] = pairs

    prev_rng_state = random.getstate()
    random.seed(seed)
    try:
        pos_counts_by_clip = positives["clip id"].astype(str).value_counts().to_dict()
        for clip_id, pos_count in pos_counts_by_clip.items():
            target_neg = int(max(0, pos_count * max(0, negatives_per_positive)))
            if target_neg == 0:
                continue
            negative_windows = sample_negative_windows_for_file(
                clip_id=clip_id,
                duration=float(assumed_clip_duration_s),
                context_duration=context_duration_s,
                calls_by_file=calls_by_file,
                max_windows=target_neg,
                margin=negative_margin_s,
                strategy="random",
            )
            for neg_idx, (start_s, end_s) in enumerate(negative_windows):
                records.append(
                    {
                        "example_id": f"neg_{clip_id}_{neg_idx:05d}",
                        "label": 0,
                        "label_name": "background",
                        "clip_id": clip_id,
                        "src": clip_id,
                        "window_start_s": float(start_s),
                        "window_end_s": float(end_s),
                        "window_duration_s": context_duration_s,
                        "call_begin_s": np.nan,
                        "call_end_s": np.nan,
                        "call_type": "negative",
                        "source_file": "",
                        "source_row_idx": -1,
                    }
                )
    finally:
        random.setstate(prev_rng_state)

    manifest = pd.DataFrame(records)
    manifest = manifest.sort_values(["clip_id", "window_start_s", "label"], kind="mergesort").reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("Generated manifest is empty.")

    summary = {
        "total_context_windows": int(len(manifest)),
        "positive_context_windows": int((manifest["label"] == 1).sum()),
        "negative_context_windows": int((manifest["label"] == 0).sum()),
        "unique_clips": int(manifest["clip_id"].nunique()),
        "context_duration_s": float(context_duration_s),
    }
    return manifest, summary


def _read_audio_window(
    audio_file: sf.SoundFile,
    start_s: float,
    window_size_s: float,
) -> Tuple[np.ndarray, int]:
    """Read a fixed-length window and zero-pad if it runs out of bounds."""
    sample_rate = int(audio_file.samplerate)
    total_frames = len(audio_file)
    target_frames = int(round(window_size_s * sample_rate))
    if target_frames <= 0:
        return np.zeros((0,), dtype=np.float32), sample_rate

    start_frame = int(round(start_s * sample_rate))
    pre_pad = 0
    if start_frame < 0:
        pre_pad = -start_frame
        start_frame = 0

    if start_frame >= total_frames:
        return np.zeros((target_frames,), dtype=np.float32), sample_rate

    audio_file.seek(start_frame)
    readable = max(0, min(target_frames - pre_pad, total_frames - start_frame))
    clip = audio_file.read(frames=readable, dtype="float32", always_2d=False)

    if clip is None:
        clip = np.zeros((0,), dtype=np.float32)
    if isinstance(clip, np.ndarray) and clip.ndim == 2:
        clip = clip.mean(axis=1)

    if pre_pad > 0:
        clip = np.pad(clip, (pre_pad, 0))
    if len(clip) < target_frames:
        clip = np.pad(clip, (0, target_frames - len(clip)))
    elif len(clip) > target_frames:
        clip = clip[:target_frames]
    return clip.astype(np.float32, copy=False), sample_rate


def _resample_fixed_length(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int,
    target_length: int,
) -> np.ndarray:
    if original_sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=target_sr,
            res_type="soxr_hq",
        )
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    elif len(audio) > target_length:
        audio = audio[:target_length]
    return audio.astype(np.float32, copy=False)


def _resolve_audio_path(audio_dir: Path, clip_id: str) -> Optional[Path]:
    """Resolve clip id to an existing audio file path."""
    clip_id = str(clip_id)
    direct = audio_dir / clip_id
    if direct.exists():
        return direct
    if not clip_id.lower().endswith(".wav"):
        with_wav = audio_dir / f"{clip_id}.wav"
        if with_wav.exists():
            return with_wav
    return None


def extract_perch_embeddings(
    manifest: pd.DataFrame,
    audio_dir: Path,
    perch_model_name: str,
    batch_size: int,
    disable_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    """Extract embeddings for each manifest row."""
    clip_ids = manifest["clip_id"].astype(str).unique().tolist()
    resolved_audio_by_clip: Dict[str, Optional[Path]] = {
        clip_id: _resolve_audio_path(audio_dir=audio_dir, clip_id=clip_id)
        for clip_id in clip_ids
    }
    available_clip_count = int(sum(1 for p in resolved_audio_by_clip.values() if p is not None))
    missing_clips = [clip_id for clip_id, p in resolved_audio_by_clip.items() if p is None]
    print(
        "Audio preflight | "
        f"manifest_clips={len(clip_ids)} "
        f"available_clips={available_clip_count} "
        f"missing_clips={len(missing_clips)}"
    )
    if available_clip_count == 0:
        sample_missing = ", ".join(missing_clips[:5])
        if len(missing_clips) > 5:
            sample_missing += ", ..."
        raise RuntimeError(
            "No manifest clip_ids could be resolved under --audio-dir. "
            "This usually means clip_id naming mismatch or over-restrictive sampling "
            "(e.g., low --max-audio-files selecting bad IDs). "
            f"Sample missing clip_ids: {sample_missing}"
        )

    if disable_gpu:
        # Must be set before TensorFlow model load.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        from perch_hoplite.zoo import model_configs
    except Exception as exc:
        raise SystemExit(
            "Perch dependencies are missing. Install with:\n"
            "  /home/sbialek/ONC/whale-call-analysis/.venv/bin/pip install \"perch-hoplite[tf]\" kagglehub\n"
            f"Import error: {exc}"
        ) from exc

    print(f"Loading Perch model preset: {perch_model_name}")
    selected_model_name = perch_model_name
    try:
        model = model_configs.load_model_by_name(perch_model_name)
    except Exception as exc:
        if perch_model_name == "perch_v2":
            # In restricted environments, perch_v2 can auto-select perch_v2_cpu
            # and fail at first-time Kaggle download. Retry with perch_v2_gpu,
            # which uses the same Perch v2 family and often exists in cache.
            selected_model_name = "perch_v2_gpu"
            print(
                "Perch preset perch_v2 failed; retrying with perch_v2_gpu. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
            model = model_configs.load_model_by_name(selected_model_name)
        else:
            raise
    model_sample_rate = int(model.sample_rate)
    model_window_s = float(getattr(model, "window_size_s", 5.0))
    target_samples = int(round(model_window_s * model_sample_rate))
    print(f"Perch model ready | sample_rate={model_sample_rate} | window={model_window_s:.2f}s")

    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    used_records: List[dict] = []
    skipped_records: List[dict] = []

    batch_audio: List[np.ndarray] = []
    batch_meta: List[dict] = []
    total_rows = len(manifest)
    processed_rows = 0

    def flush_batch() -> None:
        nonlocal batch_audio, batch_meta
        if not batch_audio:
            return
        audio_batch = np.stack(batch_audio, axis=0).astype(np.float32)
        outputs = model.batch_embed(audio_batch)
        pooled = outputs.pooled_embeddings(time_pooling="mean", channel_pooling="squeeze")
        if pooled.ndim == 1:
            pooled = pooled[np.newaxis, :]
        for meta, emb in zip(batch_meta, pooled):
            embeddings.append(np.asarray(emb, dtype=np.float32))
            labels.append(int(meta["label"]))
            used_records.append(meta)
        batch_audio = []
        batch_meta = []

    for clip_id, clip_rows in manifest.groupby("clip_id", sort=False):
        audio_path = resolved_audio_by_clip.get(str(clip_id))
        row_records = clip_rows.to_dict("records")
        if audio_path is None:
            for rec in row_records:
                bad = dict(rec)
                bad["skip_reason"] = "missing_audio"
                skipped_records.append(bad)
                processed_rows += 1
            continue

        try:
            with sf.SoundFile(audio_path) as af:
                clip_sr = int(af.samplerate)
                clip_duration_s = len(af) / max(1, clip_sr)
                for rec in row_records:
                    start_s = _safe_float(rec["window_start_s"], default=0.0)
                    req_window_s = _safe_float(rec["window_duration_s"], default=model_window_s)
                    max_start = max(0.0, clip_duration_s - req_window_s)
                    start_s = min(max(0.0, start_s), max_start)
                    audio, sr = _read_audio_window(af, start_s=start_s, window_size_s=req_window_s)
                    if audio.size == 0:
                        bad = dict(rec)
                        bad["skip_reason"] = "empty_audio"
                        skipped_records.append(bad)
                        processed_rows += 1
                        continue

                    audio = _resample_fixed_length(
                        audio=audio,
                        original_sr=sr,
                        target_sr=model_sample_rate,
                        target_length=target_samples,
                    )
                    meta = dict(rec)
                    meta["resolved_window_start_s"] = float(start_s)
                    meta["audio_sample_rate"] = int(sr)
                    meta["audio_duration_s"] = float(clip_duration_s)
                    batch_audio.append(audio)
                    batch_meta.append(meta)
                    if len(batch_audio) >= max(1, batch_size):
                        flush_batch()
                    processed_rows += 1
        except Exception as exc:
            for rec in row_records:
                bad = dict(rec)
                bad["skip_reason"] = f"audio_read_error:{type(exc).__name__}"
                bad["skip_error"] = str(exc)
                skipped_records.append(bad)
                processed_rows += 1

        if processed_rows % 250 == 0 or processed_rows == total_rows:
            print(f"Embedding progress: {processed_rows}/{total_rows} windows")

    flush_batch()

    if not embeddings:
        raise RuntimeError("No embeddings were extracted. Check audio paths and manifest filters.")

    x = np.stack(embeddings, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    used_df = pd.DataFrame(used_records).reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped_records).reset_index(drop=True)

    details = {
        "perch_model_requested": perch_model_name,
        "perch_model_loaded": selected_model_name,
        "model_sample_rate": model_sample_rate,
        "model_window_s": model_window_s,
        "embedding_dim": int(x.shape[1]),
        "manifest_subclips": int(len(manifest)),
        "manifest_unique_clips": int(len(clip_ids)),
        "manifest_available_clips": int(available_clip_count),
        "manifest_missing_clips": int(len(missing_clips)),
        "used_subclips": int(len(used_df)),
        "skipped_subclips": int(len(skipped_df)),
    }
    return x, y, used_df, skipped_df, details


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float("nan")

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": int(len(y_true)),
    }


def train_embedding_classifier(
    x: np.ndarray,
    y: np.ndarray,
    index_by_split: Dict[str, np.ndarray],
    seed: int,
    c: float,
    max_iter: int,
) -> Tuple[StandardScaler, LogisticRegression, dict]:
    """Train logistic regression from precomputed split indices."""
    if index_by_split["train"].size == 0:
        raise RuntimeError("Train split is empty. Increase sample size or adjust split ratios.")

    x_train = x[index_by_split["train"]]
    y_train = y[index_by_split["train"]]

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Train split has only one class. Increase negatives or sample size.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    clf = LogisticRegression(
        C=float(c),
        max_iter=int(max_iter),
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    clf.fit(x_train_scaled, y_train)

    metrics: Dict[str, dict] = {}
    for split_name, split_idx in index_by_split.items():
        if split_idx.size == 0:
            metrics[split_name] = {"empty_split": True}
            continue
        probs = clf.predict_proba(scaler.transform(x[split_idx]))[:, 1]
        metrics[split_name] = compute_binary_metrics(y_true=y[split_idx], y_prob=probs, threshold=0.5)

    return scaler, clf, metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train a Perch 2.0 embedding-based fin-whale call detector"
    )
    ap.add_argument("--excel-files", nargs="+", required=True, help="Excel annotation files")
    ap.add_argument("--audio-dir", type=str, required=True, help="Directory with full .wav clips")
    ap.add_argument(
        "--output-dir",
        type=str,
        default="output/perch2_embedding_training",
        help="Output root for model/artifacts",
    )

    ap.add_argument(
        "--perch-model",
        type=str,
        default="perch_v2_cpu",
        choices=["perch_v2", "perch_v2_gpu", "perch_v2_cpu"],
        help="Perch model preset from perch_hoplite",
    )
    ap.add_argument("--disable-gpu", action="store_true", help="Force TensorFlow CPU mode")
    ap.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")

    ap.add_argument("--context-seconds", type=float, default=40.0, help="Base context window length")
    ap.add_argument("--train-clip-seconds", type=float, default=10.0, help="Train subclip length sampled from context")
    ap.add_argument("--eval-clip-seconds", type=float, default=10.0, help="Val/test subclip length sampled from context")
    ap.add_argument("--assumed-clip-duration-seconds", type=float, default=300.0, help="Nominal source clip duration")
    ap.add_argument("--negatives-per-positive", type=int, default=1, help="Negative windows per positive call")
    ap.add_argument("--negative-margin-seconds", type=float, default=2.0, help="Margin around calls for negatives")
    ap.add_argument("--max-positives", type=int, default=None, help="Cap positive examples for quick runs")
    ap.add_argument("--max-audio-files", type=int, default=None, help="Cap number of unique clips for quick runs")

    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-gap-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--logreg-c", type=float, default=1.0, help="Inverse regularization strength")
    ap.add_argument("--max-iter", type=int, default=3000, help="Logistic regression max iterations")
    ap.add_argument(
        "--train-pos-augment-copies",
        type=int,
        default=1,
        help="Train subclip copies per positive context (0 uses one centered clip)",
    )
    ap.add_argument(
        "--train-neg-augment-copies",
        type=int,
        default=1,
        help="Train subclip copies per negative context (0 uses one centered clip)",
    )
    ap.add_argument(
        "--center-bias-sigma-frac",
        type=float,
        default=0.25,
        help="Decenter jitter strength; mirrors existing CNN center_bias_sigma_frac",
    )
    ap.add_argument("--skip-save-embeddings", action="store_true", help="Skip saving embeddings.npz")
    ap.add_argument("--note", type=str, default="", help="Optional note stored in summary.json")

    args = ap.parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("Invalid split ratios: require train_ratio > 0, val_ratio >= 0, and train+val < 1.")
    if args.context_seconds <= 0:
        raise SystemExit("--context-seconds must be > 0.")
    if args.train_clip_seconds <= 0:
        raise SystemExit("--train-clip-seconds must be > 0.")
    if args.eval_clip_seconds <= 0:
        raise SystemExit("--eval-clip-seconds must be > 0.")
    if args.assumed_clip_duration_seconds <= 0:
        raise SystemExit("--assumed-clip-duration-seconds must be > 0.")
    if args.train_clip_seconds > args.context_seconds:
        raise SystemExit("--train-clip-seconds must be <= --context-seconds.")
    if args.eval_clip_seconds > args.context_seconds:
        raise SystemExit("--eval-clip-seconds must be <= --context-seconds.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0.")
    if args.train_pos_augment_copies < 0:
        raise SystemExit("--train-pos-augment-copies must be >= 0.")
    if args.train_neg_augment_copies < 0:
        raise SystemExit("--train-neg-augment-copies must be >= 0.")
    if args.center_bias_sigma_frac < 0:
        raise SystemExit("--center-bias-sigma-frac must be >= 0.")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / f"perch2_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Building context manifest...")
    context_manifest, context_manifest_summary = build_window_manifest(
        excel_files=list(args.excel_files),
        context_duration_s=float(args.context_seconds),
        negatives_per_positive=int(args.negatives_per_positive),
        negative_margin_s=float(args.negative_margin_seconds),
        max_positives=args.max_positives,
        max_audio_files=args.max_audio_files,
        seed=int(args.seed),
        assumed_clip_duration_s=float(args.assumed_clip_duration_seconds),
    )
    context_manifest_path = run_dir / "context_window_manifest.csv"
    context_manifest.to_csv(context_manifest_path, index=False)
    print(f"Context manifest: {context_manifest_path} | windows={len(context_manifest)}")

    context_index_by_split, context_split_counts = build_split_indices(
        used_df=context_manifest,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        min_gap_seconds=float(args.min_gap_seconds),
        seed=int(args.seed),
    )

    subclip_manifest, subclip_manifest_summary = build_subclip_manifest_from_contexts(
        context_manifest=context_manifest,
        context_index_by_split=context_index_by_split,
        train_clip_seconds=float(args.train_clip_seconds),
        eval_clip_seconds=float(args.eval_clip_seconds),
        center_bias_sigma_frac=float(args.center_bias_sigma_frac),
        train_pos_augment_copies=int(args.train_pos_augment_copies),
        train_neg_augment_copies=int(args.train_neg_augment_copies),
        seed=int(args.seed),
    )
    if subclip_manifest.empty:
        raise RuntimeError("Subclip manifest is empty; cannot train.")
    subclip_manifest_path = run_dir / "subclip_manifest.csv"
    subclip_manifest.to_csv(subclip_manifest_path, index=False)
    print(f"Subclip manifest: {subclip_manifest_path} | subclips={len(subclip_manifest)}")

    x, y, used_df, skipped_df, embedding_details = extract_perch_embeddings(
        manifest=subclip_manifest,
        audio_dir=Path(args.audio_dir),
        perch_model_name=args.perch_model,
        batch_size=int(args.batch_size),
        disable_gpu=bool(args.disable_gpu),
    )

    used_manifest_path = run_dir / "used_subclips.csv"
    used_df.to_csv(used_manifest_path, index=False)
    print(f"Used subclips: {used_manifest_path} | count={len(used_df)}")
    skipped_path = None
    if not skipped_df.empty:
        skipped_path = run_dir / "skipped_subclips.csv"
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Skipped subclips: {skipped_path} | count={len(skipped_df)}")

    index_by_split: Dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        mask = used_df["split"].astype(str).eq(split_name)
        index_by_split[split_name] = np.where(mask.to_numpy())[0].astype(np.int64)

    used_split_counts: Dict[str, Dict[str, int]] = {}
    for split_name, split_idx in index_by_split.items():
        if split_idx.size == 0:
            used_split_counts[split_name] = {"pos": 0, "neg": 0, "total": 0}
            continue
        yy = y[split_idx]
        pos = int((yy == 1).sum())
        neg = int((yy == 0).sum())
        used_split_counts[split_name] = {"pos": pos, "neg": neg, "total": int(len(split_idx))}

    scaler, clf, metrics = train_embedding_classifier(
        x=x,
        y=y,
        index_by_split=index_by_split,
        seed=int(args.seed),
        c=float(args.logreg_c),
        max_iter=int(args.max_iter),
    )

    model_path = run_dir / "perch2_logreg.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "classifier": clf,
            "perch_model": args.perch_model,
            "embedding_dim": int(x.shape[1]),
            "context_seconds": float(args.context_seconds),
            "train_clip_seconds": float(args.train_clip_seconds),
            "eval_clip_seconds": float(args.eval_clip_seconds),
            "train_args": vars(args),
        },
        model_path,
    )
    print(f"Saved model: {model_path}")

    if not args.skip_save_embeddings:
        embeddings_path = run_dir / "embeddings.npz"
        np.savez_compressed(embeddings_path, x=x, y=y)
        print(f"Saved embeddings: {embeddings_path}")

    for split_name, split_idx in index_by_split.items():
        split_file = run_dir / f"{split_name}_subclips.csv"
        used_df.iloc[split_idx].to_csv(split_file, index=False)

    summary = {
        "run_utc": run_stamp,
        "args": vars(args),
        "context_manifest_summary": context_manifest_summary,
        "subclip_manifest_summary": subclip_manifest_summary,
        "embedding_details": embedding_details,
        "context_split_counts": context_split_counts,
        "used_split_counts": used_split_counts,
        "metrics": metrics,
        "artifacts": {
            "context_window_manifest": str(context_manifest_path),
            "subclip_manifest": str(subclip_manifest_path),
            "used_subclips": str(used_manifest_path),
            "skipped_subclips": str(skipped_path) if skipped_path else None,
            "model": str(model_path),
        },
        "note": args.note,
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    test_metrics = metrics.get("test", {})
    if test_metrics and not test_metrics.get("empty_split"):
        print(
            "Test metrics | "
            f"acc={test_metrics['acc']:.4f} "
            f"precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f} "
            f"auc={test_metrics['auc']:.4f}"
        )
    print(f"Done. Output directory: {run_dir}")


if __name__ == "__main__":
    main()
