#!/usr/bin/env python3
"""Append GAVDNet-inspired synthetic whale-call examples to an E123 H5 dataset.

The Nature/GAVDNet work synthesizes audio by perturbing clean stereotyped calls
and mixing them with real background at controlled SNRs. This script ports the
experiment idea to the existing SSAMBA H5 bridge: it operates in spectrogram
space, appends synthetic training rows only, and leaves validation/test rows
unchanged so downstream metrics still come from real examples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except Exception:
    h5py = None


@dataclass(frozen=True)
class AugmentConfig:
    snr_db_min: float = -10.0
    snr_db_max: float = 10.0
    freq_shift_min_bins: int = -12
    freq_shift_max_bins: int = 12
    time_stretch_min: float = 0.97
    time_stretch_max: float = 1.03
    transmission_loss_strength_min: float = 0.10
    transmission_loss_strength_max: float = 0.75
    reverb_smear_strength_min: float = 0.0
    reverb_smear_strength_max: float = 0.0
    reverb_smear_decay_min_bins: int = 2
    reverb_smear_decay_max_bins: int = 12
    gaussian_noise_std: float = 0.01
    seed: int = 1337


def read_strings(dataset: Any) -> List[str]:
    out: List[str] = []
    for value in dataset[:]:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def write_strings(handle: Any, name: str, values: Sequence[str], *, compression: str) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(name, data=np.asarray(list(values), dtype=object), dtype=dtype, compression=compression)


def squeeze_spec(spec: np.ndarray) -> np.ndarray:
    arr = np.asarray(spec, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D spectrogram or trailing singleton channel, got shape {arr.shape}")
    return arr


def restore_channel(spec: np.ndarray, reference_shape: Sequence[int]) -> np.ndarray:
    out = np.asarray(spec, dtype=np.float32)
    if len(reference_shape) == 3 and int(reference_shape[-1]) == 1:
        out = out[..., np.newaxis]
    return out


def robust_fill_value(spec: np.ndarray) -> float:
    return float(np.nanpercentile(np.asarray(spec, dtype=np.float32), 5))


def frequency_shift(spec: np.ndarray, bins: int, *, fill_value: Optional[float] = None) -> np.ndarray:
    """Translate a spectrogram in frequency bins without circular wrapping."""
    arr = squeeze_spec(spec)
    shift = int(bins)
    if shift == 0:
        return arr.copy()
    fill = robust_fill_value(arr) if fill_value is None else float(fill_value)
    out = np.full_like(arr, fill, dtype=np.float32)
    if abs(shift) >= arr.shape[0]:
        return out
    if shift > 0:
        out[shift:, :] = arr[:-shift, :]
    else:
        out[:shift, :] = arr[-shift:, :]
    return out


def time_stretch_to_length(spec: np.ndarray, factor: float) -> np.ndarray:
    """Stretch/compress the time axis, then center crop/pad back to input length."""
    arr = squeeze_spec(spec)
    freq, time = arr.shape
    factor = max(0.05, float(factor))
    new_time = max(1, int(round(time * factor)))
    old_x = np.linspace(0.0, 1.0, time, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, new_time, dtype=np.float32)
    stretched = np.empty((freq, new_time), dtype=np.float32)
    for idx in range(freq):
        stretched[idx] = np.interp(new_x, old_x, arr[idx]).astype(np.float32)
    if new_time == time:
        return stretched
    if new_time > time:
        start = (new_time - time) // 2
        return stretched[:, start : start + time].astype(np.float32)
    pad_left = (time - new_time) // 2
    pad_right = time - new_time - pad_left
    return np.pad(stretched, ((0, 0), (pad_left, pad_right)), mode="edge").astype(np.float32)


def smooth_random_envelope(length: int, rng: np.random.Generator, *, strength: float) -> np.ndarray:
    """Smooth amplitude envelope approximating transmission-loss variation."""
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    strength = float(np.clip(strength, 0.0, 0.95))
    walk = np.cumsum(rng.normal(0.0, 0.1, size=length)).astype(np.float32)
    window = max(3, int(round(length / 12)))
    kernel = np.hanning(window).astype(np.float32)
    if float(kernel.sum()) == 0.0:
        kernel = np.ones(window, dtype=np.float32)
    kernel /= float(kernel.sum())
    smooth = np.convolve(walk, kernel, mode="same")
    smooth -= float(smooth.min())
    denom = float(smooth.max()) or 1.0
    smooth /= denom
    return (1.0 - strength * smooth).astype(np.float32)


def apply_transmission_loss(spec: np.ndarray, rng: np.random.Generator, *, strength: float) -> np.ndarray:
    arr = squeeze_spec(spec)
    envelope = smooth_random_envelope(arr.shape[1], rng, strength=strength)
    return (arr * envelope[np.newaxis, :]).astype(np.float32)


def apply_reverb_smear(
    spec: np.ndarray,
    *,
    strength: float,
    decay_bins: int,
) -> np.ndarray:
    """Causal time-axis smear approximating simple audio reverberation."""
    arr = squeeze_spec(spec)
    strength = float(np.clip(strength, 0.0, 0.95))
    if strength <= 0.0:
        return arr.copy()
    decay = max(1, int(decay_bins))
    kernel_len = max(2, min(arr.shape[1], 4 * decay + 1))
    kernel = np.exp(-np.arange(kernel_len, dtype=np.float32) / float(decay))
    kernel /= float(kernel.sum()) or 1.0
    smeared = np.empty_like(arr, dtype=np.float32)
    for freq_idx in range(arr.shape[0]):
        conv = np.convolve(arr[freq_idx], kernel, mode="full")[: arr.shape[1]]
        smeared[freq_idx] = ((1.0 - strength) * arr[freq_idx] + strength * conv).astype(np.float32)
    return smeared


def rms(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix arrays so signal/noise RMS approximates the requested SNR."""
    sig = squeeze_spec(signal).astype(np.float32)
    bg = squeeze_spec(noise).astype(np.float32)
    sig_center = sig - float(np.mean(sig))
    bg_center = bg - float(np.mean(bg))
    sig_rms = max(rms(sig_center), 1e-6)
    bg_rms = max(rms(bg_center), 1e-6)
    noise_scale = sig_rms / (bg_rms * (10.0 ** (float(snr_db) / 20.0)))
    mixed = sig_center + bg_center * noise_scale
    # Recenter and clip to robust source/background range to keep H5 statistics sane.
    target_mean = 0.5 * (float(np.mean(sig)) + float(np.mean(bg)))
    mixed = mixed + target_mean
    low = min(float(np.nanpercentile(sig, 1)), float(np.nanpercentile(bg, 1)))
    high = max(float(np.nanpercentile(sig, 99)), float(np.nanpercentile(bg, 99)))
    if high > low:
        mixed = np.clip(mixed, low, high)
    return np.nan_to_num(mixed, nan=0.0, neginf=low, posinf=high).astype(np.float32)


def synthesize_spectrogram(
    signal: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    freq_shift_bins = int(rng.integers(config.freq_shift_min_bins, config.freq_shift_max_bins + 1))
    time_stretch = float(rng.uniform(config.time_stretch_min, config.time_stretch_max))
    tl_strength = float(
        rng.uniform(config.transmission_loss_strength_min, config.transmission_loss_strength_max)
    )
    reverb_smear_strength = float(
        rng.uniform(config.reverb_smear_strength_min, config.reverb_smear_strength_max)
    )
    reverb_smear_decay_bins = int(
        rng.integers(config.reverb_smear_decay_min_bins, config.reverb_smear_decay_max_bins + 1)
    )
    snr_db = float(rng.uniform(config.snr_db_min, config.snr_db_max))
    spec = frequency_shift(signal, freq_shift_bins)
    spec = time_stretch_to_length(spec, time_stretch)
    spec = apply_transmission_loss(spec, rng, strength=tl_strength)
    spec = apply_reverb_smear(
        spec,
        strength=reverb_smear_strength,
        decay_bins=reverb_smear_decay_bins,
    )
    mixed = mix_at_snr(spec, squeeze_spec(background), snr_db)
    if config.gaussian_noise_std > 0:
        mixed = mixed + rng.normal(0.0, config.gaussian_noise_std, size=mixed.shape).astype(np.float32)
    mixed = np.nan_to_num(mixed, nan=0.0).astype(np.float32)
    params = {
        "freq_shift_bins": freq_shift_bins,
        "time_stretch": time_stretch,
        "transmission_loss_strength": tl_strength,
        "reverb_smear_strength": reverb_smear_strength,
        "reverb_smear_decay_bins": reverb_smear_decay_bins,
        "snr_db": snr_db,
    }
    return mixed, params


def token_set(label_string: str) -> set[str]:
    return {token.strip() for token in str(label_string or "").split(";") if token.strip()}


def select_indices(
    *,
    label_strings: Sequence[str],
    splits: Sequence[str],
    target_labels: Sequence[str],
    split: str,
) -> Tuple[Dict[str, List[int]], List[int]]:
    target_to_indices: Dict[str, List[int]] = {label: [] for label in target_labels}
    normal_indices: List[int] = []
    for idx, (label_string, row_split) in enumerate(zip(label_strings, splits)):
        if str(row_split) != split:
            continue
        labels = token_set(label_string)
        if "normal" in labels:
            normal_indices.append(idx)
        for target in target_labels:
            if target in labels:
                target_to_indices[target].append(idx)
    return target_to_indices, normal_indices


def copy_numeric_dataset(src: Any, dst: Any, name: str, total_rows: int, *, compression: str) -> Any:
    source = src[name]
    shape = (total_rows, *source.shape[1:])
    target = dst.create_dataset(
        name,
        shape=shape,
        dtype=source.dtype,
        chunks=source.chunks if source.chunks and source.chunks[0] == 1 else True,
        compression=compression,
    )
    chunk = 64
    for start in range(0, source.shape[0], chunk):
        end = min(source.shape[0], start + chunk)
        target[start:end] = source[start:end]
    return target


def build_synthetic_h5(
    *,
    input_h5: Path,
    output_h5: Path,
    output_summary: Path,
    target_labels: Sequence[str],
    synthetic_per_target: int,
    split: str,
    config: AugmentConfig,
    compression: str,
) -> Dict[str, Any]:
    if h5py is None:
        raise RuntimeError("h5py is required to build an augmented H5 dataset")
    rng = np.random.default_rng(config.seed)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_h5, "r") as src:
        for name in ("spectrograms", "labels", "label_strings", "item_ids", "splits"):
            if name not in src:
                raise KeyError(f"input H5 missing required dataset {name!r}")
        label_strings = read_strings(src["label_strings"])
        item_ids = read_strings(src["item_ids"])
        splits = read_strings(src["splits"])
        sources = read_strings(src["sources"]) if "sources" in src else ["" for _ in item_ids]
        source_kinds = read_strings(src["source_kinds"]) if "source_kinds" in src else ["" for _ in item_ids]
        target_to_indices, normal_indices = select_indices(
            label_strings=label_strings,
            splits=splits,
            target_labels=target_labels,
            split=split,
        )
        missing_targets = [label for label, indices in target_to_indices.items() if not indices]
        if missing_targets:
            raise ValueError(f"no exemplar rows for target label(s): {missing_targets}")
        if not normal_indices:
            raise ValueError(f"no normal/background rows in split {split!r}")

        synth_total = len(target_labels) * int(synthetic_per_target)
        original_rows = int(src["spectrograms"].shape[0])
        total_rows = original_rows + synth_total
        with h5py.File(output_h5, "w") as dst:
            spec_ds = copy_numeric_dataset(src, dst, "spectrograms", total_rows, compression=compression)
            label_ds = copy_numeric_dataset(src, dst, "labels", total_rows, compression=compression)
            new_label_strings = list(label_strings)
            new_item_ids = list(item_ids)
            new_sources = list(sources)
            new_splits = list(splits)
            new_source_kinds = list(source_kinds)
            synth_params: List[Dict[str, Any]] = []
            write_idx = original_rows
            for target in target_labels:
                exemplar_indices = target_to_indices[target]
                for local_idx in range(int(synthetic_per_target)):
                    exemplar_idx = int(rng.choice(exemplar_indices))
                    normal_idx = int(rng.choice(normal_indices))
                    synthetic, params = synthesize_spectrogram(
                        src["spectrograms"][exemplar_idx],
                        src["spectrograms"][normal_idx],
                        rng,
                        config,
                    )
                    spec_ds[write_idx] = restore_channel(synthetic, src["spectrograms"].shape[1:])
                    label_ds[write_idx] = src["labels"][exemplar_idx]
                    new_label_strings.append(target)
                    new_item_ids.append(
                        f"synthetic_gavdnet_like::{target}::{local_idx:06d}::"
                        f"{item_ids[exemplar_idx]}::noise::{item_ids[normal_idx]}"
                    )
                    new_sources.append("synthetic_gavdnet_like")
                    new_splits.append(split)
                    new_source_kinds.append("synthetic")
                    synth_params.append(
                        {
                            "row_index": write_idx,
                            "target": target,
                            "exemplar_index": exemplar_idx,
                            "normal_index": normal_idx,
                            **params,
                        }
                    )
                    write_idx += 1
            for name in ("label_strings", "item_ids", "sources", "splits", "source_kinds"):
                values = {
                    "label_strings": new_label_strings,
                    "item_ids": new_item_ids,
                    "sources": new_sources,
                    "splits": new_splits,
                    "source_kinds": new_source_kinds,
                }[name]
                write_strings(dst, name, values, compression=compression)
            if "anomaly_label_names" in src:
                write_strings(dst, "anomaly_label_names", read_strings(src["anomaly_label_names"]), compression=compression)
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["synthetic_augmentation"] = "e127-gavdnet-like-spectrogram-v1"

    label_counts = Counter()
    split_counts = Counter()
    source_kind_counts = Counter()
    for label, row_split, source_kind in zip(new_label_strings, new_splits, new_source_kinds):
        label_counts[label] += 1
        split_counts[row_split] += 1
        source_kind_counts[source_kind] += 1
    summary = {
        "input_h5": str(input_h5),
        "output_h5": str(output_h5),
        "original_rows": original_rows,
        "synthetic_rows": synth_total,
        "rows_written": total_rows,
        "target_labels": list(target_labels),
        "synthetic_per_target": int(synthetic_per_target),
        "split": split,
        "normal_pool_rows": len(normal_indices),
        "target_pool_rows": {label: len(indices) for label, indices in target_to_indices.items()},
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "augment_config": asdict(config),
        "synthetic_param_preview": synth_params[:20],
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--target-label", action="append", default=None, help="Label string to augment; may be repeated")
    parser.add_argument("--synthetic-per-target", type=int, default=1000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--snr-db-min", type=float, default=-10.0)
    parser.add_argument("--snr-db-max", type=float, default=10.0)
    parser.add_argument("--freq-shift-min-bins", type=int, default=-12)
    parser.add_argument("--freq-shift-max-bins", type=int, default=12)
    parser.add_argument("--time-stretch-min", type=float, default=0.97)
    parser.add_argument("--time-stretch-max", type=float, default=1.03)
    parser.add_argument("--transmission-loss-strength-min", type=float, default=0.10)
    parser.add_argument("--transmission-loss-strength-max", type=float, default=0.75)
    parser.add_argument("--reverb-smear-strength-min", type=float, default=0.0)
    parser.add_argument("--reverb-smear-strength-max", type=float, default=0.0)
    parser.add_argument("--reverb-smear-decay-min-bins", type=int, default=2)
    parser.add_argument("--reverb-smear-decay-max-bins", type=int, default=12)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.01)
    parser.add_argument("--compression", default="gzip")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    targets = args.target_label or ["Bm", "Mn"]
    config = AugmentConfig(
        snr_db_min=args.snr_db_min,
        snr_db_max=args.snr_db_max,
        freq_shift_min_bins=args.freq_shift_min_bins,
        freq_shift_max_bins=args.freq_shift_max_bins,
        time_stretch_min=args.time_stretch_min,
        time_stretch_max=args.time_stretch_max,
        transmission_loss_strength_min=args.transmission_loss_strength_min,
        transmission_loss_strength_max=args.transmission_loss_strength_max,
        reverb_smear_strength_min=args.reverb_smear_strength_min,
        reverb_smear_strength_max=args.reverb_smear_strength_max,
        reverb_smear_decay_min_bins=args.reverb_smear_decay_min_bins,
        reverb_smear_decay_max_bins=args.reverb_smear_decay_max_bins,
        gaussian_noise_std=args.gaussian_noise_std,
        seed=args.seed,
    )
    summary = build_synthetic_h5(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        output_summary=args.output_summary,
        target_labels=targets,
        synthetic_per_target=args.synthetic_per_target,
        split=args.split,
        config=config,
        compression=args.compression,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
