#!/usr/bin/env python3
"""
Generate a synthetic audit dataset for spectrogram/audio clipping + event stitching.

What this script does:
1) Creates contiguous 5-minute raw audio clips with obvious whale-like synthetic calls.
2) Builds full spectrogram MAT files with edge-padding workflow (same idea as test prep).
3) Creates sliding-window crop MAT+audio items in UnifiedPredictionTracker format.
4) Assigns synthetic model scores from known overlap with injected calls.
5) Runs postprocess clustering/merge to produce stitched event media for verification app.
6) Writes an audit markdown report with alignment and seam-artifact diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io
import scipy.signal
import soundfile as sf

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.unified_prediction_tracker import UnifiedPredictionTracker


CLASS_HIERARCHY = "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"


@dataclass
class SyntheticEvent:
    clip_index: int
    start_sec: float
    duration_sec: float
    f0_hz: float
    f1_hz: float
    amplitude: float
    cluster_id: str
    event_type: str  # "isolated" | "clustered"

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


def _safe_id_token(text: str, fallback: str = "na", max_len: int = 160) -> str:
    text = (text or "").strip()
    if not text:
        text = fallback
    out_chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out_chars.append(ch)
        else:
            out_chars.append("-")
    out = "".join(out_chars).strip("-._")
    if not out:
        out = fallback
    return out[:max_len]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _spectrogram_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    win_dur: float,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win_length = max(1, int(round(win_dur * sample_rate)))
    noverlap = int(round(overlap * win_length))
    noverlap = max(0, min(noverlap, win_length - 1))
    window = scipy.signal.get_window("hann", win_length, fftbins=True)
    freqs, times, power = scipy.signal.spectrogram(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=win_length,
        scaling="density",
        mode="psd",
    )
    power = np.abs(power.astype(np.float32))
    mx = float(np.max(power)) if power.size else 0.0
    if mx > 0:
        pdB = 10.0 * np.log10(np.maximum(power / mx, 1e-10))
    else:
        pdB = np.full_like(power, -100.0, dtype=np.float32)
    return freqs, times, power, pdB


def _crop_freq(
    freqs: np.ndarray,
    power: np.ndarray,
    pdB: np.ndarray,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    if not np.any(mask):
        raise ValueError(f"No frequency bins within [{fmin}, {fmax}] Hz")
    return freqs[mask], power[mask, :], pdB[mask, :]


def _window_starts_even_coverage(total_bins: int, crop_bins: int, step_bins: int) -> List[int]:
    if total_bins <= crop_bins:
        return [0]
    max_start = total_bins - crop_bins
    step_bins = max(1, int(step_bins))
    n_windows = int(math.ceil(max_start / step_bins)) + 1
    if n_windows <= 1:
        return [0]
    starts = np.round(np.linspace(0, max_start, n_windows)).astype(int).tolist()
    # Deduplicate while preserving order
    seen = set()
    out: List[int] = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            out.append(int(s))
    if out[-1] != max_start:
        out.append(int(max_start))
    return out


def _compute_window_time_range(
    times: np.ndarray,
    start_idx: int,
    window_bins: int,
    win_dur: float,
    overlap: float,
) -> Tuple[float, float]:
    start_idx = max(0, min(int(start_idx), len(times) - 1))
    center_start = float(times[start_idx])
    if len(times) > 1:
        hop_sec = float(times[1] - times[0])
    else:
        hop_sec = float(win_dur * (1.0 - overlap))
    window_time_start = max(0.0, center_start - (float(win_dur) / 2.0))
    window_time_end = window_time_start + max(0, int(window_bins) - 1) * hop_sec + float(win_dur)
    return window_time_start, window_time_end


def _generate_call_waveform(
    sample_rate: int,
    duration_sec: float,
    f0_hz: float,
    f1_hz: float,
    amplitude: float,
) -> np.ndarray:
    n = max(1, int(round(duration_sec * sample_rate)))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    carrier = scipy.signal.chirp(
        t,
        f0=float(f0_hz),
        f1=float(f1_hz),
        t1=max(float(duration_sec), 1e-6),
        method="linear",
    ).astype(np.float32)
    harmonic = scipy.signal.chirp(
        t,
        f0=min(2.0 * float(f0_hz), 0.48 * sample_rate),
        f1=min(2.0 * float(f1_hz), 0.48 * sample_rate),
        t1=max(float(duration_sec), 1e-6),
        method="linear",
    ).astype(np.float32)
    envelope = np.hanning(n).astype(np.float32)
    return float(amplitude) * envelope * (carrier + 0.35 * harmonic)


def _build_events(num_clips: int, clip_seconds: float) -> List[SyntheticEvent]:
    events: List[SyntheticEvent] = []
    for i in range(num_clips):
        # Isolated event
        events.append(
            SyntheticEvent(
                clip_index=i,
                start_sec=55.0 + (i % 2) * 8.0,
                duration_sec=8.0,
                f0_hz=18.0,
                f1_hz=27.0,
                amplitude=0.90,
                cluster_id=f"clip{i:02d}_iso",
                event_type="isolated",
            )
        )
        # Clustered events
        cluster_base = 145.0 + (i % 2) * 6.0
        for j, (dt, dur, f0, f1) in enumerate(
            [
                (0.0, 6.0, 16.0, 24.0),
                (8.0, 7.5, 17.0, 29.0),
                (17.0, 6.0, 15.0, 23.0),
            ]
        ):
            events.append(
                SyntheticEvent(
                    clip_index=i,
                    start_sec=cluster_base + dt,
                    duration_sec=dur,
                    f0_hz=f0,
                    f1_hz=f1,
                    amplitude=0.95,
                    cluster_id=f"clip{i:02d}_clusterA",
                    event_type="clustered",
                )
            )
        # Another cluster near tail to force stitching windows near boundaries.
        tail_base = float(clip_seconds) - 28.0
        for j, (dt, dur) in enumerate([(0.0, 7.0), (9.0, 7.0)]):
            events.append(
                SyntheticEvent(
                    clip_index=i,
                    start_sec=tail_base + dt,
                    duration_sec=dur,
                    f0_hz=20.0 + j,
                    f1_hz=33.0 + j,
                    amplitude=1.00,
                    cluster_id=f"clip{i:02d}_clusterB",
                    event_type="clustered",
                )
            )
    return events


def _score_window(
    clip_events: Sequence[SyntheticEvent],
    window_start_sec: float,
    window_end_sec: float,
    rng: random.Random,
) -> float:
    win_len = max(1e-6, window_end_sec - window_start_sec)
    best_overlap = 0.0
    for ev in clip_events:
        ov = max(0.0, min(window_end_sec, ev.end_sec) - max(window_start_sec, ev.start_sec))
        best_overlap = max(best_overlap, ov)
    overlap_ratio = best_overlap / win_len
    if overlap_ratio >= 0.30:
        return float(0.90 + 0.08 * rng.random())
    if best_overlap > 0.0:
        return float(0.72 + 0.15 * rng.random())
    # Background negatives with occasional hard false positives
    if rng.random() < 0.015:
        return float(0.70 + 0.20 * rng.random())
    return float(0.01 + 0.18 * rng.random())


def _profile_lag_seconds(
    mat_path: Path,
    audio_path: Path,
    sample_rate: int,
    win_dur: float,
    overlap: float,
    freq_min: float,
    freq_max: float,
) -> Optional[float]:
    try:
        mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception:
        return None
    spec = np.asarray(mat.get("PdB_norm"))
    times = np.asarray(mat.get("T"), dtype=np.float64).ravel()
    if spec.ndim != 2 or times.size < 2 or spec.shape[1] < 4:
        return None
    dt = np.diff(times)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    spec_dt = float(np.median(dt))
    spec_profile = np.mean(spec.astype(np.float64), axis=0)

    try:
        wav, fs = sf.read(str(audio_path), always_2d=False)
    except Exception:
        return None
    if fs <= 0:
        return None
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    freqs, t_audio, _, pdB = _spectrogram_from_audio(wav, fs, win_dur=win_dur, overlap=overlap)
    freqs, _, pdB = _crop_freq(freqs, pdB, pdB, fmin=freq_min, fmax=freq_max)
    if pdB.shape[1] < 4:
        return None
    audio_profile = np.mean(pdB.astype(np.float64), axis=0)

    # Resample audio profile to spec profile length for lag estimate.
    target_n = len(spec_profile)
    src_n = len(audio_profile)
    x_src = np.linspace(0.0, 1.0, src_n, endpoint=True)
    x_tgt = np.linspace(0.0, 1.0, target_n, endpoint=True)
    audio_rs = np.interp(x_tgt, x_src, audio_profile)

    a = spec_profile - np.mean(spec_profile)
    b = audio_rs - np.mean(audio_rs)
    denom = (np.std(a) * np.std(b))
    if not np.isfinite(denom) or denom == 0:
        return None
    corr = np.correlate(a / np.std(a), b / np.std(b), mode="full")
    lag_bins = int(np.argmax(corr) - (target_n - 1))
    return float(lag_bins) * spec_dt


def _seam_artifact_score(mat_path: Path) -> Optional[float]:
    try:
        mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception:
        return None
    spec = np.asarray(mat.get("PdB_norm"))
    if spec.ndim != 2 or spec.shape[1] < 4:
        return None
    diffs = np.mean(np.abs(np.diff(spec.astype(np.float64), axis=1)), axis=0)
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med))) + 1e-9
    z = (diffs - med) / mad
    return float(np.max(z))


def _constant_column_counts(mat_path: Path) -> Tuple[int, int]:
    try:
        mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception:
        return (0, 0)
    spec = np.asarray(mat.get("PdB_norm"))
    if spec.ndim != 2 or spec.shape[1] == 0:
        return (0, 0)
    zero_cols = int(np.sum(np.all(np.isclose(spec, 0.0, atol=1e-6), axis=0)))
    const_cols = int(np.sum(np.nanstd(spec, axis=0) < 1e-6))
    return zero_cols, const_cols


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic alignment/stitching audit dataset.")
    ap.add_argument("--output-dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--device-code", type=str, default="SYNTHHF0001")
    ap.add_argument("--start-time", type=str, default="2018-07-01T00:00:00Z")
    ap.add_argument("--num-clips", type=int, default=3, help="Number of contiguous 5-minute clips")
    ap.add_argument("--clip-seconds", type=float, default=300.0, help="Clip duration in seconds")
    ap.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
        help="Audio sample rate (Hz). Use >=8000 for broad browser playback compatibility.",
    )
    ap.add_argument("--edge-padding", type=float, default=2.0, help="Padding seconds from neighboring clips")
    ap.add_argument("--win-dur", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.9)
    ap.add_argument("--freq-min", type=float, default=5.0)
    ap.add_argument("--freq-max", type=float, default=100.0)
    ap.add_argument("--crop-size", type=int, default=96)
    ap.add_argument("--window-step", type=int, default=24, help="Sliding-window step in time bins")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--low-threshold", type=float, default=0.70)
    ap.add_argument("--high-threshold", type=float, default=0.82)
    ap.add_argument("--min-members", type=int, default=2)
    ap.add_argument("--max-gap-seconds", type=float, default=15.0)
    ap.add_argument("--merge-min-score", type=float, default=0.80)
    ap.add_argument(
        "--lag-min-score",
        type=float,
        default=0.70,
        help="Only compute spectrogram/audio lag diagnostics for items with score >= this value.",
    )
    ap.add_argument("--skip-postprocess", action="store_true", help="Only build window-level dataset")
    args = ap.parse_args()
    if int(args.sample_rate) < 8000:
        print(
            f"Warning: sample_rate={int(args.sample_rate)} Hz may not play in browser audio elements. "
            "Use --sample-rate 8000 (or higher) for verification app playback."
        )

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    output_root = Path(args.output_dir).resolve()
    raw_audio_dir = _ensure_dir(output_root / "raw_audio")
    start_dt = datetime.fromisoformat(args.start_time.replace("Z", "+00:00")).astimezone(timezone.utc)

    date_str = start_dt.strftime("%Y-%m-%d")
    device_dir = _ensure_dir(output_root / date_str / args.device_code)
    full_spec_dir = _ensure_dir(device_dir / "full_spectrograms")
    spec_crop_dir = _ensure_dir(device_dir / "spectrograms")
    audio_crop_dir = _ensure_dir(device_dir / "audio")

    # 1) Build synthetic events and raw clips
    events = _build_events(num_clips=int(args.num_clips), clip_seconds=float(args.clip_seconds))
    clips: List[np.ndarray] = []
    clip_names: List[str] = []
    clip_datetimes: List[datetime] = []
    for i in range(int(args.num_clips)):
        clip_start = start_dt + timedelta(seconds=float(i) * float(args.clip_seconds))
        clip_datetimes.append(clip_start)
        ts = clip_start.strftime("%Y%m%dT%H%M%S.000Z")
        clip_name = f"{args.device_code}_{ts}.wav"
        clip_names.append(clip_name)

        n_samples = int(round(float(args.clip_seconds) * int(args.sample_rate)))
        audio = (0.025 * np.random.randn(n_samples)).astype(np.float32)
        # Add gentle low-frequency background shape so spectrogram isn't flat noise.
        t = np.arange(n_samples, dtype=np.float32) / float(args.sample_rate)
        audio += (0.015 * np.sin(2.0 * np.pi * 9.0 * t)).astype(np.float32)

        for ev in [e for e in events if e.clip_index == i]:
            s0 = int(round(ev.start_sec * args.sample_rate))
            s1 = int(round(ev.end_sec * args.sample_rate))
            s0 = max(0, min(s0, n_samples))
            s1 = max(s0, min(s1, n_samples))
            if s1 <= s0:
                continue
            call = _generate_call_waveform(
                sample_rate=int(args.sample_rate),
                duration_sec=(s1 - s0) / float(args.sample_rate),
                f0_hz=ev.f0_hz,
                f1_hz=ev.f1_hz,
                amplitude=ev.amplitude,
            )
            audio[s0:s1] += call[: (s1 - s0)]
        # Hard clip to prevent overflow.
        audio = np.clip(audio, -1.0, 1.0)
        clips.append(audio)
        sf.write(str(raw_audio_dir / clip_name), audio, int(args.sample_rate))

    # 2) Build full spectrogram MAT files using edge-padding workflow.
    full_specs: List[Dict[str, Any]] = []
    pad_samples = int(round(float(args.edge_padding) * int(args.sample_rate)))
    for i, clip_audio in enumerate(clips):
        buffer = [clip_audio]
        offset = 0.0
        if i > 0:
            buffer.insert(0, clips[i - 1][-pad_samples:])
            offset = float(args.edge_padding)
        if i < len(clips) - 1:
            buffer.append(clips[i + 1][:pad_samples])
        full_audio = np.concatenate(buffer, axis=0)

        freqs, times, power, pdB = _spectrogram_from_audio(
            full_audio,
            sample_rate=int(args.sample_rate),
            win_dur=float(args.win_dur),
            overlap=float(args.overlap),
        )

        # Trim to current clip bounds (single trim pass).
        t_start = float(offset)
        t_end = t_start + float(args.clip_seconds)
        keep = (times >= t_start) & (times <= t_end)
        if not np.any(keep):
            raise RuntimeError(f"No time bins kept for clip index {i}")
        times = times[keep] - t_start
        power = power[:, keep]
        pdB = pdB[:, keep]

        freqs, power, pdB = _crop_freq(
            freqs, power, pdB, fmin=float(args.freq_min), fmax=float(args.freq_max)
        )

        file_stem = Path(clip_names[i]).stem
        mat_path = full_spec_dir / f"{file_stem}.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "F": freqs.astype(np.float32),
                "T": times.astype(np.float64),
                "P": power.astype(np.float32),
                "PdB_norm": pdB.astype(np.float32),
                "fs": float(args.sample_rate),
            },
        )

        full_specs.append(
            {
                "clip_index": i,
                "file_stem": file_stem,
                "source_audio": clip_names[i],
                "audio_timestamp": clip_datetimes[i].isoformat(),
                "mat_path": str(mat_path.relative_to(output_root)),
                "times": times,
                "freqs": freqs,
                "power": power,
                "pdB": pdB,
            }
        )

    # 3) Export sliding-window crops + synthetic predictions JSON.
    tracker = UnifiedPredictionTracker(device_dir / "predictions.json")
    tracker.set_task_type("whale_detection")
    tracker.set_model_info(
        model_id="synthetic-audit-model",
        architecture="synthetic_score_generator",
        checkpoint_path="synthetic://none",
        trained_at=_iso_now(),
        input_shape=[int(args.crop_size), int(args.crop_size)],
        output_classes=[CLASS_HIERARCHY],
    )
    ds_id = f"synthetic_{args.device_code}_{date_str}"
    tracker.add_data_source(
        data_source_id=ds_id,
        device_code=args.device_code,
        location_name="Synthetic Alignment Test",
        date_from=clip_datetimes[0].isoformat(),
        date_to=(clip_datetimes[-1] + timedelta(seconds=float(args.clip_seconds))).isoformat(),
        sample_rate=float(args.sample_rate),
    )
    tracker.set_spectrogram_config(
        {
            "window_duration": float(args.win_dur),
            "overlap": float(args.overlap),
            "frequency_limits": {"min": float(args.freq_min), "max": float(args.freq_max)},
            "crop_size": int(args.crop_size),
            "edge_padding_sec": float(args.edge_padding),
            "source": {"type": "synthetic", "generator": "generate_synthetic_alignment_audit_dataset.py"},
        }
    )
    tracker.set_pipeline_info(
        pipeline_version="synthetic-audit-v1",
        pipeline_commit=None,
        pipeline_repo="local",
    )

    window_debug: List[Dict[str, Any]] = []
    for spec_entry in full_specs:
        i = int(spec_entry["clip_index"])
        file_stem = str(spec_entry["file_stem"])
        source_audio = str(spec_entry["source_audio"])
        clip_ts = clip_datetimes[i]
        times = np.asarray(spec_entry["times"], dtype=np.float64).ravel()
        freqs = np.asarray(spec_entry["freqs"], dtype=np.float32).ravel()
        power = np.asarray(spec_entry["power"], dtype=np.float32)
        pdB = np.asarray(spec_entry["pdB"], dtype=np.float32)
        raw_audio = clips[i]

        # Frequency dimension to square crop.
        f0_parent = 0
        if power.shape[0] < int(args.crop_size):
            pad_f = int(args.crop_size) - power.shape[0]
            power_f = np.pad(power, ((0, pad_f), (0, 0)), mode="edge")
            pdB_f = np.pad(pdB, ((0, pad_f), (0, 0)), mode="edge")
            freqs_f = np.pad(freqs, (0, pad_f), mode="edge")
        elif power.shape[0] > int(args.crop_size):
            f0 = (power.shape[0] - int(args.crop_size)) // 2
            f1 = f0 + int(args.crop_size)
            f0_parent = int(f0)
            power_f = power[f0:f1, :]
            pdB_f = pdB[f0:f1, :]
            freqs_f = freqs[f0:f1]
        else:
            power_f = power
            pdB_f = pdB
            freqs_f = freqs

        n_time = pdB_f.shape[1]
        starts = _window_starts_even_coverage(
            total_bins=int(n_time),
            crop_bins=int(args.crop_size),
            step_bins=int(args.window_step),
        )

        clip_events = [e for e in events if e.clip_index == i]
        for start_bin in starts:
            end_bin = start_bin + int(args.crop_size)
            if n_time < int(args.crop_size):
                pad_t = int(args.crop_size) - n_time
                crop_power = np.pad(power_f, ((0, 0), (0, pad_t)), mode="edge")
                crop_pdB = np.pad(pdB_f, ((0, 0), (0, pad_t)), mode="edge")
                if len(times) >= 2:
                    dt = float(np.median(np.diff(times)))
                    pad_times = times[-1] + dt * np.arange(1, pad_t + 1, dtype=np.float64)
                    crop_times = np.concatenate([times, pad_times], axis=0)
                else:
                    crop_times = np.linspace(0.0, float(args.clip_seconds), int(args.crop_size), endpoint=False)
            else:
                end_bin = min(end_bin, n_time)
                crop_power = power_f[:, start_bin:end_bin]
                crop_pdB = pdB_f[:, start_bin:end_bin]
                crop_times = times[start_bin:end_bin]
                if crop_pdB.shape[1] < int(args.crop_size):
                    pad_t = int(args.crop_size) - crop_pdB.shape[1]
                    crop_power = np.pad(crop_power, ((0, 0), (0, pad_t)), mode="edge")
                    crop_pdB = np.pad(crop_pdB, ((0, 0), (0, pad_t)), mode="edge")
                    if len(crop_times) >= 2:
                        dt = float(np.median(np.diff(crop_times)))
                        pad_times = crop_times[-1] + dt * np.arange(1, pad_t + 1, dtype=np.float64)
                        crop_times = np.concatenate([crop_times, pad_times], axis=0)

            w_start_sec, w_end_sec = _compute_window_time_range(
                times=times,
                start_idx=int(start_bin),
                window_bins=int(args.crop_size),
                win_dur=float(args.win_dur),
                overlap=float(args.overlap),
            )

            item_id = _safe_id_token(
                f"synth-{args.device_code}-{clip_ts.strftime('%Y%m%dT%H%M%S')}-w{int(start_bin):06d}",
                max_len=180,
            )
            mat_rel = Path("spectrograms") / f"{item_id}.mat"
            wav_rel = Path("audio") / f"{item_id}.wav"
            mat_abs = device_dir / mat_rel
            wav_abs = device_dir / wav_rel

            scipy.io.savemat(
                str(mat_abs),
                {
                    "F": freqs_f.astype(np.float32),
                    "T": np.asarray(crop_times, dtype=np.float64),
                    "P": crop_power.astype(np.float32),
                    "PdB_norm": crop_pdB.astype(np.float32),
                    "parent_freq_bin_start": np.int32(f0_parent),
                    "parent_freq_bin_end": np.int32(min(f0_parent + int(args.crop_size), power.shape[0])),
                    "parent_time_bin_start": np.int32(int(start_bin)),
                    "parent_time_bin_end": np.int32(int(min(start_bin + int(args.crop_size), n_time))),
                },
            )

            s0 = int(max(0.0, w_start_sec) * int(args.sample_rate))
            s1 = int(max(w_end_sec, w_start_sec) * int(args.sample_rate))
            s0 = max(0, min(s0, len(raw_audio)))
            s1 = max(s0, min(s1, len(raw_audio)))
            audio_clip = raw_audio[s0:s1]
            sf.write(str(wav_abs), audio_clip, int(args.sample_rate))

            score = _score_window(
                clip_events=clip_events,
                window_start_sec=float(w_start_sec),
                window_end_sec=float(w_end_sec),
                rng=rng,
            )
            audio_start_dt = clip_ts + timedelta(seconds=float(w_start_sec))
            audio_end_dt = clip_ts + timedelta(seconds=float(w_end_sec))
            tracker.add_item(
                item_id=item_id,
                model_outputs=[{"class_hierarchy": CLASS_HIERARCHY, "score": float(score)}],
                data_source_id=ds_id,
                audio_start_time=audio_start_dt.isoformat(),
                audio_end_time=audio_end_dt.isoformat(),
                audio_path=str(wav_rel),
                spectrogram_mat_path=str(mat_rel),
                source_audio=source_audio,
                segment_start_sec=float(w_start_sec),
                segment_end_sec=float(w_end_sec),
                window_start=int(start_bin),
                window_time_start=float(w_start_sec),
                window_time_end=float(w_end_sec),
                crop_size=int(args.crop_size),
                crop_type="sliding_window",
                crop_applied=True,
                original_shape=[int(power_f.shape[0]), int(n_time)],
                chunk_shape=[int(args.crop_size), int(args.crop_size)],
                parent_spectrogram_mat_path=str(spec_entry["mat_path"]),
                parent_audio_path=str((Path("raw_audio") / source_audio).as_posix()),
                parent_freq_bin_start=int(f0_parent),
                parent_freq_bin_end=int(min(f0_parent + int(args.crop_size), power.shape[0])),
                parent_time_bin_start=int(start_bin),
                parent_time_bin_end=int(min(start_bin + int(args.crop_size), n_time)),
            )
        window_debug.append(
                {
                    "item_id": item_id,
                    "clip_index": i,
                    "window_start_bin": int(start_bin),
                    "window_start_sec": float(w_start_sec),
                    "window_end_sec": float(w_end_sec),
                    "score": float(score),
                }
            )

    tracker.save()

    # Save synthetic truth + metadata
    metadata = {
        "version": "synthetic-audit-v1",
        "created_at": _iso_now(),
        "data_source": {
            "device_code": args.device_code,
            "date_from": clip_datetimes[0].isoformat(),
            "date_to": (clip_datetimes[-1] + timedelta(seconds=float(args.clip_seconds))).isoformat(),
            "sample_rate": float(args.sample_rate),
        },
        "spectrogram_config": {
            "window_duration": float(args.win_dur),
            "overlap": float(args.overlap),
            "frequency_limits": {"min": float(args.freq_min), "max": float(args.freq_max)},
            "crop_size": int(args.crop_size),
            "edge_padding_sec": float(args.edge_padding),
        },
        "files": [
            {
                "file_id": spec["file_stem"],
                "source_audio": spec["source_audio"],
                "audio_timestamp": spec["audio_timestamp"],
                "segment_start_sec": 0.0,
                "segment_end_sec": float(args.clip_seconds),
                "mat_path": spec["mat_path"],
                "raw_audio_path": str((Path("raw_audio") / spec["source_audio"]).as_posix()),
            }
            for spec in full_specs
        ],
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_root / "synthetic_truth_events.json").write_text(
        json.dumps([asdict(e) for e in events], indent=2)
    )
    (device_dir / "window_debug.json").write_text(json.dumps(window_debug, indent=2))

    postprocessed_json = device_dir / "predictions_postprocessed.json"
    if not args.skip_postprocess:
        post_script = REPO_ROOT / "scripts" / "inference" / "postprocess_predictions.py"
        cmd = [
            sys.executable,
            str(post_script),
            "--input-json",
            str(device_dir / "predictions.json"),
            "--output-json",
            str(postprocessed_json),
            "--low-threshold",
            str(args.low_threshold),
            "--high-threshold",
            str(args.high_threshold),
            "--min-members",
            str(args.min_members),
            "--max-gap-seconds",
            str(args.max_gap_seconds),
            "--merge-event-media",
            "--replace-items-with-events",
            "--merge-min-score",
            str(args.merge_min_score),
            "--summary-md",
            str(device_dir / "predictions_postprocessed_summary.md"),
            "--events-csv",
            str(device_dir / "predictions_postprocessed_events.csv"),
        ]
        subprocess.run(cmd, check=True)

    # 4) Numeric audit report
    pred_json = device_dir / "predictions.json"
    pred_obj = json.loads(pred_json.read_text())
    items = pred_obj.get("items", [])
    lag_values: List[float] = []
    for item in items:
        outputs = item.get("model_outputs", [])
        score = None
        if isinstance(outputs, list) and outputs:
            try:
                score = float(outputs[0].get("score"))
            except Exception:
                score = None
        if score is None or score < float(args.lag_min_score):
            continue
        paths = item.get("paths", {})
        mat_rel = paths.get("spectrogram_mat_path")
        aud_rel = paths.get("audio_path")
        if not mat_rel or not aud_rel:
            continue
        lag = _profile_lag_seconds(
            mat_path=(pred_json.parent / mat_rel).resolve(),
            audio_path=(pred_json.parent / aud_rel).resolve(),
            sample_rate=int(args.sample_rate),
            win_dur=float(args.win_dur),
            overlap=float(args.overlap),
            freq_min=float(args.freq_min),
            freq_max=float(args.freq_max),
        )
        if lag is not None and math.isfinite(lag):
            lag_values.append(float(lag))

    seam_scores: List[Tuple[str, float]] = []
    seam_zero_cols = 0
    seam_const_cols = 0
    seam_files_checked = 0
    if postprocessed_json.exists():
        post_obj = json.loads(postprocessed_json.read_text())
        for item in post_obj.get("items", []):
            paths = item.get("paths", {})
            mat_rel = paths.get("spectrogram_mat_path")
            if not mat_rel:
                continue
            mat_abs = (postprocessed_json.parent / mat_rel).resolve()
            score = _seam_artifact_score(mat_abs)
            if score is not None and math.isfinite(score):
                seam_scores.append((str(item.get("item_id")), float(score)))
            zc, cc = _constant_column_counts(mat_abs)
            seam_zero_cols += int(zc)
            seam_const_cols += int(cc)
            seam_files_checked += 1
        seam_scores.sort(key=lambda x: x[1], reverse=True)

    full_spec_diffs: List[float] = []
    for spec in full_specs:
        t = np.asarray(spec["times"], dtype=np.float64).ravel()
        if t.size < 2:
            continue
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            continue
        span = float(t[-1] - t[0]) + float(np.median(dt))
        full_spec_diffs.append(span - float(args.clip_seconds))

    lines: List[str] = []
    lines.append("# Synthetic Alignment Audit")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Device: `{args.device_code}`")
    lines.append(f"- Clips: {int(args.num_clips)} x {float(args.clip_seconds):.1f}s")
    lines.append(f"- Sample rate: {int(args.sample_rate)} Hz")
    lines.append(f"- Spectrogram params: win_dur={float(args.win_dur)}, overlap={float(args.overlap)}")
    lines.append(f"- Frequency range: {float(args.freq_min)}-{float(args.freq_max)} Hz")
    lines.append(f"- Crop/window: {int(args.crop_size)}x{int(args.crop_size)}, step={int(args.window_step)} bins")
    lines.append(f"- Injected synthetic calls: {len(events)}")
    lines.append("")
    lines.append("## Generated Files")
    lines.append(f"- Raw audio: `{raw_audio_dir}`")
    lines.append(f"- Full spectrograms: `{full_spec_dir}`")
    lines.append(f"- Window spectrograms: `{spec_crop_dir}`")
    lines.append(f"- Window audio clips: `{audio_crop_dir}`")
    lines.append(f"- Predictions JSON: `{pred_json}`")
    if postprocessed_json.exists():
        lines.append(f"- Postprocessed JSON: `{postprocessed_json}`")
        lines.append(f"- Event media dir: `{postprocessed_json.parent / (postprocessed_json.stem + '_events_media')}`")
    lines.append("")
    lines.append("## Alignment Diagnostics")
    if lag_values:
        abs_lags = np.abs(np.asarray(lag_values, dtype=np.float64))
        lines.append(
            f"- Window spectrogram/audio lag (from profile cross-correlation, score >= {float(args.lag_min_score):.2f}): "
            f"n={len(lag_values)}, median_abs={float(np.median(abs_lags)):.4f}s, "
            f"p95_abs={float(np.percentile(abs_lags, 95)):.4f}s, max_abs={float(np.max(abs_lags)):.4f}s"
        )
    else:
        lines.append("- Window lag diagnostics unavailable.")
    if full_spec_diffs:
        arr = np.asarray(full_spec_diffs, dtype=np.float64)
        lines.append(
            f"- Full-spectrogram span minus clip duration: "
            f"n={arr.size}, median={float(np.median(arr)):.4f}s, "
            f"max_abs={float(np.max(np.abs(arr))):.4f}s"
        )
    else:
        lines.append("- Full-spectrogram duration diagnostics unavailable.")
    if seam_scores:
        top = seam_scores[:5]
        lines.append("- Top seam discontinuity z-scores in stitched event spectrograms (higher is worse):")
        for eid, score in top:
            lines.append(f"  - `{eid}`: {score:.2f}")
        lines.append(
            f"- Stitched spectrogram constant-column checks: files={seam_files_checked}, "
            f"zero_cols_total={seam_zero_cols}, const_cols_total={seam_const_cols}"
        )
    else:
        lines.append("- Seam diagnostics unavailable or no postprocessed items.")
    lines.append("")
    lines.append("## Verification App")
    lines.append("Use `predictions_postprocessed.json` with:")
    lines.append(f"- spectrogram folder: `{(postprocessed_json.parent / (postprocessed_json.stem + '_events_media') / 'spectrograms') if postprocessed_json.exists() else 'N/A'}`")
    lines.append(f"- audio folder: `{(postprocessed_json.parent / (postprocessed_json.stem + '_events_media') / 'audio') if postprocessed_json.exists() else 'N/A'}`")
    lines.append(f"- predictions file: `{postprocessed_json if postprocessed_json.exists() else pred_json}`")

    report_path = output_root / "synthetic_alignment_audit_report.md"
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Synthetic dataset ready: {output_root}")
    print(f"Report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
