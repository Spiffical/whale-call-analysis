#!/usr/bin/env python3
"""
Prepare inference crops with the same preprocessing logic used for training.

Pipeline per segment:
1. Select a fixed-duration segment on each 5-minute clip timeline.
2. Add edge context on both sides (for FFT stability), using adjacent clips when needed.
3. Compute spectrogram from the extended audio.
4. Crop frequency range.
5. Trim edge context back to the segment duration.
6. Extract square crops (e.g., 96x96) along time for inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io
import soundfile as sf
import yaml
from dotenv import load_dotenv

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sequential_prep import (
    parse_datetime,
    extract_timestamp_from_filename,
    compute_window_positions,
    crop_to_freq_lims,
    load_dataset_documentation,
    get_processing_params,
)
from src.dataset.reporting import print_header, print_status


def _parse_freq_lims_arg(raw: Optional[str]) -> Optional[Tuple[float, float]]:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--freq-lims must be in 'min,max' format")
    lo = float(parts[0])
    hi = float(parts[1])
    if lo >= hi:
        raise ValueError("--freq-lims requires min < max")
    return lo, hi


def _resolve_square_crop_size(raw_crop_size: object) -> int:
    if raw_crop_size is None:
        raise ValueError("crop_size is missing; pass --crop-size or provide a checkpoint/metadata with crop_size.")
    if isinstance(raw_crop_size, (int, np.integer)):
        return int(raw_crop_size)
    if isinstance(raw_crop_size, (float, np.floating)):
        return int(raw_crop_size)
    if isinstance(raw_crop_size, str):
        if "," in raw_crop_size:
            a, b = [int(p.strip()) for p in raw_crop_size.split(",")]
            if a != b:
                raise ValueError(f"Non-square crop_size in model args: {raw_crop_size}")
            return int(a)
        return int(raw_crop_size)
    if isinstance(raw_crop_size, (list, tuple)) and len(raw_crop_size) == 2:
        a = int(raw_crop_size[0])
        b = int(raw_crop_size[1])
        if a != b:
            raise ValueError(f"Non-square crop_size in model args: {raw_crop_size}")
        return a
    raise ValueError(f"Unsupported crop_size value: {raw_crop_size}")


def _get_nested_float(d: Dict, keys: Sequence[str]) -> Optional[float]:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _audio_sort_key(path: Path) -> Tuple[int, str, str]:
    ts = extract_timestamp_from_filename(path.name)
    if ts is None:
        return 1, "", path.name
    return 0, ts.isoformat(), path.name


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, fs = sf.read(str(path))
    if data.ndim == 1:
        return data, int(fs)
    # Keep channel 0 for deterministic behavior and compatibility with 1-D spectrogram code paths.
    return data[:, 0], int(fs)


def _zeros_like(samples: int, ref: np.ndarray) -> np.ndarray:
    n = max(0, int(samples))
    return np.zeros(n, dtype=ref.dtype)


def _extract_stitched_segment(
    current_data: np.ndarray,
    fs: int,
    desired_start_s: float,
    desired_end_s: float,
    expected_duration_s: float,
    prev_data: Optional[np.ndarray],
    prev_fs: Optional[int],
    next_data: Optional[np.ndarray],
    next_fs: Optional[int],
) -> np.ndarray:
    clip_duration = len(current_data) / float(fs)
    pieces: List[np.ndarray] = []

    # Prefix from previous clip or zeros
    if desired_start_s < 0:
        need_samples = int(round(abs(desired_start_s) * fs))
        if prev_data is not None and prev_fs == fs and need_samples > 0:
            take = min(need_samples, len(prev_data))
            pieces.append(prev_data[-take:])
            if take < need_samples:
                pieces.insert(0, _zeros_like(need_samples - take, current_data))
        else:
            pieces.append(_zeros_like(need_samples, current_data))

    # Main slice
    main_start = max(0.0, desired_start_s)
    main_end = min(clip_duration, desired_end_s)
    i0 = int(round(main_start * fs))
    i1 = int(round(main_end * fs))
    i0 = max(0, min(i0, len(current_data)))
    i1 = max(i0, min(i1, len(current_data)))
    pieces.append(current_data[i0:i1])

    # Suffix from next clip or zeros
    if desired_end_s > clip_duration:
        need_samples = int(round((desired_end_s - clip_duration) * fs))
        if next_data is not None and next_fs == fs and need_samples > 0:
            take = min(need_samples, len(next_data))
            pieces.append(next_data[:take])
            if take < need_samples:
                pieces.append(_zeros_like(need_samples - take, current_data))
        else:
            pieces.append(_zeros_like(need_samples, current_data))

    out = np.concatenate(pieces) if pieces else np.zeros(0, dtype=current_data.dtype)
    expected_samples = int(round(expected_duration_s * fs))
    if len(out) > expected_samples:
        out = out[:expected_samples]
    elif len(out) < expected_samples:
        out = np.pad(out, (0, expected_samples - len(out)))
    return out


def _trim_edge_context(
    times: np.ndarray,
    power: np.ndarray,
    db: np.ndarray,
    edge_context_s: float,
    segment_duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edge_context_s <= 0:
        return times, power, db

    t_start = float(edge_context_s)
    t_end = t_start + float(segment_duration_s)
    mask = (times >= t_start) & (times <= t_end)

    if not np.any(mask):
        t0 = int(np.searchsorted(times, t_start, side="left"))
        t1 = int(np.searchsorted(times, t_end, side="right"))
        t0 = max(0, min(t0, max(0, len(times) - 1)))
        t1 = max(t0 + 1, min(t1, len(times)))
        mask = np.zeros_like(times, dtype=bool)
        mask[t0:t1] = True

    times_trim = times[mask] - t_start
    power_trim = power[:, mask]
    db_trim = db[:, mask]
    return times_trim, power_trim, db_trim


def _match_frequency_dimension(
    freqs: np.ndarray,
    power: np.ndarray,
    db: np.ndarray,
    target_f: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_freq = power.shape[0]
    if n_freq == target_f:
        return freqs, power, db
    if n_freq > target_f:
        start_f = (n_freq - target_f) // 2
        end_f = start_f + target_f
        return freqs[start_f:end_f], power[start_f:end_f, :], db[start_f:end_f, :]

    pad_f = target_f - n_freq
    power_pad = np.pad(power, ((0, pad_f), (0, 0)), mode="edge")
    db_pad = np.pad(db, ((0, pad_f), (0, 0)), mode="edge")
    freqs_pad = np.pad(freqs, (0, pad_f), mode="edge")
    return freqs_pad, power_pad, db_pad


def _compute_segment_starts(
    clip_duration_s: float,
    segment_duration_s: float,
    segment_step_s: Optional[float],
) -> List[float]:
    if clip_duration_s <= segment_duration_s:
        return [0.0]

    max_start = clip_duration_s - segment_duration_s
    if segment_step_s is not None:
        if segment_step_s <= 0:
            raise ValueError("--segment-step must be > 0")
        starts = np.arange(0.0, max_start + 1e-9, float(segment_step_s), dtype=np.float64).tolist()
        if not starts:
            starts = [0.0]
        if starts[-1] < max_start - 1e-6:
            starts.append(max_start)
        return [float(s) for s in starts]

    # Minimum overlap tiling that covers the full clip.
    n_segments = int(math.ceil(clip_duration_s / segment_duration_s))
    if n_segments <= 1:
        return [0.0]
    even_step = max_start / float(n_segments - 1)
    return [float(i * even_step) for i in range(n_segments)]


def _compute_crop_starts(total_bins: int, crop_size: int, step_bins: Optional[int]) -> List[int]:
    if total_bins <= crop_size:
        return [0]
    if step_bins is None:
        return compute_window_positions(total_bins, crop_size)
    if step_bins <= 0:
        raise ValueError("--crop-step-bins must be > 0")
    starts = list(range(0, total_bins - crop_size + 1, int(step_bins)))
    if not starts:
        starts = [0]
    last = total_bins - crop_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _crop_time_window(
    power: np.ndarray,
    db: np.ndarray,
    times: np.ndarray,
    start_bin: int,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_bins = power.shape[1]
    if total_bins < crop_size:
        pad = crop_size - total_bins
        power_out = np.pad(power, ((0, 0), (0, pad)), mode="edge")
        db_out = np.pad(db, ((0, 0), (0, pad)), mode="edge")
        times_out = times
        if len(times) > 1:
            hop = float(times[1] - times[0])
            extra = times[-1] + hop * np.arange(1, pad + 1)
            times_out = np.concatenate([times, extra])
        elif len(times) == 1:
            times_out = np.pad(times, (0, pad), mode="edge")
        else:
            times_out = np.zeros(crop_size, dtype=np.float32)
        return power_out[:, :crop_size], db_out[:, :crop_size], times_out[:crop_size]

    start = max(0, min(int(start_bin), total_bins - crop_size))
    end = start + crop_size
    return power[:, start:end], db[:, start:end], times[start:end]


def _compute_window_seconds(
    crop_times: np.ndarray,
    win_dur_s: float,
) -> Tuple[float, float]:
    if len(crop_times) == 0:
        return 0.0, float(win_dur_s)
    center_start = float(crop_times[0])
    hop = float(crop_times[1] - crop_times[0]) if len(crop_times) > 1 else 0.0
    window_start = max(0.0, center_start - (float(win_dur_s) / 2.0))
    window_end = window_start + max(0, len(crop_times) - 1) * hop + float(win_dur_s)
    return window_start, window_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train-style test crops for inference")
    parser.add_argument("--device-code", type=str, required=True, help="ONC device code")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (ISO, UTC)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (ISO, UTC)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--dataset-documentation", type=str, default=None,
                        help="Path to dataset_documentation.json or parent directory")
    parser.add_argument("--config", type=str, default="./config/dataset_config.yaml",
                        help="Fallback config path")

    # Overrides
    parser.add_argument("--crop-size", type=int, default=None, help="Override model crop size")
    parser.add_argument("--freq-lims", type=str, default=None, help="Override frequency limits as min,max")
    parser.add_argument("--win-dur", type=float, default=None, help="Override spectrogram window duration (s)")
    parser.add_argument("--overlap", type=float, default=None, help="Override overlap ratio")
    parser.add_argument("--context-duration", type=float, default=None,
                        help="Override training context duration (s), used as default segment duration")
    parser.add_argument("--edge-context", type=float, default=None,
                        help="Override edge context (s) added on both sides before spectrogram")
    parser.add_argument("--segment-duration", type=float, default=None,
                        help="Target segment duration (s). Defaults to context-duration.")
    parser.add_argument("--segment-step", type=float, default=None,
                        help="Segment step in seconds. Default: minimal-overlap tiling per clip.")
    parser.add_argument("--crop-step-bins", type=int, default=None,
                        help="Time-bin step for crops inside each segment. Default: minimal-overlap tiling.")

    # Execution options
    parser.add_argument("--spec-backend", type=str, default="auto", choices=["auto", "scipy", "torch"])
    parser.add_argument("--workers", type=int, default=4, help="Downloader workers (if supported)")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing raw audio under output-dir/raw_audio")
    parser.add_argument("--cleanup-raw-audio", action="store_true",
                        help="Delete output-dir/raw_audio after processing")
    parser.add_argument("--save-crop-audio", action="store_true", help="Save WAV audio for each spectrogram crop")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of clips for smoke tests")

    args = parser.parse_args()

    load_dotenv()
    onc_token = os.getenv("ONC_TOKEN")
    if not onc_token and not args.skip_download:
        raise SystemExit("ONC_TOKEN is required unless --skip-download is used.")

    try:
        from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator
        from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader
    except Exception as e:
        raise SystemExit(
            "Failed to import onc_hydrophone_data spectrogram components. "
            "This usually means torch/torchaudio versions are incompatible.\n"
            f"Import error: {e}"
        )

    start_dt = parse_datetime(args.start_date)
    end_dt = parse_datetime(args.end_date)
    if end_dt <= start_dt:
        raise SystemExit("--end-date must be later than --start-date")

    freq_lims_override = _parse_freq_lims_arg(args.freq_lims)

    dataset_doc: Dict = {}
    if args.dataset_documentation:
        dataset_doc = load_dataset_documentation(args.dataset_documentation)
        if not dataset_doc:
            print_status(f"Could not load dataset documentation from {args.dataset_documentation}. Using defaults/fallbacks.", "WARNING")

    config_data = _load_yaml(Path(args.config))
    temporal_cfg = (config_data.get("temporal_context") or {}) if isinstance(config_data, dict) else {}
    custom_spec_cfg = (config_data.get("custom_spectrograms") or {}) if isinstance(config_data, dict) else {}

    proc = get_processing_params(
        dataset_doc=dataset_doc,
        model_path=args.checkpoint,
        crop_size_override=args.crop_size,
        freq_lims_override=freq_lims_override,
        win_dur_override=args.win_dur,
        overlap_override=args.overlap,
    )

    crop_size = _resolve_square_crop_size(proc.get("crop_size"))
    freq_lims = tuple(proc.get("freq_lims", (5.0, 100.0)))
    win_dur = float(proc.get("win_dur", 1.0))
    overlap = float(proc.get("overlap", 0.9))
    clim = tuple(proc.get("clim", (-40.0, 0.0)))

    doc_context = _get_nested_float(dataset_doc, ["processing_parameters", "temporal_context", "context_duration_s"])
    doc_edge = _get_nested_float(dataset_doc, ["processing_parameters", "temporal_context", "edge_context_s"])
    cfg_context = None
    try:
        cfg_context = float(temporal_cfg.get("context_duration"))
    except (TypeError, ValueError):
        cfg_context = None

    context_duration = args.context_duration
    if context_duration is None:
        context_duration = doc_context if doc_context is not None else cfg_context
    if context_duration is None:
        context_duration = 40.0

    edge_context = args.edge_context
    if edge_context is None:
        edge_context = doc_edge if doc_edge is not None else 2.0

    segment_duration = float(args.segment_duration) if args.segment_duration is not None else float(context_duration)
    segment_step = float(args.segment_step) if args.segment_step is not None else None

    if segment_duration <= 0:
        raise SystemExit("--segment-duration must be > 0")
    if edge_context < 0:
        raise SystemExit("--edge-context must be >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_audio_dir = output_dir / "raw_audio"
    raw_audio_dir.mkdir(parents=True, exist_ok=True)

    print_header("PREPARING TRAIN-STYLE TEST CROPS")
    print(f"Device: {args.device_code}")
    print(f"Date range: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"crop_size: {crop_size}")
    print(f"freq_lims: {freq_lims[0]}-{freq_lims[1]} Hz")
    print(f"win_dur: {win_dur}s | overlap: {overlap}")
    print(f"context_duration: {context_duration}s | edge_context: {edge_context}s")
    print(f"segment_duration: {segment_duration}s | segment_step: {segment_step if segment_step is not None else 'auto(min-overlap)'}")
    print(f"crop_step_bins: {args.crop_step_bins if args.crop_step_bins is not None else 'auto(min-overlap)'}")

    if not args.skip_download:
        print_header("PHASE 1: DOWNLOADING AUDIO")
        downloader = HydrophoneDownloader(onc_token, str(raw_audio_dir))
        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        try:
            downloader.download_flac_files(
                args.device_code,
                start_str,
                end_str,
                max_download_workers=max(1, int(args.workers)),
            )
        except Exception as e:
            print_status(f"Downloader raised an error: {e}", "WARNING")

    audio_files = sorted(
        list(raw_audio_dir.glob("**/*.flac")) + list(raw_audio_dir.glob("**/*.wav")),
        key=_audio_sort_key,
    )
    if args.max_files is not None:
        audio_files = audio_files[:max(0, int(args.max_files))]

    if not audio_files:
        raise SystemExit(f"No audio files found in {raw_audio_dir}")

    print_status(f"Found {len(audio_files)} audio clips to process.", "SUCCESS")

    spec_gen = SpectrogramGenerator(
        win_dur=win_dur,
        overlap=overlap,
        freq_lims=freq_lims,
        clim=clim,
        log_freq=bool(custom_spec_cfg.get("log_frequency", False)),
        colormap=custom_spec_cfg.get("colormap", "viridis"),
        crop_freq_lims=False,
        backend=args.spec_backend,
        quiet=True,
    )

    print_header("PHASE 2: SEGMENT + CROP GENERATION")
    files_meta: List[Dict] = []
    total_segments = 0
    total_crops = 0

    for clip_idx, audio_path in enumerate(audio_files):
        print_status(f"Clip {clip_idx + 1}/{len(audio_files)}: {audio_path.name}", "PROGRESS")
        try:
            current_data, fs = _read_audio(audio_path)
        except Exception as e:
            print_status(f"Failed to read {audio_path}: {e}", "WARNING")
            continue

        prev_data = None
        prev_fs = None
        if clip_idx > 0:
            try:
                prev_data, prev_fs = _read_audio(audio_files[clip_idx - 1])
            except Exception:
                prev_data, prev_fs = None, None

        next_data = None
        next_fs = None
        if clip_idx + 1 < len(audio_files):
            try:
                next_data, next_fs = _read_audio(audio_files[clip_idx + 1])
            except Exception:
                next_data, next_fs = None, None

        clip_duration = len(current_data) / float(fs)
        segment_starts = _compute_segment_starts(clip_duration, segment_duration, segment_step)
        total_segments += len(segment_starts)

        clip_ts = extract_timestamp_from_filename(audio_path.name)
        if clip_ts is None:
            clip_ts = datetime.fromtimestamp(audio_path.stat().st_mtime, tz=timezone.utc)
        date_str = clip_ts.strftime("%Y-%m-%d")

        spec_dir = output_dir / date_str / args.device_code / "spectrograms"
        spec_dir.mkdir(parents=True, exist_ok=True)
        crop_audio_dir = output_dir / date_str / args.device_code / "audio"
        if args.save_crop_audio:
            crop_audio_dir.mkdir(parents=True, exist_ok=True)

        for seg_idx, seg_start in enumerate(segment_starts):
            seg_end = seg_start + segment_duration
            ext_start = seg_start - edge_context
            ext_end = seg_end + edge_context
            ext_duration = segment_duration + (2.0 * edge_context)

            seg_audio = _extract_stitched_segment(
                current_data=current_data,
                fs=fs,
                desired_start_s=ext_start,
                desired_end_s=ext_end,
                expected_duration_s=ext_duration,
                prev_data=prev_data,
                prev_fs=prev_fs,
                next_data=next_data,
                next_fs=next_fs,
            )

            try:
                freqs, times, power, db = spec_gen.compute_spectrogram(seg_audio, fs)
            except Exception as e:
                print_status(f"Spectrogram failed for {audio_path.name} seg#{seg_idx}: {e}", "WARNING")
                continue

            freqs_c, power_c = crop_to_freq_lims(freqs, power, freq_lims[0], freq_lims[1])
            _, db_c = crop_to_freq_lims(freqs, db, freq_lims[0], freq_lims[1])
            times_t, power_t, db_t = _trim_edge_context(times, power_c, db_c, edge_context, segment_duration)
            if power_t.size == 0 or power_t.shape[1] == 0:
                print_status(f"Empty segment after trim for {audio_path.name} seg#{seg_idx}; skipping.", "WARNING")
                continue

            freqs_f, power_f, db_f = _match_frequency_dimension(freqs_c, power_t, db_t, crop_size)
            n_freq, n_time = power_f.shape
            crop_starts = _compute_crop_starts(n_time, crop_size, args.crop_step_bins)

            for win_idx, start_bin in enumerate(crop_starts):
                power_crop, db_crop, times_crop = _crop_time_window(power_f, db_f, times_t, start_bin, crop_size)
                crop_id = f"{audio_path.stem}_s{seg_idx:03d}_w{win_idx:03d}"
                mat_path = spec_dir / f"{crop_id}.mat"

                scipy.io.savemat(
                    str(mat_path),
                    {
                        "F": freqs_f[:crop_size],
                        "T": times_crop,
                        "P": power_crop[:crop_size, :crop_size],
                        "PdB_norm": db_crop[:crop_size, :crop_size],
                        "fs": float(fs),
                        "segment_start_sec": float(seg_start),
                        "segment_end_sec": float(seg_end),
                        "window_start_bin": int(start_bin),
                    },
                )

                rel_win_start, rel_win_end = _compute_window_seconds(times_crop, win_dur)
                abs_win_start = float(seg_start + rel_win_start)
                abs_win_end = float(seg_start + rel_win_end)
                abs_win_start = max(0.0, min(abs_win_start, clip_duration))
                abs_win_end = max(abs_win_start, min(abs_win_end, clip_duration))

                crop_audio_rel = None
                if args.save_crop_audio:
                    i0 = int(round(abs_win_start * fs))
                    i1 = int(round(abs_win_end * fs))
                    i0 = max(0, min(i0, len(current_data)))
                    i1 = max(i0, min(i1, len(current_data)))
                    crop_audio = current_data[i0:i1]
                    crop_audio_path = crop_audio_dir / f"{crop_id}.wav"
                    sf.write(str(crop_audio_path), crop_audio, fs)
                    crop_audio_rel = str(crop_audio_path.relative_to(output_dir))

                file_timestamp = clip_ts + timedelta(seconds=abs_win_start)
                files_meta.append(
                    {
                        "file_id": crop_id,
                        "source_audio": audio_path.name,
                        "audio_timestamp": file_timestamp.isoformat(),
                        "date": date_str,
                        "mat_path": str(mat_path.relative_to(output_dir)),
                        "audio_path": crop_audio_rel,
                        "raw_audio_path": str(audio_path.relative_to(output_dir)),
                        "segment_index": int(seg_idx),
                        "segment_start_sec": float(seg_start),
                        "segment_end_sec": float(seg_end),
                        "window_index": int(win_idx),
                        "window_start": int(start_bin),
                        "window_time_start": abs_win_start,
                        "window_time_end": abs_win_end,
                        "chunk_shape": [int(crop_size), int(crop_size)],
                        "original_shape": [int(n_freq), int(n_time)],
                        "sample_rate": float(fs),
                    }
                )
                total_crops += 1

    metadata = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": {
            "device_code": args.device_code,
            "date_from": args.start_date,
            "date_to": args.end_date,
        },
        "spectrogram_config": {
            "window_duration": win_dur,
            "overlap": overlap,
            "frequency_limits": {"min": freq_lims[0], "max": freq_lims[1]},
            "color_limits": {"min": clim[0], "max": clim[1]},
            "crop_size": crop_size,
            "context_duration": float(context_duration),
            "edge_context": float(edge_context),
            "source": {
                "type": "computed",
                "generator": "onc_hydrophone_data.SpectrogramGenerator",
                "pipeline": "train_style_segment_crops",
            },
        },
        "processing": {
            "segment_duration_s": float(segment_duration),
            "segment_step_s": float(segment_step) if segment_step is not None else None,
            "crop_step_bins": int(args.crop_step_bins) if args.crop_step_bins is not None else None,
            "spec_backend": args.spec_backend,
            "save_crop_audio": bool(args.save_crop_audio),
            "total_clips": len(audio_files),
            "total_segments": int(total_segments),
            "total_crops": int(total_crops),
        },
        "dataset_documentation_source": args.dataset_documentation,
        "checkpoint": str(args.checkpoint),
        "files": files_meta,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if args.cleanup_raw_audio:
        print_status("Cleaning up raw audio cache...", "PROGRESS")
        shutil.rmtree(raw_audio_dir, ignore_errors=True)

    print_header("PREPARATION COMPLETE")
    print(f"Clips processed: {len(audio_files)}")
    print(f"Segments generated: {total_segments}")
    print(f"Crops generated: {total_crops}")
    print_status(f"Metadata written: {metadata_path}", "SUCCESS")


if __name__ == "__main__":
    main()
