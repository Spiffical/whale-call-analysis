#!/usr/bin/env python3
"""Prepare aligned low/mid/high 40s spectrogram MATs from call-window CSVs.

The original multi-species experiments used one train-style 5-100 Hz MAT per
row. This utility keeps the same 40s context convention, but emits three
frequency bands per row so training can fuse low-frequency baleen calls with
mid/high-frequency Mn/Oo evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import soundfile as sf
except Exception:
    sf = None


@dataclass(frozen=True)
class BandConfig:
    name: str
    fmin_hz: float
    fmax_hz: float
    target_sample_rate_hz: int
    window_seconds: float
    hop_seconds: float
    freq_bins: int
    freq_scale: str


DEFAULT_BANDS: Tuple[BandConfig, ...] = (
    BandConfig(
        name="low",
        fmin_hz=5.0,
        fmax_hz=200.0,
        target_sample_rate_hz=512,
        window_seconds=2.0,
        hop_seconds=0.2,
        freq_bins=391,
        freq_scale="linear",
    ),
    BandConfig(
        name="mid",
        fmin_hz=100.0,
        fmax_hz=2000.0,
        target_sample_rate_hz=4096,
        window_seconds=0.5,
        hop_seconds=0.1,
        freq_bins=256,
        freq_scale="log",
    ),
    BandConfig(
        name="high",
        fmin_hz=500.0,
        fmax_hz=32000.0,
        target_sample_rate_hz=64000,
        window_seconds=0.128,
        hop_seconds=0.032,
        freq_bins=256,
        freq_scale="log",
    ),
)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_clip_ts(clip: str) -> Optional[datetime]:
    match = re.search(r"_(\d{8}T\d{6})", str(clip))
    if not match:
        return None
    ts = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    return ts.replace(tzinfo=timezone.utc)


def build_audio_index(audio_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    by_stem: Dict[str, Path] = {}
    by_second: Dict[str, Path] = {}
    pattern = re.compile(r"^(?P<device>[^_]+)_(?P<ts>\d{8}T\d{6})")
    for path in audio_dir.rglob("*"):
        try:
            if not path.is_file() or path.suffix.lower() not in (".flac", ".wav"):
                continue
            by_stem.setdefault(path.stem, path)
            match = pattern.match(path.stem)
            if match:
                by_second.setdefault(f"{match.group('device')}_{match.group('ts')}", path)
        except FileNotFoundError:
            continue
    return by_stem, by_second


def _find_audio_file_with_index(
    audio_dir: Path,
    clip_name: str,
    audio_index: Mapping[str, Path],
    audio_index_by_second: Mapping[str, Path],
) -> Optional[Path]:
    stem = Path(clip_name).stem if clip_name.endswith((".flac", ".wav")) else clip_name
    if stem in audio_index:
        return audio_index[stem]
    match = re.search(r"^(?P<device>[^_]+)_(?P<ts>\d{8}T\d{6})", clip_name)
    if match:
        key = f"{match.group('device')}_{match.group('ts')}"
        if key in audio_index_by_second:
            return audio_index_by_second[key]
    if clip_name.endswith((".flac", ".wav")):
        direct = audio_dir / clip_name
        if direct.exists():
            return direct
    for ext in (".flac", ".wav"):
        direct = audio_dir / f"{clip_name}{ext}"
        if direct.exists():
            return direct
    matches = list(audio_dir.rglob(f"{stem}*.flac")) + list(audio_dir.rglob(f"{stem}*.wav"))
    return matches[0] if matches else None


def _find_adjacent_file_with_index(
    audio_dir: Path,
    device: str,
    ts: datetime,
    audio_index_by_second: Mapping[str, Path],
) -> Optional[Path]:
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    key = f"{device}_{stamp}"
    if key in audio_index_by_second:
        return audio_index_by_second[key]
    for ext in (".flac", ".wav"):
        matches = list(audio_dir.rglob(f"{device}_{stamp}*{ext}"))
        if matches:
            return matches[0]
    return None


def _to_mono_audio(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        return arr.mean(axis=1).astype(np.float32, copy=False)
    return arr.reshape(arr.shape[0], -1).mean(axis=1).astype(np.float32, copy=False)


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    if sf is not None:
        data, sr = sf.read(str(path))
        return _to_mono_audio(data), int(sr)
    if path.suffix.lower() != ".wav":
        raise RuntimeError("soundfile is required for non-WAV audio")
    sr, data = scipy.io.wavfile.read(str(path))
    arr = _to_mono_audio(data)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        arr = arr.astype(np.float32) / float(scale)
    return arr.astype(np.float32, copy=False), int(sr)


def _load_context_audio(
    audio_dir: Path,
    clip: str,
    start_s: float,
    end_s: float,
    context_s: float,
    audio_index: Mapping[str, Path],
    audio_index_by_second: Mapping[str, Path],
) -> Tuple[np.ndarray, int]:
    cur_file = _find_audio_file_with_index(audio_dir, clip, audio_index, audio_index_by_second)
    if cur_file is None:
        raise FileNotFoundError(f"Audio file not found for {clip}")
    data, fs = _read_audio(cur_file)
    total_s = len(data) / fs

    def _slice(arr: np.ndarray, s0: float, s1: float) -> np.ndarray:
        s0 = max(0.0, s0)
        s1 = min(total_s, s1)
        i0 = int(round(s0 * fs))
        i1 = int(round(s1 * fs))
        if i1 <= i0:
            return np.zeros(0, dtype=arr.dtype)
        return arr[i0:i1]

    chunks: List[np.ndarray] = []
    if start_s < 0:
        clip_dt = parse_clip_ts(clip)
        prev_file = (
            _find_adjacent_file_with_index(audio_dir, clip.split("_")[0], clip_dt - timedelta(minutes=5), audio_index_by_second)
            if clip_dt
            else None
        )
        if prev_file is not None and prev_file.exists():
            prev_data, prev_fs = _read_audio(prev_file)
            if prev_fs == fs:
                chunks.append(prev_data[-int(round(-start_s * fs)) :])
            else:
                chunks.append(np.zeros(int(round(-start_s * fs)), dtype=np.float32))
        else:
            chunks.append(np.zeros(int(round(-start_s * fs)), dtype=np.float32))

    chunks.append(_slice(data, start_s, end_s))

    if end_s > total_s:
        clip_dt = parse_clip_ts(clip)
        next_file = (
            _find_adjacent_file_with_index(audio_dir, clip.split("_")[0], clip_dt + timedelta(minutes=5), audio_index_by_second)
            if clip_dt
            else None
        )
        if next_file is not None and next_file.exists():
            next_data, next_fs = _read_audio(next_file)
            if next_fs == fs:
                chunks.append(next_data[: int(round((end_s - total_s) * fs))])
            else:
                chunks.append(np.zeros(int(round((end_s - total_s) * fs)), dtype=np.float32))
        else:
            chunks.append(np.zeros(int(round((end_s - total_s) * fs)), dtype=np.float32))

    full = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    target = int(round(float(context_s) * fs))
    if len(full) > target:
        full = full[:target]
    elif len(full) < target:
        full = np.pad(full, (0, target - len(full)))
    return full.astype(np.float32, copy=False), int(fs)


def power_to_db_norm(power: np.ndarray) -> np.ndarray:
    power = np.abs(np.asarray(power, dtype=np.float32))
    max_power = float(np.max(power)) if power.size else 0.0
    if max_power > 0:
        normalized = np.maximum(power / max_power, 1e-10)
        return 10.0 * np.log10(normalized).astype(np.float32)
    return np.full_like(power, -100.0, dtype=np.float32)


def _target_frequencies(band: BandConfig) -> np.ndarray:
    if band.freq_scale == "log":
        return np.geomspace(float(band.fmin_hz), float(band.fmax_hz), int(band.freq_bins)).astype(np.float32)
    return np.linspace(float(band.fmin_hz), float(band.fmax_hz), int(band.freq_bins), dtype=np.float32)


def _resample_down_if_needed(audio: np.ndarray, source_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    """Downsample to target_sr, but never upsample lower-rate sources."""

    compute_sr = int(min(int(source_sr), int(target_sr)))
    if compute_sr <= 0:
        raise ValueError(f"Invalid sample rate: {source_sr}")
    if int(source_sr) == compute_sr:
        return np.asarray(audio, dtype=np.float32), compute_sr
    divisor = math.gcd(int(source_sr), compute_sr)
    up = compute_sr // divisor
    down = int(source_sr) // divisor
    resampled = scipy.signal.resample_poly(np.asarray(audio, dtype=np.float32), up, down)
    return resampled.astype(np.float32, copy=False), compute_sr


def _empty_db(band: BandConfig, compute_sr: int, context_seconds: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = max(8, int(round(float(band.window_seconds) * compute_sr)))
    hop = max(1, int(round(float(band.hop_seconds) * compute_sr)))
    total_samples = max(nperseg, int(round(float(context_seconds) * compute_sr)))
    frame_count = 1 + max(0, (total_samples - nperseg) // hop)
    times = ((np.arange(frame_count, dtype=np.float32) * hop) + (0.5 * nperseg)) / float(compute_sr)
    freqs = _target_frequencies(band)
    db = np.full((len(freqs), frame_count), -100.0, dtype=np.float32)
    return freqs, times.astype(np.float32), db


def compute_band_spectrogram(
    audio: np.ndarray,
    source_sr: int,
    band: BandConfig,
    *,
    context_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return target-frequency dB spectrogram [freq, time] for one band."""

    source_nyquist = float(source_sr) / 2.0
    compute_audio, compute_sr = _resample_down_if_needed(audio, source_sr, band.target_sample_rate_hz)
    target_freqs = _target_frequencies(band)
    if source_nyquist <= float(band.fmin_hz):
        freqs, times, db = _empty_db(band, compute_sr, context_seconds)
        return freqs, times, db, {
            "source_sample_rate_hz": int(source_sr),
            "compute_sample_rate_hz": int(compute_sr),
            "source_nyquist_hz": source_nyquist,
            "empty_reason": "source_nyquist_below_band_min",
        }

    nperseg = max(8, int(round(float(band.window_seconds) * compute_sr)))
    hop = max(1, int(round(float(band.hop_seconds) * compute_sr)))
    noverlap = max(0, min(nperseg - 1, nperseg - hop))
    if compute_audio.size < nperseg:
        compute_audio = np.pad(compute_audio, (0, nperseg - compute_audio.size))

    freqs, times, power = scipy.signal.spectrogram(
        compute_audio,
        fs=float(compute_sr),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    db = power_to_db_norm(np.asarray(power, dtype=np.float32))
    interp = np.empty((target_freqs.shape[0], db.shape[1]), dtype=np.float32)
    for col_idx in range(db.shape[1]):
        interp[:, col_idx] = np.interp(target_freqs, freqs, db[:, col_idx], left=-100.0, right=-100.0)
    info = {
        "source_sample_rate_hz": int(source_sr),
        "compute_sample_rate_hz": int(compute_sr),
        "source_nyquist_hz": source_nyquist,
        "nperseg": int(nperseg),
        "hop_samples": int(hop),
        "hop_seconds_actual": float(hop / compute_sr),
        "freq_resolution_hz": float(compute_sr / nperseg),
        "empty_reason": "",
    }
    return target_freqs, np.asarray(times, dtype=np.float32), interp, info


def _row_out_stem(row: Mapping[str, Any], row_idx: int) -> str:
    for key in ("expected_mat_name", "out_mat", "mat_path"):
        value = _clean(row.get(key))
        if value:
            return Path(value).stem
    item_id = _clean(row.get("item_id"))
    if item_id:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)
        return safe[:180]
    clip = _clean(row.get("clip")) or _clean(row.get("source_audio")) or f"row{row_idx:08d}"
    begin_s = float(row.get("begin_s") or row.get("begin_time_s") or 0.0)
    end_s = float(row.get("end_s") or row.get("end_time_s") or begin_s)
    digest = hashlib.sha1(f"{clip}|{begin_s:.6f}|{end_s:.6f}|{row_idx}".encode("utf-8")).hexdigest()[:10]
    safe_clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(clip).stem)
    return f"{safe_clip}_{begin_s:.1f}s_{end_s:.1f}s_{digest}"


def _read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def prepare_multiband_windows(
    *,
    calls_csv: Path,
    audio_dir: Path,
    out_dir: Path,
    window_seconds: float,
    n: Optional[int] = None,
    bands: Sequence[BandConfig] = DEFAULT_BANDS,
    combined_mat: bool = False,
) -> Dict[str, Any]:
    rows, input_fieldnames = _read_rows(calls_csv)
    if n is not None and int(n) > 0:
        rows = rows[: int(n)]
    if not rows:
        raise ValueError(f"No rows in {calls_csv}")
    for needed in ("clip", "begin_s", "end_s"):
        if needed not in input_fieldnames:
            raise ValueError(f"{calls_csv} is missing required column {needed!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_index, audio_index_by_second = build_audio_index(audio_dir)
    report_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(tqdm(rows, desc="multiband", unit="row"), start=1):
        clip = _clean(row.get("clip"))
        begin_s = float(row.get("begin_s") or 0.0)
        end_s = float(row.get("end_s") or begin_s)
        padding = (float(window_seconds) - (end_s - begin_s)) / 2.0
        start_s = begin_s - padding
        stop_s = end_s + padding
        out_stem = _row_out_stem(row, row_idx)
        report = dict(row)
        report["multiband_context_seconds"] = float(window_seconds)
        report["multiband_context_start_s"] = f"{start_s:.6f}"
        report["multiband_context_end_s"] = f"{stop_s:.6f}"
        try:
            audio, source_sr = _load_context_audio(
                audio_dir,
                clip,
                start_s,
                stop_s,
                float(window_seconds),
                audio_index,
                audio_index_by_second,
            )
            combined_payload: Dict[str, Any] = {}
            combined_path = out_dir / "multiband" / f"{out_stem}__multiband.mat"
            for band in bands:
                freqs, times, db, info = compute_band_spectrogram(
                    np.asarray(audio, dtype=np.float32),
                    int(source_sr),
                    band,
                    context_seconds=float(window_seconds),
                )
                payload = {
                        "F": freqs.astype(np.float32),
                        "T": times.astype(np.float32),
                        "PdB_norm": db.astype(np.float32),
                        "freq_min": float(band.fmin_hz),
                        "freq_max": float(band.fmax_hz),
                        "freq_scale": band.freq_scale,
                        "window_s": float(window_seconds),
                        "analysis_window_s": float(band.window_seconds),
                        "hop_s": float(band.hop_seconds),
                        "source_sample_rate_hz": int(source_sr),
                        "compute_sample_rate_hz": int(info["compute_sample_rate_hz"]),
                        "source_nyquist_hz": float(info["source_nyquist_hz"]),
                        "empty_reason": str(info.get("empty_reason", "")),
                        "time_axis_reference": "context_window_center",
                        "band_name": band.name,
                }
                if combined_mat:
                    combined_path.parent.mkdir(parents=True, exist_ok=True)
                    for key, value in payload.items():
                        combined_payload[f"{band.name}_{key}"] = value
                    band_path = combined_path
                else:
                    band_dir = out_dir / band.name
                    band_dir.mkdir(parents=True, exist_ok=True)
                    band_path = band_dir / f"{out_stem}__{band.name}.mat"
                    scipy.io.savemat(str(band_path), payload)
                report[f"{band.name}_mat_path"] = str(band_path)
                report[f"{band.name}_shape"] = f"{db.shape[0]}x{db.shape[1]}"
                report[f"{band.name}_compute_sample_rate_hz"] = int(info["compute_sample_rate_hz"])
                report[f"{band.name}_source_nyquist_hz"] = float(info["source_nyquist_hz"])
                report[f"{band.name}_empty_reason"] = str(info.get("empty_reason", ""))
            if combined_mat:
                combined_payload["mat_storage"] = "combined_multiband"
                combined_payload["window_s"] = float(window_seconds)
                scipy.io.savemat(str(combined_path), combined_payload)
                report["mat_path"] = str(combined_path)
            report["status"] = "ok"
            report_rows.append(report)
        except Exception as exc:  # noqa: BLE001 - batch prep should report row-level failures.
            failure = dict(row)
            failure["status"] = "failed"
            failure["error"] = repr(exc)
            failures.append(failure)

    output_fieldnames: List[str] = []
    for source in (input_fieldnames,):
        for field in source:
            if field not in output_fieldnames:
                output_fieldnames.append(field)
    for extra in (
        "multiband_context_seconds",
        "multiband_context_start_s",
        "multiband_context_end_s",
        "mat_path",
        *[f"{band.name}_mat_path" for band in bands],
        *[f"{band.name}_shape" for band in bands],
        *[f"{band.name}_compute_sample_rate_hz" for band in bands],
        *[f"{band.name}_source_nyquist_hz" for band in bands],
        *[f"{band.name}_empty_reason" for band in bands],
        "status",
        "error",
    ):
        if extra not in output_fieldnames:
            output_fieldnames.append(extra)

    report_csv = out_dir / "multiband_report.csv"
    _write_rows(report_csv, report_rows, output_fieldnames)
    failure_csv = out_dir / "multiband_failures.csv"
    if failures:
        _write_rows(failure_csv, failures, output_fieldnames)
    summary = {
        "calls_csv": str(calls_csv),
        "audio_dir": str(audio_dir),
        "out_dir": str(out_dir),
        "input_rows": len(rows),
        "prepared_rows": len(report_rows),
        "failed_rows": len(failures),
        "report_csv": str(report_csv),
        "failure_csv": str(failure_csv) if failures else "",
        "bands": [band.__dict__ for band in bands],
        "combined_mat": bool(combined_mat),
    }
    (out_dir / "multiband_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        examples = "\n".join(f"{_clean(row.get('clip'))}: {row.get('error')}" for row in failures[:10])
        raise RuntimeError(f"{len(failures)} rows failed during multiband prep; examples:\n{examples}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls-csv", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-s", type=float, default=40.0)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--combined-mat", action="store_true", help="Store all bands for each row in one MAT file.")
    args = parser.parse_args()
    summary = prepare_multiband_windows(
        calls_csv=Path(args.calls_csv),
        audio_dir=Path(args.audio_dir),
        out_dir=Path(args.out_dir),
        window_seconds=float(args.window_s),
        n=args.n,
        combined_mat=bool(args.combined_mat),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
