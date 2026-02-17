#!/usr/bin/env python3
"""
Audit window-level spectrogram/audio alignment.

Compares saved MAT power spectrograms (`P`) against power spectrograms
recomputed from paired audio clips using the same FFT params.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io
import scipy.signal
import soundfile as sf


@dataclass
class Row:
    item_id: str
    mae_power: float
    max_abs_power: float
    shape_saved: str
    shape_recomputed: str


def _load_power(mat_path: Path) -> Optional[np.ndarray]:
    data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    power = data.get("P")
    if power is None:
        return None
    arr = np.asarray(power, dtype=np.float64)
    if arr.ndim != 2:
        return None
    return arr


def _compute_power_from_audio(
    audio_path: Path,
    *,
    win_dur: float,
    overlap: float,
    freq_min: float,
    freq_max: float,
) -> np.ndarray:
    wav, sr = sf.read(str(audio_path), always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    win_len = max(1, int(round(float(win_dur) * float(sr))))
    noverlap = int(round(float(overlap) * win_len))
    noverlap = max(0, min(noverlap, win_len - 1))
    window = scipy.signal.get_window("hann", win_len, fftbins=True)
    f, _, sxx = scipy.signal.spectrogram(
        wav,
        fs=sr,
        window=window,
        nperseg=win_len,
        noverlap=noverlap,
        nfft=win_len,
        scaling="density",
        mode="psd",
    )
    mask = (f >= float(freq_min)) & (f <= float(freq_max))
    return np.asarray(sxx[mask, :], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit MAT/audio window alignment using raw power.")
    ap.add_argument("--spec-dir", required=True, type=str, help="Directory with window MAT files")
    ap.add_argument("--audio-dir", required=True, type=str, help="Directory with paired window WAV files")
    ap.add_argument("--sample-count", type=int, default=40, help="Number of matched pairs to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--win-dur", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.9)
    ap.add_argument("--freq-min", type=float, default=5.0)
    ap.add_argument("--freq-max", type=float, default=100.0)
    ap.add_argument("--out-json", type=str, default=None, help="Optional output JSON report")
    args = ap.parse_args()

    spec_dir = Path(args.spec_dir).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    mats = sorted(spec_dir.glob("*.mat"))
    pairs: List[Path] = []
    for mat in mats:
        if (audio_dir / f"{mat.stem}.wav").exists():
            pairs.append(mat)
    if not pairs:
        raise SystemExit("No matched MAT/WAV window pairs found.")

    rng = random.Random(int(args.seed))
    sample_count = min(int(args.sample_count), len(pairs))
    chosen = rng.sample(pairs, sample_count)

    rows: List[Row] = []
    for mat_path in chosen:
        wav_path = audio_dir / f"{mat_path.stem}.wav"
        p_saved = _load_power(mat_path)
        if p_saved is None:
            continue
        p_rec = _compute_power_from_audio(
            wav_path,
            win_dur=float(args.win_dur),
            overlap=float(args.overlap),
            freq_min=float(args.freq_min),
            freq_max=float(args.freq_max),
        )
        h = min(p_saved.shape[0], p_rec.shape[0])
        w = min(p_saved.shape[1], p_rec.shape[1])
        a = p_saved[:h, :w]
        b = p_rec[:h, :w]
        diff = a - b
        rows.append(
            Row(
                item_id=mat_path.stem,
                mae_power=float(np.mean(np.abs(diff))),
                max_abs_power=float(np.max(np.abs(diff))),
                shape_saved=f"{p_saved.shape[0]}x{p_saved.shape[1]}",
                shape_recomputed=f"{p_rec.shape[0]}x{p_rec.shape[1]}",
            )
        )

    if not rows:
        raise SystemExit("No comparable rows found (missing `P` in sampled MAT files?).")

    maes = np.array([r.mae_power for r in rows], dtype=np.float64)
    maxes = np.array([r.max_abs_power for r in rows], dtype=np.float64)
    summary = {
        "spec_dir": str(spec_dir),
        "audio_dir": str(audio_dir),
        "sampled_pairs": int(len(rows)),
        "mae_power_median": float(np.median(maes)),
        "mae_power_p95": float(np.quantile(maes, 0.95)),
        "mae_power_max": float(np.max(maes)),
        "max_abs_power_median": float(np.median(maxes)),
        "max_abs_power_p95": float(np.quantile(maxes, 0.95)),
        "max_abs_power_max": float(np.max(maxes)),
        "rows": [r.__dict__ for r in rows],
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote detailed report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

