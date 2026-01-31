#!/usr/bin/env python3
"""
Generate a small spectrogram and report which backend was actually used.

Default behavior prefers the PyTorch backend if torch is importable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer local onc-hydrophone-data repo if present
ONC_REPO = Path("/home/sbialek/ONC/onc-hydrophone-data")
if ONC_REPO.exists() and str(ONC_REPO) not in sys.path:
    sys.path.insert(0, str(ONC_REPO))

from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _load_dataset_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with cfg_path.open("r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    cs = data.get("custom_spectrograms", {}) or {}
    freq = cs.get("frequency_limits", {}) or {}
    clim = cs.get("color_limits", {}) or {}
    return {
        "win_dur": cs.get("window_duration"),
        "overlap": cs.get("overlap"),
        "freq_min": freq.get("min"),
        "freq_max": freq.get("max"),
        "clim_min": clim.get("min"),
        "clim_max": clim.get("max"),
        "log_freq": cs.get("log_frequency"),
        "colormap": cs.get("colormap"),
    }


def _resolve_params(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "win_dur": 1.0,
        "overlap": 0.9,
        "freq_min": 5.0,
        "freq_max": 100.0,
        "clim_min": -60.0,
        "clim_max": 0.0,
        "log_freq": False,
        "colormap": "turbo",
    }
    cfg = _load_dataset_config(args.dataset_config)
    win_dur = args.win_dur if args.win_dur is not None else cfg.get("win_dur", defaults["win_dur"])
    overlap = args.overlap if args.overlap is not None else cfg.get("overlap", defaults["overlap"])
    freq_min = args.freq_min if args.freq_min is not None else cfg.get("freq_min", defaults["freq_min"])
    freq_max = args.freq_max if args.freq_max is not None else cfg.get("freq_max", defaults["freq_max"])
    clim_min = args.clim_min if args.clim_min is not None else cfg.get("clim_min", defaults["clim_min"])
    clim_max = args.clim_max if args.clim_max is not None else cfg.get("clim_max", defaults["clim_max"])
    log_freq = cfg.get("log_freq", defaults["log_freq"])
    colormap = cfg.get("colormap", defaults["colormap"])
    return {
        "win_dur": float(win_dur),
        "overlap": float(overlap),
        "freq_lims": (float(freq_min), float(freq_max)),
        "clim": (float(clim_min), float(clim_max)),
        "log_freq": bool(log_freq),
        "colormap": str(colormap),
    }


def _load_audio(path: Path, max_seconds: Optional[float]) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    data, fs = sf.read(str(path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if max_seconds is not None:
        n = int(round(max_seconds * fs))
        if n > 0:
            data = data[:n]
    return data.astype(np.float32, copy=False), int(fs)


def _synthetic_audio(duration_s: float, sample_rate: int, tone_hz: float, noise: float) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    audio = np.sin(2.0 * np.pi * float(tone_hz) * t)
    if noise > 0:
        audio = audio + (noise * np.random.normal(0.0, 1.0, size=n).astype(np.float32))
    return audio.astype(np.float32, copy=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a spectrogram and report backend used")
    ap.add_argument("--audio", type=str, default=None, help="Optional audio file path to test")
    ap.add_argument("--max-seconds", type=float, default=20.0, help="Max seconds to use from audio file")
    ap.add_argument("--duration-s", type=float, default=10.0, help="Duration for synthetic audio")
    ap.add_argument("--sample-rate", type=int, default=200, help="Sample rate for synthetic audio")
    ap.add_argument("--tone-hz", type=float, default=20.0, help="Tone frequency for synthetic audio")
    ap.add_argument("--noise", type=float, default=0.02, help="Noise level for synthetic audio")
    ap.add_argument("--backend", type=str, default=None, choices=["auto", "torch", "scipy"],
                    help="Backend to request (default: torch if installed, else scipy)")
    ap.add_argument("--torch-device", type=str, default="auto",
                    help="Torch device for spectrogram (cpu, cuda, auto)")
    ap.add_argument("--dataset-config", type=str, default="config/dataset_config.yaml",
                    help="Path to dataset_config.yaml for spectrogram params")
    ap.add_argument("--win-dur", type=float, default=None, help="Override window duration (s)")
    ap.add_argument("--overlap", type=float, default=None, help="Override overlap ratio")
    ap.add_argument("--freq-min", type=float, default=None, help="Override min frequency (Hz)")
    ap.add_argument("--freq-max", type=float, default=None, help="Override max frequency (Hz)")
    ap.add_argument("--clim-min", type=float, default=None, help="Override color min (dB)")
    ap.add_argument("--clim-max", type=float, default=None, help="Override color max (dB)")
    ap.add_argument("--save", action="store_true", help="Save PNG/MAT outputs")
    ap.add_argument("--out-dir", type=str, default="output/backend_test", help="Output directory if saving")
    args = ap.parse_args()

    torch_ok = _try_import("torch")
    torchaudio_ok = _try_import("torchaudio")
    if args.backend is None:
        backend_requested = "torch" if torch_ok else "scipy"
    else:
        backend_requested = args.backend

    params = _resolve_params(args)
    spec_gen = SpectrogramGenerator(
        win_dur=params["win_dur"],
        overlap=params["overlap"],
        freq_lims=params["freq_lims"],
        clim=params["clim"],
        log_freq=params["log_freq"],
        colormap=params["colormap"],
        crop_freq_lims=False,
        backend=backend_requested,
        torch_device=args.torch_device,
        quiet=True,
    )

    audio_path = Path(args.audio).expanduser().resolve() if args.audio else None
    if audio_path:
        if not audio_path.exists():
            print(f"ERROR: audio file not found: {audio_path}")
            return 1
        audio, fs = _load_audio(audio_path, args.max_seconds)
        audio_source = str(audio_path)
    else:
        audio = _synthetic_audio(args.duration_s, args.sample_rate, args.tone_hz, args.noise)
        fs = int(args.sample_rate)
        audio_source = f"synthetic tone {args.tone_hz}Hz, {args.duration_s}s @ {fs}Hz"

    duration_s = float(len(audio) / fs) if fs else 0.0
    freqs, times, Sxx, PdB = spec_gen.compute_spectrogram(audio, fs)
    backend_used = getattr(spec_gen, "_last_backend", None)
    device_used = getattr(spec_gen, "_last_device", None)

    print("SPECTROGRAM BACKEND CHECK")
    print(f"  requested_backend: {backend_requested}")
    print(f"  torch_import: {'ok' if torch_ok else 'missing'}")
    print(f"  torchaudio_import: {'ok' if torchaudio_ok else 'missing'}")
    print(f"  backend_used: {backend_used}")
    if backend_used and backend_used != backend_requested:
        print(f"  backend_fallback: {backend_requested} -> {backend_used}")
    if backend_used == "torch":
        print(f"  torch_device_used: {device_used}")
    print(f"  audio_source: {audio_source}")
    print(f"  audio_duration_s: {duration_s:.2f}")
    print(f"  sample_rate_hz: {fs}")
    print(f"  spectrogram_shape: {Sxx.shape[0]} x {Sxx.shape[1]}")

    if args.save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = "backend_test"
        if audio_path:
            base = audio_path.stem + "_backend_test"
        png_path = out_dir / f"{base}.png"
        mat_path = out_dir / f"{base}.mat"
        metadata = {
            "backend_requested": backend_requested,
            "backend_used": backend_used,
            "torch_device_requested": args.torch_device,
            "torch_device_used": device_used,
            "audio_source": audio_source,
            "sample_rate_hz": fs,
        }
        spec_gen.plot_spectrogram(freqs, times, PdB, title="Backend Test Spectrogram", save_path=png_path)
        spec_gen.save_matlab_format(freqs, times, Sxx, PdB, mat_path, metadata=metadata)
        print(f"  saved_png: {png_path}")
        print(f"  saved_mat: {mat_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
