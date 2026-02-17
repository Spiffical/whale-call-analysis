#!/usr/bin/env python3
"""
Compare saved spectrogram MATs against spectrograms recomputed from their paired audio clips.

Outputs:
- side-by-side PNG plots (saved / recomputed-on-saved-grid / abs-diff)
- CSV summary with lag/correlation/error metrics
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal
import soundfile as sf


@dataclass
class CompareMetrics:
    item_id: str
    audio_samplerate: int
    saved_shape: Tuple[int, int]
    recomputed_shape: Tuple[int, int]
    lag_seconds_profile: float
    corr_flat: float
    mae_db: float
    rmse_db: float


def _load_saved_mat(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = scipy.io.loadmat(str(path), simplify_cells=True)
    spec = np.asarray(data.get("PdB_norm"))
    f = np.asarray(data.get("F"), dtype=np.float64).ravel()
    t = np.asarray(data.get("T"), dtype=np.float64).ravel()
    if spec.ndim != 2:
        raise ValueError(f"Unexpected spec ndim={spec.ndim} for {path}")
    # Orientation check
    if f.size and t.size and spec.shape == (t.size, f.size):
        spec = spec.T
    return spec.astype(np.float32), f, t


def _compute_spectrogram_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    win_dur: float,
    overlap: float,
    freq_min: float,
    freq_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    win_len = max(1, int(round(float(win_dur) * int(sample_rate))))
    noverlap = int(round(float(overlap) * win_len))
    noverlap = max(0, min(noverlap, win_len - 1))
    window = scipy.signal.get_window("hann", win_len, fftbins=True)
    f, t, sxx = scipy.signal.spectrogram(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=win_len,
        noverlap=noverlap,
        nfft=win_len,
        scaling="density",
        mode="psd",
    )
    sxx = np.abs(sxx.astype(np.float32))
    mx = float(np.max(sxx)) if sxx.size else 0.0
    if mx > 0:
        pdB = 10.0 * np.log10(np.maximum(sxx / mx, 1e-10))
    else:
        pdB = np.full_like(sxx, -100.0, dtype=np.float32)
    mask = (f >= float(freq_min)) & (f <= float(freq_max))
    if not np.any(mask):
        raise ValueError(f"No frequency bins in [{freq_min}, {freq_max}] Hz")
    return pdB[mask, :], f[mask].astype(np.float64), t.astype(np.float64)


def _interp_rows_1d(values: np.ndarray, src_x: np.ndarray, tgt_x: np.ndarray) -> np.ndarray:
    out = np.empty((values.shape[0], tgt_x.size), dtype=np.float32)
    for i in range(values.shape[0]):
        out[i, :] = np.interp(tgt_x, src_x, values[i, :], left=values[i, 0], right=values[i, -1])
    return out


def _interp_cols_1d(values: np.ndarray, src_y: np.ndarray, tgt_y: np.ndarray) -> np.ndarray:
    out = np.empty((tgt_y.size, values.shape[1]), dtype=np.float32)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(tgt_y, src_y, values[:, j], left=values[0, j], right=values[-1, j])
    return out


def _regrid_to_saved(
    spec_src: np.ndarray,
    f_src: np.ndarray,
    t_src: np.ndarray,
    f_tgt: np.ndarray,
    t_tgt: np.ndarray,
) -> np.ndarray:
    # Compare clip-relative time axes. Some saved MATs keep absolute-in-parent
    # time coordinates, while recomputed audio spectrograms are clip-relative.
    t_src = np.asarray(t_src, dtype=np.float64) - float(np.asarray(t_src, dtype=np.float64)[0])
    t_tgt = np.asarray(t_tgt, dtype=np.float64) - float(np.asarray(t_tgt, dtype=np.float64)[0])

    # Frequency interpolate first -> (len(f_tgt), src_time)
    freq_interp = _interp_cols_1d(spec_src, f_src, f_tgt)
    # Time interpolate second -> (len(f_tgt), len(t_tgt))
    out = _interp_rows_1d(freq_interp, t_src, t_tgt)
    return out


def _profile_lag_seconds(saved: np.ndarray, recomp: np.ndarray, dt: float) -> float:
    a = np.mean(saved.astype(np.float64), axis=0)
    b = np.mean(recomp.astype(np.float64), axis=0)
    a = a - np.mean(a)
    b = b - np.mean(b)
    sa = np.std(a)
    sb = np.std(b)
    if sa == 0 or sb == 0:
        return 0.0
    corr = np.correlate(a / sa, b / sb, mode="full")
    lag_bins = int(np.argmax(corr) - (len(a) - 1))
    return float(lag_bins) * float(dt)


def _flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.dot(x, y) / (len(x) * sx * sy))


def _plot_compare(
    out_png: Path,
    item_id: str,
    saved: np.ndarray,
    recomp: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    metrics: CompareMetrics,
) -> None:
    diff = np.abs(saved - recomp)
    extent = [float(t[0]), float(t[-1]), float(f[0]), float(f[-1])]
    vmin = float(min(np.min(saved), np.min(recomp)))
    vmax = float(max(np.max(saved), np.max(recomp)))
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)

    im0 = axes[0].imshow(saved, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Saved MAT Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.01)

    im1 = axes[1].imshow(recomp, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Recomputed from Audio (regridded to saved axes)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.01)

    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axes[2].set_title("Absolute Difference |saved - recomputed|")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.01)

    fig.suptitle(
        f"{item_id}\n"
        f"lag={metrics.lag_seconds_profile:.4f}s, corr={metrics.corr_flat:.4f}, "
        f"MAE={metrics.mae_db:.4f} dB, RMSE={metrics.rmse_db:.4f} dB, sr={metrics.audio_samplerate} Hz"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _resolve_paths(pred_json: Path, item: Dict) -> Tuple[Optional[Path], Optional[Path]]:
    paths = item.get("paths") if isinstance(item.get("paths"), dict) else {}
    mat_rel = (
        paths.get("spectrogram_mat_path")
        or item.get("spectrogram_mat_path")
        or paths.get("mat_path")
        or item.get("mat_path")
    )
    aud_rel = paths.get("audio_path") or item.get("audio_path")
    mat = (pred_json.parent / mat_rel).resolve() if mat_rel else None
    aud = (pred_json.parent / aud_rel).resolve() if aud_rel else None
    return mat, aud


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare saved MAT spectrograms with recomputed audio spectrograms.")
    ap.add_argument("--predictions-json", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument(
        "--item-ids",
        type=str,
        default=None,
        help="Comma-separated item_ids. Default: first N items. When --use-source-segments is set, these are event item_ids.",
    )
    ap.add_argument("--max-items", type=int, default=3)
    ap.add_argument(
        "--use-source-segments",
        action="store_true",
        help="For postprocessed event JSONs, compare source window segments instead of merged event media.",
    )
    ap.add_argument(
        "--max-source-segments",
        type=int,
        default=0,
        help="Cap number of source segments when --use-source-segments is enabled (0 = no cap).",
    )
    ap.add_argument("--win-dur", type=float, default=None, help="Override spectrogram window duration")
    ap.add_argument("--overlap", type=float, default=None, help="Override overlap")
    ap.add_argument("--freq-min", type=float, default=None, help="Override min frequency")
    ap.add_argument("--freq-max", type=float, default=None, help="Override max frequency")
    ap.add_argument("--no-plots", action="store_true", help="Skip writing per-item PNG comparison plots.")
    args = ap.parse_args()

    pred_json = Path(args.predictions_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = json.loads(pred_json.read_text())
    items = obj.get("items", [])

    spec_cfg = obj.get("spectrogram_config", {})
    win_dur = float(args.win_dur if args.win_dur is not None else spec_cfg.get("window_duration", 1.0))
    overlap = float(args.overlap if args.overlap is not None else spec_cfg.get("overlap", 0.9))
    freq_limits = spec_cfg.get("frequency_limits", {})
    freq_min = float(args.freq_min if args.freq_min is not None else freq_limits.get("min", 5.0))
    freq_max = float(args.freq_max if args.freq_max is not None else freq_limits.get("max", 100.0))

    selected: List[Dict] = []
    if args.item_ids:
        wanted = {x.strip() for x in args.item_ids.split(",") if x.strip()}
        selected = [it for it in items if str(it.get("item_id")) in wanted]
    else:
        selected = items[: max(1, int(args.max_items))]

    if args.use_source_segments:
        seg_items: List[Dict] = []
        for evt in selected:
            evt_id = str(evt.get("item_id"))
            for seg in evt.get("source_segments", []) or []:
                seg_items.append(
                    {
                        "item_id": f"{evt_id}::{seg.get('source_item_id', 'segment')}",
                        "paths": {
                            "spectrogram_mat_path": seg.get("spectrogram_mat_path"),
                            "audio_path": seg.get("audio_path"),
                        },
                    }
                )
        if args.max_source_segments and args.max_source_segments > 0:
            seg_items = seg_items[: int(args.max_source_segments)]
        selected = seg_items

    metrics_rows: List[CompareMetrics] = []
    for item in selected:
        item_id = str(item.get("item_id"))
        mat_path, aud_path = _resolve_paths(pred_json, item)
        if mat_path is None or aud_path is None or not mat_path.exists() or not aud_path.exists():
            continue

        saved, f_saved, t_saved = _load_saved_mat(mat_path)
        audio, sr = sf.read(str(aud_path), always_2d=False)
        recomp_raw, f_rec, t_rec = _compute_spectrogram_from_audio(
            audio=np.asarray(audio),
            sample_rate=int(sr),
            win_dur=float(win_dur),
            overlap=float(overlap),
            freq_min=float(freq_min),
            freq_max=float(freq_max),
        )
        recomp = _regrid_to_saved(
            spec_src=recomp_raw,
            f_src=f_rec,
            t_src=t_rec,
            f_tgt=f_saved,
            t_tgt=t_saved,
        )

        dt = float(np.median(np.diff(t_saved))) if t_saved.size > 1 else 0.0
        lag = _profile_lag_seconds(saved, recomp, dt) if dt > 0 else 0.0
        corr = _flat_corr(saved, recomp)
        err = (saved.astype(np.float64) - recomp.astype(np.float64))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        metrics = CompareMetrics(
            item_id=item_id,
            audio_samplerate=int(sr),
            saved_shape=(int(saved.shape[0]), int(saved.shape[1])),
            recomputed_shape=(int(recomp.shape[0]), int(recomp.shape[1])),
            lag_seconds_profile=float(lag),
            corr_flat=float(corr),
            mae_db=mae,
            rmse_db=rmse,
        )
        metrics_rows.append(metrics)
        if not args.no_plots:
            _plot_compare(
                out_png=out_dir / f"{item_id}_compare.png",
                item_id=item_id,
                saved=saved,
                recomp=recomp,
                f=f_saved,
                t=t_saved,
                metrics=metrics,
            )

    csv_path = out_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item_id",
                "audio_samplerate",
                "saved_shape",
                "recomputed_shape",
                "lag_seconds_profile",
                "corr_flat",
                "mae_db",
                "rmse_db",
            ]
        )
        for m in metrics_rows:
            w.writerow(
                [
                    m.item_id,
                    m.audio_samplerate,
                    f"{m.saved_shape[0]}x{m.saved_shape[1]}",
                    f"{m.recomputed_shape[0]}x{m.recomputed_shape[1]}",
                    f"{m.lag_seconds_profile:.6f}",
                    f"{m.corr_flat:.6f}",
                    f"{m.mae_db:.6f}",
                    f"{m.rmse_db:.6f}",
                ]
            )

    md_path = out_dir / "comparison_summary.md"
    lines: List[str] = []
    lines.append("# Saved vs Audio-Recomputed Spectrogram Comparison")
    lines.append("")
    lines.append(f"- predictions_json: `{pred_json}`")
    lines.append(f"- win_dur={win_dur}, overlap={overlap}, freq=[{freq_min}, {freq_max}] Hz")
    lines.append(f"- compared_items={len(metrics_rows)}")
    lines.append(f"- metrics_csv: `{csv_path}`")
    lines.append("")
    if metrics_rows:
        lags = np.array([m.lag_seconds_profile for m in metrics_rows], dtype=np.float64)
        cors = np.array([m.corr_flat for m in metrics_rows], dtype=np.float64)
        maes = np.array([m.mae_db for m in metrics_rows], dtype=np.float64)
        lines.append(
            f"- lag_seconds_profile: median={np.median(lags):.6f}, max_abs={np.max(np.abs(lags)):.6f}"
        )
        lines.append(
            f"- corr_flat: median={np.median(cors):.6f}, min={np.min(cors):.6f}, max={np.max(cors):.6f}"
        )
        lines.append(
            f"- mae_db: median={np.median(maes):.6f}, max={np.max(maes):.6f}"
        )
        lines.append("")
        lines.append("## Items")
        for m in metrics_rows:
            plot_ref = "(skipped)" if args.no_plots else str(out_dir / f"{m.item_id}_compare.png")
            lines.append(
                f"- `{m.item_id}`: sr={m.audio_samplerate}, lag={m.lag_seconds_profile:.6f}s, "
                f"corr={m.corr_flat:.6f}, mae={m.mae_db:.6f} dB, rmse={m.rmse_db:.6f} dB, "
                f"plot=`{plot_ref}`"
            )
    else:
        lines.append("- No comparable items found.")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote plots and metrics to: {out_dir}")
    print(f"Summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
