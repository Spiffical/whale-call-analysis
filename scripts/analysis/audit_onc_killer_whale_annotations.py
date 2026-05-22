#!/usr/bin/env python3
"""Render ONC killer whale annotation review sheets from raw audio.

The review sheets are intended as non-committed QA artifacts. They show the raw
ONC annotation window against mid, high, and wide-high spectrogram views so we
can decide whether the ONC killer whale support set is strong enough for
training or should remain diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import clean_text, write_csv_rows  # noqa: E402


REVIEW_LABELS = (
    "true_killer_whale_call",
    "odontocete_non_killer",
    "non_target_biological",
    "noise_or_artifact",
    "boundary_wrong",
    "unclear",
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def safe_float(value: Any) -> Optional[float]:
    try:
        text = clean_text(value)
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def is_onc_killer_whale_annotation(row: Mapping[str, Any]) -> bool:
    # ONC call_type_raw=CK is mostly generic odontocete click under species OD.
    # The killer whale support set is keyed by species code Oo.
    return clean_text(row.get("species")) == "Oo"


def annotation_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    begin = safe_float(row.get("begin_time_s") or row.get("begin_time") or row.get("begin_s") or row.get("window_start_s"))
    return clean_text(row.get("filename") or row.get("clip") or Path(clean_text(row.get("source_audio"))).name), (
        "" if begin is None else f"{begin:.3f}"
    )


def manifest_keys(path: Optional[Path]) -> set[Tuple[str, str]]:
    if path is None or not path.exists():
        return set()
    out: set[Tuple[str, str]] = set()
    for row in read_csv(path):
        labels = clean_text(row.get("label_ids"))
        if "species:Oo" not in labels:
            continue
        out.add(annotation_key(row))
    return out


def find_audio(audio_dir: Path, filename: str) -> Optional[Path]:
    direct = audio_dir / filename
    if direct.exists():
        return direct
    matches = list(audio_dir.rglob(filename))
    return matches[0] if matches else None


def load_audio_segment(audio_path: Path, start_s: float, duration_s: float) -> Tuple[np.ndarray, float]:
    import soundfile as sf

    info = sf.info(str(audio_path))
    fs = float(info.samplerate)
    start_frame = max(0, int(round(start_s * fs)))
    frames = max(1, int(round(duration_s * fs)))
    data, sr = sf.read(str(audio_path), start=start_frame, frames=frames, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), float(sr)


def maybe_downsample(data: np.ndarray, fs: float, target_fs: float) -> Tuple[np.ndarray, float]:
    if fs <= target_fs * 1.25:
        return data, fs
    from scipy import signal

    # Keep integer-ratio reductions exact for common 256 kHz ONC audio.
    ratio = int(round(fs / target_fs))
    if ratio > 1 and abs(fs / ratio - target_fs) < 1e-6:
        return signal.resample_poly(data, up=1, down=ratio), fs / ratio
    n = max(1, int(round(len(data) * target_fs / fs)))
    return signal.resample(data, n), target_fs


def spectrogram_db(
    data: np.ndarray,
    fs: float,
    *,
    window_s: float,
    hop_s: float,
    fmin: float,
    fmax: float,
    downsample_to: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import signal

    work = data
    work_fs = fs
    if downsample_to is not None:
        work, work_fs = maybe_downsample(work, work_fs, downsample_to)
    nperseg = max(64, int(round(window_s * work_fs)))
    hop = max(1, int(round(hop_s * work_fs)))
    noverlap = max(0, min(nperseg - 1, nperseg - hop))
    freqs, times, power = signal.spectrogram(
        work,
        fs=work_fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="magnitude",
    )
    mask = (freqs >= fmin) & (freqs <= min(fmax, work_fs / 2.0))
    freqs = freqs[mask]
    power = power[mask, :]
    db = 20.0 * np.log10(np.maximum(power, 1e-12))
    if db.size:
        db = db - np.nanpercentile(db, 99.5)
    return freqs, times, db


def draw_annotation_box(ax: Any, begin_s: float, end_s: float, low_hz: float, high_hz: float, fmin: float, fmax: float) -> None:
    if end_s <= begin_s or high_hz <= low_hz:
        return
    lo = max(low_hz, fmin)
    hi = min(high_hz, fmax)
    if hi <= lo:
        return
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (begin_s, lo),
        end_s - begin_s,
        hi - lo,
        fill=False,
        edgecolor="#65d9ff",
        linewidth=1.0,
        alpha=0.95,
    )
    ax.add_patch(rect)


def render_contact_sheets(
    rows: Sequence[Mapping[str, Any]],
    *,
    audio_dir: Path,
    output_dir: Path,
    context_s: float,
    rows_per_sheet: int,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    missing_audio: Counter[str] = Counter()
    sheet_paths: List[str] = []
    for sheet_idx, start_idx in enumerate(range(0, len(rows), rows_per_sheet), start=1):
        chunk = rows[start_idx : start_idx + rows_per_sheet]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(16, max(3.2, 2.15 * len(chunk))), squeeze=False)
        fig.suptitle("ONC killer whale annotation review: mid, high, and wide-high views", fontsize=14)
        for row_idx, row in enumerate(chunk):
            filename = clean_text(row.get("filename"))
            audio_path = find_audio(audio_dir, filename)
            begin_s = safe_float(row.get("begin_time_s")) or 0.0
            end_s = safe_float(row.get("end_time_s")) or begin_s
            low_hz = safe_float(row.get("low_freq_hz")) or 0.0
            high_hz = safe_float(row.get("high_freq_hz")) or 0.0
            title = (
                f"{start_idx + row_idx + 1}. {filename}\n"
                f"{begin_s:.1f}-{end_s:.1f}s, {low_hz:.0f}-{high_hz:.0f} Hz"
            )
            if audio_path is None:
                missing_audio[filename] += 1
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "missing audio", ha="center", va="center")
                    ax.set_axis_off()
                axes[row_idx, 0].set_ylabel(title, fontsize=7)
                continue
            try:
                info = sf.info(str(audio_path))
                center = 0.5 * (begin_s + end_s)
                window_start = max(0.0, min(center - context_s / 2.0, max(0.0, float(info.duration) - context_s)))
                data, fs = load_audio_segment(audio_path, window_start, context_s)
                specs = [
                    ("Mid 100-2000 Hz", 100.0, 2000.0, 0.5, 0.1, 8000.0),
                    ("High 500-32000 Hz", 500.0, 32000.0, 0.128, 0.032, None),
                    ("Wide high 500-64000 Hz", 500.0, 64000.0, 0.128, 0.032, None),
                ]
                for col, (label, fmin, fmax, window_s, hop_s, downsample_to) in enumerate(specs):
                    ax = axes[row_idx, col]
                    freqs, times, db = spectrogram_db(
                        data,
                        fs,
                        window_s=window_s,
                        hop_s=hop_s,
                        fmin=fmin,
                        fmax=fmax,
                        downsample_to=downsample_to,
                    )
                    if db.size:
                        ax.imshow(
                            db,
                            origin="lower",
                            aspect="auto",
                            extent=[window_start + float(times[0]), window_start + float(times[-1]), float(freqs[0]), float(freqs[-1])],
                            cmap="magma",
                            vmin=-75,
                            vmax=0,
                        )
                    ax.set_title(label, fontsize=8)
                    ax.set_ylim(fmin, min(fmax, fs / 2.0))
                    ax.set_yscale("log")
                    ymax = min(fmax, fs / 2.0)
                    ticks = [tick for tick in (fmin, 1000, 2000, 5000, 10000, 20000, ymax) if fmin <= tick <= ymax]
                    ax.set_yticks(sorted(set(round(tick, 6) for tick in ticks)))
                    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    draw_annotation_box(ax, begin_s, end_s, low_hz, high_hz, fmin, min(fmax, fs / 2.0))
                    if row_idx == len(chunk) - 1:
                        ax.set_xlabel("Time in source clip (s)")
                    if col == 0:
                        ax.set_ylabel(title, fontsize=7)
                    else:
                        ax.set_ylabel("Hz")
                rendered += 1
            except Exception as exc:  # pragma: no cover - audit artifact resilience
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, f"render error:\n{exc}", ha="center", va="center", fontsize=7)
                    ax.set_axis_off()
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = sheet_dir / f"onc_killer_whale_review_sheet_{sheet_idx:02d}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        sheet_paths.append(str(out))
    return {
        "rendered_annotation_count": rendered,
        "missing_audio_count": int(sum(missing_audio.values())),
        "missing_audio_top": dict(missing_audio.most_common(20)),
        "contact_sheets": sheet_paths,
    }


def build_review_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    manifest_match_keys: set[Tuple[str, str]],
) -> List[Dict[str, str]]:
    review_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(raw_rows, start=1):
        key = annotation_key(row)
        begin_s = safe_float(row.get("begin_time_s"))
        end_s = safe_float(row.get("end_time_s"))
        low_hz = safe_float(row.get("low_freq_hz"))
        high_hz = safe_float(row.get("high_freq_hz"))
        review_rows.append(
            {
                "review_index": str(idx),
                "included_in_e16_manifest": "1" if key in manifest_match_keys else "0",
                "sheet": clean_text(row.get("sheet")),
                "row_index": clean_text(row.get("row_index")),
                "filename": clean_text(row.get("filename")),
                "begin_time_s": "" if begin_s is None else f"{begin_s:.6f}",
                "end_time_s": "" if end_s is None else f"{end_s:.6f}",
                "duration_s": "" if begin_s is None or end_s is None else f"{(end_s - begin_s):.6f}",
                "low_freq_hz": "" if low_hz is None else f"{low_hz:.6f}",
                "high_freq_hz": "" if high_hz is None else f"{high_hz:.6f}",
                "fully_inside_mid_100_2000": "1" if low_hz is not None and high_hz is not None and low_hz >= 100 and high_hz <= 2000 else "0",
                "intersects_mid_100_2000": "1" if low_hz is not None and high_hz is not None and high_hz >= 100 and low_hz <= 2000 else "0",
                "fully_inside_high_500_32000": "1" if low_hz is not None and high_hz is not None and low_hz >= 500 and high_hz <= 32000 else "0",
                "intersects_high_500_32000": "1" if low_hz is not None and high_hz is not None and high_hz >= 500 and low_hz <= 32000 else "0",
                "extends_above_32000": "1" if high_hz is not None and high_hz > 32000 else "0",
                "verified_flag": clean_text(row.get("verified_flag")),
                "granularity": clean_text(row.get("granularity")),
                "comments": clean_text(row.get("comments")),
                "review_label": "",
                "reviewer_notes": "",
                "allowed_review_labels": "|".join(REVIEW_LABELS),
            }
        )
    return review_rows


def summarize_review_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def numeric(key: str) -> List[float]:
        vals = []
        for row in rows:
            value = safe_float(row.get(key))
            if value is not None and math.isfinite(value):
                vals.append(value)
        return vals

    def min_median_max(vals: Sequence[float]) -> List[float]:
        if not vals:
            return []
        sorted_vals = sorted(vals)
        return [float(sorted_vals[0]), float(np.median(sorted_vals)), float(sorted_vals[-1])]

    durations = numeric("duration_s")
    lows = numeric("low_freq_hz")
    highs = numeric("high_freq_hz")
    return {
        "annotation_count": len(rows),
        "included_in_e16_manifest_count": sum(1 for row in rows if clean_text(row.get("included_in_e16_manifest")) == "1"),
        "duration_s_min_median_max": min_median_max(durations),
        "low_freq_hz_min_median_max": min_median_max(lows),
        "high_freq_hz_min_median_max": min_median_max(highs),
        "fully_inside_mid_100_2000": sum(1 for row in rows if clean_text(row.get("fully_inside_mid_100_2000")) == "1"),
        "intersects_mid_100_2000": sum(1 for row in rows if clean_text(row.get("intersects_mid_100_2000")) == "1"),
        "fully_inside_high_500_32000": sum(1 for row in rows if clean_text(row.get("fully_inside_high_500_32000")) == "1"),
        "intersects_high_500_32000": sum(1 for row in rows if clean_text(row.get("intersects_high_500_32000")) == "1"),
        "extends_above_32000": sum(1 for row in rows if clean_text(row.get("extends_above_32000")) == "1"),
        "verified_flag_counts": dict(Counter(clean_text(row.get("verified_flag")) for row in rows).most_common()),
        "granularity_counts": dict(Counter(clean_text(row.get("granularity")) for row in rows).most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True, type=Path)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--raw-audio-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context-s", type=float, default=40.0)
    parser.add_argument("--rows-per-sheet", type=int, default=8)
    parser.add_argument("--max-annotations", type=int, default=0)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [row for row in read_csv(args.annotations_csv) if is_onc_killer_whale_annotation(row)]
    raw_rows = sorted(raw_rows, key=lambda row: (clean_text(row.get("filename")), safe_float(row.get("begin_time_s")) or 0.0))
    if args.max_annotations > 0:
        raw_rows = raw_rows[: args.max_annotations]
    review_rows = build_review_rows(raw_rows, manifest_match_keys=manifest_keys(args.manifest_csv))
    review_csv = args.output_dir / "onc_killer_whale_annotation_review_queue.csv"
    write_csv_rows(review_csv, review_rows)

    image_summary: Dict[str, Any] = {"skipped": bool(args.skip_images)}
    if not args.skip_images and args.raw_audio_dir is not None:
        image_summary = render_contact_sheets(
            raw_rows,
            audio_dir=args.raw_audio_dir,
            output_dir=args.output_dir,
            context_s=float(args.context_s),
            rows_per_sheet=max(1, int(args.rows_per_sheet)),
        )
    summary = summarize_review_rows(review_rows)
    summary.update(
        {
            "annotations_csv": str(args.annotations_csv),
            "manifest_csv": "" if args.manifest_csv is None else str(args.manifest_csv),
            "raw_audio_dir": "" if args.raw_audio_dir is None else str(args.raw_audio_dir),
            "review_csv": str(review_csv),
            "image_summary": image_summary,
        }
    )
    (args.output_dir / "onc_killer_whale_annotation_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
