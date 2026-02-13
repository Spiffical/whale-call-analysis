#!/usr/bin/env python3
"""
Audit timing alignment for inference datasets and prediction media.

Checks:
1) Full-spectrogram MAT time span vs metadata segment duration.
2) Per-item spectrogram MAT duration vs clipped audio duration in predictions JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import soundfile as sf


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_rel(base: Path, rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _mat_time_span_seconds(mat_path: Path) -> Optional[float]:
    try:
        d = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception:
        return None
    t = d.get("T")
    if t is None:
        t = d.get("times")
    if t is None:
        return None
    t_arr = np.asarray(t, dtype=np.float64).ravel()
    if t_arr.size < 2:
        return None
    diffs = np.diff(t_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    dt = float(np.median(diffs))
    return float(t_arr[-1] - t_arr[0]) + dt


def _audio_duration_seconds(audio_path: Path) -> Optional[float]:
    try:
        info = sf.info(str(audio_path))
    except Exception:
        return None
    if info.samplerate <= 0:
        return None
    return float(info.frames) / float(info.samplerate)


def _item_media_paths(item: Dict[str, Any], base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    paths = item.get("paths") if isinstance(item.get("paths"), dict) else {}
    mat_rel = (
        item.get("spectrogram_mat_path")
        or item.get("mat_path")
        or paths.get("spectrogram_mat_path")
        or paths.get("mat_path")
    )
    audio_rel = item.get("audio_path") or paths.get("audio_path")
    return _resolve_rel(base, mat_rel), _resolve_rel(base, audio_rel)


@dataclass
class DiffStats:
    label: str
    n: int
    median_diff: float
    p95_abs_diff: float
    max_abs_diff: float


def _summarize_diffs(label: str, diffs: List[float]) -> Optional[DiffStats]:
    if not diffs:
        return None
    arr = np.asarray(diffs, dtype=np.float64)
    return DiffStats(
        label=label,
        n=int(arr.size),
        median_diff=float(np.median(arr)),
        p95_abs_diff=float(np.percentile(np.abs(arr), 95)),
        max_abs_diff=float(np.max(np.abs(arr))),
    )


def _format_stats(s: DiffStats) -> str:
    return (
        f"{s.label}: n={s.n}, median_diff={s.median_diff:.3f}s, "
        f"p95_abs_diff={s.p95_abs_diff:.3f}s, max_abs_diff={s.max_abs_diff:.3f}s"
    )


def audit_dataset_metadata(metadata_path: Path) -> Tuple[List[float], int]:
    meta = _load_json(metadata_path)
    files = meta.get("files", [])
    base = metadata_path.parent
    diffs: List[float] = []
    checked = 0
    for entry in files:
        mat_rel = entry.get("mat_path")
        mat_path = _resolve_rel(base, mat_rel)
        if mat_path is None or not mat_path.exists():
            continue
        span = _mat_time_span_seconds(mat_path)
        start = entry.get("segment_start_sec")
        end = entry.get("segment_end_sec")
        if span is None or start is None or end is None:
            continue
        expected = float(end) - float(start)
        diffs.append(float(span - expected))
        checked += 1
    return diffs, checked


def audit_predictions(predictions_json: Path) -> Tuple[List[float], int]:
    obj = _load_json(predictions_json)
    items = obj.get("items", [])
    base = predictions_json.parent
    diffs: List[float] = []
    checked = 0
    for item in items:
        mat_path, audio_path = _item_media_paths(item, base)
        if mat_path is None or audio_path is None:
            continue
        if not mat_path.exists() or not audio_path.exists():
            continue
        spec_dur = _mat_time_span_seconds(mat_path)
        audio_dur = _audio_duration_seconds(audio_path)
        if spec_dur is None or audio_dur is None:
            continue
        diffs.append(float(audio_dur - spec_dur))
        checked += 1
    return diffs, checked


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit timing alignment in inference artifacts.")
    ap.add_argument("--dataset-metadata", type=str, required=True, help="Path to metadata.json from test prep")
    ap.add_argument("--predictions-json", type=str, default=None, help="Optional predictions JSON to audit media alignment")
    ap.add_argument("--out-md", type=str, default=None, help="Optional markdown summary path")
    args = ap.parse_args()

    metadata_path = Path(args.dataset_metadata).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    md_diffs, md_checked = audit_dataset_metadata(metadata_path)
    md_stats = _summarize_diffs("full_spec_span_minus_segment_duration", md_diffs)

    pred_stats = None
    pred_checked = 0
    if args.predictions_json:
        pred_json = Path(args.predictions_json).resolve()
        if not pred_json.exists():
            raise FileNotFoundError(f"Predictions JSON not found: {pred_json}")
        pred_diffs, pred_checked = audit_predictions(pred_json)
        pred_stats = _summarize_diffs("audio_duration_minus_spectrogram_duration", pred_diffs)

    lines: List[str] = []
    lines.append("# Inference Timing Audit")
    lines.append("")
    lines.append(f"- metadata: `{metadata_path}`")
    lines.append(f"- full-spec entries checked: {md_checked}")
    if md_stats:
        lines.append(f"- {_format_stats(md_stats)}")
    else:
        lines.append("- full-spec timing stats: unavailable")
    if args.predictions_json:
        lines.append(f"- predictions: `{Path(args.predictions_json).resolve()}`")
        lines.append(f"- prediction items with both media checked: {pred_checked}")
        if pred_stats:
            lines.append(f"- {_format_stats(pred_stats)}")
        else:
            lines.append("- prediction media timing stats: unavailable")

    summary = "\n".join(lines) + "\n"
    print(summary)

    if args.out_md:
        out_path = Path(args.out_md).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
