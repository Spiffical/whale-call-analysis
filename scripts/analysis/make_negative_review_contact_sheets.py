#!/usr/bin/env python3
"""Create small visual review sheets for dry-run ONC negatives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.analyze_weekend_multispecies import _load_mat_image, ensure_matplotlib  # noqa: E402
from src.dataset.multilabel import clean_text, write_csv_rows  # noqa: E402


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sample_rows(rows: Sequence[Mapping[str, Any]], *, max_rows: int, bucket: str) -> List[Dict[str, Any]]:
    bucket_rows = [dict(row) for row in rows if clean_text(row.get("negative_bucket")) == bucket]
    if len(bucket_rows) <= max_rows:
        return bucket_rows
    by_split = {split: [row for row in bucket_rows if clean_text(row.get("split")) == split] for split in ("train", "val", "test")}
    selected: List[Dict[str, Any]] = []
    seen_sources: set[str] = set()
    split_order = ("val", "test", "train")
    while len(selected) < max_rows and any(by_split.values()):
        progressed = False
        for split in split_order:
            candidates = by_split.get(split, [])
            while candidates:
                row = candidates.pop(0)
                source = clean_text(row.get("source_audio") or row.get("filename") or row.get("clip"))
                if source not in seen_sources or len(seen_sources) >= max_rows:
                    selected.append(row)
                    seen_sources.add(source)
                    progressed = True
                    break
            if len(selected) >= max_rows:
                break
        if not progressed:
            break
    if len(selected) < max_rows:
        for row in bucket_rows:
            if row not in selected:
                selected.append(row)
                if len(selected) >= max_rows:
                    break
    return selected[:max_rows]


def _load_audio_spectrogram(row: Mapping[str, Any]) -> Optional[Any]:
    try:
        import numpy as np
        import soundfile as sf
        from scipy import signal
    except Exception:
        return None

    audio_path = Path(clean_text(row.get("source_audio") or row.get("filename") or row.get("clip")))
    if not audio_path.exists():
        return None
    begin_s = _float_or_none(row.get("begin_s") or row.get("window_start_s"))
    end_s = _float_or_none(row.get("end_s"))
    if end_s is None and begin_s is not None:
        duration_s = _float_or_none(row.get("duration_s"))
        if duration_s is not None:
            end_s = begin_s + duration_s
    if begin_s is None or end_s is None or end_s <= begin_s:
        return None

    try:
        info = sf.info(str(audio_path))
        start = max(0, int(round(begin_s * info.samplerate)))
        stop = min(int(info.frames), int(round(end_s * info.samplerate)))
        audio, sr = sf.read(str(audio_path), start=start, stop=stop, always_2d=False)
    except Exception:
        return None
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size < 8:
        return None
    audio = audio.astype("float32")
    audio = audio - np.nanmean(audio)
    nperseg = min(max(256, int(sr * 0.25)), max(256, audio.size))
    noverlap = int(nperseg * 0.75)
    try:
        freqs, _, spec = signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
    except Exception:
        return None
    spec_db = 10.0 * np.log10(np.maximum(spec, 1e-12))
    keep = freqs <= 250.0
    if keep.any():
        spec_db = spec_db[keep, :]
    lo, hi = np.nanpercentile(spec_db, [5, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(spec_db)), float(np.nanmax(spec_db))
    if hi > lo:
        spec_db = (spec_db - lo) / (hi - lo)
    return np.clip(np.nan_to_num(spec_db), 0.0, 1.0)


def _image_for_row(row: Mapping[str, Any]) -> Optional[Any]:
    mat_path = Path(clean_text(row.get("mat_path")))
    if mat_path.exists():
        return _load_mat_image(mat_path)
    return _load_audio_spectrogram(row)


def make_sheet(rows: Sequence[Mapping[str, Any]], out_path: Path, *, title: str) -> int:
    if not rows:
        return 0
    plt = ensure_matplotlib()
    cols = 4
    grid_rows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3.8, grid_rows * 3.0))
    axes_list = [axes] if grid_rows == 1 and cols == 1 else (list(axes) if grid_rows == 1 else [ax for row in axes for ax in row])
    rendered = 0
    for idx, (ax, row) in enumerate(zip(axes_list, rows), start=1):
        image = _image_for_row(row)
        if image is not None:
            ax.imshow(image, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
            rendered += 1
        else:
            ax.text(0.5, 0.5, "missing audio/MAT", ha="center", va="center", fontsize=7)
        item = clean_text(row.get("item_id"))[:42]
        split = clean_text(row.get("split"))
        start = clean_text(row.get("begin_s") or row.get("window_start_s"))
        bucket = clean_text(row.get("negative_bucket"))
        ax.set_title(f"{idx}. {split} {start}s\n{bucket}\n{item}", fontsize=7)
        ax.axis("off")
    for ax in axes_list[len(rows) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rendered


def build_review_artifacts(*, queue_csv: Path, output_dir: Path, max_primary_gap: int = 32, max_ambiguous: int = 24) -> Dict[str, Any]:
    rows = read_csv_rows(queue_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    primary_gap = _sample_rows(rows, max_rows=max_primary_gap, bucket="primary_adjacent_gap")
    ambiguous = _sample_rows(rows, max_rows=max_ambiguous, bucket="ambiguous_hard_negative")
    for row in primary_gap:
        row["suggested_review_label"] = "needs_visual_review"
        row["review_question"] = "Is this 10s primary-adjacent gap clean enough for reviewed_background?"
    for row in ambiguous:
        row["suggested_review_label"] = "ambiguous_hard_negative"
        row["review_question"] = "Confirm this should stay out of clean background gates."

    write_csv_rows(tables_dir / "onc_primary_adjacent_gap_review_sample.csv", primary_gap)
    write_csv_rows(tables_dir / "onc_ambiguous_hard_negative_review_sample.csv", ambiguous)
    primary_rendered = make_sheet(
        primary_gap,
        figures_dir / "onc_primary_adjacent_gap_review_contact_sheet.png",
        title="ONC primary-adjacent gap review sample",
    )
    ambiguous_rendered = make_sheet(
        ambiguous,
        figures_dir / "onc_ambiguous_hard_negative_review_contact_sheet.png",
        title="ONC ambiguous hard-negative review sample",
    )

    summary = {
        "queue_csv": str(queue_csv.resolve()),
        "row_count": len(rows),
        "queue_bucket_counts": dict(Counter(clean_text(row.get("negative_bucket")) for row in rows).most_common()),
        "queue_split_counts": dict(Counter(clean_text(row.get("split")) for row in rows).most_common()),
        "primary_gap_sample_count": len(primary_gap),
        "primary_gap_rendered_count": primary_rendered,
        "ambiguous_sample_count": len(ambiguous),
        "ambiguous_rendered_count": ambiguous_rendered,
        "decision": "visual_review_required_before_training",
    }
    (output_dir / "negative_review_visual_sample_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# ONC Negative Review Visual Sample",
        "",
        f"- Queue rows: `{len(rows)}`.",
        f"- Primary-adjacent gap sample: `{len(primary_gap)}` rows, `{primary_rendered}` rendered panels.",
        f"- Ambiguous hard-negative sample: `{len(ambiguous)}` rows, `{ambiguous_rendered}` rendered panels.",
        "",
        "## Decision",
        "",
        "- Keep training blocked until enough primary-adjacent gaps are reviewed as clean `reviewed_background`.",
        "- Ambiguous hard negatives remain useful for training pressure but must not define the deployment background gate.",
        "",
    ]
    (output_dir / "negative_review_visual_sample_report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-primary-gap", type=int, default=32)
    parser.add_argument("--max-ambiguous", type=int, default=24)
    args = parser.parse_args()
    summary = build_review_artifacts(
        queue_csv=Path(args.queue_csv),
        output_dir=Path(args.output_dir),
        max_primary_gap=int(args.max_primary_gap),
        max_ambiguous=int(args.max_ambiguous),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
