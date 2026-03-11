#!/usr/bin/env python3
"""
Create a Perch2 training dataset of 40s context audio clips (positive + negative).

This mirrors the spectrogram training data selection logic (call-centered positives,
margin-safe sampled negatives), but writes 40s WAV context clips + a manifest that
can be shipped to DRAC/Nibi and consumed directly by train_perch2_embeddings.py in
prebuilt-context mode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

# Ensure repo root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.train_perch2_embeddings import (  # noqa: E402
    _read_audio_window,
    _resolve_audio_path,
    _safe_float,
    build_window_manifest,
)


def _create_archive(
    dataset_dir: Path,
    output_path: Path,
    fmt: str,
    threads: int,
    zstd_level: int,
    gzip_level: int,
) -> None:
    members = _list_archive_members(dataset_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    file_list_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"{output_path.name}.",
            suffix=".tar-members",
            delete=False,
        ) as f:
            file_list_path = Path(f.name)
            for member in members:
                f.write(member.encode("utf-8"))
                f.write(b"\0")

        cmd: List[str]
        base_cmd = ["tar", "-C", str(dataset_dir), "--null", "-T", str(file_list_path)]
        if fmt == "tar":
            cmd = [*base_cmd, "-cf", str(tmp_path)]
        elif fmt == "tar.gz":
            if shutil.which("pigz"):
                cmd = [
                    *base_cmd,
                    "-I",
                    f"pigz -p {max(1, int(threads))} -{int(gzip_level)}",
                    "-cf",
                    str(tmp_path),
                ]
            else:
                cmd = [*base_cmd, "-czf", str(tmp_path)]
        elif fmt == "tar.zst":
            if not shutil.which("zstd"):
                raise RuntimeError("zstd is required for --archive-format tar.zst")
            cmd = [
                *base_cmd,
                "-I",
                f"zstd -T{max(1, int(threads))} -{int(zstd_level)}",
                "-cf",
                str(tmp_path),
            ]
        else:
            raise RuntimeError(f"Unsupported archive format: {fmt}")

        print("Creating archive:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        tmp_path.rename(output_path)
        print(f"Archive ready: {output_path}")
    finally:
        if file_list_path is not None:
            file_list_path.unlink(missing_ok=True)


def _list_archive_members(dataset_dir: Path) -> List[str]:
    members = sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if not path.is_dir()
    )
    if not members:
        raise RuntimeError(f"No files found to archive in dataset directory: {dataset_dir}")
    return members


def _normalize_context_record(
    row: dict,
    context_filename: str,
) -> dict:
    out = dict(row)
    original_clip_id = str(row["clip_id"])
    original_start = _safe_float(row.get("window_start_s"), default=0.0)
    duration = _safe_float(row.get("window_duration_s"), default=40.0)

    out["original_clip_id"] = original_clip_id
    out["original_window_start_s"] = float(original_start)
    out["original_window_end_s"] = float(original_start + duration)
    out["split_start_s"] = float(original_start)
    out["split_src"] = str(row.get("src", original_clip_id))

    out["clip_id"] = context_filename
    out["window_start_s"] = 0.0
    out["window_end_s"] = float(duration)
    out["window_duration_s"] = float(duration)

    if int(row.get("label", 0)) == 1:
        begin = _safe_float(row.get("call_begin_s"), default=np.nan)
        end = _safe_float(row.get("call_end_s"), default=np.nan)
        if np.isfinite(begin):
            out["call_begin_s"] = float(np.clip(begin - original_start, 0.0, duration))
        if np.isfinite(end):
            out["call_end_s"] = float(np.clip(end - original_start, 0.0, duration))
    return out


def build_context_audio_dataset(
    excel_files: List[str],
    source_audio_dir: Path,
    output_dir: Path,
    context_seconds: float,
    negatives_per_positive: int,
    negative_margin_seconds: float,
    max_positives: Optional[int],
    max_audio_files: Optional[int],
    seed: int,
    assumed_clip_duration_seconds: float,
    wav_subtype: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    base_manifest, base_summary = build_window_manifest(
        excel_files=excel_files,
        context_duration_s=float(context_seconds),
        negatives_per_positive=int(negatives_per_positive),
        negative_margin_s=float(negative_margin_seconds),
        max_positives=max_positives,
        max_audio_files=max_audio_files,
        seed=int(seed),
        assumed_clip_duration_s=float(assumed_clip_duration_seconds),
    )

    context_audio_dir = output_dir / "context_audio"
    context_audio_dir.mkdir(parents=True, exist_ok=True)

    rows_out: List[dict] = []
    rows_skipped: List[dict] = []
    next_idx = 0

    for clip_id, clip_rows in base_manifest.groupby("clip_id", sort=False):
        audio_path = _resolve_audio_path(source_audio_dir, str(clip_id))
        records = clip_rows.to_dict("records")
        if audio_path is None:
            for rec in records:
                bad = dict(rec)
                bad["skip_reason"] = "missing_source_audio"
                rows_skipped.append(bad)
            continue

        try:
            with sf.SoundFile(audio_path) as af:
                for rec in records:
                    start_s = _safe_float(rec.get("window_start_s"), default=0.0)
                    dur_s = _safe_float(rec.get("window_duration_s"), default=context_seconds)
                    audio, sr = _read_audio_window(af, start_s=float(start_s), window_size_s=float(dur_s))
                    if audio.size == 0:
                        bad = dict(rec)
                        bad["skip_reason"] = "empty_audio_window"
                        rows_skipped.append(bad)
                        continue

                    context_name = f"context_{next_idx:09d}.wav"
                    next_idx += 1
                    context_path = context_audio_dir / context_name
                    sf.write(context_path, audio, sr, subtype=wav_subtype)
                    out_row = _normalize_context_record(rec, context_filename=context_name)
                    rows_out.append(out_row)
        except Exception as exc:
            for rec in records:
                bad = dict(rec)
                bad["skip_reason"] = f"audio_read_error:{type(exc).__name__}"
                bad["skip_error"] = str(exc)
                rows_skipped.append(bad)

    out_df = pd.DataFrame(rows_out).reset_index(drop=True)
    skipped_df = pd.DataFrame(rows_skipped).reset_index(drop=True)
    if out_df.empty:
        skip_counts = {}
        sample_missing = []
        if not skipped_df.empty and "skip_reason" in skipped_df.columns:
            skip_counts = (
                skipped_df["skip_reason"].astype(str).value_counts().sort_values(ascending=False).to_dict()
            )
            if "clip_id" in skipped_df.columns:
                sample_missing = (
                    skipped_df.loc[skipped_df["skip_reason"] == "missing_source_audio", "clip_id"]
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )
        raise RuntimeError(
            "No context windows were extracted into audio clips. "
            f"skip_reason_counts={skip_counts}; sample_missing_clip_ids={sample_missing}"
        )

    summary: Dict[str, object] = {
        "base_context_manifest_summary": base_summary,
        "context_audio_clips": int(len(out_df)),
        "skipped_context_windows": int(len(skipped_df)),
        "positive_context_audio_clips": int((out_df["label"].astype(int) == 1).sum()),
        "negative_context_audio_clips": int((out_df["label"].astype(int) == 0).sum()),
        "unique_source_clips": int(out_df["src"].astype(str).nunique()) if "src" in out_df.columns else None,
        "context_seconds": float(context_seconds),
        "wav_subtype": wav_subtype,
    }
    return out_df, skipped_df, summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create 40s context-audio dataset for Perch2 DRAC training")
    ap.add_argument("--excel-files", nargs="+", required=True, help="Excel annotation files")
    ap.add_argument("--audio-dir", required=True, help="Source directory with 5-minute WAV clips")
    ap.add_argument(
        "--output-dir",
        default="output/perch2_context_dataset",
        help="Output root for prepared dataset",
    )
    ap.add_argument("--context-seconds", type=float, default=40.0)
    ap.add_argument("--negatives-per-positive", type=int, default=1)
    ap.add_argument("--negative-margin-seconds", type=float, default=2.0)
    ap.add_argument("--max-positives", type=int, default=None)
    ap.add_argument("--max-audio-files", type=int, default=None)
    ap.add_argument("--assumed-clip-duration-seconds", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wav-subtype", type=str, default="PCM_16")
    ap.add_argument("--create-archive", action="store_true")
    ap.add_argument("--archive-path", type=str, default=None)
    ap.add_argument(
        "--archive-format",
        type=str,
        default="tar.zst",
        choices=["tar", "tar.gz", "tar.zst"],
    )
    ap.add_argument("--archive-threads", type=int, default=8)
    ap.add_argument("--archive-zstd-level", type=int, default=3)
    ap.add_argument("--archive-gzip-level", type=int, default=3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    source_audio_dir = Path(args.audio_dir)
    if not source_audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {source_audio_dir}")
    if args.context_seconds <= 0:
        raise SystemExit("--context-seconds must be > 0")
    if args.negatives_per_positive < 0:
        raise SystemExit("--negatives-per-positive must be >= 0")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_dir = Path(args.output_dir) / f"perch2_context_dataset_{stamp}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing context-audio dataset...")
    context_df, skipped_df, summary = build_context_audio_dataset(
        excel_files=list(args.excel_files),
        source_audio_dir=source_audio_dir,
        output_dir=dataset_dir,
        context_seconds=float(args.context_seconds),
        negatives_per_positive=int(args.negatives_per_positive),
        negative_margin_seconds=float(args.negative_margin_seconds),
        max_positives=args.max_positives,
        max_audio_files=args.max_audio_files,
        seed=int(args.seed),
        assumed_clip_duration_seconds=float(args.assumed_clip_duration_seconds),
        wav_subtype=str(args.wav_subtype),
    )

    manifest_path = dataset_dir / "context_window_manifest.csv"
    context_df.to_csv(manifest_path, index=False)
    skipped_path = None
    if not skipped_df.empty:
        skipped_path = dataset_dir / "skipped_context_windows.csv"
        skipped_df.to_csv(skipped_path, index=False)

    summary_payload = {
        "run_utc": stamp,
        "args": vars(args),
        "summary": summary,
        "artifacts": {
            "context_manifest_csv": str(manifest_path),
            "context_audio_dir": str(dataset_dir / "context_audio"),
            "skipped_context_windows_csv": str(skipped_path) if skipped_path else None,
        },
    }
    summary_path = dataset_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Context manifest: {manifest_path} | rows={len(context_df)}")
    if skipped_path:
        print(f"Skipped windows: {skipped_path} | rows={len(skipped_df)}")
    print(f"Summary: {summary_path}")

    archive_path = None
    if args.create_archive:
        if args.archive_path:
            archive_path = Path(args.archive_path)
        else:
            ext = {"tar": ".tar", "tar.gz": ".tar.gz", "tar.zst": ".tar.zst"}[args.archive_format]
            archive_path = dataset_dir / f"context_dataset{ext}"
        _create_archive(
            dataset_dir=dataset_dir,
            output_path=archive_path,
            fmt=args.archive_format,
            threads=int(args.archive_threads),
            zstd_level=int(args.archive_zstd_level),
            gzip_level=int(args.archive_gzip_level),
        )
    if archive_path:
        print(f"Archive: {archive_path}")
    print(f"Done. Dataset directory: {dataset_dir}")


if __name__ == "__main__":
    main()
