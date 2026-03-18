#!/usr/bin/env python3
"""Prepare the Part 2 VM-side bundle for Nibi.

This script is intended for execution on the ONC VM near the mounted drive.
It performs the heavy data-preparation steps:

1. Normalize the workbook into stable manifests.
2. Stage or download the exact candidate audio files near the output bundle.
3. Generate train-style 40s MAT windows for the union of fin-positive and
   annotated non-fin clips.
4. Emit a metadata.json compatible with run_inference.py.
5. Optionally create a deterministic archive for transfer to Nibi.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.part2_annotations import (
    build_part2_manifests,
    parse_filename_timestamp,
    parse_window_mat_stem,
    write_part2_manifests,
)


def _audio_candidates(audio_root: Path) -> Iterable[Path]:
    for path in sorted(audio_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}:
            yield path


def _index_audio(audio_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in _audio_candidates(audio_root):
        index.setdefault(path.name, path)
    return index


def _copy_selected_audio(
    clip_names: Sequence[str],
    audio_root: Path,
    staged_audio_dir: Path,
) -> Tuple[List[str], List[str]]:
    staged_audio_dir.mkdir(parents=True, exist_ok=True)
    index = _index_audio(audio_root)
    copied: List[str] = []
    missing: List[str] = []
    for clip_name in sorted(set(clip_names)):
        source = index.get(clip_name)
        if source is None:
            missing.append(clip_name)
            continue
        target = staged_audio_dir / clip_name
        if not target.exists():
            shutil.copy2(source, target)
        copied.append(clip_name)
    return copied, missing


def _download_missing_audio(
    clip_names: Sequence[str],
    target_dir: Path,
    *,
    onc_token: str,
    show_onc_warnings: bool = False,
) -> Tuple[List[str], List[str]]:
    if not clip_names:
        return [], []
    try:
        from onc import ONC
    except Exception as exc:
        raise SystemExit(
            "Downloading Part 2 audio requires the ONC Python client on the VM. "
            f"Import error: {exc}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    client = ONC(onc_token, showWarning=show_onc_warnings)
    client.outPath = str(target_dir)

    downloaded: List[str] = []
    failed: List[str] = []
    for clip_name in sorted(set(clip_names)):
        target_path = target_dir / clip_name
        if target_path.exists() and target_path.stat().st_size > 0:
            downloaded.append(clip_name)
            continue
        try:
            client.getFile(clip_name)
        except Exception:
            failed.append(clip_name)
            continue
        if target_path.exists() and target_path.stat().st_size > 0:
            downloaded.append(clip_name)
        else:
            failed.append(clip_name)
    return downloaded, failed


def _run_prepare_trainstyle_windows(
    *,
    clip_list_path: Path,
    audio_dir: Path,
    dataset_doc: Path,
    out_dir: Path,
    window_s: float,
    step_s: float,
    spec_backend: str,
) -> None:
    script = REPO_ROOT / "scripts" / "data" / "test" / "prepare_trainstyle_windows.py"
    cmd = [
        sys.executable,
        str(script),
        "--slide",
        "--clip-list",
        str(clip_list_path),
        "--audio-dir",
        str(audio_dir),
        "--dataset-doc",
        str(dataset_doc),
        "--out-dir",
        str(out_dir),
        "--window-s",
        str(window_s),
        "--step-s",
        str(step_s),
        "--spec-backend",
        str(spec_backend),
    ]
    subprocess.run(cmd, check=True)


def _metadata_rows_from_mats(
    mat_dir: Path,
    bundle_dir: Path,
) -> Tuple[List[Dict[str, object]], Optional[str], Optional[str]]:
    rows: List[Dict[str, object]] = []
    min_dt = None
    max_dt = None
    for mat_path in sorted(mat_dir.glob("*.mat")):
        parsed = parse_window_mat_stem(mat_path.stem)
        if parsed is None:
            continue
        source_audio, start_s, end_s = parsed
        file_id = mat_path.stem
        source_ts = parse_filename_timestamp(source_audio)
        audio_timestamp = None
        if source_ts is not None:
            window_ts = source_ts + timedelta(seconds=float(start_s))
            audio_timestamp = window_ts.isoformat()
            min_dt = window_ts if min_dt is None or window_ts < min_dt else min_dt
            max_dt = window_ts if max_dt is None or window_ts > max_dt else max_dt

        raw_audio_rel = None
        raw_audio_path = bundle_dir / "raw_audio" / source_audio
        if raw_audio_path.exists():
            raw_audio_rel = str(raw_audio_path.relative_to(bundle_dir))

        rows.append(
            {
                "file_id": file_id,
                "mat_path": str(mat_path.relative_to(bundle_dir)),
                "source_audio": source_audio,
                "raw_audio_path": raw_audio_rel,
                "segment_index": "",
                "segment_start_sec": start_s,
                "segment_end_sec": end_s,
                "window_index": "",
                "window_start": "",
                "window_time_start": start_s,
                "window_time_end": end_s,
                "audio_timestamp": audio_timestamp or "",
            }
        )
    return rows, min_dt.isoformat() if min_dt else None, max_dt.isoformat() if max_dt else None


def _infer_data_source(clip_names: Sequence[str]) -> Dict[str, object]:
    device_codes = sorted({name.split("_", 1)[0] for name in clip_names if "_" in name})
    device_code = device_codes[0] if len(device_codes) == 1 else "mixed"
    return {
        "device_code": device_code,
        "date_from": "",
        "date_to": "",
    }


def _load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _create_archive(bundle_dir: Path, archive_path: Path) -> None:
    script = REPO_ROOT / "drac" / "scripts" / "create_finwhale_audio_archive.sh"
    cmd = [
        "bash",
        str(script),
        "--dataset-dir",
        str(bundle_dir),
        "--output-path",
        str(archive_path),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the VM-side Part 2 bundle for Nibi")
    ap.add_argument(
        "--workbook",
        type=str,
        default="data/finwhales/Clayoquot_2025_annotations_Mar18.xlsx",
        help="Path to the Part 2 workbook",
    )
    ap.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory with cached/source 5-minute audio clips; missing clips can be downloaded on the VM",
    )
    ap.add_argument("--dataset-doc", type=str, required=True, help="dataset_documentation.json for train-style prep")
    ap.add_argument("--output-dir", type=str, required=True, help="Bundle output directory on the mounted drive")
    ap.add_argument("--window-s", type=float, default=40.0)
    ap.add_argument("--step-s", type=float, default=40.0)
    ap.add_argument("--spec-backend", type=str, default="auto", choices=["auto", "scipy", "torch"])
    ap.add_argument(
        "--adjacent-boundary-seconds",
        type=float,
        default=20.0,
        help="Download/stage previous or next 5-minute clips when annotations fall within this many seconds of a clip edge",
    )
    ap.add_argument(
        "--include-adjacent-in-prep",
        action="store_true",
        help="Include adjacent boundary-context clips in prep_clips.txt rather than keeping them as download-only context",
    )
    ap.add_argument(
        "--download-missing-audio",
        action="store_true",
        help="Download missing candidate/adjacent clips from ONC on the VM instead of failing on missing audio",
    )
    ap.add_argument(
        "--onc-token-env",
        type=str,
        default="ONC_TOKEN",
        help="Environment variable that holds the ONC API token when --download-missing-audio is used",
    )
    ap.add_argument(
        "--show-onc-warnings",
        action="store_true",
        help="Show ONC client warnings during VM-side downloads",
    )
    ap.add_argument("--skip-prep", action="store_true", help="Only build manifests and metadata scaffolding")
    ap.add_argument(
        "--no-stage-selected-audio",
        dest="stage_selected_audio",
        action="store_false",
        help="Do not copy the selected raw audio clips into the bundle",
    )
    ap.set_defaults(stage_selected_audio=True)
    ap.add_argument("--archive-path", type=str, default=None, help="Optional archive path for transfer to Nibi")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    workbook = Path(args.workbook)
    audio_dir = Path(args.audio_dir)
    dataset_doc = Path(args.dataset_doc)
    bundle_dir = Path(args.output_dir)
    manifests_dir = bundle_dir / "manifests"
    raw_audio_dir = bundle_dir / "raw_audio"
    mat_dir = bundle_dir / "mat_files"

    manifests = build_part2_manifests(
        workbook,
        adjacent_boundary_seconds=max(0.0, float(args.adjacent_boundary_seconds)),
        include_adjacent_in_prep=bool(args.include_adjacent_in_prep),
        seed=int(args.seed),
    )
    write_part2_manifests(manifests_dir, manifests)
    candidate_clip_names = [row["filename"] for row in manifests["candidate_clips"]]
    adjacent_clip_names = [row["filename"] for row in manifests.get("adjacent_context_clips", [])]
    download_clip_names = [row["filename"] for row in manifests.get("download_clips", manifests["candidate_clips"])]
    prep_clip_names = [row["filename"] for row in manifests.get("prep_clips", manifests["candidate_clips"])]
    prep_clip_list = manifests_dir / "prep_clips.txt"

    copied_audio: List[str] = []
    downloaded_audio: List[str] = []
    download_failures: List[str] = []
    prep_audio_dir = raw_audio_dir if args.stage_selected_audio else audio_dir
    staging_target_dir = raw_audio_dir if args.stage_selected_audio else audio_dir
    if args.stage_selected_audio:
        copied_audio, missing_before_download = _copy_selected_audio(download_clip_names, audio_dir, raw_audio_dir)
    else:
        indexed_audio = _index_audio(audio_dir)
        missing_before_download = [name for name in download_clip_names if name not in indexed_audio]

    missing_audio = list(missing_before_download)
    if missing_before_download and args.download_missing_audio:
        onc_token = os.getenv(args.onc_token_env, "").strip()
        if not onc_token:
            raise SystemExit(
                f"{args.onc_token_env} is required when --download-missing-audio is enabled."
            )
        downloaded_audio, download_failures = _download_missing_audio(
            missing_before_download,
            staging_target_dir,
            onc_token=onc_token,
            show_onc_warnings=bool(args.show_onc_warnings),
        )
        downloaded_set = set(downloaded_audio)
        missing_audio = [name for name in missing_before_download if name not in downloaded_set]

    missing_path = bundle_dir / "missing_audio.txt"
    with open(missing_path, "w", encoding="utf-8") as handle:
        for clip_name in missing_audio:
            handle.write(f"{clip_name}\n")

    downloaded_path = bundle_dir / "downloaded_audio.txt"
    with open(downloaded_path, "w", encoding="utf-8") as handle:
        for clip_name in downloaded_audio:
            handle.write(f"{clip_name}\n")

    failed_downloads_path = bundle_dir / "failed_downloads.txt"
    with open(failed_downloads_path, "w", encoding="utf-8") as handle:
        for clip_name in download_failures:
            handle.write(f"{clip_name}\n")

    if missing_audio:
        raise SystemExit(
            f"Missing {len(missing_audio)} required audio clips. "
            f"See {missing_path} before running the heavy prep step."
        )

    if not args.skip_prep:
        mat_dir.mkdir(parents=True, exist_ok=True)
        _run_prepare_trainstyle_windows(
            clip_list_path=prep_clip_list,
            audio_dir=prep_audio_dir,
            dataset_doc=dataset_doc,
            out_dir=mat_dir,
            window_s=float(args.window_s),
            step_s=float(args.step_s),
            spec_backend=str(args.spec_backend),
        )

    metadata_rows, date_from, date_to = _metadata_rows_from_mats(mat_dir, bundle_dir)
    data_source = _infer_data_source(prep_clip_names)
    if date_from:
        data_source["date_from"] = date_from
    if date_to:
        data_source["date_to"] = date_to

    summary = _load_summary(manifests_dir / "summary.json")
    metadata = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "spectrogram_config": {
            "context_duration": float(args.window_s),
            "window_duration": "",
            "overlap": "",
            "source": {
                "type": "computed",
                "pipeline": "prepare_trainstyle_windows.slide",
                "dataset_doc": str(dataset_doc),
            },
        },
        "files": metadata_rows,
        "part2_summary": summary,
        "bundle": {
            "stage_selected_audio": bool(args.stage_selected_audio),
            "download_missing_audio": bool(args.download_missing_audio),
            "include_adjacent_in_prep": bool(args.include_adjacent_in_prep),
            "adjacent_boundary_seconds": float(args.adjacent_boundary_seconds),
            "candidate_clip_count": len(candidate_clip_names),
            "adjacent_context_clip_count": len(adjacent_clip_names),
            "download_clip_count": len(download_clip_names),
            "prep_clip_count": len(prep_clip_names),
            "copied_audio_count": len(copied_audio),
            "downloaded_audio_count": len(downloaded_audio),
            "staged_audio_count": len(copied_audio) + len(downloaded_audio) if args.stage_selected_audio else len(download_clip_names) - len(missing_audio),
            "mat_count": len(metadata_rows),
            "window_s": float(args.window_s),
            "step_s": float(args.step_s),
        },
    }
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    prep_summary = {
        "bundle_dir": str(bundle_dir),
        "candidate_clip_count": len(candidate_clip_names),
        "adjacent_context_clip_count": len(adjacent_clip_names),
        "download_clip_count": len(download_clip_names),
        "prep_clip_count": len(prep_clip_names),
        "copied_audio_count": len(copied_audio),
        "downloaded_audio_count": len(downloaded_audio),
        "download_failure_count": len(download_failures),
        "staged_audio_count": len(copied_audio) + len(downloaded_audio) if args.stage_selected_audio else len(download_clip_names) - len(missing_audio),
        "mat_count": len(metadata_rows),
        "missing_audio_count": len(missing_audio),
        "download_missing_audio": bool(args.download_missing_audio),
        "include_adjacent_in_prep": bool(args.include_adjacent_in_prep),
        "adjacent_boundary_seconds": float(args.adjacent_boundary_seconds),
        "archive_path": args.archive_path or "",
    }
    with open(bundle_dir / "prep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(prep_summary, handle, indent=2, sort_keys=True)

    if args.archive_path:
        _create_archive(bundle_dir, Path(args.archive_path))

    print("Prepared Part 2 VM bundle:")
    print(f"  bundle_dir: {bundle_dir}")
    print(f"  manifests: {manifests_dir}")
    print(f"  raw_audio: {prep_audio_dir}")
    print(f"  candidate_clips: {len(candidate_clip_names)}")
    print(f"  adjacent_context_clips: {len(adjacent_clip_names)}")
    print(f"  prep_clips: {len(prep_clip_names)}")
    print(f"  copied_audio: {len(copied_audio)}")
    print(f"  downloaded_audio: {len(downloaded_audio)}")
    print(f"  mat_files: {mat_dir}")
    print(f"  metadata: {metadata_path}")
    if args.archive_path:
        print(f"  archive: {args.archive_path}")


if __name__ == "__main__":
    main()
