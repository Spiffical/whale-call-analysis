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
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv as _dotenv_load
except Exception:
    _dotenv_load = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.part2_annotations import (
    build_part2_manifests,
    parse_filename_timestamp,
    parse_window_mat_stem,
    write_part2_manifests,
)


_TQDM = None
try:
    from tqdm import tqdm as _TQDM
except Exception:
    _TQDM = None


def _log(message: str, status: str = "INFO") -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{status}] {message}", flush=True)


def _print_header(title: str) -> None:
    line = "=" * 88
    _log(line, "PHASE")
    _log(title, "PHASE")
    _log(line, "PHASE")


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes:d}m {sec:02d}s"
    return f"{seconds:.1f}s"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    if _dotenv_load is not None:
        _dotenv_load(path)
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _progress_iter(
    items: Sequence[str],
    *,
    desc: str,
) -> Iterable[Tuple[int, str]]:
    total = len(items)
    if _TQDM is not None:
        for idx, item in enumerate(_TQDM(items, desc=desc, unit="clip"), start=1):
            yield idx, item
        return

    started = time.monotonic()
    last_report = 0
    for idx, item in enumerate(items, start=1):
        if idx == 1 or idx == total or idx - last_report >= max(1, min(100, total // 20 or 1)):
            pct = (100.0 * idx / total) if total else 100.0
            _log(f"{desc}: {idx}/{total} ({pct:.1f}%) after {_format_duration(time.monotonic() - started)}", "PROGRESS")
            last_report = idx
        yield idx, item


def _audio_candidates(audio_root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(audio_root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in {".wav", ".flac"}:
                yield path


def _index_audio(audio_root: Path) -> Dict[str, Path]:
    _log(f"Indexing audio cache under {audio_root} ...", "PROGRESS")
    index: Dict[str, Path] = {}
    started = time.monotonic()
    last_report = 0
    for scanned, path in enumerate(_audio_candidates(audio_root), start=1):
        index.setdefault(path.name, path)
        if scanned == 1 or scanned - last_report >= 1000:
            _log(
                f"Indexed {scanned:,} audio files so far "
                f"({len(index):,} unique names) in {_format_duration(time.monotonic() - started)}",
                "PROGRESS",
            )
            last_report = scanned
    _log(
        f"Completed audio index: {len(index):,} unique files found in {_format_duration(time.monotonic() - started)}",
        "SUCCESS",
    )
    return index


def _copy_selected_audio(
    clip_names: Sequence[str],
    audio_root: Path,
    staged_audio_dir: Path,
) -> Tuple[List[str], List[str], int]:
    staged_audio_dir.mkdir(parents=True, exist_ok=True)
    index = _index_audio(audio_root)
    copied: List[str] = []
    missing: List[str] = []
    reused_existing = 0
    unique_names = sorted(set(clip_names))
    _log(
        f"Staging {len(unique_names):,} requested clips into {staged_audio_dir} "
        f"from cache {audio_root}",
        "PROGRESS",
    )
    for _, clip_name in _progress_iter(unique_names, desc="Stage audio"):
        target = staged_audio_dir / clip_name
        if target.exists() and target.stat().st_size > 0:
            reused_existing += 1
            copied.append(clip_name)
            continue
        source = index.get(clip_name)
        if source is None:
            missing.append(clip_name)
            continue
        shutil.copy2(source, target)
        copied.append(clip_name)
    _log(
        f"Audio staging finished: {len(copied):,} available, {len(missing):,} missing, "
        f"{reused_existing:,} already present at destination",
        "SUCCESS",
    )
    return copied, missing, reused_existing


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
    unique_names = sorted(set(clip_names))
    _log(
        f"Downloading {len(unique_names):,} missing clips from ONC into {target_dir}",
        "PROGRESS",
    )
    started = time.monotonic()
    for idx, clip_name in _progress_iter(unique_names, desc="Download audio"):
        target_path = target_dir / clip_name
        if target_path.exists() and target_path.stat().st_size > 0:
            downloaded.append(clip_name)
            continue
        try:
            client.getFile(clip_name)
        except Exception as exc:
            _log(f"Download failed for {clip_name}: {exc}", "WARNING")
            failed.append(clip_name)
            continue
        if target_path.exists() and target_path.stat().st_size > 0:
            downloaded.append(clip_name)
        else:
            failed.append(clip_name)
        if idx % 25 == 0:
            _log(
                f"Download checkpoint: {len(downloaded):,} ready, {len(failed):,} failed, "
                f"elapsed {_format_duration(time.monotonic() - started)}",
                "PROGRESS",
            )
    _log(
        f"Download step finished: {len(downloaded):,} ready, {len(failed):,} failed in "
        f"{_format_duration(time.monotonic() - started)}",
        "SUCCESS" if not failed else "WARNING",
    )
    return downloaded, failed


def _trainstyle_helper_script() -> Path:
    return REPO_ROOT / "scripts" / "data" / "test" / "prepare_trainstyle_windows.py"


def _preflight_or_die(
    *,
    workbook: Path,
    audio_dir: Path,
    dataset_doc: Path,
    bundle_dir: Path,
    archive_path: Optional[Path],
) -> None:
    problems: List[str] = []
    if not workbook.exists():
        problems.append(f"Workbook not found: {workbook}")
    if not dataset_doc.exists():
        problems.append(f"dataset_documentation.json not found: {dataset_doc}")
    if not audio_dir.exists():
        problems.append(f"Audio directory does not exist yet: {audio_dir}")
    helper_script = _trainstyle_helper_script()
    if not helper_script.exists():
        problems.append(
            "Required helper script is missing from this repo checkout: "
            f"{helper_script}. Sync the latest repo to the VM before rerunning."
        )
    if archive_path is not None and not archive_path.parent.exists():
        problems.append(f"Archive parent directory does not exist: {archive_path.parent}")

    if problems:
        for problem in problems:
            _log(problem, "ERROR")
        raise SystemExit("Preflight checks failed; fix the issues above and rerun.")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Preflight checks passed. Helper script found at {helper_script}", "SUCCESS")


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
    script = _trainstyle_helper_script()
    if not script.exists():
        raise SystemExit(
            "Required helper script is missing: "
            f"{script}. Sync the latest repo to this machine before rerunning."
        )
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
    _log("Launching train-style MAT generation:", "PROGRESS")
    _log(" ".join(cmd), "INFO")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, env=env)


def _metadata_rows_from_mats(
    mat_dir: Path,
    bundle_dir: Path,
) -> Tuple[List[Dict[str, object]], Optional[str], Optional[str]]:
    rows: List[Dict[str, object]] = []
    min_dt = None
    max_dt = None
    mat_paths = sorted(mat_dir.glob("*.mat"))
    if mat_paths:
        _log(f"Scanning {len(mat_paths):,} MAT files to build metadata", "PROGRESS")
    for _, mat_path in _progress_iter([str(path) for path in mat_paths], desc="Scan MAT metadata"):
        mat_path = Path(mat_path)
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
    _log(f"Creating archive at {archive_path}", "PROGRESS")
    _log(" ".join(cmd), "INFO")
    subprocess.run(cmd, check=True)


def main() -> None:
    _load_env_file(REPO_ROOT / ".env")

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

    _print_header("PART 2 VM PREP START")
    _log(f"Workbook: {workbook}", "INFO")
    _log(f"Audio cache dir: {audio_dir}", "INFO")
    _log(f"Dataset doc: {dataset_doc}", "INFO")
    _log(f"Output bundle: {bundle_dir}", "INFO")
    _log(f"Adjacent boundary seconds: {float(args.adjacent_boundary_seconds):.1f}", "INFO")
    _log(f"Include adjacent clips in prep windows: {bool(args.include_adjacent_in_prep)}", "INFO")
    _log(f"Stage selected audio into bundle: {bool(args.stage_selected_audio)}", "INFO")
    _log(f"Download missing audio: {bool(args.download_missing_audio)}", "INFO")
    if args.download_missing_audio:
        onc_token_present = bool(os.getenv(args.onc_token_env, "").strip())
        _log(f"{args.onc_token_env} loaded from environment/.env: {onc_token_present}", "INFO")

    _print_header("PHASE 0: PREFLIGHT CHECKS")
    _preflight_or_die(
        workbook=workbook,
        audio_dir=audio_dir,
        dataset_doc=dataset_doc,
        bundle_dir=bundle_dir,
        archive_path=Path(args.archive_path) if args.archive_path else None,
    )

    _print_header("PHASE 1: BUILD MANIFESTS")
    manifest_started = time.monotonic()
    _log("Parsing workbook and normalizing Part 2 manifests. This is usually quick.", "PROGRESS")
    manifests = build_part2_manifests(
        workbook,
        adjacent_boundary_seconds=max(0.0, float(args.adjacent_boundary_seconds)),
        include_adjacent_in_prep=bool(args.include_adjacent_in_prep),
        seed=int(args.seed),
    )
    write_part2_manifests(manifests_dir, manifests)
    summary = manifests["summary"]
    _log(
        "Manifest summary: "
        f"candidate={summary['candidate_clip_count']:,}, "
        f"adjacent_context={summary['adjacent_context_clip_count']:,}, "
        f"download={summary['download_clip_count']:,}, "
        f"prep={summary['prep_clip_count']:,}",
        "SUCCESS",
    )
    _log(f"Manifest phase finished in {_format_duration(time.monotonic() - manifest_started)}", "SUCCESS")
    candidate_clip_names = [row["filename"] for row in manifests["candidate_clips"]]
    adjacent_clip_names = [row["filename"] for row in manifests.get("adjacent_context_clips", [])]
    download_clip_names = [row["filename"] for row in manifests.get("download_clips", manifests["candidate_clips"])]
    prep_clip_names = [row["filename"] for row in manifests.get("prep_clips", manifests["candidate_clips"])]
    prep_clip_list = manifests_dir / "prep_clips.txt"
    required_audio_names = set(prep_clip_names)
    optional_audio_names = set(download_clip_names) - required_audio_names

    copied_audio: List[str] = []
    downloaded_audio: List[str] = []
    download_failures: List[str] = []
    reused_existing_audio = 0
    prep_audio_dir = raw_audio_dir if args.stage_selected_audio else audio_dir
    staging_target_dir = raw_audio_dir if args.stage_selected_audio else audio_dir

    _print_header("PHASE 2: RESOLVE RAW AUDIO")
    if args.stage_selected_audio:
        copied_audio, missing_before_download, reused_existing_audio = _copy_selected_audio(
            download_clip_names,
            audio_dir,
            raw_audio_dir,
        )
    else:
        _log(
            f"Using audio in place from {audio_dir}; checking {len(download_clip_names):,} required clips",
            "PROGRESS",
        )
        indexed_audio = _index_audio(audio_dir)
        missing_before_download = [name for name in download_clip_names if name not in indexed_audio]
        _log(
            f"In-place audio check finished: {len(download_clip_names) - len(missing_before_download):,} present, "
            f"{len(missing_before_download):,} missing",
            "SUCCESS" if not missing_before_download else "WARNING",
        )

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
    elif missing_before_download:
        _log(
            f"{len(missing_before_download):,} clips are missing locally and downloads are disabled.",
            "WARNING",
        )

    missing_path = bundle_dir / "missing_audio.txt"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with open(missing_path, "w", encoding="utf-8") as handle:
        for clip_name in missing_audio:
            handle.write(f"{clip_name}\n")

    missing_required_audio = [clip_name for clip_name in missing_audio if clip_name in required_audio_names]
    missing_optional_audio = [clip_name for clip_name in missing_audio if clip_name in optional_audio_names]
    missing_required_path = bundle_dir / "missing_required_audio.txt"
    with open(missing_required_path, "w", encoding="utf-8") as handle:
        for clip_name in missing_required_audio:
            handle.write(f"{clip_name}\n")
    missing_optional_path = bundle_dir / "missing_optional_adjacent_audio.txt"
    with open(missing_optional_path, "w", encoding="utf-8") as handle:
        for clip_name in missing_optional_audio:
            handle.write(f"{clip_name}\n")

    downloaded_path = bundle_dir / "downloaded_audio.txt"
    with open(downloaded_path, "w", encoding="utf-8") as handle:
        for clip_name in downloaded_audio:
            handle.write(f"{clip_name}\n")

    failed_downloads_path = bundle_dir / "failed_downloads.txt"
    with open(failed_downloads_path, "w", encoding="utf-8") as handle:
        for clip_name in download_failures:
            handle.write(f"{clip_name}\n")

    _log(
        f"Audio resolution summary: copied_or_found={len(copied_audio):,}, reused_existing={reused_existing_audio:,}, "
        f"downloaded={len(downloaded_audio):,}, still_missing={len(missing_audio):,} "
        f"(required={len(missing_required_audio):,}, optional_adjacent={len(missing_optional_audio):,})",
        "SUCCESS" if not missing_required_audio else "WARNING",
    )
    _log(f"Missing list: {missing_path}", "INFO")
    _log(f"Missing required list: {missing_required_path}", "INFO")
    _log(f"Missing optional adjacent list: {missing_optional_path}", "INFO")
    _log(f"Downloaded list: {downloaded_path}", "INFO")
    _log(f"Failed downloads list: {failed_downloads_path}", "INFO")

    if missing_required_audio:
        raise SystemExit(
            f"Missing {len(missing_required_audio)} required prep clips. "
            f"See {missing_required_path} before running the heavy prep step."
        )
    if missing_optional_audio:
        _log(
            f"{len(missing_optional_audio):,} adjacent-context clips are still missing. "
            "Prep will continue; boundary windows that need those clips will fall back to "
            "the existing zero-padding behavior.",
            "WARNING",
        )

    if not args.skip_prep:
        _print_header("PHASE 3: GENERATE MAT WINDOWS")
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
    else:
        _log("Skipping MAT generation because --skip-prep was provided.", "WARNING")

    _print_header("PHASE 4: WRITE METADATA")
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
            "reused_existing_audio_count": reused_existing_audio,
            "downloaded_audio_count": len(downloaded_audio),
            "staged_audio_count": len(copied_audio) + len(downloaded_audio) if args.stage_selected_audio else len(download_clip_names) - len(missing_audio),
            "mat_count": len(metadata_rows),
            "window_s": float(args.window_s),
            "step_s": float(args.step_s),
            "missing_required_audio_count": len(missing_required_audio),
            "missing_optional_adjacent_audio_count": len(missing_optional_audio),
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
        "reused_existing_audio_count": reused_existing_audio,
        "downloaded_audio_count": len(downloaded_audio),
        "download_failure_count": len(download_failures),
        "staged_audio_count": len(copied_audio) + len(downloaded_audio) if args.stage_selected_audio else len(download_clip_names) - len(missing_audio),
        "mat_count": len(metadata_rows),
        "missing_audio_count": len(missing_audio),
        "missing_required_audio_count": len(missing_required_audio),
        "missing_optional_adjacent_audio_count": len(missing_optional_audio),
        "download_missing_audio": bool(args.download_missing_audio),
        "include_adjacent_in_prep": bool(args.include_adjacent_in_prep),
        "adjacent_boundary_seconds": float(args.adjacent_boundary_seconds),
        "archive_path": args.archive_path or "",
    }
    with open(bundle_dir / "prep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(prep_summary, handle, indent=2, sort_keys=True)
    _log(f"Wrote metadata: {metadata_path}", "SUCCESS")
    _log(f"Wrote prep summary: {bundle_dir / 'prep_summary.json'}", "SUCCESS")

    if args.archive_path:
        _print_header("PHASE 5: CREATE ARCHIVE")
        _create_archive(bundle_dir, Path(args.archive_path))

    _print_header("PART 2 VM PREP COMPLETE")
    _log(f"bundle_dir: {bundle_dir}", "SUCCESS")
    _log(f"manifests: {manifests_dir}", "SUCCESS")
    _log(f"raw_audio: {prep_audio_dir}", "SUCCESS")
    _log(f"candidate_clips: {len(candidate_clip_names)}", "SUCCESS")
    _log(f"adjacent_context_clips: {len(adjacent_clip_names)}", "SUCCESS")
    _log(f"prep_clips: {len(prep_clip_names)}", "SUCCESS")
    _log(f"copied_audio: {len(copied_audio)}", "SUCCESS")
    _log(f"downloaded_audio: {len(downloaded_audio)}", "SUCCESS")
    _log(f"mat_files: {mat_dir}", "SUCCESS")
    _log(f"metadata: {metadata_path}", "SUCCESS")
    if args.archive_path:
        _log(f"archive: {args.archive_path}", "SUCCESS")


if __name__ == "__main__":
    main()
