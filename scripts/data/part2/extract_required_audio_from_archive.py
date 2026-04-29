#!/usr/bin/env python3
"""Extract a flat subset of required raw-audio clips from the canonical Nibi archive."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CANONICAL_NIBI_RAW_AUDIO_ARCHIVE = Path(
    "/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio.tar.zst"
)
CANONICAL_NIBI_AVAILABLE_FILENAMES = Path(
    "/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio_available_filenames.txt"
)


def _read_lines(path: Path) -> List[str]:
    values: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            value = raw_line.strip()
            if value:
                values.append(value)
    return values


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _archive_list_command(archive_path: Path) -> List[str]:
    name = archive_path.name.lower()
    if name.endswith(".tar.zst"):
        return ["tar", "--use-compress-program=unzstd", "-tf", str(archive_path)]
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return ["tar", "-tzf", str(archive_path)]
    if name.endswith(".tar"):
        return ["tar", "-tf", str(archive_path)]
    raise SystemExit(f"Unsupported archive format: {archive_path}")


def _archive_extract_command(archive_path: Path, member_list_path: Path, output_dir: Path) -> List[str]:
    name = archive_path.name.lower()
    if name.endswith(".tar.zst"):
        return [
            "tar",
            "--use-compress-program=unzstd",
            "-xf",
            str(archive_path),
            "-C",
            str(output_dir),
            "-T",
            str(member_list_path),
        ]
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return ["tar", "-xzf", str(archive_path), "-C", str(output_dir), "-T", str(member_list_path)]
    if name.endswith(".tar"):
        return ["tar", "-xf", str(archive_path), "-C", str(output_dir), "-T", str(member_list_path)]
    raise SystemExit(f"Unsupported archive format: {archive_path}")


def _iter_archive_members(archive_path: Path) -> Iterator[str]:
    proc = subprocess.Popen(
        _archive_list_command(archive_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            member = raw_line.rstrip("\n")
            if member:
                yield member
    finally:
        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read()
        rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"Archive listing failed for {archive_path}:\n{stderr_text.strip()}")


def _map_required_members(archive_path: Path, required_filenames: Sequence[str]) -> Dict[str, str]:
    required_set = set(required_filenames)
    member_map: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}
    for member in _iter_archive_members(archive_path):
        basename = Path(member).name
        if basename not in required_set:
            continue
        if basename in member_map and member_map[basename] != member:
            duplicates.setdefault(basename, [member_map[basename]]).append(member)
            continue
        member_map[basename] = member
    if duplicates:
        problems = "\n".join(
            f"  {basename}: {', '.join(paths)}" for basename, paths in sorted(duplicates.items())
        )
        raise SystemExit(
            "Archive contains duplicate members for one or more required basenames.\n"
            "Please disambiguate before extraction:\n"
            f"{problems}"
        )
    return member_map


def _move_extracted_members(
    *,
    temp_dir: Path,
    member_map: Dict[str, str],
    output_dir: Path,
    overwrite: bool,
) -> List[str]:
    moved: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for basename, member in sorted(member_map.items()):
        extracted_path = temp_dir / member
        if not extracted_path.exists():
            raise SystemExit(f"Expected extracted member missing: {extracted_path}")
        destination = output_dir / basename
        if destination.exists():
            if not overwrite:
                moved.append(basename)
                continue
            destination.unlink()
        shutil.move(str(extracted_path), str(destination))
        moved.append(basename)
    return moved


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract required final-2025 raw-audio clips from the canonical archive")
    ap.add_argument(
        "--archive-path",
        type=str,
        default=str(CANONICAL_NIBI_RAW_AUDIO_ARCHIVE),
        help="Raw-audio archive to read from",
    )
    ap.add_argument("--required-filenames-txt", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True, help="Flat output directory for extracted clips")
    ap.add_argument(
        "--available-filenames-txt",
        type=str,
        default=str(CANONICAL_NIBI_AVAILABLE_FILENAMES) if CANONICAL_NIBI_AVAILABLE_FILENAMES.exists() else None,
        help="Optional available-files manifest to check before archive extraction",
    )
    ap.add_argument("--allow-missing", action="store_true", help="Do not fail if required clips are unavailable")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite already extracted destination files")
    args = ap.parse_args()

    archive_path = Path(args.archive_path).resolve()
    required_txt = Path(args.required_filenames_txt).resolve()
    output_dir = Path(args.output_dir).resolve()
    available_txt = Path(args.available_filenames_txt).resolve() if args.available_filenames_txt else None

    if not archive_path.exists():
        raise SystemExit(f"Archive not found: {archive_path}")
    if not required_txt.exists():
        raise SystemExit(f"Required filename list not found: {required_txt}")
    if available_txt is not None and not available_txt.exists():
        raise SystemExit(f"Available filename list not found: {available_txt}")

    required_filenames = sorted(dict.fromkeys(_read_lines(required_txt)))
    if not required_filenames:
        raise SystemExit(f"No filenames found in required list: {required_txt}")

    available_set = set(_read_lines(available_txt)) if available_txt is not None else None
    available_required = (
        [name for name in required_filenames if name in available_set] if available_set is not None else list(required_filenames)
    )
    missing_from_available = (
        [name for name in required_filenames if name not in available_set] if available_set is not None else []
    )

    member_map = _map_required_members(archive_path, available_required)
    extractable = sorted(member_map)
    missing_from_archive = [name for name in available_required if name not in member_map]
    missing_filenames = sorted(dict.fromkeys(missing_from_available + missing_from_archive))

    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir = output_dir / "_archive_extract"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(output_dir / "required_audio_filenames.txt", required_filenames)
    _write_lines(output_dir / "available_required_audio_filenames.txt", available_required)
    _write_lines(output_dir / "extractable_audio_filenames.txt", extractable)
    _write_lines(output_dir / "missing_audio_filenames.txt", missing_filenames)

    extracted_filenames: List[str] = []
    if extractable:
        with tempfile.TemporaryDirectory(prefix="clayoquot_extract_", dir=str(extraction_dir)) as tmpdir:
            temp_dir = Path(tmpdir)
            member_list_path = temp_dir / "archive_members.txt"
            _write_lines(member_list_path, [member_map[name] for name in extractable])
            subprocess.run(
                _archive_extract_command(archive_path, member_list_path, temp_dir),
                check=True,
            )
            extracted_filenames = _move_extracted_members(
                temp_dir=temp_dir,
                member_map=member_map,
                output_dir=output_dir,
                overwrite=bool(args.overwrite),
            )

    summary = {
        "archive_path": str(archive_path),
        "required_filenames_txt": str(required_txt),
        "available_filenames_txt": str(available_txt) if available_txt is not None else None,
        "required_count": len(required_filenames),
        "available_required_count": len(available_required),
        "extractable_count": len(extractable),
        "extracted_count": len(extracted_filenames),
        "missing_count": len(missing_filenames),
        "missing_filenames": missing_filenames,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"required_count={len(required_filenames)}")
    print(f"extractable_count={len(extractable)}")
    print(f"extracted_count={len(extracted_filenames)}")
    print(f"missing_count={len(missing_filenames)}")
    print(f"output_dir={output_dir}")

    if missing_filenames and not args.allow_missing:
        raise SystemExit(
            "One or more required clips are unavailable. "
            f"See {output_dir / 'missing_audio_filenames.txt'} for details."
        )


if __name__ == "__main__":
    main()
