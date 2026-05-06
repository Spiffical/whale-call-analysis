#!/usr/bin/env python3
"""Create a reusable MAT cache archive from a manifest.

The training manifests usually contain absolute MAT paths created by a prep
job. This utility packages the referenced MAT files into one tar archive and
writes a remapped manifest whose ``mat_path`` values are relative archive
members. After extraction, pass the extraction directory as ``--dataset-root``
to the training script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _resolve_mat_path(value: str, dataset_root: Optional[Path]) -> Path:
    path = Path(value)
    if path.is_absolute() or dataset_root is None:
        return path
    return (dataset_root / path).resolve()


def _member_name(path: Path, prefix: str) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    safe_name = path.name.replace("/", "_")
    return f"{prefix.strip('/')}/{digest}__{safe_name}"


def build_archive_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_root: Optional[Path] = None,
    member_prefix: str = "mat_files",
    allow_missing: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[Path, str], Dict[str, Any]]:
    path_to_member: Dict[Path, str] = {}
    missing_paths: List[str] = []
    skipped_row_count = 0
    remapped_rows: List[Dict[str, Any]] = []

    for raw in rows:
        mat_text = _clean_text(raw.get("mat_path"))
        if not mat_text:
            if allow_missing:
                skipped_row_count += 1
                continue
            raise ValueError("Manifest row is missing mat_path")
        mat_path = _resolve_mat_path(mat_text, dataset_root)
        if not mat_path.exists():
            missing_paths.append(str(mat_path))
            if allow_missing:
                skipped_row_count += 1
                continue
        else:
            path_to_member.setdefault(mat_path, _member_name(mat_path, member_prefix))
            row = dict(raw)
            row["mat_path"] = path_to_member[mat_path]
            remapped_rows.append(row)

    if missing_paths and not allow_missing:
        sample = "\n".join(missing_paths[:10])
        raise FileNotFoundError(f"{len(missing_paths)} MAT files are missing; examples:\n{sample}")

    total_bytes = sum(path.stat().st_size for path in path_to_member)
    summary = {
        "input_row_count": len(rows),
        "output_row_count": len(remapped_rows),
        "skipped_row_count": int(skipped_row_count),
        "unique_mat_count": len(path_to_member),
        "duplicate_mat_reference_count": max(0, len(remapped_rows) - len(path_to_member)),
        "missing_mat_count": len(missing_paths),
        "missing_mat_examples": missing_paths[:10],
        "member_prefix": member_prefix,
        "total_mat_bytes": int(total_bytes),
    }
    return remapped_rows, path_to_member, summary


def create_mat_archive(
    *,
    manifest_csv: Path,
    output_dir: Path,
    archive_path: Path,
    dataset_root: Optional[Path] = None,
    archive_format: str = "tar",
    member_prefix: str = "mat_files",
    remapped_manifest_name: str = "archive_manifest.csv",
    summary_name: str = "archive_summary.json",
    include_files: Sequence[Path] = (),
    allow_missing: bool = False,
) -> Dict[str, Any]:
    rows, fieldnames = _read_csv(manifest_csv)
    if "mat_path" not in fieldnames:
        fieldnames = [*fieldnames, "mat_path"]
    remapped_rows, path_to_member, summary = build_archive_plan(
        rows,
        dataset_root=dataset_root,
        member_prefix=member_prefix,
        allow_missing=allow_missing,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    remapped_manifest = output_dir / remapped_manifest_name
    summary_path = output_dir / summary_name
    _write_csv(remapped_manifest, remapped_rows, fieldnames)

    summary.update(
        {
            "manifest_csv": str(manifest_csv.resolve()),
            "dataset_root": "" if dataset_root is None else str(dataset_root.resolve()),
            "archive_path": str(archive_path.resolve()),
            "archive_format": archive_format,
            "remapped_manifest": str(remapped_manifest.resolve()),
            "summary_json": str(summary_path.resolve()),
            "include_files": [str(path.resolve()) for path in include_files],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    mode = {"tar": "w", "tar.gz": "w:gz"}[archive_format]
    tmp_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp_archive.exists():
        tmp_archive.unlink()
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(tmp_archive, mode) as tar:
        for path, member in sorted(path_to_member.items(), key=lambda item: item[1]):
            tar.add(path, arcname=member, recursive=False)
        tar.add(remapped_manifest, arcname=remapped_manifest.name, recursive=False)
        tar.add(summary_path, arcname=summary_path.name, recursive=False)
        for include_path in include_files:
            if include_path.exists():
                tar.add(include_path, arcname=include_path.name, recursive=False)
    tmp_archive.rename(archive_path)

    summary["archive_bytes"] = int(archive_path.stat().st_size)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--archive-format", choices=["tar", "tar.gz"], default="tar")
    parser.add_argument("--member-prefix", default="mat_files")
    parser.add_argument("--remapped-manifest-name", default="archive_manifest.csv")
    parser.add_argument("--summary-name", default="archive_summary.json")
    parser.add_argument("--include-file", action="append", default=[])
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    summary = create_mat_archive(
        manifest_csv=Path(args.manifest_csv),
        output_dir=Path(args.output_dir),
        archive_path=Path(args.archive_path),
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        archive_format=str(args.archive_format),
        member_prefix=str(args.member_prefix),
        remapped_manifest_name=str(args.remapped_manifest_name),
        summary_name=str(args.summary_name),
        include_files=[Path(path) for path in args.include_file],
        allow_missing=bool(args.allow_missing),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
