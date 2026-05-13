#!/usr/bin/env python3
"""Create a reusable tar archive for manifests with low/mid/high MAT columns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_PATH_COLUMNS = ("low_mat_path", "mid_mat_path", "high_mat_path")


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


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _resolve_path(value: str, dataset_root: Optional[Path]) -> Path:
    path = Path(value)
    if path.is_absolute() or dataset_root is None:
        return path
    return (dataset_root / path).resolve()


def _member_name(path: Path, column: str, prefix: str) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    band = column.removesuffix("_mat_path")
    safe = path.name.replace("/", "_")
    return f"{prefix.strip('/')}/{band}/{digest}__{safe}"


def build_archive_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    path_columns: Sequence[str],
    dataset_root: Optional[Path],
    member_prefix: str,
    allow_missing: bool,
) -> Tuple[List[Dict[str, Any]], Dict[Path, str], Dict[str, Any]]:
    path_to_member: Dict[Path, str] = {}
    missing: List[Dict[str, str]] = []
    remapped: List[Dict[str, Any]] = []
    skipped = 0
    for row_idx, raw in enumerate(rows, start=2):
        row = dict(raw)
        row_missing = False
        for column in path_columns:
            value = _clean(raw.get(column))
            if not value:
                missing.append({"row": str(row_idx), "column": column, "path": "<blank>"})
                row_missing = True
                continue
            path = _resolve_path(value, dataset_root)
            if not path.exists():
                missing.append({"row": str(row_idx), "column": column, "path": str(path)})
                row_missing = True
                continue
            if path not in path_to_member:
                path_to_member[path] = _member_name(path, column, member_prefix)
            row[column] = path_to_member[path]
        if row_missing:
            if allow_missing:
                skipped += 1
                continue
            break
        if "mat_path" in row and "low_mat_path" in row:
            row["mat_path"] = row["low_mat_path"]
        remapped.append(row)

    if missing and not allow_missing:
        examples = "\n".join(f"row {item['row']} {item['column']}: {item['path']}" for item in missing[:10])
        raise FileNotFoundError(f"{len(missing)} multiband MAT references are missing; examples:\n{examples}")
    total_bytes = sum(path.stat().st_size for path in path_to_member)
    summary = {
        "input_row_count": len(rows),
        "output_row_count": len(remapped),
        "skipped_row_count": int(skipped),
        "unique_mat_count": len(path_to_member),
        "path_columns": list(path_columns),
        "member_prefix": member_prefix,
        "missing_reference_count": len(missing),
        "missing_reference_examples": missing[:25],
        "total_mat_bytes": int(total_bytes),
    }
    return remapped, path_to_member, summary


def create_multiband_archive(
    *,
    manifest_csv: Path,
    output_dir: Path,
    archive_path: Path,
    path_columns: Sequence[str] = DEFAULT_PATH_COLUMNS,
    dataset_root: Optional[Path] = None,
    archive_format: str = "tar",
    member_prefix: str = "mat_files",
    remapped_manifest_name: str = "archive_manifest.csv",
    summary_name: str = "archive_summary.json",
    allow_missing: bool = False,
    include_files: Sequence[Path] = (),
) -> Dict[str, Any]:
    rows, fieldnames = _read_csv(manifest_csv)
    for column in path_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    remapped_rows, path_to_member, summary = build_archive_plan(
        rows,
        path_columns=path_columns,
        dataset_root=dataset_root,
        member_prefix=member_prefix,
        allow_missing=allow_missing,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    remapped_manifest = output_dir / remapped_manifest_name
    summary_path = output_dir / summary_name
    member_map_path = output_dir / "archive_members.json"
    _write_csv(remapped_manifest, remapped_rows, fieldnames)
    member_map = {str(path.resolve()): member for path, member in sorted(path_to_member.items(), key=lambda item: item[1])}
    member_map_path.write_text(json.dumps(member_map, indent=2, sort_keys=True), encoding="utf-8")

    summary.update(
        {
            "manifest_csv": str(manifest_csv.resolve()),
            "dataset_root": "" if dataset_root is None else str(dataset_root.resolve()),
            "archive_path": str(archive_path.resolve()),
            "archive_format": archive_format,
            "remapped_manifest": str(remapped_manifest.resolve()),
            "summary_json": str(summary_path.resolve()),
            "archive_members_json": str(member_map_path.resolve()),
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
        tar.add(member_map_path, arcname=member_map_path.name, recursive=False)
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
    parser.add_argument("--path-columns", default=",".join(DEFAULT_PATH_COLUMNS))
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--archive-format", choices=["tar", "tar.gz"], default="tar")
    parser.add_argument("--member-prefix", default="mat_files")
    parser.add_argument("--remapped-manifest-name", default="archive_manifest.csv")
    parser.add_argument("--summary-name", default="archive_summary.json")
    parser.add_argument("--include-file", action="append", default=[])
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    summary = create_multiband_archive(
        manifest_csv=Path(args.manifest_csv),
        output_dir=Path(args.output_dir),
        archive_path=Path(args.archive_path),
        path_columns=[token.strip() for token in str(args.path_columns).split(",") if token.strip()],
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
