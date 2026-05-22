#!/usr/bin/env python3
"""Build unreviewed month sliding-window rows for deployment inference.

This mirrors the target-selection step used by the fin-whale high-confidence
month job, but emits 10 s centered windows that can be expanded to 40 s
multiband contexts for the E24 expert ensemble.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TS_RE = re.compile(r"^(?P<device>[^_]+)_(?P<ts>\d{8}T\d{6})(?:\.\d+)?Z")


def file_key(name: str) -> Optional[str]:
    match = TS_RE.search(Path(name).name)
    if not match:
        return None
    return f"{match.group('device')}_{match.group('ts')}"


def file_ts(name: str) -> Optional[datetime]:
    match = TS_RE.search(Path(name).name)
    if not match:
        return None
    return datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def month_bounds(month: str) -> Tuple[datetime, datetime]:
    year, mon = [int(part) for part in month.split("-")]
    start = datetime(year, mon, 1, tzinfo=timezone.utc)
    if mon == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, mon + 1, 1, tzinfo=timezone.utc)
    return start, end


def reviewed_clip_keys(
    *,
    workbook: Path,
    device: str,
    month_start: datetime,
    month_end: datetime,
) -> Tuple[set[str], set[str]]:
    from src.dataset.part2_annotations import load_workbook_sheets, normalize_audio_filename

    keys: set[str] = set()
    names: set[str] = set()
    for sheet in load_workbook_sheets(workbook):
        if sheet.name == "READ ME":
            continue
        for row in sheet.rows:
            name = normalize_audio_filename(row.get("filename", ""))
            if not name or not name.startswith(device + "_"):
                continue
            ts = file_ts(name)
            if ts is None or not (month_start <= ts < month_end):
                continue
            key = file_key(name)
            if key:
                keys.add(key)
            names.add(Path(name).name)
    return keys, names


def read_available(path: Path, device: str) -> Dict[str, Tuple[datetime, str]]:
    out: Dict[str, Tuple[datetime, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            name = Path(raw.strip()).name
            if not name or not name.startswith(device + "_"):
                continue
            ts = file_ts(name)
            key = file_key(name)
            if ts is None or key is None:
                continue
            out.setdefault(key, (ts, name))
    return out


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(str(value) + "\n")


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def build_rows(
    *,
    target: List[Tuple[datetime, str]],
    month: str,
    device: str,
    crop_seconds: float,
    step_seconds: float,
    clip_seconds: float,
) -> List[dict[str, str]]:
    rows: List[dict[str, str]] = []
    first_center = crop_seconds / 2.0
    last_center = clip_seconds - (crop_seconds / 2.0)
    centers: List[float] = []
    cur = first_center
    while cur <= last_center + 1e-6:
        centers.append(round(cur, 6))
        cur += step_seconds

    for clip_ts, name in target:
        clip_stem = Path(name).stem
        for center_s in centers:
            begin_s = max(0.0, center_s - (crop_seconds / 2.0))
            end_s = min(clip_seconds, center_s + (crop_seconds / 2.0))
            abs_begin = clip_ts + timedelta(seconds=begin_s)
            abs_end = clip_ts + timedelta(seconds=end_s)
            item_id = safe_id(f"{clip_stem}__center_{center_s:07.2f}s")
            rows.append(
                {
                    "item_id": item_id,
                    "clip": name,
                    "filename": name,
                    "source_audio": name,
                    "raw_audio_path": f"raw_audio/{name}",
                    "begin_s": f"{begin_s:.6f}",
                    "end_s": f"{end_s:.6f}",
                    "window_center_s": f"{center_s:.6f}",
                    "window_step_s": f"{step_seconds:.6f}",
                    "crop_seconds": f"{crop_seconds:.6f}",
                    "clip_seconds": f"{clip_seconds:.6f}",
                    "absolute_begin_time": abs_begin.isoformat(),
                    "absolute_end_time": abs_end.isoformat(),
                    "month": month,
                    "device": device,
                    "source_kind": "ONC",
                    "source_dataset": "ONC_Clayoquot_unreviewed_month",
                    "split": "inference",
                    "label_ids": "",
                    "target_label_ids": "",
                    "canonical_label_ids": "",
                    "source_label_ids": "",
                    "analysis_label_ids": "",
                    "is_background": "",
                    "review_status": "unreviewed",
                    "negative_bucket": "",
                    "context_tags": "deployment_unreviewed_sliding_window",
                    "event_group": "",
                }
            )
    return rows


def write_csv(path: Path, rows: List[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, help="Month as YYYY-MM")
    parser.add_argument("--device-code", default="ICLISTENHF6016")
    parser.add_argument("--available-filenames", required=True, type=Path)
    parser.add_argument("--reviewed-workbook", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument("--adjacent-minutes", type=float, default=5.0)
    parser.add_argument("--clip-seconds", type=float, default=300.0)
    parser.add_argument("--crop-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=10.0)
    parser.add_argument("--max-target-files", type=int, default=0, help="Optional smoke-test cap after reviewed-file exclusion")
    args = parser.parse_args()

    month_start, month_end = month_bounds(args.month)
    available = read_available(args.available_filenames, args.device_code)
    reviewed_keys: set[str] = set()
    reviewed_names: set[str] = set()
    if not args.include_reviewed:
        if args.reviewed_workbook is None:
            raise SystemExit("--reviewed-workbook is required unless --include-reviewed is set")
        reviewed_keys, reviewed_names = reviewed_clip_keys(
            workbook=args.reviewed_workbook,
            device=args.device_code,
            month_start=month_start,
            month_end=month_end,
        )

    all_month: List[Tuple[datetime, str]] = []
    target: List[Tuple[datetime, str]] = []
    excluded: List[Tuple[datetime, str]] = []
    for key, value in available.items():
        ts, name = value
        if not (month_start <= ts < month_end):
            continue
        all_month.append(value)
        if not args.include_reviewed and (key in reviewed_keys or name in reviewed_names):
            excluded.append(value)
        else:
            target.append(value)

    all_month = sorted(set(all_month))
    target = sorted(set(target))
    excluded = sorted(set(excluded))
    uncapped_target_count = len(target)
    if int(args.max_target_files) > 0:
        target = target[: int(args.max_target_files)]
    if not all_month:
        raise SystemExit(f"No available month audio files found for {args.device_code} {args.month}")
    if not target:
        raise SystemExit(f"All {len(all_month)} month files were excluded as reviewed")

    selected_by_key: Dict[str, Tuple[datetime, str]] = {}
    adjacent = timedelta(minutes=float(args.adjacent_minutes))
    for ts, name in target:
        key = file_key(name)
        if key:
            selected_by_key[key] = (ts, name)
        for delta in (-adjacent, adjacent):
            adj_ts = ts + delta
            adj_key = f"{args.device_code}_{adj_ts.strftime('%Y%m%dT%H%M%S')}"
            if adj_key in available:
                selected_by_key.setdefault(adj_key, available[adj_key])
    selected = sorted(set(selected_by_key.values()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.out_dir
    write_lines(manifest_dir / "target_clip_list_all_available.txt", [name for _, name in all_month])
    write_lines(manifest_dir / "target_clip_list.txt", [name for _, name in target])
    write_lines(manifest_dir / "excluded_reviewed_clip_list.txt", [name for _, name in excluded])
    write_lines(manifest_dir / "reviewed_workbook_clip_keys.txt", sorted(reviewed_keys))
    write_lines(manifest_dir / "selected_archive_members.txt", [f"raw_audio/{name}" for _, name in selected])
    write_lines(manifest_dir / "selected_filenames.txt", [name for _, name in selected])
    rows = build_rows(
        target=target,
        month=args.month,
        device=args.device_code,
        crop_seconds=float(args.crop_seconds),
        step_seconds=float(args.step_seconds),
        clip_seconds=float(args.clip_seconds),
    )
    write_csv(manifest_dir / "sliding_windows.csv", rows)

    summary = {
        "month": args.month,
        "device": args.device_code,
        "include_reviewed": bool(args.include_reviewed),
        "available_month_files": len(all_month),
        "reviewed_or_examined_files_excluded": len(excluded),
        "uncapped_unreviewed_target_files": uncapped_target_count,
        "unreviewed_target_files": len(target),
        "max_target_files": int(args.max_target_files),
        "selected_files_including_adjacent": len(selected),
        "sliding_window_rows": len(rows),
        "crop_seconds": float(args.crop_seconds),
        "step_seconds": float(args.step_seconds),
        "target_start": target[0][0].isoformat(),
        "target_end": target[-1][0].isoformat(),
        "sliding_windows_csv": str(manifest_dir / "sliding_windows.csv"),
    }
    (manifest_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (manifest_dir / "selection_summary.txt").open("w", encoding="utf-8") as handle:
        for key, value in summary.items():
            handle.write(f"{key}={value}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
