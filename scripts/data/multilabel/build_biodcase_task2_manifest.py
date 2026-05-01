#!/usr/bin/env python3
"""Build train-style manifests for BioDCASE 2026 Task 2 / ATBFL data.

BioDCASE Task 2 annotations are strongly labelled sound events with absolute
start/end datetimes and call labels in {bma, bmb, bmz, bmd, bpd, bp20,
bp20plus}. This converter maps those rows into the same call-centered CSV shape
used by scripts/data/part2/prepare_trainstyle_windows.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    build_vocabulary_from_rows,
    call_type_display_name,
    clean_text,
    label_ids_from_row,
    normalize_call_type,
    species_display_name,
    write_csv_rows,
)


CALL_TO_SPECIES = {
    "BmA": "Bm",
    "BmB": "Bm",
    "BmZ": "Bm",
    "BmD": "Bm",
    "Bp20": "Bp",
    "Bp20plus": "Bp",
    "BpD": "Bp",
}

AUDIO_COLUMNS = (
    "filename",
    "file",
    "audio_filename",
    "audio_file",
    "recording",
    "recording_filename",
    "recording_id",
)
DATASET_COLUMNS = ("dataset", "site_year", "deployment", "site", "source_dataset")
LABEL_COLUMNS = ("annotation", "label", "event_label", "call_type", "call_type_raw", "class")
START_COLUMNS = ("start_s", "begin_s", "begin_time_s", "start_time_s", "event_start_s", "onset_s")
END_COLUMNS = ("end_s", "end_time_s", "event_end_s", "offset_s")
DURATION_COLUMNS = ("duration_s", "duration", "event_duration_s")
START_DATETIME_COLUMNS = ("start_datetime", "start_time", "begin_datetime", "event_start_datetime")
END_DATETIME_COLUMNS = ("end_datetime", "end_time", "event_end_datetime")
LOW_FREQ_COLUMNS = ("low_frequency", "low_freq", "freq_low", "fmin")
HIGH_FREQ_COLUMNS = ("high_frequency", "high_freq", "freq_high", "fmax")
CONFIDENCE_COLUMNS = ("confidence", "detection_confidence", "score")


def _read_table(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:4096]
    if path.suffix.lower() in {".tsv", ".tab"}:
        delimiter = "\t"
    else:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def _first(row: Dict[str, Any], names: Sequence[str]) -> str:
    lower = {str(key).lower(): key for key in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            value = clean_text(row.get(key))
            if value:
                return value
    return ""


def _float_or_none(value: Any) -> Optional[float]:
    try:
        text = clean_text(value)
        return None if not text else float(text)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_audio_start(filename: str) -> Optional[datetime]:
    stem = Path(filename).stem
    patterns = (
        # BioDCASE/ATBFL example: 2015-02-04T03-00-00_000.wav
        (r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})", "%Y-%m-%dT%H:%M:%S"),
        # ONC-style or common recorder filename: DEVICE_20250101T123456...
        (r"(?P<stamp>\d{8}T\d{6})", "%Y%m%dT%H%M%S"),
        # ISO-like filename where colons survived.
        (r"(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", "%Y-%m-%dT%H:%M:%S"),
    )
    for pattern, fmt in patterns:
        match = re.search(pattern, stem)
        if not match:
            continue
        if "stamp" in match.groupdict() and match.group("stamp"):
            text = match.group("stamp")
        else:
            text = f"{match.group('date')}T{match.group('h')}:{match.group('m')}:{match.group('s')}"
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _timing_from_row(row: Dict[str, Any], filename: str) -> Tuple[Optional[float], Optional[float]]:
    begin = _float_or_none(_first(row, START_COLUMNS))
    end = _float_or_none(_first(row, END_COLUMNS))
    if begin is not None and end is None:
        duration = _float_or_none(_first(row, DURATION_COLUMNS))
        if duration is not None:
            end = begin + duration
    if begin is not None and end is not None:
        return begin, end

    event_start = _parse_datetime(_first(row, START_DATETIME_COLUMNS))
    event_end = _parse_datetime(_first(row, END_DATETIME_COLUMNS))
    audio_start = _parse_audio_start(filename)
    if event_start is not None and event_end is not None and audio_start is not None:
        return (event_start - audio_start).total_seconds(), (event_end - audio_start).total_seconds()
    return None, None


def _find_existing_audio(filename: str, dataset: str, audio_root: Optional[Path]) -> Optional[Path]:
    if audio_root is None:
        return None
    candidates = [
        audio_root / dataset / filename,
        audio_root / filename,
        audio_root / Path(filename).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    stem = Path(filename).stem
    for suffix in (Path(filename).suffix, ".wav", ".flac"):
        if not suffix:
            continue
        matches = list(audio_root.rglob(f"{stem}{suffix}"))
        if matches:
            return matches[0]
    return None


def _audio_names_from_list(paths: Sequence[Path]) -> List[str]:
    names: List[str] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = clean_text(line)
            if not text or text.startswith("#"):
                continue
            names.append(text)
    return list(dict.fromkeys(names))


def _audio_names_from_root(audio_root: Optional[Path]) -> List[str]:
    if audio_root is None:
        return []
    names: List[str] = []
    for pattern in ("*.wav", "*.flac"):
        names.extend(str(path.relative_to(audio_root)) for path in sorted(audio_root.rglob(pattern)))
    return names


def _expected_mat_name(filename: str, begin_s: float, end_s: float) -> str:
    return f"{filename}_{float(begin_s):.1f}s_{float(end_s):.1f}s_trainstyle.mat"


def _clip_name(filename: str, dataset: str, mode: str) -> str:
    if mode == "filename":
        return filename
    if mode == "dataset_prefix":
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", clean_text(dataset) or "dataset").strip("-")
        return f"{prefix}__{Path(filename).name}"
    raise ValueError(f"Unknown clip name mode: {mode}")


def _label_record(*, species_code: str, call_type: str, dataset: str, row: Dict[str, Any]) -> Dict[str, Any]:
    confidence = _float_or_none(_first(row, CONFIDENCE_COLUMNS))
    return {
        "species_code": species_code,
        "species": species_display_name(species_code),
        "call_type": call_type,
        "call_type_name": call_type_display_name(call_type),
        "source": dataset or "biodcase2026_task2_atbfl",
        "review_status": "reviewed",
        "confidence": confidence,
        "trainable": True,
    }


def _manifest_row(
    *,
    filename: str,
    clip_name: str,
    dataset: str,
    begin_s: float,
    end_s: float,
    species_code: str,
    call_type: str,
    labels_json: str,
    low_freq_hz: str = "",
    high_freq_hz: str = "",
    is_background: bool = False,
    mat_rel_dir: str = "mat_files",
) -> Dict[str, Any]:
    expected_mat = _expected_mat_name(clip_name, begin_s, end_s)
    labels = "" if is_background else labels_json
    row = {
        "item_id": Path(expected_mat).stem,
        "clip": clip_name,
        "filename": filename,
        "source_audio": filename,
        "begin_s": f"{begin_s:.6f}",
        "end_s": f"{end_s:.6f}",
        "begin_time_s": f"{begin_s:.6f}",
        "end_time_s": f"{end_s:.6f}",
        "window_start_s": f"{begin_s:.6f}",
        "duration_s": f"{(end_s - begin_s):.6f}",
        "expected_mat_name": expected_mat,
        "mat_path": str(Path(mat_rel_dir) / expected_mat),
        "source_dataset": dataset or "biodcase2026_task2_atbfl",
        "review_status": "reviewed_background" if is_background else "reviewed",
        "species": "" if is_background else species_code,
        "species_code": "" if is_background else species_code,
        "call_type": "" if is_background else call_type,
        "call_type_std": "" if is_background else call_type,
        "call_type_raw": "" if is_background else call_type,
        "low_frequency_hz": low_freq_hz,
        "high_frequency_hz": high_freq_hz,
        "is_background": "1" if is_background else "0",
        "event_group": f"{dataset}:{filename}:{begin_s:.3f}" if not is_background else f"{dataset}:{filename}:background:{begin_s:.3f}",
        "labels_json": labels if labels else "[]",
    }
    row["label_ids"] = "" if is_background else "|".join(label_ids_from_row(row))
    return row


def _cap_per_label(rows: Sequence[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    if cap <= 0:
        return list(rows)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[clean_text(row.get("call_type"))].append(dict(row))
    selected: List[Dict[str, Any]] = []
    for label, label_rows in sorted(by_label.items()):
        label_rows = sorted(label_rows, key=lambda r: (clean_text(r.get("source_dataset")), clean_text(r.get("filename")), float(r.get("begin_s") or 0.0)))
        if len(label_rows) <= cap:
            selected.extend(label_rows)
            continue
        if cap == 1:
            selected.append(label_rows[0])
            continue
        step = (len(label_rows) - 1) / float(cap - 1)
        indices = sorted({int(round(i * step)) for i in range(cap)})
        selected.extend(label_rows[min(idx, len(label_rows) - 1)] for idx in indices[:cap])
    return selected


def build_biodcase_manifest(
    *,
    annotations_csvs: Sequence[Path],
    output_dir: Path,
    dataset_name: str = "biodcase2026_task2_atbfl",
    audio_root: Optional[Path] = None,
    audio_lists: Sequence[Path] = (),
    require_existing_audio: bool = False,
    max_per_label: int = 0,
    max_background: int = 0,
    background_window_s: float = 40.0,
    background_start_s: float = 130.0,
    mat_rel_dir: str = "mat_files",
    vocab_min_count: int = 1,
    clip_name_mode: str = "filename",
) -> Dict[str, Any]:
    positive_rows: List[Dict[str, Any]] = []
    skipped = Counter()
    missing_audio = Counter()
    per_file_positive_count = Counter()

    for annotation_csv in annotations_csvs:
        for row in _read_table(annotation_csv):
            filename = _first(row, AUDIO_COLUMNS)
            raw_label = _first(row, LABEL_COLUMNS)
            call_type = normalize_call_type(raw_label)
            species_code = CALL_TO_SPECIES.get(call_type, "")
            dataset = _first(row, DATASET_COLUMNS) or dataset_name
            begin_s, end_s = _timing_from_row(row, filename)
            if not filename:
                skipped["missing_filename"] += 1
                continue
            if not call_type or not species_code:
                skipped["unsupported_label"] += 1
                continue
            if begin_s is None or end_s is None or end_s <= begin_s:
                skipped["missing_or_invalid_time"] += 1
                continue
            if require_existing_audio and _find_existing_audio(filename, dataset, audio_root) is None:
                skipped["missing_audio"] += 1
                missing_audio[filename] += 1
                continue

            labels = [_label_record(species_code=species_code, call_type=call_type, dataset=dataset, row=row)]
            manifest_row = _manifest_row(
                filename=filename,
                clip_name=_clip_name(filename, dataset, clip_name_mode),
                dataset=dataset,
                begin_s=float(begin_s),
                end_s=float(end_s),
                species_code=species_code,
                call_type=call_type,
                labels_json=json.dumps(labels, sort_keys=True, separators=(",", ":")),
                low_freq_hz=_first(row, LOW_FREQ_COLUMNS),
                high_freq_hz=_first(row, HIGH_FREQ_COLUMNS),
                mat_rel_dir=mat_rel_dir,
            )
            positive_rows.append(manifest_row)
            per_file_positive_count[f"{dataset}/{filename}"] += 1

    positive_rows = _cap_per_label(positive_rows, int(max_per_label))

    audio_names = _audio_names_from_list(audio_lists)
    if not audio_names:
        audio_names = _audio_names_from_root(audio_root)
    background_rows: List[Dict[str, Any]] = []
    if max_background > 0 and audio_names:
        positive_keys = {key for key, count in per_file_positive_count.items() if count > 0}
        candidates = []
        for raw_name in audio_names:
            filename = Path(raw_name).name
            dataset = str(Path(raw_name).parent) if str(Path(raw_name).parent) != "." else dataset_name
            if f"{dataset}/{filename}" in positive_keys or f"{dataset_name}/{filename}" in positive_keys:
                continue
            candidates.append((dataset, filename))
        for dataset, filename in sorted(dict.fromkeys(candidates))[: int(max_background)]:
            begin_s = float(background_start_s)
            end_s = begin_s + float(background_window_s)
            background_rows.append(
                _manifest_row(
                    filename=filename,
                    clip_name=_clip_name(filename, dataset, clip_name_mode),
                    dataset=dataset,
                    begin_s=begin_s,
                    end_s=end_s,
                    species_code="",
                    call_type="",
                    labels_json="[]",
                    is_background=True,
                    mat_rel_dir=mat_rel_dir,
                )
            )

    rows = sorted(
        positive_rows + background_rows,
        key=lambda r: (
            clean_text(r.get("source_dataset")),
            clean_text(r.get("filename")),
            float(r.get("begin_s") or 0.0),
            clean_text(r.get("item_id")),
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "selected_calls.csv", rows)
    write_csv_rows(output_dir / "positive_calls.csv", positive_rows)
    write_csv_rows(output_dir / "background_windows.csv", background_rows)
    write_csv_rows(output_dir / "expected_multilabel_manifest.csv", rows)
    required = sorted({clean_text(row.get("source_audio")) for row in rows if clean_text(row.get("source_audio"))})
    (output_dir / "required_audio_filenames.txt").write_text("\n".join(required) + ("\n" if required else ""), encoding="utf-8")
    write_csv_rows(
        output_dir / "required_audio_sources.csv",
        [
            {
                "clip": clean_text(row.get("clip")),
                "source_dataset": clean_text(row.get("source_dataset")),
                "source_audio": clean_text(row.get("source_audio")),
            }
            for row in rows
            if clean_text(row.get("clip")) and clean_text(row.get("source_audio"))
        ],
    )
    vocab = build_vocabulary_from_rows(rows, min_count=max(1, int(vocab_min_count)))
    vocab.save(output_dir / "label_vocabulary.json")

    label_counts = Counter()
    species_counts = Counter()
    call_counts = Counter()
    for row in rows:
        ids = label_ids_from_row(row)
        if ids:
            label_counts.update(ids)
        else:
            label_counts["<background>"] += 1
        species_counts[clean_text(row.get("species")) or "<background>"] += 1
        call_counts[clean_text(row.get("call_type")) or "<blank>"] += 1

    summary = {
        "dataset_name": dataset_name,
        "annotation_csvs": [str(path.resolve()) for path in annotations_csvs],
        "audio_root": "" if audio_root is None else str(audio_root.resolve()),
        "row_count": len(rows),
        "positive_count": len(positive_rows),
        "background_count": len(background_rows),
        "required_audio_count": len(required),
        "species_counts": dict(species_counts.most_common()),
        "call_type_counts": dict(call_counts.most_common()),
        "label_counts": dict(label_counts.most_common()),
        "skipped_counts": dict(skipped.most_common()),
        "missing_audio_top": dict(missing_audio.most_common(25)),
        "vocabulary_size": vocab.size,
        "vocabulary_label_ids": list(vocab.label_ids),
        "config": {
            "require_existing_audio": bool(require_existing_audio),
            "max_per_label": int(max_per_label),
            "max_background": int(max_background),
            "background_window_s": float(background_window_s),
            "background_start_s": float(background_start_s),
            "mat_rel_dir": mat_rel_dir,
            "vocab_min_count": max(1, int(vocab_min_count)),
            "clip_name_mode": clip_name_mode,
        },
    }
    (output_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BioDCASE Task 2 train-style manifests")
    parser.add_argument("--annotations-csv", action="append", required=True, help="BioDCASE/ATBFL annotation CSV/TSV. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="biodcase2026_task2_atbfl")
    parser.add_argument("--audio-root", default="", help="Optional root used for audio availability checks and background candidates")
    parser.add_argument("--audio-list", action="append", default=[], help="Optional text file of audio names for background candidates")
    parser.add_argument("--require-existing-audio", action="store_true")
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--max-background", type=int, default=0)
    parser.add_argument("--background-window-s", type=float, default=40.0)
    parser.add_argument("--background-start-s", type=float, default=130.0)
    parser.add_argument("--mat-rel-dir", default="mat_files")
    parser.add_argument("--vocab-min-count", type=int, default=1)
    parser.add_argument(
        "--clip-name-mode",
        choices=["filename", "dataset_prefix"],
        default="filename",
        help="Use dataset_prefix when audio files from multiple site-year folders are staged into one directory.",
    )
    args = parser.parse_args()

    summary = build_biodcase_manifest(
        annotations_csvs=[Path(path) for path in args.annotations_csv],
        output_dir=Path(args.output_dir),
        dataset_name=str(args.dataset_name),
        audio_root=Path(args.audio_root) if str(args.audio_root).strip() else None,
        audio_lists=[Path(path) for path in args.audio_list],
        require_existing_audio=bool(args.require_existing_audio),
        max_per_label=int(args.max_per_label),
        max_background=int(args.max_background),
        background_window_s=float(args.background_window_s),
        background_start_s=float(args.background_start_s),
        mat_rel_dir=str(args.mat_rel_dir),
        vocab_min_count=int(args.vocab_min_count),
        clip_name_mode=str(args.clip_name_mode),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
