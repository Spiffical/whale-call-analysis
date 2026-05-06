#!/usr/bin/env python3
"""Build a small multi-species/call-type prep manifest from Part 2 annotations.

The output is intentionally compatible with the existing train-style MAT
generator:

    scripts/data/part2/prepare_trainstyle_windows.py --calls-csv selected_calls.csv

Positive rows are annotation-centered. Background rows are synthetic windows
sampled from pure-negative clips and are centered inside the 300s source clip so
the normal 40s context path does not require adjacent audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    NONBIOLOGICAL_SPECIES_CODES,
    PRIMARY_SPECIES_LABEL_IDS,
    annotation_call_type,
    annotation_species_code,
    build_vocabulary_from_rows,
    call_type_display_name,
    clean_text,
    label_ids_from_row,
    parse_filename_timestamp,
    review_status,
    species_display_name,
    write_csv_rows,
)


DEFAULT_SPECIES_EXCLUDE = frozenset({"Bp"})
DEFAULT_BACKGROUND_WINDOW_START_S = 130.0
PRIMARY_SPECIES_CODES = frozenset(label.partition(":")[2] for label in PRIMARY_SPECIES_LABEL_IDS)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if clean_text(value) == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_csv_arg(value: str) -> Tuple[str, ...]:
    if not value:
        return ()
    return tuple(token.strip() for token in value.split(",") if token.strip())


def _evenly_capped(rows: Sequence[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    if cap == 1:
        return [dict(rows[0])]
    step = (len(rows) - 1) / float(cap - 1)
    indices = sorted({int(round(i * step)) for i in range(cap)})
    while len(indices) < cap:
        indices.append(len(indices))
    return [dict(rows[min(idx, len(rows) - 1)]) for idx in indices[:cap]]


def _parse_clip_name(filename: str) -> Tuple[str, Optional[datetime], str, str]:
    """Return device, timestamp, suffix before extension, extension."""
    match = re.match(r"^(?P<device>[^_]+)_(?P<ts>\d{8}T\d{6})(?P<tail>.*)$", filename)
    if not match:
        return "", None, "", Path(filename).suffix
    dt = datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return match.group("device"), dt, match.group("tail") or "", ""


def _format_clip_name(device: str, dt: datetime, suffix: str, ext: str) -> str:
    return f"{device}_{dt.strftime('%Y%m%dT%H%M%S')}{suffix}{ext}"


def _adjacent_audio_names(
    filename: str,
    *,
    begin_s: float,
    end_s: float,
    context_s: float,
    edge_context_s: float,
    clip_duration_s: float,
) -> List[str]:
    device, clip_dt, suffix, ext = _parse_clip_name(filename)
    if clip_dt is None or not device:
        return []
    duration = max(0.0, float(end_s) - float(begin_s))
    padding = max(0.0, (float(context_s) - duration) / 2.0)
    needed_start = float(begin_s) - padding - float(edge_context_s)
    needed_end = float(end_s) + padding + float(edge_context_s)
    names: List[str] = []
    if needed_start < 0.0:
        names.append(_format_clip_name(device, clip_dt - timedelta(seconds=clip_duration_s), suffix, ext))
    if needed_end > float(clip_duration_s):
        names.append(_format_clip_name(device, clip_dt + timedelta(seconds=clip_duration_s), suffix, ext))
    return names


def _required_audio_names(
    filename: str,
    *,
    begin_s: float,
    end_s: float,
    context_s: float,
    edge_context_s: float,
    clip_duration_s: float,
) -> set[str]:
    names = {filename}
    names.update(
        _adjacent_audio_names(
            filename,
            begin_s=begin_s,
            end_s=end_s,
            context_s=context_s,
            edge_context_s=edge_context_s,
            clip_duration_s=clip_duration_s,
        )
    )
    return {name for name in names if name}


def _available_audio_names(audio_dir: Optional[Path]) -> Optional[set[str]]:
    if audio_dir is None:
        return None
    names: set[str] = set()
    for pattern in ("*.flac", "*.wav"):
        names.update(path.name for path in audio_dir.rglob(pattern))
    return names


def _label_record(row: Dict[str, Any]) -> Dict[str, Any]:
    species = annotation_species_code(row)
    call_type = annotation_call_type(row)
    return {
        "species_code": species or None,
        "species": species_display_name(species) if species else None,
        "call_type": call_type or None,
        "call_type_name": call_type_display_name(call_type) if call_type else None,
        "source": clean_text(row.get("source_dataset")) or "part2_full_bundle",
        "review_status": review_status(row),
        "confidence": None,
        "trainable": bool(species and species not in NONBIOLOGICAL_SPECIES_CODES) or bool(call_type),
    }


def _expected_mat_name(filename: str, begin_s: float, end_s: float) -> str:
    return f"{filename}_{float(begin_s):.1f}s_{float(end_s):.1f}s_trainstyle.mat"


def _positive_manifest_row(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    filename = clean_text(row.get("filename"))
    begin_s = _float_or_none(row.get("begin_time_s") or row.get("begin_time"))
    end_s = _float_or_none(row.get("end_time_s") or row.get("end_time"))
    if begin_s is None or end_s is None:
        raise ValueError("positive annotation row missing begin/end")
    labels = [_label_record(row)]
    species_code = annotation_species_code(row)
    negative_bucket = "" if species_code in PRIMARY_SPECIES_CODES else "nonprimary_biological_signal"
    out = {
        "item_id": Path(_expected_mat_name(filename, begin_s, end_s)).stem,
        "clip": filename,
        "source_audio": filename,
        "begin_s": f"{begin_s:.6f}",
        "end_s": f"{end_s:.6f}",
        "window_start_s": f"{begin_s:.6f}",
        "duration_s": f"{(end_s - begin_s):.6f}",
        "expected_mat_name": _expected_mat_name(filename, begin_s, end_s),
        "source_dataset": dataset_name,
        "source_kind": "ONC",
        "review_status": review_status(row),
        "species": species_code,
        "call_type": annotation_call_type(row),
        "is_background": "0",
        "event_group": f"{filename}:{begin_s:.3f}",
        "negative_bucket": negative_bucket,
        "context_tags": "" if not negative_bucket else negative_bucket,
        "labels_json": json.dumps(labels, sort_keys=True, separators=(",", ":")),
    }
    out["label_ids"] = "|".join(label_ids_from_row(out))
    return out


def _background_manifest_row(
    *,
    filename: str,
    begin_s: float,
    end_s: float,
    dataset_name: str,
    context_tags: str,
) -> Dict[str, Any]:
    out = {
        "item_id": Path(_expected_mat_name(filename, begin_s, end_s)).stem,
        "clip": filename,
        "source_audio": filename,
        "begin_s": f"{begin_s:.6f}",
        "end_s": f"{end_s:.6f}",
        "window_start_s": f"{begin_s:.6f}",
        "duration_s": f"{(end_s - begin_s):.6f}",
        "expected_mat_name": _expected_mat_name(filename, begin_s, end_s),
        "source_dataset": dataset_name,
        "source_kind": "ONC",
        "review_status": "pure_negative_candidate",
        "species": "",
        "call_type": "",
        "is_background": "1",
        "event_group": f"{filename}:background:{begin_s:.3f}",
        "negative_bucket": "ambiguous_hard_negative",
        "context_tags": context_tags,
        "labels_json": "[]",
        "label_ids": "",
    }
    return out


def build_prep_manifest(
    *,
    annotations_csv: Path,
    clip_manifest_csv: Path,
    output_dir: Path,
    dataset_name: str,
    species: Sequence[str],
    include_fin: bool,
    include_nonbiological: bool,
    max_per_species: int,
    max_fin: int,
    max_background: int,
    context_s: float,
    edge_context_s: float,
    clip_duration_s: float,
    background_window_s: float,
    background_windows_per_clip: int,
    available_audio_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    rows = _read_csv(annotations_csv)
    species_filter = set(species)
    if include_fin and species_filter:
        species_filter.add("Bp")

    available_audio = _available_audio_names(available_audio_dir)
    missing_audio_counter: Counter[str] = Counter()
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for row in rows:
        filename = clean_text(row.get("filename"))
        begin_s = _float_or_none(row.get("begin_time_s") or row.get("begin_time"))
        end_s = _float_or_none(row.get("end_time_s") or row.get("end_time"))
        code = annotation_species_code(row)
        if not filename or begin_s is None or end_s is None or end_s <= begin_s:
            skipped["missing_time_or_filename"] += 1
            continue
        if not code:
            skipped["blank_species"] += 1
            continue
        if code in NONBIOLOGICAL_SPECIES_CODES and not include_nonbiological:
            skipped["nonbiological"] += 1
            continue
        if code in DEFAULT_SPECIES_EXCLUDE and not include_fin:
            skipped["fin_excluded"] += 1
            continue
        if species_filter and code not in species_filter:
            skipped["species_filter"] += 1
            continue
        if available_audio is not None:
            required_names = _required_audio_names(
                filename,
                begin_s=begin_s,
                end_s=end_s,
                context_s=context_s,
                edge_context_s=edge_context_s,
                clip_duration_s=clip_duration_s,
            )
            missing_names = sorted(required_names.difference(available_audio))
            if missing_names:
                skipped["missing_required_audio"] += 1
                missing_audio_counter.update(missing_names)
                continue
        grouped[code].append(dict(row))

    positive_rows: List[Dict[str, Any]] = []
    for code, code_rows in sorted(grouped.items()):
        code_rows = sorted(
            code_rows,
            key=lambda r: (
                clean_text(r.get("filename")),
                float(_float_or_none(r.get("begin_time_s") or r.get("begin_time")) or 0.0),
                float(_float_or_none(r.get("end_time_s") or r.get("end_time")) or 0.0),
            ),
        )
        cap = max_fin if code == "Bp" and max_fin > 0 else max_per_species
        for row in _evenly_capped(code_rows, cap):
            positive_rows.append(_positive_manifest_row(row, dataset_name))

    clip_rows = _read_csv(clip_manifest_csv)
    pure_negative_clips = [
        row
        for row in clip_rows
        if clean_text(row.get("is_pure_negative_candidate")) == "1"
        and clean_text(row.get("filename"))
        and clean_text(row.get("is_fin_positive")) != "1"
        and clean_text(row.get("is_annotated_non_fin")) != "1"
    ]
    if available_audio is not None:
        audio_filtered_negative_clips = []
        for row in pure_negative_clips:
            filename = clean_text(row.get("filename"))
            missing_names = sorted({filename}.difference(available_audio))
            if missing_names:
                skipped["background_missing_audio"] += 1
                missing_audio_counter.update(missing_names)
                continue
            audio_filtered_negative_clips.append(row)
        pure_negative_clips = audio_filtered_negative_clips
    pure_negative_clips = sorted(pure_negative_clips, key=lambda row: clean_text(row.get("filename")))
    pure_negative_clips = _evenly_capped(pure_negative_clips, max_background)

    background_rows: List[Dict[str, Any]] = []
    for clip_row in pure_negative_clips:
        filename = clean_text(clip_row.get("filename"))
        if not filename:
            continue
        windows = max(1, int(background_windows_per_clip))
        if windows == 1:
            starts = [DEFAULT_BACKGROUND_WINDOW_START_S]
        else:
            usable = max(0.0, clip_duration_s - background_window_s)
            step = usable / float(windows - 1)
            starts = [round(i * step, 3) for i in range(windows)]
        for start in starts:
            start = max(0.0, min(float(start), float(clip_duration_s) - float(background_window_s)))
            end = start + float(background_window_s)
            background_rows.append(
                _background_manifest_row(
                    filename=filename,
                    begin_s=start,
                    end_s=end,
                    dataset_name=dataset_name,
                    context_tags=clean_text(clip_row.get("context_tags")) or "pure_negative",
                )
            )

    manifest_rows = sorted(
        positive_rows + background_rows,
        key=lambda r: (clean_text(r.get("clip")), float(_float_or_none(r.get("begin_s")) or 0.0), clean_text(r.get("item_id"))),
    )

    required_audio: set[str] = set()
    for row in manifest_rows:
        filename = clean_text(row.get("clip"))
        begin_s = float(_float_or_none(row.get("begin_s")) or 0.0)
        end_s = float(_float_or_none(row.get("end_s")) or begin_s)
        required_audio.add(filename)
        required_audio.update(
            _adjacent_audio_names(
                filename,
                begin_s=begin_s,
                end_s=end_s,
                context_s=context_s,
                edge_context_s=edge_context_s,
                clip_duration_s=clip_duration_s,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "selected_calls.csv", manifest_rows)
    write_csv_rows(output_dir / "positive_calls.csv", positive_rows)
    write_csv_rows(output_dir / "background_windows.csv", background_rows)
    write_csv_rows(output_dir / "expected_multilabel_manifest.csv", manifest_rows)

    with open(output_dir / "required_audio_filenames.txt", "w", encoding="utf-8") as handle:
        for filename in sorted(required_audio):
            handle.write(f"{filename}\n")
    with open(output_dir / "selected_source_clips.txt", "w", encoding="utf-8") as handle:
        for filename in sorted({clean_text(row.get("clip")) for row in manifest_rows if clean_text(row.get("clip"))}):
            handle.write(f"{filename}\n")

    vocab = build_vocabulary_from_rows(manifest_rows, min_count=1)
    vocab.save(output_dir / "label_vocabulary.json")

    label_counts = Counter()
    species_counts = Counter()
    call_counts = Counter()
    for row in manifest_rows:
        for raw_id in label_ids_from_row(row):
            label_counts[raw_id] += 1
        species_counts[clean_text(row.get("species")) or "<background>"] += 1
        call_counts[clean_text(row.get("call_type")) or "<blank>"] += 1

    summary = {
        "annotations_csv": str(annotations_csv.resolve()),
        "clip_manifest_csv": str(clip_manifest_csv.resolve()),
        "dataset_name": dataset_name,
        "positive_count": len(positive_rows),
        "background_count": len(background_rows),
        "row_count": len(manifest_rows),
        "required_audio_count": len(required_audio),
        "selected_source_clip_count": len({clean_text(row.get("clip")) for row in manifest_rows}),
        "species_counts": dict(species_counts.most_common()),
        "call_type_counts": dict(call_counts.most_common()),
        "label_counts": dict(label_counts.most_common()),
        "skipped_annotation_counts": dict(skipped.most_common()),
        "available_audio_dir": "" if available_audio_dir is None else str(available_audio_dir.resolve()),
        "available_audio_count": None if available_audio is None else len(available_audio),
        "missing_required_audio_top": dict(missing_audio_counter.most_common(25)),
        "config": {
            "species": list(species),
            "include_fin": bool(include_fin),
            "include_nonbiological": bool(include_nonbiological),
            "max_per_species": int(max_per_species),
            "max_fin": int(max_fin),
            "max_background": int(max_background),
            "context_s": float(context_s),
            "edge_context_s": float(edge_context_s),
            "clip_duration_s": float(clip_duration_s),
            "background_window_s": float(background_window_s),
            "background_windows_per_clip": int(background_windows_per_clip),
        },
    }
    with open(output_dir / "prep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build multi-species train-style prep manifests")
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--clip-manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="final2025_multispecies_prep")
    parser.add_argument("--species", default="", help="Comma-separated species codes to include; default is all non-fin biological labels")
    parser.add_argument("--include-fin", action="store_true", help="Include Bp rows, optionally capped by --max-fin")
    parser.add_argument("--include-nonbiological", action="store_true", help="Promote INSTRUMENT/EQ/SONAR/UNKNOWN to trainable species labels")
    parser.add_argument("--max-per-species", type=int, default=0, help="Cap positives per non-fin species; 0 keeps all")
    parser.add_argument("--max-fin", type=int, default=0, help="Cap Bp positives when --include-fin is set; 0 keeps all")
    parser.add_argument("--max-background", type=int, default=0, help="Cap pure-negative clips; 0 keeps all")
    parser.add_argument("--background-windows-per-clip", type=int, default=1)
    parser.add_argument("--background-window-s", type=float, default=40.0)
    parser.add_argument("--context-s", type=float, default=40.0)
    parser.add_argument("--edge-context-s", type=float, default=10.5)
    parser.add_argument("--clip-duration-s", type=float, default=300.0)
    parser.add_argument("--available-audio-dir", default="", help="Optional directory used to skip rows whose required source audio is unavailable")
    args = parser.parse_args()

    summary = build_prep_manifest(
        annotations_csv=Path(args.annotations_csv),
        clip_manifest_csv=Path(args.clip_manifest_csv),
        output_dir=Path(args.output_dir),
        dataset_name=str(args.dataset_name),
        species=_split_csv_arg(args.species),
        include_fin=bool(args.include_fin),
        include_nonbiological=bool(args.include_nonbiological),
        max_per_species=int(args.max_per_species),
        max_fin=int(args.max_fin),
        max_background=int(args.max_background),
        context_s=float(args.context_s),
        edge_context_s=float(args.edge_context_s),
        clip_duration_s=float(args.clip_duration_s),
        background_window_s=float(args.background_window_s),
        background_windows_per_clip=int(args.background_windows_per_clip),
        available_audio_dir=Path(args.available_audio_dir) if str(args.available_audio_dir).strip() else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
