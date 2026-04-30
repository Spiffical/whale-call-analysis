#!/usr/bin/env python3
"""Audit multi-species and multi-call-type labels.

The script is read-only with respect to source data. It writes audit artifacts
under the caller-provided output directory and can also build a small
clip/window-level candidate manifest for smoke tests when bundle MATs are
available.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    NONBIOLOGICAL_SPECIES_CODES,
    TRAINABLE_CALL_TYPES,
    annotation_call_type,
    annotation_device,
    annotation_filename,
    annotation_month,
    annotation_species_code,
    annotation_year,
    build_vocabulary_from_rows,
    call_type_display_name,
    clean_text,
    label_ids_from_row,
    normalize_call_type,
    normalize_species_code,
    parse_filename_timestamp,
    read_csv_rows,
    review_status,
    split_pipe,
    species_display_name,
    write_csv_rows,
)
from src.training.mat_utils import parse_mat_filename  # noqa: E402


def _counter_rows(counter: Counter, *fieldnames: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, count in counter.most_common():
        if not isinstance(key, tuple):
            key = (key,)
        row = {field: key[idx] if idx < len(key) else "" for idx, field in enumerate(fieldnames)}
        row["count"] = int(count)
        rows.append(row)
    return rows


def _read_annotation_sources(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            out = dict(row)
            out["_source_manifest"] = str(path)
            rows.append(out)
    return rows


def _bundle_annotation_csv(bundle_dir: Path) -> Optional[Path]:
    manifest_dir = bundle_dir / "manifests"
    for name in ("unified_annotations.csv", "annotations_all.csv", "fin_annotations.csv"):
        path = manifest_dir / name
        if path.exists():
            return path
    return None


def _bundle_clip_manifest_csv(bundle_dir: Path) -> Optional[Path]:
    for name in ("clip_manifest.csv", "clip_inventory.csv"):
        path = bundle_dir / "manifests" / name
        if path.exists():
            return path
    return None


def _bundle_mat_dirs(bundle_dir: Path) -> List[Path]:
    out = []
    for name in ("mat_files", "neg_mat_files"):
        path = bundle_dir / name
        if path.exists():
            out.append(path)
    return out


def discover_bundle_sources(bundle_dirs: Sequence[Path]) -> Tuple[List[Path], List[Path]]:
    annotation_paths: List[Path] = []
    clip_paths: List[Path] = []
    for bundle_dir in bundle_dirs:
        ann = _bundle_annotation_csv(bundle_dir)
        clip = _bundle_clip_manifest_csv(bundle_dir)
        if ann is not None:
            annotation_paths.append(ann)
        if clip is not None:
            clip_paths.append(clip)
    return annotation_paths, clip_paths


def summarize_annotations(rows: Sequence[Dict[str, Any]], *, rare_threshold: int) -> Dict[str, Any]:
    species_counts: Counter[str] = Counter()
    call_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    month_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    duplicate_key_counts: Counter[Tuple[str, str, str, str, str]] = Counter()
    duration_values: List[float] = []

    for row in rows:
        species = annotation_species_code(row) or "<blank>"
        call_type = annotation_call_type(row) or "<blank>"
        filename = annotation_filename(row)
        begin_s = clean_text(row.get("begin_time_s") or row.get("begin_time"))
        end_s = clean_text(row.get("end_time_s") or row.get("end_time"))
        duplicate_key_counts[(filename, species, call_type, begin_s, end_s)] += 1
        species_counts[species] += 1
        call_counts[call_type] += 1
        pair_counts[(species, call_type)] += 1
        month_counts[annotation_month(row) or "<blank>"] += 1
        year_counts[annotation_year(row) or "<blank>"] += 1
        device_counts[annotation_device(row) or "<blank>"] += 1
        source_counts[clean_text(row.get("source_dataset")) or clean_text(row.get("_source_manifest")) or "<blank>"] += 1
        review_counts[review_status(row)] += 1
        try:
            dur = float(row.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            begin = row.get("begin_time_s") or row.get("begin_time")
            end = row.get("end_time_s") or row.get("end_time")
            try:
                dur = float(end) - float(begin)
            except (TypeError, ValueError):
                dur = 0.0
        if dur > 0:
            duration_values.append(float(dur))

    rare_species = {key: int(value) for key, value in species_counts.items() if 0 < value < rare_threshold}
    rare_calls = {key: int(value) for key, value in call_counts.items() if key != "<blank>" and 0 < value < rare_threshold}
    duplicate_keys = {key: count for key, count in duplicate_key_counts.items() if count > 1 and key[0]}

    duration_summary = {
        "count": len(duration_values),
        "min": min(duration_values) if duration_values else None,
        "median": sorted(duration_values)[len(duration_values) // 2] if duration_values else None,
        "max": max(duration_values) if duration_values else None,
    }
    return {
        "annotation_count": int(len(rows)),
        "species_counts": dict(species_counts.most_common()),
        "call_type_counts": dict(call_counts.most_common()),
        "species_call_type_counts": {f"{species}::{call}": int(count) for (species, call), count in pair_counts.most_common()},
        "month_counts": dict(month_counts.most_common()),
        "year_counts": dict(year_counts.most_common()),
        "device_counts": dict(device_counts.most_common()),
        "source_dataset_counts": dict(source_counts.most_common()),
        "review_status_counts": dict(review_counts.most_common()),
        "rare_species": rare_species,
        "rare_call_types": rare_calls,
        "duplicate_annotation_key_count": int(len(duplicate_keys)),
        "duplicate_annotation_row_count": int(sum(duplicate_keys.values())),
        "duration_seconds": duration_summary,
        "_counters": {
            "species": species_counts,
            "call_type": call_counts,
            "pair": pair_counts,
            "month": month_counts,
            "year": year_counts,
            "device": device_counts,
            "source": source_counts,
            "review": review_counts,
            "duplicates": Counter(duplicate_keys),
        },
    }


def summarize_clip_manifests(paths: Sequence[Path]) -> Dict[str, Any]:
    clip_count = 0
    multi_species = 0
    multi_call = 0
    state_counts: Counter[Tuple[str, str, str]] = Counter()
    species_combo_counts: Counter[str] = Counter()
    call_combo_counts: Counter[str] = Counter()

    for path in paths:
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            clip_count += 1
            species = split_pipe(row.get("species_codes") or row.get("species_code") or row.get("species"))
            calls = split_pipe(row.get("fin_call_type_stds") or row.get("fin_call_type_buckets") or row.get("call_type_stds"))
            if len(species) > 1:
                multi_species += 1
            if len(calls) > 1:
                multi_call += 1
            state_counts[
                (
                    clean_text(row.get("is_fin_positive")) or "0",
                    clean_text(row.get("is_annotated_non_fin")) or "0",
                    clean_text(row.get("is_pure_negative_candidate")) or "0",
                )
            ] += 1
            species_combo_counts["|".join(species) if species else "<blank>"] += 1
            call_combo_counts["|".join(calls) if calls else "<blank>"] += 1

    return {
        "clip_count": int(clip_count),
        "multi_species_clip_count": int(multi_species),
        "multi_call_type_clip_count": int(multi_call),
        "clip_state_counts": {"/".join(key): int(value) for key, value in state_counts.most_common()},
        "species_combo_counts": dict(species_combo_counts.most_common()),
        "call_type_combo_counts": dict(call_combo_counts.most_common()),
    }


def _label_records_from_clip_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_species in split_pipe(row.get("species_codes") or row.get("species_code") or row.get("species")):
        code = normalize_species_code(raw_species)
        if not code:
            continue
        records.append(
            {
                "species_code": code,
                "species": species_display_name(code),
                "call_type": None,
                "source": "manifest",
                "review_status": review_status(row),
                "trainable": code not in NONBIOLOGICAL_SPECIES_CODES,
            }
        )
    raw_calls = split_pipe(row.get("fin_call_type_stds") or row.get("fin_call_type_buckets") or row.get("call_type_stds"))
    for raw_call in raw_calls:
        call_type = normalize_call_type(raw_call)
        if not call_type:
            continue
        records.append(
            {
                "species_code": None,
                "species": None,
                "call_type": call_type,
                "call_type_name": call_type_display_name(call_type),
                "source": "manifest",
                "review_status": review_status(row),
                "trainable": call_type in TRAINABLE_CALL_TYPES,
            }
        )
    return records


def _mat_source_audio(mat_path: Path) -> Tuple[str, Optional[float], Optional[float]]:
    source, start, duration = parse_mat_filename(mat_path.name)
    return source, start, duration


def _manifest_time_fields(source_audio: str, start_s: Optional[float], duration_s: Optional[float]) -> Dict[str, str]:
    clip_start = parse_filename_timestamp(source_audio)
    if clip_start is None:
        return {"start_time": "", "end_time": ""}
    rel_start = float(start_s or 0.0)
    dur = float(duration_s) if duration_s is not None else 0.0
    start = clip_start + timedelta(seconds=rel_start)
    end = start + timedelta(seconds=dur) if dur > 0 else None
    return {
        "start_time": start.isoformat(),
        "end_time": end.isoformat() if end is not None else "",
    }


def build_candidate_manifest(bundle_dirs: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    missing_media: List[Dict[str, Any]] = []

    for bundle_dir in bundle_dirs:
        clip_manifest = _bundle_clip_manifest_csv(bundle_dir)
        if clip_manifest is None:
            continue
        clip_rows = {clean_text(row.get("filename")): row for row in read_csv_rows(clip_manifest)}
        mat_dirs = _bundle_mat_dirs(bundle_dir)
        seen_sources: set[str] = set()
        for mat_dir in mat_dirs:
            for mat_path in sorted(mat_dir.glob("*.mat")):
                source_audio, start_s, duration_s = _mat_source_audio(mat_path)
                seen_sources.add(source_audio)
                clip_row = dict(clip_rows.get(source_audio, {}))
                labels = _label_records_from_clip_row(clip_row)
                time_fields = _manifest_time_fields(source_audio, start_s, duration_s)
                row = {
                    "item_id": mat_path.stem,
                    "source_audio": source_audio,
                    "mat_path": str(mat_path.resolve()),
                    "device": source_audio.split("_", 1)[0] if "_" in source_audio else "",
                    "start_time": time_fields["start_time"],
                    "end_time": time_fields["end_time"],
                    "window_start_s": "" if start_s is None else f"{float(start_s):.6f}",
                    "duration_s": "" if duration_s is None else f"{float(duration_s):.6f}",
                    "source_dataset": clean_text(clip_row.get("source_dataset")) or bundle_dir.name,
                    "review_status": review_status(clip_row),
                    "event_group": source_audio,
                    "labels_json": json.dumps(labels, sort_keys=True, separators=(",", ":")),
                }
                row["label_ids"] = "|".join(label_ids_from_row(row))
                row["is_background"] = "1" if not row["label_ids"] else "0"
                rows.append(row)

        for filename in sorted(clip_rows):
            if filename and filename not in seen_sources:
                missing_media.append(
                    {
                        "bundle_dir": str(bundle_dir),
                        "source_audio": filename,
                        "missing": "mat",
                    }
                )

    return rows, missing_media


def write_audit_outputs(
    output_dir: Path,
    *,
    annotation_summary: Dict[str, Any],
    clip_summary: Dict[str, Any],
    candidate_rows: Sequence[Dict[str, Any]],
    missing_media: Sequence[Dict[str, Any]],
    vocab_min_count: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = annotation_summary.pop("_counters")
    write_csv_rows(output_dir / "species_counts.csv", _counter_rows(counters["species"], "species_code"))
    write_csv_rows(output_dir / "call_type_counts.csv", _counter_rows(counters["call_type"], "call_type"))
    write_csv_rows(output_dir / "species_call_type_counts.csv", _counter_rows(counters["pair"], "species_code", "call_type"))
    write_csv_rows(output_dir / "month_counts.csv", _counter_rows(counters["month"], "month"))
    write_csv_rows(output_dir / "year_counts.csv", _counter_rows(counters["year"], "year"))
    write_csv_rows(output_dir / "device_counts.csv", _counter_rows(counters["device"], "device"))
    write_csv_rows(output_dir / "source_dataset_counts.csv", _counter_rows(counters["source"], "source_dataset"))
    write_csv_rows(output_dir / "review_status_counts.csv", _counter_rows(counters["review"], "review_status"))
    duplicate_rows = []
    for (filename, species, call_type, begin_s, end_s), count in counters["duplicates"].most_common():
        duplicate_rows.append(
            {
                "filename": filename,
                "species_code": species,
                "call_type": call_type,
                "begin_time_s": begin_s,
                "end_time_s": end_s,
                "count": int(count),
            }
        )
    write_csv_rows(output_dir / "duplicate_annotation_keys.csv", duplicate_rows)

    rare_rows: List[Dict[str, Any]] = []
    for species, count in annotation_summary["rare_species"].items():
        rare_rows.append({"label_group": "species", "label": species, "count": count})
    for call_type, count in annotation_summary["rare_call_types"].items():
        rare_rows.append({"label_group": "call_type", "label": call_type, "count": count})
    write_csv_rows(output_dir / "rare_labels.csv", rare_rows)
    write_csv_rows(output_dir / "missing_media.csv", list(missing_media))

    if candidate_rows:
        write_csv_rows(output_dir / "candidate_multilabel_manifest.csv", candidate_rows)
        vocab = build_vocabulary_from_rows(candidate_rows, min_count=int(vocab_min_count))
        vocab.save(output_dir / "label_vocabulary.json")

    summary = {
        "annotation_summary": annotation_summary,
        "clip_summary": clip_summary,
        "candidate_manifest": {
            "row_count": int(len(candidate_rows)),
            "background_row_count": int(sum(1 for row in candidate_rows if clean_text(row.get("is_background")) == "1")),
            "label_min_count": int(vocab_min_count),
        },
        "missing_media_count": int(len(missing_media)),
    }
    with open(output_dir / "audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_markdown_summary(output_dir / "DATA_AUDIT.generated.md", summary)
    return summary


def _top_lines(mapping: Dict[str, Any], limit: int = 12) -> List[str]:
    return [f"- `{key}`: {value}" for key, value in list(mapping.items())[:limit]]


def write_markdown_summary(path: Path, summary: Dict[str, Any]) -> None:
    ann = summary["annotation_summary"]
    clip = summary["clip_summary"]
    lines = [
        "# Generated Data Audit Summary",
        "",
        "This file is generated by `scripts/data/multilabel/audit_labels.py`.",
        "",
        "## Overall",
        "",
        f"- Annotation rows: {ann['annotation_count']:,}",
        f"- Clip manifest rows: {clip['clip_count']:,}",
        f"- Multi-species clips: {clip['multi_species_clip_count']:,}",
        f"- Multi-call-type clips: {clip['multi_call_type_clip_count']:,}",
        f"- Candidate smoke manifest rows: {summary['candidate_manifest']['row_count']:,}",
        f"- Candidate background rows: {summary['candidate_manifest']['background_row_count']:,}",
        f"- Missing media records: {summary['missing_media_count']:,}",
        f"- Duplicate annotation keys: {ann['duplicate_annotation_key_count']:,}",
        f"- Rows involved in duplicate keys: {ann['duplicate_annotation_row_count']:,}",
        "",
        "## Top Species",
        "",
        *_top_lines(ann["species_counts"]),
        "",
        "## Top Call Types",
        "",
        *_top_lines(ann["call_type_counts"]),
        "",
        "## Top Species/Call-Type Pairs",
        "",
        *_top_lines(ann["species_call_type_counts"]),
        "",
        "## Devices",
        "",
        *_top_lines(ann["device_counts"]),
        "",
        "## Months",
        "",
        *_top_lines(ann["month_counts"], limit=24),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit multi-label whale/acoustic annotations")
    parser.add_argument("--annotation-csv", action="append", default=[], help="Annotation CSV path; repeatable")
    parser.add_argument("--clip-manifest-csv", action="append", default=[], help="Clip manifest CSV path; repeatable")
    parser.add_argument("--bundle-dir", action="append", default=[], help="Bundle directory containing manifests/ and MATs; repeatable")
    parser.add_argument("--output-dir", required=True, help="Directory for audit outputs")
    parser.add_argument("--rare-threshold", type=int, default=10)
    parser.add_argument("--vocab-min-count", type=int, default=1)
    args = parser.parse_args()

    bundle_dirs = [Path(path).resolve() for path in args.bundle_dir]
    bundle_annotation_paths, bundle_clip_paths = discover_bundle_sources(bundle_dirs)
    annotation_paths = [Path(path).resolve() for path in args.annotation_csv] + bundle_annotation_paths
    clip_paths = [Path(path).resolve() for path in args.clip_manifest_csv] + bundle_clip_paths

    annotation_rows = _read_annotation_sources(annotation_paths)
    annotation_summary = summarize_annotations(annotation_rows, rare_threshold=max(1, int(args.rare_threshold)))
    clip_summary = summarize_clip_manifests(clip_paths)
    candidate_rows, missing_media = build_candidate_manifest(bundle_dirs)
    summary = write_audit_outputs(
        Path(args.output_dir).resolve(),
        annotation_summary=annotation_summary,
        clip_summary=clip_summary,
        candidate_rows=candidate_rows,
        missing_media=missing_media,
        vocab_min_count=max(1, int(args.vocab_min_count)),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
