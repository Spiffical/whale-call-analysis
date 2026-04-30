#!/usr/bin/env python3
"""Build a call-level multi-label manifest from train-style MAT files."""

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
    annotation_filename,
    annotation_species_code,
    build_vocabulary_from_rows,
    call_type_display_name,
    clean_text,
    label_ids_from_row,
    parse_filename_timestamp,
    read_csv_rows,
    review_status,
    species_display_name,
    write_csv_rows,
)
from src.training.mat_utils import parse_mat_filename  # noqa: E402


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annotation_times(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    begin = _float_or_none(row.get("begin_time_s") or row.get("begin_time"))
    end = _float_or_none(row.get("end_time_s") or row.get("end_time"))
    return begin, end


def _label_record(row: Dict[str, Any]) -> Dict[str, Any]:
    species = annotation_species_code(row)
    call_type = annotation_call_type(row)
    return {
        "species_code": species or None,
        "species": species_display_name(species) if species else None,
        "call_type": call_type or None,
        "call_type_name": call_type_display_name(call_type) if call_type else None,
        "source": clean_text(row.get("source_dataset")) or clean_text(row.get("source_workbook")) or "annotation",
        "review_status": review_status(row),
        "confidence": _float_or_none(row.get("confidence")),
        "trainable": bool((species and species not in NONBIOLOGICAL_SPECIES_CODES) or (call_type and call_type in TRAINABLE_CALL_TYPES)),
    }


def _device_from_source(source_audio: str) -> str:
    return source_audio.split("_", 1)[0] if "_" in source_audio else ""


def _time_fields(source_audio: str, start_s: Optional[float], duration_s: Optional[float]) -> Dict[str, str]:
    clip_start = parse_filename_timestamp(source_audio)
    if clip_start is None:
        return {"start_time": "", "end_time": ""}
    rel_start = float(start_s or 0.0)
    dur = float(duration_s or 0.0)
    start = clip_start + timedelta(seconds=rel_start)
    end = start + timedelta(seconds=dur)
    return {"start_time": start.isoformat(), "end_time": end.isoformat()}


def _build_annotation_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        filename = annotation_filename(row)
        if filename:
            by_file[filename].append(dict(row))
    return by_file


def _matching_annotations(
    rows: Sequence[Dict[str, Any]],
    *,
    start_s: Optional[float],
    duration_s: Optional[float],
    tolerance_s: float,
) -> List[Dict[str, Any]]:
    if start_s is None or duration_s is None:
        return []
    end_s = float(start_s) + float(duration_s)
    matches: List[Dict[str, Any]] = []
    for row in rows:
        begin, end = _annotation_times(row)
        if begin is None or end is None:
            continue
        if abs(begin - float(start_s)) <= tolerance_s and abs(end - end_s) <= tolerance_s:
            matches.append(dict(row))
    return matches


def build_call_manifest(
    *,
    annotations_csv: Path,
    mat_dir: Path,
    dataset_name: str,
    tolerance_s: float = 0.25,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    annotation_rows = read_csv_rows(annotations_csv)
    by_file = _build_annotation_index(annotation_rows)
    out_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []
    match_count_counter: Counter[int] = Counter()
    label_counter: Counter[str] = Counter()

    mat_paths = sorted(mat_dir.glob("*.mat"))
    if limit is not None and int(limit) > 0:
        mat_paths = mat_paths[: int(limit)]

    for mat_path in mat_paths:
        source_audio, start_s, duration_s = parse_mat_filename(mat_path.name)
        candidates = by_file.get(source_audio, [])
        matches = _matching_annotations(candidates, start_s=start_s, duration_s=duration_s, tolerance_s=float(tolerance_s))
        labels = [_label_record(row) for row in matches]
        labels = [label for label in labels if label.get("species_code") or label.get("call_type")]
        time_fields = _time_fields(source_audio, start_s, duration_s)
        row = {
            "item_id": mat_path.stem,
            "source_audio": source_audio,
            "mat_path": str(mat_path.resolve()),
            "device": _device_from_source(source_audio),
            "start_time": time_fields["start_time"],
            "end_time": time_fields["end_time"],
            "window_start_s": "" if start_s is None else f"{float(start_s):.6f}",
            "duration_s": "" if duration_s is None else f"{float(duration_s):.6f}",
            "source_dataset": dataset_name,
            "review_status": review_status(matches[0]) if matches else "unreviewed",
            "event_group": f"{source_audio}:{'' if start_s is None else f'{float(start_s):.3f}'}",
            "matched_annotation_count": len(matches),
            "labels_json": json.dumps(labels, sort_keys=True, separators=(",", ":")),
        }
        row["label_ids"] = "|".join(label_ids_from_row(row))
        row["is_background"] = "1" if not row["label_ids"] else "0"
        out_rows.append(row)
        for label_id in label_ids_from_row(row):
            label_counter[label_id] += 1
        match_count_counter[len(matches)] += 1
        if not matches:
            unmatched_rows.append(
                {
                    "mat_path": str(mat_path.resolve()),
                    "source_audio": source_audio,
                    "window_start_s": "" if start_s is None else f"{float(start_s):.6f}",
                    "duration_s": "" if duration_s is None else f"{float(duration_s):.6f}",
                    "candidate_annotation_count_for_source": len(candidates),
                }
            )

    summary = {
        "annotations_csv": str(annotations_csv.resolve()),
        "mat_dir": str(mat_dir.resolve()),
        "dataset_name": dataset_name,
        "tolerance_s": float(tolerance_s),
        "row_count": len(out_rows),
        "unmatched_count": len(unmatched_rows),
        "background_count": sum(1 for row in out_rows if clean_text(row.get("is_background")) == "1"),
        "match_count_distribution": {str(key): int(value) for key, value in sorted(match_count_counter.items())},
        "label_counts": dict(label_counter.most_common()),
    }
    return out_rows, unmatched_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a call-level multi-label manifest from train-style MATs")
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--mat-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="call_mat_dataset")
    parser.add_argument("--match-tolerance-s", type=float, default=0.25)
    parser.add_argument("--vocab-min-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, unmatched, summary = build_call_manifest(
        annotations_csv=Path(args.annotations_csv),
        mat_dir=Path(args.mat_dir),
        dataset_name=str(args.dataset_name),
        tolerance_s=float(args.match_tolerance_s),
        limit=args.limit,
    )
    write_csv_rows(output_dir / "call_multilabel_manifest.csv", rows)
    write_csv_rows(output_dir / "unmatched_mats.csv", unmatched)
    vocab = build_vocabulary_from_rows(rows, min_count=max(1, int(args.vocab_min_count)))
    vocab.save(output_dir / "label_vocabulary.json")
    summary["vocab_min_count"] = max(1, int(args.vocab_min_count))
    summary["vocabulary_size"] = vocab.size
    summary["vocabulary_label_ids"] = list(vocab.label_ids)
    with open(output_dir / "call_manifest_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
