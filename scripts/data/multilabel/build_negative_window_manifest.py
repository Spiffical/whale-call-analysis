#!/usr/bin/env python3
"""Build auditable no-primary negative-window manifests.

This utility is deliberately conservative: it creates named negative buckets
and keeps source groups intact, but it does not bless any unreviewed row as
clean deployment background.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    NONBIOLOGICAL_SPECIES_CODES,
    PRIMARY_SPECIES_LABEL_IDS,
    TRUTHY_VALUES,
    clean_text,
    group_key_for_split,
    label_balanced_grouped_split,
    label_ids_from_row,
    normalize_species_code,
    split_pipe,
    write_csv_rows,
)
from src.dataset.negative_sampler import enumerate_negative_windows_for_file  # noqa: E402


NEGATIVE_BUCKETS = (
    "reviewed_background",
    "primary_adjacent_gap",
    "nonprimary_biological_signal",
    "nonbiological_signal",
    "external_source_gap",
    "ambiguous_hard_negative",
)

PRIMARY_SPECIES = tuple(label.partition(":")[2] for label in PRIMARY_SPECIES_LABEL_IDS)
NONPRIMARY_BIOLOGICAL_CLASSES = frozenset({"OD", "CE", "UN", "P", "Bb", "Pm", "Lo", "UndBio"})
NONBIOLOGICAL_CLASSES = frozenset({"AB", "INSTRUMENT", "EQ", "SONAR", "UNKNOWN"})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _label_set(row: Mapping[str, Any]) -> set[str]:
    ids = set(label_ids_from_row(dict(row)))
    for field in ("canonical_label_ids", "source_label_ids", "analysis_label_ids", "target_label_ids"):
        ids.update(split_pipe(row.get(field)))
    return {clean_text(label) for label in ids if clean_text(label)}


def _source_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        clean_text(row.get(field)).lower()
        for field in (
            "source_dataset",
            "source_dataset_raw",
            "source_provider",
            "source_audio",
            "mat_path",
            "filename",
            "clip",
        )
    )


def has_primary_species(row: Mapping[str, Any], primary_species: Sequence[str] = PRIMARY_SPECIES) -> bool:
    primary_ids = {f"species:{normalize_species_code(code)}" for code in primary_species}
    return bool(_label_set(row) & primary_ids)


def negative_bucket_from_row(row: Mapping[str, Any], primary_species: Sequence[str] = PRIMARY_SPECIES) -> str:
    """Classify a no-primary row into an explicit negative bucket."""

    if has_primary_species(row, primary_species):
        return ""

    review_label = clean_text(row.get("review_label")).lower().replace("-", "_").replace(" ", "_")
    if review_label in set(NEGATIVE_BUCKETS):
        return review_label

    review_status_text = clean_text(row.get("review_status")).lower().replace("-", "_").replace(" ", "_")
    context_tags = {tag.lower().replace("-", "_").replace(" ", "_") for tag in split_pipe(row.get("context_tags"))}
    source_dataset = _source_text(row)
    source_class = normalize_species_code(row.get("source_class_species") or row.get("species_code") or row.get("species"))
    labels = _label_set(row)

    if review_status_text in {"reviewed_background", "reviewed_negative"} and (
        "biodcase" in source_dataset or "dclde" in source_dataset
    ):
        return "external_source_gap"
    if review_status_text in {"reviewed_background", "reviewed_negative"}:
        return "reviewed_background"
    if "primary_adjacent_gap" in context_tags:
        return "primary_adjacent_gap"
    if source_class in NONBIOLOGICAL_CLASSES or any(
        label.startswith("confounder:abiotic") or label in {"species:EQ", "species:INSTRUMENT", "species:SONAR", "species:UNKNOWN"}
        for label in labels
    ):
        return "nonbiological_signal"
    if source_class in NONPRIMARY_BIOLOGICAL_CLASSES or any(
        label.startswith("group:") or label in {"confounder:undetermined_biological"} for label in labels
    ):
        return "nonprimary_biological_signal"
    if any(label.startswith("species:") and label.split(":", 1)[1] not in set(primary_species) for label in labels):
        return "nonprimary_biological_signal"
    if "dclde" in source_dataset or "biodcase" in source_dataset:
        return "external_source_gap"
    if clean_text(row.get("is_background")).lower() in TRUTHY_VALUES:
        return "ambiguous_hard_negative"
    return "ambiguous_hard_negative"


def _clip_id(row: Mapping[str, Any]) -> str:
    return clean_text(row.get("source_audio") or row.get("filename") or row.get("clip"))


def primary_intervals_by_clip(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Tuple[float, float]]]:
    intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if not has_primary_species(row):
            continue
        clip = _clip_id(row)
        begin_s = _float_or_none(
            row.get("begin_s")
            or row.get("begin_time_s")
            or row.get("begin_time")
            or row.get("window_start_s")
        )
        end_s = _float_or_none(row.get("end_s") or row.get("end_time_s") or row.get("end_time"))
        if end_s is None and begin_s is not None:
            duration_s = _float_or_none(row.get("duration_s"))
            if duration_s is not None and duration_s > 0:
                end_s = begin_s + duration_s
        if clip and begin_s is not None and end_s is not None and end_s > begin_s:
            intervals[clip].append((begin_s, end_s))
    return dict(intervals)


def primary_adjacent_gap_rows(
    *,
    annotation_rows: Sequence[Mapping[str, Any]],
    clip_durations: Mapping[str, float],
    window_s: float = 10.0,
    exclusion_buffer_s: float = 2.0,
    step_s: Optional[float] = 10.0,
    max_windows_per_clip: int = 0,
    source_dataset: str = "onc_primary_adjacent_gap",
) -> List[Dict[str, Any]]:
    calls_by_file = primary_intervals_by_clip(annotation_rows)
    out: List[Dict[str, Any]] = []
    for clip in sorted(calls_by_file):
        duration = float(clip_durations.get(clip, 0.0))
        if duration <= 0:
            continue
        windows = enumerate_negative_windows_for_file(
            clip_id=clip,
            duration=duration,
            context_duration=float(window_s),
            calls_by_file=calls_by_file,
            margin=float(exclusion_buffer_s),
            step_seconds=step_s,
        )
        if max_windows_per_clip > 0:
            windows = windows[: int(max_windows_per_clip)]
        for idx, (begin_s, end_s) in enumerate(windows):
            item_id = f"{Path(clip).stem}_gap_{idx:04d}_{begin_s:.1f}s_{end_s:.1f}s"
            out.append(
                {
                    "item_id": item_id,
                    "clip": clip,
                    "source_audio": clip,
                    "filename": clip,
                    "begin_s": f"{begin_s:.6f}",
                    "end_s": f"{end_s:.6f}",
                    "duration_s": f"{(end_s - begin_s):.6f}",
                    "source_dataset": source_dataset,
                    "negative_bucket": "primary_adjacent_gap",
                    "review_status": "candidate_primary_adjacent_gap",
                    "context_tags": "primary_adjacent_gap",
                    "is_background": "1",
                    "label_ids": "",
                    "event_group": clip,
                }
            )
    return out


def no_primary_negative_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in rows:
        if has_primary_species(raw):
            continue
        bucket = negative_bucket_from_row(raw)
        if not bucket:
            continue
        row = dict(raw)
        row["negative_bucket"] = bucket
        row["is_background"] = "1"
        row.setdefault("label_ids", "")
        row.setdefault("event_group", group_key_for_split(row))
        out.append(row)
    return out


def leaked_groups_by_split(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[str]]:
    seen: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = clean_text(row.get("split"))
        if split:
            seen[group_key_for_split(dict(row))].add(split)
    return {group: sorted(splits) for group, splits in sorted(seen.items()) if len(splits) > 1}


def build_negative_manifest(
    *,
    annotations_csv: Path,
    output_csv: Path,
    clip_duration_csv: Optional[Path] = None,
    window_s: float = 10.0,
    exclusion_buffer_s: float = 2.0,
    step_s: float = 10.0,
    max_windows_per_clip: int = 0,
    split: bool = False,
) -> Dict[str, Any]:
    rows = _read_csv(annotations_csv)
    duration_by_clip: Dict[str, float] = {}
    if clip_duration_csv is not None:
        for row in _read_csv(clip_duration_csv):
            clip = _clip_id(row)
            duration = _float_or_none(row.get("duration_s") or row.get("clip_duration_s"))
            if clip and duration is not None:
                duration_by_clip[clip] = duration

    manifest_rows = no_primary_negative_rows(rows)
    if duration_by_clip:
        manifest_rows.extend(
            primary_adjacent_gap_rows(
                annotation_rows=rows,
                clip_durations=duration_by_clip,
                window_s=window_s,
                exclusion_buffer_s=exclusion_buffer_s,
                step_s=step_s,
                max_windows_per_clip=max_windows_per_clip,
            )
        )

    if split and manifest_rows:
        split_rows = label_balanced_grouped_split(manifest_rows)
        manifest_rows = [row for split_items in split_rows.values() for row in split_items]

    manifest_rows = sorted(
        manifest_rows,
        key=lambda row: (
            clean_text(row.get("source_dataset")),
            clean_text(row.get("clip") or row.get("filename")),
            float(_float_or_none(row.get("begin_s") or row.get("begin_time_s")) or 0.0),
            clean_text(row.get("item_id")),
        ),
    )
    write_csv_rows(output_csv, manifest_rows)

    bucket_counts = Counter(clean_text(row.get("negative_bucket")) or "<blank>" for row in manifest_rows)
    leak_count = len(leaked_groups_by_split(manifest_rows))
    summary = {
        "annotations_csv": str(annotations_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "row_count": len(manifest_rows),
        "negative_bucket_counts": dict(bucket_counts.most_common()),
        "leaked_group_count": leak_count,
        "config": {
            "clip_duration_csv": "" if clip_duration_csv is None else str(clip_duration_csv.resolve()),
            "window_s": float(window_s),
            "exclusion_buffer_s": float(exclusion_buffer_s),
            "step_s": float(step_s),
            "max_windows_per_clip": int(max_windows_per_clip),
            "split": bool(split),
        },
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--clip-duration-csv", default="")
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--exclusion-buffer-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=10.0)
    parser.add_argument("--max-windows-per-clip", type=int, default=0)
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    summary = build_negative_manifest(
        annotations_csv=Path(args.annotations_csv),
        output_csv=Path(args.output_csv),
        clip_duration_csv=Path(args.clip_duration_csv) if args.clip_duration_csv else None,
        window_s=float(args.window_s),
        exclusion_buffer_s=float(args.exclusion_buffer_s),
        step_s=float(args.step_s),
        max_windows_per_clip=int(args.max_windows_per_clip),
        split=bool(args.split),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
