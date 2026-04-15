"""Audit fin-whale bbox audio requirements, including boundary-adjacent files."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .finwhale_bbox import (
    HISTORICAL_DATASET,
    PURE_NEGATIVE_DATASET,
    SPECIES_TEMPORAL_DATASET,
    load_annotation_manifest,
    load_clip_manifest,
    load_split_assignments,
)
from .finwhale_bbox_export import build_context_manifest
from .part2_annotations import parse_filename_timestamp


EVENT_CONTEXT_TYPES = {"fin_positive", "annotated_nonfin"}
COHORT_HISTORICAL = "historical_2018_2019"
COHORT_2025 = "all_2025"


@dataclass(frozen=True)
class AudioAuditConfig:
    historical_audio_dir: Path
    audio_2025_dir: Path
    context_duration_s: float = 40.0
    clip_duration_s: float = 300.0
    edge_buffer_s: float = 2.0
    pure_zero_ratio: float = 0.5
    negative_margin_s: float = 2.0


def _index_audio_dir(audio_dir: Path | str) -> set[str]:
    root = Path(audio_dir)
    out: set[str] = set()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix not in {".wav", ".flac"}:
                continue
            out.add(filename)
    return out


def _cohort_for_source_dataset(source_dataset: str) -> str:
    if str(source_dataset) == HISTORICAL_DATASET:
        return COHORT_HISTORICAL
    return COHORT_2025


def _audio_index_for_source_dataset(source_dataset: str, *, hist_index: set[str], audio_2025_index: set[str]) -> set[str]:
    if str(source_dataset) == HISTORICAL_DATASET:
        return hist_index
    return audio_2025_index


def _adjacent_filename(filename: str, delta_seconds: float) -> str:
    ts = parse_filename_timestamp(filename)
    if ts is None:
        return ""
    path = Path(filename)
    device_code = str(filename).split("_", 1)[0]
    adj_ts = ts + timedelta(seconds=float(delta_seconds))
    return f"{device_code}_{adj_ts.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z{path.suffix}"


def _summarize_requirement_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    summary: Dict[str, Any] = {}
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["cohort"]), str(row["policy"]))].append(row)

    for (cohort, policy), group_rows in sorted(grouped.items()):
        existing_required = {str(row["required_filename"]) for row in group_rows if bool(row["exists"])}
        missing_rows = [row for row in group_rows if not bool(row["exists"])]
        summary_key = f"{cohort}:{policy}"
        summary[summary_key] = {
            "cohort": cohort,
            "policy": policy,
            "requirement_row_count": int(len(group_rows)),
            "unique_existing_required_files": int(len(existing_required)),
            "missing_requirement_count": int(len(missing_rows)),
            "missing_by_role": dict(Counter(str(row["role"]) for row in missing_rows)),
            "missing_by_source_dataset": dict(Counter(str(row["source_dataset"]) for row in missing_rows)),
        }
    return summary


def audit_audio_requirements(
    *,
    annotation_manifest_csv: Path | str,
    clip_manifest_csv: Path | str,
    split_assignments_csv: Path | str,
    config: AudioAuditConfig,
    allowed_filenames: Optional[set[str]] = None,
) -> Dict[str, Any]:
    annotation_df = load_annotation_manifest(annotation_manifest_csv)
    clip_df = load_clip_manifest(clip_manifest_csv)
    assignments_df = load_split_assignments(split_assignments_csv)

    context_df, context_summary = build_context_manifest(
        annotation_df,
        clip_df,
        assignments_df,
        context_duration_s=float(config.context_duration_s),
        pure_zero_ratio=float(config.pure_zero_ratio),
        negative_margin_s=float(config.negative_margin_s),
        allowed_filenames=allowed_filenames,
    )

    hist_index = _index_audio_dir(config.historical_audio_dir)
    audio_2025_index = _index_audio_dir(config.audio_2025_dir)

    requirement_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    for row in context_df.to_dict("records"):
        source_dataset = str(row["source_dataset"])
        filename = str(row["filename"])
        split_name = str(row["split_name"])
        context_type = str(row["context_type"])
        cohort = _cohort_for_source_dataset(source_dataset)
        audio_index = _audio_index_for_source_dataset(source_dataset, hist_index=hist_index, audio_2025_index=audio_2025_index)

        # Main file is always required.
        requirement_rows.append(
            {
                "cohort": cohort,
                "policy": "current_export_render",
                "source_dataset": source_dataset,
                "split_name": split_name,
                "context_id": str(row["context_id"]),
                "context_type": context_type,
                "filename": filename,
                "required_filename": filename,
                "role": "main",
                "exists": int(filename in audio_index),
                "reason": "context_main_file",
            }
        )

        if float(row["context_start_s"]) < float(config.edge_buffer_s):
            prev_name = _adjacent_filename(filename, -float(config.clip_duration_s))
            requirement_rows.append(
                {
                    "cohort": cohort,
                    "policy": "current_export_render",
                    "source_dataset": source_dataset,
                    "split_name": split_name,
                    "context_id": str(row["context_id"]),
                    "context_type": context_type,
                    "filename": filename,
                    "required_filename": prev_name,
                    "role": "prev",
                    "exists": int(bool(prev_name) and prev_name in audio_index),
                    "reason": "edge_buffer_before_context",
                }
            )

        if float(row["context_end_s"]) > float(config.clip_duration_s) - float(config.edge_buffer_s):
            next_name = _adjacent_filename(filename, float(config.clip_duration_s))
            requirement_rows.append(
                {
                    "cohort": cohort,
                    "policy": "current_export_render",
                    "source_dataset": source_dataset,
                    "split_name": split_name,
                    "context_id": str(row["context_id"]),
                    "context_type": context_type,
                    "filename": filename,
                    "required_filename": next_name,
                    "role": "next",
                    "exists": int(bool(next_name) and next_name in audio_index),
                    "reason": "edge_buffer_after_context",
                }
            )

        if context_type not in EVENT_CONTEXT_TYPES:
            continue

        event_begin = float(row["event_begin_s"])
        event_end = float(row["event_end_s"])
        event_center = 0.5 * (event_begin + event_end)
        distance_to_edge = min(event_center, float(config.clip_duration_s) - event_center)
        need_prev_centered = int((event_center - 0.5 * float(config.context_duration_s)) < 0.0)
        need_next_centered = int((event_center + 0.5 * float(config.context_duration_s)) > float(config.clip_duration_s))
        prev_center_name = _adjacent_filename(filename, -float(config.clip_duration_s)) if need_prev_centered else ""
        next_center_name = _adjacent_filename(filename, float(config.clip_duration_s)) if need_next_centered else ""

        event_rows.append(
            {
                "cohort": cohort,
                "source_dataset": source_dataset,
                "split_name": split_name,
                "context_id": str(row["context_id"]),
                "context_type": context_type,
                "filename": filename,
                "event_begin_s": event_begin,
                "event_end_s": event_end,
                "event_center_s": float(event_center),
                "distance_to_edge_s": float(distance_to_edge),
                "within_20s_of_edge": int(distance_to_edge < 20.0),
                "within_40s_of_edge": int(distance_to_edge < 40.0),
                "needs_adjacent_for_centered_40s_prev": int(need_prev_centered),
                "needs_adjacent_for_centered_40s_next": int(need_next_centered),
                "centered_prev_filename": prev_center_name,
                "centered_prev_exists": int(bool(prev_center_name) and prev_center_name in audio_index),
                "centered_next_filename": next_center_name,
                "centered_next_exists": int(bool(next_center_name) and next_center_name in audio_index),
            }
        )

        for role, need_flag, adj_name in (
            ("prev", need_prev_centered, prev_center_name),
            ("next", need_next_centered, next_center_name),
        ):
            if not need_flag:
                continue
            requirement_rows.append(
                {
                    "cohort": cohort,
                    "policy": "centered_40s_event_context",
                    "source_dataset": source_dataset,
                    "split_name": split_name,
                    "context_id": str(row["context_id"]),
                    "context_type": context_type,
                    "filename": filename,
                    "required_filename": adj_name,
                    "role": role,
                    "exists": int(bool(adj_name) and adj_name in audio_index),
                    "reason": "centered_40s_event_context",
                }
            )

    requirement_df = pd.DataFrame(requirement_rows)
    event_df = pd.DataFrame(event_rows)

    file_lists: Dict[str, List[str]] = {}
    if not requirement_df.empty:
        for (cohort, policy), group in requirement_df.groupby(["cohort", "policy"], sort=True):
            names = sorted(set(group.loc[group["exists"] == 1, "required_filename"].astype(str).tolist()))
            file_lists[f"{cohort}:{policy}"] = names

    event_summary: Dict[str, Any] = {}
    if not event_df.empty:
        for cohort, group in event_df.groupby("cohort", sort=True):
            centered_need = group[
                (group["needs_adjacent_for_centered_40s_prev"] == 1)
                | (group["needs_adjacent_for_centered_40s_next"] == 1)
            ]
            centered_missing = centered_need[
                ((centered_need["needs_adjacent_for_centered_40s_prev"] == 1) & (centered_need["centered_prev_exists"] == 0))
                | ((centered_need["needs_adjacent_for_centered_40s_next"] == 1) & (centered_need["centered_next_exists"] == 0))
            ]
            event_summary[str(cohort)] = {
                "event_context_count": int(len(group)),
                "within_20s_of_edge_count": int(group["within_20s_of_edge"].sum()),
                "within_40s_of_edge_count": int(group["within_40s_of_edge"].sum()),
                "centered_40s_adjacent_needed_count": int(len(centered_need)),
                "centered_40s_adjacent_missing_count": int(len(centered_missing)),
            }

    summary = {
        "historical_audio_dir": str(Path(config.historical_audio_dir).resolve()),
        "audio_2025_dir": str(Path(config.audio_2025_dir).resolve()),
        "historical_audio_index_count": int(len(hist_index)),
        "audio_2025_index_count": int(len(audio_2025_index)),
        "context_summary": context_summary,
        "event_summary": event_summary,
        "requirement_summary": _summarize_requirement_records(requirement_rows),
    }

    return {
        "summary": summary,
        "requirement_df": requirement_df,
        "event_df": event_df,
        "file_lists": file_lists,
    }


def write_audio_audit(output_dir: Path | str, audit: Dict[str, Any]) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(audit["summary"], handle, indent=2, sort_keys=True)

    requirement_path = out_dir / "requirement_records.csv"
    audit["requirement_df"].to_csv(requirement_path, index=False)

    event_path = out_dir / "event_edge_audit.csv"
    audit["event_df"].to_csv(event_path, index=False)

    written = {
        "summary": summary_path,
        "requirement_records": requirement_path,
        "event_edge_audit": event_path,
    }

    for key, names in sorted(audit["file_lists"].items()):
        safe_key = key.replace(":", "__")
        path = out_dir / f"{safe_key}.txt"
        path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        written[safe_key] = path
    return written
