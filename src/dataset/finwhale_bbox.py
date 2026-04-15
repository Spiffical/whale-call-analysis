"""Fin-whale bbox dataset parsing, normalization, and split helpers."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .negative_sampler import enumerate_negative_windows_for_file
from .part2_annotations import normalize_audio_filename, parse_filename_timestamp


FIN_SPECIES_CODE = "Bp"
HISTORICAL_DATASET = "historical_2018_2019"
SPECIES_TEMPORAL_DATASET = "species_temporal_2025"
PURE_NEGATIVE_DATASET = "mar26_verified_pure_negative_2025"
SEI_SPECIES_CODE = "Bb"

HISTORICAL_WORKBOOK_DEFAULT = "data/finwhales/Clayoquot_Call_Library_copy.xlsx"
SPECIES_TEMPORAL_WORKBOOK_DEFAULT = "data/finwhales/Clayoquot_2025_SpeciesTemporalAnalysis.xlsx"
MAR26_WORKBOOK_DEFAULT = "data/finwhales/Clayoquot_2025_Analysis_Mar26_Final.xlsx"
MAR18_WORKBOOK_DEFAULT = "data/finwhales/Clayoquot_2025_annotations_Mar18.xlsx"

FIN_LABEL_20 = "20Hz"
FIN_LABEL_30 = "30Hz"
FIN_LABEL_40 = "40Hz"
FIN_LABEL_SONG = "song"
FIN_LABEL_OTHER = "other_fin"

PURE_NEGATIVE_FLAG_COLUMNS = (
    "Bp",
    "Bm",
    "Mn",
    "Bb",
    "OD",
    "OD_CK",
    "OD_CK_low",
    "OD_CK_high",
    "OD_W",
    "OD_BP",
    "CE_unknown",
)

ANNOTATION_COLUMNS = [
    "annotation_id",
    "source_dataset",
    "source_workbook",
    "source_sheet",
    "source_row_index",
    "filename",
    "device_code",
    "clip_start_utc",
    "recording_day_utc",
    "species_code",
    "is_target_species",
    "call_type_raw",
    "call_type_std",
    "begin_time_s",
    "end_time_s",
    "duration_s",
    "low_freq_hz",
    "high_freq_hz",
    "peak_freq_hz",
    "peak_power_dbfs",
    "annotator",
    "verified_flag",
    "vessel_flag",
    "granularity",
    "comments",
    "context_tags",
    "timestamp_fix",
    "quality_flags",
]

CLIP_COLUMNS = [
    "source_dataset",
    "inventory_source",
    "filename",
    "device_code",
    "clip_start_utc",
    "recording_day_utc",
    "is_fin_positive",
    "is_annotated_non_fin",
    "is_pure_negative_candidate",
    "annotation_count",
    "fin_annotation_count",
    "non_fin_annotation_count",
    "species_codes",
    "fin_call_type_stds",
    "source_workbooks",
    "verified_flag",
]


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_colname(name: Any) -> str:
    text = _clean_text(name).lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _truthy_flag(value: Any) -> int:
    text = _clean_text(value)
    if not text:
        return 0
    if text in {"1", "1.0", "true", "True", "yes", "Yes"}:
        return 1
    try:
        return 1 if float(text) > 0 else 0
    except ValueError:
        return 0


def _device_code(filename: str) -> str:
    text = _clean_text(filename)
    if not text or "_" not in text:
        return ""
    return text.split("_", 1)[0]


def _clip_start_iso(filename: str) -> str:
    ts = parse_filename_timestamp(filename)
    return ts.isoformat() if ts is not None else ""


def _recording_day(filename: str) -> str:
    ts = parse_filename_timestamp(filename)
    return ts.strftime("%Y-%m-%d") if ts is not None else ""


def standardize_fin_call_type(raw_value: Any, species_code: str) -> str:
    raw_text = _clean_text(raw_value)
    if species_code != FIN_SPECIES_CODE:
        return raw_text
    if not raw_text:
        return FIN_LABEL_OTHER
    lowered = raw_text.lower().replace("_", " ").replace("-", " ")
    lowered = " ".join(lowered.split())
    compact = lowered.replace(" ", "")
    if compact == "20hz":
        return FIN_LABEL_20
    if compact == "30hz":
        return FIN_LABEL_30
    if compact == "40hz":
        return FIN_LABEL_40
    if compact in {"s", "song"} or "song" in lowered:
        return FIN_LABEL_SONG
    if re.search(r"\b20\s*hz\b", lowered):
        return FIN_LABEL_20
    if re.search(r"\b30\s*hz\b", lowered):
        return FIN_LABEL_30
    if re.search(r"\b40\s*hz\b", lowered):
        return FIN_LABEL_40
    return FIN_LABEL_OTHER


def _historical_time_fix(begin_s: float, end_s: float) -> Tuple[float, float, str]:
    if 300.0 <= begin_s < 600.0 and 300.0 <= end_s < 600.0:
        return begin_s - 300.0, end_s - 300.0, "minus_300s"
    if begin_s < 300.0 < end_s <= 600.0:
        return begin_s, 300.0, "clip_end_to_300s"
    return begin_s, end_s, "none"


def _annotation_id(
    source_dataset: str,
    sheet_name: str,
    row_index: int,
    filename: str,
    begin_s: float,
    end_s: float,
) -> str:
    clip_stub = Path(filename).name.replace(".", "_")
    return (
        f"{source_dataset}__{sheet_name.replace(' ', '_')}__r{int(row_index):06d}__"
        f"{clip_stub}__{begin_s:.3f}__{end_s:.3f}"
    )


def _find_column(columns: Sequence[str], *patterns: str) -> Optional[str]:
    normalized = [(_normalize_colname(col), col) for col in columns]
    for pattern in patterns:
        for norm_col, raw_col in normalized:
            if pattern in norm_col:
                return raw_col
    return None


def _combine_comments(row: pd.Series, *keys: Optional[str]) -> str:
    parts: List[str] = []
    for key in keys:
        if not key:
            continue
        value = _clean_text(row.get(key, ""))
        if value and value not in parts:
            parts.append(value)
    return " | ".join(parts)


def parse_historical_workbook(workbook_path: Path | str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = Path(workbook_path)
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "parsed_row_count": 0,
        "kept_row_count": 0,
        "sheet_counts": {},
        "drop_reasons": Counter(),
        "timestamp_fix_counts": Counter(),
    }

    with pd.ExcelFile(path) as excel:
        for sheet_name in excel.sheet_names:
            if sheet_name.startswith("~$"):
                continue
            is_sei_sheet = _clean_text(sheet_name).lower() == "sei whale calls"
            df = pd.read_excel(path, sheet_name=sheet_name)
            file_col = _find_column(df.columns, "clip id", "filename", "file")
            begin_col = _find_column(df.columns, "begin time (s)", "begin time")
            end_col = _find_column(df.columns, "end time (s)", "end time")
            low_col = _find_column(df.columns, "low freq")
            high_col = _find_column(df.columns, "high freq")
            peak_col = _find_column(df.columns, "peak freq")
            power_col = _find_column(df.columns, "peak power density", "peak power")
            type_col = _find_column(df.columns, "call type", "class")
            comments_col = _find_column(df.columns, "comments")
            note_col = _find_column(df.columns, "note or pattern analysis")
            indiv_col = _find_column(df.columns, "individual note comments")

            sheet_kept = 0
            for idx, row in df.iterrows():
                filename = normalize_audio_filename(row.get(file_col, "")) if file_col else ""
                begin_s = _as_float(row.get(begin_col)) if begin_col else None
                end_s = _as_float(row.get(end_col)) if end_col else None
                low_hz = _as_float(row.get(low_col)) if low_col else None
                high_hz = _as_float(row.get(high_col)) if high_col else None
                if not filename or begin_s is None or end_s is None or low_hz is None or high_hz is None:
                    continue

                summary["parsed_row_count"] += 1
                begin_fixed, end_fixed, timestamp_fix = _historical_time_fix(begin_s, end_s)
                summary["timestamp_fix_counts"][timestamp_fix] += 1

                duration_s = end_fixed - begin_fixed
                freq_span = high_hz - low_hz
                if duration_s <= 0:
                    summary["drop_reasons"]["nonpositive_duration"] += 1
                    continue
                if freq_span <= 0:
                    summary["drop_reasons"]["nonpositive_freq_span"] += 1
                    continue

                species_code = SEI_SPECIES_CODE if is_sei_sheet else FIN_SPECIES_CODE
                raw_type = _clean_text(row.get(type_col, "")) if type_col else ""
                call_type_std = standardize_fin_call_type(raw_type, species_code)
                comment_text = _combine_comments(row, note_col, indiv_col, comments_col)

                rows.append(
                    {
                        "annotation_id": _annotation_id(
                            HISTORICAL_DATASET,
                            sheet_name,
                            int(idx) + 2,
                            filename,
                            begin_fixed,
                            end_fixed,
                        ),
                        "source_dataset": HISTORICAL_DATASET,
                        "source_workbook": str(path.resolve()),
                        "source_sheet": sheet_name,
                        "source_row_index": int(idx) + 2,
                        "filename": filename,
                        "device_code": _device_code(filename),
                        "clip_start_utc": _clip_start_iso(filename),
                        "recording_day_utc": _recording_day(filename),
                        "species_code": species_code,
                        "is_target_species": 1 if species_code == FIN_SPECIES_CODE else 0,
                        "call_type_raw": raw_type,
                        "call_type_std": call_type_std,
                        "begin_time_s": float(begin_fixed),
                        "end_time_s": float(end_fixed),
                        "duration_s": float(duration_s),
                        "low_freq_hz": float(low_hz),
                        "high_freq_hz": float(high_hz),
                        "peak_freq_hz": _as_float(row.get(peak_col)) if peak_col else None,
                        "peak_power_dbfs": _as_float(row.get(power_col)) if power_col else None,
                        "annotator": "",
                        "verified_flag": 0,
                        "vessel_flag": 0,
                        "granularity": "",
                        "comments": comment_text,
                        "context_tags": "",
                        "timestamp_fix": timestamp_fix,
                        "quality_flags": "",
                    }
                )
                sheet_kept += 1

            summary["sheet_counts"][sheet_name] = int(sheet_kept)
            summary["kept_row_count"] += int(sheet_kept)

    out = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    return out, summary


def parse_species_temporal_workbook(workbook_path: Path | str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = Path(workbook_path)
    sheet_species = {
        "Bp_all": FIN_SPECIES_CODE,
        "Bm": "Bm",
        "Bb": "Bb",
        "Mn": "Mn",
        "Pm": "Pm",
        "OD_all": "OD",
    }
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "parsed_row_count": 0,
        "kept_row_count": 0,
        "sheet_counts": {},
        "drop_reasons": Counter(),
    }

    with pd.ExcelFile(path) as excel:
        for sheet_name, species_code in sheet_species.items():
            if sheet_name not in excel.sheet_names:
                continue
            df = pd.read_excel(path, sheet_name=sheet_name)
            comments_col = _find_column(df.columns, "comments")
            annotator_col = _find_column(df.columns, "annotator")
            granularity_col = _find_column(df.columns, "granularity")
            peak_col = _find_column(df.columns, "peak freq")
            power_col = _find_column(df.columns, "peak power")
            type_col = _find_column(df.columns, "call type")

            kept = 0
            for idx, row in df.iterrows():
                filename = normalize_audio_filename(row.get("filename", ""))
                begin_s = _as_float(row.get("begin_time"))
                end_s = _as_float(row.get("end_time"))
                low_hz = _as_float(row.get("low_freq"))
                high_hz = _as_float(row.get("high_freq"))
                if not filename or begin_s is None or end_s is None or low_hz is None or high_hz is None:
                    continue

                summary["parsed_row_count"] += 1
                duration_s = end_s - begin_s
                freq_span = high_hz - low_hz
                if duration_s <= 0:
                    summary["drop_reasons"]["nonpositive_duration"] += 1
                    continue
                if freq_span <= 0:
                    summary["drop_reasons"]["nonpositive_freq_span"] += 1
                    continue

                raw_type = _clean_text(row.get(type_col, "")) if type_col else ""
                rows.append(
                    {
                        "annotation_id": _annotation_id(
                            SPECIES_TEMPORAL_DATASET,
                            sheet_name,
                            int(idx) + 2,
                            filename,
                            begin_s,
                            end_s,
                        ),
                        "source_dataset": SPECIES_TEMPORAL_DATASET,
                        "source_workbook": str(path.resolve()),
                        "source_sheet": sheet_name,
                        "source_row_index": int(idx) + 2,
                        "filename": filename,
                        "device_code": _device_code(filename),
                        "clip_start_utc": _clip_start_iso(filename),
                        "recording_day_utc": _recording_day(filename),
                        "species_code": species_code,
                        "is_target_species": 1 if species_code == FIN_SPECIES_CODE else 0,
                        "call_type_raw": raw_type,
                        "call_type_std": standardize_fin_call_type(raw_type, species_code),
                        "begin_time_s": float(begin_s),
                        "end_time_s": float(end_s),
                        "duration_s": float(duration_s),
                        "low_freq_hz": float(low_hz),
                        "high_freq_hz": float(high_hz),
                        "peak_freq_hz": _as_float(row.get(peak_col)) if peak_col else None,
                        "peak_power_dbfs": _as_float(row.get(power_col)) if power_col else None,
                        "annotator": _clean_text(row.get(annotator_col, "")) if annotator_col else "",
                        "verified_flag": 0,
                        "vessel_flag": 0,
                        "granularity": _clean_text(row.get(granularity_col, "")) if granularity_col else "",
                        "comments": _clean_text(row.get(comments_col, "")) if comments_col else "",
                        "context_tags": "",
                        "timestamp_fix": "none",
                        "quality_flags": "",
                    }
                )
                kept += 1

            summary["sheet_counts"][sheet_name] = int(kept)
            summary["kept_row_count"] += int(kept)

    out = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    return out, summary


def collect_mar18_guardrail_filenames(workbook_path: Path | str) -> Tuple[set[str], Dict[str, Any]]:
    path = Path(workbook_path)
    filenames: set[str] = set()
    summary: Dict[str, Any] = {"kept_row_count": 0, "sheet_counts": {}}
    with pd.ExcelFile(path) as excel:
        for sheet_name in excel.sheet_names:
            if sheet_name in {"READ ME", "file_list"} or sheet_name.startswith("~$"):
                continue
            df = pd.read_excel(path, sheet_name=sheet_name)
            kept = 0
            for _, row in df.iterrows():
                filename = normalize_audio_filename(row.get("filename", ""))
                species = _clean_text(row.get("species", ""))
                begin_s = _as_float(row.get("begin_time"))
                end_s = _as_float(row.get("end_time"))
                low_hz = _as_float(row.get("low_freq"))
                high_hz = _as_float(row.get("high_freq"))
                if not filename or not species or species == "nan":
                    continue
                if begin_s is None or end_s is None or low_hz is None or high_hz is None:
                    continue
                if end_s <= begin_s or high_hz <= low_hz:
                    continue
                filenames.add(filename)
                kept += 1
            summary["sheet_counts"][sheet_name] = int(kept)
            summary["kept_row_count"] += int(kept)
    return filenames, summary


def collect_mar26_pure_negative_clips(
    workbook_path: Path | str,
    *,
    exclude_filenames: Optional[set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = Path(workbook_path)
    exclude = set(exclude_filenames or set())
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    summary: Dict[str, Any] = {
        "verified_rows": 0,
        "pure_negative_rows": 0,
        "drop_reasons": Counter(),
        "sheet_counts": {},
    }

    with pd.ExcelFile(path) as excel:
        for sheet_name in excel.sheet_names:
            if sheet_name in {"READ ME", "file_list"} or sheet_name.startswith("~$"):
                continue
            df = pd.read_excel(path, sheet_name=sheet_name)
            kept = 0
            for idx, row in df.iterrows():
                filename = normalize_audio_filename(row.get("filename", ""))
                if not filename:
                    continue
                verified = _truthy_flag(row.get("verified"))
                if verified != 1:
                    continue
                summary["verified_rows"] += 1
                if filename in exclude:
                    summary["drop_reasons"]["excluded_by_annotation_guardrail"] += 1
                    continue
                if filename in seen:
                    summary["drop_reasons"]["duplicate_filename"] += 1
                    continue

                any_species_flag = any(_truthy_flag(row.get(col)) for col in PURE_NEGATIVE_FLAG_COLUMNS if col in df.columns)
                if any_species_flag:
                    summary["drop_reasons"]["species_flag_present"] += 1
                    continue

                rows.append(
                    {
                        "source_dataset": PURE_NEGATIVE_DATASET,
                        "inventory_source": str(path.resolve()),
                        "filename": filename,
                        "device_code": _device_code(filename),
                        "clip_start_utc": _clip_start_iso(filename),
                        "recording_day_utc": _recording_day(filename),
                        "is_fin_positive": 0,
                        "is_annotated_non_fin": 0,
                        "is_pure_negative_candidate": 1,
                        "annotation_count": 0,
                        "fin_annotation_count": 0,
                        "non_fin_annotation_count": 0,
                        "species_codes": "",
                        "fin_call_type_stds": "",
                        "source_workbooks": str(path.resolve()),
                        "verified_flag": 1,
                    }
                )
                seen.add(filename)
                kept += 1

            summary["sheet_counts"][sheet_name] = int(kept)
            summary["pure_negative_rows"] += int(kept)

    out = pd.DataFrame(rows, columns=CLIP_COLUMNS)
    return out, summary


def build_clip_manifest(
    annotation_df: pd.DataFrame,
    pure_negative_df: pd.DataFrame,
) -> pd.DataFrame:
    grouped_rows: List[Dict[str, Any]] = []
    if not annotation_df.empty:
        for (source_dataset, filename), group in annotation_df.groupby(["source_dataset", "filename"], sort=True):
            species_codes = sorted({_clean_text(v) for v in group["species_code"].tolist() if _clean_text(v)})
            fin_rows = group[group["species_code"] == FIN_SPECIES_CODE]
            non_fin_rows = group[group["species_code"] != FIN_SPECIES_CODE]
            grouped_rows.append(
                {
                    "source_dataset": source_dataset,
                    "inventory_source": str(group["source_workbook"].iloc[0]),
                    "filename": filename,
                    "device_code": _device_code(filename),
                    "clip_start_utc": _clip_start_iso(filename),
                    "recording_day_utc": _recording_day(filename),
                    "is_fin_positive": 1 if not fin_rows.empty else 0,
                    "is_annotated_non_fin": 1 if not non_fin_rows.empty else 0,
                    "is_pure_negative_candidate": 0,
                    "annotation_count": int(len(group)),
                    "fin_annotation_count": int(len(fin_rows)),
                    "non_fin_annotation_count": int(len(non_fin_rows)),
                    "species_codes": "|".join(species_codes),
                    "fin_call_type_stds": "|".join(
                        sorted({_clean_text(v) for v in fin_rows["call_type_std"].tolist() if _clean_text(v)})
                    ),
                    "source_workbooks": "|".join(sorted({str(v) for v in group["source_workbook"].tolist() if _clean_text(v)})),
                    "verified_flag": int(group["verified_flag"].astype(int).max()) if "verified_flag" in group.columns else 0,
                }
            )

    clip_df = pd.DataFrame(grouped_rows, columns=CLIP_COLUMNS)
    if pure_negative_df is not None and not pure_negative_df.empty:
        clip_df = pd.concat([clip_df, pure_negative_df[CLIP_COLUMNS]], ignore_index=True)
    if clip_df.empty:
        return pd.DataFrame(columns=CLIP_COLUMNS)
    clip_df = clip_df.sort_values(["source_dataset", "recording_day_utc", "filename"]).reset_index(drop=True)
    return clip_df


def build_joint_bbox_manifests(
    *,
    historical_workbook: Path | str,
    species_temporal_workbook: Path | str,
    mar26_workbook: Path | str,
    mar18_workbook: Optional[Path | str] = None,
) -> Dict[str, Any]:
    historical_df, historical_summary = parse_historical_workbook(historical_workbook)
    species_temporal_df, species_temporal_summary = parse_species_temporal_workbook(species_temporal_workbook)

    guardrail_filenames: set[str] = set()
    mar18_summary: Dict[str, Any] = {"kept_row_count": 0, "sheet_counts": {}}
    if mar18_workbook:
        guardrail_filenames, mar18_summary = collect_mar18_guardrail_filenames(mar18_workbook)

    exclude_filenames = set(species_temporal_df["filename"].astype(str).tolist()) | set(guardrail_filenames)
    pure_negative_df, pure_negative_summary = collect_mar26_pure_negative_clips(
        mar26_workbook,
        exclude_filenames=exclude_filenames,
    )

    frames = [frame for frame in (historical_df, species_temporal_df) if frame is not None and not frame.empty]
    if frames:
        records: List[Dict[str, Any]] = []
        for frame in frames:
            records.extend(frame[ANNOTATION_COLUMNS].to_dict("records"))
        annotation_df = pd.DataFrame(records, columns=ANNOTATION_COLUMNS)
    else:
        annotation_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
    clip_df = build_clip_manifest(annotation_df, pure_negative_df)

    summary = {
        "historical_workbook": str(Path(historical_workbook).resolve()),
        "species_temporal_workbook": str(Path(species_temporal_workbook).resolve()),
        "mar26_workbook": str(Path(mar26_workbook).resolve()),
        "mar18_guardrail_workbook": str(Path(mar18_workbook).resolve()) if mar18_workbook else "",
        "annotation_count": int(len(annotation_df)),
        "clip_count": int(len(clip_df)),
        "pure_negative_clip_count": int((clip_df["is_pure_negative_candidate"] == 1).sum()) if not clip_df.empty else 0,
        "source_annotation_counts": dict(
            Counter(annotation_df["source_dataset"].astype(str).tolist())
        ),
        "species_counts": dict(Counter(annotation_df["species_code"].astype(str).tolist())),
        "fin_call_type_counts": dict(
            Counter(
                annotation_df.loc[annotation_df["species_code"] == FIN_SPECIES_CODE, "call_type_std"]
                .astype(str)
                .tolist()
            )
        ),
        "historical_summary": {
            "parsed_row_count": int(historical_summary["parsed_row_count"]),
            "kept_row_count": int(historical_summary["kept_row_count"]),
            "sheet_counts": historical_summary["sheet_counts"],
            "drop_reasons": dict(historical_summary["drop_reasons"]),
            "timestamp_fix_counts": dict(historical_summary["timestamp_fix_counts"]),
        },
        "species_temporal_summary": {
            "parsed_row_count": int(species_temporal_summary["parsed_row_count"]),
            "kept_row_count": int(species_temporal_summary["kept_row_count"]),
            "sheet_counts": species_temporal_summary["sheet_counts"],
            "drop_reasons": dict(species_temporal_summary["drop_reasons"]),
        },
        "mar18_guardrail_summary": mar18_summary,
        "mar26_pure_negative_summary": {
            "verified_rows": int(pure_negative_summary["verified_rows"]),
            "pure_negative_rows": int(pure_negative_summary["pure_negative_rows"]),
            "sheet_counts": pure_negative_summary["sheet_counts"],
            "drop_reasons": dict(pure_negative_summary["drop_reasons"]),
        },
    }
    return {
        "annotations": annotation_df,
        "clip_manifest": clip_df,
        "pure_negative_clips": pure_negative_df,
        "summary": summary,
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_joint_bbox_manifests(output_dir: Path | str, manifests: Dict[str, Any]) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    ann_path = out_dir / "unified_annotations.csv"
    clip_path = out_dir / "clip_manifest.csv"
    pure_neg_path = out_dir / "pure_negative_clip_inventory.csv"
    summary_path = out_dir / "summary.json"

    _write_csv(ann_path, manifests["annotations"])
    _write_csv(clip_path, manifests["clip_manifest"])
    _write_csv(pure_neg_path, manifests["pure_negative_clips"])
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(manifests["summary"], handle, indent=2, sort_keys=True)

    written["annotations"] = ann_path
    written["clip_manifest"] = clip_path
    written["pure_negative_clips"] = pure_neg_path
    written["summary"] = summary_path
    return written


def load_annotation_manifest(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    for col in ("begin_time_s", "end_time_s", "duration_s", "low_freq_hz", "high_freq_hz", "peak_freq_hz", "peak_power_dbfs"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("source_row_index", "is_target_species", "verified_flag", "vessel_flag"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def load_clip_manifest(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=CLIP_COLUMNS)
    for col in (
        "is_fin_positive",
        "is_annotated_non_fin",
        "is_pure_negative_candidate",
        "annotation_count",
        "fin_annotation_count",
        "non_fin_annotation_count",
        "verified_flag",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _round_robin_day_order(day_rows: pd.DataFrame, weight_column: str) -> List[str]:
    month_groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for row in day_rows.to_dict("records"):
        day = str(row["recording_day_utc"])
        month = day[:7]
        month_groups[month].append((day, float(row.get(weight_column, 0.0))))
    for month in month_groups:
        month_groups[month] = sorted(
            month_groups[month],
            key=lambda item: (-item[1], item[0]),
        )
    ordered: List[str] = []
    progress = True
    months = sorted(month_groups)
    while progress:
        progress = False
        for month in months:
            if not month_groups[month]:
                continue
            ordered.append(month_groups[month].pop(0)[0])
            progress = True
    return ordered


def _assign_days_by_weight(
    day_rows: pd.DataFrame,
    *,
    weight_column: str,
    split_names: Tuple[str, str, str],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    if day_rows.empty:
        return {}
    ordered_days = _round_robin_day_order(day_rows, weight_column=weight_column)
    weight_by_day = {
        str(row["recording_day_utc"]): float(row.get(weight_column, 0.0))
        for row in day_rows.to_dict("records")
    }
    total_weight = sum(max(0.0, w) for w in weight_by_day.values())
    train_cut = total_weight * train_ratio
    val_cut = total_weight * (train_ratio + val_ratio)
    split_map: Dict[str, str] = {}
    cumulative = 0.0
    train_name, val_name, test_name = split_names
    for idx, day in enumerate(ordered_days):
        weight = max(0.0, weight_by_day.get(day, 0.0))
        if total_weight <= 0:
            frac = (idx + 1) / max(1, len(ordered_days))
            if frac <= train_ratio:
                split = train_name
            elif frac <= train_ratio + val_ratio:
                split = val_name
            else:
                split = test_name
        else:
            if cumulative < train_cut:
                split = train_name
            elif cumulative < val_cut:
                split = val_name
            else:
                split = test_name
        split_map[day] = split
        cumulative += weight
    return split_map


def build_bbox_splits(
    annotation_df: pd.DataFrame,
    clip_df: pd.DataFrame,
) -> Dict[str, Any]:
    assignments: List[Dict[str, Any]] = []

    hist_annotated = clip_df[
        (clip_df["source_dataset"] == HISTORICAL_DATASET)
        & (clip_df["is_pure_negative_candidate"] == 0)
    ].copy()
    hist_days = (
        hist_annotated.groupby("recording_day_utc", as_index=False)
        .agg(fin_annotation_count=("fin_annotation_count", "sum"))
    )
    hist_day_split = _assign_days_by_weight(
        hist_days,
        weight_column="fin_annotation_count",
        split_names=("train", "val_hist", "test_hist"),
        train_ratio=0.8,
        val_ratio=0.1,
    )

    ann_2025 = clip_df[
        (clip_df["source_dataset"] == SPECIES_TEMPORAL_DATASET)
        & (clip_df["is_pure_negative_candidate"] == 0)
    ].copy()
    ann_2025_days = (
        ann_2025.groupby("recording_day_utc", as_index=False)
        .agg(fin_annotation_count=("fin_annotation_count", "sum"))
    )
    ann_2025_day_split = _assign_days_by_weight(
        ann_2025_days,
        weight_column="fin_annotation_count",
        split_names=("train", "val_2025", "test_2025"),
        train_ratio=0.7,
        val_ratio=0.15,
    )

    pure_2025 = clip_df[clip_df["source_dataset"] == PURE_NEGATIVE_DATASET].copy()
    pure_inherit_mask = pure_2025["recording_day_utc"].isin(ann_2025_day_split)
    pure_inherit = pure_2025[pure_inherit_mask].copy()
    pure_only = pure_2025[~pure_inherit_mask].copy()
    pure_only_days = (
        pure_only.groupby("recording_day_utc", as_index=False)
        .agg(clip_count=("filename", "count"))
    )
    pure_only_day_split = _assign_days_by_weight(
        pure_only_days,
        weight_column="clip_count",
        split_names=("train", "val_2025", "test_2025"),
        train_ratio=0.7,
        val_ratio=0.15,
    )

    for row in clip_df.to_dict("records"):
        source_dataset = str(row["source_dataset"])
        day = str(row["recording_day_utc"])
        if source_dataset == HISTORICAL_DATASET:
            split_name = hist_day_split.get(day, "train")
            split_reason = "historical_day_group"
        elif source_dataset == SPECIES_TEMPORAL_DATASET:
            split_name = ann_2025_day_split.get(day, "train")
            split_reason = "species_temporal_day_group"
        elif source_dataset == PURE_NEGATIVE_DATASET:
            if day in ann_2025_day_split:
                split_name = ann_2025_day_split[day]
                split_reason = "inherit_species_temporal_day_group"
            else:
                split_name = pure_only_day_split.get(day, "train")
                split_reason = "pure_negative_day_group"
        else:
            split_name = "train"
            split_reason = "fallback"
        assignments.append(
            {
                "source_dataset": source_dataset,
                "filename": str(row["filename"]),
                "recording_day_utc": day,
                "split_name": split_name,
                "split_reason": split_reason,
                "is_fin_positive": int(row["is_fin_positive"]),
                "is_annotated_non_fin": int(row["is_annotated_non_fin"]),
                "is_pure_negative_candidate": int(row["is_pure_negative_candidate"]),
                "fin_annotation_count": int(row["fin_annotation_count"]),
                "non_fin_annotation_count": int(row["non_fin_annotation_count"]),
            }
        )

    assignment_df = pd.DataFrame(assignments)
    split_summary: Dict[str, Any] = {}
    for split_name, group in assignment_df.groupby("split_name", sort=True):
        filenames = set(group["filename"].astype(str).tolist())
        anns = annotation_df[annotation_df["filename"].astype(str).isin(filenames)]
        split_summary[split_name] = {
            "clip_count": int(len(group)),
            "fin_positive_clip_count": int(group["is_fin_positive"].sum()),
            "annotated_non_fin_clip_count": int(group["is_annotated_non_fin"].sum()),
            "pure_negative_clip_count": int(group["is_pure_negative_candidate"].sum()),
            "annotation_count": int(len(anns)),
            "fin_annotation_count": int((anns["species_code"] == FIN_SPECIES_CODE).sum()),
            "non_fin_annotation_count": int((anns["species_code"] != FIN_SPECIES_CODE).sum()),
        }

    return {
        "assignments": assignment_df.sort_values(["split_name", "source_dataset", "recording_day_utc", "filename"]).reset_index(drop=True),
        "summary": split_summary,
    }


def write_bbox_splits(output_dir: Path | str, split_data: Dict[str, Any]) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = out_dir / "assignments.csv"
    summary_path = out_dir / "summary.json"
    split_data["assignments"].to_csv(assignment_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(split_data["summary"], handle, indent=2, sort_keys=True)

    written = {"assignments": assignment_path, "summary": summary_path}
    for split_name, group in split_data["assignments"].groupby("split_name", sort=True):
        path = out_dir / f"{split_name}.txt"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            for filename in sorted(group["filename"].astype(str).tolist()):
                handle.write(f"{filename}\n")
        written[split_name] = path
    return written


def load_split_assignments(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in (
        "is_fin_positive",
        "is_annotated_non_fin",
        "is_pure_negative_candidate",
        "fin_annotation_count",
        "non_fin_annotation_count",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def annotation_calls_by_file(annotation_df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    calls: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in annotation_df.to_dict("records"):
        filename = str(row["filename"])
        begin_s = _as_float(row.get("begin_time_s"))
        end_s = _as_float(row.get("end_time_s"))
        if begin_s is None or end_s is None or end_s <= begin_s:
            continue
        calls[filename].append((float(begin_s), float(end_s)))
    for filename in calls:
        calls[filename] = sorted(calls[filename])
    return calls


def enumerate_gap_negative_contexts(
    annotation_df: pd.DataFrame,
    *,
    context_duration_s: float = 40.0,
    clip_duration_s: float = 300.0,
    negative_margin_s: float = 2.0,
) -> pd.DataFrame:
    calls_by_file = annotation_calls_by_file(annotation_df)
    rows: List[Dict[str, Any]] = []
    for filename, windows in sorted(calls_by_file.items()):
        negatives = enumerate_negative_windows_for_file(
            clip_id=filename,
            duration=float(clip_duration_s),
            context_duration=float(context_duration_s),
            calls_by_file=calls_by_file,
            margin=float(negative_margin_s),
            step_seconds=None,
        )
        for neg_idx, (start_s, end_s) in enumerate(negatives):
            rows.append(
                {
                    "filename": filename,
                    "context_type": "gap_negative",
                    "context_start_s": float(start_s),
                    "context_end_s": float(end_s),
                    "context_duration_s": float(context_duration_s),
                    "context_index": int(neg_idx),
                }
            )
    return pd.DataFrame(rows)
