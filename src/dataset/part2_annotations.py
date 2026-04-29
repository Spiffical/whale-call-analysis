"""Helpers for parsing and normalizing the 2025 Part 2 fin-whale workbook.

This module intentionally avoids non-stdlib spreadsheet dependencies so that we
can inspect and normalize the annotation workbook in lightweight environments.
The resulting manifests are used by:

- VM-side prep on ONC infrastructure
- Nibi evaluation and report generation
- local smoke tests
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PKGREL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOCREL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {
    "main": MAIN_NS,
    "pkgrel": PKGREL_NS,
}

FIN_SPECIES_CODE = "Bp"
ANNOTATIONS_2025_DATASET = "clayoquot_2025_final_annotations"
PURE_NEGATIVE_DATASET = "mar26_verified_pure_negative_2025"
READ_ME_SHEET = "READ ME"
INVENTORY_SHEET = "file_list"
ANNOTATIONS_2025_WORKBOOK_DEFAULT = (
    "data/finwhales/ONC_ClayoquotSlope2025_Annotations_Cetaceans_Instrument_EQ_Sonar_Unknown.xlsx"
)
MAR26_WORKBOOK_DEFAULT = "data/finwhales/Clayoquot_2025_Analysis_Mar26_Final.xlsx"
MAR18_WORKBOOK_DEFAULT = "data/finwhales/Clayoquot_2025_annotations_Mar18.xlsx"
FIN_BUCKET_20 = "20Hz"
FIN_BUCKET_40 = "40Hz"
FIN_BUCKET_OTHER = "other_fin"
UNKNOWN_CONTEXT = "unknown_other"
DEFAULT_CLIP_DURATION_S = 300.0
DEFAULT_ADJACENT_BOUNDARY_SECONDS = 20.0
INSTRUMENT_SIGNAL_CODE = "INSTRUMENT"
EARTHQUAKE_SIGNAL_CODE = "EQ"
UNKNOWN_SIGNAL_CODE = "UNKNOWN"
SONAR_SIGNAL_CODE = "SONAR"
PURE_NEGATIVE_FLAG_COLUMNS = (
    "bp",
    "bm",
    "mn",
    "bb",
    "od",
    "od_ck",
    "od_ck_low",
    "od_ck_high",
    "od_w",
    "od_bp",
    "ce_unknown",
)

_CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")
_WHITESPACE_RE = re.compile(r"\s+")
_FILENAME_TS_RE = re.compile(r"(\d{8}T\d{6})(?:\.(\d{3}))?Z")
_AUDIO_FILENAME_RE = re.compile(
    r"([A-Za-z0-9_-]+_\d{8}T\d{6}(?:\.\d{3})?Z\.(?:wav|flac))",
    re.IGNORECASE,
)
_FILE_WINDOW_RE = re.compile(
    r"^(?P<source>.+)_(?P<start>-?\d+(?:\.\d+)?)s_(?P<end>-?\d+(?:\.\d+)?)s_window$"
)


@dataclass(frozen=True)
class WorkbookSheet:
    name: str
    rows: List[Dict[str, str]]


def _column_letters(cell_ref: str) -> str:
    match = _CELL_REF_RE.match(cell_ref)
    if not match:
        return ""
    return match.group(1)


def _normalize_header(value: str) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value or "").strip()).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "column"


def _make_unique_headers(values: Sequence[str]) -> Dict[str, str]:
    seen: Counter[str] = Counter()
    out: Dict[str, str] = {}
    for col_ref, raw in values:
        base = _normalize_header(raw)
        seen[base] += 1
        if seen[base] > 1:
            base = f"{base}_{seen[base]}"
        out[col_ref] = base
    return out


def _parse_shared_strings(zf: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    shared: List[str] = []
    for item in root.findall("main:si", NS):
        text = "".join(node.text or "" for node in item.iterfind(".//main:t", NS))
        shared.append(text)
    return shared


def _cell_text(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iterfind(".//main:t", NS))

    value_node = cell.find("main:v", NS)
    if value_node is None:
        return ""
    value = value_node.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except Exception:
            return value
    return value


def _sheet_targets(zf: ZipFile) -> List[Tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pkgrel:Relationship", NS)
    }

    out: List[Tuple[str, str]] = []
    for sheet in workbook.findall("main:sheets/main:sheet", NS):
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(f"{{{DOCREL_NS}}}id", "")
        target = rel_map.get(rel_id, "")
        if not target:
            continue
        out.append((name, f"xl/{target}" if not target.startswith("xl/") else target))
    return out


def load_workbook_sheets(path: Path | str) -> List[WorkbookSheet]:
    """Load workbook rows as plain-text dictionaries keyed by normalized headers."""
    workbook_path = Path(path)
    with ZipFile(workbook_path) as zf:
        shared_strings = _parse_shared_strings(zf)
        sheets: List[WorkbookSheet] = []
        for sheet_name, target in _sheet_targets(zf):
            root = ET.fromstring(zf.read(target))
            row_nodes = root.findall(".//main:sheetData/main:row", NS)
            header_map: Dict[str, str] = {}
            rows: List[Dict[str, str]] = []
            for row_idx, row_node in enumerate(row_nodes, start=1):
                cells: List[Tuple[str, str]] = []
                for cell in row_node.findall("main:c", NS):
                    cell_ref = cell.attrib.get("r", "")
                    col_ref = _column_letters(cell_ref)
                    if not col_ref:
                        continue
                    cells.append((col_ref, _cell_text(cell, shared_strings)))
                if row_idx == 1:
                    header_map = _make_unique_headers(cells)
                    continue
                if not header_map:
                    continue
                row_dict = {name: "" for name in header_map.values()}
                for col_ref, value in cells:
                    header = header_map.get(col_ref)
                    if header:
                        row_dict[header] = value.strip()
                rows.append(row_dict)
            sheets.append(WorkbookSheet(name=sheet_name, rows=rows))
    return sheets


def _clean_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = _clean_text(str(value))
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        parts = text.split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
        except Exception:
            return None
    return None


def _as_int_flag(*values: str) -> int:
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        if text in {"1", "1.0", "true", "True", "yes", "Yes"}:
            return 1
        try:
            return 1 if float(text) > 0 else 0
        except ValueError:
            continue
    return 0


def _truthy_flag(value: object) -> int:
    text = _clean_text(value)
    if not text or text.lower() == "nan":
        return 0
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return 1
    try:
        return 1 if float(text) > 0 else 0
    except ValueError:
        return 0


def _normalize_species(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    lowered = text.lower()
    alias_map = {
        "bp": FIN_SPECIES_CODE,
        "bm": "Bm",
        "bb": "Bb",
        "mn": "Mn",
        "pm": "Pm",
        "oo": "Oo",
        "lo": "Lo",
        "od": "OD",
        "ce": "CE",
        "un": "UN",
        "ma": "MA",
        "ba": "BA",
        "p": "P",
        "eq": EARTHQUAKE_SIGNAL_CODE,
        "earthquake": EARTHQUAKE_SIGNAL_CODE,
        "sonar": SONAR_SIGNAL_CODE,
        "unknown": UNKNOWN_SIGNAL_CODE,
        "instrument": INSTRUMENT_SIGNAL_CODE,
        "hydrophone_thud": INSTRUMENT_SIGNAL_CODE,
        "hydrophone thud": INSTRUMENT_SIGNAL_CODE,
    }
    return alias_map.get(lowered, text)


def _normalize_call_type_raw(value: str) -> str:
    text = _clean_text(value)
    text = text.replace("  ", " ")
    return text


def normalize_audio_filename(value: str) -> str:
    raw_text = str(value or "").strip()
    if not raw_text:
        return ""
    match = _AUDIO_FILENAME_RE.search(raw_text)
    if match:
        return match.group(1)
    return _clean_text(raw_text)


def bucket_fin_call_type(raw_value: str, species_code: str) -> str:
    normalized = re.sub(r"\s+", "", _normalize_call_type_raw(raw_value).lower())
    if normalized == "20hz":
        return FIN_BUCKET_20
    if normalized == "40hz":
        return FIN_BUCKET_40
    if species_code == FIN_SPECIES_CODE:
        return FIN_BUCKET_OTHER
    return ""


def infer_context_tags(
    comments: str,
    vessel_flag: int,
    species_code: str,
    clip_species_codes: Optional[Sequence[str]] = None,
) -> List[str]:
    text = _normalize_call_type_raw(comments).lower()
    tags = set()

    if species_code in {
        INSTRUMENT_SIGNAL_CODE,
        EARTHQUAKE_SIGNAL_CODE,
        UNKNOWN_SIGNAL_CODE,
        SONAR_SIGNAL_CODE,
    }:
        tags.add("non_biological")
    if species_code == INSTRUMENT_SIGNAL_CODE:
        tags.add("instrument_noise")
    if species_code == EARTHQUAKE_SIGNAL_CODE:
        tags.add("earthquake")
    if species_code == UNKNOWN_SIGNAL_CODE:
        tags.add("unknown_signal")
    if species_code == SONAR_SIGNAL_CODE:
        tags.add("sonar")

    if vessel_flag or any(token in text for token in ("vessel", "mask", "masking", "ship", "lloyd", "lpf", "noise")):
        tags.add("vessel_or_masking")
    if any(token in text for token in ("faint", "very faint", "weak")):
        tags.add("faint")
    if any(token in text for token in ("song", "double", "doubles", "twin note", "twin notes", "ab song", "b song")):
        tags.add("song")
    if any(token in text for token in ("irregular", "variable ici", "varibale", "no obvious spectral notch")):
        tags.add("irregular")
    if any(token in text for token in ("click", "click train", "whistle", "spectral notch", "dolphin", "odontocete", "ck")):
        tags.add("click_overlap")
    if species_code in {"UN", "Oo"} or "?" in text:
        tags.add(UNKNOWN_CONTEXT)

    if clip_species_codes:
        unique_species = {code for code in clip_species_codes if code}
        if len(unique_species) > 1:
            tags.add("mixed_species")

    if not tags:
        tags.add(UNKNOWN_CONTEXT)
    return sorted(tags)


def _is_annotation_row(row: Dict[str, str]) -> bool:
    return any(
        _clean_text(row.get(key, ""))
        for key in ("species", "call_type", "begin_time", "end_time", "low_freq", "high_freq")
    )


def _monthly_sheets(sheets: Sequence[WorkbookSheet]) -> Iterator[WorkbookSheet]:
    for sheet in sheets:
        if sheet.name in {READ_ME_SHEET, INVENTORY_SHEET}:
            continue
        yield sheet


def _inventory_sheet(sheets: Sequence[WorkbookSheet]) -> Optional[WorkbookSheet]:
    for sheet in sheets:
        if sheet.name == INVENTORY_SHEET:
            return sheet
    return None


def _build_annotation_row(sheet_name: str, row_index: int, row: Dict[str, str]) -> Optional[Dict[str, str]]:
    if not _is_annotation_row(row):
        return None

    sheet_defaults = {
        "hydrophone_thuds": {
            "species": INSTRUMENT_SIGNAL_CODE,
            "call_type": "hydrophone_thud",
        },
        "earthquakes": {
            "species": EARTHQUAKE_SIGNAL_CODE,
            "call_type": "earthquake",
        },
        "unknown": {
            "species": UNKNOWN_SIGNAL_CODE,
            "call_type": "unknown",
        },
        "sonar": {
            "species": SONAR_SIGNAL_CODE,
            "call_type": "sonar",
        },
    }
    defaults = sheet_defaults.get(sheet_name, {})
    filename = normalize_audio_filename(row.get("filename", ""))
    species = _normalize_species(
        row.get("species", "")
        or row.get("signal_source", "")
        or defaults.get("species", "")
    )
    call_type_raw = _normalize_call_type_raw(
        row.get("call_type", "")
        or row.get("signal_type", "")
        or defaults.get("call_type", "")
    )
    begin_time_s = _as_float(row.get("begin_time"))
    end_time_s = _as_float(row.get("end_time"))
    low_freq_hz = _as_float(row.get("low_freq"))
    high_freq_hz = _as_float(row.get("high_freq"))
    peak_freq_hz = _as_float(row.get("peak_freq"))
    peak_power = _as_float(row.get("peak_power"))
    comments = _clean_text(row.get("comments", "") or row.get("comment", ""))
    verified_flag = _as_int_flag(row.get("verified", ""))
    vessel_flag = _as_int_flag(
        row.get("vessel_presence", ""),
        row.get("vessel_present", ""),
        row.get("vessel_lloyds", ""),
    )
    granularity = _clean_text(row.get("granularity", ""))

    if not filename:
        return None
    if begin_time_s is None or end_time_s is None or low_freq_hz is None or high_freq_hz is None:
        return None
    if end_time_s <= begin_time_s:
        return None
    if high_freq_hz <= low_freq_hz:
        return None

    return {
        "sheet": sheet_name,
        "row_index": str(row_index),
        "filename": filename,
        "species": species,
        "call_type_raw": call_type_raw,
        "call_type_bucket": bucket_fin_call_type(call_type_raw, species),
        "begin_time_s": "" if begin_time_s is None else f"{begin_time_s:.6f}",
        "end_time_s": "" if end_time_s is None else f"{end_time_s:.6f}",
        "low_freq_hz": "" if low_freq_hz is None else f"{low_freq_hz:.6f}",
        "high_freq_hz": "" if high_freq_hz is None else f"{high_freq_hz:.6f}",
        "peak_freq_hz": "" if peak_freq_hz is None else f"{peak_freq_hz:.6f}",
        "peak_power": "" if peak_power is None else f"{peak_power:.6f}",
        "comments": comments,
        "verified_flag": str(verified_flag),
        "vessel_flag": str(vessel_flag),
        "granularity": granularity,
        "context_tags": "",
    }


def _clip_inventory_rows(sheet: WorkbookSheet) -> List[Dict[str, str]]:
    seen = set()
    rows: List[Dict[str, str]] = []
    for row in sheet.rows:
        filename = normalize_audio_filename(row.get("filename", ""))
        if not filename or filename in seen:
            continue
        seen.add(filename)
        rows.append({"filename": filename})
    return rows


def _fallback_inventory_rows(*row_groups: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    rows: List[Dict[str, str]] = []
    for row_group in row_groups:
        for row in row_group:
            filename = normalize_audio_filename(row.get("filename", ""))
            if not filename or filename in seen:
                continue
            seen.add(filename)
            rows.append({"filename": filename})
    rows.sort(key=lambda row: row["filename"])
    return rows


def _load_optional_inventory_rows(workbook_path: Optional[Path | str]) -> Tuple[List[Dict[str, str]], str]:
    if workbook_path is None:
        return [], ""
    path = Path(workbook_path)
    if not path.exists():
        return [], ""
    sheets = load_workbook_sheets(path)
    inventory_sheet = _inventory_sheet(sheets)
    if inventory_sheet is None:
        return [], ""
    return _clip_inventory_rows(inventory_sheet), str(path.resolve())


def _device_code(filename: str) -> str:
    text = _clean_text(filename)
    return text.split("_", 1)[0] if "_" in text else ""


def _clip_start_iso(filename: str) -> str:
    timestamp = parse_filename_timestamp(filename)
    return timestamp.isoformat() if timestamp is not None else ""


def _recording_day(filename: str) -> str:
    timestamp = parse_filename_timestamp(filename)
    return timestamp.strftime("%Y-%m-%d") if timestamp is not None else ""


def _clip_manifest_rows(
    annotation_rows: Sequence[Dict[str, str]],
    *,
    inventory_source: str,
    source_workbook: str,
) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in annotation_rows:
        grouped[row["filename"]].append(row)

    clip_rows: List[Dict[str, str]] = []
    for filename in sorted(grouped):
        rows = grouped[filename]
        species_codes = sorted({row["species"] for row in rows if row.get("species")})
        fin_rows = [row for row in rows if row.get("species") == FIN_SPECIES_CODE]
        non_fin_rows = [row for row in rows if row.get("species") and row.get("species") != FIN_SPECIES_CODE]
        clip_tags = set()
        for row in rows:
            clip_tags.update(filter(None, row.get("context_tags", "").split("|")))
        clip_rows.append(
            {
                "source_dataset": ANNOTATIONS_2025_DATASET,
                "inventory_source": inventory_source,
                "filename": filename,
                "device_code": _device_code(filename),
                "clip_start_utc": _clip_start_iso(filename),
                "recording_day_utc": _recording_day(filename),
                "is_fin_positive": "1" if fin_rows else "0",
                "is_annotated_non_fin": "1" if non_fin_rows else "0",
                "is_pure_negative_candidate": "0",
                "annotation_count": str(len(rows)),
                "fin_annotation_count": str(len(fin_rows)),
                "non_fin_annotation_count": str(len(non_fin_rows)),
                "species_codes": "|".join(species_codes),
                "fin_call_type_buckets": "|".join(sorted({row["call_type_bucket"] for row in fin_rows if row.get("call_type_bucket")})),
                "fin_call_type_raws": "|".join(sorted({row["call_type_raw"] for row in fin_rows if row.get("call_type_raw")})),
                "context_tags": "|".join(sorted(clip_tags)) if clip_tags else UNKNOWN_CONTEXT,
                "source_workbooks": source_workbook,
                "verified_flag": str(max(int(row.get("verified_flag") or 0) for row in rows)),
            }
        )
    return clip_rows


def _collect_mar18_guardrail_filenames(
    workbook_path: Optional[Path | str],
) -> Tuple[set[str], Dict[str, object]]:
    summary: Dict[str, object] = {"kept_row_count": 0, "sheet_counts": {}}
    if workbook_path is None:
        return set(), summary
    path = Path(workbook_path)
    if not path.exists():
        return set(), summary

    filenames: set[str] = set()
    sheets = load_workbook_sheets(path)
    for sheet in _monthly_sheets(sheets):
        kept = 0
        for row in sheet.rows:
            filename = normalize_audio_filename(row.get("filename", ""))
            species = _normalize_species(row.get("species", ""))
            begin_s = _as_float(row.get("begin_time"))
            end_s = _as_float(row.get("end_time"))
            low_hz = _as_float(row.get("low_freq"))
            high_hz = _as_float(row.get("high_freq"))
            if not filename or not species or species.lower() == "nan":
                continue
            if begin_s is None or end_s is None or low_hz is None or high_hz is None:
                continue
            if end_s <= begin_s or high_hz <= low_hz:
                continue
            filenames.add(filename)
            kept += 1
        summary["sheet_counts"][sheet.name] = int(kept)
        summary["kept_row_count"] = int(summary["kept_row_count"]) + int(kept)
    return filenames, summary


def _collect_mar26_pure_negative_rows(
    workbook_path: Optional[Path | str],
    *,
    exclude_filenames: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    drop_reasons: Counter[str] = Counter()
    summary: Dict[str, object] = {
        "verified_rows": 0,
        "pure_negative_rows": 0,
        "drop_reasons": drop_reasons,
        "sheet_counts": {},
    }
    if workbook_path is None:
        summary["drop_reasons"] = {}
        return [], summary
    path = Path(workbook_path)
    if not path.exists():
        summary["drop_reasons"] = {}
        return [], summary

    exclude = set(exclude_filenames or set())
    rows: List[Dict[str, str]] = []
    seen: set[str] = set()
    sheets = load_workbook_sheets(path)
    for sheet in _monthly_sheets(sheets):
        kept = 0
        for row in sheet.rows:
            filename = normalize_audio_filename(row.get("filename", ""))
            if not filename:
                continue
            if parse_filename_timestamp(filename) is None:
                drop_reasons["invalid_filename"] += 1
                continue
            verified = _truthy_flag(row.get("verified", ""))
            if verified != 1:
                continue
            summary["verified_rows"] = int(summary["verified_rows"]) + 1
            if filename in exclude:
                drop_reasons["excluded_by_annotation_guardrail"] += 1
                continue
            if filename in seen:
                drop_reasons["duplicate_filename"] += 1
                continue
            any_species_flag = any(
                _truthy_flag(row.get(column_name, ""))
                for column_name in PURE_NEGATIVE_FLAG_COLUMNS
                if column_name in row
            )
            if any_species_flag:
                drop_reasons["species_flag_present"] += 1
                continue

            rows.append(
                {
                    "source_dataset": PURE_NEGATIVE_DATASET,
                    "inventory_source": str(path.resolve()),
                    "filename": filename,
                    "device_code": _device_code(filename),
                    "clip_start_utc": _clip_start_iso(filename),
                    "recording_day_utc": _recording_day(filename),
                    "is_fin_positive": "0",
                    "is_annotated_non_fin": "0",
                    "is_pure_negative_candidate": "1",
                    "annotation_count": "0",
                    "fin_annotation_count": "0",
                    "non_fin_annotation_count": "0",
                    "species_codes": "",
                    "fin_call_type_buckets": "",
                    "fin_call_type_raws": "",
                    "context_tags": "pure_negative",
                    "source_workbooks": str(path.resolve()),
                    "verified_flag": "1",
                }
            )
            seen.add(filename)
            kept += 1
        summary["sheet_counts"][sheet.name] = int(kept)
        summary["pure_negative_rows"] = int(summary["pure_negative_rows"]) + int(kept)
    summary["drop_reasons"] = dict(drop_reasons)
    return rows, summary


def _smoke_subset_rows(
    clip_rows: Sequence[Dict[str, str]],
    smoke_per_bucket: int,
    smoke_non_fin: int,
    seed: int,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in clip_rows:
        if row["is_fin_positive"] == "1":
            buckets = row["fin_call_type_buckets"].split("|") if row["fin_call_type_buckets"] else [FIN_BUCKET_OTHER]
            for bucket in filter(None, buckets):
                grouped[bucket].append(row)
        elif row["is_annotated_non_fin"] == "1":
            grouped["annotated_non_fin"].append(row)

    selected: Dict[str, Dict[str, str]] = {}
    for bucket, limit in (
        (FIN_BUCKET_20, smoke_per_bucket),
        (FIN_BUCKET_40, smoke_per_bucket),
        (FIN_BUCKET_OTHER, smoke_per_bucket),
        ("annotated_non_fin", smoke_non_fin),
    ):
        candidates = list(grouped.get(bucket, []))
        rng.shuffle(candidates)
        for row in candidates[:limit]:
            selected[row["filename"]] = row
    return [selected[name] for name in sorted(selected)]


def adjacent_clip_filename(
    filename: str,
    *,
    clip_delta: int,
    clip_minutes: float = 5.0,
) -> Optional[str]:
    """Return the adjacent 5-minute clip filename while preserving prefix/ext."""
    if clip_delta == 0:
        return _clean_text(filename)
    source_name = _clean_text(filename)
    match = _FILENAME_TS_RE.search(source_name)
    source_ts = parse_filename_timestamp(source_name)
    if match is None or source_ts is None:
        return None

    shifted = source_ts + timedelta(minutes=float(clip_minutes) * int(clip_delta))
    if match.group(2) is None:
        replacement = shifted.strftime("%Y%m%dT%H%M%SZ")
    else:
        replacement = (
            f"{shifted.strftime('%Y%m%dT%H%M%S')}"
            f".{shifted.microsecond // 1000:03d}Z"
        )
    start, end = match.span()
    return f"{source_name[:start]}{replacement}{source_name[end:]}"


def _boundary_context_rows(
    annotation_rows: Sequence[Dict[str, str]],
    clip_row_by_name: Dict[str, Dict[str, str]],
    *,
    adjacent_boundary_seconds: float,
    clip_duration_s: float,
) -> List[Dict[str, str]]:
    if adjacent_boundary_seconds <= 0 or clip_duration_s <= 0:
        return []

    aggregated: Dict[str, Dict[str, object]] = {}
    candidate_names = {
        name
        for name, row in clip_row_by_name.items()
        if row.get("is_fin_positive") == "1" or row.get("is_annotated_non_fin") == "1"
    }

    for row in annotation_rows:
        source_filename = row.get("filename", "")
        if source_filename not in candidate_names:
            continue

        begin_time_s = _as_float(row.get("begin_time_s"))
        end_time_s = _as_float(row.get("end_time_s"))
        boundary_hits: List[Tuple[str, str, float]] = []
        if begin_time_s is not None and begin_time_s <= adjacent_boundary_seconds:
            prev_name = adjacent_clip_filename(source_filename, clip_delta=-1)
            if prev_name:
                boundary_hits.append((prev_name, "previous", begin_time_s))
        if end_time_s is not None and end_time_s >= clip_duration_s - adjacent_boundary_seconds:
            next_name = adjacent_clip_filename(source_filename, clip_delta=1)
            if next_name:
                boundary_hits.append((next_name, "next", max(0.0, clip_duration_s - end_time_s)))

        for adjacent_name, boundary_side, distance_s in boundary_hits:
            info = aggregated.setdefault(
                adjacent_name,
                {
                    "filename": adjacent_name,
                    "boundary_sides": set(),
                    "source_filenames": set(),
                    "trigger_annotation_count": 0,
                    "min_distance_to_edge_s": None,
                },
            )
            info["boundary_sides"].add(boundary_side)
            info["source_filenames"].add(source_filename)
            info["trigger_annotation_count"] = int(info["trigger_annotation_count"]) + 1
            prev_distance = info["min_distance_to_edge_s"]
            if prev_distance is None or distance_s < float(prev_distance):
                info["min_distance_to_edge_s"] = distance_s

    rows: List[Dict[str, str]] = []
    for filename in sorted(aggregated):
        info = aggregated[filename]
        min_distance = info.get("min_distance_to_edge_s")
        rows.append(
            {
                "filename": filename,
                "boundary_sides": "|".join(sorted(info["boundary_sides"])),
                "source_filenames": "|".join(sorted(info["source_filenames"])),
                "trigger_annotation_count": str(int(info["trigger_annotation_count"])),
                "min_distance_to_edge_s": (
                    "" if min_distance is None else f"{float(min_distance):.6f}"
                ),
            }
        )
    return rows


def _merge_named_rows(
    primary_rows: Sequence[Dict[str, str]],
    secondary_rows: Sequence[Dict[str, str]],
    *,
    primary_role: str,
    secondary_role: str,
) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    roles_by_name: Dict[str, set[str]] = defaultdict(set)

    for row in primary_rows:
        filename = row.get("filename", "")
        if not filename:
            continue
        merged[filename] = dict(row)
        roles_by_name[filename].add(primary_role)

    for row in secondary_rows:
        filename = row.get("filename", "")
        if not filename:
            continue
        current = merged.setdefault(filename, {})
        for key, value in row.items():
            if key == "filename":
                current[key] = value
                continue
            if key not in current or not str(current.get(key, "")).strip():
                current[key] = value
        roles_by_name[filename].add(secondary_role)

    rows: List[Dict[str, str]] = []
    for filename in sorted(merged):
        row = dict(merged[filename])
        row["roles"] = "|".join(sorted(roles_by_name[filename]))
        rows.append(row)
    return rows


def build_part2_manifests(
    workbook_path: Path | str,
    *,
    mar26_workbook: Optional[Path | str] = MAR26_WORKBOOK_DEFAULT,
    mar18_workbook: Optional[Path | str] = MAR18_WORKBOOK_DEFAULT,
    smoke_per_bucket: int = 6,
    smoke_non_fin: int = 6,
    adjacent_boundary_seconds: float = DEFAULT_ADJACENT_BOUNDARY_SECONDS,
    include_adjacent_in_prep: bool = False,
    clip_duration_s: float = DEFAULT_CLIP_DURATION_S,
    seed: int = 1337,
) -> Dict[str, object]:
    sheets = load_workbook_sheets(workbook_path)
    inventory_sheet = _inventory_sheet(sheets)

    annotation_rows: List[Dict[str, str]] = []
    clip_species: Dict[str, List[str]] = defaultdict(list)
    sheet_counts: Counter[str] = Counter()

    for sheet in _monthly_sheets(sheets):
        for row_index, row in enumerate(sheet.rows, start=2):
            out_row = _build_annotation_row(sheet.name, row_index, row)
            if out_row is None:
                continue
            annotation_rows.append(out_row)
            sheet_counts[sheet.name] += 1
            if out_row.get("species"):
                clip_species[out_row["filename"]].append(out_row["species"])

    for row in annotation_rows:
        tags = infer_context_tags(
            comments=row.get("comments", ""),
            vessel_flag=int(row.get("vessel_flag") or 0),
            species_code=row.get("species", ""),
            clip_species_codes=clip_species.get(row["filename"], []),
        )
        row["context_tags"] = "|".join(tags)

    annotation_rows.sort(
        key=lambda row: (
            row["filename"],
            float(row["begin_time_s"] or 0.0),
            row["sheet"],
            int(row["row_index"]),
        )
    )

    workbook_source = str(Path(workbook_path).resolve())
    guardrail_filenames, mar18_summary = _collect_mar18_guardrail_filenames(mar18_workbook)
    pure_negative_rows, pure_negative_summary = _collect_mar26_pure_negative_rows(
        mar26_workbook,
        exclude_filenames={row["filename"] for row in annotation_rows} | guardrail_filenames,
    )

    inventory_rows: List[Dict[str, str]]
    inventory_source: str
    if inventory_sheet is not None:
        inventory_rows = _clip_inventory_rows(inventory_sheet)
        inventory_source = workbook_source
    else:
        inventory_rows, inventory_source = _load_optional_inventory_rows(mar26_workbook)
        if not inventory_rows:
            inventory_rows, inventory_source = _load_optional_inventory_rows(mar18_workbook)
        if not inventory_rows:
            inventory_rows = _fallback_inventory_rows(annotation_rows, pure_negative_rows)
            inventory_source = "annotation_union"

    annotation_clip_rows = _clip_manifest_rows(
        annotation_rows,
        inventory_source=inventory_source,
        source_workbook=workbook_source,
    )
    clip_rows = sorted(annotation_clip_rows + pure_negative_rows, key=lambda row: row["filename"])
    clip_row_by_name = {row["filename"]: row for row in clip_rows}

    fin_rows = [row for row in annotation_rows if row.get("species") == FIN_SPECIES_CODE]
    fin_positive_rows = [row for row in annotation_clip_rows if row["is_fin_positive"] == "1"]
    annotated_non_fin_rows = [
        row for row in annotation_clip_rows if row["is_fin_positive"] == "0" and row["is_annotated_non_fin"] == "1"
    ]
    candidate_rows = sorted(
        fin_positive_rows + annotated_non_fin_rows,
        key=lambda row: row["filename"],
    )
    adjacent_context_rows = _boundary_context_rows(
        annotation_rows=annotation_rows,
        clip_row_by_name=clip_row_by_name,
        adjacent_boundary_seconds=float(adjacent_boundary_seconds),
        clip_duration_s=float(clip_duration_s),
    )
    download_base_rows = _merge_named_rows(
        candidate_rows,
        pure_negative_rows,
        primary_role="candidate",
        secondary_role="pure_negative_candidate",
    )
    download_rows = _merge_named_rows(
        download_base_rows,
        adjacent_context_rows,
        primary_role="selected_training_audio",
        secondary_role="adjacent_context",
    )
    prep_rows = _merge_named_rows(
        candidate_rows,
        adjacent_context_rows if include_adjacent_in_prep else [],
        primary_role="candidate",
        secondary_role="adjacent_context",
    )
    smoke_rows = _smoke_subset_rows(
        clip_rows=clip_rows,
        smoke_per_bucket=smoke_per_bucket,
        smoke_non_fin=smoke_non_fin,
        seed=seed,
    )

    summary = {
        "workbook_path": str(Path(workbook_path)),
        "sheet_count": len(sheets),
        "inventory_clip_count": len(inventory_rows),
        "inventory_source": inventory_source,
        "annotated_row_count": len(annotation_rows),
        "fin_annotation_count": len(fin_rows),
        "annotated_clip_count": len(annotation_clip_rows),
        "clip_manifest_clip_count": len(clip_rows),
        "fin_positive_clip_count": len(fin_positive_rows),
        "annotated_non_fin_clip_count": len(annotated_non_fin_rows),
        "pure_negative_clip_count": len(pure_negative_rows),
        "candidate_clip_count": len(candidate_rows),
        "adjacent_context_clip_count": len(adjacent_context_rows),
        "download_clip_count": len(download_rows),
        "prep_clip_count": len(prep_rows),
        "smoke_clip_count": len(smoke_rows),
        "adjacent_boundary_seconds": float(adjacent_boundary_seconds),
        "include_adjacent_in_prep": bool(include_adjacent_in_prep),
        "clip_duration_s": float(clip_duration_s),
        "sheet_counts": dict(sheet_counts),
        "species_counts": dict(Counter(row["species"] for row in annotation_rows if row.get("species"))),
        "fin_call_type_counts": dict(Counter(row["call_type_bucket"] for row in fin_rows if row.get("call_type_bucket"))),
        "mar18_guardrail_summary": mar18_summary,
        "mar26_pure_negative_summary": pure_negative_summary,
    }

    return {
        "summary": summary,
        "clip_inventory": inventory_rows,
        "annotations_all": annotation_rows,
        "fin_annotations": fin_rows,
        "clip_manifest": clip_rows,
        "fin_positive_clips": fin_positive_rows,
        "annotated_non_fin_clips": annotated_non_fin_rows,
        "pure_negative_clips": pure_negative_rows,
        "candidate_clips": candidate_rows,
        "adjacent_context_clips": adjacent_context_rows,
        "download_clips": download_rows,
        "prep_clips": prep_rows,
        "smoke_clips": smoke_rows,
        "clip_manifest_by_name": clip_row_by_name,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def write_part2_manifests(
    output_dir: Path | str,
    manifests: Dict[str, object],
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    csv_keys = [
        "clip_inventory",
        "annotations_all",
        "fin_annotations",
        "clip_manifest",
        "fin_positive_clips",
        "annotated_non_fin_clips",
        "pure_negative_clips",
        "candidate_clips",
        "adjacent_context_clips",
        "download_clips",
        "prep_clips",
        "smoke_clips",
    ]
    for key in csv_keys:
        rows = manifests.get(key, [])
        path = out_dir / f"{key}.csv"
        _write_csv(path, rows if isinstance(rows, list) else [])
        written[key] = path

    txt_keys = [
        "fin_positive_clips",
        "annotated_non_fin_clips",
        "pure_negative_clips",
        "candidate_clips",
        "adjacent_context_clips",
        "download_clips",
        "prep_clips",
        "smoke_clips",
    ]
    for key in txt_keys:
        rows = manifests.get(key, [])
        lines = [row["filename"] for row in rows if isinstance(row, dict) and row.get("filename")]
        path = out_dir / f"{key}.txt"
        _write_text_lines(path, lines)
        written[f"{key}_txt"] = path

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(manifests.get("summary", {}), handle, indent=2, sort_keys=True)
    written["summary"] = summary_path
    return written


def parse_window_mat_stem(stem: str) -> Optional[Tuple[str, float, float]]:
    match = _FILE_WINDOW_RE.match(stem)
    if not match:
        return None
    return (
        match.group("source"),
        float(match.group("start")),
        float(match.group("end")),
    )


def parse_filename_timestamp(filename: str) -> Optional[datetime]:
    match = _FILENAME_TS_RE.search(str(filename))
    if not match:
        return None
    base = match.group(1)
    millis = match.group(2)
    try:
        if millis is None:
            return datetime.strptime(base, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return datetime.strptime(f"{base}.{millis}", "%Y%m%dT%H%M%S.%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
