"""Multi-label acoustic dataset helpers.

This module is intentionally small and additive. It reuses the MAT loading and
normalization conventions from the existing fin-whale ResNet pipeline, but
represents labels as independent binary targets instead of a two-class softmax
target.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback base so label-only utilities can run without PyTorch."""

        pass

try:
    import scipy.io as sio
except Exception:
    sio = None

from src.dataset.part2_annotations import parse_filename_timestamp
try:
    from src.training.mat_dataset import (
        DB_KEYS,
        FREQ_KEYS,
        POWER_KEYS,
        SPECTRO_KEYS,
        TIME_KEYS,
        _choose_start_idx,
        _find_key,
        _infer_time_bin_seconds,
        _normalize_db_to_unit,
        _power_to_db_norm,
        parse_crop_size,
    )
except Exception:
    DB_KEYS: Tuple[str, ...] = ()
    FREQ_KEYS: Tuple[str, ...] = ()
    POWER_KEYS: Tuple[str, ...] = ()
    SPECTRO_KEYS: Tuple[str, ...] = ()
    TIME_KEYS: Tuple[str, ...] = ()

    def _missing_training_dependency(*_: Any, **__: Any) -> Any:
        raise RuntimeError("PyTorch training dependencies are required for MAT dataset loading")

    _choose_start_idx = _missing_training_dependency
    _find_key = _missing_training_dependency
    _infer_time_bin_seconds = _missing_training_dependency
    _normalize_db_to_unit = _missing_training_dependency
    _power_to_db_norm = _missing_training_dependency

    def parse_crop_size(crop_size: Any) -> Tuple[Optional[int], Optional[int]]:
        _missing_training_dependency(crop_size)


NONBIOLOGICAL_SPECIES_CODES = frozenset({"INSTRUMENT", "EQ", "SONAR", "UNKNOWN"})
TRAINABLE_CALL_TYPES = frozenset(
    {
        "20Hz",
        "30Hz",
        "40Hz",
        "song",
        "other_fin",
        "BmA",
        "BmB",
        "BmZ",
        "BmD",
        "Bp20",
        "Bp20plus",
        "BpD",
        "fin_20hz",
        "fin_20hz_plus",
        "fin_30hz",
        "fin_40hz",
        "fin_downsweep",
        "fin_song",
        "fin_other",
        "blue_A",
        "blue_B",
        "blue_Z",
        "blue_D",
        "humpback_song",
        "orca_call",
        "orca_click",
        "orca_whistle",
    }
)

PRIMARY_SPECIES_LABEL_IDS = ("species:Bm", "species:Bp", "species:Mn", "species:Oo")
TRUTHY_VALUES = frozenset({"1", "true", "t", "yes", "y"})

SPECIES_CODE_TO_NAME = {
    "Bp": "Fin whale",
    "Bm": "Blue whale",
    "Mn": "Humpback whale",
    "Bb": "Sei whale",
    "Oo": "Killer whale",
    "Pm": "Sperm whale",
    "OD": "Odontocete",
    "CE": "Cetacean",
    "UN": "Unknown cetacean",
    "P": "Porpoise",
    "Lo": "Pacific white-sided dolphin",
    "INSTRUMENT": "Instrument sound",
    "EQ": "Earthquake",
    "SONAR": "Sonar",
    "UNKNOWN": "Unknown sound",
}

SPECIES_CODE_TO_CLASS_PATH = {
    "Bp": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
    "Bm": "Biophony > Marine mammal > Cetacean > Baleen whale > Blue whale",
    "Mn": "Biophony > Marine mammal > Cetacean > Baleen whale > Humpback whale",
    "Bb": "Biophony > Marine mammal > Cetacean > Baleen whale > Sei whale",
    "Oo": "Biophony > Marine mammal > Cetacean > Toothed whale > Killer whale",
    "Pm": "Biophony > Marine mammal > Cetacean > Toothed whale > Sperm whale",
    "OD": "Biophony > Marine mammal > Cetacean > Toothed whale",
    "CE": "Biophony > Marine mammal > Cetacean",
    "UN": "Biophony > Marine mammal > Cetacean",
    "P": "Biophony > Marine mammal > Cetacean > Toothed whale > Porpoise",
    "Lo": "Biophony > Marine mammal > Cetacean > Toothed whale > Dolphin > Pacific white-sided dolphin",
    "INSTRUMENT": "Instrumentation",
    "EQ": "Geophony > Geology > Earthquake",
    "SONAR": "Anthropophony > Sonar",
    "UNKNOWN": "Other > Unknown sound of interest",
}

CALL_TYPE_TO_NAME = {
    "20Hz": "20 Hz pulse",
    "30Hz": "30 Hz call",
    "40Hz": "40 Hz call",
    "song": "Song unit",
    "other_fin": "Other fin-whale call",
    "BmA": "Antarctic blue whale A-call",
    "BmB": "Antarctic blue whale B-call",
    "BmZ": "Antarctic blue whale Z-call",
    "BmD": "Antarctic blue whale D-call",
    "Bp20": "Fin whale 20 Hz pulse",
    "Bp20plus": "Fin whale 20 Hz pulse with overtone",
    "BpD": "Fin whale 40 Hz downsweep",
    "fin_20hz": "Fin whale 20 Hz pulse",
    "fin_20hz_plus": "Fin whale 20 Hz pulse plus",
    "fin_30hz": "Fin whale 30 Hz call",
    "fin_40hz": "Fin whale 40 Hz call",
    "fin_downsweep": "Fin whale downsweep",
    "fin_song": "Fin whale song unit",
    "fin_other": "Other fin-whale call",
    "blue_A": "Blue whale A-call",
    "blue_B": "Blue whale B-call",
    "blue_Z": "Blue whale Z-call",
    "blue_D": "Blue whale D-call",
    "humpback_song": "Humpback whale song",
    "orca_call": "Killer whale call",
    "orca_click": "Killer whale click",
    "orca_whistle": "Killer whale whistle",
}

_WHITESPACE_RE = re.compile(r"\s+")


def split_pipe(value: Any) -> Tuple[str, ...]:
    """Split a pipe-delimited manifest field into unique non-empty tokens."""
    if value is None:
        return ()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ()
    tokens = [token.strip() for token in text.split("|") if token.strip()]
    return tuple(dict.fromkeys(tokens))


def clean_text(value: Any) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value or "").strip())
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_species_code(value: Any) -> str:
    """Normalize known species/noise aliases into the code vocabulary."""
    text = clean_text(value)
    if not text:
        return ""
    aliases = {
        "bp": "Bp",
        "fin whale": "Bp",
        "bm": "Bm",
        "blue whale": "Bm",
        "mn": "Mn",
        "humpback whale": "Mn",
        "bb": "Bb",
        "sei whale": "Bb",
        "od": "OD",
        "odontocete": "OD",
        "oo": "Oo",
        "killer whale": "Oo",
        "pm": "Pm",
        "sperm whale": "Pm",
        "ce": "CE",
        "un": "UN",
        "unknown": "UNKNOWN",
        "instrument": "INSTRUMENT",
        "hydrophone_thud": "INSTRUMENT",
        "hydrophone thud": "INSTRUMENT",
        "eq": "EQ",
        "earthquake": "EQ",
        "sonar": "SONAR",
    }
    return aliases.get(text.lower(), text)


def normalize_call_type(value: Any, species_code: str = "") -> str:
    """Normalize common call-type spellings while preserving unknown raw codes."""
    text = clean_text(value)
    if not text:
        return ""
    compact = re.sub(r"[\s_-]+", "", text.lower())
    if compact in {"20hz", "20hzcall", "20hzcalls"}:
        return "20Hz"
    if compact in {"30hz", "30hzcall", "30hzcalls", "30hz+echo", "30hzwith echo", "30hzwithEcho".lower()}:
        return "30Hz"
    if compact in {"40hz", "40hzcall", "40hzcalls"}:
        return "40Hz"
    lowered = text.lower()
    if compact == "s" or "song" in lowered or "twin note" in lowered or "ab song" in lowered:
        return "song"
    if species_code == "Bp" and compact in {"otherfin", "other"}:
        return "other_fin"
    biodcase_aliases = {
        "bma": "BmA",
        "bmb": "BmB",
        "bmz": "BmZ",
        "bmd": "BmD",
        "bp20": "Bp20",
        "bp20plus": "Bp20plus",
        "bp20p": "Bp20plus",
        "bpd": "BpD",
    }
    if compact in biodcase_aliases:
        return biodcase_aliases[compact]
    return text


def species_display_name(code: str) -> str:
    return SPECIES_CODE_TO_NAME.get(code, f"Species code {code}")


def species_class_path(code: str) -> str:
    return SPECIES_CODE_TO_CLASS_PATH.get(code, f"Bioacoustic label > Species code > {code}")


def call_type_display_name(code: str) -> str:
    return CALL_TYPE_TO_NAME.get(code, code)


def call_type_class_path(code: str) -> str:
    return f"Bioacoustic call type > {call_type_display_name(code)}"


def label_id(group: str, code: str) -> str:
    return f"{group}:{code}"


def canonicalize_source_label_ids(
    source_label_ids: Iterable[str],
    *,
    include_species: bool = True,
    include_call_types: bool = True,
    primary_species: Sequence[str] = ("Bm", "Bp", "Mn", "Oo"),
) -> Tuple[List[str], List[str]]:
    """Map source-specific label ids into the weekend canonical ontology.

    Returns `(trainable_ids, analysis_ids)`. Analysis ids preserve useful
    non-trainable concepts such as broad odontocete labels without making them
    positive training targets.
    """
    source_ids = [clean_text(label) for label in source_label_ids if clean_text(label)]
    species_context = {label.partition(":")[2] for label in source_ids if label.startswith("species:")}
    primary = set(primary_species)
    canonical: List[str] = []
    analysis: List[str] = []

    def add_trainable(value: str) -> None:
        if value and value not in canonical:
            canonical.append(value)

    def add_analysis(value: str) -> None:
        if value and value not in analysis:
            analysis.append(value)

    for raw_id in source_ids:
        group, sep, code = raw_id.partition(":")
        if not sep:
            continue
        if group == "species":
            norm = normalize_species_code(code)
            if include_species and norm in primary:
                add_trainable(f"species:{norm}")
            elif norm == "OD":
                add_analysis("group:odontocete_unknown")
            elif norm in {"CE", "UN", "P"}:
                add_analysis(f"group:{norm}")
            continue

        if group != "call" or not include_call_types:
            continue
        norm_call = normalize_call_type(code)
        mapping = {
            "20Hz": "call:fin_20hz",
            "30Hz": "call:fin_30hz",
            "40Hz": "call:fin_40hz",
            "other_fin": "call:fin_other",
            "Bp20": "call:fin_20hz",
            "Bp20plus": "call:fin_20hz_plus",
            "BpD": "call:fin_downsweep",
            "BmA": "call:blue_A",
            "BmB": "call:blue_B",
            "BmZ": "call:blue_Z",
            "BmD": "call:blue_D",
        }
        mapped = mapping.get(norm_call)
        if mapped is None and norm_call == "song":
            mapped = "call:fin_song" if "Bp" in species_context else "call:humpback_song"
        if mapped is None and norm_call.lower() in {"ck", "call", "orca_call"}:
            mapped = "call:orca_call"
        if mapped is None and norm_call.lower() in {"click", "clicks", "echolocation"}:
            mapped = "call:orca_click"
        if mapped is None and norm_call.lower() in {"whistle", "whistles"}:
            mapped = "call:orca_whistle"
        if mapped:
            add_trainable(mapped)

    return sorted(canonical), sorted(analysis)


def parse_labels_json(value: Any) -> List[Dict[str, Any]]:
    text = clean_text(value)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def label_ids_from_row(row: Dict[str, Any]) -> List[str]:
    """Read label ids from a manifest row.

    Supports the candidate manifest emitted by `audit_labels.py`, plus common
    existing manifest fields for tests and small local conversions.
    """
    if clean_text(row.get("canonical_schema_version")):
        explicit_canonical = split_pipe(row.get("label_ids") or row.get("canonical_label_ids"))
        return list(explicit_canonical)

    explicit = split_pipe(row.get("label_ids") or row.get("labels"))
    if explicit:
        return list(explicit)

    labels = parse_labels_json(row.get("labels_json"))
    ids: List[str] = []
    for label in labels:
        if not isinstance(label, dict):
            continue
        if label.get("trainable") is False:
            continue
        species = normalize_species_code(label.get("species_code") or label.get("species"))
        call_type = normalize_call_type(label.get("call_type") or label.get("call_type_std"), species)
        if species and species not in NONBIOLOGICAL_SPECIES_CODES:
            ids.append(label_id("species", species))
        if call_type and call_type in TRAINABLE_CALL_TYPES:
            ids.append(label_id("call", call_type))
    if ids:
        return sorted(dict.fromkeys(ids))

    species_codes = split_pipe(row.get("species_codes") or row.get("species_code") or row.get("species"))
    call_types = split_pipe(
        row.get("call_type_stds")
        or row.get("fin_call_type_stds")
        or row.get("call_type_std")
        or row.get("fin_call_type_buckets")
        or row.get("call_type_bucket")
        or row.get("call_type_raw")
    )
    for raw_species in species_codes:
        code = normalize_species_code(raw_species)
        if code and code not in NONBIOLOGICAL_SPECIES_CODES:
            ids.append(label_id("species", code))
    for raw_call in call_types:
        norm = normalize_call_type(raw_call)
        if norm and norm in TRAINABLE_CALL_TYPES:
            ids.append(label_id("call", norm))
    return sorted(dict.fromkeys(ids))


def evaluation_bucket_from_row(
    row: Dict[str, Any],
    *,
    primary_species_label_ids: Sequence[str] = PRIMARY_SPECIES_LABEL_IDS,
) -> str:
    """Classify empty-target rows for deployment metrics.

    A row with no primary `Bm/Bp/Mn/Oo` label is not always true background.
    Species-only manifests can intentionally demote OD/unknown labels and call
    labels out of the trainable target vector. Those rows should be analyzed as
    non-primary biological/confounder signal rather than counted as silent
    background.
    """

    primary = set(primary_species_label_ids)
    target_ids = set(split_pipe(row.get("target_label_ids")))
    canonical_ids = set(split_pipe(row.get("canonical_label_ids") or row.get("label_ids")))
    source_ids = set(split_pipe(row.get("source_label_ids")))
    analysis_ids = set(split_pipe(row.get("analysis_label_ids")))
    known_ids = target_ids | canonical_ids | source_ids | analysis_ids
    review = clean_text(row.get("review_status")).lower()
    context_tags = {tag.lower() for tag in split_pipe(row.get("context_tags"))}

    if (target_ids | canonical_ids | source_ids) & primary:
        return "primary_species_positive"
    if analysis_ids:
        return "demoted_nonprimary_signal"
    if any(label.startswith("species:") and label not in primary for label in known_ids):
        return "known_nonprimary_signal"
    if any(label.startswith("call:") for label in known_ids):
        return "known_call_signal_no_primary"
    if (
        "candidate" in review
        or "pure_negative" in review
        or "pure_negative" in context_tags
        or "pure-negative" in context_tags
    ):
        return "candidate_background"
    if clean_text(row.get("is_background")).lower() in TRUTHY_VALUES:
        if review in {"reviewed_background", "reviewed background", "reviewed_negative", "reviewed negative"}:
            return "reviewed_background"
        return "candidate_background"
    return "empty_unverified"


def annotation_species_code(row: Dict[str, Any]) -> str:
    return normalize_species_code(row.get("species_code") or row.get("species"))


def annotation_call_type(row: Dict[str, Any]) -> str:
    species = annotation_species_code(row)
    raw_call = normalize_call_type(row.get("call_type_raw"), species)
    if raw_call == "30Hz":
        return raw_call
    return normalize_call_type(row.get("call_type_std") or row.get("call_type_bucket") or row.get("call_type_raw"), species)


def annotation_filename(row: Dict[str, Any]) -> str:
    return clean_text(row.get("filename") or row.get("source_audio"))


def annotation_device(row: Dict[str, Any]) -> str:
    device = clean_text(row.get("device_code"))
    if device:
        return device
    filename = annotation_filename(row)
    return filename.split("_", 1)[0] if "_" in filename else ""


def annotation_month(row: Dict[str, Any]) -> str:
    day = clean_text(row.get("recording_day_utc"))
    if len(day) >= 7:
        return day[:7]
    timestamp = clean_text(row.get("clip_start_utc") or row.get("start_time"))
    if len(timestamp) >= 7:
        return timestamp[:7]
    parsed = parse_filename_timestamp(annotation_filename(row))
    return parsed.strftime("%Y-%m") if parsed is not None else ""


def annotation_year(row: Dict[str, Any]) -> str:
    month = annotation_month(row)
    return month[:4] if len(month) >= 4 else ""


def review_status(row: Dict[str, Any]) -> str:
    raw = clean_text(row.get("review_status"))
    if raw:
        return raw
    flag = clean_text(row.get("verified_flag"))
    return "reviewed" if flag in {"1", "true", "True", "yes"} else "unreviewed"


def read_csv_rows(path: Path | str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path | str, rows: Sequence[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True)
class LabelVocabulary:
    labels: Tuple[Dict[str, Any], ...]

    @property
    def label_ids(self) -> Tuple[str, ...]:
        return tuple(str(label["id"]) for label in self.labels)

    @property
    def size(self) -> int:
        return len(self.labels)

    def index(self) -> Dict[str, int]:
        return {label_id_: idx for idx, label_id_ in enumerate(self.label_ids)}

    def vectorize(self, ids: Iterable[str]) -> np.ndarray:
        lookup = self.index()
        y = np.zeros(self.size, dtype=np.float32)
        for raw_id in ids:
            idx = lookup.get(str(raw_id))
            if idx is not None:
                y[idx] = 1.0
        return y

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": "multilabel-v1", "labels": list(self.labels)}

    def save(self, path: Path | str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path | str) -> "LabelVocabulary":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        labels = payload.get("labels", []) if isinstance(payload, dict) else []
        return cls(labels=tuple(dict(label) for label in labels))


def label_metadata(label_id_: str, count: int = 0) -> Dict[str, Any]:
    group, _, code = label_id_.partition(":")
    if group == "species":
        return {
            "id": label_id_,
            "group": "species",
            "code": code,
            "name": species_display_name(code),
            "class_hierarchy": species_class_path(code),
            "count": int(count),
        }
    if group == "call":
        return {
            "id": label_id_,
            "group": "call_type",
            "code": code,
            "name": call_type_display_name(code),
            "class_hierarchy": call_type_class_path(code),
            "count": int(count),
        }
    return {"id": label_id_, "group": group or "unknown", "code": code, "name": code, "count": int(count)}


def build_vocabulary_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    min_count: int = 1,
    include_species: bool = True,
    include_call_types: bool = True,
) -> LabelVocabulary:
    counts: Counter[str] = Counter()
    for row in rows:
        for raw_id in label_ids_from_row(row):
            if raw_id.startswith("species:") and not include_species:
                continue
            if raw_id.startswith("call:") and not include_call_types:
                continue
            counts[raw_id] += 1

    def sort_key(item: Tuple[str, int]) -> Tuple[int, str]:
        raw_id, _ = item
        group_rank = 0 if raw_id.startswith("species:") else 1
        return group_rank, raw_id

    labels = [
        label_metadata(raw_id, count)
        for raw_id, count in sorted(counts.items(), key=sort_key)
        if int(count) >= int(min_count)
    ]
    return LabelVocabulary(labels=tuple(labels))


def _resolve_manifest_path(row: Dict[str, Any], dataset_root: Optional[Path]) -> Path:
    raw = clean_text(
        row.get("mat_path")
        or row.get("spectrogram_mat_path")
        or row.get("spectrogram_path")
        or row.get("relative_path")
    )
    if not raw:
        raise ValueError("Manifest row is missing mat_path/spectrogram_mat_path/relative_path")
    path = Path(raw)
    if path.is_absolute():
        return path
    if dataset_root is not None:
        return (dataset_root / path).resolve()
    return path.resolve()


class MultiLabelMatDataset(Dataset):
    """MAT spectrogram dataset with independent binary labels."""

    def __init__(
        self,
        manifest_csv: str | Path,
        vocabulary: LabelVocabulary | str | Path,
        *,
        split: Optional[str] = None,
        dataset_root: Optional[str | Path] = None,
        crop_size: Optional[int | Sequence[int]] = None,
        crop_time_seconds: Optional[float] = None,
        crop_freq_range_hz: Optional[Tuple[float, float]] = None,
        min_db: float = -80.0,
        max_db: float = 0.0,
        center_bias_sigma_frac: float = 0.25,
        positive_crop_mode: str = "centered_gaussian",
        seed: int = 0,
        return_meta: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to load MultiLabelMatDataset")
        if sio is None:
            raise RuntimeError("scipy is required to load .mat files")
        self.manifest_csv = Path(manifest_csv)
        self.vocabulary = vocabulary if isinstance(vocabulary, LabelVocabulary) else LabelVocabulary.load(vocabulary)
        self.dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
        self.split = split
        self.freq_crop, self.time_crop = parse_crop_size(crop_size)
        self.crop_time_seconds = float(crop_time_seconds) if crop_time_seconds is not None else None
        if self.crop_time_seconds is not None and self.crop_time_seconds <= 0:
            raise ValueError("crop_time_seconds must be > 0")
        self.crop_freq_range_hz = tuple(crop_freq_range_hz) if crop_freq_range_hz is not None else None
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.center_bias_sigma_frac = float(center_bias_sigma_frac)
        self.positive_crop_mode = str(positive_crop_mode)
        self.rng = np.random.default_rng(seed)
        self.return_meta = bool(return_meta)

        rows = read_csv_rows(self.manifest_csv)
        if split is not None:
            rows = [row for row in rows if clean_text(row.get("split")) == str(split)]
        self.rows: List[Dict[str, Any]] = rows
        self.files: List[Tuple[Path, np.ndarray, Dict[str, Any]]] = []
        for row in self.rows:
            ids = label_ids_from_row(row)
            target = self.vocabulary.vectorize(ids)
            mat_path = _resolve_manifest_path(row, self.dataset_root)
            self.files.append((mat_path, target, dict(row)))

    def __len__(self) -> int:
        return len(self.files)

    def _load_spectrogram_raw(self, path: Path) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
        data = sio.loadmat(str(path), simplify_cells=True)
        key = _find_key(data, POWER_KEYS)
        spec_kind = "power"
        if key is None:
            key = _find_key(data, DB_KEYS) or _find_key(data, SPECTRO_KEYS)
            spec_kind = "db"
        if key is None:
            raise KeyError(f"No spectrogram-like key found in {path.name}")
        spec = np.asarray(data[key])
        if spec.ndim != 2:
            raise ValueError(f"Unexpected spectrogram ndim {spec.ndim} in {path.name}")

        freq_key = _find_key(data, FREQ_KEYS)
        time_key = _find_key(data, TIME_KEYS)
        freqs = np.asarray(data[freq_key]).squeeze() if freq_key in data else None
        times = np.asarray(data[time_key]).squeeze() if time_key in data else None
        if freqs is not None and times is not None:
            freq_len = int(np.asarray(freqs).ravel().shape[0])
            time_len = int(np.asarray(times).ravel().shape[0])
            rows, cols = spec.shape[:2]
            if (rows, cols) == (time_len, freq_len):
                spec = spec.T
        return spec, spec_kind, freqs, times

    def _crop(
        self,
        spec: np.ndarray,
        is_positive: bool,
        freqs: Optional[np.ndarray],
        times: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, int, int]:
        freq_bins, time_bins = spec.shape
        if self.crop_freq_range_hz is not None and freqs is not None:
            freq_arr = np.asarray(freqs).ravel()
            if freq_arr.shape[0] == freq_bins:
                fmin, fmax = self.crop_freq_range_hz
                mask = (freq_arr >= float(fmin)) & (freq_arr <= float(fmax))
                if np.any(mask):
                    idx = np.where(mask)[0]
                    spec = spec[idx[0] : idx[-1] + 1, :]
                    freq_bins, time_bins = spec.shape

        target_f = self.freq_crop if self.freq_crop is not None else freq_bins
        if self.crop_time_seconds is not None:
            dt = _infer_time_bin_seconds(times)
            target_t = max(1, int(round(self.crop_time_seconds / dt))) if dt else (self.time_crop or target_f)
        else:
            target_t = self.time_crop if self.time_crop is not None else target_f

        if freq_bins < target_f:
            spec = np.pad(spec, ((0, target_f - freq_bins), (0, 0)), mode="edge")
            freq_bins = target_f
        elif freq_bins > target_f:
            f_start = max(0, (freq_bins - target_f) // 2)
            spec = spec[f_start : f_start + target_f, :]
            freq_bins = target_f

        start = _choose_start_idx(
            time_bins,
            int(target_t),
            self.split or "eval",
            bool(is_positive),
            center_bias_sigma_frac=self.center_bias_sigma_frac,
            positive_crop_mode=self.positive_crop_mode,
            rng=self.rng,
            augment_eval=False,
        )
        if time_bins < target_t:
            spec = np.pad(spec, ((0, 0), (0, int(target_t) - time_bins)), mode="edge")
        else:
            spec = spec[:, start : start + int(target_t)]
        return spec, int(start), int(target_t)

    def __getitem__(self, index: int):
        mat_path, target, row = self.files[index]
        spec, spec_kind, freqs, times = self._load_spectrogram_raw(mat_path)
        full_shape = list(spec.shape)
        spec, crop_start, crop_t = self._crop(spec, bool(np.any(target > 0)), freqs=freqs, times=times)
        if spec_kind == "power":
            spec = _power_to_db_norm(spec)
        spec = _normalize_db_to_unit(spec, self.min_db, self.max_db)
        x = torch.from_numpy(spec).unsqueeze(0).float()
        y = torch.from_numpy(target.astype(np.float32))
        if not self.return_meta:
            return x, y
        meta = {
            "item_id": row.get("item_id") or mat_path.stem,
            "mat_path": str(mat_path),
            "source_audio": row.get("source_audio") or row.get("filename"),
            "source_dataset": row.get("source_dataset") or "",
            "source_kind": row.get("source_kind") or "",
            "negative_bucket": row.get("negative_bucket") or "",
            "split": row.get("split") or self.split or "",
            "label_ids": label_ids_from_row(row),
            "source_label_ids": row.get("source_label_ids") or "",
            "canonical_label_ids": row.get("canonical_label_ids") or row.get("label_ids") or "",
            "analysis_label_ids": row.get("analysis_label_ids") or "",
            "original_label_ids": row.get("original_label_ids") or "",
            "original_target_label_ids": row.get("original_target_label_ids") or "",
            "gate_positive_source_labels": row.get("gate_positive_source_labels") or "",
            "is_background": row.get("is_background") or "",
            "review_status": row.get("review_status") or "",
            "context_tags": row.get("context_tags") or "",
            "begin_s": row.get("begin_s") or row.get("begin_time_s") or "",
            "end_s": row.get("end_s") or row.get("end_time_s") or "",
            "event_group": row.get("event_group") or "",
            "full_shape": full_shape,
            "crop_start": crop_start,
            "crop_time_bins": crop_t,
        }
        return x, y, meta


def multilabel_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_score: np.ndarray | torch.Tensor,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute thresholded multi-label metrics."""
    true_np = y_true.detach().cpu().numpy() if torch is not None and isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    score_np = y_score.detach().cpu().numpy() if torch is not None and isinstance(y_score, torch.Tensor) else np.asarray(y_score)
    if true_np.ndim != 2:
        raise ValueError("y_true must be [n_samples, n_labels]")
    if score_np.shape != true_np.shape:
        raise ValueError("y_score shape must match y_true")
    pred_np = (score_np >= float(threshold)).astype(np.int64)
    true_bin = (true_np >= 0.5).astype(np.int64)

    tp = np.sum((pred_np == 1) & (true_bin == 1), axis=0).astype(np.float64)
    fp = np.sum((pred_np == 1) & (true_bin == 0), axis=0).astype(np.float64)
    fn = np.sum((pred_np == 0) & (true_bin == 1), axis=0).astype(np.float64)
    support = np.sum(true_bin == 1, axis=0).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    tp_micro = float(tp.sum())
    fp_micro = float(fp.sum())
    fn_micro = float(fn.sum())
    micro_precision = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) else 0.0
    micro_recall = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    present = support > 0
    macro_f1 = float(np.mean(f1[present])) if np.any(present) else 0.0

    return {
        "threshold": float(threshold),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "index": int(idx),
                "support": int(support[idx]),
                "tp": int(tp[idx]),
                "fp": int(fp[idx]),
                "fn": int(fn[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
            for idx in range(true_np.shape[1])
        ],
    }


def parse_manifest_time(row: Dict[str, Any]) -> Optional[datetime]:
    for key in ("start_time", "clip_start_utc", "audio_start_time", "timestamp"):
        text = clean_text(row.get(key))
        if not text:
            continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
    source = clean_text(row.get("source_audio") or row.get("filename"))
    return parse_filename_timestamp(source)


def group_key_for_split(row: Dict[str, Any]) -> str:
    return clean_text(row.get("event_group") or row.get("source_audio") or row.get("filename") or row.get("item_id"))


def temporal_grouped_split(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split rows by group in chronological order.

    All rows with the same event/source group stay together. This is a first
    candidate split, not the final scientific split.
    """
    if not rows:
        return {"train": [], "val": [], "test": []}
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = group_key_for_split(row)
        groups[key].append(dict(row))

    def group_sort_value(item: Tuple[str, List[Dict[str, Any]]]) -> Tuple[datetime, str]:
        key, group_rows = item
        times = [parse_manifest_time(row) for row in group_rows]
        times = [dt for dt in times if dt is not None]
        fallback = datetime.max.replace(tzinfo=timezone.utc)
        return min(times) if times else fallback, key

    ordered = sorted(groups.items(), key=group_sort_value)
    n_groups = len(ordered)
    n_train = max(1, int(math.floor(n_groups * float(train_ratio)))) if n_groups > 1 else 1
    n_val = int(math.floor(n_groups * float(val_ratio)))
    if n_groups >= 3 and n_val < 1:
        n_val = 1
    if n_train + n_val >= n_groups and n_groups > 1:
        n_train = max(1, n_groups - n_val - 1)

    split_groups = {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name, group_items in split_groups.items():
        for _, group_rows in group_items:
            for row in group_rows:
                row_out = dict(row)
                row_out["split"] = split_name
                out[split_name].append(row_out)
    return out


def label_balanced_grouped_split(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split rows by group with a greedy multi-label balance heuristic.

    This is intended for smoke experiments where every split should see a
    representative slice of background and rare labels, while still preserving
    event/source grouping. It is not a substitute for the final temporal or
    deployment-level evaluation design.
    """
    if not rows:
        return {"train": [], "val": [], "test": []}
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key_for_split(row)].append(dict(row))

    group_items = list(groups.items())
    n_groups = len(group_items)
    if n_groups <= 2:
        return temporal_grouped_split(rows, train_ratio=train_ratio, val_ratio=val_ratio)

    n_train = max(1, int(math.floor(n_groups * float(train_ratio))))
    n_val = int(math.floor(n_groups * float(val_ratio)))
    if n_val < 1:
        n_val = 1
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - n_val - 1)
    target_group_counts = {
        "train": n_train,
        "val": n_val,
        "test": max(1, n_groups - n_train - n_val),
    }
    split_ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": max(0.0, 1.0 - float(train_ratio) - float(val_ratio)),
    }

    def row_label_counts(group_rows: Sequence[Dict[str, Any]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for row in group_rows:
            labels = label_ids_from_row(row)
            if labels:
                counts.update(labels)
            else:
                counts["<background>"] += 1
        return counts

    group_label_counts: Dict[str, Counter[str]] = {key: row_label_counts(group_rows) for key, group_rows in group_items}
    total_label_counts: Counter[str] = Counter()
    label_group_presence: Counter[str] = Counter()
    for key, counts in group_label_counts.items():
        total_label_counts.update(counts)
        for label in counts:
            label_group_presence[label] += 1

    target_label_counts: Dict[str, Dict[str, float]] = {
        split: {label: float(count) * ratio for label, count in total_label_counts.items()}
        for split, ratio in split_ratios.items()
    }
    current_label_counts: Dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    assigned_groups: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]] = {
        split: [] for split in ("train", "val", "test")
    }
    assigned_keys: set[str] = set()

    rng = np.random.default_rng(int(seed))
    jitter = {key: float(rng.random()) for key, _ in group_items}

    def group_time(key: str, group_rows: Sequence[Dict[str, Any]]) -> datetime:
        times = [parse_manifest_time(row) for row in group_rows]
        times = [dt for dt in times if dt is not None]
        return min(times) if times else datetime.max.replace(tzinfo=timezone.utc)

    def rarity_score(key: str) -> float:
        labels = group_label_counts[key]
        return float(sum(1.0 / max(1, label_group_presence[label]) for label in labels))

    group_lookup = {key: group_rows for key, group_rows in group_items}

    def has_capacity(split: str) -> bool:
        return len(assigned_groups[split]) < target_group_counts[split]

    def assign_group(split: str, key: str) -> None:
        group_rows = group_lookup[key]
        assigned_groups[split].append((key, group_rows))
        current_label_counts[split].update(group_label_counts[key])
        assigned_keys.add(key)

    # First reserve a small amount of coverage for validation/test labels.
    # The later greedy balance step otherwise tends to satisfy the large train
    # target for blocky species/time distributions before holdout splits ever
    # see those labels.
    coverage_split_order = ("val", "test", "train")
    labels_by_rarity = sorted(
        total_label_counts,
        key=lambda label: (label_group_presence[label], total_label_counts[label], label),
    )
    for label in labels_by_rarity:
        candidate_splits = [
            split
            for split in coverage_split_order
            if target_label_counts[split].get(label, 0.0) > 0.0
        ]
        if not candidate_splits:
            continue
        # If a label exists in only a few groups, spread it as far as it can go
        # without pretending every split can receive a positive example.
        candidate_splits = candidate_splits[: min(len(candidate_splits), label_group_presence[label])]
        for split in candidate_splits:
            if current_label_counts[split].get(label, 0) > 0 or not has_capacity(split):
                continue
            candidates = [
                key
                for key in group_lookup
                if key not in assigned_keys and group_label_counts[key].get(label, 0) > 0
            ]
            if not candidates:
                continue
            best_key = min(
                candidates,
                key=lambda key: (
                    -group_label_counts[key].get(label, 0),
                    len(group_label_counts[key]),
                    group_time(key, group_lookup[key]),
                    jitter[key],
                    key,
                ),
            )
            assign_group(split, best_key)

    ordered = sorted(
        [(key, group_rows) for key, group_rows in group_items if key not in assigned_keys],
        key=lambda item: (
            -rarity_score(item[0]),
            group_time(item[0], item[1]),
            jitter[item[0]],
            item[0],
        ),
    )

    def score_assignment(split: str, key: str) -> float:
        counts = group_label_counts[key]
        label_score = 0.0
        for label, count in counts.items():
            target = max(1.0, target_label_counts[split].get(label, 0.0))
            current = float(current_label_counts[split].get(label, 0))
            deficit = (target - current) / target
            label_score += float(count) * deficit
            if label_group_presence[label] >= 2 and current == 0.0:
                label_score += 1.0
        capacity_deficit = (
            float(target_group_counts[split] - len(assigned_groups[split]))
            / max(1.0, float(target_group_counts[split]))
        )
        return label_score + (0.25 * capacity_deficit)

    for key, group_rows in ordered:
        available = [
            split for split in ("train", "val", "test") if len(assigned_groups[split]) < target_group_counts[split]
        ]
        if not available:
            available = ["train", "val", "test"]
        best_split = max(
            available,
            key=lambda split: (
                score_assignment(split, key),
                target_group_counts[split] - len(assigned_groups[split]),
                {"val": 2, "test": 1, "train": 0}[split],
            ),
        )
        assign_group(best_split, key)

    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name, split_group_items in assigned_groups.items():
        sorted_groups = sorted(split_group_items, key=lambda item: (group_time(item[0], item[1]), item[0]))
        for _, group_rows in sorted_groups:
            for row in group_rows:
                row_out = dict(row)
                row_out["split"] = split_name
                out[split_name].append(row_out)
    return out
