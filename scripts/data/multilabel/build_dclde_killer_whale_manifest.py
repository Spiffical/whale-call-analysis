#!/usr/bin/env python3
"""Build bounded DCLDE killer-whale manifests for ONC Oo-repair experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    build_vocabulary_from_rows,
    call_type_display_name,
    clean_text,
    label_ids_from_row,
    species_display_name,
    write_csv_rows,
)


DEFAULT_GCS_ROOT = "gs://noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales"
DEFAULT_HTTPS_ROOT = "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales"

PROVIDER_TO_GCS_SLUG = {
    "DFO_CRP": "dfo_crp",
    "DFO_WDLP": "dfo_wdlp",
    "JASCO_VFPA": "vfpa",
    "JASCO_VFPA_ONC": "vfpa",
    "ONC": "onc",
    "OrcaSound": "orcasound",
    "SIO": "scripps",
    "SIMRES": "simres",
    "SMRUConsulting": "smru",
    "UAF_NGOS": "uaf",
}

DCLDE_PRIMARY_CLASS_TO_LABELS = {
    "KW": ("Oo", "orca_call"),
    "HW": ("Mn", ""),
}

CLASS_TO_ANALYSIS_LABEL = {
    "AB": "confounder:abiotic",
    "UndBio": "confounder:undetermined_biological",
}
CLASS_TO_NEGATIVE_BUCKET = {
    "AB": "nonbiological_signal",
    "UndBio": "nonprimary_biological_signal",
}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{**row, "__source_row_id": str(idx)} for idx, row in enumerate(csv.DictReader(handle), start=1)]


def _float_or_none(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _true_text(value: Any) -> bool:
    return clean_text(value).lower() in {"true", "1", "yes", "y"}


def _safe_token(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", clean_text(value)).strip("-")
    return text or "unknown"


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def _clip_name(provider: str, dataset: str, soundfile: str) -> str:
    return f"dclde_{_safe_token(provider)}_{_safe_token(dataset)}__{Path(soundfile).name}"


def _expected_mat_name(clip: str, begin_s: float, end_s: float) -> str:
    return f"{clip}_{float(begin_s):.1f}s_{float(end_s):.1f}s_trainstyle.mat"


def _labels_json(*, species_code: str, call_type: str, source: str) -> str:
    labels = [
        {
            "species_code": species_code,
            "species": species_display_name(species_code),
            "call_type": call_type or None,
            "call_type_name": call_type_display_name(call_type) if call_type else None,
            "source": source or "dclde_2027_killer_whales",
            "review_status": "reviewed",
            "trainable": True,
        }
    ]
    return json.dumps(labels, sort_keys=True, separators=(",", ":"))


def _load_gcs_object_index(paths: Sequence[Path]) -> Tuple[Dict[Tuple[str, str, str], str], Dict[str, List[str]]]:
    by_key: Dict[Tuple[str, str, str], str] = {}
    by_soundfile: Dict[str, List[str]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                name = clean_text(line)
                if not name or name.startswith("#"):
                    continue
                parts = name.split("/")
                try:
                    provider_idx = parts.index("dclde_2027_killer_whales") + 1
                except ValueError:
                    continue
                if len(parts) < provider_idx + 4 or parts[provider_idx + 1].lower() != "audio":
                    continue
                provider_slug = parts[provider_idx]
                dataset_slug = parts[provider_idx + 2]
                soundfile = parts[-1]
                by_key[(provider_slug, _slug(dataset_slug), soundfile)] = name
                by_soundfile[soundfile].append(name)
    return by_key, by_soundfile


def _infer_gcs_object(
    *,
    provider: str,
    dataset: str,
    soundfile: str,
    by_key: Dict[Tuple[str, str, str], str],
    by_soundfile: Dict[str, List[str]],
) -> str:
    provider_slug = PROVIDER_TO_GCS_SLUG.get(provider, _slug(provider))
    key = (provider_slug, _slug(dataset), soundfile)
    if key in by_key:
        return by_key[key]
    matches = by_soundfile.get(soundfile, [])
    if matches:
        provider_matches = [name for name in matches if f"/{provider_slug}/" in name]
        if provider_matches:
            return sorted(provider_matches)[0]
        return sorted(matches)[0]
    return f"dclde/2027/dclde_2027_killer_whales/{provider_slug}/audio/{_slug(dataset)}/{soundfile}"


def _balanced_take(rows: Sequence[Dict[str, Any]], cap: int, group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(clean_text(row.get(field)) for field in group_fields)
        grouped[key].append(dict(row))
    for key in grouped:
        grouped[key] = sorted(
            grouped[key],
            key=lambda row: (
                clean_text(row.get("source_provider")),
                clean_text(row.get("source_dataset")),
                clean_text(row.get("filename")),
                float(row.get("begin_s") or 0.0),
            ),
        )

    selected: List[Dict[str, Any]] = []
    keys = sorted(grouped)
    while len(selected) < cap and keys:
        next_keys: List[str] = []
        for key in keys:
            if len(selected) >= cap:
                break
            bucket = grouped[key]
            if bucket:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _class_balanced_take(
    rows: Sequence[Dict[str, Any]],
    cap: int,
    *,
    class_field: str,
    group_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[clean_text(row.get(class_field)) or "<blank>"].append(dict(row))
    ordered_by_class = {
        class_name: _balanced_take(class_rows, len(class_rows), group_fields)
        for class_name, class_rows in sorted(by_class.items())
    }
    selected: List[Dict[str, Any]] = []
    active_classes = sorted(ordered_by_class)
    while len(selected) < cap and active_classes:
        next_classes: List[str] = []
        for class_name in active_classes:
            if len(selected) >= cap:
                break
            bucket = ordered_by_class[class_name]
            if bucket:
                selected.append(bucket.pop(0))
            if bucket:
                next_classes.append(class_name)
        active_classes = next_classes
    return selected


def _manifest_row(
    *,
    raw: Dict[str, Any],
    class_species: str,
    is_positive: bool,
    gcs_object: str,
    mat_rel_dir: str,
) -> Dict[str, Any]:
    provider = clean_text(raw.get("Provider"))
    dataset = clean_text(raw.get("Dataset"))
    soundfile = clean_text(raw.get("Soundfile"))
    begin_s = float(raw["FileBeginSec"])
    end_s = float(raw["FileEndSec"])
    clip = _clip_name(provider, dataset, soundfile)
    expected_mat = _expected_mat_name(clip, begin_s, end_s)
    source_dataset = f"dclde_2027_{_safe_token(provider)}_{_safe_token(dataset)}"
    species_code, call_type = DCLDE_PRIMARY_CLASS_TO_LABELS.get(class_species, ("", ""))
    labels_json = _labels_json(species_code=species_code, call_type=call_type, source=source_dataset) if is_positive else "[]"
    positive_ids = [f"species:{species_code}"] if species_code else []
    if call_type:
        positive_ids.append(f"call:{call_type}")
    source_ids = "|".join(positive_ids) if is_positive else CLASS_TO_ANALYSIS_LABEL.get(class_species, f"confounder:{class_species}")
    canonical_label_ids = "|".join(sorted(positive_ids)) if is_positive else ""
    row = {
        "item_id": Path(expected_mat).stem,
        "clip": clip,
        "filename": soundfile,
        "source_audio": soundfile,
        "begin_s": f"{begin_s:.6f}",
        "end_s": f"{end_s:.6f}",
        "begin_time_s": f"{begin_s:.6f}",
        "end_time_s": f"{end_s:.6f}",
        "window_start_s": f"{begin_s:.6f}",
        "duration_s": f"{(end_s - begin_s):.6f}",
        "expected_mat_name": expected_mat,
        "mat_path": str(Path(mat_rel_dir) / expected_mat),
        "source_dataset": source_dataset,
        "source_kind": "DCLDE",
        "source_row_id": clean_text(raw.get("__source_row_id")),
        "source_provider": provider,
        "source_dataset_raw": dataset,
        "source_soundfile": soundfile,
        "source_class_species": class_species,
        "source_label_ids": source_ids,
        "canonical_label_ids": canonical_label_ids,
        "canonical_species": species_code if is_positive else "",
        "canonical_call_type": call_type if is_positive else "",
        "analysis_label_ids": "" if is_positive else source_ids,
        "negative_bucket": "" if is_positive else CLASS_TO_NEGATIVE_BUCKET.get(class_species, "ambiguous_hard_negative"),
        "context_tags": "" if is_positive else CLASS_TO_NEGATIVE_BUCKET.get(class_species, "ambiguous_hard_negative"),
        "dclde_ecotype": clean_text(raw.get("Ecotype")),
        "dclde_annotation_level": clean_text(raw.get("AnnotationLevel")),
        "dclde_kw": clean_text(raw.get("KW")),
        "dclde_kw_certain": clean_text(raw.get("KW_certain")),
        "low_frequency_hz": clean_text(raw.get("LowFreqHz")),
        "high_frequency_hz": clean_text(raw.get("HighFreqHz")),
        "gcs_object": gcs_object,
        "gcs_uri": f"gs://noaa-passive-bioacoustic/{gcs_object}",
        "https_url": f"https://storage.googleapis.com/noaa-passive-bioacoustic/{gcs_object}",
        "is_background": "0" if is_positive else "1",
        "review_status": "reviewed" if is_positive else "reviewed_confounder",
        "species": species_code if is_positive else "",
        "species_code": species_code if is_positive else "",
        "call_type": call_type if is_positive else "",
        "call_type_std": call_type if is_positive else "",
        "call_type_raw": call_type if is_positive else "",
        "event_group": f"{provider}:{dataset}:{soundfile}",
        "labels_json": labels_json,
    }
    row["label_ids"] = "|".join(label_ids_from_row(row)) if is_positive else ""
    return row


def build_dclde_manifest(
    *,
    annotations_csv: Path,
    output_dir: Path,
    gcs_object_lists: Sequence[Path] = (),
    require_gcs_audio: bool = False,
    max_positive: int = 200,
    max_hard_negative: int = 200,
    hard_negative_classes: Sequence[str] = ("UndBio", "AB"),
    mat_rel_dir: str = "mat_files",
    vocab_min_count: int = 1,
) -> Dict[str, Any]:
    gcs_by_key, gcs_by_soundfile = _load_gcs_object_index(gcs_object_lists)
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    class_counts_raw: Counter[str] = Counter()

    for raw in _read_rows(annotations_csv):
        class_species = clean_text(raw.get("ClassSpecies"))
        class_counts_raw[class_species or "<blank>"] += 1
        if not _true_text(raw.get("FileOk", "TRUE")):
            skipped["file_not_ok"] += 1
            continue
        soundfile = clean_text(raw.get("Soundfile"))
        provider = clean_text(raw.get("Provider"))
        dataset = clean_text(raw.get("Dataset"))
        begin_s = _float_or_none(raw.get("FileBeginSec"))
        end_s = _float_or_none(raw.get("FileEndSec"))
        if not soundfile or not provider or not dataset:
            skipped["missing_source_fields"] += 1
            continue
        if begin_s is None or end_s is None or end_s <= begin_s:
            skipped["invalid_time"] += 1
            continue
        is_positive = class_species in DCLDE_PRIMARY_CLASS_TO_LABELS
        is_hard_negative = class_species in set(hard_negative_classes)
        if not is_positive and not is_hard_negative:
            skipped["unsupported_class_species"] += 1
            continue
        gcs_object = _infer_gcs_object(
            provider=provider,
            dataset=dataset,
            soundfile=soundfile,
            by_key=gcs_by_key,
            by_soundfile=gcs_by_soundfile,
        )
        if require_gcs_audio and gcs_object not in set(gcs_by_soundfile.get(soundfile, [])):
            skipped["missing_gcs_audio"] += 1
            continue
        row = _manifest_row(
            raw=raw,
            class_species=class_species,
            is_positive=is_positive,
            gcs_object=gcs_object,
            mat_rel_dir=mat_rel_dir,
        )
        if is_positive:
            positives.append(row)
        else:
            negatives.append(row)

    positives = _class_balanced_take(
        positives,
        int(max_positive),
        class_field="source_class_species",
        group_fields=("source_provider", "source_dataset_raw", "dclde_ecotype"),
    )
    negatives = _balanced_take(negatives, int(max_hard_negative), ("source_class_species", "source_provider", "source_dataset_raw"))
    rows = sorted(
        positives + negatives,
        key=lambda row: (
            clean_text(row.get("source_provider")),
            clean_text(row.get("source_dataset_raw")),
            clean_text(row.get("filename")),
            float(row.get("begin_s") or 0.0),
            clean_text(row.get("source_class_species")),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "selected_calls.csv", rows)
    write_csv_rows(output_dir / "positive_calls.csv", positives)
    write_csv_rows(output_dir / "hard_negative_windows.csv", negatives)
    write_csv_rows(output_dir / "expected_multilabel_manifest.csv", rows)
    write_csv_rows(
        output_dir / "required_audio_sources.csv",
        [
            {
                "clip": row["clip"],
                "source_provider": row["source_provider"],
                "source_dataset_raw": row["source_dataset_raw"],
                "source_audio": row["source_audio"],
                "gcs_object": row["gcs_object"],
                "gcs_uri": row["gcs_uri"],
                "https_url": row["https_url"],
            }
            for row in rows
        ],
    )
    required = sorted({clean_text(row.get("clip")) for row in rows if clean_text(row.get("clip"))})
    (output_dir / "required_audio_filenames.txt").write_text("\n".join(required) + ("\n" if required else ""), encoding="utf-8")
    vocab = build_vocabulary_from_rows(rows, min_count=max(1, int(vocab_min_count)))
    vocab.save(output_dir / "label_vocabulary.json")

    label_counts = Counter()
    source_class_counts = Counter()
    provider_counts = Counter()
    for row in rows:
        ids = label_ids_from_row(row)
        label_counts.update(ids or ["<background>"])
        source_class_counts[clean_text(row.get("source_class_species")) or "<blank>"] += 1
        provider_counts[clean_text(row.get("source_provider")) or "<blank>"] += 1

    summary = {
        "dataset_name": "dclde_2027_killer_whales",
        "annotations_csv": str(annotations_csv.resolve()),
        "row_count": len(rows),
        "positive_count": len(positives),
        "hard_negative_count": len(negatives),
        "required_audio_count": len(required),
        "label_counts": dict(label_counts.most_common()),
        "source_class_counts": dict(source_class_counts.most_common()),
        "provider_counts": dict(provider_counts.most_common()),
        "raw_class_counts": dict(class_counts_raw.most_common()),
        "skipped_counts": dict(skipped.most_common()),
        "vocabulary_size": vocab.size,
        "vocabulary_label_ids": list(vocab.label_ids),
        "config": {
            "gcs_object_lists": [str(path.resolve()) for path in gcs_object_lists],
            "require_gcs_audio": bool(require_gcs_audio),
            "max_positive": int(max_positive),
            "max_hard_negative": int(max_hard_negative),
            "hard_negative_classes": list(hard_negative_classes),
            "mat_rel_dir": mat_rel_dir,
            "vocab_min_count": max(1, int(vocab_min_count)),
        },
    }
    (output_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gcs-object-list", action="append", default=[])
    parser.add_argument("--require-gcs-audio", action="store_true")
    parser.add_argument("--max-positive", type=int, default=200)
    parser.add_argument("--max-hard-negative", type=int, default=200)
    parser.add_argument("--hard-negative-classes", default="UndBio,AB")
    parser.add_argument("--mat-rel-dir", default="mat_files")
    parser.add_argument("--vocab-min-count", type=int, default=1)
    args = parser.parse_args()
    summary = build_dclde_manifest(
        annotations_csv=Path(args.annotations_csv),
        output_dir=Path(args.output_dir),
        gcs_object_lists=[Path(path) for path in args.gcs_object_list],
        require_gcs_audio=bool(args.require_gcs_audio),
        max_positive=int(args.max_positive),
        max_hard_negative=int(args.max_hard_negative),
        hard_negative_classes=[token.strip() for token in args.hard_negative_classes.split(",") if token.strip()],
        mat_rel_dir=str(args.mat_rel_dir),
        vocab_min_count=int(args.vocab_min_count),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
