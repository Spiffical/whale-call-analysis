#!/usr/bin/env python3
"""Build E23 killer whale localization diagnostic manifests.

E23 focuses on the ONC killer whale support set after E22 showed that DCLDE
killer whale is easy in-domain while ONC killer whale remains weak. It repairs
existing E16/E22-style manifests by hydrating ONC annotation frequency metadata
from the raw Part 2 annotation table, then creates split variants that keep
source clips or source dates together.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.multilabel import build_e20_diagnostic_variants as e20  # noqa: E402


E23_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "E23_killer_whale_onc_only_midhigh_clip_split",
        "description": "Killer whale ONC-only mid+high expert with source clips kept within one split.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
        "drop_oversampled_rows": True,
        "split_grouping": "source_audio",
        "split_grouping_source_kinds": ["ONC"],
        "split_grouping_label_ids": ["species:Oo"],
    },
    {
        "name": "E23_killer_whale_onc_only_midhigh_date_split",
        "description": "Killer whale ONC-only mid+high expert with source dates kept within one split.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
        "drop_oversampled_rows": True,
        "split_grouping": "source_date",
        "split_grouping_source_kinds": ["ONC"],
        "split_grouping_label_ids": ["species:Oo"],
    },
    {
        "name": "E23_killer_whale_onc_dclde_midhigh_sourcecap_clip_split",
        "description": "Killer whale ONC+DCLDE source-capped mid+high expert with ONC source clips kept within one split.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "source_label_train_cap",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
        "drop_oversampled_rows": True,
        "split_grouping": "source_audio",
        "split_grouping_source_kinds": ["ONC"],
        "split_grouping_label_ids": ["species:Oo"],
    },
    {
        "name": "E23_killer_whale_onc_dclde_midhigh_sourcecap_date_split",
        "description": "Killer whale ONC+DCLDE source-capped mid+high expert with ONC source dates kept within one split.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "source_label_train_cap",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
        "drop_oversampled_rows": True,
        "split_grouping": "source_date",
        "split_grouping_source_kinds": ["ONC"],
        "split_grouping_label_ids": ["species:Oo"],
    },
]


ANNOTATION_METADATA_FIELDS = [
    "sheet",
    "row_index",
    "filename",
    "call_type_raw",
    "call_type_bucket",
    "begin_time_s",
    "end_time_s",
    "low_freq_hz",
    "high_freq_hz",
    "peak_freq_hz",
    "peak_power",
    "comments",
    "verified_flag",
    "vessel_flag",
    "granularity",
    "low_frequency_hz",
    "high_frequency_hz",
]


def clean(value: Any) -> str:
    return str(value or "").strip()


def safe_float(value: Any) -> float | None:
    try:
        text = clean(value)
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    numeric = safe_float(value)
    return "" if numeric is None else f"{numeric:.6f}"


def annotation_key_from_raw(row: Mapping[str, Any]) -> Tuple[str, str]:
    begin = safe_float(row.get("begin_time_s") or row.get("begin_time"))
    return clean(row.get("filename")), "" if begin is None else f"{begin:.3f}"


def annotation_key_from_manifest(row: Mapping[str, Any]) -> Tuple[str, str]:
    clip = clean(row.get("clip") or row.get("filename") or Path(clean(row.get("source_audio"))).name)
    begin = safe_float(row.get("begin_s") or row.get("begin_time_s") or row.get("window_start_s"))
    return clip, "" if begin is None else f"{begin:.3f}"


def metadata_from_raw(row: Mapping[str, Any]) -> Dict[str, str]:
    out = {
        "sheet": clean(row.get("sheet")),
        "row_index": clean(row.get("row_index")),
        "filename": clean(row.get("filename")),
        "call_type_raw": clean(row.get("call_type_raw")),
        "call_type_bucket": clean(row.get("call_type_bucket")),
        "begin_time_s": fmt(row.get("begin_time_s") or row.get("begin_time")),
        "end_time_s": fmt(row.get("end_time_s") or row.get("end_time")),
        "low_freq_hz": fmt(row.get("low_freq_hz") or row.get("low_freq")),
        "high_freq_hz": fmt(row.get("high_freq_hz") or row.get("high_freq")),
        "peak_freq_hz": fmt(row.get("peak_freq_hz") or row.get("peak_freq")),
        "peak_power": fmt(row.get("peak_power")),
        "comments": clean(row.get("comments")),
        "verified_flag": clean(row.get("verified_flag")),
        "vessel_flag": clean(row.get("vessel_flag")),
        "granularity": clean(row.get("granularity")),
    }
    out["low_frequency_hz"] = out["low_freq_hz"]
    out["high_frequency_hz"] = out["high_freq_hz"]
    return out


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        return list(reader.fieldnames), rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hydrate_onc_annotation_metadata(
    *,
    input_manifest: Path,
    raw_onc_annotations: Path,
    output_manifest: Path,
) -> Dict[str, Any]:
    fieldnames, rows = read_csv(input_manifest)
    _, raw_rows = read_csv(raw_onc_annotations)
    # ONC uses call_type_raw=CK for many generic odontocete clicks. Killer
    # whale positives are the rows whose species code is Oo, regardless of
    # their raw call type.
    raw_by_key = {
        annotation_key_from_raw(row): metadata_from_raw(row)
        for row in raw_rows
        if clean(row.get("species")) == "Oo"
    }
    fieldnames_out = list(fieldnames)
    for field in ANNOTATION_METADATA_FIELDS:
        if field not in fieldnames_out:
            fieldnames_out.append(field)

    matched = 0
    onc_killer_rows = 0
    rows_out: List[Dict[str, str]] = []
    for row in rows:
        out = dict(row)
        is_onc = clean(out.get("source_kind")) == "ONC"
        is_killer = "species:Oo" in clean(out.get("label_ids"))
        if is_onc and is_killer:
            onc_killer_rows += 1
            meta = raw_by_key.get(annotation_key_from_manifest(out))
            if meta:
                matched += 1
                for key, value in meta.items():
                    out[key] = value
        for field in fieldnames_out:
            out.setdefault(field, "")
        rows_out.append(out)

    write_csv(output_manifest, fieldnames_out, rows_out)
    summary = {
        "input_manifest": str(input_manifest),
        "raw_onc_annotations": str(raw_onc_annotations),
        "output_manifest": str(output_manifest),
        "raw_onc_killer_annotations": len(raw_by_key),
        "onc_killer_manifest_rows": onc_killer_rows,
        "matched_onc_killer_rows": matched,
        "unmatched_onc_killer_rows": onc_killer_rows - matched,
        "added_fields": [field for field in ANNOTATION_METADATA_FIELDS if field not in fieldnames],
    }
    output_manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--input-vocab", required=True, type=Path)
    parser.add_argument("--raw-onc-annotations", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hydrated_dir = args.output_root / "_hydrated_input"
    hydrated_manifest = hydrated_dir / "standardized_manifest.csv"
    hydrate_summary = hydrate_onc_annotation_metadata(
        input_manifest=args.input_manifest,
        raw_onc_annotations=args.raw_onc_annotations,
        output_manifest=hydrated_manifest,
    )
    hydrated_dir.mkdir(parents=True, exist_ok=True)
    (hydrated_dir / "label_vocabulary.json").write_text(args.input_vocab.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(hydrate_summary, indent=2, sort_keys=True))

    original_variants = e20.VARIANTS
    try:
        e20.VARIANTS = E23_VARIANTS
        e20.build_variants(
            input_manifest=hydrated_manifest,
            input_vocab=args.input_vocab,
            output_root=args.output_root,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
        )
    finally:
        e20.VARIANTS = original_variants
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
