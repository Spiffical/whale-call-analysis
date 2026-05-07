#!/usr/bin/env python3
"""Standardize multi-source manifests into the weekend canonical label schema."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    build_vocabulary_from_rows,
    canonicalize_source_label_ids,
    clean_text,
    label_ids_from_row,
    read_csv_rows,
    split_pipe,
    write_csv_rows,
)


SCHEMA_VERSION = "weekend-canonical-v1"


def _parse_input_spec(spec: str) -> Tuple[str, Path, Optional[Path]]:
    parts = spec.split("|")
    if len(parts) not in {2, 3}:
        raise ValueError(
            "--input must be formatted as source_dataset|manifest_csv|dataset_root "
            "or source_dataset|manifest_csv"
        )
    source_dataset = clean_text(parts[0])
    manifest = Path(parts[1]).expanduser()
    root = Path(parts[2]).expanduser() if len(parts) == 3 and clean_text(parts[2]) else None
    return source_dataset, manifest, root


def _resolve_path(value: str, root: Optional[Path]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute() or root is None:
        return str(path)
    return str((root / path).resolve())


def _mat_path_text(row: Dict[str, Any]) -> str:
    explicit = (
        clean_text(row.get("mat_path"))
        or clean_text(row.get("spectrogram_mat_path"))
        or clean_text(row.get("spectrogram_path"))
        or clean_text(row.get("relative_path"))
    )
    if explicit:
        return explicit
    expected_name = clean_text(row.get("expected_mat_name"))
    if not expected_name:
        return ""
    expected_path = Path(expected_name)
    if len(expected_path.parts) == 1:
        return str(Path("mat_files") / expected_path)
    return expected_name


def _source_label_ids(row: Dict[str, Any]) -> List[str]:
    explicit = split_pipe(row.get("source_label_ids"))
    if explicit:
        return list(explicit)
    return label_ids_from_row(row)


def standardize_rows(
    input_specs: Sequence[str],
    *,
    include_species: bool,
    include_call_types: bool,
    primary_species: Sequence[str],
    drop_empty: bool = False,
    dedupe_key_fields: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": [],
        "row_count": 0,
        "dropped_empty_count": 0,
        "dedupe_key_fields": list(dedupe_key_fields),
        "dedupe_dropped_count": 0,
        "dedupe_examples": [],
        "source_label_counts": {},
        "canonical_label_counts": {},
        "analysis_label_counts": {},
        "source_dataset_counts": {},
        "split_counts": {},
        "source_split_counts": {},
    }
    source_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    analysis_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    seen_dedupe_keys: set[Tuple[str, ...]] = set()

    for input_spec in input_specs:
        default_source, manifest_csv, dataset_root = _parse_input_spec(input_spec)
        rows = read_csv_rows(manifest_csv)
        summary["inputs"].append(
            {
                "source_dataset": default_source,
                "manifest_csv": str(manifest_csv),
                "dataset_root": str(dataset_root) if dataset_root is not None else "",
                "row_count": len(rows),
            }
        )
        for row in rows:
            source_dataset = clean_text(row.get("source_dataset")) or default_source
            source_ids = _source_label_ids(row)
            canonical_ids, analysis_ids = canonicalize_source_label_ids(
                source_ids,
                include_species=include_species,
                include_call_types=include_call_types,
                primary_species=primary_species,
            )
            if drop_empty and not canonical_ids:
                summary["dropped_empty_count"] += 1
                continue
            out = dict(row)
            out["source_dataset"] = source_dataset
            out["source_label_ids"] = "|".join(source_ids)
            out["canonical_label_ids"] = "|".join(canonical_ids)
            out["analysis_label_ids"] = "|".join(analysis_ids)
            out["label_ids"] = "|".join(canonical_ids)
            out["canonical_schema_version"] = SCHEMA_VERSION
            out["mat_path"] = _resolve_path(_mat_path_text(out), dataset_root)
            if clean_text(out.get("source_audio")):
                out["source_audio"] = _resolve_path(clean_text(out.get("source_audio")), dataset_root)
            split = clean_text(out.get("split")) or "unsplit"
            if dedupe_key_fields:
                dedupe_key = tuple(clean_text(out.get(field)) for field in dedupe_key_fields)
                if any(dedupe_key):
                    if dedupe_key in seen_dedupe_keys:
                        summary["dedupe_dropped_count"] += 1
                        if len(summary["dedupe_examples"]) < 10:
                            summary["dedupe_examples"].append(
                                {
                                    "item_id": clean_text(out.get("item_id")),
                                    "source_dataset": source_dataset,
                                    "key": dict(zip(dedupe_key_fields, dedupe_key)),
                                }
                            )
                        continue
                    seen_dedupe_keys.add(dedupe_key)
            rows_out.append(out)
            source_counts.update(source_ids or ["<background>"])
            canonical_counts.update(canonical_ids or ["<background>"])
            analysis_counts.update(analysis_ids)
            dataset_counts[source_dataset] += 1
            split_counts[split] += 1
            source_split_counts[source_dataset][split] += 1

    summary["row_count"] = len(rows_out)
    summary["source_label_counts"] = dict(source_counts.most_common())
    summary["canonical_label_counts"] = dict(canonical_counts.most_common())
    summary["analysis_label_counts"] = dict(analysis_counts.most_common())
    summary["source_dataset_counts"] = dict(dataset_counts.most_common())
    summary["split_counts"] = dict(split_counts.most_common())
    summary["source_split_counts"] = {
        source: dict(counts.most_common()) for source, counts in sorted(source_split_counts.items())
    }
    return rows_out, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input spec: source_dataset|manifest_csv|dataset_root. Repeat to combine sources.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["species", "species_call"], default="species_call")
    parser.add_argument("--primary-species", default="Bm,Bp,Mn,Oo")
    parser.add_argument("--vocab-min-count", type=int, default=1)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument(
        "--dedupe-key",
        action="append",
        default=[],
        help="Field name or comma-separated field names used to drop duplicate rows after path resolution.",
    )
    args = parser.parse_args()

    primary_species = [token.strip() for token in args.primary_species.split(",") if token.strip()]
    dedupe_key_fields = [
        field.strip()
        for raw in args.dedupe_key
        for field in str(raw).split(",")
        if field.strip()
    ]
    rows, summary = standardize_rows(
        args.input,
        include_species=True,
        include_call_types=args.mode == "species_call",
        primary_species=primary_species,
        drop_empty=bool(args.drop_empty),
        dedupe_key_fields=dedupe_key_fields,
    )
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_dir / "standardized_manifest.csv", rows)
    vocab = build_vocabulary_from_rows(rows, min_count=int(args.vocab_min_count))
    vocab.save(out_dir / "label_vocabulary.json")
    summary["mode"] = args.mode
    summary["primary_species"] = primary_species
    summary["vocabulary_size"] = vocab.size
    summary["vocabulary_label_ids"] = list(vocab.label_ids)
    with open(out_dir / "standardization_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
