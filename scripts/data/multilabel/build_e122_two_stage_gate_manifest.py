#!/usr/bin/env python3
"""Build the E122 binary whale-call gate manifest.

Rows containing any requested target species are rewritten to one synthetic
binary label. Rows without those species keep blank training labels and serve as
background/no-whale negatives. Original label fields are retained in audit
columns so the gate manifest can still be inspected later.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_POSITIVE_LABELS = ("species:Bp", "species:Bm", "species:Mn")
DEFAULT_GATE_LABEL = "task:whale_call"


def clean(value: Any) -> str:
    return str(value or "").strip()


def split_tokens(value: Any) -> List[str]:
    return [token.strip() for token in clean(value).replace(",", "|").split("|") if token.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_labels(row: Mapping[str, Any]) -> List[str]:
    labels: List[str] = []
    for key in ("label_ids", "target_label_ids", "canonical_label_ids", "analysis_label_ids", "source_label_ids"):
        for label in split_tokens(row.get(key)):
            if label not in labels:
                labels.append(label)
    species = clean(row.get("species")) or clean(row.get("species_code")) or clean(row.get("canonical_species"))
    if species and not species.startswith("species:"):
        species = f"species:{species}"
    if species and species not in labels:
        labels.append(species)
    return labels


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_vocab(path: Path, gate_label: str, positives: Sequence[str], positive_count: int) -> None:
    payload = {
        "schema_version": "multilabel-v1",
        "labels": [
            {
                "id": gate_label,
                "group": "task",
                "code": gate_label.split(":", 1)[-1],
                "name": "Any target whale call",
                "class_hierarchy": "Task > Whale call gate",
                "count": int(positive_count),
                "positive_source_labels": list(positives),
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_gate_manifest(
    *,
    input_manifest: Path,
    output_csv: Path,
    output_vocab: Path,
    output_summary: Path,
    positive_labels: Sequence[str],
    gate_label: str,
    source_kinds: Sequence[str] = (),
) -> Dict[str, Any]:
    positive_set = set(positive_labels)
    selected_sources = {clean(source) for source in source_kinds if clean(source)}
    input_rows = read_csv(input_manifest)
    fieldnames = list(input_rows[0].keys()) if input_rows else []
    for field in (
        "original_label_ids",
        "original_target_label_ids",
        "gate_positive_source_labels",
        "label_ids",
        "canonical_label_ids",
        "analysis_label_ids",
        "target_label_ids",
        "is_background",
    ):
        if field not in fieldnames:
            fieldnames.append(field)

    rows: List[Dict[str, Any]] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in input_rows:
        if selected_sources and clean(row.get("source_kind")) not in selected_sources:
            continue
        labels = row_labels(row)
        matched = [label for label in labels if label in positive_set]
        out = dict(row)
        out["original_label_ids"] = "|".join(labels)
        out["original_target_label_ids"] = clean(row.get("target_label_ids"))
        out["gate_positive_source_labels"] = "|".join(matched)
        target = gate_label if matched else ""
        for field in ("label_ids", "canonical_label_ids", "analysis_label_ids", "target_label_ids"):
            out[field] = target
        out["is_background"] = "0" if target else "1"
        rows.append(out)
        counts[(clean(out.get("split")), clean(out.get("source_kind")), target or "<background>")] += 1

    write_csv(output_csv, rows, fieldnames)
    write_vocab(output_vocab, gate_label, positive_labels, sum(1 for row in rows if clean(row.get("label_ids")) == gate_label))
    summary = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_csv),
        "output_vocab": str(output_vocab),
        "gate_label": gate_label,
        "positive_labels": list(positive_labels),
        "source_kinds": sorted(selected_sources),
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if clean(row.get("label_ids")) == gate_label),
        "background_rows": sum(1 for row in rows if clean(row.get("label_ids")) != gate_label),
        "split_source_label_counts": {
            f"{split}|{source}|{label}": count
            for (split, source, label), count in sorted(counts.items())
        },
    }
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--positive-labels", default=",".join(DEFAULT_POSITIVE_LABELS))
    parser.add_argument("--gate-label", default=DEFAULT_GATE_LABEL)
    parser.add_argument("--source-kind", action="append", default=None, help="Keep only this source kind; may be repeated")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_gate_manifest(
        input_manifest=args.input_manifest,
        output_csv=args.output_dir / "standardized_manifest.csv",
        output_vocab=args.output_dir / "label_vocabulary.json",
        output_summary=args.output_dir / "manifest_counts.json",
        positive_labels=split_tokens(args.positive_labels),
        gate_label=clean(args.gate_label),
        source_kinds=list(args.source_kind or []),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
