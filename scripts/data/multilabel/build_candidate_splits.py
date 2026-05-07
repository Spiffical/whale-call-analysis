#!/usr/bin/env python3
"""Build leakage-aware candidate splits for multi-label smoke experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import (  # noqa: E402
    clean_text,
    group_key_for_split,
    label_balanced_grouped_split,
    label_ids_from_row,
    read_csv_rows,
    temporal_grouped_split,
    write_csv_rows,
)


def _split_text_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}.txt"


def _label_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        labels = label_ids_from_row(row)
        if labels:
            counts.update(labels)
        else:
            counts["<background>"] += 1
    return dict(counts.most_common())


def _source_key(row: Dict[str, Any], source_key_fields: Sequence[str]) -> str:
    values = [clean_text(row.get(field)) for field in source_key_fields]
    text = "|".join(value or "<blank>" for value in values)
    return text or "<unknown>"


def _stable_seed_offset(text: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 1_000_003


def _group_leakage(split_rows: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    split_to_groups: Dict[str, set[str]] = {}
    group_to_splits: Dict[str, set[str]] = defaultdict(set)
    for split, rows in split_rows.items():
        groups = {group_key_for_split(row) for row in rows}
        split_to_groups[split] = groups
        for group in groups:
            group_to_splits[group].add(split)
    leaked = {group: sorted(splits) for group, splits in group_to_splits.items() if len(splits) > 1}
    return {
        "leaked_group_count": len(leaked),
        "leaked_groups": leaked,
        "group_counts": {split: len(groups) for split, groups in split_to_groups.items()},
    }


def _source_split_label_counts(
    split_rows: Dict[str, Sequence[Dict[str, Any]]],
    source_key_fields: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split, rows in split_rows.items():
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_source[_source_key(row, source_key_fields)].append(dict(row))
        out[split] = {source: _label_counts(source_rows) for source, source_rows in sorted(by_source.items())}
    return out


def _summarize(
    split_rows: Dict[str, Sequence[Dict[str, Any]]],
    *,
    source_key_fields: Sequence[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"splits": {}, "leakage": _group_leakage(split_rows)}
    for split, rows in split_rows.items():
        summary["splits"][split] = {
            "row_count": len(rows),
            "group_count": len({group_key_for_split(row) for row in rows}),
            "background_row_count": sum(1 for row in rows if not label_ids_from_row(row)),
            "label_counts": _label_counts(rows),
        }
    summary["source_key_fields"] = list(source_key_fields)
    summary["source_split_label_counts"] = _source_split_label_counts(split_rows, source_key_fields)
    return summary


def source_label_balanced_grouped_split(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
    source_key_fields: Sequence[str] = ("source_kind",),
) -> Dict[str, List[Dict[str, Any]]]:
    """Run the label-balanced grouped splitter independently per source.

    A combined multi-source label-balanced split can satisfy rare-label
    validation coverage using only an external source. For ONC deployment
    checks we need each major source, especially ONC, to carry its own
    validation/test support when that source has enough groups.
    """

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[_source_key(row, source_key_fields)].append(dict(row))

    combined: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for source_key, source_rows in sorted(by_source.items()):
        source_split = label_balanced_grouped_split(
            source_rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=int(seed) + _stable_seed_offset(source_key),
        )
        for split in ("train", "val", "test"):
            combined[split].extend(source_split[split])
    return combined


def write_split_outputs(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    strategy: str = "temporal",
    seed: int = 0,
    source_key_fields: Sequence[str] = ("source_kind",),
) -> Dict[str, Any]:
    if strategy == "temporal":
        split_rows = temporal_grouped_split(rows, train_ratio=train_ratio, val_ratio=val_ratio)
    elif strategy == "label_balanced":
        split_rows = label_balanced_grouped_split(
            rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    elif strategy == "source_label_balanced":
        split_rows = source_label_balanced_grouped_split(
            rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            source_key_fields=source_key_fields,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    all_rows: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        all_rows.extend(split_rows[split])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "split_manifest.csv", all_rows)
    for split in ("train", "val", "test"):
        lines = []
        for row in split_rows[split]:
            identifier = clean_text(row.get("item_id")) or clean_text(row.get("mat_path"))
            lines.append(identifier)
        _split_text_path(output_dir, split).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    summary = _summarize(split_rows, source_key_fields=source_key_fields)
    summary["config"] = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "strategy": strategy,
        "seed": int(seed),
        "source_key_fields": list(source_key_fields),
    }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build grouped candidate splits for a multi-label manifest")
    parser.add_argument("--manifest-csv", required=True, help="Candidate multi-label manifest CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for split outputs")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--strategy", choices=["temporal", "label_balanced", "source_label_balanced"], default="temporal")
    parser.add_argument(
        "--source-key-fields",
        default="source_kind",
        help="Comma-separated fields used to partition source_label_balanced splits",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_csv_rows(Path(args.manifest_csv))
    summary = write_split_outputs(
        rows,
        Path(args.output_dir).resolve(),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        strategy=str(args.strategy),
        seed=int(args.seed),
        source_key_fields=[field.strip() for field in str(args.source_key_fields).split(",") if field.strip()],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
