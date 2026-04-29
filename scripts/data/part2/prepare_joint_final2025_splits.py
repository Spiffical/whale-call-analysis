#!/usr/bin/env python3
"""Build explicit joint historical + final-2025 training splits for ResNet benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.part2_finetune import (  # noqa: E402
    assign_time_pools,
    inventory_rows_from_dataset,
    load_finetune_clip_records_from_dataset,
    split_inventory_rows,
)


def _resolve_split_entry_path(
    raw_path: str,
    *,
    label: int,
    pos_dir: Path,
    neg_dir: Path,
) -> Path:
    path = Path(raw_path)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                (pos_dir / path.name) if int(label) == 1 else (neg_dir / path.name),
                pos_dir / path,
                neg_dir / path,
            ]
        )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve split entry path '{raw_path}' for label={label}")


def _load_split_entries(split_path: Path, *, pos_dir: Path, neg_dir: Path) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                raw_path, raw_label = line.split("\t", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid split line {split_path}:{line_no}: expected '<path>\\t<label>'")
                raw_path, raw_label = parts[0], parts[1]
            label = int(raw_label)
            resolved = _resolve_split_entry_path(raw_path, label=label, pos_dir=pos_dir, neg_dir=neg_dir)
            items.append((resolved, label))
    return items


def _count_rows(rows: Sequence[Tuple[Path, int]]) -> Dict[str, int]:
    pos = sum(1 for _, label in rows if int(label) == 1)
    neg = sum(1 for _, label in rows if int(label) == 0)
    return {"total": len(rows), "pos": pos, "neg": neg}


def _write_split_txt(path: Path, rows: Iterable[Tuple[Path, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for mat_path, label in rows:
            handle.write(f"{Path(mat_path).resolve()}\t{int(label)}\n")


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _dedupe_rows(rows: Sequence[Tuple[Path, int]]) -> List[Tuple[Path, int]]:
    out: List[Tuple[Path, int]] = []
    seen: set[Tuple[str, int]] = set()
    for mat_path, label in rows:
        key = (str(Path(mat_path).resolve()), int(label))
        if key in seen:
            continue
        seen.add(key)
        out.append((Path(key[0]), key[1]))
    return out


def _absolutize_inventory_rows(dataset_dir: Path, inventory_rows: Sequence[Dict[str, str]]) -> List[Tuple[Path, int]]:
    rows: List[Tuple[Path, int]] = []
    for row in inventory_rows:
        relative_path = str(row.get("relative_path", "")).strip()
        if not relative_path:
            continue
        abs_path = (dataset_dir / relative_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Sample inventory path does not exist: {abs_path}")
        rows.append((abs_path, int(str(row.get("label", "0")).strip() or "0")))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare explicit joint historical + final-2025 MAT splits")
    ap.add_argument("--historical-pos-dir", type=str, required=True)
    ap.add_argument("--historical-neg-dir", type=str, required=True)
    ap.add_argument("--historical-splits-dir", type=str, required=True)
    ap.add_argument("--part2-dataset-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--part2-train-ratio", type=float, default=0.7)
    ap.add_argument("--part2-val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    historical_pos_dir = Path(args.historical_pos_dir).resolve()
    historical_neg_dir = Path(args.historical_neg_dir).resolve()
    historical_splits_dir = Path(args.historical_splits_dir).resolve()
    part2_dataset_dir = Path(args.part2_dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    sample_inventory_csv = part2_dataset_dir / "sample_inventory.csv"
    call_inventory_csv = part2_dataset_dir / "call_inventory.csv"
    if not sample_inventory_csv.exists():
        raise SystemExit(f"Missing sample inventory: {sample_inventory_csv}")

    historical_train = _load_split_entries(
        historical_splits_dir / "train.txt",
        pos_dir=historical_pos_dir,
        neg_dir=historical_neg_dir,
    )
    historical_val = _load_split_entries(
        historical_splits_dir / "val.txt",
        pos_dir=historical_pos_dir,
        neg_dir=historical_neg_dir,
    )
    historical_test = _load_split_entries(
        historical_splits_dir / "test.txt",
        pos_dir=historical_pos_dir,
        neg_dir=historical_neg_dir,
    )

    part2_records = load_finetune_clip_records_from_dataset(
        sample_inventory_csv=sample_inventory_csv,
        call_inventory_csv=call_inventory_csv if call_inventory_csv.exists() else None,
    )
    split_map = assign_time_pools(
        part2_records,
        train_ratio=float(args.part2_train_ratio),
        val_ratio=float(args.part2_val_ratio),
    )
    inventory_rows = inventory_rows_from_dataset(sample_inventory_csv=sample_inventory_csv)
    part2_inventory_split_rows = split_inventory_rows(
        inventory_rows,
        train_clips={record.filename for record in split_map["train"]},
        val_clips={record.filename for record in split_map["val"]},
        test_clips={record.filename for record in split_map["test"]},
    )

    part2_train = _absolutize_inventory_rows(part2_dataset_dir, part2_inventory_split_rows["train"])
    part2_val = _absolutize_inventory_rows(part2_dataset_dir, part2_inventory_split_rows["val"])
    part2_test = _absolutize_inventory_rows(part2_dataset_dir, part2_inventory_split_rows["test"])

    joint_train = _dedupe_rows(list(historical_train) + list(part2_train))
    joint_val = _dedupe_rows(list(part2_val))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_split_txt(output_dir / "train.txt", joint_train)
    _write_split_txt(output_dir / "val.txt", joint_val)
    _write_split_txt(output_dir / "historical_train.txt", historical_train)
    _write_split_txt(output_dir / "historical_val.txt", historical_val)
    _write_split_txt(output_dir / "historical_test.txt", historical_test)
    _write_split_txt(output_dir / "part2_train.txt", part2_train)
    _write_split_txt(output_dir / "part2_val.txt", part2_val)
    _write_split_txt(output_dir / "part2_test.txt", part2_test)
    _write_lines(output_dir / "part2_train_clips.txt", sorted(record.filename for record in split_map["train"]))
    _write_lines(output_dir / "part2_val_clips.txt", sorted(record.filename for record in split_map["val"]))
    _write_lines(output_dir / "part2_test_clips.txt", sorted(record.filename for record in split_map["test"]))

    summary = {
        "historical": {
            "train": _count_rows(historical_train),
            "val": _count_rows(historical_val),
            "test": _count_rows(historical_test),
        },
        "part2": {
            "train": _count_rows(part2_train),
            "val": _count_rows(part2_val),
            "test": _count_rows(part2_test),
            "train_clip_count": len(split_map["train"]),
            "val_clip_count": len(split_map["val"]),
            "test_clip_count": len(split_map["test"]),
            "train_fin_clip_count": len(split_map["train_fin"]),
            "val_fin_clip_count": len(split_map["val_fin"]),
            "test_fin_clip_count": len(split_map["test_fin"]),
            "train_nonfin_clip_count": len(split_map["train_nonfin"]),
            "val_nonfin_clip_count": len(split_map["val_nonfin"]),
            "test_nonfin_clip_count": len(split_map["test_nonfin"]),
            "train_pure_negative_clip_count": len(split_map.get("train_pure_negative", [])),
            "val_pure_negative_clip_count": len(split_map.get("val_pure_negative", [])),
            "test_pure_negative_clip_count": len(split_map.get("test_pure_negative", [])),
        },
        "joint_training": {
            "train": _count_rows(joint_train),
            "val": _count_rows(joint_val),
        },
        "historical_pos_dir": str(historical_pos_dir),
        "historical_neg_dir": str(historical_neg_dir),
        "historical_splits_dir": str(historical_splits_dir),
        "part2_dataset_dir": str(part2_dataset_dir),
        "part2_train_ratio": float(args.part2_train_ratio),
        "part2_val_ratio": float(args.part2_val_ratio),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"joint_train_total={summary['joint_training']['train']['total']}")
    print(f"joint_val_total={summary['joint_training']['val']['total']}")
    print(f"part2_test_clip_count={summary['part2']['test_clip_count']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
