#!/usr/bin/env python3
"""Prepare clip-based learning-curve splits for Part 2 fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.part2_finetune import (
    build_learning_curve_plan,
    clip_rollup,
    inventory_rows_from_dataset,
    load_finetune_clip_records,
    split_inventory_rows,
)


def _parse_int_list(raw: str) -> List[int]:
    values = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return [int(token) for token in values]


def _parse_str_list(raw: str) -> List[str]:
    values = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one string value")
    return values


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _write_split_txt(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    lines = [f"{row['relative_path']}\t{row['label']}" for row in rows if row.get("relative_path")]
    _write_lines(path, lines)


def _filter_rows(rows: Sequence[Dict[str, str]], allowed_clip_names: set[str]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("filename", "")).strip() in allowed_clip_names]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare clip-based learning-curve splits for Part 2 fine-tuning")
    ap.add_argument("--dataset-dir", type=str, required=True, help="Fine-tune dataset directory with sample_inventory.csv")
    ap.add_argument("--fin-annotations-csv", type=str, required=True)
    ap.add_argument("--clip-manifest-csv", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--budgets", type=str, default="25,50,100,250,500,1000,2500,5000,10000")
    ap.add_argument("--sampling-modes", type=str, default="chronological,month_stratified_clip")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    sample_inventory_csv = dataset_dir / "sample_inventory.csv"
    if not sample_inventory_csv.exists():
        raise SystemExit(f"Missing sample inventory: {sample_inventory_csv}")

    records = load_finetune_clip_records(
        fin_annotations_csv=args.fin_annotations_csv,
        clip_manifest_csv=args.clip_manifest_csv,
    )
    plan_rows, split_map = build_learning_curve_plan(
        records=records,
        budgets=_parse_int_list(args.budgets),
        sampling_modes=_parse_str_list(args.sampling_modes),
        repeats=int(args.repeats),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        base_seed=int(args.seed),
    )
    inventory_rows = inventory_rows_from_dataset(sample_inventory_csv=sample_inventory_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    val_clip_names = {record.filename for record in split_map["val"] if record.is_fin_positive or record.is_annotated_non_fin}
    test_clip_names = {record.filename for record in split_map["test"] if record.is_fin_positive or record.is_annotated_non_fin}

    for run in plan_rows:
        run_id = str(run["run_id"])
        split_dir = output_dir / "runs" / run_id
        train_fin_clip_names = {
            token for token in str(run.get("train_fin_clip_names", "")).split("|") if token
        }
        train_nonfin_clip_names = {
            token for token in str(run.get("train_nonfin_clip_names", "")).split("|") if token
        }
        train_clip_names = train_fin_clip_names | train_nonfin_clip_names

        split_rows = split_inventory_rows(
            inventory_rows,
            train_clips=train_clip_names,
            val_clips=val_clip_names,
            test_clips=test_clip_names,
        )
        for split_name, rows in split_rows.items():
            _write_split_txt(split_dir / f"{split_name}.txt", rows)

        train_records = [record for record in split_map["train"] if record.filename in train_fin_clip_names]
        val_records = [record for record in split_map["val"] if record.filename in val_clip_names]
        test_records = [record for record in split_map["test"] if record.filename in test_clip_names]

        run_summary = {
            "run_id": run_id,
            "sampling_mode": run["sampling_mode"],
            "repeat_index": int(run["repeat_index"]),
            "seed": int(run["seed"]),
            "target_budget_calls": int(run["target_budget_calls"]),
            "actual_budget_calls": int(run["actual_budget_calls"]),
            "train_fin_clip_count": int(run["train_fin_clip_count"]),
            "train_nonfin_clip_count": int(run["train_nonfin_clip_count"]),
            "train_total_clip_count": int(run["train_fin_clip_count"]) + int(run["train_nonfin_clip_count"]),
            "train_rollup": clip_rollup(train_records),
            "val_rollup": clip_rollup(val_records),
            "test_rollup": clip_rollup(test_records),
            "sample_counts": {
                split_name: {
                    "total": len(rows),
                    "pos": sum(1 for row in rows if str(row.get("label", "")) == "1"),
                    "neg": sum(1 for row in rows if str(row.get("label", "")) == "0"),
                }
                for split_name, rows in split_rows.items()
            },
        }
        with open(split_dir / "run_summary.json", "w", encoding="utf-8") as handle:
            json.dump(run_summary, handle, indent=2, sort_keys=True)
        _write_lines(split_dir / "train_fin_clips.txt", sorted(train_fin_clip_names))
        _write_lines(split_dir / "train_nonfin_clips.txt", sorted(train_nonfin_clip_names))
        _write_lines(split_dir / "train_all_clips.txt", sorted(train_clip_names))

    _write_csv(output_dir / "learning_curve_plan.csv", plan_rows)

    with open(args.fin_annotations_csv, "r", encoding="utf-8", newline="") as handle:
        fin_annotation_rows = _filter_rows(list(csv.DictReader(handle)), test_clip_names)
    with open(args.clip_manifest_csv, "r", encoding="utf-8", newline="") as handle:
        clip_manifest_rows = _filter_rows(list(csv.DictReader(handle)), test_clip_names)
    eval_dir = output_dir / "part2_eval_test"
    _write_csv(eval_dir / "fin_annotations.csv", fin_annotation_rows)
    _write_csv(eval_dir / "clip_manifest.csv", clip_manifest_rows)
    _write_lines(eval_dir / "test_clips.txt", sorted(test_clip_names))
    _write_lines(eval_dir / "val_clips.txt", sorted(val_clip_names))

    summary = {
        "dataset_dir": str(dataset_dir),
        "plan_csv": str(output_dir / "learning_curve_plan.csv"),
        "run_count": len(plan_rows),
        "sampling_modes": _parse_str_list(args.sampling_modes),
        "budgets": _parse_int_list(args.budgets),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "val_clip_count": len(val_clip_names),
        "test_clip_count": len(test_clip_names),
        "part2_eval_test_dir": str(eval_dir),
    }
    with open(output_dir / "learning_curve_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
