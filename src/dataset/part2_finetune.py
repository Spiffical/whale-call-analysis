"""Helpers for Part 2 fine-tuning and learning-curve experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .part2_annotations import FIN_SPECIES_CODE, parse_filename_timestamp


@dataclass(frozen=True)
class FineTuneClipRecord:
    filename: str
    timestamp: datetime
    fin_call_count: int
    fin_call_type_buckets: Tuple[str, ...]
    context_tags: Tuple[str, ...]
    is_fin_positive: bool
    is_annotated_non_fin: bool
    is_pure_negative_candidate: bool


def _split_pipe(raw: Any) -> Tuple[str, ...]:
    tokens = [token.strip() for token in str(raw or "").split("|") if token.strip()]
    return tuple(sorted(dict.fromkeys(tokens)))


def _read_csv(path: Path | str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(raw: Any) -> bool:
    value = str(raw or "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def load_finetune_clip_records(
    *,
    fin_annotations_csv: Path | str,
    clip_manifest_csv: Path | str,
) -> List[FineTuneClipRecord]:
    fin_counts: Dict[str, int] = {}
    for row in _read_csv(fin_annotations_csv):
        if str(row.get("species", "")).strip() != FIN_SPECIES_CODE:
            continue
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        fin_counts[filename] = fin_counts.get(filename, 0) + 1

    rows: List[FineTuneClipRecord] = []
    for row in _read_csv(clip_manifest_csv):
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        timestamp = parse_filename_timestamp(filename)
        if timestamp is None:
            continue
        rows.append(
            FineTuneClipRecord(
                filename=filename,
                timestamp=timestamp,
                fin_call_count=int(fin_counts.get(filename, 0)),
                fin_call_type_buckets=_split_pipe(row.get("fin_call_type_buckets", "")),
                context_tags=_split_pipe(row.get("context_tags", "")),
                is_fin_positive=str(row.get("is_fin_positive", "0")).strip() == "1",
                is_annotated_non_fin=str(row.get("is_annotated_non_fin", "0")).strip() == "1",
                is_pure_negative_candidate=str(row.get("is_pure_negative_candidate", "0")).strip() == "1",
            )
        )
    rows.sort(key=lambda item: (item.timestamp, item.filename))
    return rows


def load_finetune_clip_records_from_dataset(
    *,
    sample_inventory_csv: Path | str,
    call_inventory_csv: Path | str | None = None,
) -> List[FineTuneClipRecord]:
    clip_rows: Dict[str, Dict[str, Any]] = {}

    for row in _read_csv(sample_inventory_csv):
        filename = str(row.get("source_audio", "") or row.get("filename", "")).strip()
        if not filename:
            continue
        timestamp = parse_filename_timestamp(filename)
        if timestamp is None:
            continue
        state = clip_rows.setdefault(
            filename,
            {
                "timestamp": timestamp,
                "fin_call_count": 0,
                "fin_call_type_buckets": set(),
                "context_tags": set(),
                "is_fin_positive": False,
                "is_annotated_non_fin": False,
                "is_pure_negative_candidate": False,
            },
        )
        state["context_tags"].update(_split_pipe(row.get("context_tags", "")))
        state["is_fin_positive"] = bool(state["is_fin_positive"]) or _as_bool(row.get("is_fin_positive", "0"))
        state["is_annotated_non_fin"] = bool(state["is_annotated_non_fin"]) or _as_bool(
            row.get("is_annotated_non_fin", "0")
        )
        state["is_pure_negative_candidate"] = bool(state["is_pure_negative_candidate"]) or _as_bool(
            row.get("is_pure_negative_candidate", "0")
        )

    if call_inventory_csv is not None and Path(call_inventory_csv).exists():
        for row in _read_csv(call_inventory_csv):
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            timestamp = parse_filename_timestamp(filename)
            if timestamp is None:
                continue
            state = clip_rows.setdefault(
                filename,
                {
                    "timestamp": timestamp,
                    "fin_call_count": 0,
                    "fin_call_type_buckets": set(),
                    "context_tags": set(),
                    "is_fin_positive": False,
                    "is_annotated_non_fin": False,
                    "is_pure_negative_candidate": False,
                },
            )
            state["fin_call_count"] = int(state["fin_call_count"]) + 1
            state["fin_call_type_buckets"].update(_split_pipe(row.get("call_type_bucket", "")))
            state["context_tags"].update(_split_pipe(row.get("context_tags", "")))
            state["is_fin_positive"] = True
    else:
        # Fallback for older dataset exports where only sample_inventory.csv is available.
        for row in _read_csv(sample_inventory_csv):
            filename = str(row.get("source_audio", "") or row.get("filename", "")).strip()
            if not filename or not _as_bool(row.get("is_fin_positive", "0")):
                continue
            state = clip_rows.get(filename)
            if state is None:
                continue
            state["fin_call_count"] = int(state["fin_call_count"]) + 1
            state["fin_call_type_buckets"].update(_split_pipe(row.get("call_type_bucket", "")))

    records: List[FineTuneClipRecord] = []
    for filename, state in clip_rows.items():
        records.append(
            FineTuneClipRecord(
                filename=filename,
                timestamp=state["timestamp"],
                fin_call_count=int(state["fin_call_count"]),
                fin_call_type_buckets=tuple(sorted(state["fin_call_type_buckets"])),
                context_tags=tuple(sorted(state["context_tags"])),
                is_fin_positive=bool(state["is_fin_positive"]),
                is_annotated_non_fin=bool(state["is_annotated_non_fin"]),
                is_pure_negative_candidate=bool(state["is_pure_negative_candidate"]),
            )
        )
    records.sort(key=lambda item: (item.timestamp, item.filename))
    return records


def compute_time_boundaries(
    fin_positive_records: Sequence[FineTuneClipRecord],
    *,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Optional[datetime]]:
    if not fin_positive_records:
        return {"train_end": None, "val_end": None}
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1")

    ordered = sorted(fin_positive_records, key=lambda item: (item.timestamp, item.filename))
    n_total = len(ordered)
    train_end_index = max(0, min(n_total - 1, int(train_ratio * n_total) - 1))
    val_end_index = max(train_end_index, min(n_total - 1, int((train_ratio + val_ratio) * n_total) - 1))
    return {
        "train_end": ordered[train_end_index].timestamp,
        "val_end": ordered[val_end_index].timestamp,
    }


def assign_time_pools(
    records: Sequence[FineTuneClipRecord],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Dict[str, List[FineTuneClipRecord]]:
    def _assign_subset(subset: Sequence[FineTuneClipRecord]) -> Dict[str, List[FineTuneClipRecord]]:
        boundaries = compute_time_boundaries(subset, train_ratio=train_ratio, val_ratio=val_ratio)
        train_end = boundaries["train_end"]
        val_end = boundaries["val_end"]
        subset_split: Dict[str, List[FineTuneClipRecord]] = {"train": [], "val": [], "test": []}
        for record in sorted(subset, key=lambda item: (item.timestamp, item.filename)):
            if train_end is None or record.timestamp <= train_end:
                subset_split["train"].append(record)
            elif val_end is None or record.timestamp <= val_end:
                subset_split["val"].append(record)
            else:
                subset_split["test"].append(record)
        return subset_split

    fin_positive = [record for record in records if record.is_fin_positive]
    nonfin_only = [
        record
        for record in records
        if record.is_annotated_non_fin and not record.is_fin_positive and not record.is_pure_negative_candidate
    ]
    pure_negative_only = [
        record
        for record in records
        if record.is_pure_negative_candidate and not record.is_fin_positive
    ]
    fin_split = _assign_subset(fin_positive)
    nonfin_split = _assign_subset(nonfin_only)

    split_map: Dict[str, List[FineTuneClipRecord]] = {}
    for split_name in ("train", "val", "test"):
        pure_negative_split = list(pure_negative_only) if split_name == "train" else []
        split_map[f"{split_name}_fin"] = list(fin_split[split_name])
        split_map[f"{split_name}_annotated_nonfin"] = list(nonfin_split[split_name])
        split_map[f"{split_name}_pure_negative"] = pure_negative_split
        split_map[f"{split_name}_nonfin"] = sorted(
            list(nonfin_split[split_name]) + pure_negative_split,
            key=lambda item: (item.timestamp, item.filename),
        )
        split_map[split_name] = sorted(
            fin_split[split_name] + split_map[f"{split_name}_nonfin"],
            key=lambda item: (item.timestamp, item.filename),
        )
    return split_map


def _month_round_robin_order(
    records: Sequence[FineTuneClipRecord],
    *,
    seed: int,
) -> List[FineTuneClipRecord]:
    month_groups: Dict[str, List[FineTuneClipRecord]] = {}
    for record in records:
        month_key = record.timestamp.strftime("%Y%m")
        month_groups.setdefault(month_key, []).append(record)

    rng = random.Random(seed)
    months = sorted(month_groups)
    for month in months:
        rng.shuffle(month_groups[month])

    ordered: List[FineTuneClipRecord] = []
    progress = True
    while progress:
        progress = False
        for month in months:
            group = month_groups[month]
            if not group:
                continue
            ordered.append(group.pop(0))
            progress = True
    return ordered


def order_train_pool(
    records: Sequence[FineTuneClipRecord],
    *,
    sampling_mode: str,
    seed: int = 0,
) -> List[FineTuneClipRecord]:
    mode = str(sampling_mode).strip().lower()
    ordered = list(sorted(records, key=lambda item: (item.timestamp, item.filename)))
    if mode == "chronological":
        return ordered
    if mode == "random_clip":
        rng = random.Random(seed)
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        return shuffled
    if mode == "month_stratified_clip":
        return _month_round_robin_order(ordered, seed=seed)
    raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")


def select_budget_clips(
    train_pool: Sequence[FineTuneClipRecord],
    *,
    budget_calls: int,
    sampling_mode: str,
    seed: int = 0,
) -> List[FineTuneClipRecord]:
    if budget_calls <= 0:
        raise ValueError("budget_calls must be > 0")
    ordered = order_train_pool(train_pool, sampling_mode=sampling_mode, seed=seed)
    selected: List[FineTuneClipRecord] = []
    total_calls = 0
    for record in ordered:
        if record.fin_call_count <= 0:
            continue
        selected.append(record)
        total_calls += int(record.fin_call_count)
        if total_calls >= int(budget_calls):
            break
    return selected


def clip_rollup(records: Sequence[FineTuneClipRecord]) -> Dict[str, Any]:
    bucket_counts: Dict[str, int] = {}
    context_counts: Dict[str, int] = {}
    for record in records:
        for bucket in record.fin_call_type_buckets:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        for tag in record.context_tags:
            context_counts[tag] = context_counts.get(tag, 0) + 1
    return {
        "clip_count": len(records),
        "fin_call_count": sum(int(record.fin_call_count) for record in records),
        "months": sorted({record.timestamp.strftime("%Y%m") for record in records}),
        "bucket_clip_counts": dict(sorted(bucket_counts.items())),
        "context_clip_counts": dict(sorted(context_counts.items())),
        "first_timestamp": records[0].timestamp.isoformat() if records else None,
        "last_timestamp": records[-1].timestamp.isoformat() if records else None,
    }


def inventory_rows_from_dataset(
    *,
    sample_inventory_csv: Path | str,
) -> List[Dict[str, str]]:
    rows = _read_csv(sample_inventory_csv)
    for row in rows:
        row["label"] = str(row.get("label", "")).strip()
        row["source_audio"] = str(row.get("source_audio", "")).strip()
        row["relative_path"] = str(row.get("relative_path", "")).strip()
    return rows


def split_inventory_rows(
    inventory_rows: Sequence[Dict[str, str]],
    *,
    train_clips: Iterable[str],
    val_clips: Iterable[str],
    test_clips: Iterable[str],
) -> Dict[str, List[Dict[str, str]]]:
    clip_sets = {
        "train": set(train_clips),
        "val": set(val_clips),
        "test": set(test_clips),
    }
    split_rows: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in inventory_rows:
        source_audio = str(row.get("source_audio", "")).strip()
        if source_audio in clip_sets["train"]:
            split_rows["train"].append(dict(row))
        elif source_audio in clip_sets["val"]:
            split_rows["val"].append(dict(row))
        elif source_audio in clip_sets["test"]:
            split_rows["test"].append(dict(row))
    return split_rows


def build_learning_curve_plan(
    *,
    records: Sequence[FineTuneClipRecord],
    budgets: Sequence[int],
    sampling_modes: Sequence[str],
    repeats: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    base_seed: int = 1337,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[FineTuneClipRecord]]]:
    split_map = assign_time_pools(records, train_ratio=train_ratio, val_ratio=val_ratio)
    train_pool = list(split_map["train_fin"])
    train_nonfin_pool = list(split_map["train_nonfin"])
    val_pool = list(split_map["val"])
    test_pool = list(split_map["test"])

    rows: List[Dict[str, Any]] = []
    for sampling_mode in sampling_modes:
        n_repeats = 1 if sampling_mode == "chronological" else max(1, int(repeats))
        for repeat_index in range(n_repeats):
            seed = int(base_seed) + repeat_index
            for budget_calls in budgets:
                selected_train = select_budget_clips(
                    train_pool,
                    budget_calls=int(budget_calls),
                    sampling_mode=sampling_mode,
                    seed=seed,
                )
                selected_train_clip_names = {record.filename for record in selected_train}
                # Keep the non-fin training background fixed across budgets so the
                # learning-curve question isolates how many new fin-whale calls are needed.
                selected_train_nonfin = list(train_nonfin_pool)
                run_id = (
                    f"{sampling_mode}_calls{int(budget_calls):05d}_"
                    f"rep{int(repeat_index):02d}"
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "sampling_mode": sampling_mode,
                        "repeat_index": int(repeat_index),
                        "seed": int(seed),
                        "target_budget_calls": int(budget_calls),
                        "actual_budget_calls": sum(record.fin_call_count for record in selected_train),
                        "train_fin_clip_count": len(selected_train),
                        "train_nonfin_clip_count": len(selected_train_nonfin),
                        "val_fin_clip_count": len(split_map["val_fin"]),
                        "val_nonfin_clip_count": len(split_map["val_nonfin"]),
                        "val_clip_count": len(val_pool),
                        "test_fin_clip_count": len(split_map["test_fin"]),
                        "test_nonfin_clip_count": len(split_map["test_nonfin"]),
                        "test_clip_count": len(test_pool),
                        "train_last_timestamp": max((record.timestamp for record in selected_train), default=None).isoformat() if selected_train else "",
                        "train_fin_clip_names": "|".join(sorted(selected_train_clip_names)),
                        "train_nonfin_clip_names": "|".join(sorted(record.filename for record in selected_train_nonfin)),
                    }
                )
    return rows, split_map
