#!/usr/bin/env python3
"""Build split-manifest variants for ONC calibration experiments.

The 40s MAT archive is expensive to build, but training manifests are cheap.
This utility rewrites an already-split archive manifest by optionally capping
external-source train rows and duplicating train-set ONC rare-label rows. The
validation and test rows are copied unchanged unless the caller filters the
input before invoking this script.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import clean_text, label_ids_from_row, read_csv_rows, write_csv_rows  # noqa: E402

BACKGROUND_LABEL = "<background>"


def _label_key(row: Mapping[str, Any]) -> str:
    labels = label_ids_from_row(row)
    if not labels:
        return BACKGROUND_LABEL
    return "|".join(labels)


def _parse_source_label_spec(value: str, *, value_name: str) -> Tuple[str, str, int]:
    parts = [part.strip() for part in str(value).split(":")]
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(f"{value_name} must be SOURCE:LABEL:N, got {value!r}")
    source = parts[0]
    amount_text = parts[-1]
    label = ":".join(parts[1:-1])
    if not source or not label:
        raise argparse.ArgumentTypeError(f"{value_name} must be SOURCE:LABEL:N, got {value!r}")
    try:
        amount = int(amount_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value_name} count must be an integer: {value!r}") from exc
    if amount < 0:
        raise argparse.ArgumentTypeError(f"{value_name} count must be non-negative: {value!r}")
    return source, label, amount


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return clean_text(row.get("source_kind")) or "<blank>", _label_key(row)


def _summarize(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    split_counts: Counter[str] = Counter()
    source_split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    source_split_label_counts: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    oversampled_count = 0
    for row in rows:
        split = clean_text(row.get("split")) or "<blank>"
        source = clean_text(row.get("source_kind")) or "<blank>"
        label = _label_key(row)
        split_counts[split] += 1
        source_split_counts[source][split] += 1
        source_split_label_counts[split][source][label] += 1
        if clean_text(row.get("is_oversampled")) == "1":
            oversampled_count += 1
    return {
        "row_count": sum(split_counts.values()),
        "split_counts": dict(split_counts.most_common()),
        "source_split_counts": {
            source: dict(counts.most_common())
            for source, counts in sorted(source_split_counts.items())
        },
        "source_split_label_counts": {
            split: {
                source: dict(counts.most_common())
                for source, counts in sorted(source_counts.items())
            }
            for split, source_counts in sorted(source_split_label_counts.items())
        },
        "oversampled_row_count": oversampled_count,
    }


def _copy_row(row: Mapping[str, Any], *, oversampled: bool = False) -> Dict[str, Any]:
    out = dict(row)
    out.setdefault("is_oversampled", "0")
    out.setdefault("oversample_policy", "")
    out.setdefault("oversample_source_item_id", "")
    if oversampled:
        out["is_oversampled"] = "1"
    return out


def _apply_train_caps(
    rows: Sequence[Mapping[str, Any]],
    caps: Mapping[Tuple[str, str], int],
    *,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not caps:
        return [_copy_row(row) for row in rows], {"dropped_row_count": 0, "caps": {}}

    train_rows: List[Mapping[str, Any]] = []
    passthrough_rows: List[Mapping[str, Any]] = []
    for row in rows:
        if clean_text(row.get("split")) == "train":
            train_rows.append(row)
        else:
            passthrough_rows.append(row)

    shuffled = list(train_rows)
    rng.shuffle(shuffled)
    kept_train: List[Dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    kept_by_key: Counter[Tuple[str, str]] = Counter()
    for row in shuffled:
        key = _row_key(row)
        cap = caps.get(key)
        if cap is not None and kept_by_key[key] >= cap:
            dropped[f"{key[0]}:{key[1]}"] += 1
            continue
        kept_by_key[key] += 1
        kept_train.append(_copy_row(row))

    kept_train.sort(key=lambda row: (clean_text(row.get("source_kind")), _label_key(row), clean_text(row.get("item_id")), clean_text(row.get("mat_path"))))
    passthrough = [_copy_row(row) for row in passthrough_rows]
    return kept_train + passthrough, {
        "dropped_row_count": int(sum(dropped.values())),
        "dropped_by_cap": dict(dropped.most_common()),
        "kept_by_capped_key": {f"{source}:{label}": count for (source, label), count in sorted(kept_by_key.items()) if (source, label) in caps},
        "caps": {f"{source}:{label}": cap for (source, label), cap in sorted(caps.items())},
    }


def _oversample_train_labels(
    rows: Sequence[Mapping[str, Any]],
    targets: Mapping[Tuple[str, str], int],
    *,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = [_copy_row(row) for row in rows]
    if not targets:
        return output, {"added_row_count": 0, "targets": {}}

    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in output:
        if clean_text(row.get("split")) == "train":
            by_key[_row_key(row)].append(row)

    added_rows: List[Dict[str, Any]] = []
    target_summary: Dict[str, Dict[str, Any]] = {}
    for key, target in sorted(targets.items()):
        candidates = list(by_key.get(key, []))
        key_text = f"{key[0]}:{key[1]}"
        current = len(candidates)
        if not candidates:
            target_summary[key_text] = {"original_count": 0, "target_count": int(target), "added_count": 0, "status": "no_candidates"}
            continue
        if current >= target:
            target_summary[key_text] = {"original_count": current, "target_count": int(target), "added_count": 0, "status": "already_at_or_above_target"}
            continue
        rng.shuffle(candidates)
        need = target - current
        for idx in range(need):
            source = candidates[idx % len(candidates)]
            dup = dict(source)
            base_item = clean_text(source.get("item_id")) or clean_text(source.get("mat_path")) or f"{key_text}_{idx}"
            dup["item_id"] = f"{base_item}__oversample_{idx + 1:05d}"
            dup["is_oversampled"] = "1"
            dup["oversample_policy"] = f"target_train_count={target}"
            dup["oversample_source_item_id"] = base_item
            added_rows.append(dup)
        target_summary[key_text] = {
            "original_count": current,
            "target_count": int(target),
            "added_count": int(need),
            "status": "oversampled",
        }
    output.extend(added_rows)
    return output, {
        "added_row_count": len(added_rows),
        "targets": {f"{source}:{label}": target for (source, label), target in sorted(targets.items())},
        "target_summary": target_summary,
    }


def build_variant(
    *,
    manifest_csv: Path,
    output_dir: Path,
    variant_name: str,
    train_caps: Mapping[Tuple[str, str], int],
    oversample_targets: Mapping[Tuple[str, str], int],
    seed: int,
    vocab_json: Path | None = None,
) -> Dict[str, Any]:
    rows = read_csv_rows(manifest_csv)
    rng = random.Random(seed)
    capped_rows, cap_summary = _apply_train_caps(rows, train_caps, rng=rng)
    variant_rows, oversample_summary = _oversample_train_labels(capped_rows, oversample_targets, rng=rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = output_dir / "standardized_manifest.csv"
    write_csv_rows(manifest_out, variant_rows)
    if vocab_json is not None:
        shutil.copy2(vocab_json, output_dir / "label_vocabulary.json")
    summary = {
        "variant_name": variant_name,
        "input_manifest": str(manifest_csv),
        "output_manifest": str(manifest_out),
        "input_summary": _summarize(rows),
        "output_summary": _summarize(variant_rows),
        "cap_summary": cap_summary,
        "oversample_summary": oversample_summary,
        "seed": int(seed),
    }
    (output_dir / "manifest_variant_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an ONC calibration training-manifest variant")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--vocab-json", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--train-source-label-cap",
        action="append",
        default=[],
        help="Cap train rows for an exact source/label key: SOURCE:LABEL:N. Use <background> for unlabeled rows.",
    )
    parser.add_argument(
        "--oversample-train-source-label",
        action="append",
        default=[],
        help="Duplicate train rows until an exact source/label key reaches N rows: SOURCE:LABEL:N.",
    )
    args = parser.parse_args()

    caps = {
        (source, label): amount
        for source, label, amount in (
            _parse_source_label_spec(value, value_name="--train-source-label-cap")
            for value in args.train_source_label_cap
        )
    }
    targets = {
        (source, label): amount
        for source, label, amount in (
            _parse_source_label_spec(value, value_name="--oversample-train-source-label")
            for value in args.oversample_train_source_label
        )
    }
    summary = build_variant(
        manifest_csv=Path(args.manifest_csv),
        output_dir=Path(args.output_dir),
        variant_name=str(args.variant_name),
        train_caps=caps,
        oversample_targets=targets,
        seed=int(args.seed),
        vocab_json=Path(args.vocab_json) if args.vocab_json else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
