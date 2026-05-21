#!/usr/bin/env python3
"""Build E20 diagnostic manifests for staged multiband failure analysis."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PRIMARY_LABELS = ("species:Bp", "species:Bm", "species:Mn", "species:Oo")


VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "E20_bp_mn_lowmid_cumulative",
        "description": "Pairwise Bp+Mn low+mid cumulative interference probe.",
        "active_label_ids": ["species:Bp", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "none",
    },
    {
        "name": "E20_bm_mn_lowmid_cumulative",
        "description": "Pairwise Bm+Mn low+mid cumulative interference probe.",
        "active_label_ids": ["species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "none",
    },
    {
        "name": "E20_bp_bm_mn_lowmid_labelcap",
        "description": "Bp+Bm+Mn low+mid cumulative probe with train-positive label caps.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "label_train_cap",
    },
    {
        "name": "E20_bp_bm_mn_lowmid_sourcecap",
        "description": "Bp+Bm+Mn low+mid cumulative probe with external source-label train caps.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "source_label_train_cap",
    },
    {
        "name": "E20_full_allbands_nomask",
        "description": "Full low+mid+high routed fusion with no class-band mask.",
        "active_label_ids": list(PRIMARY_LABELS),
        "eval_label_ids": list(PRIMARY_LABELS),
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid", "high"],
        "cap_strategy": "none",
    },
    {
        "name": "E20_full_allbands_labelcap",
        "description": "Full low+mid+high routed fusion with train-positive label caps.",
        "active_label_ids": list(PRIMARY_LABELS),
        "eval_label_ids": list(PRIMARY_LABELS),
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid", "high"],
        "cap_strategy": "label_train_cap",
    },
    {
        "name": "E20_oo_mid_only",
        "description": "Oo-only mid-band localization/domain probe.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid"],
        "cap_strategy": "none",
    },
    {
        "name": "E20_oo_high_only",
        "description": "Oo-only high-band localization/domain probe.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["high"],
        "cap_strategy": "none",
    },
    {
        "name": "E20_oo_allbands",
        "description": "Oo-only low+mid+high single-target probe.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["low", "mid", "high"],
        "cap_strategy": "none",
    },
    {
        "name": "E20_oo_midhigh_wide20",
        "description": "Oo-only mid+high probe with a 20s crop.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "crop_time_seconds": 20,
    },
    {
        "name": "E20_oo_midhigh_full40",
        "description": "Oo-only mid+high probe with the full 40s context crop.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "crop_time_seconds": 40,
    },
    {
        "name": "E20_oo_onc_only_midhigh",
        "description": "Oo-only ONC-only mid+high probe.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
    },
    {
        "name": "E20_oo_dclde_only_midhigh",
        "description": "Oo-only DCLDE-only mid+high in-domain source-fit probe.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "DCLDE",
        "eval_source_kind": "DCLDE",
    },
]


SOURCE_LABEL_TRAIN_CAPS = {
    ("BioDCASE", "species:Bm"): 8000,
    ("BioDCASE", "species:Bp"): 8000,
    ("BioDCASE", "<background>"): 1000,
    ("DCLDE", "species:Mn"): 3000,
    ("DCLDE", "species:Oo"): 3000,
    ("DCLDE", "<background>"): 3000,
}


DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)"),
    re.compile(r"(20\d{2})[-_]?([01]\d)"),
]


def clean(value: Any) -> str:
    return str(value or "").strip()


def split_pipe(value: Any) -> List[str]:
    return [token.strip() for token in clean(value).split("|") if token.strip()]


def is_truthy(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def labels(row: Mapping[str, Any]) -> Tuple[str, ...]:
    for key in ("label_ids", "target_label_ids", "canonical_label_ids", "analysis_label_ids", "source_label_ids"):
        value = clean(row.get(key))
        if value:
            return tuple(split_pipe(value))
    return tuple()


def label_key(row: Mapping[str, Any]) -> str:
    labs = labels(row)
    return "|".join(labs) if labs else "<background>"


def active_label_key(row: Mapping[str, Any], active_label_ids: Sequence[str]) -> str:
    active = set(active_label_ids)
    labs = [label for label in labels(row) if label in active]
    return labs[0] if labs else "<background>"


def month_bin(row: Mapping[str, Any]) -> str:
    text = " ".join(
        clean(row.get(key))
        for key in ("clip", "source_audio", "item_id", "mat_path", "expected_mat_name")
    )
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
    return "<unknown>"


def source_audio_basename(row: Mapping[str, Any]) -> str:
    for key in ("clip", "source_audio", "filename", "source_soundfile", "item_id"):
        text = clean(row.get(key))
        if text:
            return Path(text).name
    return clean(row.get("item_id")) or "<missing>"


def source_date_key(row: Mapping[str, Any]) -> str:
    text = " ".join(
        clean(row.get(key))
        for key in ("clip", "source_audio", "filename", "source_soundfile", "item_id", "mat_path")
    )
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", text)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    month = month_bin(row)
    return month if month != "<unknown>" else source_audio_basename(row)


def split_group_key(row: Mapping[str, Any], mode: str) -> str:
    source = clean(row.get("source_kind")) or "<source>"
    if mode == "source_audio":
        return f"{source}:{source_audio_basename(row)}"
    if mode == "source_date":
        return f"{source}:{source_date_key(row)}"
    if mode == "event_group":
        return f"{source}:{clean(row.get('event_group')) or source_audio_basename(row)}"
    raise ValueError(f"Unknown split_grouping mode {mode!r}")


def reassign_splits_by_group(
    rows: Sequence[Dict[str, str]],
    *,
    mode: str,
    source_kinds: Sequence[str],
    stratify_label_ids: Sequence[str] = (),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    selected_sources = {clean(source) for source in source_kinds if clean(source)}
    target_rows = [
        dict(row)
        for row in rows
        if not selected_sources or clean(row.get("source_kind")) in selected_sources
    ]
    untouched_rows = [
        dict(row)
        for row in rows
        if selected_sources and clean(row.get("source_kind")) not in selected_sources
    ]
    if not target_rows:
        return list(rows), {
            "mode": mode,
            "source_kinds": sorted(selected_sources),
            "reason": "no_matching_rows",
        }

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in target_rows:
        groups[split_group_key(row, mode)].append(row)

    ordered = sorted(
        groups.items(),
        key=lambda item: (
            min(month_bin(row) for row in item[1]),
            item[0],
        ),
    )

    def desired_counts(n_groups: int) -> Dict[str, int]:
        if n_groups == 1:
            return {"train": 1, "val": 0, "test": 0}
        n_train = max(1, int(n_groups * float(train_ratio)))
        n_val = int(n_groups * float(val_ratio))
        if n_groups >= 3 and n_val < 1:
            n_val = 1
        if n_train + n_val >= n_groups:
            n_train = max(1, n_groups - n_val - 1)
        return {
            "train": n_train,
            "val": n_val,
            "test": n_groups - n_train - n_val,
        }

    n_groups = len(ordered)
    split_group_counts = desired_counts(n_groups)
    stratify_labels = {clean(label) for label in stratify_label_ids if clean(label)}
    positive_group_counts = {"train": 0, "val": 0, "test": 0}
    assigned: Dict[str, str] = {}
    if stratify_labels:
        positive = [
            (key, group_rows)
            for key, group_rows in ordered
            if any(stratify_labels.intersection(labels(row)) for row in group_rows)
        ]
        negative = [(key, group_rows) for key, group_rows in ordered if key not in {item[0] for item in positive}]
        positive_counts = desired_counts(len(positive)) if positive else {"train": 0, "val": 0, "test": 0}
        cursor = 0
        for split in ("train", "val", "test"):
            for key, _ in positive[cursor : cursor + positive_counts[split]]:
                assigned[key] = split
                positive_group_counts[split] += 1
            cursor += positive_counts[split]

        remaining_counts = {
            split: max(0, split_group_counts[split] - positive_counts.get(split, 0))
            for split in ("train", "val", "test")
        }
        cursor = 0
        for split in ("train", "val", "test"):
            for key, _ in negative[cursor : cursor + remaining_counts[split]]:
                assigned[key] = split
            cursor += remaining_counts[split]
        for key, _ in negative[cursor:]:
            assigned[key] = min(("train", "val", "test"), key=lambda split: sum(1 for value in assigned.values() if value == split))
    else:
        cursor = 0
        for split in ("train", "val", "test"):
            for key, _ in ordered[cursor : cursor + split_group_counts[split]]:
                assigned[key] = split
            cursor += split_group_counts[split]

    reassigned: List[Dict[str, str]] = []
    for key, group_rows in ordered:
        split = assigned.get(key, "train")
        for row in group_rows:
            out = dict(row)
            out["split"] = split
            reassigned.append(out)

    split_rows = Counter(clean(row.get("split")) for row in reassigned)
    leaked = 0
    for key, group_rows in groups.items():
        splits = {assigned.get(key, clean(row.get("split"))) for row in group_rows}
        if len(splits) > 1:
            leaked += 1
    summary = {
        "mode": mode,
        "source_kinds": sorted(selected_sources) if selected_sources else ["<all>"],
        "group_count": n_groups,
        "target_row_count": len(target_rows),
        "untouched_row_count": len(untouched_rows),
        "split_group_counts": dict(split_group_counts),
        "stratify_label_ids": sorted(stratify_labels),
        "positive_group_counts": dict(positive_group_counts),
        "split_row_counts": dict(split_rows.most_common()),
        "leaked_group_count": leaked,
    }
    return sorted(untouched_rows + reassigned, key=lambda row: (clean(row.get("split")), source_audio_basename(row), clean(row.get("item_id")))), summary


def subset_vocab(vocab_payload: Mapping[str, Any], active_labels: Sequence[str]) -> Dict[str, Any]:
    active = set(active_labels)
    labels_out = [dict(label) for label in vocab_payload.get("labels", []) if str(label.get("id")) in active]
    found = {str(label.get("id")) for label in labels_out}
    missing = sorted(active.difference(found))
    if missing:
        raise SystemExit(f"Active labels missing from vocabulary: {missing}")
    return {
        "schema_version": vocab_payload.get("schema_version", "multilabel-v1"),
        "labels": labels_out,
    }


def rewrite_label_fields(row: Mapping[str, Any], active_labels: Sequence[str]) -> Dict[str, str]:
    active = set(active_labels)
    labs = [label for label in labels(row) if label in active]
    out = {str(key): str(value) for key, value in row.items()}
    text = "|".join(labs)
    for key in ("label_ids", "canonical_label_ids", "target_label_ids"):
        if key in out:
            out[key] = text
    if "is_background" in out:
        out["is_background"] = "0" if labs else "1"
    return out


def keep_row(row: Mapping[str, Any], variant: Mapping[str, Any]) -> bool:
    source = clean(row.get("source_kind"))
    if source not in set(variant["sources"]):
        return False
    labs = set(labels(row))
    if not labs:
        return True
    return bool(labs.intersection(set(variant["active_label_ids"])))


def deterministic_sample(rows: Sequence[Dict[str, str]], cap: int, rng: random.Random) -> List[Dict[str, str]]:
    rows = list(rows)
    if len(rows) <= cap:
        return rows
    return rng.sample(rows, int(cap))


def apply_label_train_cap(
    rows: Sequence[Dict[str, str]],
    active_label_ids: Sequence[str],
    *,
    rng: random.Random,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    train_rows = [row for row in rows if clean(row.get("split")) == "train"]
    other_rows = [row for row in rows if clean(row.get("split")) != "train"]
    positive_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    background: List[Dict[str, str]] = []
    for row in train_rows:
        key = active_label_key(row, active_label_ids)
        if key == "<background>":
            background.append(row)
        else:
            positive_groups[key].append(row)
    nonempty_counts = [len(positive_groups[label]) for label in active_label_ids if positive_groups[label]]
    if not nonempty_counts:
        return list(rows), {"strategy": "label_train_cap", "reason": "no_positive_train_rows"}
    target = min(nonempty_counts)
    capped: List[Dict[str, str]] = []
    caps: Dict[str, Any] = {"strategy": "label_train_cap", "positive_cap": target, "input_counts": {}}
    for label in active_label_ids:
        group = positive_groups[label]
        caps["input_counts"][label] = len(group)
        capped.extend(deterministic_sample(group, target, rng))
    background_cap = max(target * max(len(active_label_ids), 1), target)
    caps["background_input_count"] = len(background)
    caps["background_cap"] = background_cap
    capped.extend(deterministic_sample(background, background_cap, rng))
    capped.extend(other_rows)
    return capped, caps


def apply_source_label_train_cap(
    rows: Sequence[Dict[str, str]],
    active_label_ids: Sequence[str],
    *,
    rng: random.Random,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    train_rows = [row for row in rows if clean(row.get("split")) == "train"]
    other_rows = [row for row in rows if clean(row.get("split")) != "train"]
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        groups[(clean(row.get("source_kind")), active_label_key(row, active_label_ids))].append(row)
    capped: List[Dict[str, str]] = []
    cap_info: Dict[str, Any] = {
        "strategy": "source_label_train_cap",
        "caps": {f"{src}:{label}": cap for (src, label), cap in SOURCE_LABEL_TRAIN_CAPS.items()},
        "input_counts": {},
    }
    for key, group in sorted(groups.items()):
        source, label = key
        cap_info["input_counts"][f"{source}:{label}"] = len(group)
        cap = SOURCE_LABEL_TRAIN_CAPS.get(key)
        if cap is None:
            capped.extend(group)
        else:
            capped.extend(deterministic_sample(group, cap, rng))
    capped.extend(other_rows)
    return capped, cap_info


def apply_cap(
    rows: Sequence[Dict[str, str]],
    variant: Mapping[str, Any],
    *,
    seed: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    strategy = str(variant.get("cap_strategy") or "none")
    rng = random.Random(int(seed) + sum(ord(ch) for ch in str(variant["name"])))
    if strategy == "none":
        return list(rows), {"strategy": "none"}
    if strategy == "label_train_cap":
        return apply_label_train_cap(rows, variant["active_label_ids"], rng=rng)
    if strategy == "source_label_train_cap":
        return apply_source_label_train_cap(rows, variant["active_label_ids"], rng=rng)
    raise ValueError(f"Unknown cap_strategy {strategy!r}")


def summarize(
    selected_rows: Sequence[Mapping[str, Any]],
    variant: Mapping[str, Any],
    fieldnames: Sequence[str],
    cap_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    split_counts = Counter(clean(row.get("split")) or "<blank>" for row in selected_rows)
    split_source_label = Counter(
        (
            clean(row.get("split")) or "<blank>",
            clean(row.get("source_kind")) or "<blank>",
            label_key(row),
            clean(row.get("negative_bucket")) or "",
        )
        for row in selected_rows
    )
    time_counts = Counter(
        (
            clean(row.get("split")) or "<blank>",
            clean(row.get("source_kind")) or "<blank>",
            label_key(row),
            month_bin(row),
        )
        for row in selected_rows
    )
    missing_by_band = {
        band: sum(1 for row in selected_rows if not clean(row.get(f"{band}_mat_path")))
        for band in variant["bands"]
    }
    return {
        "variant_name": variant["name"],
        "description": variant["description"],
        "active_label_ids": list(variant["active_label_ids"]),
        "eval_label_ids": list(variant["eval_label_ids"]),
        "sources": list(variant["sources"]),
        "bands": list(variant["bands"]),
        "cap_summary": dict(cap_summary),
        "crop_time_seconds": variant.get("crop_time_seconds", 10),
        "calibration_source_kind": variant.get("calibration_source_kind", "ONC"),
        "eval_source_kind": variant.get("eval_source_kind", "ONC"),
        "row_count": len(selected_rows),
        "split_counts": dict(split_counts.most_common()),
        "missing_mat_path_by_band": missing_by_band,
        "columns": list(fieldnames),
        "split_source_label_counts": [
            {
                "split": split,
                "source_kind": source,
                "label": label,
                "negative_bucket": bucket,
                "rows": count,
            }
            for (split, source, label, bucket), count in split_source_label.most_common()
        ],
        "time_counts": [
            {
                "split": split,
                "source_kind": source,
                "label": label,
                "month": month,
                "rows": count,
            }
            for (split, source, label, month), count in time_counts.most_common()
        ],
    }


def read_manifest(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {path}")
        return list(reader.fieldnames), rows


def write_manifest(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_variants(
    *,
    input_manifest: Path,
    input_vocab: Path,
    output_root: Path,
    seed: int,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    fieldnames, rows = read_manifest(input_manifest)
    vocab_payload = json.loads(input_vocab.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    index: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        out_dir = output_root / str(variant["name"])
        selected = [
            rewrite_label_fields(row, variant["active_label_ids"])
            for row in rows
            if keep_row(row, variant)
        ]
        oversample_summary: Dict[str, Any] = {"dropped_oversampled_rows": 0}
        if variant.get("drop_oversampled_rows"):
            before = len(selected)
            selected = [row for row in selected if not is_truthy(row.get("is_oversampled"))]
            oversample_summary = {
                "dropped_oversampled_rows": before - len(selected),
                "remaining_rows": len(selected),
            }
        split_grouping_summary: Dict[str, Any] = {"mode": "preserve_input"}
        if variant.get("split_grouping"):
            selected, split_grouping_summary = reassign_splits_by_group(
                selected,
                mode=str(variant["split_grouping"]),
                source_kinds=variant.get("split_grouping_source_kinds", []),
                stratify_label_ids=variant.get("split_grouping_label_ids", []),
                train_ratio=float(variant.get("train_ratio", 0.7)),
                val_ratio=float(variant.get("val_ratio", 0.15)),
            )
        selected, cap_summary = apply_cap(selected, variant, seed=seed)
        summary = summarize(selected, variant, fieldnames, cap_summary)
        summary["oversample_summary"] = dict(oversample_summary)
        summary["split_grouping_summary"] = dict(split_grouping_summary)
        summary.update(
            {
                "input_manifest": str(input_manifest),
                "manifest_csv": str(out_dir / "standardized_manifest.csv"),
                "vocab_json": str(out_dir / "label_vocabulary.json"),
            }
        )
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_manifest(out_dir / "standardized_manifest.csv", fieldnames, selected)
            (out_dir / "label_vocabulary.json").write_text(
                json.dumps(subset_vocab(vocab_payload, variant["active_label_ids"]), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (out_dir / "manifest_variant_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        index.append(summary)
        print(
            f"{variant['name']}: {len(selected)} rows; splits={summary['split_counts']}; "
            f"labels={variant['active_label_ids']}; bands={variant['bands']}; "
            f"cap={summary['cap_summary'].get('strategy')}"
        )
    if not dry_run:
        (output_root / "variant_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--input-vocab", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build_variants(
        input_manifest=args.input_manifest,
        input_vocab=args.input_vocab,
        output_root=args.output_root,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
