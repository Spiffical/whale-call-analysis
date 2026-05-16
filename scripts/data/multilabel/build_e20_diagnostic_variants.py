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
        selected, cap_summary = apply_cap(selected, variant, seed=seed)
        summary = summarize(selected, variant, fieldnames, cap_summary)
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
