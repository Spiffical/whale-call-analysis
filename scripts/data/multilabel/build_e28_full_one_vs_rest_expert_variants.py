#!/usr/bin/env python3
"""Build E28 full-source one-vs-rest expert ablation manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.multilabel import build_e20_diagnostic_variants as e20  # noqa: E402


FULL_SOURCES = ["ONC", "BioDCASE", "DCLDE"]

E28_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "E28_fin_whale_low_full_one_vs_rest",
        "description": "Fin whale low-band full-source one-vs-rest expert; non-fin labels are retained as negatives.",
        "active_label_ids": ["species:Bp"],
        "eval_label_ids": ["species:Bp"],
        "sources": FULL_SOURCES,
        "bands": ["low"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
        "keep_nonactive_labels_as_background": True,
    },
    {
        "name": "E28_blue_whale_low_full_one_vs_rest",
        "description": "Blue whale low-band full-source one-vs-rest expert; non-blue labels are retained as negatives.",
        "active_label_ids": ["species:Bm"],
        "eval_label_ids": ["species:Bm"],
        "sources": FULL_SOURCES,
        "bands": ["low"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
        "keep_nonactive_labels_as_background": True,
    },
    {
        "name": "E28_humpback_whale_lowmid_full_one_vs_rest",
        "description": "Humpback whale low+mid full-source one-vs-rest expert; non-humpback labels are retained as negatives.",
        "active_label_ids": ["species:Mn"],
        "eval_label_ids": ["species:Mn"],
        "sources": FULL_SOURCES,
        "bands": ["low", "mid"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
        "keep_nonactive_labels_as_background": True,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--input-vocab", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    original_variants = e20.VARIANTS
    try:
        e20.VARIANTS = E28_VARIANTS
        e20.build_variants(
            input_manifest=args.input_manifest,
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
