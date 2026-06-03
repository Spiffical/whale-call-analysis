#!/usr/bin/env python3
"""Build E26 ONC-only expert-ensemble ablation manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.multilabel import build_e20_diagnostic_variants as e20  # noqa: E402


E26_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "E26_fin_whale_low_onc_only",
        "description": "Fin whale low-band expert using ONC annotations only.",
        "active_label_ids": ["species:Bp"],
        "eval_label_ids": ["species:Bp"],
        "sources": ["ONC"],
        "bands": ["low"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
    },
    {
        "name": "E26_blue_whale_low_onc_only",
        "description": "Blue whale low-band expert using ONC annotations only.",
        "active_label_ids": ["species:Bm"],
        "eval_label_ids": ["species:Bm"],
        "sources": ["ONC"],
        "bands": ["low"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
    },
    {
        "name": "E26_humpback_whale_lowmid_onc_only",
        "description": "Humpback whale low+mid expert using ONC annotations only.",
        "active_label_ids": ["species:Mn"],
        "eval_label_ids": ["species:Mn"],
        "sources": ["ONC"],
        "bands": ["low", "mid"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
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
        e20.VARIANTS = E26_VARIANTS
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
