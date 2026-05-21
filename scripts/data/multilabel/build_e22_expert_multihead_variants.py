#!/usr/bin/env python3
"""Build E22 expert and multi-head diagnostic manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.multilabel import build_e20_diagnostic_variants as e20  # noqa: E402


E22_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "E22_fin_whale_low_expert",
        "description": "Fin whale low-band expert using ONC and BioDCASE.",
        "active_label_ids": ["species:Bp"],
        "eval_label_ids": ["species:Bp"],
        "sources": ["ONC", "BioDCASE"],
        "bands": ["low"],
        "cap_strategy": "none",
    },
    {
        "name": "E22_blue_whale_low_expert",
        "description": "Blue whale low-band expert using ONC and BioDCASE.",
        "active_label_ids": ["species:Bm"],
        "eval_label_ids": ["species:Bm"],
        "sources": ["ONC", "BioDCASE"],
        "bands": ["low"],
        "cap_strategy": "none",
    },
    {
        "name": "E22_humpback_whale_lowmid_expert",
        "description": "Humpback whale low+mid expert using ONC and DCLDE.",
        "active_label_ids": ["species:Mn"],
        "eval_label_ids": ["species:Mn"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "none",
    },
    {
        "name": "E22_killer_whale_onc_only_midhigh_expert",
        "description": "Killer whale ONC-only mid+high expert for ONC support sanity checking.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
    },
    {
        "name": "E22_killer_whale_dclde_only_midhigh_expert",
        "description": "Killer whale DCLDE-only mid+high expert for in-domain source checking.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "DCLDE",
        "eval_source_kind": "DCLDE",
    },
    {
        "name": "E22_killer_whale_onc_dclde_midhigh_sourcecap",
        "description": "Killer whale ONC+DCLDE mid+high expert with source-label train caps.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "source_label_train_cap",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
    },
    {
        "name": "E22_three_species_multihead_lowmid",
        "description": "Fin whale, blue whale, and humpback whale low+mid multi-head model.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "none",
    },
    {
        "name": "E22_three_species_multihead_lowmid_sourcecap",
        "description": "Three-species low+mid multi-head model with external source-label train caps.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "source_label_train_cap",
    },
    {
        "name": "E22_three_species_multihead_lowmid_labelcap",
        "description": "Three-species low+mid multi-head model with train-positive label caps.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "label_train_cap",
    },
    {
        "name": "E22_four_species_multihead_allbands_sourcecap",
        "description": "Full low+mid+high four-species multi-head model with source-label train caps.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn", "species:Oo"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn", "species:Oo"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid", "high"],
        "cap_strategy": "source_label_train_cap",
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
        e20.VARIANTS = E22_VARIANTS
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
