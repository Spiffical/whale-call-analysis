#!/usr/bin/env python3
"""Build E21 diagnostic manifests for balanced/frozen multiband follow-ups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.multilabel import build_e20_diagnostic_variants as e20  # noqa: E402


PRIMARY_LABELS = ("species:Bp", "species:Bm", "species:Mn", "species:Oo")


E21_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "E21_bp_bm_low_cumulative",
        "description": "Bp+Bm low-band cumulative probe to isolate baleen-label interference.",
        "active_label_ids": ["species:Bp", "species:Bm"],
        "eval_label_ids": ["species:Bp", "species:Bm"],
        "sources": ["ONC", "BioDCASE"],
        "bands": ["low"],
        "cap_strategy": "none",
    },
    {
        "name": "E21_bp_bm_low_sourcecap",
        "description": "Bp+Bm low-band probe with external source-label train caps.",
        "active_label_ids": ["species:Bp", "species:Bm"],
        "eval_label_ids": ["species:Bp", "species:Bm"],
        "sources": ["ONC", "BioDCASE"],
        "bands": ["low"],
        "cap_strategy": "source_label_train_cap",
    },
    {
        "name": "E21_bp_mn_lowmid_cumulative",
        "description": "Bp+Mn low+mid pairwise interference probe.",
        "active_label_ids": ["species:Bp", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "none",
    },
    {
        "name": "E21_bm_mn_lowmid_cumulative",
        "description": "Bm+Mn low+mid pairwise positive-control probe.",
        "active_label_ids": ["species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "none",
    },
    {
        "name": "E21_bp_bm_mn_lowmid_labelcap",
        "description": "Bp+Bm+Mn low+mid probe with train-positive label caps.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "label_train_cap",
    },
    {
        "name": "E21_bp_bm_mn_lowmid_sourcecap",
        "description": "Bp+Bm+Mn low+mid probe with external source-label train caps.",
        "active_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "eval_label_ids": ["species:Bp", "species:Bm", "species:Mn"],
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid"],
        "cap_strategy": "source_label_train_cap",
    },
    {
        "name": "E21_full_allbands_labelcap",
        "description": "Full low+mid+high probe with train-positive label caps.",
        "active_label_ids": list(PRIMARY_LABELS),
        "eval_label_ids": list(PRIMARY_LABELS),
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid", "high"],
        "cap_strategy": "label_train_cap",
    },
    {
        "name": "E21_full_allbands_sourcecap",
        "description": "Full low+mid+high probe with external source-label train caps.",
        "active_label_ids": list(PRIMARY_LABELS),
        "eval_label_ids": list(PRIMARY_LABELS),
        "sources": ["ONC", "BioDCASE", "DCLDE"],
        "bands": ["low", "mid", "high"],
        "cap_strategy": "source_label_train_cap",
    },
    {
        "name": "E21_oo_midhigh_sourcecap",
        "description": "Oo-only ONC+DCLDE mid+high probe with source-label train caps.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC", "DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "source_label_train_cap",
    },
    {
        "name": "E21_oo_onc_only_midhigh",
        "description": "Oo-only ONC-only mid+high probe for ONC support sanity checking.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["ONC"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "ONC",
        "eval_source_kind": "ONC",
    },
    {
        "name": "E21_oo_dclde_only_midhigh",
        "description": "Oo-only DCLDE-only mid+high in-domain control with balanced loss.",
        "active_label_ids": ["species:Oo"],
        "eval_label_ids": ["species:Oo"],
        "sources": ["DCLDE"],
        "bands": ["mid", "high"],
        "cap_strategy": "none",
        "calibration_source_kind": "DCLDE",
        "eval_source_kind": "DCLDE",
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
        e20.VARIANTS = E21_VARIANTS
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
