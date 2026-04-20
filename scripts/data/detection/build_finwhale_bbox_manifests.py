#!/usr/bin/env python3
"""Build unified fin-whale bbox manifests from historical + 2025 workbooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.finwhale_bbox import (
    ANNOTATIONS_2025_WORKBOOK_DEFAULT,
    HISTORICAL_WORKBOOK_DEFAULT,
    MAR18_WORKBOOK_DEFAULT,
    MAR26_WORKBOOK_DEFAULT,
    build_joint_bbox_manifests,
    write_joint_bbox_manifests,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified fin-whale bbox manifests")
    ap.add_argument("--historical-workbook", type=str, default=HISTORICAL_WORKBOOK_DEFAULT)
    ap.add_argument(
        "--annotations-2025-workbook",
        "--species-temporal-workbook",
        dest="annotations_2025_workbook",
        type=str,
        default=ANNOTATIONS_2025_WORKBOOK_DEFAULT,
        help="Canonical 2025 annotation workbook (mixed cetacean + non-biological sheets)",
    )
    ap.add_argument("--mar26-workbook", type=str, default=MAR26_WORKBOOK_DEFAULT)
    ap.add_argument(
        "--mar18-workbook",
        type=str,
        default=MAR18_WORKBOOK_DEFAULT,
        help="Guardrail workbook used only to exclude false pure negatives",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="output/finwhale_bbox/manifests/joint_v1",
        help="Directory where the unified manifests should be written",
    )
    args = ap.parse_args()

    manifests = build_joint_bbox_manifests(
        historical_workbook=args.historical_workbook,
        species_temporal_workbook=args.annotations_2025_workbook,
        mar26_workbook=args.mar26_workbook,
        mar18_workbook=args.mar18_workbook,
    )
    written = write_joint_bbox_manifests(args.output_dir, manifests)

    print("Wrote fin-whale bbox manifests:")
    for key in sorted(written):
        print(f"  {key}: {written[key]}")
    print("")
    print(json.dumps(manifests["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
