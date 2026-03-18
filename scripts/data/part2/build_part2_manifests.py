#!/usr/bin/env python3
"""Build normalized Part 2 manifests from the 2025 Clayoquot workbook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.part2_annotations import build_part2_manifests, write_part2_manifests


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize the 2025 Part 2 workbook into CSV/TXT manifests")
    ap.add_argument(
        "--workbook",
        type=str,
        default="data/finwhales/Clayoquot_2025_annotations_Mar18.xlsx",
        help="Path to the 2025 annotation workbook",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where normalized manifests should be written",
    )
    ap.add_argument(
        "--smoke-per-bucket",
        type=int,
        default=6,
        help="Number of smoke-test clips to sample per fin subtype bucket",
    )
    ap.add_argument(
        "--smoke-non-fin",
        type=int,
        default=6,
        help="Number of smoke-test clips to sample from annotated non-fin clips",
    )
    ap.add_argument(
        "--adjacent-boundary-seconds",
        type=float,
        default=20.0,
        help="Add previous/next 5-minute context clips when annotations fall within this many seconds of a clip edge",
    )
    ap.add_argument(
        "--include-adjacent-in-prep",
        action="store_true",
        help="Include the boundary-adjacent context clips in prep_clips.txt in addition to candidate clips",
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    manifests = build_part2_manifests(
        workbook_path=args.workbook,
        smoke_per_bucket=max(0, int(args.smoke_per_bucket)),
        smoke_non_fin=max(0, int(args.smoke_non_fin)),
        adjacent_boundary_seconds=max(0.0, float(args.adjacent_boundary_seconds)),
        include_adjacent_in_prep=bool(args.include_adjacent_in_prep),
        seed=int(args.seed),
    )
    written = write_part2_manifests(args.output_dir, manifests)

    print("Wrote Part 2 manifests:")
    for key in sorted(written):
        print(f"  {key}: {written[key]}")
    print("")
    print(json.dumps(manifests["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
