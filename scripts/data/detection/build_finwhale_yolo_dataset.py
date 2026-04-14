#!/usr/bin/env python3
"""Build a YOLO-format dataset from the exported fin-whale COCO dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.finwhale_yolo import build_yolo_dataset_from_coco


def main() -> None:
    ap = argparse.ArgumentParser(description="Build YOLO labels/yamls from fin-whale COCO export")
    ap.add_argument(
        "--coco-export-dir",
        type=str,
        default="output/finwhale_bbox/exports/fin_1to200_detector_v1",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="output/finwhale_bbox/exports/fin_1to200_yolo26_v1",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="",
        help="Optional comma-separated split list. Default: all COCO splits under the export dir.",
    )
    ap.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
    )
    args = ap.parse_args()

    split_names = [part.strip() for part in args.splits.split(",") if part.strip()] or None
    result = build_yolo_dataset_from_coco(
        coco_export_dir=args.coco_export_dir,
        output_dir=args.output_dir,
        split_names=split_names,
        link_mode=args.link_mode,
    )

    print("Wrote YOLO dataset:")
    print(f"  train_yaml: {result['train_yaml']}")
    print(f"  summary:    {result['summary_path']}")
    print("")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
