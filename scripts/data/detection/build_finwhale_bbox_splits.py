#!/usr/bin/env python3
"""Build leakage-safe train/val/test splits for the fin-whale bbox pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.finwhale_bbox import (
    build_bbox_splits,
    load_annotation_manifest,
    load_clip_manifest,
    write_bbox_splits,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fin-whale bbox split assignments")
    ap.add_argument(
        "--annotation-manifest",
        type=str,
        default="output/finwhale_bbox/manifests/joint_v1/unified_annotations.csv",
    )
    ap.add_argument(
        "--clip-manifest",
        type=str,
        default="output/finwhale_bbox/manifests/joint_v1/clip_manifest.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="output/finwhale_bbox/splits/joint_v1",
    )
    args = ap.parse_args()

    annotation_df = load_annotation_manifest(args.annotation_manifest)
    clip_df = load_clip_manifest(args.clip_manifest)
    split_data = build_bbox_splits(annotation_df, clip_df)
    written = write_bbox_splits(args.output_dir, split_data)

    print("Wrote fin-whale bbox splits:")
    for key in sorted(written):
        print(f"  {key}: {written[key]}")
    print("")
    print(json.dumps(split_data["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
