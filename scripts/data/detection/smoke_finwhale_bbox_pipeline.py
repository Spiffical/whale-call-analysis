#!/usr/bin/env python3
"""Run a small local smoke test of the fin-whale bbox manifest/split/export pipeline."""

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
    build_bbox_splits,
    build_joint_bbox_manifests,
    write_bbox_splits,
    write_joint_bbox_manifests,
)
from src.dataset.finwhale_bbox_export import export_bbox_dataset
from src.training.finwhale_yolo import build_yolo_dataset_from_coco


def _list_audio_filenames(audio_dir: Path, limit: int) -> list[str]:
    names = sorted(path.name for path in audio_dir.iterdir() if path.is_file())
    return names[: max(0, int(limit))] if limit > 0 else names


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the fin-whale bbox pipeline")
    ap.add_argument("--historical-workbook", type=str, default=HISTORICAL_WORKBOOK_DEFAULT)
    ap.add_argument(
        "--annotations-2025-workbook",
        "--species-temporal-workbook",
        dest="annotations_2025_workbook",
        type=str,
        default=ANNOTATIONS_2025_WORKBOOK_DEFAULT,
    )
    ap.add_argument("--mar26-workbook", type=str, default=MAR26_WORKBOOK_DEFAULT)
    ap.add_argument("--mar18-workbook", type=str, default=MAR18_WORKBOOK_DEFAULT)
    ap.add_argument("--audio-dir", type=str, default="data/finwhale_part2_smoke_bundle/raw_audio")
    ap.add_argument("--config-path", type=str, default="config/dataset_config.yaml")
    ap.add_argument("--output-dir", type=str, default="tmp/finwhale_bbox_smoke")
    ap.add_argument("--audio-limit", type=int, default=18)
    ap.add_argument("--qc-limit", type=int, default=12)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    manifests_dir = output_dir / "manifests"
    splits_dir = output_dir / "splits"
    export_dir = output_dir / "export"
    yolo_dir = output_dir / "yolo"

    allowed_filenames = set(_list_audio_filenames(audio_dir, args.audio_limit))
    if not allowed_filenames:
        raise SystemExit(f"No audio files found under {audio_dir}")

    manifests = build_joint_bbox_manifests(
        historical_workbook=args.historical_workbook,
        species_temporal_workbook=args.annotations_2025_workbook,
        mar26_workbook=args.mar26_workbook,
        mar18_workbook=args.mar18_workbook,
    )
    written_manifests = write_joint_bbox_manifests(manifests_dir, manifests)

    split_data = build_bbox_splits(manifests["annotations"], manifests["clip_manifest"])
    written_splits = write_bbox_splits(splits_dir, split_data)

    export_results = export_bbox_dataset(
        annotation_manifest_csv=written_manifests["annotations"],
        clip_manifest_csv=written_manifests["clip_manifest"],
        split_assignments_csv=written_splits["assignments"],
        audio_dir=audio_dir,
        output_dir=export_dir,
        config_path=args.config_path,
        allowed_filenames=allowed_filenames,
        qc_limit=int(args.qc_limit),
    )
    yolo_results = build_yolo_dataset_from_coco(
        coco_export_dir=export_dir,
        output_dir=yolo_dir,
        link_mode="copy",
    )

    summary = {
        "allowed_audio_count": int(len(allowed_filenames)),
        "manifests_dir": str(manifests_dir),
        "splits_dir": str(splits_dir),
        "export_dir": str(export_dir),
        "yolo_dir": str(yolo_dir),
        "manifest_summary": manifests["summary"],
        "split_summary": split_data["summary"],
        "export_summary": json.loads(Path(export_results["summary"]).read_text(encoding="utf-8")),
        "yolo_summary": yolo_results["summary"],
    }
    summary_path = output_dir / "smoke_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("Fin-whale bbox smoke run complete:")
    print(f"  manifests: {manifests_dir}")
    print(f"  splits:    {splits_dir}")
    print(f"  export:    {export_dir}")
    print(f"  yolo:      {yolo_dir}")
    print(f"  summary:   {summary_path}")
    print("")
    print(json.dumps(summary["export_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
