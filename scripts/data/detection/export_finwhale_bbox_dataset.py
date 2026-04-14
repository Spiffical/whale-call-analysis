#!/usr/bin/env python3
"""Export detector-ready spectrogram crops and COCO annotations for fin-whale bbox training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.finwhale_bbox_export import export_bbox_dataset


def _read_name_list(path: str | None) -> set[str] | None:
    if not path:
        return None
    names: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                names.add(value)
    return names or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Export detector-ready fin-whale bbox data")
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
        "--split-assignments",
        type=str,
        default="output/finwhale_bbox/splits/joint_v1/assignments.csv",
    )
    ap.add_argument("--audio-dir", type=str, required=True)
    ap.add_argument(
        "--output-dir",
        type=str,
        default="output/finwhale_bbox/exports/fin_1to200_detector_v1",
    )
    ap.add_argument("--config-path", type=str, default="config/dataset_config.yaml")
    ap.add_argument("--allowed-filenames-txt", type=str, default=None)
    ap.add_argument("--context-duration-s", type=float, default=40.0)
    ap.add_argument("--train-crop-duration-s", type=float, default=10.0)
    ap.add_argument("--eval-crop-duration-s", type=float, default=10.0)
    ap.add_argument("--freq-min-hz", type=float, default=1.0)
    ap.add_argument("--freq-max-hz", type=float, default=200.0)
    ap.add_argument("--edge-buffer-s", type=float, default=2.0)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--pure-zero-ratio", type=float, default=0.5)
    ap.add_argument("--negative-margin-s", type=float, default=2.0)
    ap.add_argument("--center-bias-sigma-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--qc-limit", type=int, default=0)
    args = ap.parse_args()

    results = export_bbox_dataset(
        annotation_manifest_csv=args.annotation_manifest,
        clip_manifest_csv=args.clip_manifest,
        split_assignments_csv=args.split_assignments,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        allowed_filenames=_read_name_list(args.allowed_filenames_txt),
        context_duration_s=float(args.context_duration_s),
        train_crop_duration_s=float(args.train_crop_duration_s),
        eval_crop_duration_s=float(args.eval_crop_duration_s),
        freq_min_hz=float(args.freq_min_hz),
        freq_max_hz=float(args.freq_max_hz),
        edge_buffer_s=float(args.edge_buffer_s),
        image_size=int(args.image_size),
        pure_zero_ratio=float(args.pure_zero_ratio),
        negative_margin_s=float(args.negative_margin_s),
        center_bias_sigma_frac=float(args.center_bias_sigma_frac),
        seed=int(args.seed),
        qc_limit=int(args.qc_limit),
    )

    print("Wrote fin-whale bbox export:")
    for key in sorted(results):
        print(f"  {key}: {results[key]}")
    print("")
    summary_path = Path(results["summary"])
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
