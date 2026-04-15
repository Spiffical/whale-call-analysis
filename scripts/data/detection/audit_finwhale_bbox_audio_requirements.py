#!/usr/bin/env python3
"""Audit raw-audio requirements for fin-whale bbox export, including adjacent files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.finwhale_bbox_audio_audit import AudioAuditConfig, audit_audio_requirements, write_audio_audit


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
    ap = argparse.ArgumentParser(description="Audit fin-whale bbox audio requirements")
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
    ap.add_argument("--historical-audio-dir", type=str, required=True)
    ap.add_argument("--audio-2025-dir", type=str, required=True)
    ap.add_argument("--allowed-filenames-txt", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--context-duration-s", type=float, default=40.0)
    ap.add_argument("--clip-duration-s", type=float, default=300.0)
    ap.add_argument("--edge-buffer-s", type=float, default=2.0)
    ap.add_argument("--pure-zero-ratio", type=float, default=0.5)
    ap.add_argument("--negative-margin-s", type=float, default=2.0)
    args = ap.parse_args()

    audit = audit_audio_requirements(
        annotation_manifest_csv=args.annotation_manifest,
        clip_manifest_csv=args.clip_manifest,
        split_assignments_csv=args.split_assignments,
        config=AudioAuditConfig(
            historical_audio_dir=Path(args.historical_audio_dir),
            audio_2025_dir=Path(args.audio_2025_dir),
            context_duration_s=float(args.context_duration_s),
            clip_duration_s=float(args.clip_duration_s),
            edge_buffer_s=float(args.edge_buffer_s),
            pure_zero_ratio=float(args.pure_zero_ratio),
            negative_margin_s=float(args.negative_margin_s),
        ),
        allowed_filenames=_read_name_list(args.allowed_filenames_txt),
    )
    written = write_audio_audit(args.output_dir, audit)

    print("Wrote fin-whale bbox audio audit:")
    for key in sorted(written):
        print(f"  {key}: {written[key]}")
    print("")
    print(Path(written["summary"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
