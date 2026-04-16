#!/usr/bin/env python3
"""Write the exact tar members needed for fin-whale bbox export raw audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.finwhale_bbox import (
    load_annotation_manifest,
    load_clip_manifest,
    load_split_assignments,
)
from src.dataset.finwhale_bbox_vm_audio import build_export_required_audio_filenames


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
    ap = argparse.ArgumentParser(description="Build raw-audio tar member list for fin-whale bbox jobs")
    ap.add_argument("--annotation-manifest", type=str, required=True)
    ap.add_argument("--clip-manifest", type=str, required=True)
    ap.add_argument("--split-assignments", type=str, required=True)
    ap.add_argument("--output-path", type=str, required=True)
    ap.add_argument("--tar-prefix", type=str, default="raw_audio")
    ap.add_argument("--allowed-filenames-txt", type=str, default=None)
    ap.add_argument(
        "--available-audio-filenames-txt",
        type=str,
        default=None,
        help="Optional list of audio filenames actually present in the source tar/archive; "
        "required filenames are intersected with this list after context selection.",
    )
    ap.add_argument("--context-duration-s", type=float, default=40.0)
    ap.add_argument("--clip-duration-s", type=float, default=300.0)
    ap.add_argument("--edge-buffer-s", type=float, default=2.0)
    ap.add_argument("--pure-zero-ratio", type=float, default=0.5)
    ap.add_argument("--negative-margin-s", type=float, default=2.0)
    ap.add_argument("--summary-path", type=str, default=None)
    args = ap.parse_args()

    annotation_df = load_annotation_manifest(args.annotation_manifest)
    clip_df = load_clip_manifest(args.clip_manifest)
    assignments_df = load_split_assignments(args.split_assignments)
    required = build_export_required_audio_filenames(
        annotation_df,
        clip_df,
        assignments_df,
        context_duration_s=float(args.context_duration_s),
        clip_duration_s=float(args.clip_duration_s),
        edge_buffer_s=float(args.edge_buffer_s),
        pure_zero_ratio=float(args.pure_zero_ratio),
        negative_margin_s=float(args.negative_margin_s),
        allowed_filenames=_read_name_list(args.allowed_filenames_txt),
    )

    required_names = list(required["required_filenames"])
    available_audio = _read_name_list(args.available_audio_filenames_txt)
    if available_audio is not None:
        filtered_names = [name for name in required_names if name in available_audio]
    else:
        filtered_names = required_names

    members = [f"{args.tar_prefix.rstrip('/')}/{name}" for name in filtered_names]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(members) + ("\n" if members else ""), encoding="utf-8")

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = dict(required["summary"])
        summary["required_file_count_before_available_filter"] = int(len(required_names))
        summary["required_file_count_after_available_filter"] = int(len(filtered_names))
        summary["available_audio_filter_applied"] = bool(available_audio is not None)
        summary["excluded_missing_archive_member_count"] = int(len(required_names) - len(filtered_names))
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(members)} tar members to {output_path}")
    print(
        json.dumps(
            {
                **required["summary"],
                "required_file_count_before_available_filter": int(len(required_names)),
                "required_file_count_after_available_filter": int(len(filtered_names)),
                "available_audio_filter_applied": bool(available_audio is not None),
                "excluded_missing_archive_member_count": int(len(required_names) - len(filtered_names)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
