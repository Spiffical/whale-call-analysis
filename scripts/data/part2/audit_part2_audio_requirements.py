#!/usr/bin/env python3
"""Audit required raw audio for the final 2025 Part 2 workbook.

This script computes the exact 5-minute clip filenames needed for Part 2
bundle prep, including:

1. candidate fin-positive and annotated non-fin clips
2. boundary-adjacent clips requested by the workbook annotations
3. extra previous/next clips needed when full-clip inference windows use edge
   context from the training spectrogram configuration

Optionally, it compares that required set against a flat available-filenames
manifest such as the canonical Nibi list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sequential_prep import get_processing_params, load_dataset_documentation
from src.dataset.part2_annotations import (
    ANNOTATIONS_2025_WORKBOOK_DEFAULT,
    adjacent_clip_filename,
    build_part2_manifests,
)


CANONICAL_NIBI_RAW_AUDIO_ARCHIVE = Path(
    "/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio.tar.zst"
)
CANONICAL_NIBI_AVAILABLE_FILENAMES = Path(
    "/project/6070467/merileo/data/finwhales/archives/clayoquot_raw_audio_available_filenames.txt"
)


def _resolve_edge_context_seconds(
    dataset_doc_path: Optional[Path],
    explicit_value: Optional[float],
) -> float:
    if explicit_value is not None:
        return max(0.0, float(explicit_value))
    if dataset_doc_path is None:
        return 0.0
    dataset_doc = load_dataset_documentation(str(dataset_doc_path))
    proc = get_processing_params(dataset_doc=dataset_doc, model_path=None)
    crop_size = int(proc.get("crop_size") or 96)
    win_dur = float(proc.get("win_dur") or 1.0)
    overlap = float(proc.get("overlap") or 0.9)
    hop_s = win_dur * max(0.0, 1.0 - overlap)
    return max(0.0, win_dur + max(0, crop_size - 1) * hop_s)


def _adjacent_context_for_prep_clips(
    clip_names: Sequence[str],
    inventory_names: Sequence[str],
) -> List[str]:
    available = set(inventory_names)
    adjacent: Set[str] = set()
    for clip_name in clip_names:
        for clip_delta in (-1, 1):
            neighbor = adjacent_clip_filename(clip_name, clip_delta=clip_delta)
            if neighbor and neighbor in available:
                adjacent.add(neighbor)
    return sorted(adjacent)


def _read_name_list(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None or not path.exists():
        return None
    names: Set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                names.add(value)
    return names


def _write_name_list(path: Path, names: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(names)
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Part 2 raw-audio requirements")
    ap.add_argument(
        "--workbook",
        type=str,
        default=ANNOTATIONS_2025_WORKBOOK_DEFAULT,
        help="Path to the 2025 annotation workbook",
    )
    ap.add_argument(
        "--dataset-doc",
        type=str,
        default=None,
        help="Optional dataset_documentation.json used to infer edge context seconds",
    )
    ap.add_argument(
        "--edge-context-s",
        type=float,
        default=None,
        help="Explicit edge context seconds. Overrides --dataset-doc when provided.",
    )
    ap.add_argument(
        "--adjacent-boundary-seconds",
        type=float,
        default=20.0,
        help="Boundary threshold for previous/next 5-minute context clips",
    )
    ap.add_argument(
        "--include-adjacent-in-prep",
        action="store_true",
        help="Mirror the prep setting where adjacent boundary clips are included in prep_clips",
    )
    default_available = (
        str(CANONICAL_NIBI_AVAILABLE_FILENAMES)
        if CANONICAL_NIBI_AVAILABLE_FILENAMES.exists()
        else None
    )
    ap.add_argument(
        "--available-filenames-txt",
        type=str,
        default=default_available,
        help=(
            "Optional newline-delimited available-filenames manifest. "
            "On Nibi this defaults to the canonical clayoquot raw-audio list when present."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the audit lists and summary should be written",
    )
    args = ap.parse_args()

    workbook_path = Path(args.workbook).resolve()
    dataset_doc_path = Path(args.dataset_doc).resolve() if args.dataset_doc else None
    available_path = Path(args.available_filenames_txt).resolve() if args.available_filenames_txt else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = build_part2_manifests(
        workbook_path,
        adjacent_boundary_seconds=max(0.0, float(args.adjacent_boundary_seconds)),
        include_adjacent_in_prep=bool(args.include_adjacent_in_prep),
    )
    inventory_clip_names = [row["filename"] for row in manifests.get("clip_inventory", [])]
    candidate_clip_names = [row["filename"] for row in manifests.get("candidate_clips", [])]
    adjacent_clip_names = [row["filename"] for row in manifests.get("adjacent_context_clips", [])]
    prep_clip_names = [row["filename"] for row in manifests.get("prep_clips", manifests.get("candidate_clips", []))]
    download_clip_names = [row["filename"] for row in manifests.get("download_clips", manifests.get("candidate_clips", []))]

    edge_context_s = _resolve_edge_context_seconds(dataset_doc_path, args.edge_context_s)
    inference_context_clip_names = (
        _adjacent_context_for_prep_clips(prep_clip_names, inventory_clip_names)
        if edge_context_s > 0
        else []
    )
    required_audio_names = sorted(set(download_clip_names) | set(inference_context_clip_names))

    available_names = _read_name_list(available_path)
    available_required_names: List[str] = []
    missing_required_names: List[str] = []
    if available_names is not None:
        available_required_names = [name for name in required_audio_names if name in available_names]
        missing_required_names = [name for name in required_audio_names if name not in available_names]

    _write_name_list(output_dir / "candidate_clips.txt", sorted(candidate_clip_names))
    _write_name_list(output_dir / "adjacent_boundary_clips.txt", sorted(adjacent_clip_names))
    _write_name_list(output_dir / "prep_clips.txt", sorted(prep_clip_names))
    _write_name_list(output_dir / "download_clips.txt", sorted(download_clip_names))
    _write_name_list(output_dir / "inference_context_clips.txt", sorted(inference_context_clip_names))
    _write_name_list(output_dir / "required_audio_filenames.txt", required_audio_names)
    if available_names is not None:
        _write_name_list(output_dir / "available_required_audio_filenames.txt", available_required_names)
        _write_name_list(output_dir / "missing_required_audio_filenames.txt", missing_required_names)

    summary: Dict[str, object] = {
        "workbook": str(workbook_path),
        "dataset_doc": "" if dataset_doc_path is None else str(dataset_doc_path),
        "edge_context_s": float(edge_context_s),
        "adjacent_boundary_seconds": float(args.adjacent_boundary_seconds),
        "include_adjacent_in_prep": bool(args.include_adjacent_in_prep),
        "candidate_clip_count": len(candidate_clip_names),
        "adjacent_boundary_clip_count": len(adjacent_clip_names),
        "prep_clip_count": len(prep_clip_names),
        "download_clip_count": len(download_clip_names),
        "inference_context_clip_count": len(inference_context_clip_names),
        "required_audio_file_count": len(required_audio_names),
        "available_filenames_txt": "" if available_path is None else str(available_path),
        "available_required_audio_file_count": len(available_required_names),
        "missing_required_audio_file_count": len(missing_required_names),
        "canonical_nibi_raw_audio_archive": (
            str(CANONICAL_NIBI_RAW_AUDIO_ARCHIVE)
            if CANONICAL_NIBI_RAW_AUDIO_ARCHIVE.exists()
            else ""
        ),
        "canonical_nibi_available_filenames": (
            str(CANONICAL_NIBI_AVAILABLE_FILENAMES)
            if CANONICAL_NIBI_AVAILABLE_FILENAMES.exists()
            else ""
        ),
        "manifest_summary": manifests.get("summary", {}),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
