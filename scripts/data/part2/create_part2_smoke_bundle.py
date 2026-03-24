#!/usr/bin/env python3
"""Create a small transferable Part 2 debug bundle from a full VM bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.part2_annotations import adjacent_clip_filename, parse_window_mat_stem


MANIFEST_CSV_KEYS = [
    "clip_inventory",
    "annotations_all",
    "fin_annotations",
    "clip_manifest",
    "fin_positive_clips",
    "annotated_non_fin_clips",
    "candidate_clips",
    "adjacent_context_clips",
    "download_clips",
    "prep_clips",
    "smoke_clips",
]
MANIFEST_TXT_KEYS = [
    "fin_positive_clips",
    "annotated_non_fin_clips",
    "candidate_clips",
    "adjacent_context_clips",
    "download_clips",
    "prep_clips",
    "smoke_clips",
]
CLIP_FILTER_KEYS = {
    "annotations_all",
    "fin_annotations",
    "clip_manifest",
    "fin_positive_clips",
    "annotated_non_fin_clips",
    "candidate_clips",
    "prep_clips",
    "smoke_clips",
}
CONTEXT_FILTER_KEYS = {
    "adjacent_context_clips",
    "download_clips",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _selected_clip_names(manifests_dir: Path, manifest_key: str, limit: int | None) -> List[str]:
    csv_rows = _read_csv(manifests_dir / f"{manifest_key}.csv")
    clip_names = sorted({str(row.get("filename", "")).strip() for row in csv_rows if row.get("filename")})
    if limit is not None:
        clip_names = clip_names[: max(0, int(limit))]
    return clip_names


def _neighbor_clip_names(clip_names: Sequence[str], inventory_names: Sequence[str]) -> List[str]:
    inventory = set(inventory_names)
    neighbors = set()
    for clip_name in clip_names:
        for delta in (-1, 1):
            candidate = adjacent_clip_filename(clip_name, clip_delta=delta)
            if candidate and candidate in inventory:
                neighbors.add(candidate)
    return sorted(neighbors)


def _filter_rows(rows: Sequence[Dict[str, str]], names: set[str]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if str(row.get("filename", "")).strip() in names]


def _copy_filtered_mats(source_dir: Path, target_dir: Path, selected_clips: set[str]) -> List[str]:
    copied: List[str] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for mat_path in sorted(source_dir.glob("*.mat")):
        parsed = parse_window_mat_stem(mat_path.stem)
        if parsed is None:
            continue
        source_audio, _, _ = parsed
        if source_audio not in selected_clips:
            continue
        _copy_file(mat_path, target_dir / mat_path.name)
        copied.append(mat_path.name)
    return copied


def _copy_filtered_report(source_report: Path, target_report: Path, selected_clips: set[str]) -> None:
    rows = _read_csv(source_report)
    filtered = [row for row in rows if str(row.get("clip", "")).strip() in selected_clips]
    _write_csv(target_report, filtered)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a tiny local Part 2 bundle for smoke tests")
    ap.add_argument("--source-bundle", type=str, required=True, help="Existing full Part 2 bundle directory")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory for the smoke bundle")
    ap.add_argument(
        "--manifest-key",
        type=str,
        default="smoke_clips",
        choices=[
            "smoke_clips",
            "prep_clips",
            "candidate_clips",
            "fin_positive_clips",
            "annotated_non_fin_clips",
        ],
        help="Manifest list used to choose the subset clips",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on selected clips after manifest ordering")
    ap.add_argument(
        "--no-inference-context",
        dest="include_inference_context",
        action="store_false",
        help="Do not copy previous/next raw audio clips for boundary context",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it already exists")
    ap.set_defaults(include_inference_context=True)
    args = ap.parse_args()

    source_bundle = Path(args.source_bundle).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifests_dir = source_bundle / "manifests"
    source_mat_dir = source_bundle / "mat_files"
    source_audio_dir = source_bundle / "raw_audio"

    if not source_bundle.exists():
        raise SystemExit(f"Source bundle not found: {source_bundle}")
    if not manifests_dir.exists():
        raise SystemExit(f"Source manifests directory not found: {manifests_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir} (use --overwrite to replace it)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = {key: _read_csv(manifests_dir / f"{key}.csv") for key in MANIFEST_CSV_KEYS}
    selected_clips = _selected_clip_names(manifests_dir, args.manifest_key, args.limit)
    if not selected_clips:
        raise SystemExit(f"No clip names found for manifest key '{args.manifest_key}'")
    selected_clip_set = set(selected_clips)

    inventory_names = [
        str(row.get("filename", "")).strip()
        for row in manifest_rows.get("clip_inventory", [])
        if row.get("filename")
    ]
    inference_context = (
        _neighbor_clip_names(selected_clips, inventory_names) if args.include_inference_context else []
    )
    audio_clip_set = selected_clip_set | set(inference_context)

    out_manifests = output_dir / "manifests"
    out_audio = output_dir / "raw_audio"
    out_mats = output_dir / "mat_files"

    filtered_manifests: Dict[str, List[Dict[str, str]]] = {}
    for key in MANIFEST_CSV_KEYS:
        rows = manifest_rows.get(key, [])
        if key == "clip_inventory":
            filtered_manifests[key] = _filter_rows(rows, audio_clip_set)
        elif key in CLIP_FILTER_KEYS:
            filtered_manifests[key] = _filter_rows(rows, selected_clip_set)
        elif key in CONTEXT_FILTER_KEYS:
            filtered_manifests[key] = _filter_rows(rows, audio_clip_set)
        else:
            filtered_manifests[key] = list(rows)
        _write_csv(out_manifests / f"{key}.csv", filtered_manifests[key])

    for key in MANIFEST_TXT_KEYS:
        lines = [row["filename"] for row in filtered_manifests.get(key, []) if row.get("filename")]
        _write_lines(out_manifests / f"{key}.txt", lines)

    copied_audio = 0
    missing_audio: List[str] = []
    out_audio.mkdir(parents=True, exist_ok=True)
    for clip_name in sorted(audio_clip_set):
        src = source_audio_dir / clip_name
        if not src.exists():
            missing_audio.append(clip_name)
            continue
        _copy_file(src, out_audio / clip_name)
        copied_audio += 1

    copied_mats = _copy_filtered_mats(source_mat_dir, out_mats, selected_clip_set)
    source_report = source_mat_dir / "report.csv"
    if source_report.exists():
        _copy_filtered_report(source_report, out_mats / "report.csv", selected_clip_set)

    summary = _load_summary(manifests_dir / "summary.json")
    subset_summary = {
        "source_bundle": str(source_bundle),
        "manifest_key": args.manifest_key,
        "selected_clip_count": len(selected_clips),
        "selected_audio_with_context_count": len(audio_clip_set),
        "selected_inference_context_count": len(inference_context),
        "copied_audio_count": copied_audio,
        "missing_audio_count": len(missing_audio),
        "copied_mat_count": len(copied_mats),
        "source_summary": summary,
    }
    with open(out_manifests / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(subset_summary, handle, indent=2, sort_keys=True)

    source_metadata_path = source_bundle / "metadata.json"
    if source_metadata_path.exists():
        with open(source_metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["files"] = [
            row
            for row in metadata.get("files", [])
            if str(row.get("source_audio", "")).strip() in selected_clip_set
        ]
        bundle_meta = metadata.get("bundle", {})
        if isinstance(bundle_meta, dict):
            bundle_meta["prep_clip_count"] = len(selected_clips)
            bundle_meta["inference_context_clip_count"] = len(inference_context)
            bundle_meta["staged_audio_count"] = copied_audio
            bundle_meta["mat_count"] = len(metadata.get("files", []))
        metadata["part2_summary"] = subset_summary
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    prep_summary = {
        "bundle_dir": str(output_dir),
        "source_bundle": str(source_bundle),
        "manifest_key": args.manifest_key,
        "selected_clip_count": len(selected_clips),
        "inference_context_clip_count": len(inference_context),
        "staged_audio_count": copied_audio,
        "missing_audio_count": len(missing_audio),
        "mat_count": len(copied_mats),
    }
    with open(output_dir / "prep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(prep_summary, handle, indent=2, sort_keys=True)

    if missing_audio:
        _write_lines(output_dir / "missing_audio.txt", sorted(missing_audio))

    print(f"Created smoke bundle at: {output_dir}")
    print(f"  selected clips: {len(selected_clips)}")
    print(f"  copied mats:    {len(copied_mats)}")
    print(f"  copied audio:   {copied_audio}")
    print(f"  context clips:  {len(inference_context)}")
    if missing_audio:
        print(f"  missing audio:  {len(missing_audio)} -> {output_dir / 'missing_audio.txt'}")


if __name__ == "__main__":
    main()
