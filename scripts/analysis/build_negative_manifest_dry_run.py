#!/usr/bin/env python3
"""Create a bounded negative-manifest dry run for the species-first ladder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.multilabel.build_dclde_killer_whale_manifest import build_dclde_manifest  # noqa: E402
from scripts.data.multilabel.build_negative_window_manifest import build_negative_manifest  # noqa: E402
from src.dataset.multilabel import clean_text, label_ids_from_row, split_pipe, write_csv_rows  # noqa: E402


DEFAULT_E09_MANIFEST = "manifests/E09_onc_biod_dclde_species/standardized_manifest.csv"
DEFAULT_DCLDE_ANNOTATIONS = "audits/dclde_2027_killer_whales/Annotations.csv"
DEFAULT_DCLDE_GCS_OBJECTS = "audits/dclde_2027_killer_whales/gcs_objects.txt"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_kind(row: Mapping[str, Any]) -> str:
    text = " ".join(
        clean_text(row.get(key)).lower()
        for key in ("source_dataset", "source_dataset_raw", "source_provider", "mat_path", "source_audio")
    )
    if "dclde" in text:
        return "DCLDE"
    if "biodcase" in text or "task2" in text:
        return "BioDCASE"
    if "final2025" in text or "part2" in text or "iclisten" in text:
        return "ONC"
    return "unknown"


def has_primary(row: Mapping[str, Any]) -> bool:
    ids = set(label_ids_from_row(dict(row)))
    ids.update(split_pipe(row.get("canonical_label_ids")))
    return bool(ids & {"species:Bp", "species:Bm", "species:Mn", "species:Oo"})


def labels_or_background(row: Mapping[str, Any]) -> List[str]:
    labels = label_ids_from_row(dict(row))
    return labels or ["<background>"]


def write_counter_table(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counts[tuple(clean_text(row.get(field)) or "<blank>" for field in fields)] += 1
    out = [{field: key[idx] for idx, field in enumerate(fields)} | {"count": count} for key, count in counts.most_common()]
    write_csv_rows(path, out)


def build_dry_run(
    *,
    weekend_root: Path,
    output_dir: Path,
    e09_manifest: Path,
    dclde_annotations: Path,
    dclde_gcs_object_lists: Sequence[Path],
    dclde_max_positive: int,
    dclde_max_hard_negative: int,
    max_gap_windows_per_clip: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    e09_rows = read_csv_rows(e09_manifest)
    non_dclde_rows = [row for row in e09_rows if source_kind(row) != "DCLDE"]

    dclde_dir = output_dir / "dclde_hw_mn_prep"
    dclde_summary = build_dclde_manifest(
        annotations_csv=dclde_annotations,
        output_dir=dclde_dir,
        gcs_object_lists=dclde_gcs_object_lists,
        require_gcs_audio=bool(dclde_gcs_object_lists),
        max_positive=int(dclde_max_positive),
        max_hard_negative=int(dclde_max_hard_negative),
    )
    dclde_rows = read_csv_rows(dclde_dir / "selected_calls.csv")

    combined_rows: List[Dict[str, Any]] = [dict(row) for row in non_dclde_rows] + [dict(row) for row in dclde_rows]
    input_csv = tables_dir / "negative_manifest_dry_run_input.csv"
    write_csv_rows(input_csv, combined_rows)

    onc_sources = sorted(
        {
            clean_text(row.get("source_audio") or row.get("filename") or row.get("clip"))
            for row in combined_rows
            if source_kind(row) == "ONC" and has_primary(row)
        }
    )
    duration_csv = tables_dir / "negative_manifest_onc_clip_durations.csv"
    write_csv_rows(duration_csv, [{"source_audio": source, "duration_s": "300.0"} for source in onc_sources])

    negative_csv = tables_dir / "negative_manifest_dry_run.csv"
    negative_summary = build_negative_manifest(
        annotations_csv=input_csv,
        output_csv=negative_csv,
        clip_duration_csv=duration_csv,
        window_s=10.0,
        exclusion_buffer_s=5.0,
        step_s=30.0,
        max_windows_per_clip=int(max_gap_windows_per_clip),
        split=True,
    )
    negative_rows = read_csv_rows(negative_csv)

    for row in combined_rows:
        row["source_kind"] = source_kind(row)
        row["has_primary"] = "1" if has_primary(row) else "0"
    for row in negative_rows:
        row["source_kind"] = source_kind(row)

    write_counter_table(tables_dir / "dry_run_positive_label_source_counts.csv", (
        {"source_kind": source_kind(row), "label_id": label}
        for row in combined_rows
        if has_primary(row)
        for label in labels_or_background(row)
        if label.startswith("species:")
    ), ("source_kind", "label_id"))
    write_counter_table(tables_dir / "dry_run_negative_bucket_source_counts.csv", negative_rows, ("source_kind", "negative_bucket"))
    write_counter_table(tables_dir / "dry_run_negative_bucket_split_counts.csv", negative_rows, ("split", "negative_bucket"))

    onc_review_queue = [
        row
        for row in negative_rows
        if source_kind(row) == "ONC"
        and clean_text(row.get("negative_bucket")) in {"ambiguous_hard_negative", "primary_adjacent_gap", "reviewed_background"}
    ]
    write_csv_rows(tables_dir / "dry_run_onc_negative_review_queue.csv", onc_review_queue)

    source_counts = Counter(source_kind(row) for row in combined_rows)
    positive_source_counts = Counter(source_kind(row) for row in combined_rows if has_primary(row))
    no_primary_source_counts = Counter(source_kind(row) for row in combined_rows if not has_primary(row))
    negative_bucket_counts = Counter(clean_text(row.get("negative_bucket")) or "<blank>" for row in negative_rows)
    negative_source_counts = Counter(source_kind(row) for row in negative_rows)
    reviewed_background_rows = [
        row for row in negative_rows if source_kind(row) == "ONC" and clean_text(row.get("negative_bucket")) == "reviewed_background"
    ]

    summary = {
        "weekend_root": str(weekend_root),
        "e09_manifest": str(e09_manifest),
        "dclde_annotations": str(dclde_annotations),
        "dclde_gcs_object_lists": [str(path) for path in dclde_gcs_object_lists],
        "combined_input_row_count": len(combined_rows),
        "combined_source_counts": dict(source_counts.most_common()),
        "positive_source_counts": dict(positive_source_counts.most_common()),
        "no_primary_source_counts": dict(no_primary_source_counts.most_common()),
        "negative_manifest_row_count": len(negative_rows),
        "negative_bucket_counts": dict(negative_bucket_counts.most_common()),
        "negative_source_counts": dict(negative_source_counts.most_common()),
        "onc_review_queue_count": len(onc_review_queue),
        "onc_reviewed_background_count": len(reviewed_background_rows),
        "dclde_summary": dclde_summary,
        "negative_summary": negative_summary,
        "decision": (
            "training_blocked_no_onc_reviewed_background"
            if not reviewed_background_rows
            else "negative_manifest_ready_for_visual_audit"
        ),
    }
    (output_dir / "negative_manifest_dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report_lines = [
        "# Negative Manifest Dry Run",
        "",
        f"- Combined input rows: `{len(combined_rows)}`.",
        f"- Negative manifest rows: `{len(negative_rows)}`.",
        f"- ONC negative review queue rows: `{len(onc_review_queue)}`.",
        f"- ONC reviewed-background rows: `{len(reviewed_background_rows)}`.",
        "",
        "## Negative Buckets",
        "",
    ]
    for bucket, count in negative_bucket_counts.most_common():
        report_lines.append(f"- `{bucket}`: `{count}`")
    report_lines.extend(["", "## Source Counts", ""])
    for kind, count in source_counts.most_common():
        report_lines.append(f"- `{kind}` combined input rows: `{count}`")
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Do not launch GPU training from this dry run.",
            "- The ONC reviewed-background bucket is still empty, so deployment background calibration remains blocked.",
            "- Use `tables/dry_run_onc_negative_review_queue.csv` for the next human/visual review pass.",
            "",
        ]
    )
    (output_dir / "negative_manifest_dry_run_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekend-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--e09-manifest", default="")
    parser.add_argument("--dclde-annotations", default="")
    parser.add_argument("--dclde-gcs-object-list", action="append", default=[])
    parser.add_argument("--dclde-max-positive", type=int, default=400)
    parser.add_argument("--dclde-max-hard-negative", type=int, default=200)
    parser.add_argument("--max-gap-windows-per-clip", type=int, default=3)
    args = parser.parse_args()

    weekend_root = Path(args.weekend_root)
    dclde_gcs_object_lists = [Path(path) for path in args.dclde_gcs_object_list]
    default_gcs_objects = weekend_root / DEFAULT_DCLDE_GCS_OBJECTS
    if not dclde_gcs_object_lists and default_gcs_objects.exists():
        dclde_gcs_object_lists = [default_gcs_objects]
    summary = build_dry_run(
        weekend_root=weekend_root,
        output_dir=Path(args.output_dir),
        e09_manifest=Path(args.e09_manifest) if args.e09_manifest else weekend_root / DEFAULT_E09_MANIFEST,
        dclde_annotations=Path(args.dclde_annotations) if args.dclde_annotations else weekend_root / DEFAULT_DCLDE_ANNOTATIONS,
        dclde_gcs_object_lists=dclde_gcs_object_lists,
        dclde_max_positive=int(args.dclde_max_positive),
        dclde_max_hard_negative=int(args.dclde_max_hard_negative),
        max_gap_windows_per_clip=int(args.max_gap_windows_per_clip),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
