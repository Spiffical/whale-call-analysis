#!/usr/bin/env python3
"""Build species-first manifests with auto-screened hard negatives.

This script intentionally keeps the visual decision narrow: rows marked
`unlabeled_signal_suspect` by the contact-sheet pass are excluded from the
negative set, while ambiguous rows remain hard negatives rather than clean
reviewed background.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.multilabel.build_negative_window_manifest import leaked_groups_by_split  # noqa: E402
from src.dataset.multilabel import (  # noqa: E402
    clean_text,
    label_ids_from_row,
    read_csv_rows,
    split_pipe,
    write_csv_rows,
)


PRIMARY_LABELS = {"species:Bp", "species:Bm", "species:Mn", "species:Oo"}
OBVIOUS_SIGNAL_LABEL = "unlabeled_signal_suspect"


def _label_set(row: Mapping[str, Any]) -> set[str]:
    labels = set(label_ids_from_row(dict(row)))
    for field in ("canonical_label_ids", "source_label_ids", "analysis_label_ids", "target_label_ids"):
        labels.update(split_pipe(row.get(field)))
    return {label for label in labels if label}


def _has_primary(row: Mapping[str, Any]) -> bool:
    return bool(_label_set(row) & PRIMARY_LABELS)


def _classify_source(row: Mapping[str, Any]) -> str:
    text = " ".join(
        clean_text(row.get(field)).lower()
        for field in ("source_dataset", "source_dataset_raw", "source_provider", "source_audio", "mat_path", "clip")
    )
    if "dclde" in text:
        return "DCLDE"
    if "biodcase" in text or "task2" in text:
        return "BioDCASE"
    if "iclisten" in text or "final2025" in text or "part2" in text:
        return "ONC"
    return "unknown"


def _load_model_labels(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        item_id = clean_text(row.get("item_id"))
        if item_id:
            out[item_id] = dict(row)
    return out


def _gap_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    clip = Path(clean_text(row.get("source_audio") or row.get("filename") or row.get("clip"))).name
    begin = f"{float(clean_text(row.get('begin_s') or row.get('begin_time_s') or 0.0)):.6f}"
    end = f"{float(clean_text(row.get('end_s') or row.get('end_time_s') or 0.0)):.6f}"
    return clip, begin, end


def _load_gap_report(path: Optional[Path]) -> Dict[Tuple[str, str, str], str]:
    if path is None or not path.exists():
        return {}
    out: Dict[Tuple[str, str, str], str] = {}
    for row in read_csv_rows(path):
        clip = Path(clean_text(row.get("clip"))).name
        begin = f"{float(clean_text(row.get('begin_s') or 0.0)):.6f}"
        end = f"{float(clean_text(row.get('end_s') or 0.0)):.6f}"
        mat_path = clean_text(row.get("out_mat"))
        if clip and mat_path:
            out[(clip, begin, end)] = mat_path
    return out


def _resolve_under_root(value: str, root: Optional[Path]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute() or root is None:
        return str(path)
    return str((root / path).resolve())


def _force_labels(row: Dict[str, Any], labels: Sequence[str]) -> Dict[str, Any]:
    labels_text = "|".join(labels)
    row["source_label_ids"] = labels_text
    row["canonical_label_ids"] = labels_text
    row["label_ids"] = labels_text
    row["analysis_label_ids"] = labels_text
    row["is_background"] = "0" if labels else "1"
    return row


def _dclde_positive_labels(row: Mapping[str, Any]) -> List[str]:
    source_class = clean_text(row.get("source_class_species"))
    if source_class == "KW":
        return ["species:Oo"]
    if source_class == "HW":
        return ["species:Mn"]
    return []


def build_positive_manifest(
    *,
    source_csv: Path,
    output_csv: Path,
    source_kind: str,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    rows_out: List[Dict[str, Any]] = []
    skipped = Counter()
    labels = Counter()
    for raw in read_csv_rows(source_csv):
        row = dict(raw)
        if source_kind == "DCLDE":
            forced = _dclde_positive_labels(row)
            if not forced:
                skipped["dclde_non_primary_or_negative"] += 1
                continue
            row = _force_labels(row, forced)
            row["species_code"] = forced[0].split(":", 1)[1]
            row["species"] = row["species_code"]
            row["review_status"] = "reviewed"
        elif not _has_primary(row):
            skipped["no_primary_label"] += 1
            continue
        row["mat_path"] = _resolve_under_root(clean_text(row.get("mat_path")), source_root)
        row["source_kind"] = source_kind
        rows_out.append(row)
        labels.update(label for label in _label_set(row) if label in PRIMARY_LABELS)
    write_csv_rows(output_csv, rows_out)
    return {
        "path": str(output_csv),
        "source_kind": source_kind,
        "row_count": len(rows_out),
        "skipped": dict(skipped),
        "primary_label_counts": dict(labels.most_common()),
    }


def build_negative_manifest(
    *,
    negative_csv: Path,
    output_csv: Path,
    gap_calls_csv: Path,
    excluded_csv: Path,
    model_labels_csv: Optional[Path],
    gap_report_csv: Optional[Path],
    source_roots: Optional[Mapping[str, Path]] = None,
    require_existing_mats: bool = False,
) -> Dict[str, Any]:
    model_labels = _load_model_labels(model_labels_csv)
    gap_paths = _load_gap_report(gap_report_csv)
    rows_out: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    gap_calls: List[Dict[str, Any]] = []
    pending_gap_count = 0
    bucket_counts = Counter()
    source_counts = Counter()
    auto_counts = Counter()
    missing_gap_mat_examples: List[str] = []
    missing_mat_path_examples: List[str] = []
    missing_mat_path_count = 0
    resolved_relative_mat_count = 0
    source_roots = source_roots or {}

    for raw in read_csv_rows(negative_csv):
        row = dict(raw)
        item_id = clean_text(row.get("item_id"))
        auto = model_labels.get(item_id, {})
        auto_label = clean_text(auto.get("model_assisted_review_label"))
        if auto_label == OBVIOUS_SIGNAL_LABEL:
            row["auto_screen_decision"] = "excluded_obvious_signal"
            row["auto_screen_label"] = auto_label
            row["auto_screen_notes"] = clean_text(auto.get("visual_notes"))
            excluded.append(row)
            continue
        if _has_primary(row):
            row["auto_screen_decision"] = "excluded_primary_label_present"
            excluded.append(row)
            continue

        bucket = clean_text(row.get("negative_bucket")) or "ambiguous_hard_negative"
        row["auto_screen_label"] = auto_label or "not_visually_sampled"
        row["auto_screen_notes"] = clean_text(auto.get("visual_notes"))
        row["auto_screen_decision"] = "kept_as_hard_negative"
        row["review_status"] = "auto_screened_hard_negative"
        row["label_ids"] = ""
        row["canonical_label_ids"] = ""
        row["source_label_ids"] = ""
        row["is_background"] = "1"
        row["source_kind"] = _classify_source(row)

        if bucket == "primary_adjacent_gap":
            clip, begin, end = _gap_key(row)
            if gap_paths:
                mat_path = gap_paths.get((clip, begin, end))
                if not mat_path:
                    if len(missing_gap_mat_examples) < 10:
                        missing_gap_mat_examples.append(f"{clip}:{begin}-{end}")
                    row["auto_screen_decision"] = "excluded_missing_gap_mat"
                    excluded.append(row)
                    continue
                row["mat_path"] = mat_path
            else:
                pending_gap_count += 1
                gap_calls.append(
                    {
                        "clip": clip,
                        "begin_s": begin,
                        "end_s": end,
                        "item_id": item_id,
                        "source_audio_original": clean_text(row.get("source_audio")),
                    }
                )
                row["mat_path"] = ""
        else:
            raw_mat_path = clean_text(row.get("mat_path"))
            resolved_mat_path = _resolve_under_root(raw_mat_path, source_roots.get(row["source_kind"]))
            if raw_mat_path and resolved_mat_path != raw_mat_path:
                resolved_relative_mat_count += 1
            row["mat_path"] = resolved_mat_path

        if require_existing_mats and clean_text(row.get("mat_path")):
            mat_path = Path(clean_text(row.get("mat_path")))
            if not mat_path.exists():
                missing_mat_path_count += 1
                if len(missing_mat_path_examples) < 10:
                    missing_mat_path_examples.append(str(mat_path))
                row["auto_screen_decision"] = "excluded_missing_mat_path"
                excluded.append(row)
                continue

        rows_out.append(row)
        bucket_counts[bucket] += 1
        source_counts[row["source_kind"]] += 1
        auto_counts[row["auto_screen_label"]] += 1

    if not gap_calls and not gap_paths:
        gap_calls = [
            {
                "clip": _gap_key(row)[0],
                "begin_s": _gap_key(row)[1],
                "end_s": _gap_key(row)[2],
                "item_id": clean_text(row.get("item_id")),
                "source_audio_original": clean_text(row.get("source_audio")),
            }
            for row in read_csv_rows(negative_csv)
            if clean_text(row.get("negative_bucket")) == "primary_adjacent_gap"
            and clean_text(model_labels.get(clean_text(row.get("item_id")), {}).get("model_assisted_review_label")) != OBVIOUS_SIGNAL_LABEL
        ]
    write_csv_rows(gap_calls_csv, gap_calls)
    write_csv_rows(output_csv, rows_out)
    write_csv_rows(excluded_csv, excluded)
    leaked = leaked_groups_by_split(rows_out)
    return {
        "path": str(output_csv),
        "row_count": len(rows_out),
        "excluded_row_count": len(excluded),
        "gap_calls_csv": str(gap_calls_csv),
        "gap_calls_row_count": len(gap_calls),
        "pending_gap_mat_count": int(pending_gap_count),
        "bucket_counts": dict(bucket_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "auto_screen_label_counts": dict(auto_counts.most_common()),
        "leaked_group_count": len(leaked),
        "leaked_group_examples": dict(list(leaked.items())[:10]),
        "missing_gap_mat_examples": missing_gap_mat_examples,
        "missing_mat_path_count": missing_mat_path_count,
        "missing_mat_path_examples": missing_mat_path_examples,
        "resolved_relative_mat_count": resolved_relative_mat_count,
    }


def build_manifests(
    *,
    output_dir: Path,
    onc_csv: Path,
    biodcase_csv: Optional[Path],
    dclde_csv: Path,
    negative_csv: Path,
    model_labels_csv: Optional[Path],
    gap_report_csv: Optional[Path],
    onc_root: Optional[Path] = None,
    biodcase_root: Optional[Path] = None,
    dclde_root: Optional[Path] = None,
    require_existing_mats: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"output_dir": str(output_dir), "positive_manifests": {}}
    summary["positive_manifests"]["ONC"] = build_positive_manifest(
        source_csv=onc_csv,
        output_csv=tables_dir / "onc_primary_positive_manifest.csv",
        source_kind="ONC",
        source_root=onc_root,
    )
    if biodcase_csv is not None:
        summary["positive_manifests"]["BioDCASE"] = build_positive_manifest(
            source_csv=biodcase_csv,
            output_csv=tables_dir / "biodcase_primary_positive_manifest.csv",
            source_kind="BioDCASE",
            source_root=biodcase_root,
        )
    summary["positive_manifests"]["DCLDE"] = build_positive_manifest(
        source_csv=dclde_csv,
        output_csv=tables_dir / "dclde_kw_hw_primary_positive_manifest.csv",
        source_kind="DCLDE",
        source_root=dclde_root,
    )
    source_roots = {
        key: root
        for key, root in {
            "ONC": onc_root,
            "BioDCASE": biodcase_root,
            "DCLDE": dclde_root,
        }.items()
        if root is not None
    }
    summary["negative_manifest"] = build_negative_manifest(
        negative_csv=negative_csv,
        output_csv=tables_dir / "autoscreened_negative_manifest.csv",
        gap_calls_csv=tables_dir / "primary_adjacent_gap_calls_for_mat.csv",
        excluded_csv=tables_dir / "autoscreened_negative_excluded_rows.csv",
        model_labels_csv=model_labels_csv,
        gap_report_csv=gap_report_csv,
        source_roots=source_roots,
        require_existing_mats=require_existing_mats,
    )
    (output_dir / "autoscreened_training_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--onc-csv", required=True)
    parser.add_argument("--biodcase-csv", default="")
    parser.add_argument("--dclde-csv", required=True)
    parser.add_argument("--negative-csv", required=True)
    parser.add_argument("--model-labels-csv", default="")
    parser.add_argument("--gap-report-csv", default="")
    parser.add_argument("--onc-root", default="")
    parser.add_argument("--biodcase-root", default="")
    parser.add_argument("--dclde-root", default="")
    parser.add_argument("--require-existing-mats", action="store_true")
    args = parser.parse_args()

    summary = build_manifests(
        output_dir=Path(args.output_dir),
        onc_csv=Path(args.onc_csv),
        biodcase_csv=Path(args.biodcase_csv) if args.biodcase_csv else None,
        dclde_csv=Path(args.dclde_csv),
        negative_csv=Path(args.negative_csv),
        model_labels_csv=Path(args.model_labels_csv) if args.model_labels_csv else None,
        gap_report_csv=Path(args.gap_report_csv) if args.gap_report_csv else None,
        onc_root=Path(args.onc_root) if args.onc_root else None,
        biodcase_root=Path(args.biodcase_root) if args.biodcase_root else None,
        dclde_root=Path(args.dclde_root) if args.dclde_root else None,
        require_existing_mats=bool(args.require_existing_mats),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
