#!/usr/bin/env python3
"""Attach multiband MAT paths to an existing standardized manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multilabel import read_csv_rows, write_csv_rows  # noqa: E402


DEFAULT_BANDS = ("low", "mid", "high")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _read_report(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: Mapping[str, Any]) -> str:
    item_id = _clean(row.get("item_id"))
    if item_id:
        return f"item:{item_id}"
    mat_path = _clean(row.get("mat_path"))
    if mat_path:
        return f"mat:{Path(mat_path).stem}"
    clip = _clean(row.get("clip"))
    begin_s = _clean(row.get("begin_s") or row.get("begin_time_s"))
    end_s = _clean(row.get("end_s") or row.get("end_time_s"))
    return f"window:{clip}:{begin_s}:{end_s}"


def build_multiband_manifest(
    *,
    manifest_csv: Path,
    report_csvs: Sequence[Path],
    output_dir: Path,
    bands: Sequence[str] = DEFAULT_BANDS,
    set_mat_path_to_band: str = "low",
) -> Dict[str, Any]:
    manifest_rows = read_csv_rows(manifest_csv)
    report_rows: List[Dict[str, str]] = []
    for path in report_csvs:
        report_rows.extend(_read_report(path))
    band_cols = [f"{band}_mat_path" for band in bands]

    by_key: Dict[str, Dict[str, str]] = {}
    duplicate_keys: List[str] = []
    for row in report_rows:
        key = _row_key(row)
        if key in by_key:
            duplicate_keys.append(key)
        by_key[key] = row

    out_rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for row in manifest_rows:
        key = _row_key(row)
        report = by_key.get(key)
        if report is None:
            missing.append({"key": key, "item_id": _clean(row.get("item_id")), "mat_path": _clean(row.get("mat_path"))})
            continue
        out = dict(row)
        for col in band_cols:
            value = _clean(report.get(col))
            if not value:
                missing.append({"key": key, "item_id": _clean(row.get("item_id")), "missing_column": col})
                report = None
                break
            out[col] = value
        if report is None:
            continue
        if set_mat_path_to_band:
            out["mat_path"] = out.get(f"{set_mat_path_to_band}_mat_path", out.get("mat_path", ""))
        out_rows.append(out)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "standardized_manifest.csv"
    write_csv_rows(out_manifest, out_rows)
    summary = {
        "input_manifest": str(manifest_csv),
        "report_csvs": [str(path) for path in report_csvs],
        "output_manifest": str(out_manifest),
        "input_rows": len(manifest_rows),
        "report_rows": len(report_rows),
        "output_rows": len(out_rows),
        "missing_rows": len(missing),
        "missing_examples": missing[:25],
        "duplicate_report_keys": len(duplicate_keys),
        "duplicate_report_key_examples": duplicate_keys[:25],
        "bands": list(bands),
        "set_mat_path_to_band": set_mat_path_to_band,
    }
    (output_dir / "multiband_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if missing:
        raise RuntimeError(f"{len(missing)} manifest rows could not be matched to multiband MATs")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-csv", action="append", required=True)
    parser.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    parser.add_argument("--set-mat-path-to-band", default="low")
    args = parser.parse_args()
    summary = build_multiband_manifest(
        manifest_csv=Path(args.manifest_csv),
        report_csvs=[Path(path) for path in args.report_csv],
        output_dir=Path(args.output_dir),
        bands=[token.strip() for token in str(args.bands).split(",") if token.strip()],
        set_mat_path_to_band=str(args.set_mat_path_to_band),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
