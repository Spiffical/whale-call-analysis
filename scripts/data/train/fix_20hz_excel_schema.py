#!/usr/bin/env python3
"""
Repair known schema-shift rows in FinWhale20Hz_CallLibrary_Rannankari.xlsx.

The affected block contains two extra timestamp fields inserted after
"End time (UTC)", which shifts subsequent columns to the right.
This script creates a new corrected Excel file and leaves the original untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _to_seconds(value: object) -> float:
    """Convert mixed time-like values to seconds; return NaN if conversion fails."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
        return (
            float(value.hour) * 3600.0
            + float(value.minute) * 60.0
            + float(value.second)
            + float(getattr(value, "microsecond", 0)) / 1e6
        )
    text = str(value).strip()
    if not text:
        return float("nan")
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(text)
    except Exception:
        return float("nan")


def _contiguous_ranges(indices: List[int]) -> List[Tuple[int, int]]:
    """Return contiguous [start, end] index ranges from sorted indices."""
    if not indices:
        return []
    ranges: List[Tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = idx
        prev = idx
    ranges.append((start, prev))
    return ranges


def detect_shifted_rows(df: pd.DataFrame) -> pd.Series:
    """Detect rows that appear shifted by two columns."""
    begin_num = df["begin time (s)"].apply(_to_seconds)
    end_num = df["end time (s)"].apply(_to_seconds)
    duration_num = df["Duration (s)"].apply(_to_seconds)
    low_num = pd.to_numeric(df["low freq"], errors="coerce")
    high_num = pd.to_numeric(df["high freq"], errors="coerce")

    # In the malformed block, begin/end are non-numeric and low > high after shift.
    non_numeric_times = pd.to_numeric(df["begin time (s)"], errors="coerce").isna() | pd.to_numeric(
        df["end time (s)"], errors="coerce"
    ).isna()
    shifted_payload = duration_num.notna() & low_num.notna() & high_num.notna()
    inverted_freq = low_num > high_num
    return non_numeric_times & shifted_payload & inverted_freq


def repair_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Repair shifted rows and return (fixed_df, shifted_row_mask)."""
    required = [
        "begin time (s)",
        "end time (s)",
        "Duration (s)",
        "low freq",
        "high freq",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fixed = df.copy()
    shifted = detect_shifted_rows(fixed)
    if not shifted.any():
        return fixed, shifted

    # Undo the two-column shift for affected rows from begin-time onward.
    start_idx = list(fixed.columns).index("begin time (s)")
    tail_cols = list(fixed.columns)[start_idx:]
    arr = fixed.loc[shifted, tail_cols].to_numpy(dtype=object)
    shifted_arr = np.empty_like(arr, dtype=object)
    shifted_arr[:, :] = np.nan
    shifted_arr[:, :-2] = arr[:, 2:]
    fixed.loc[shifted, tail_cols] = shifted_arr

    # Promote known category text when call category is missing in shifted rows.
    if "Call Category" in fixed.columns:
        category = fixed["Call Category"]
        for src_col in ("Comments", "Unnamed: 18", "Unnamed: 17"):
            if src_col not in fixed.columns:
                continue
            src_text = fixed[src_col].astype(str).str.strip()
            label_like = src_text.str.lower().isin({"20 hz", "20hz", "20 hz call", "20hz call"})
            fill_mask = shifted & category.isna() & fixed[src_col].notna() & label_like
            fixed.loc[fill_mask, "Call Category"] = src_text[fill_mask]

    # Coerce repaired numeric columns to usable numeric values.
    for col in ("begin time (s)", "end time (s)", "Duration (s)", "low freq", "high freq", "peak freq"):
        if col in fixed.columns:
            fixed[col] = fixed[col].apply(_to_seconds)

    # Fallback for a few rows where begin remains malformed after shift.
    b = pd.to_numeric(fixed["begin time (s)"], errors="coerce")
    e = pd.to_numeric(fixed["end time (s)"], errors="coerce")
    d = pd.to_numeric(fixed["Duration (s)"], errors="coerce")

    bad_begin = shifted & ((b.isna()) | (b > (e + 5.0)))
    fix_from_end_duration = bad_begin & e.notna() & d.notna() & (e >= d)
    fixed.loc[fix_from_end_duration, "begin time (s)"] = e[fix_from_end_duration] - d[fix_from_end_duration]

    b = pd.to_numeric(fixed["begin time (s)"], errors="coerce")
    e = pd.to_numeric(fixed["end time (s)"], errors="coerce")
    d = pd.to_numeric(fixed["Duration (s)"], errors="coerce")
    bad_end = shifted & ((e.isna()) | (e < b))
    fix_end_from_duration = bad_end & b.notna() & d.notna()
    fixed.loc[fix_end_from_duration, "end time (s)"] = b[fix_end_from_duration] + d[fix_end_from_duration]

    return fixed, shifted


def summarize(df: pd.DataFrame, label: str) -> dict:
    """Return basic data-quality diagnostics for key training columns."""
    b = pd.to_numeric(df["begin time (s)"], errors="coerce")
    e = pd.to_numeric(df["end time (s)"], errors="coerce")
    d = e - b
    low = pd.to_numeric(df["low freq"], errors="coerce")
    high = pd.to_numeric(df["high freq"], errors="coerce")
    return {
        "label": label,
        "rows": int(len(df)),
        "begin_non_numeric": int(b.isna().sum()),
        "end_non_numeric": int(e.isna().sum()),
        "duration_negative": int((d < 0).fillna(False).sum()),
        "low_gt_high": int((low > high).fillna(False).sum()),
        "duration_min": float(np.nanmin(d.to_numpy())),
        "duration_max": float(np.nanmax(d.to_numpy())),
        "low_min": float(np.nanmin(low.to_numpy())),
        "low_max": float(np.nanmax(low.to_numpy())),
        "high_min": float(np.nanmin(high.to_numpy())),
        "high_max": float(np.nanmax(high.to_numpy())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair shifted rows in 20Hz fin-whale annotation workbook")
    parser.add_argument(
        "--input",
        type=str,
        default="data/finwhales/FinWhale20Hz_CallLibrary_Rannankari.xlsx",
        help="Input Excel file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/finwhales/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx",
        help="Output Excel file path",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original = pd.read_excel(in_path)
    fixed, shifted_mask = repair_dataframe(original)

    # Write fixed workbook.
    fixed.to_excel(out_path, index=False)

    before = summarize(original, "before")
    after = summarize(fixed, "after")
    shifted_indices = shifted_mask[shifted_mask].index.tolist()
    ranges = _contiguous_ranges(shifted_indices)

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Shifted rows detected: {int(shifted_mask.sum())}")
    if ranges:
        print("Shifted index ranges (0-based):")
        for start, end in ranges:
            print(f"  - {start}..{end} ({end - start + 1} rows)")
    print("")
    print("Diagnostics:")
    for key in (
        "begin_non_numeric",
        "end_non_numeric",
        "duration_negative",
        "low_gt_high",
        "duration_min",
        "duration_max",
        "low_min",
        "low_max",
        "high_min",
        "high_max",
    ):
        print(f"  {key}: {before[key]} -> {after[key]}")


if __name__ == "__main__":
    main()
