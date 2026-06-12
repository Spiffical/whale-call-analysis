#!/usr/bin/env python3
"""Export a multispecies standardized manifest to a SSAMBA-compatible H5 file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except Exception:
    h5py = None

try:
    from scipy import ndimage
except Exception as exc:  # pragma: no cover - exercised by import failure only
    raise RuntimeError("scipy is required for E123 H5 export") from exc

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.multiband import (  # noqa: E402
    _crop_freq,
    _crop_time,
    _extract_spectrogram_raw,
    _load_mat_data,
    _power_to_db_norm,
    _resolve_path,
)
from src.dataset.multilabel import (  # noqa: E402
    clean_text,
    label_ids_from_row,
    read_csv_rows,
    split_pipe,
)


DEFAULT_TARGET_LABELS = {
    "species:Bm": "Bm",
    "species:Bp": "Bp",
    "species:Mn": "Mn",
}


def parse_target_label_map(value: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in str(value or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        label_id, sep, label_name = chunk.partition("=")
        if not sep:
            raise ValueError(f"Target label mapping must look like label_id=name, got {chunk!r}")
        out[label_id.strip()] = label_name.strip()
    if not out:
        raise ValueError("At least one target label mapping is required")
    return out


def parse_size(value: str) -> Tuple[int, int]:
    left, sep, right = str(value).lower().partition("x")
    if not sep:
        raise ValueError(f"Size must look like FxT, got {value!r}")
    return int(left), int(right)


def all_known_label_ids(row: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    for label_id in label_ids_from_row(dict(row)):
        ids.append(label_id)
    for key in ("target_label_ids", "canonical_label_ids", "source_label_ids", "analysis_label_ids"):
        for label_id in split_pipe(row.get(key)):
            ids.append(label_id)
    return list(dict.fromkeys(ids))


def label_string_for_row(
    row: Mapping[str, Any],
    *,
    target_label_map: Mapping[str, str],
    non_target_mode: str,
    ambiguous_mode: str,
) -> Tuple[Optional[str], str]:
    labels = all_known_label_ids(row)
    target_names = [target_label_map[label_id] for label_id in labels if label_id in target_label_map]
    if len(target_names) == 1:
        return target_names[0], "target"
    if len(target_names) > 1:
        if ambiguous_mode == "skip":
            return None, "ambiguous_target"
        if ambiguous_mode == "first":
            return target_names[0], "target_first"
        if ambiguous_mode == "semicolon":
            return ";".join(dict.fromkeys(target_names)), "target_multilabel"
        raise ValueError(f"Unknown ambiguous mode: {ambiguous_mode}")

    if labels:
        if non_target_mode == "skip":
            return None, "non_target_labeled"
        if non_target_mode == "normal":
            return "normal", "non_target_as_normal"
        raise ValueError(f"Unknown non-target mode: {non_target_mode}")
    return "normal", "background"


def resolve_band_path(row: Mapping[str, Any], *, band: str, dataset_root: Optional[Path]) -> Path:
    raw = row.get(f"{band}_mat_path") or row.get("mat_path")
    return _resolve_path(raw, dataset_root)


def crop_start_seconds(*, context_seconds: float, crop_time_seconds: float) -> float:
    return max(0.0, (float(context_seconds) - float(crop_time_seconds)) / 2.0)


def crop_start_offsets(
    *,
    context_seconds: float,
    crop_time_seconds: float,
    crops_per_row: int,
    centered_if_single: bool = True,
) -> List[float]:
    crops = max(1, int(crops_per_row))
    usable = max(0.0, float(context_seconds) - float(crop_time_seconds))
    if crops == 1:
        if centered_if_single:
            return [crop_start_seconds(context_seconds=context_seconds, crop_time_seconds=crop_time_seconds)]
        return [0.0]
    if crops == 2:
        return [0.0, usable]
    step = usable / float(crops - 1) if crops > 1 else 0.0
    return [round(i * step, 6) for i in range(crops)]


def resize_spec(spec: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    target_f, target_t = output_shape
    if spec.shape == output_shape:
        return np.asarray(spec, dtype=np.float32)
    zoom = (target_f / max(1, spec.shape[0]), target_t / max(1, spec.shape[1]))
    resized = ndimage.zoom(np.asarray(spec, dtype=np.float32), zoom, order=1)
    if resized.shape != output_shape:
        out = np.zeros(output_shape, dtype=np.float32)
        f = min(target_f, resized.shape[0])
        t = min(target_t, resized.shape[1])
        out[:f, :t] = resized[:f, :t]
        if f < target_f and f > 0:
            out[f:, :t] = out[f - 1 : f, :t]
        if t < target_t and t > 0:
            out[:, t:] = out[:, t - 1 : t]
        return out
    return resized.astype(np.float32)


def extract_band_spectrogram(
    row: Mapping[str, Any],
    *,
    band: str,
    dataset_root: Optional[Path],
    band_crop_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    context_seconds: float,
    crop_time_seconds: float,
    crop_start_s: Optional[float] = None,
) -> np.ndarray:
    path = resolve_band_path(row, band=band, dataset_root=dataset_root)
    data = _load_mat_data(path)
    spec, kind, _, times = _extract_spectrogram_raw(data, path, band=band)
    spec = np.asarray(spec, dtype=np.float32)
    if kind == "power":
        spec = _power_to_db_norm(spec)
    target_f, target_t = band_crop_shape
    spec = _crop_freq(spec, int(target_f))
    spec, _ = _crop_time(
        spec,
        times=times,
        crop_start_s=(
            crop_start_seconds(context_seconds=context_seconds, crop_time_seconds=crop_time_seconds)
            if crop_start_s is None
            else float(crop_start_s)
        ),
        target_t=int(target_t),
    )
    spec = np.nan_to_num(spec, nan=-100.0, neginf=-100.0, posinf=0.0).astype(np.float32)
    return resize_spec(spec, output_shape)


def write_h5(
    output_h5: Path,
    rows: Sequence[Mapping[str, Any]],
    label_strings: Sequence[str],
    *,
    target_label_names: Sequence[str],
    sources: Sequence[str],
    item_ids: Sequence[str],
    splits: Sequence[str],
    source_kinds: Sequence[str],
    data_arrays: Sequence[np.ndarray],
    compression: str,
) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to write the E123 SSAMBA H5 dataset")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((len(rows), len(target_label_names)), dtype=np.int8)
    label_to_idx = {label: idx for idx, label in enumerate(target_label_names)}
    for row_idx, label_str in enumerate(label_strings):
        for token in label_str.split(";"):
            idx = label_to_idx.get(token)
            if idx is not None:
                labels[row_idx, idx] = 1

    spectrograms = np.asarray(data_arrays, dtype=np.float32)[..., np.newaxis]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_h5, "w") as h5:
        h5.create_dataset(
            "spectrograms",
            data=spectrograms,
            chunks=(1, spectrograms.shape[1], spectrograms.shape[2], 1),
            compression=compression,
        )
        h5.create_dataset("labels", data=labels, chunks=True, compression=compression)
        h5.create_dataset("sources", data=np.asarray(sources, dtype=object), dtype=string_dtype)
        h5.create_dataset(
            "label_strings",
            data=np.asarray(label_strings, dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset("item_ids", data=np.asarray(item_ids, dtype=object), dtype=string_dtype)
        h5.create_dataset("splits", data=np.asarray(splits, dtype=object), dtype=string_dtype)
        h5.create_dataset(
            "source_kinds",
            data=np.asarray(source_kinds, dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "anomaly_label_names",
            data=np.asarray(target_label_names, dtype=object),
            dtype=string_dtype,
        )
        h5.attrs["schema"] = "e123-ssamba-multispecies-v1"


def select_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    splits: Optional[set[str]],
    source_kinds: Optional[set[str]],
    target_label_map: Mapping[str, str],
    non_target_mode: str,
    ambiguous_mode: str,
    max_normal: int,
    max_per_target: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[str], Counter[str]]:
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}
    skip_reasons: Counter[str] = Counter()
    for row in rows:
        split = clean_text(row.get("split"))
        if splits is not None and split not in splits:
            skip_reasons["split_filter"] += 1
            continue
        source_kind = clean_text(row.get("source_kind"))
        if source_kinds is not None and source_kind not in source_kinds:
            skip_reasons["source_kind_filter"] += 1
            continue
        label_str, reason = label_string_for_row(
            row,
            target_label_map=target_label_map,
            non_target_mode=non_target_mode,
            ambiguous_mode=ambiguous_mode,
        )
        if label_str is None:
            skip_reasons[reason] += 1
            continue
        buckets.setdefault(label_str, []).append((dict(row), reason))

    selected_pairs: List[Tuple[Dict[str, Any], str]] = []
    for label_str, pairs in sorted(buckets.items()):
        limit = max_normal if label_str == "normal" else max_per_target
        ordered = list(pairs)
        if limit > 0 and len(ordered) > limit:
            idx = np.sort(rng.choice(len(ordered), size=limit, replace=False))
            ordered = [ordered[int(i)] for i in idx]
            skip_reasons[f"cap_{label_str}"] += len(pairs) - limit
        selected_pairs.extend(ordered)

    selected_pairs.sort(
        key=lambda pair: (
            clean_text(pair[0].get("split")),
            pair[1],
            clean_text(pair[0].get("source_kind")),
            clean_text(pair[0].get("item_id")) or clean_text(pair[0].get("mat_path")),
        )
    )
    return [row for row, _ in selected_pairs], [label for _, label in selected_pairs], skip_reasons


def build_e123_h5(
    *,
    manifest_csv: Path,
    output_h5: Path,
    output_summary: Path,
    dataset_root: Optional[Path],
    band: str,
    band_crop_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    target_label_map: Mapping[str, str],
    splits: Optional[set[str]],
    source_kinds: Optional[set[str]],
    non_target_mode: str,
    ambiguous_mode: str,
    max_normal: int,
    max_per_target: int,
    normal_crops_per_row: int,
    context_seconds: float,
    crop_time_seconds: float,
    seed: int,
    compression: str,
) -> Dict[str, Any]:
    rows = read_csv_rows(manifest_csv)
    selected_rows, selected_label_strings, skip_reasons = select_rows(
        rows,
        splits=splits,
        source_kinds=source_kinds,
        target_label_map=target_label_map,
        non_target_mode=non_target_mode,
        ambiguous_mode=ambiguous_mode,
        max_normal=max_normal,
        max_per_target=max_per_target,
        seed=seed,
    )
    if not selected_rows:
        raise ValueError("No rows selected for E123 H5 export")

    data_arrays: List[np.ndarray] = []
    kept_rows: List[Dict[str, Any]] = []
    kept_labels: List[str] = []
    missing_or_bad: List[Dict[str, str]] = []
    for row, label_str in zip(selected_rows, selected_label_strings):
        crop_starts = [None]
        if label_str == "normal" and int(normal_crops_per_row) > 1:
            crop_starts = [
                float(start)
                for start in crop_start_offsets(
                    context_seconds=context_seconds,
                    crop_time_seconds=crop_time_seconds,
                    crops_per_row=int(normal_crops_per_row),
                    centered_if_single=True,
                )
            ]
        for crop_index, crop_start_s in enumerate(crop_starts):
            try:
                arr = extract_band_spectrogram(
                    row,
                    band=band,
                    dataset_root=dataset_root,
                    band_crop_shape=band_crop_shape,
                    output_shape=output_shape,
                    context_seconds=context_seconds,
                    crop_time_seconds=crop_time_seconds,
                    crop_start_s=crop_start_s,
                )
            except Exception as exc:
                missing_or_bad.append(
                    {
                        "item_id": clean_text(row.get("item_id")),
                        "split": clean_text(row.get("split")),
                        "source_kind": clean_text(row.get("source_kind")),
                        "reason": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            kept_row = dict(row)
            if crop_start_s is not None:
                kept_row["ssl_crop_index"] = str(crop_index)
                kept_row["ssl_crop_start_s"] = f"{float(crop_start_s):.6f}"
            data_arrays.append(arr)
            kept_rows.append(kept_row)
            kept_labels.append(label_str)

    if not kept_rows:
        raise ValueError("All selected rows failed during MAT loading")

    sources = [
        clean_text(row.get("source_audio") or row.get("filename") or row.get("mat_path") or row.get(f"{band}_mat_path"))
        for row in kept_rows
    ]
    item_ids = []
    for row in kept_rows:
        item_id = clean_text(row.get("item_id")) or Path(clean_text(row.get(f"{band}_mat_path") or row.get("mat_path"))).stem
        crop_index = clean_text(row.get("ssl_crop_index"))
        item_ids.append(f"{item_id}::crop{crop_index}" if crop_index else item_id)
    row_splits = [clean_text(row.get("split")) for row in kept_rows]
    row_source_kinds = [clean_text(row.get("source_kind")) for row in kept_rows]
    target_label_names = list(dict.fromkeys(target_label_map.values()))
    write_h5(
        output_h5,
        kept_rows,
        kept_labels,
        target_label_names=target_label_names,
        sources=sources,
        item_ids=item_ids,
        splits=row_splits,
        source_kinds=row_source_kinds,
        data_arrays=data_arrays,
        compression=compression,
    )

    counts = Counter(kept_labels)
    summary: Dict[str, Any] = {
        "manifest_csv": str(manifest_csv),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "output_h5": str(output_h5),
        "rows_read": len(rows),
        "rows_selected_before_mat_load": len(selected_rows),
        "rows_written": len(kept_rows),
        "label_counts": dict(sorted(counts.items())),
        "split_counts": dict(sorted(Counter(row_splits).items())),
        "source_kind_counts": dict(sorted(Counter(row_source_kinds).items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "mat_load_failures": len(missing_or_bad),
        "mat_load_failure_examples": missing_or_bad[:20],
        "target_label_map": dict(target_label_map),
        "band": band,
        "band_crop_shape": list(band_crop_shape),
        "output_shape": list(output_shape),
        "non_target_mode": non_target_mode,
        "ambiguous_mode": ambiguous_mode,
        "max_normal": max_normal,
        "max_per_target": max_per_target,
        "normal_crops_per_row": int(normal_crops_per_row),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if missing_or_bad:
        failure_csv = output_summary.with_name("e123_h5_mat_load_failures.csv")
        with failure_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["item_id", "split", "source_kind", "reason", "message"])
            writer.writeheader()
            writer.writerows(missing_or_bad)
        summary["failure_csv"] = str(failure_csv)
        output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--band", default="low")
    parser.add_argument("--band-crop-shape", default="391x50")
    parser.add_argument("--output-shape", default="512x512")
    parser.add_argument(
        "--target-label-map",
        default=",".join(f"{key}={value}" for key, value in DEFAULT_TARGET_LABELS.items()),
        help="Comma-separated label_id=name mappings, e.g. species:Bm=Bm,species:Bp=Bp",
    )
    parser.add_argument("--splits", default="train,val", help="Comma-separated manifest splits to export; empty means all")
    parser.add_argument("--source-kind", action="append", default=[], help="Optional source_kind filter; may be repeated")
    parser.add_argument("--non-target-mode", choices=["skip", "normal"], default="skip")
    parser.add_argument("--ambiguous-mode", choices=["skip", "first", "semicolon"], default="skip")
    parser.add_argument("--max-normal", type=int, default=10000, help="Cap normal/background rows; <=0 keeps all")
    parser.add_argument("--max-per-target", type=int, default=0, help="Cap each target class; <=0 keeps all")
    parser.add_argument(
        "--normal-crops-per-row",
        type=int,
        default=1,
        help="Export this many deterministic 10s crops for each normal/background row; positives stay single-crop.",
    )
    parser.add_argument("--context-seconds", type=float, default=40.0)
    parser.add_argument("--crop-time-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", None])
    args = parser.parse_args()

    output_h5 = Path(args.output_h5)
    output_summary = Path(args.output_summary) if args.output_summary else output_h5.with_suffix(".summary.json")
    splits = {token.strip() for token in str(args.splits).split(",") if token.strip()} or None
    source_kinds = set(args.source_kind) if args.source_kind else None
    summary = build_e123_h5(
        manifest_csv=Path(args.manifest_csv),
        output_h5=output_h5,
        output_summary=output_summary,
        dataset_root=Path(args.dataset_root).resolve() if args.dataset_root else None,
        band=str(args.band),
        band_crop_shape=parse_size(args.band_crop_shape),
        output_shape=parse_size(args.output_shape),
        target_label_map=parse_target_label_map(args.target_label_map),
        splits=splits,
        source_kinds=source_kinds,
        non_target_mode=str(args.non_target_mode),
        ambiguous_mode=str(args.ambiguous_mode),
        max_normal=int(args.max_normal),
        max_per_target=int(args.max_per_target),
        normal_crops_per_row=int(args.normal_crops_per_row),
        context_seconds=float(args.context_seconds),
        crop_time_seconds=float(args.crop_time_seconds),
        seed=int(args.seed),
        compression=str(args.compression),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
