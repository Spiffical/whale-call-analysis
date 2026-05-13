"""Multi-band MAT dataset for aligned acoustic spectrogram fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

try:
    import scipy.io as sio
except Exception:
    sio = None

from src.dataset.multilabel import LabelVocabulary, clean_text, label_ids_from_row, read_csv_rows
from src.training.mat_dataset import (
    DB_KEYS,
    FREQ_KEYS,
    POWER_KEYS,
    SPECTRO_KEYS,
    TIME_KEYS,
    _find_key,
    _infer_time_bin_seconds,
    _normalize_db_to_unit,
    _power_to_db_norm,
    _sample_positive_crop_fraction,
)


DEFAULT_BANDS = ("low", "mid", "high")
DEFAULT_BAND_CROP_SHAPES: Dict[str, Tuple[int, int]] = {
    "low": (391, 50),
    "mid": (256, 100),
    "high": (256, 312),
}


def parse_band_crop_shapes(value: str | Mapping[str, Sequence[int]] | None) -> Dict[str, Tuple[int, int]]:
    if value is None or value == "":
        return dict(DEFAULT_BAND_CROP_SHAPES)
    if isinstance(value, Mapping):
        return {str(key): (int(val[0]), int(val[1])) for key, val in value.items()}
    shapes: Dict[str, Tuple[int, int]] = {}
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, shape_text = chunk.partition(":")
        if not name or not shape_text:
            raise ValueError(f"Band crop shape must look like band:FxT, got {chunk!r}")
        freq_text, sep, time_text = shape_text.lower().partition("x")
        if not sep:
            raise ValueError(f"Band crop shape must look like band:FxT, got {chunk!r}")
        shapes[name.strip()] = (int(freq_text), int(time_text))
    return shapes


def _resolve_path(value: Any, dataset_root: Optional[Path]) -> Path:
    raw = clean_text(value)
    if not raw:
        raise ValueError("Empty multiband MAT path")
    path = Path(raw)
    if path.is_absolute():
        return path
    if dataset_root is not None:
        return (dataset_root / path).resolve()
    return path.resolve()


def _load_spectrogram_raw(path: Path) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    if sio is None:
        raise RuntimeError("scipy is required to load .mat files")
    data = sio.loadmat(str(path), simplify_cells=True)
    key = _find_key(data, POWER_KEYS)
    kind = "power"
    if key is None:
        key = _find_key(data, DB_KEYS) or _find_key(data, SPECTRO_KEYS)
        kind = "db"
    if key is None:
        raise KeyError(f"No spectrogram-like key found in {path.name}")
    spec = np.asarray(data[key])
    if spec.ndim != 2:
        raise ValueError(f"Unexpected spectrogram ndim {spec.ndim} in {path.name}")

    freq_key = _find_key(data, FREQ_KEYS)
    time_key = _find_key(data, TIME_KEYS)
    freqs = np.asarray(data[freq_key]).squeeze() if freq_key in data else None
    times = np.asarray(data[time_key]).squeeze() if time_key in data else None
    if freqs is not None and times is not None:
        freq_len = int(np.asarray(freqs).ravel().shape[0])
        time_len = int(np.asarray(times).ravel().shape[0])
        rows, cols = spec.shape[:2]
        if (rows, cols) == (time_len, freq_len):
            spec = spec.T
    return spec, kind, freqs, times


def _crop_freq(spec: np.ndarray, target_f: int) -> np.ndarray:
    freq_bins = spec.shape[0]
    if freq_bins < target_f:
        pad = target_f - freq_bins
        return np.pad(spec, ((0, pad), (0, 0)), mode="edge")
    if freq_bins > target_f:
        start = max(0, (freq_bins - target_f) // 2)
        return spec[start : start + target_f, :]
    return spec


def _crop_time(
    spec: np.ndarray,
    *,
    times: Optional[np.ndarray],
    crop_start_s: float,
    target_t: int,
) -> Tuple[np.ndarray, int]:
    time_bins = spec.shape[1]
    if target_t <= 0:
        raise ValueError("target_t must be positive")
    if times is not None:
        t = np.asarray(times).ravel().astype(np.float64)
        if t.shape[0] == time_bins and t.size:
            start = int(np.searchsorted(t, float(crop_start_s), side="left"))
        else:
            start = 0
    else:
        dt = _infer_time_bin_seconds(times)
        start = int(round(float(crop_start_s) / dt)) if dt else 0
    start = max(0, min(start, max(0, time_bins - 1)))
    if time_bins < target_t:
        spec = np.pad(spec, ((0, 0), (0, target_t - time_bins)), mode="edge")
        time_bins = spec.shape[1]
    end = start + target_t
    if end > time_bins:
        start = max(0, time_bins - target_t)
        end = start + target_t
    return spec[:, start:end], int(start)


class MultiBandMatDataset(Dataset):
    """Load aligned low/mid/high MATs and independent binary labels."""

    def __init__(
        self,
        manifest_csv: str | Path,
        vocabulary: LabelVocabulary | str | Path,
        *,
        split: Optional[str] = None,
        dataset_root: Optional[str | Path] = None,
        bands: Sequence[str] = DEFAULT_BANDS,
        band_crop_shapes: Optional[Mapping[str, Sequence[int]] | str] = None,
        crop_time_seconds: float = 10.0,
        context_seconds: float = 40.0,
        min_db: float = -80.0,
        max_db: float = 0.0,
        center_bias_sigma_frac: float = 0.25,
        positive_crop_mode: str = "edge_mix",
        seed: int = 0,
        return_meta: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to load MultiBandMatDataset")
        self.manifest_csv = Path(manifest_csv)
        self.vocabulary = vocabulary if isinstance(vocabulary, LabelVocabulary) else LabelVocabulary.load(vocabulary)
        self.dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
        self.split = split
        self.bands = tuple(str(band) for band in bands)
        self.band_crop_shapes = parse_band_crop_shapes(band_crop_shapes)
        self.crop_time_seconds = float(crop_time_seconds)
        self.context_seconds = float(context_seconds)
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.center_bias_sigma_frac = float(center_bias_sigma_frac)
        self.positive_crop_mode = str(positive_crop_mode)
        self.rng = np.random.default_rng(seed)
        self.return_meta = bool(return_meta)

        rows = read_csv_rows(self.manifest_csv)
        if split is not None:
            rows = [row for row in rows if clean_text(row.get("split")) == str(split)]
        self.rows: List[Dict[str, Any]] = rows
        self.files: List[Tuple[Dict[str, Path], np.ndarray, Dict[str, Any]]] = []
        for row in self.rows:
            target = self.vocabulary.vectorize(label_ids_from_row(row))
            paths = {
                band: _resolve_path(row.get(f"{band}_mat_path"), self.dataset_root)
                for band in self.bands
            }
            self.files.append((paths, target, dict(row)))

    def __len__(self) -> int:
        return len(self.files)

    def _sample_crop_start_s(self, is_positive: bool) -> float:
        max_start = max(0.0, self.context_seconds - self.crop_time_seconds)
        if is_positive:
            if self.split == "train":
                frac = _sample_positive_crop_fraction(
                    self.rng,
                    center_bias_sigma_frac=self.center_bias_sigma_frac,
                    positive_crop_mode=self.positive_crop_mode,
                )
            else:
                frac = 0.5
            start = (self.context_seconds / 2.0) - (float(frac) * self.crop_time_seconds)
        else:
            if self.split == "train":
                start = float(self.rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
            else:
                start = max_start / 2.0
        return float(np.clip(start, 0.0, max_start))

    def __getitem__(self, index: int):
        paths, target, row = self.files[index]
        is_positive = bool(np.any(target > 0))
        crop_start_s = self._sample_crop_start_s(is_positive)
        tensors: Dict[str, torch.Tensor] = {}
        full_shapes: Dict[str, List[int]] = {}
        crop_starts: Dict[str, int] = {}
        for band in self.bands:
            spec, kind, _, times = _load_spectrogram_raw(paths[band])
            full_shapes[band] = list(spec.shape)
            if kind == "power":
                spec = _power_to_db_norm(spec)
            freq_target, time_target = self.band_crop_shapes.get(band, (spec.shape[0], spec.shape[1]))
            spec = _crop_freq(np.asarray(spec, dtype=np.float32), int(freq_target))
            spec, crop_idx = _crop_time(spec, times=times, crop_start_s=crop_start_s, target_t=int(time_target))
            crop_starts[band] = crop_idx
            spec = _normalize_db_to_unit(spec, self.min_db, self.max_db)
            tensors[band] = torch.from_numpy(spec).unsqueeze(0).float()
        y = torch.from_numpy(target.astype(np.float32))
        if not self.return_meta:
            return tensors, y
        meta = {
            "item_id": row.get("item_id") or Path(next(iter(paths.values()))).stem,
            "mat_path": str(paths.get("low") or next(iter(paths.values()))),
            "low_mat_path": str(paths.get("low", "")),
            "mid_mat_path": str(paths.get("mid", "")),
            "high_mat_path": str(paths.get("high", "")),
            "source_audio": row.get("source_audio") or row.get("filename"),
            "source_dataset": row.get("source_dataset") or "",
            "source_kind": row.get("source_kind") or "",
            "negative_bucket": row.get("negative_bucket") or "",
            "split": row.get("split") or self.split or "",
            "label_ids": label_ids_from_row(row),
            "source_label_ids": row.get("source_label_ids") or "",
            "canonical_label_ids": row.get("canonical_label_ids") or row.get("label_ids") or "",
            "analysis_label_ids": row.get("analysis_label_ids") or "",
            "is_background": row.get("is_background") or "",
            "review_status": row.get("review_status") or "",
            "context_tags": row.get("context_tags") or "",
            "begin_s": row.get("begin_s") or row.get("begin_time_s") or "",
            "end_s": row.get("end_s") or row.get("end_time_s") or "",
            "event_group": row.get("event_group") or "",
            "full_shape": full_shapes,
            "crop_start_s": crop_start_s,
            "crop_start": crop_starts,
            "crop_time_seconds": self.crop_time_seconds,
        }
        return tensors, y, meta
