#!/usr/bin/env python3
"""
Run Model Inference on Sequential Spectrograms

Runs a trained model on processed spectrogram data and saves predictions
to JSON with full versioning metadata for expert review.

Usage:
    python scripts/inference/run_inference.py \
        --mat-dir output/test_windows/spectrograms/2024-01-01/ICLISTENHF1951 \
        --checkpoint checkpoints/best.pt \
        --output-json output/test_windows/predictions.json \
        --dataset-metadata output/test_windows/metadata.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io
import soundfile as sf

from src.models.fin_models import create_model
from src.utils.model_utils import extract_model_info, verify_model_hash, compute_model_hash
from src.utils.unified_prediction_tracker import UnifiedPredictionTracker
from src.dataset.reporting import print_status, print_header
from src.data.sequential_prep import extract_timestamp_from_filename, parse_datetime


def _parse_crop_size(crop_size: Optional[Any]) -> Tuple[Optional[int], Optional[int]]:
    """Parse crop size into (freq_bins, time_bins)."""
    if crop_size is None:
        return (None, None)
    if isinstance(crop_size, int):
        return (int(crop_size), int(crop_size))
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        return (
            int(crop_size[0]) if crop_size[0] is not None else None,
            int(crop_size[1]) if crop_size[1] is not None else None,
        )
    raise ValueError(f"crop_size must be int or [freq,time], got {crop_size}")


def _infer_time_bin_seconds(times: Optional[np.ndarray]) -> Optional[float]:
    if times is None:
        return None
    t = np.asarray(times).ravel()
    if t.size < 2:
        return None
    diffs = np.diff(t.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _infer_time_bin_seconds_from_mat(mat_path: Optional[Path]) -> Optional[float]:
    """Best-effort time-bin inference directly from a MAT file."""
    if mat_path is None or not mat_path.exists():
        return None
    try:
        data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception:
        return None
    tk = _find_key(data, InferenceDataset.TIME_KEYS)
    if tk not in data:
        return None
    return _infer_time_bin_seconds(np.asarray(data[tk]).squeeze())


def _maybe_convert_window_times_from_bins(
    window_time_start: Optional[Any],
    window_time_end: Optional[Any],
    window_start_bin: Optional[Any],
    crop_time_bins: Optional[Any],
    time_bin_seconds: Optional[Any],
) -> Tuple[Optional[float], Optional[float]]:
    """Convert window_time_* from bin units to seconds when values are clearly bin-index based."""
    try:
        wts = float(window_time_start) if window_time_start is not None else None
        wte = float(window_time_end) if window_time_end is not None else None
    except (TypeError, ValueError):
        return window_time_start, window_time_end
    if wts is None or wte is None:
        return window_time_start, window_time_end

    try:
        tbin = float(time_bin_seconds) if time_bin_seconds is not None else None
    except (TypeError, ValueError):
        tbin = None
    if tbin is None or tbin <= 0:
        return wts, wte

    ws = None
    try:
        ws = float(window_start_bin) if window_start_bin is not None else None
    except (TypeError, ValueError):
        ws = None

    ct = None
    try:
        ct = float(crop_time_bins) if crop_time_bins is not None else None
    except (TypeError, ValueError):
        ct = None

    looks_like_bins = False
    if ws is not None and abs(wts - ws) < 1e-6:
        looks_like_bins = True
    if ct is not None and abs((wte - wts) - ct) < 1e-3:
        looks_like_bins = True

    if looks_like_bins:
        return wts * tbin, wte * tbin
    return wts, wte


class InferenceDataset(torch.utils.data.Dataset):
    """Dataset for inference on MAT spectrograms with optional sliding window.
    
    Matches the data preparation from FinWhaleMatDataset in training.
    Supports sliding window mode for exhaustive scanning of spectrograms.
    """
    
    # Same keys as training dataset
    SPECTRO_KEYS = ('spectrogram', 'PdB_norm', 'power_db_norm', 'PdB', 'P_db',
                    'P', 'PSD', 'psd', 'Sxx', 'S', 'spec', 'power_spectrogram')
    POWER_KEYS = ('P', 'Sxx', 'PSD', 'psd', 'power_spectrogram')
    DB_KEYS = ('PdB_norm', 'power_db_norm', 'PdB', 'P_db')
    FREQ_KEYS = ('frequencies', 'F', 'freqs', 'freq', 'f')
    TIME_KEYS = ('times', 'T', 'time', 't')
    
    def __init__(
        self,
        mat_dir: str,
        crop_size: Optional[Any] = None,
        crop_time_seconds: Optional[float] = None,
        crop_freq_range_hz: Optional[Tuple[float, float]] = None,
        min_db: float = -80.0,
        max_db: float = 0.0,
        sliding_window: bool = False,
        window_step: Optional[int] = None,  # None = same as crop_size (no overlap)
        window_step_seconds: Optional[float] = None,  # None = derive from window_step or crop size
        windows_per_file: Optional[int] = None,  # Evenly distribute N windows per file
    ):
        """Initialize the inference dataset.
        
        Args:
            mat_dir: Directory containing MAT files
            crop_size: Crop size. int for square, [freq,time] for non-square.
            crop_time_seconds: Physical time span for final crop. Overrides crop_size time bins.
            crop_freq_range_hz: Physical frequency range [min_hz, max_hz] for final crop.
            min_db: Minimum dB for normalization
            max_db: Maximum dB for normalization
            sliding_window: If True, slide window across time axis
            window_step: Step size for sliding window. None = crop_size (no overlap)
            window_step_seconds: Sliding step in seconds. Overrides window_step when provided.
            windows_per_file: If set, evenly distribute exactly N windows per file
        """
        self.mat_dir = Path(mat_dir)
        self.crop_size = crop_size
        self.freq_crop, self.time_crop = _parse_crop_size(crop_size)
        self.crop_time_seconds = float(crop_time_seconds) if crop_time_seconds is not None else None
        if self.crop_time_seconds is not None and self.crop_time_seconds <= 0:
            raise ValueError("crop_time_seconds must be > 0")
        if crop_freq_range_hz is not None:
            fmin, fmax = float(crop_freq_range_hz[0]), float(crop_freq_range_hz[1])
            if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
                raise ValueError("crop_freq_range_hz must be [min_hz, max_hz] with max > min")
            self.crop_freq_range_hz = (fmin, fmax)
        else:
            self.crop_freq_range_hz = None
        self.min_db = min_db
        self.max_db = max_db
        self.sliding_window = sliding_window
        self.window_step = window_step
        self.window_step_seconds = float(window_step_seconds) if window_step_seconds is not None else None
        if self.window_step_seconds is not None and self.window_step_seconds <= 0:
            raise ValueError("window_step_seconds must be > 0")
        self.windows_per_file = windows_per_file
        self.window_step_by_file: Dict[int, Optional[float]] = {}
        self.crop_dims_by_file: Dict[int, Tuple[int, int]] = {}
        self.freq_slice_by_file: Dict[int, Optional[Tuple[int, int]]] = {}
        self.time_bin_seconds_by_file: Dict[int, Optional[float]] = {}
        self.reference_time_bin_seconds: Optional[float] = None
        self.output_crop_shape: Optional[List[int]] = None
        
        # Find all MAT files
        self.mat_files = sorted(list(self.mat_dir.glob("*.mat")))
        if not self.mat_files:
            raise ValueError(f"No MAT files found in {mat_dir}")

        # Initialize reference time resolution for seconds-based options.
        if self.crop_time_seconds is not None or self.window_step_seconds is not None:
            first_spec, _, _, first_times = self._load_spectrogram_raw(self.mat_files[0])
            _ = first_spec  # shape is not needed here; keeps explicit intent
            self.reference_time_bin_seconds = _infer_time_bin_seconds(first_times)
            if self.reference_time_bin_seconds is None or self.reference_time_bin_seconds <= 0:
                raise ValueError(
                    "crop-time-seconds/window-step-seconds requires MAT time axis with at least 2 increasing entries"
                )
        
        # Build index: list of (file_idx, window_start) tuples
        self.samples = []
        if sliding_window:
            # Pre-scan files to build window indices
            for file_idx, mat_path in enumerate(self.mat_files):
                spec, _, freqs, times = self._load_spectrogram_raw(mat_path)
                F_dim, T_dim = spec.shape
                crop_f, crop_t, freq_slice = self._resolve_crop_dims(F_dim, T_dim, freqs=freqs, times=times)
                self.crop_dims_by_file[file_idx] = (crop_f, crop_t)
                self.freq_slice_by_file[file_idx] = freq_slice
                self.time_bin_seconds_by_file[file_idx] = _infer_time_bin_seconds(times) or self.reference_time_bin_seconds

                max_start = max(T_dim - crop_t, 0)
                if T_dim < crop_t:
                    raise ValueError(
                        f"Spectrogram time bins ({T_dim}) smaller than crop target ({crop_t}) for {mat_path.name}"
                    )

                if max_start == 0:
                    # Single window if spectrogram is exactly crop size
                    self.samples.append((file_idx, 0))
                    self.window_step_by_file[file_idx] = 0.0
                    continue

                if self.windows_per_file is not None:
                    n_windows = int(self.windows_per_file)
                    if n_windows <= 0:
                        raise ValueError("windows_per_file must be >= 1")
                    if n_windows == 1:
                        starts = [0]
                        step = 0.0
                    else:
                        if n_windows > (max_start + 1):
                            raise ValueError(
                                f"windows_per_file ({n_windows}) too large for {mat_path.name} "
                                f"with {T_dim} time bins and crop_size {crop_size}"
                            )
                        step = max_start / (n_windows - 1)
                        starts = np.round(np.linspace(0, max_start, n_windows)).astype(int).tolist()

                    for win_start in starts:
                        self.samples.append((file_idx, int(win_start)))
                    self.window_step_by_file[file_idx] = step
                    continue

                # Calculate minimum overlap windows based on requested step (or crop_size)
                if self.window_step_seconds is not None:
                    dt = self.time_bin_seconds_by_file[file_idx] or self.reference_time_bin_seconds
                    if dt is None or dt <= 0:
                        raise ValueError("window_step_seconds requires valid MAT time axis")
                    step_bins = max(1, int(round(self.window_step_seconds / dt)))
                elif self.window_step is not None:
                    step_bins = int(self.window_step)
                else:
                    step_bins = crop_t
                if step_bins is None or step_bins <= 0:
                    raise ValueError("window_step must be >= 1 when sliding_window is enabled")

                n_windows = int(np.ceil(max_start / step_bins)) + 1
                if n_windows <= 1:
                    starts = [0]
                    step = 0.0
                else:
                    step = max_start / (n_windows - 1)
                    starts = np.round(np.linspace(0, max_start, n_windows)).astype(int).tolist()

                for win_start in starts:
                    self.samples.append((file_idx, int(win_start)))
                self.window_step_by_file[file_idx] = step
        else:
            # One sample per file (center crop)
            self.samples = [(i, None) for i in range(len(self.mat_files))]

        # Determine output shape for metadata/logging.
        first_spec, _, first_freqs, first_times = self._load_spectrogram_raw(self.mat_files[0])
        f0, t0 = first_spec.shape
        crop_f0, crop_t0, _ = self._resolve_crop_dims(f0, t0, freqs=first_freqs, times=first_times)
        self.output_crop_shape = [int(crop_f0), int(crop_t0)]

    def _resolve_crop_dims(
        self,
        F_dim: int,
        T_dim: int,
        freqs: Optional[np.ndarray],
        times: Optional[np.ndarray],
    ) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """Resolve target crop dimensions and optional frequency pre-slice."""
        freq_slice: Optional[Tuple[int, int]] = None
        effective_F = int(F_dim)

        if self.crop_freq_range_hz is not None and freqs is not None:
            freq_arr = np.asarray(freqs).ravel()
            if freq_arr.shape[0] == F_dim:
                fmin, fmax = self.crop_freq_range_hz
                mask = (freq_arr >= fmin) & (freq_arr <= fmax)
                if np.any(mask):
                    idx = np.where(mask)[0]
                    freq_slice = (int(idx[0]), int(idx[-1]) + 1)
                    effective_F = int(freq_slice[1] - freq_slice[0])

        crop_f = int(self.freq_crop) if self.freq_crop is not None else int(effective_F)
        if self.crop_time_seconds is not None:
            dt = _infer_time_bin_seconds(times) or self.reference_time_bin_seconds
            if dt is None or dt <= 0:
                raise ValueError("crop_time_seconds requires valid MAT time axis")
            crop_t = max(1, int(round(self.crop_time_seconds / dt)))
        else:
            crop_t = int(self.time_crop) if self.time_crop is not None else int(crop_f)

        return int(crop_f), int(crop_t), freq_slice
    
    def _find_key(self, data: dict, keys: tuple) -> Optional[str]:
        """Find matching key in data dict (same as training)."""
        for k in keys:
            if k in data:
                return k
        # Case-insensitive fallback
        lowered = {k.lower(): k for k in data.keys()}
        for k in keys:
            if k.lower() in lowered:
                return lowered[k.lower()]
        return None
    
    def _load_spectrogram_raw(
        self, mat_path: Path
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """Load raw spectrogram from MAT file without normalization.

        Returns:
            Tuple of (spec, spec_kind, freqs, times) where spec_kind is 'power' or 'db'.
        """
        data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
        
        k = self._find_key(data, self.POWER_KEYS)
        spec_kind = 'power'
        if k is None:
            k = self._find_key(data, self.DB_KEYS) or self._find_key(data, self.SPECTRO_KEYS)
            spec_kind = 'db'
        if k is None:
            raise KeyError(f"No spectrogram-like key found in {mat_path.name}")
        
        spec = np.asarray(data[k])
        if spec.ndim != 2:
            raise ValueError(f"Unexpected spectrogram ndim {spec.ndim} in {mat_path.name}")
        
        # Check orientation using freq/time vectors if available
        fk = self._find_key(data, self.FREQ_KEYS)
        tk = self._find_key(data, self.TIME_KEYS)
        freqs = np.asarray(data[fk]).squeeze() if fk in data else None
        times = np.asarray(data[tk]).squeeze() if tk in data else None
        if freqs is not None and times is not None:
            f_len = int(np.asarray(freqs).ravel().shape[0])
            t_len = int(np.asarray(times).ravel().shape[0])
            r, c = spec.shape[:2]
            if (r, c) == (t_len, f_len):
                spec = spec.T  # now (F, T)
        
        return spec, spec_kind, freqs, times
    
    def _normalize_db_to_unit(self, x: np.ndarray) -> np.ndarray:
        """Normalize dB to [0, 1] (exactly like training: clip then normalize)."""
        x = x.astype(np.float32)
        x = np.clip(x, self.min_db, self.max_db)
        return (x - self.min_db) / (self.max_db - self.min_db)

    def _power_to_db_norm(self, power: np.ndarray) -> np.ndarray:
        power = np.abs(power.astype(np.float32))
        max_power = float(np.max(power)) if power.size else 0.0
        if max_power > 0:
            normalized = power / max_power
            normalized = np.maximum(normalized, 1e-10)
            return 10.0 * np.log10(normalized)
        return np.full_like(power, -100.0, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict]:
        file_idx, window_start = self.samples[idx]
        mat_path = self.mat_files[file_idx]
        file_id = mat_path.stem
        
        # Load spectrogram
        spec, spec_kind, freqs, times = self._load_spectrogram_raw(mat_path)
        F_dim, T_dim = spec.shape
        crop_f, crop_t, freq_slice = self._resolve_crop_dims(F_dim, T_dim, freqs=freqs, times=times)
        time_bin_seconds = _infer_time_bin_seconds(times) or self.reference_time_bin_seconds

        meta = {
            'original_shape': [F_dim, T_dim],
            'crop_size': [crop_f, crop_t],
            'crop_freq_bins': int(crop_f),
            'crop_time_bins': int(crop_t),
            'crop_time_seconds': float(self.crop_time_seconds) if self.crop_time_seconds is not None else None,
            'crop_freq_range_hz': list(self.crop_freq_range_hz) if self.crop_freq_range_hz is not None else None,
            'time_bin_seconds': float(time_bin_seconds) if time_bin_seconds is not None else None,
            'sliding_window': self.sliding_window,
            'window_start': window_start,
            'window_step': self.window_step_by_file.get(file_idx),
            'window_step_requested': self.window_step,
            'window_step_seconds_requested': self.window_step_seconds,
            'windows_per_file': self.windows_per_file,
            'freq_slice_start': int(freq_slice[0]) if freq_slice is not None else None,
            'freq_slice_end': int(freq_slice[1]) if freq_slice is not None else None,
        }

        # Optional physical frequency pre-slice.
        if freq_slice is not None:
            f0, f1 = freq_slice
            spec = spec[f0:f1, :]
            F_dim, T_dim = spec.shape

        # Frequency axis: pad or center-crop (match training behavior).
        if F_dim < crop_f:
            pad = crop_f - F_dim
            spec = np.pad(spec, ((0, pad), (0, 0)), mode='edge')
            F_dim = crop_f
        elif F_dim > crop_f:
            start_f = max(0, (F_dim - crop_f) // 2)
            spec = spec[start_f:start_f + crop_f, :]
            F_dim = crop_f

        # Time axis: sliding window or center crop.
        if self.sliding_window and window_start is not None:
            start_t = int(window_start)
            end_t = start_t + int(crop_t)
            if end_t > T_dim:
                start_t = max(0, T_dim - int(crop_t))
                end_t = start_t + int(crop_t)
            if T_dim < crop_t:
                # Sliding mode generally pre-validates this, but keep it robust.
                pad = crop_t - T_dim
                spec = np.pad(spec, ((0, 0), (0, pad)), mode='edge')
                start_t = 0
                end_t = crop_t
            else:
                spec = spec[:, start_t:end_t]
            meta['crop_type'] = 'sliding_window'
            meta['window_time_start'] = int(start_t)
            meta['window_time_end'] = int(end_t)
        else:
            if T_dim < crop_t:
                pad = crop_t - T_dim
                spec = np.pad(spec, ((0, 0), (0, pad)), mode='edge')
                start_t = 0
                end_t = crop_t
            elif T_dim > crop_t:
                start_t = max(0, (T_dim - crop_t) // 2)
                end_t = start_t + crop_t
                spec = spec[:, start_t:end_t]
            else:
                start_t = 0
                end_t = crop_t
            meta['crop_type'] = 'center_crop'
            meta['window_time_start'] = int(start_t)
            meta['window_time_end'] = int(end_t)

        meta['output_shape'] = list(spec.shape)
        meta['crop_applied'] = True

        # Normalize after cropping for consistent context
        if spec_kind == 'power':
            spec = self._power_to_db_norm(spec)
        spec = self._normalize_db_to_unit(spec)
        
        # Create unique file_id for sliding windows
        if self.sliding_window and window_start is not None:
            file_id = f"{file_id}_win{window_start}"
        
        # Convert to tensor [1, F, T]
        tensor = torch.from_numpy(spec).unsqueeze(0).float()
        
        return tensor, file_id, meta


def _resolve_path(path_value: Optional[str], base_dir: Optional[Path]) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p
    if base_dir is None:
        return Path(os.path.abspath(str(p)))
    return Path(os.path.abspath(str(base_dir / p)))


def _path_to_json_ref(path_value: Optional[Path], json_base_dir: Path) -> Optional[str]:
    """Serialize a filesystem path for JSON `paths.*` fields.

    Prefers path relative to the predictions JSON directory (including ../../
    style paths for siblings) so the verification app can relocate the whole
    folder tree.
    """
    if path_value is None:
        return None
    try:
        return os.path.relpath(str(path_value), str(json_base_dir))
    except Exception:
        return str(path_value)


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _compact_utc_timestamp(value: Optional[str]) -> Optional[str]:
    dt = _parse_iso_datetime(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"


def _safe_id_token(value: Optional[str], *, fallback: str, max_len: int = 80) -> str:
    raw = str(value) if value is not None else fallback
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


def _derive_device_token(
    *,
    device_code: Optional[str],
    source_audio: Optional[str],
    base_id: Optional[str],
) -> str:
    if device_code:
        return _safe_id_token(device_code, fallback="unknown-device", max_len=32)
    if source_audio:
        src_name = Path(source_audio).name
        if "_" in src_name:
            return _safe_id_token(src_name.split("_", 1)[0], fallback="unknown-device", max_len=32)
        return _safe_id_token(src_name, fallback="unknown-device", max_len=32)
    if base_id:
        if "_" in base_id:
            return _safe_id_token(base_id.split("_", 1)[0], fallback="unknown-device", max_len=32)
        return _safe_id_token(base_id, fallback="unknown-device", max_len=32)
    return "unknown-device"


def _build_descriptive_window_id(
    *,
    file_id: str,
    base_id: str,
    source_audio: Optional[str],
    device_code: Optional[str],
    audio_start_time: Optional[str],
    audio_end_time: Optional[str],
    window_start: Optional[Any],
    window_time_start: Optional[Any],
    window_time_end: Optional[Any],
) -> str:
    device = _derive_device_token(
        device_code=device_code,
        source_audio=source_audio,
        base_id=base_id,
    )
    src_token = _safe_id_token(Path(source_audio).stem if source_audio else base_id, fallback=base_id, max_len=48)
    start_token = _compact_utc_timestamp(audio_start_time)
    end_token = _compact_utc_timestamp(audio_end_time)
    if start_token and end_token:
        time_part = f"{start_token}-{end_token}"
    else:
        if window_time_start is None:
            wts = "na"
        else:
            wts = f"{float(window_time_start):.3f}".replace(".", "p")
        if window_time_end is None:
            wte = "na"
        else:
            wte = f"{float(window_time_end):.3f}".replace(".", "p")
        time_part = f"s{wts}-e{wte}"

    if window_start is None:
        win_tag = "wna"
    else:
        try:
            win_tag = f"w{int(window_start):06d}"
        except (TypeError, ValueError):
            win_tag = f"w{_safe_id_token(str(window_start), fallback='na', max_len=12)}"

    digest = hashlib.sha1(
        "|".join(
            [
                str(file_id),
                str(base_id),
                str(source_audio or ""),
                str(audio_start_time or ""),
                str(audio_end_time or ""),
                str(window_start),
                str(window_time_start),
                str(window_time_end),
            ]
        ).encode("utf-8")
    ).hexdigest()[:8]
    return f"fw-{device}-{time_part}-{src_token}-{win_tag}-h{digest}"


def _first_existing_path(candidates: List[Optional[Path]]) -> Optional[Path]:
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def _load_mat_with_axes(mat_path: Path) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load spectrogram and axes from MAT file.

    Returns (spec, spec_kind, freqs, times)
    """
    data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    # Use same key logic as InferenceDataset
    def _find_key(d: dict, keys: tuple) -> Optional[str]:
        for k in keys:
            if k in d:
                return k
        lowered = {k.lower(): k for k in d.keys()}
        for k in keys:
            if k.lower() in lowered:
                return lowered[k.lower()]
        return None

    k = _find_key(data, InferenceDataset.POWER_KEYS)
    spec_kind = 'power'
    if k is None:
        k = _find_key(data, InferenceDataset.DB_KEYS) or _find_key(data, InferenceDataset.SPECTRO_KEYS)
        spec_kind = 'db'
    if k is None:
        raise KeyError(f"No spectrogram-like key found in {mat_path.name}")

    spec = np.asarray(data[k])
    if spec.ndim != 2:
        raise ValueError(f"Unexpected spectrogram ndim {spec.ndim} in {mat_path.name}")

    freqs = None
    times = None
    fk = _find_key(data, InferenceDataset.FREQ_KEYS)
    tk = _find_key(data, InferenceDataset.TIME_KEYS)
    if fk in data:
        freqs = np.asarray(data[fk]).squeeze()
    if tk in data:
        times = np.asarray(data[tk]).squeeze()

    # Orient using freq/time vectors if available
    if freqs is not None and times is not None:
        f_len = int(np.asarray(freqs).ravel().shape[0])
        t_len = int(np.asarray(times).ravel().shape[0])
        r, c = spec.shape[:2]
        if (r, c) == (t_len, f_len):
            spec = spec.T  # now (F, T)

    return spec, spec_kind, freqs, times


def _power_to_db_norm(power: np.ndarray) -> np.ndarray:
    power = np.abs(power.astype(np.float32))
    max_power = float(np.max(power)) if power.size else 0.0
    if max_power > 0:
        normalized = power / max_power
        normalized = np.maximum(normalized, 1e-10)
        return 10.0 * np.log10(normalized)
    return np.full_like(power, -100.0, dtype=np.float32)


def _compute_window_time_range(
    times: Optional[np.ndarray],
    start_idx: int,
    window_bins: int,
    win_dur: Optional[float],
    overlap: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute window start/end times in seconds using time-bin centers."""
    if times is None or len(times) == 0:
        return None, None
    start_idx = max(0, min(int(start_idx), len(times) - 1))
    center_start = float(times[start_idx])
    if len(times) > 1:
        hop_sec = float(times[1] - times[0])
    else:
        hop_sec = (win_dur * (1.0 - overlap)) if (win_dur is not None and overlap is not None) else 0.0
    if win_dur is None:
        win_dur = 0.0
    window_time_start = max(0.0, center_start - (win_dur / 2.0))
    window_time_end = window_time_start + max(0, window_bins - 1) * hop_sec + win_dur
    return window_time_start, window_time_end

def extract_crop_size_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """Extract crop_size from checkpoint's training args.
    
    Checks both the checkpoint dict and args.pkl in the same directory.
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Try args.pkl first (more complete)
    args_pkl = checkpoint_path.parent / 'args.pkl'
    if args_pkl.exists():
        try:
            import pickle
            with open(args_pkl, 'rb') as f:
                args = pickle.load(f)
            if hasattr(args, 'crop_size') and args.crop_size is not None:
                return int(args.crop_size) if isinstance(args.crop_size, (int, float)) else args.crop_size
        except Exception:
            pass
    
    # Try checkpoint dict
    try:
        import torch
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        if 'training_args' in ckpt:
            args = ckpt['training_args']
            if isinstance(args, dict) and 'crop_size' in args:
                return args['crop_size']
        if 'args' in ckpt:
            args = ckpt['args']
            if isinstance(args, dict) and 'crop_size' in args:
                return args['crop_size']
    except Exception:
        pass
    
    return None


def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run inference on all samples in dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with inference samples
        device: Device to run on
        
    Returns:
        List of {file_id, confidence, meta} dicts
    """
    model.eval()
    results = []
    
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                x, file_ids, metas = batch
            else:
                x, file_ids = batch
                metas = [{}] * len(file_ids)
            
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            # Get positive class probability
            pos_probs = probs[:, 1].cpu().numpy()
            
            for i, (file_id, prob) in enumerate(zip(file_ids, pos_probs)):
                meta = {}
                if isinstance(metas, dict):
                    # Batch collation turned it into a dict of tensors
                    for k, v in metas.items():
                        try:
                            meta[k] = v[i].item() if hasattr(v[i], 'item') else v[i]
                        except:
                            meta[k] = None
                elif isinstance(metas, (list, tuple)) and i < len(metas):
                    meta = metas[i] if isinstance(metas[i], dict) else {}
                
                results.append({
                    'file_id': file_id,
                    'confidence': float(prob),
                    'meta': meta
                })
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                print(f"  Batch {batch_idx + 1}/{total_batches}", end='\r')
    
    print()  # newline after progress
    return results


def _normalize_spec_config_from_test_metadata(proc_params: Dict[str, Any]) -> Dict[str, Any]:
    # Allow caller to inject provenance; default to computed if missing
    source_override = proc_params.get("spectrogram_source")
    freq_lims = proc_params.get('freq_lims_hz')
    freq_limits = None
    if isinstance(freq_lims, (list, tuple)) and len(freq_lims) >= 2:
        freq_limits = {"min": freq_lims[0], "max": freq_lims[1]}
    elif isinstance(freq_lims, dict):
        freq_limits = {"min": freq_lims.get("min"), "max": freq_lims.get("max")}

    clim = proc_params.get('clim_db')
    color_limits = None
    if isinstance(clim, (list, tuple)) and len(clim) >= 2:
        color_limits = {"min": clim[0], "max": clim[1]}
    elif isinstance(clim, dict):
        color_limits = {"min": clim.get("min"), "max": clim.get("max")}

    spec_config = {
        "window_duration": proc_params.get("win_dur_s"),
        "overlap": proc_params.get("overlap"),
        "frequency_limits": freq_limits,
        "color_limits": color_limits,
        "crop_size": proc_params.get("crop_size"),
        "pipeline": "test_windows",
    }
    if source_override:
        if isinstance(source_override, dict):
            spec_config["source"] = source_override
        else:
            spec_config["source"] = {"type": str(source_override)}
    else:
        spec_config["source"] = {
            "type": "computed",
            "generator": "onc_hydrophone_data.SpectrogramGenerator",
        }
    # Remove empty keys
    return {k: v for k, v in spec_config.items() if v is not None}


def _infer_spectrogram_source(dataset_meta: Dict[str, Any], default_type: str) -> Dict[str, Any]:
    """Infer spectrogram source/provenance metadata when not explicitly provided."""
    source = {}
    if "spectrogram_source" in dataset_meta:
        value = dataset_meta.get("spectrogram_source")
        if isinstance(value, dict):
            return value
        return {"type": str(value)}
    if "spectrogram_download" in dataset_meta and isinstance(dataset_meta.get("spectrogram_download"), dict):
        return dataset_meta["spectrogram_download"]

    plot_res = dataset_meta.get("plot_res", dataset_meta.get("plotRes"))
    if plot_res is not None:
        source = {"type": "onc_download", "provider": "ONC", "plot_res": plot_res}
        return source

    return {"type": default_type}


def load_inference_metadata(path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    """Load metadata and normalize to unified tracker fields.

    Returns (data_source, spectrogram_config, file_info_map, metadata_type)
    """
    data_source: Dict[str, Any] = {}
    spec_config: Dict[str, Any] = {}
    file_info_map: Dict[str, Any] = {}
    metadata_type = "unknown"

    if not path or not Path(path).exists():
        return data_source, spec_config, file_info_map, metadata_type

    with open(path, 'r') as f:
        dataset_meta = json.load(f)

    # Legacy sequential pipeline metadata.json
    if "data_source" in dataset_meta and "files" in dataset_meta:
        metadata_type = "legacy_segments"
        data_source = dataset_meta.get("data_source", {})
        spec_config = dataset_meta.get("spectrogram_config", {})
        if "source" not in spec_config:
            spec_config["source"] = _infer_spectrogram_source(dataset_meta, "computed")
        for file_info in dataset_meta.get("files", []):
            file_info_map[file_info.get("file_id")] = file_info
        return data_source, spec_config, file_info_map, metadata_type

    # Test windows metadata.json (prepare_test_windows.py)
    if "chunks" in dataset_meta and "processing_parameters" in dataset_meta:
        metadata_type = "test_windows"
        data_source = {
            "device_code": dataset_meta.get("device_code", "unknown"),
            "date_from": dataset_meta.get("start_date", ""),
            "date_to": dataset_meta.get("end_date", ""),
        }
        proc_params = dataset_meta.get("processing_parameters", {})
        if "spectrogram_source" not in proc_params:
            proc_params["spectrogram_source"] = _infer_spectrogram_source(dataset_meta, "computed")
        spec_config = _normalize_spec_config_from_test_metadata(proc_params)
        for chunk in dataset_meta.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            file_info_map[chunk_id] = {
                "mat_path": chunk.get("mat_path"),
                "audio_path": chunk.get("audio_path"),
                "audio_timestamp": chunk.get("timestamp"),
                "chunk_shape": chunk.get("chunk_shape"),
                "original_shape": chunk.get("original_shape"),
                "window_index": chunk.get("window_index"),
                "window_start": chunk.get("window_start"),
                "window_time_start": chunk.get("window_time_start"),
                "window_time_end": chunk.get("window_time_end"),
                "source_audio": chunk.get("source_audio"),
                "date": chunk.get("date"),
            }
        return data_source, spec_config, file_info_map, metadata_type

    return data_source, spec_config, file_info_map, metadata_type


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on sequential spectrograms"
    )
    parser.add_argument('--mat-dir', type=str, required=True,
                        help='Directory with MAT spectrograms')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-json', type=str, required=True,
                        help='Output predictions JSON path')
    parser.add_argument('--dataset-metadata', type=str, default=None,
                        help='Path to metadata JSON (auto-detects legacy vs test windows)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--crop-size', type=str, default=None,
                        help='Crop size: int for square or "freq,time" for non-square. '
                             'Auto-detected from checkpoint if not specified.')
    parser.add_argument('--crop-time-seconds', type=float, default=None,
                        help='Physical time span (seconds) for final crop. Overrides crop-size time bins.')
    parser.add_argument('--crop-freq-range-hz', type=float, nargs=2, default=None, metavar=('MIN_HZ', 'MAX_HZ'),
                        help='Physical frequency range for final crop. Default: full MAT frequency axis.')
    parser.add_argument('--sliding-window', action='store_true',
                        help='Use sliding window to scan entire spectrogram')
    parser.add_argument('--window-step', type=int, default=None,
                        help='Step size for sliding window (default: crop_size = no overlap). '
                             'Windows are evenly distributed to avoid padding.')
    parser.add_argument('--window-step-seconds', type=float, default=None,
                        help='Sliding step in seconds. Overrides --window-step when provided.')
    parser.add_argument('--windows-per-file', type=int, default=None,
                        help='Evenly distribute exactly N windows per 5-min spectrogram '
                             '(minimum overlap). Overrides window-step.')
    parser.add_argument('--min-db', type=float, default=-80.0,
                        help='Min dB for normalization')
    parser.add_argument('--max-db', type=float, default=0.0,
                        help='Max dB for normalization')
    parser.add_argument('--verify-hash', action='store_true',
                        help='Verify model hash matches checkpoint')
    # Export options for verification app
    parser.add_argument('--export-crops', action='store_true',
                        help='Export cropped MATs/audio for windows above threshold')
    parser.add_argument('--export-threshold', type=float, default=0.7,
                        help='Score threshold for exporting crops')
    parser.add_argument('--export-dir', type=str, default=None,
                        help='Base directory for exported crops (default: output-json directory)')
    parser.add_argument('--export-all', action='store_true',
                        help='Include predictions below threshold even when exporting crops')
    parser.add_argument('--raw-audio-dir', type=str, default=None,
                        help='Directory containing raw 5-min audio files (for cropped audio export)')
    parser.add_argument('--no-export-audio', dest='export_audio', action='store_false',
                        help='Disable exporting cropped audio (MATs still exported)')
    parser.set_defaults(export_audio=True)
    parser.add_argument('--postprocess', action='store_true',
                        help='Run temporal clustering/hysteresis postprocessing on predictions JSON')
    parser.add_argument('--postprocess-output-json', type=str, default=None,
                        help='Output JSON path for postprocessed predictions '
                             '(default: <output-json stem>_postprocessed.json)')
    parser.add_argument('--postprocess-class-hierarchy', type=str, default=None,
                        help='Class hierarchy to postprocess (default: first model output class)')
    parser.add_argument('--postprocess-low-threshold', type=float, default=0.70,
                        help='Low threshold for postprocess candidate windows')
    parser.add_argument('--postprocess-high-threshold', type=float, default=0.82,
                        help='High threshold required inside each kept event')
    parser.add_argument('--postprocess-min-members', type=int, default=2,
                        help='Minimum number of windows per kept event')
    parser.add_argument('--postprocess-min-duration-sec', type=float, default=0.0,
                        help='Minimum event duration in seconds')
    parser.add_argument('--postprocess-max-gap-seconds', type=float, default=None,
                        help='Maximum gap (sec) between adjacent windows in an event '
                             '(default: auto-inferred by postprocessor)')
    parser.add_argument('--postprocess-events-csv', type=str, default=None,
                        help='Optional CSV path for postprocessed event summary')
    parser.add_argument('--postprocess-summary-md', type=str, default=None,
                        help='Optional markdown summary path for postprocessing')
    parser.add_argument('--postprocess-debug-json', type=str, default=None,
                        help='Optional debug JSON path for postprocessing diagnostics')
    parser.add_argument('--postprocess-merge-event-media', action='store_true',
                        help='Merge clustered windows into event-level spectrogram/audio clips')
    parser.add_argument('--postprocess-merge-min-score', type=float, default=None,
                        help='Optional score floor for members included in event-media merge')
    parser.add_argument('--postprocess-event-media-dir', type=str, default=None,
                        help='Directory for merged event media (default: <postprocess output stem>_events_media)')
    parser.add_argument('--postprocess-replace-items-with-events', action='store_true',
                        help='Replace window-level items with one event-level item per kept cluster')
    parser.add_argument('--postprocess-merge-across-source-audio', action='store_true',
                        help='Allow postprocess event clustering across adjacent source audio files')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print_header("MODEL INFERENCE ON SEQUENTIAL DATA")
    print(f"MAT directory: {args.mat_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Load dataset metadata early (to infer crop_size if needed)
    data_source, spec_config, file_info_map, metadata_type = load_inference_metadata(args.dataset_metadata)
    if args.dataset_metadata:
        print(f"Metadata type: {metadata_type}")

    # Parse crop size - auto-detect from checkpoint / metadata / first MAT
    crop_size = None
    if args.crop_size:
        if ',' in args.crop_size:
            parts = args.crop_size.split(',')
            crop_size = [int(p.strip()) for p in parts]
        else:
            crop_size = int(args.crop_size)
    crop_freq_range_hz = tuple(args.crop_freq_range_hz) if args.crop_freq_range_hz is not None else None
    if (args.crop_time_seconds is not None or crop_freq_range_hz is not None) and args.crop_size is not None:
        raise SystemExit(
            "Use either --crop-size or physical crop args (--crop-time-seconds/--crop-freq-range-hz), not both."
        )
    if args.crop_size is None and args.crop_time_seconds is None and crop_freq_range_hz is None:
        # Try to auto-detect from checkpoint
        crop_size = extract_crop_size_from_checkpoint(args.checkpoint)
        if crop_size:
            print(f"Auto-detected crop_size from checkpoint: {crop_size}")
        else:
            # Try metadata
            meta_crop = spec_config.get("crop_size") if spec_config else None
            if meta_crop:
                if isinstance(meta_crop, (list, tuple)) and len(meta_crop) == 2:
                    crop_size = [int(meta_crop[0]), int(meta_crop[1])]
                else:
                    crop_size = int(meta_crop)
                print(f"Auto-detected crop_size from metadata: {crop_size}")
            else:
                # Fallback: infer from first MAT file's freq bins
                try:
                    first_mat = next(iter(sorted(Path(args.mat_dir).glob("*.mat"))), None)
                    if first_mat:
                        spec, _, freqs, _ = _load_mat_with_axes(first_mat)
                        if freqs is not None and len(np.atleast_1d(freqs)) > 0:
                            crop_size = int(len(np.atleast_1d(freqs)))
                        else:
                            crop_size = int(spec.shape[0])
                        print(f"Inferred crop_size from MAT ({first_mat.name}): {crop_size}")
                except Exception:
                    pass
            if crop_size is None:
                print("Warning: crop_size not specified and could not be auto-detected")

    # Normalize auto-detected crop_size representations (e.g., "96,192" from args.pkl).
    if isinstance(crop_size, str):
        text = crop_size.strip()
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) == 2:
                crop_size = [int(parts[0]), int(parts[1])]
            else:
                raise SystemExit(f"Invalid auto-detected crop_size string: {crop_size}")
        elif text:
            crop_size = int(text)

    if args.sliding_window and crop_size is None and args.crop_time_seconds is None:
        raise SystemExit(
            "sliding_window requires either crop-size bins or crop-time-seconds. "
            "Provide --crop-size / --crop-time-seconds or ensure crop_size exists in args.pkl/metadata."
        )
    if args.window_step is not None and args.windows_per_file is not None:
        raise SystemExit("Use either --window-step or --windows-per-file (not both).")
    if args.window_step_seconds is not None and args.windows_per_file is not None:
        raise SystemExit("Use either --window-step-seconds or --windows-per-file (not both).")
    if args.window_step is not None and args.window_step_seconds is not None:
        raise SystemExit("Use either --window-step (bins) or --window-step-seconds, not both.")

    # Record effective crop intent in spectrogram config metadata.
    if spec_config is None:
        spec_config = {}
    if crop_size is not None:
        spec_config["crop_size"] = crop_size
    if args.crop_time_seconds is not None:
        spec_config["crop_time_seconds"] = float(args.crop_time_seconds)
    if crop_freq_range_hz is not None:
        spec_config["crop_freq_range_hz"] = {"min": float(crop_freq_range_hz[0]), "max": float(crop_freq_range_hz[1])}
    if args.window_step_seconds is not None:
        spec_config["window_step_seconds"] = float(args.window_step_seconds)

    # Load checkpoint
    print_status("Loading checkpoint...", "PROGRESS")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract model info
    model_info = extract_model_info(checkpoint)
    model_info['checkpoint_path'] = str(Path(args.checkpoint).resolve())
    
    print(f"  Model ID: {model_info['model_id']}")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Trained at: {model_info['trained_at']}")
    
    # Verify hash if requested
    if args.verify_hash:
        if verify_model_hash(checkpoint):
            print_status("Model hash verified ✓", "SUCCESS")
        else:
            print_status("WARNING: Model hash mismatch!", "WARNING")
    
    # Create model
    architecture = model_info['architecture']
    model = create_model(architecture, num_classes=2, in_ch=1).to(device)
    
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    print_status(f"Model loaded: {architecture}", "SUCCESS")
    
    # Create dataset
    print_status("Loading dataset...", "PROGRESS")
    dataset = InferenceDataset(
        mat_dir=args.mat_dir,
        crop_size=crop_size,
        crop_time_seconds=args.crop_time_seconds,
        crop_freq_range_hz=crop_freq_range_hz,
        min_db=args.min_db,
        max_db=args.max_db,
        sliding_window=args.sliding_window,
        window_step=args.window_step,
        window_step_seconds=args.window_step_seconds,
        windows_per_file=args.windows_per_file,
    )
    
    n_files = len(dataset.mat_files)
    n_samples = len(dataset)
    mode = "sliding window" if args.sliding_window else "center crop"
    print(f"  Found {n_files} MAT files -> {n_samples} samples ({mode})")
    if dataset.output_crop_shape is not None:
        print(f"  Effective crop shape [freq, time]: {dataset.output_crop_shape}")
    
    # Create dataloader with custom collate for variable metadata
    def collate_fn(batch):
        tensors = torch.stack([b[0] for b in batch])
        file_ids = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        return tensors, file_ids, metas
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Run inference
    print_status("Running inference...", "PROGRESS")
    results = run_inference(model, dataloader, device)
    print_status(f"Inference complete: {len(results)} predictions", "SUCCESS")
    
    json_base_dir = Path(args.output_json).resolve().parent
    metadata_base = Path(args.dataset_metadata).resolve().parent if args.dataset_metadata else None

    # Optional: export cropped MATs/audio for verification
    export_info: Dict[str, Dict[str, Any]] = {}
    if args.export_crops:
        export_dir = Path(args.export_dir).resolve() if args.export_dir else json_base_dir
        spec_out_dir = export_dir / "spectrograms"
        audio_out_dir = export_dir / "audio"
        spec_out_dir.mkdir(parents=True, exist_ok=True)
        if args.export_audio:
            audio_out_dir.mkdir(parents=True, exist_ok=True)

        # Map MAT stem -> path for quick lookup
        mat_path_map = {p.stem: p for p in dataset.mat_files}
        win_dur = spec_config.get("window_duration") or spec_config.get("window_duration_sec") or spec_config.get("win_dur") or spec_config.get("win_dur_s")
        overlap = spec_config.get("overlap") or spec_config.get("overlap_ratio")

        export_cutoff = float(args.export_threshold)
        if args.postprocess:
            # Ensure postprocessed/kept windows can reference existing exported media.
            export_cutoff = min(export_cutoff, float(args.postprocess_low_threshold))
        export_results = [r for r in results if r['confidence'] >= export_cutoff]
        missing_audio_sources: Dict[str, int] = {}
        print_status(
            f"Exporting crops for {len(export_results)} windows >= {export_cutoff:.2f}",
            "PROGRESS",
        )

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in export_results:
            base_id = r['file_id'].rsplit('_win', 1)[0] if '_win' in r['file_id'] else r['file_id']
            grouped[base_id].append(r)

        grouped_items = list(grouped.items())
        total_groups = len(grouped_items)
        total_export_targets = len(export_results)
        print_status(
            f"Export pass over {total_groups} source clips ({total_export_targets} candidate windows)",
            "PROGRESS",
        )

        for group_idx, (base_id, group) in enumerate(grouped_items, start=1):
            file_info = file_info_map.get(base_id, {})

            # Resolve MAT path for this base clip
            mat_path = mat_path_map.get(base_id)
            if mat_path is None and file_info.get("mat_path"):
                mat_path = _resolve_path(file_info.get("mat_path"), metadata_base)
            if mat_path is None:
                mat_path = Path(args.mat_dir) / f"{base_id}.mat"
            if not mat_path.exists():
                print_status(f"Missing MAT for {base_id}: {mat_path}", "WARNING")
                continue

            # Load MAT once for all windows in this clip
            try:
                spec, spec_kind, freqs, times = _load_mat_with_axes(mat_path)
            except Exception as e:
                print_status(f"Failed to load MAT {mat_path}: {e}", "WARNING")
                continue

            F_dim, T_dim = spec.shape
            # Build fallback axes if missing
            if freqs is None or len(np.atleast_1d(freqs)) == 0:
                freqs = np.arange(F_dim)
            if times is None or len(np.atleast_1d(times)) == 0:
                hop_sec = (win_dur * (1.0 - overlap)) if (win_dur is not None and overlap is not None) else 0.0
                times = np.arange(T_dim, dtype=np.float32) * hop_sec
            group_time_bin_seconds = _infer_time_bin_seconds(np.asarray(times) if times is not None else None)

            # Resolve raw audio file with robust fallbacks.
            source_audio = file_info.get("source_audio") or f"{base_id}.wav"
            raw_candidates: List[Optional[Path]] = []
            if args.raw_audio_dir:
                base_raw_dir = Path(args.raw_audio_dir)
                raw_candidates.append(base_raw_dir / source_audio)
                stem = Path(source_audio).stem
                for ext in (".wav", ".flac", ".mp3"):
                    raw_candidates.append(base_raw_dir / f"{stem}{ext}")
            if file_info.get("raw_audio_path"):
                raw_candidates.append(_resolve_path(file_info.get("raw_audio_path"), metadata_base))
            if metadata_base and source_audio:
                raw_candidates.append(metadata_base / "raw_audio" / source_audio)
                stem = Path(source_audio).stem
                for ext in (".wav", ".flac", ".mp3"):
                    raw_candidates.append(metadata_base / "raw_audio" / f"{stem}{ext}")

            raw_audio_path = _first_existing_path(raw_candidates)
            if args.export_audio and raw_audio_path is None:
                missing_audio_sources[source_audio] = missing_audio_sources.get(source_audio, 0) + 1

            # Parse clip start timestamp
            clip_ts = None
            if file_info.get("audio_timestamp"):
                try:
                    clip_ts = parse_datetime(file_info.get("audio_timestamp"))
                except Exception:
                    clip_ts = None
            if clip_ts is None:
                # Try source audio filename or base_id
                source_name = file_info.get("source_audio") or f"{base_id}.wav"
                clip_ts = extract_timestamp_from_filename(source_name)

            # Export each window for this clip
            for r in group:
                file_id = r['file_id']
                meta = r.get('meta', {})
                start_idx = meta.get('window_start')
                if start_idx is None:
                    continue

                start_idx = int(start_idx)
                crop_f = int(meta.get('crop_freq_bins') or (meta.get('output_shape', [F_dim, T_dim])[0]))
                crop_t = int(meta.get('crop_time_bins') or (meta.get('output_shape', [F_dim, T_dim])[1]))
                spec_work = spec
                freqs_work = np.asarray(freqs).ravel() if freqs is not None else None
                times_work = np.asarray(times).ravel() if times is not None else None
                F_work, T_work = spec_work.shape
                parent_freq_offset = 0

                # Optional physical frequency pre-slice to match inference dataset.
                freq_range_meta = meta.get('crop_freq_range_hz')
                if freq_range_meta is not None and freqs_work is not None:
                    if isinstance(freq_range_meta, dict):
                        fmin = freq_range_meta.get('min')
                        fmax = freq_range_meta.get('max')
                    elif isinstance(freq_range_meta, (list, tuple)) and len(freq_range_meta) >= 2:
                        fmin, fmax = freq_range_meta[0], freq_range_meta[1]
                    else:
                        fmin, fmax = None, None
                    if fmin is not None and fmax is not None:
                        mask = (freqs_work >= float(fmin)) & (freqs_work <= float(fmax))
                        if np.any(mask):
                            idx = np.where(mask)[0]
                            spec_work = spec_work[idx[0]:idx[-1] + 1, :]
                            freqs_work = freqs_work[idx[0]:idx[-1] + 1]
                            parent_freq_offset = int(idx[0])
                            F_work, T_work = spec_work.shape

                # Frequency crop (center)
                f_start = 0
                if F_work > crop_f:
                    f_start = (F_work - crop_f) // 2
                f_end = f_start + crop_f
                parent_freq_start = parent_freq_offset + int(f_start)
                if F_work < crop_f:
                    parent_freq_end = parent_freq_offset + int(F_work)
                else:
                    parent_freq_end = parent_freq_offset + int(min(f_end, F_work))
                if F_work < crop_f:
                    spec_f = np.pad(spec_work, ((0, crop_f - F_work), (0, 0)), mode='edge')
                    if freqs_work is not None:
                        if len(freqs_work) > 0:
                            pad_vals = np.full((crop_f - F_work,), freqs_work[-1], dtype=freqs_work.dtype)
                            freqs_f = np.concatenate([freqs_work, pad_vals], axis=0)
                        else:
                            freqs_f = None
                    else:
                        freqs_f = None
                else:
                    spec_f = spec_work[f_start:f_end, :]
                    freqs_f = freqs_work[f_start:f_end] if freqs_work is not None else None

                # Time crop (window)
                max_start = T_work - crop_t
                if max_start >= 0 and start_idx > max_start:
                    print_status(
                        f"Skipping {file_id}: window_start {start_idx} exceeds max {max_start}",
                        "WARNING",
                    )
                    continue
                t_start = max(0, int(start_idx)) if max_start >= 0 else 0
                t_end = t_start + crop_t
                parent_time_start = int(t_start)
                parent_time_end = int(min(t_end, T_work))
                if T_work < crop_t:
                    spec_crop = np.pad(spec_f, ((0, 0), (0, crop_t - T_work)), mode='edge')
                    if times_work is not None:
                        if len(times_work) >= 2:
                            dt = float(np.median(np.diff(times_work.astype(np.float64))))
                            pad_vals = times_work[-1] + dt * np.arange(1, (crop_t - T_work) + 1, dtype=np.float64)
                            times_crop = np.concatenate([times_work, pad_vals], axis=0)
                        elif len(times_work) == 1:
                            times_crop = np.concatenate([times_work, np.full((crop_t - T_work,), times_work[-1])], axis=0)
                        else:
                            times_crop = None
                    else:
                        times_crop = None
                    parent_time_start = 0
                    parent_time_end = int(T_work)
                else:
                    spec_crop = spec_f[:, t_start:t_end]
                    times_crop = times_work[t_start:t_end] if times_work is not None else None

                if spec_crop.shape[1] < crop_t or spec_crop.shape[0] < crop_f:
                    print_status(
                        f"Skipping {file_id}: crop shape {spec_crop.shape} smaller than {crop_f}x{crop_t}",
                        "WARNING",
                    )
                    continue

                # Compute PdB_norm for this crop (match training normalization scope)
                if spec_kind == 'power':
                    pdB_crop = _power_to_db_norm(spec_crop)
                else:
                    pdB_crop = spec_crop.astype(np.float32)

                # Compute window timing for audio + metadata
                window_time_start, window_time_end = _compute_window_time_range(
                    times=np.asarray(times) if times is not None else None,
                    start_idx=t_start,
                    window_bins=crop_t,
                    win_dur=float(win_dur) if win_dur is not None else None,
                    overlap=float(overlap) if overlap is not None else None,
                )

                audio_start_time = None
                audio_end_time = None
                if clip_ts and window_time_start is not None and window_time_end is not None:
                    audio_start_time = (clip_ts + timedelta(seconds=float(window_time_start))).isoformat()
                    audio_end_time = (clip_ts + timedelta(seconds=float(window_time_end))).isoformat()

                item_id = _build_descriptive_window_id(
                    file_id=file_id,
                    base_id=base_id,
                    source_audio=source_audio,
                    device_code=data_source.get("device_code"),
                    audio_start_time=audio_start_time,
                    audio_end_time=audio_end_time,
                    window_start=start_idx,
                    window_time_start=window_time_start,
                    window_time_end=window_time_end,
                )

                out_mat = spec_out_dir / f"{item_id}.mat"
                mat_payload = {
                    "F": np.asarray(freqs_f) if freqs_f is not None else None,
                    "T": np.asarray(times_crop) if times_crop is not None else None,
                    "PdB_norm": pdB_crop,
                    "parent_freq_bin_start": np.int32(parent_freq_start),
                    "parent_freq_bin_end": np.int32(parent_freq_end),
                    "parent_time_bin_start": np.int32(parent_time_start),
                    "parent_time_bin_end": np.int32(parent_time_end),
                }
                if spec_kind == 'power':
                    mat_payload["P"] = spec_crop
                # Remove None entries
                mat_payload = {k: v for k, v in mat_payload.items() if v is not None}
                scipy.io.savemat(out_mat, mat_payload)

                # Export audio crop if requested and available
                out_audio = None
                if args.export_audio and raw_audio_path and raw_audio_path.exists() and window_time_start is not None and window_time_end is not None:
                    try:
                        with sf.SoundFile(str(raw_audio_path)) as f:
                            fs = f.samplerate
                            start_frame = int(max(0.0, float(window_time_start)) * fs)
                            end_frame = int(max(float(window_time_end), float(window_time_start)) * fs)
                            start_frame = max(0, min(start_frame, len(f)))
                            end_frame = max(start_frame, min(end_frame, len(f)))
                            f.seek(start_frame)
                            audio_data = f.read(end_frame - start_frame)
                        # Pad/trim to exact expected length
                        expected_samples = int(max(0.0, float(window_time_end) - float(window_time_start)) * fs)
                        if len(audio_data) < expected_samples:
                            audio_data = np.pad(audio_data, (0, expected_samples - len(audio_data)))
                        elif len(audio_data) > expected_samples:
                            audio_data = audio_data[:expected_samples]
                        out_audio = audio_out_dir / f"{item_id}.wav"
                        sf.write(str(out_audio), audio_data, fs)
                    except Exception as e:
                        print_status(f"Audio export failed for {item_id}: {e}", "WARNING")

                export_info[file_id] = {
                    "item_id": item_id,
                    "spectrogram_mat_path": _path_to_json_ref(out_mat, json_base_dir),
                    "audio_path": _path_to_json_ref(out_audio, json_base_dir) if out_audio else None,
                    "audio_start_time": audio_start_time,
                    "audio_end_time": audio_end_time,
                    "window_time_start": window_time_start,
                    "window_time_end": window_time_end,
                    "time_bin_seconds": group_time_bin_seconds,
                    "source_audio": source_audio,
                    "parent_spectrogram_mat_path": _path_to_json_ref(mat_path.resolve(), json_base_dir),
                    "parent_audio_path": _path_to_json_ref(raw_audio_path.resolve(), json_base_dir) if raw_audio_path else None,
                    "parent_freq_bin_start": int(parent_freq_start),
                    "parent_freq_bin_end": int(parent_freq_end),
                    "parent_time_bin_start": int(parent_time_start),
                    "parent_time_bin_end": int(parent_time_end),
                    "parent_original_shape": [int(F_dim), int(T_dim)],
                }

            if group_idx % 10 == 0 or group_idx == total_groups:
                print(
                    f"  Export groups {group_idx}/{total_groups} | "
                    f"windows written {len(export_info)}/{total_export_targets}",
                    end='\r',
                )

        if total_groups:
            print()

        print_status(f"Export complete: {len(export_info)} crops", "SUCCESS")
        if args.export_audio and missing_audio_sources:
            print_status(
                f"Audio source files not found for {len(missing_audio_sources)} base clips; "
                "those exported items have no clipped audio.",
                "WARNING",
            )

    # Decide which results to include in predictions.json
    results_for_tracker = results
    if args.export_crops and not args.export_all and not args.postprocess:
        results_for_tracker = [r for r in results if r['file_id'] in export_info]
        print_status(
            "Predictions JSON is filtered to exported windows only. "
            "Use --export-all to keep all inference windows in output-json.",
            "WARNING",
        )
    elif args.export_crops and not args.export_all and args.postprocess:
        print_status(
            "Postprocessing enabled: keeping all inference windows in predictions JSON "
            "(ignoring export-only JSON filtering) so temporal clustering can run correctly.",
            "WARNING",
        )

    # Create prediction tracker
    tracker = UnifiedPredictionTracker(args.output_json)
    
    # Set model info
    tracker.set_model_info(
        model_id=model_info['model_id'],
        architecture=model_info['architecture'],
        checkpoint_path=model_info['checkpoint_path'],
        trained_at=model_info['trained_at'],
        wandb_run_id=model_info['wandb_run_id'],
        input_shape=dataset.output_crop_shape,
        output_classes=["Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"]
    )
    
    # Set task type
    tracker.set_task_type('whale_detection')
    
    # Set data source and config
    if data_source:
        tracker.set_data_source(
            device_code=data_source.get('device_code', 'unknown'),
            date_from=data_source.get('date_from', ''),
            date_to=data_source.get('date_to', ''),
            sample_rate=data_source.get('sample_rate'),
        )
    
    if spec_config:
        tracker.set_spectrogram_config(spec_config)
    
    # Add predictions
    total_tracker_items = len(results_for_tracker)
    print_status(f"Building predictions JSON items ({total_tracker_items})...", "PROGRESS")
    progress_every = 2000 if total_tracker_items >= 10000 else 500
    time_bin_cache: Dict[str, Optional[float]] = {}
    for idx, result in enumerate(results_for_tracker, start=1):
        file_id = result['file_id']
        base_id = file_id.rsplit('_win', 1)[0] if '_win' in file_id else file_id
        file_info = file_info_map.get(file_id, file_info_map.get(base_id, {}))
        meta = result.get('meta', {})
        
        # Build model_outputs in unified format
        model_outputs = [{
            "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
            "score": result['confidence'],  # Store raw score (not thresholded)
        }]
        
        # Add item with unified format
        spectrogram_mat_path = None
        spectrogram_png_path = None
        audio_path = None

        mat_from_meta = _resolve_path(file_info.get('mat_path'), metadata_base)
        if mat_from_meta is not None:
            spectrogram_mat_path = _path_to_json_ref(mat_from_meta, json_base_dir)
        else:
            fallback_mat = (Path(args.mat_dir) / f"{base_id}.mat").resolve()
            if fallback_mat.exists():
                spectrogram_mat_path = _path_to_json_ref(fallback_mat, json_base_dir)

        png_from_meta = _resolve_path(
            file_info.get("spectrogram_png_path") or file_info.get("spectrogram_path"),
            metadata_base,
        )
        if png_from_meta is not None:
            spectrogram_png_path = _path_to_json_ref(png_from_meta, json_base_dir)

        source_audio = file_info.get('source_audio') or f"{base_id}.wav"
        audio_from_meta = _resolve_path(file_info.get('audio_path'), metadata_base)
        if audio_from_meta is not None:
            audio_path = _path_to_json_ref(audio_from_meta, json_base_dir)
        time_bin_seconds = meta.get('time_bin_seconds')
        if time_bin_seconds is None:
            time_bin_seconds = file_info.get('time_bin_seconds')
        if time_bin_seconds is None:
            if base_id not in time_bin_cache:
                mat_for_tbin = mat_from_meta
                if mat_for_tbin is None:
                    fallback_mat = (Path(args.mat_dir) / f"{base_id}.mat").resolve()
                    if fallback_mat.exists():
                        mat_for_tbin = fallback_mat
                time_bin_cache[base_id] = _infer_time_bin_seconds_from_mat(mat_for_tbin)
            time_bin_seconds = time_bin_cache.get(base_id)
        parent_audio_from_meta = _resolve_path(
            file_info.get('raw_audio_path') or file_info.get('audio_path'),
            metadata_base,
        )
        if parent_audio_from_meta is None and args.raw_audio_dir and source_audio:
            candidate_parent_audio = Path(args.raw_audio_dir) / source_audio
            if candidate_parent_audio.exists():
                parent_audio_from_meta = candidate_parent_audio.resolve()
        # Prefer richer metadata from file_info_map when available (e.g., test windows)
        window_start = meta.get('window_start')
        window_time_start = meta.get('window_time_start')
        window_time_end = meta.get('window_time_end')
        original_shape = meta.get('original_shape')
        parent_freq_bin_start = None
        parent_freq_bin_end = None
        parent_time_bin_start = None
        parent_time_bin_end = None
        if file_info:
            window_start = file_info.get('window_start', window_start)
            window_time_start = file_info.get('window_time_start', window_time_start)
            window_time_end = file_info.get('window_time_end', window_time_end)
            original_shape = file_info.get('original_shape', original_shape)

        # Override paths and timing if we exported crops
        export_item = export_info.get(file_id)
        if export_item:
            window_time_start = export_item.get('window_time_start', window_time_start)
            window_time_end = export_item.get('window_time_end', window_time_end)
            time_bin_seconds = export_item.get('time_bin_seconds', time_bin_seconds)
            parent_freq_bin_start = export_item.get('parent_freq_bin_start')
            parent_freq_bin_end = export_item.get('parent_freq_bin_end')
            parent_time_bin_start = export_item.get('parent_time_bin_start')
            parent_time_bin_end = export_item.get('parent_time_bin_end')
        # Backward-compatibility: older inference exports can store window_time_*
        # in bin units. Convert here so downstream JSON stays in seconds.
        window_time_start, window_time_end = _maybe_convert_window_times_from_bins(
            window_time_start=window_time_start,
            window_time_end=window_time_end,
            window_start_bin=window_start,
            crop_time_bins=meta.get('crop_time_bins'),
            time_bin_seconds=time_bin_seconds,
        )

        duration_sec = spec_config.get('context_duration') if spec_config else None
        if duration_sec is None and window_time_start is not None and window_time_end is not None:
            duration_sec = max(0.0, float(window_time_end) - float(window_time_start))

        audio_start_time = None
        audio_end_time = None
        parent_spectrogram_mat_path = spectrogram_mat_path
        parent_audio_path = _path_to_json_ref(parent_audio_from_meta, json_base_dir) if parent_audio_from_meta else audio_path
        if export_item:
            if export_item.get('spectrogram_mat_path'):
                spectrogram_mat_path = export_item.get('spectrogram_mat_path')
            if export_item.get('audio_path'):
                audio_path = export_item.get('audio_path')
            audio_start_time = export_item.get('audio_start_time')
            audio_end_time = export_item.get('audio_end_time')
            parent_spectrogram_mat_path = export_item.get('parent_spectrogram_mat_path', parent_spectrogram_mat_path)
            parent_audio_path = export_item.get('parent_audio_path', parent_audio_path)

        # Fallback parent crop indices for non-exported windows.
        if parent_time_bin_start is None or parent_time_bin_end is None:
            try:
                ws = int(window_start) if window_start is not None else None
                crop_t = int(meta.get('crop_time_bins')) if meta.get('crop_time_bins') is not None else None
                if ws is not None and crop_t is not None:
                    parent_time_bin_start = ws
                    parent_time_bin_end = ws + crop_t
            except (TypeError, ValueError):
                pass
        if parent_freq_bin_start is None or parent_freq_bin_end is None:
            try:
                crop_f = int(meta.get('crop_freq_bins')) if meta.get('crop_freq_bins') is not None else None
                orig_f = int(original_shape[0]) if isinstance(original_shape, (list, tuple)) and len(original_shape) >= 1 else None
                fs0 = int(meta.get('freq_slice_start')) if meta.get('freq_slice_start') is not None else 0
                fs1 = int(meta.get('freq_slice_end')) if meta.get('freq_slice_end') is not None else orig_f
                if crop_f is not None:
                    if fs1 is not None:
                        span = max(0, int(fs1) - int(fs0))
                        center_offset = max(0, (span - crop_f) // 2)
                        parent_freq_bin_start = int(fs0) + int(center_offset)
                    else:
                        parent_freq_bin_start = int(fs0)
                    parent_freq_bin_end = int(parent_freq_bin_start) + int(crop_f)
            except (TypeError, ValueError):
                pass

        # Compute absolute window timestamps for non-exported windows when possible.
        if audio_start_time is None or audio_end_time is None:
            clip_ts = None
            if file_info.get("audio_timestamp"):
                try:
                    clip_ts = parse_datetime(file_info.get("audio_timestamp"))
                except Exception:
                    clip_ts = None
            if clip_ts is None:
                source_name = source_audio or f"{base_id}.wav"
                clip_ts = extract_timestamp_from_filename(source_name)
            if clip_ts and window_time_start is not None and window_time_end is not None:
                if audio_start_time is None:
                    audio_start_time = (clip_ts + timedelta(seconds=float(window_time_start))).isoformat()
                if audio_end_time is None:
                    audio_end_time = (clip_ts + timedelta(seconds=float(window_time_end))).isoformat()

        item_id = (
            export_item.get("item_id")
            if export_item and export_item.get("item_id")
            else _build_descriptive_window_id(
                file_id=file_id,
                base_id=base_id,
                source_audio=source_audio,
                device_code=data_source.get("device_code"),
                audio_start_time=audio_start_time,
                audio_end_time=audio_end_time,
                window_start=window_start,
                window_time_start=window_time_start,
                window_time_end=window_time_end,
            )
        )

        tracker.add_item(
            item_id=item_id,
            model_outputs=model_outputs,
            mat_path=spectrogram_mat_path,
            audio_path=audio_path,
            spectrogram_path=spectrogram_png_path,
            audio_timestamp=audio_start_time or file_info.get('audio_timestamp', ''),
            duration_sec=duration_sec,
            # Additional metadata
            source_audio=source_audio,
            segment_start_sec=file_info.get('segment_start_sec', window_time_start),
            segment_end_sec=file_info.get('segment_end_sec', window_time_end),
            segment_index=file_info.get('segment_index', file_info.get('window_index')),
            chunk_shape=file_info.get('chunk_shape'),
            # Descriptive aliases for spectrogram paths
            spectrogram_mat_path=spectrogram_mat_path,
            spectrogram_png_path=spectrogram_png_path,
            # Crop/window metadata
            original_shape=original_shape,
            crop_size=meta.get('crop_size'),
            crop_applied=meta.get('crop_applied'),
            crop_type=meta.get('crop_type'),
            window_start=window_start,
            window_start_bin=window_start,
            window_time_start=window_time_start,
            window_time_end=window_time_end,
            time_bin_seconds=time_bin_seconds,
            parent_spectrogram_mat_path=parent_spectrogram_mat_path,
            parent_audio_path=parent_audio_path,
            parent_freq_bin_start=parent_freq_bin_start,
            parent_freq_bin_end=parent_freq_bin_end,
            parent_time_bin_start=parent_time_bin_start,
            parent_time_bin_end=parent_time_bin_end,
            audio_start_time=audio_start_time,
            audio_end_time=audio_end_time,
        )
        if idx % progress_every == 0 or idx == total_tracker_items:
            print(f"  Added {idx}/{total_tracker_items} items", end='\r')
    if total_tracker_items:
        print()
    
    # Save predictions
    print_status("Saving predictions JSON...", "PROGRESS")
    tracker.save()
    
    # Print summary
    print_header("RESULTS")
    summary = tracker.summary()
    print(f"Total items: {summary.get('total_items', 0)}")
    if 'mean_score' in summary:
        print(f"Mean confidence: {summary['mean_score']:.4f}")
        print(f"Min confidence: {summary['min_score']:.4f}")
        print(f"Max confidence: {summary['max_score']:.4f}")
    
    # Show threshold-based counts as preview
    class_name = "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        above = len(tracker.get_items_by_score_threshold(class_name, threshold, above=True))
        total_items = max(summary.get('total_items', 0), 1)
        print(f"  >= {threshold:.1f}: {above} ({100*above/total_items:.1f}%)")
    
    print(f"\nPredictions saved to: {args.output_json}")

    if args.postprocess:
        postprocess_script = REPO_ROOT / "scripts" / "inference" / "postprocess_predictions.py"
        if not postprocess_script.exists():
            raise FileNotFoundError(f"Postprocess script not found: {postprocess_script}")

        post_out = (
            Path(args.postprocess_output_json)
            if args.postprocess_output_json
            else Path(args.output_json).with_name(f"{Path(args.output_json).stem}_postprocessed.json")
        )
        post_cmd: List[str] = [
            sys.executable,
            str(postprocess_script),
            "--input-json",
            str(args.output_json),
            "--output-json",
            str(post_out),
            "--low-threshold",
            str(args.postprocess_low_threshold),
            "--high-threshold",
            str(args.postprocess_high_threshold),
            "--min-members",
            str(args.postprocess_min_members),
            "--min-duration-sec",
            str(args.postprocess_min_duration_sec),
        ]
        if args.postprocess_class_hierarchy:
            post_cmd.extend(["--class-hierarchy", str(args.postprocess_class_hierarchy)])
        if args.postprocess_max_gap_seconds is not None:
            post_cmd.extend(["--max-gap-seconds", str(args.postprocess_max_gap_seconds)])
        if args.postprocess_events_csv:
            post_cmd.extend(["--events-csv", str(args.postprocess_events_csv)])
        if args.postprocess_summary_md:
            post_cmd.extend(["--summary-md", str(args.postprocess_summary_md)])
        if args.postprocess_debug_json:
            post_cmd.extend(["--debug-json", str(args.postprocess_debug_json)])
        if args.postprocess_merge_event_media:
            post_cmd.append("--merge-event-media")
        if args.postprocess_merge_min_score is not None:
            post_cmd.extend(["--merge-min-score", str(args.postprocess_merge_min_score)])
        if args.postprocess_event_media_dir:
            post_cmd.extend(["--event-media-dir", str(args.postprocess_event_media_dir)])
        if args.postprocess_replace_items_with_events:
            post_cmd.append("--replace-items-with-events")
        if args.postprocess_merge_across_source_audio:
            post_cmd.append("--merge-across-source-audio")

        print_status("Running postprocessing on predictions...", "PROGRESS")
        subprocess.run(post_cmd, check=True)
        print(f"Postprocessed predictions saved to: {post_out}")

    print_status("Inference complete!", "SUCCESS")


if __name__ == "__main__":
    main()
