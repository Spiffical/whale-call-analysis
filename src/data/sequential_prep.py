from __future__ import annotations

import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_datetime(dt_str: str) -> datetime:
    """Parse ISO format datetime string to timezone-aware datetime."""
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from ONC audio filename (DEVICE_YYYYMMDDTHHMMSSmmm.*)."""
    try:
        base = Path(filename).stem
        parts = base.split('_')
        if len(parts) >= 2:
            ts_str = parts[1].replace('Z', '')
            if '.' in ts_str:
                ts_str = ts_str.split('.')[0]
            dt = datetime.strptime(ts_str[:15], '%Y%m%dT%H%M%S')
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def compute_window_positions(total_bins: int, window_size: int) -> List[int]:
    """Compute start positions that tile the spectrogram with minimal overlap."""
    if total_bins <= window_size:
        return [0]
    n_windows = math.ceil(total_bins / window_size)
    if n_windows > 1:
        step = (total_bins - window_size) / (n_windows - 1)
        return [int(round(i * step)) for i in range(n_windows)]
    return [0]


def crop_to_freq_lims(
    freqs: np.ndarray,
    data: np.ndarray,
    freq_min: float,
    freq_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop frequency dimension to specified limits."""
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[freq_mask], data[freq_mask, :]


def load_dataset_documentation(doc_path: str) -> Dict[str, Any]:
    """Load parameters from dataset_documentation.json."""
    path = Path(doc_path)
    if path.is_dir():
        path = path / 'dataset_documentation.json'
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def get_processing_params(
    dataset_doc: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    crop_size_override: Optional[int] = None,
    freq_lims_override: Optional[Tuple[float, float]] = None,
    win_dur_override: Optional[float] = None,
    overlap_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get processing parameters with precedence: CLI override > dataset_doc > model args.pkl > defaults.

    Returns dict with keys: crop_size, freq_lims, win_dur, overlap, clim
    """
    # Defaults
    params = {
        'crop_size': 96,
        'freq_lims': (5, 100),
        'win_dur': 1.0,
        'overlap': 0.9,
        'clim': (-40, 0),
    }

    # Load from model args.pkl if available
    if model_path:
        args_path = Path(model_path) / 'args.pkl'
        if args_path.exists():
            with open(args_path, 'rb') as f:
                model_args = pickle.load(f)
                if hasattr(model_args, 'crop_size'):
                    params['crop_size'] = model_args.crop_size
                elif isinstance(model_args, dict) and 'crop_size' in model_args:
                    params['crop_size'] = model_args['crop_size']

    # Load from dataset documentation if available
    if dataset_doc:
        proc_params = dataset_doc.get('processing_parameters', {})
        spec_gen = proc_params.get('spectrogram_generation', {})
        freq_filt = proc_params.get('frequency_filtering', {})

        # Extract frequency limits
        freq_limits = spec_gen.get('frequency_limits_hz', {})
        if freq_limits:
            params['freq_lims'] = (freq_limits.get('min', 5), freq_limits.get('max', 100))

        # Window duration and overlap
        if 'window_duration_s' in spec_gen:
            params['win_dur'] = spec_gen['window_duration_s']
        if 'overlap_ratio' in spec_gen:
            params['overlap'] = spec_gen['overlap_ratio']

        # Color limits
        clim = spec_gen.get('color_limits_db', {})
        if clim:
            params['clim'] = (clim.get('min', -40), clim.get('max', 0))

        # Crop size from actual_freq_bins (e.g., "96 bins" -> 96)
        actual_bins = freq_filt.get('actual_freq_bins', '')
        if actual_bins and isinstance(actual_bins, str):
            try:
                params['crop_size'] = int(actual_bins.split()[0])
            except (ValueError, IndexError):
                pass

    # Apply CLI overrides (highest precedence)
    if crop_size_override is not None:
        params['crop_size'] = crop_size_override
    if freq_lims_override is not None:
        params['freq_lims'] = freq_lims_override
    if win_dur_override is not None:
        params['win_dur'] = win_dur_override
    if overlap_override is not None:
        params['overlap'] = overlap_override

    return params
