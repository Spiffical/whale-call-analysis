#!/usr/bin/env python3
"""
Check consistency between training prep settings and test-window prep outputs.

This script is intentionally lightweight: it uses only stdlib + repo utilities
so it can run even in minimal environments.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

# Keep stdlib-only for minimal environments
def compute_window_positions(total_bins: int, window_size: int):
    """Compute start positions that tile the spectrogram with minimal overlap."""
    if total_bins <= window_size:
        return [0]
    n_windows = int(math.ceil(total_bins / window_size))
    if n_windows > 1:
        step = (total_bins - window_size) / (n_windows - 1)
        return [int(round(i * step)) for i in range(n_windows)]
    return [0]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def _load_args_pkl(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        with open(path, 'rb') as f:
            args = pickle.load(f)
        if isinstance(args, dict):
            return args
        # argparse.Namespace
        return {k: getattr(args, k) for k in dir(args) if not k.startswith('_')}
    except Exception:
        return {}


def _extract_from_yaml_text(text: str) -> Dict[str, Any]:
    """Best-effort parser for dataset_config.yaml without external deps."""
    def _find_float(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _find_bool(pattern: str) -> Optional[bool]:
        m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if not m:
            return None
        return m.group(1).strip().lower() == 'true'

    def _find_float_in_block(block: str, key: str) -> Optional[float]:
        m = re.search(rf'^\s*{re.escape(key)}:\s*([0-9.]+)\s*$', block, re.MULTILINE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    out: Dict[str, Any] = {}
    out['window_duration'] = _find_float(r'^\s*window_duration:\s*([0-9.]+)\s*$')
    out['overlap'] = _find_float(r'^\s*overlap:\s*([0-9.]+)\s*$')

    freq_block = re.search(r'frequency_limits:\s*\n(?P<blk>(?:\s+.*\n)+)', text)
    if freq_block:
        blk = freq_block.group('blk')
        out['freq_min'] = _find_float_in_block(blk, 'min')
        out['freq_max'] = _find_float_in_block(blk, 'max')

    color_block = re.search(r'color_limits:\s*\n(?P<blk>(?:\s+.*\n)+)', text)
    if color_block:
        blk = color_block.group('blk')
        out['clim_min'] = _find_float_in_block(blk, 'min')
        out['clim_max'] = _find_float_in_block(blk, 'max')

    out['log_frequency'] = _find_bool(r'^\s*log_frequency:\s*(true|false)\s*$')
    out['context_duration'] = _find_float(r'^\s*context_duration:\s*([0-9.]+)\s*$')
    return out


def _load_dataset_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        cs = data.get('custom_spectrograms', {})
        tc = data.get('temporal_context', {})
        return {
            'window_duration': cs.get('window_duration'),
            'overlap': cs.get('overlap'),
            'freq_min': (cs.get('frequency_limits') or {}).get('min'),
            'freq_max': (cs.get('frequency_limits') or {}).get('max'),
            'clim_min': (cs.get('color_limits') or {}).get('min'),
            'clim_max': (cs.get('color_limits') or {}).get('max'),
            'log_frequency': cs.get('log_frequency'),
            'context_duration': tc.get('context_duration'),
        }
    except Exception:
        text = path.read_text()
        return _extract_from_yaml_text(text)


def _fmt(v: Any) -> str:
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check training vs test prep consistency")
    parser.add_argument('--metadata', type=str, required=True, help='Path to test_windows metadata.json')
    parser.add_argument('--dataset-config', type=str, default='config/dataset_config.yaml',
                        help='Path to dataset_config.yaml used for training')
    parser.add_argument('--model-args', type=str, default=None,
                        help='Path to args.pkl (trained model) for crop/min/max DB checks')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Tolerance for timing checks (seconds)')
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    cfg_path = Path(args.dataset_config)
    args_path = Path(args.model_args) if args.model_args else None

    metadata = _load_json(meta_path)
    proc = metadata.get('processing_parameters', {})

    train_cfg = _load_dataset_config(cfg_path) if cfg_path.exists() else {}
    model_args = _load_args_pkl(args_path) if args_path else {}

    crop_size = proc.get('crop_size')
    freq_lims = proc.get('freq_lims_hz')
    win_dur = proc.get('win_dur_s')
    overlap = proc.get('overlap')

    print("TRAINING CONFIG (dataset_config.yaml)")
    print(f"  window_duration: {_fmt(train_cfg.get('window_duration'))}")
    print(f"  overlap: {_fmt(train_cfg.get('overlap'))}")
    print(f"  freq_min: {_fmt(train_cfg.get('freq_min'))}")
    print(f"  freq_max: {_fmt(train_cfg.get('freq_max'))}")
    print(f"  clim_min: {_fmt(train_cfg.get('clim_min'))}")
    print(f"  clim_max: {_fmt(train_cfg.get('clim_max'))}")
    print(f"  log_frequency: {_fmt(train_cfg.get('log_frequency'))}")
    print(f"  context_duration: {_fmt(train_cfg.get('context_duration'))}")

    print("\nMODEL ARGS (args.pkl)")
    print(f"  crop_size: {_fmt(model_args.get('crop_size'))}")
    print(f"  min_db: {_fmt(model_args.get('min_db'))}")
    print(f"  max_db: {_fmt(model_args.get('max_db'))}")

    print("\nTEST WINDOW METADATA")
    print(f"  crop_size: {_fmt(crop_size)}")
    print(f"  freq_lims_hz: {_fmt(freq_lims)}")
    print(f"  win_dur_s: {_fmt(win_dur)}")
    print(f"  overlap: {_fmt(overlap)}")
    print(f"  clim_db: {_fmt(proc.get('clim_db'))}")

    print("\nPARAMETER MISMATCHES")
    mismatches = []
    if train_cfg.get('window_duration') is not None and win_dur is not None:
        if float(train_cfg['window_duration']) != float(win_dur):
            mismatches.append(f"window_duration: train={train_cfg['window_duration']} vs test={win_dur}")
    if train_cfg.get('overlap') is not None and overlap is not None:
        if float(train_cfg['overlap']) != float(overlap):
            mismatches.append(f"overlap: train={train_cfg['overlap']} vs test={overlap}")
    if train_cfg.get('freq_min') is not None and freq_lims:
        if float(train_cfg['freq_min']) != float(freq_lims[0]):
            mismatches.append(f"freq_min: train={train_cfg['freq_min']} vs test={freq_lims[0]}")
    if train_cfg.get('freq_max') is not None and freq_lims:
        if float(train_cfg['freq_max']) != float(freq_lims[1]):
            mismatches.append(f"freq_max: train={train_cfg['freq_max']} vs test={freq_lims[1]}")
    if model_args.get('crop_size') is not None and crop_size is not None:
        if int(model_args['crop_size']) != int(crop_size):
            mismatches.append(f"crop_size: model={model_args['crop_size']} vs test={crop_size}")
    if train_cfg.get('clim_min') is not None and proc.get('clim_db'):
        if float(train_cfg['clim_min']) != float(proc['clim_db'][0]):
            mismatches.append(f"clim_min: train={train_cfg['clim_min']} vs test={proc['clim_db'][0]}")
    if train_cfg.get('clim_max') is not None and proc.get('clim_db'):
        if float(train_cfg['clim_max']) != float(proc['clim_db'][1]):
            mismatches.append(f"clim_max: train={train_cfg['clim_max']} vs test={proc['clim_db'][1]}")

    if mismatches:
        for m in mismatches:
            print(f"  - {m}")
    else:
        print("  None")

    # Chunk-level checks
    chunks = metadata.get('chunks', [])
    print("\nCHUNK CONSISTENCY CHECKS")
    if not chunks:
        print("  No chunks found in metadata.")
        return 1

    # Group by source audio
    by_source = defaultdict(list)
    for ch in chunks:
        by_source[ch.get('source_audio', 'UNKNOWN')].append(ch)

    bad_shapes = 0
    bad_orig_freq = 0
    bad_time_bounds = 0
    bad_time_calc = 0
    start_mismatches = 0

    if win_dur is None or overlap is None or crop_size is None:
        print("  Missing win_dur/overlap/crop_size in metadata; skipping timing checks.")
    else:
        nominal_hop = float(win_dur) * (1.0 - float(overlap))
        offset_notes = []
        clip_durations = []
        for source_audio, chs in by_source.items():
            # All chunks from same file should share original_shape[1]
            total_bins = None
            hop_candidates = []
            for c in chs:
                orig = c.get('original_shape') or [None, None]
                if total_bins is None and len(orig) >= 2:
                    total_bins = orig[1]
                # Shape checks
                chunk_shape = c.get('chunk_shape') or [None, None]
                if chunk_shape[0] != crop_size or chunk_shape[1] != crop_size:
                    bad_shapes += 1
                if orig[0] != crop_size:
                    bad_orig_freq += 1

                # Timing checks
                ws = c.get('window_start')
                wts = c.get('window_time_start')
                wte = c.get('window_time_end')
                if ws is None or wts is None or wte is None:
                    continue

                if total_bins is not None:
                    window_bins = min(int(crop_size), int(total_bins) - int(ws))
                else:
                    window_bins = int(crop_size)

                if window_bins > 1:
                    hop_est = (float(wte) - float(wts) - float(win_dur)) / float(window_bins - 1)
                    if hop_est > 0:
                        hop_candidates.append(hop_est)

                # Bounds: should not exceed clip duration implied by total bins
                if total_bins is not None:
                    clip_dur = float(win_dur) + (float(total_bins) - 1) * nominal_hop
                    if float(wte) > clip_dur + args.tolerance:
                        bad_time_bounds += 1

            # Use estimated hop if available; otherwise nominal hop
            if hop_candidates:
                hop_candidates.sort()
                hop = hop_candidates[len(hop_candidates) // 2]
            else:
                hop = nominal_hop

            # Estimate systematic time offset (e.g., edge-padded clips)
            offset_candidates = []
            for c in chs:
                ws = c.get('window_start')
                wts = c.get('window_time_start')
                if ws is None or wts is None:
                    continue
                if int(ws) == 0:
                    continue
                offset_candidates.append(float(wts) - float(ws) * hop)
            if offset_candidates:
                offset_candidates.sort()
                offset = offset_candidates[len(offset_candidates) // 2]
            else:
                offset = 0.0
            if abs(offset) > args.tolerance:
                offset_notes.append(offset)

            # Second pass to validate timing with the chosen hop
            for c in chs:
                ws = c.get('window_start')
                wts = c.get('window_time_start')
                wte = c.get('window_time_end')
                if ws is None or wts is None or wte is None:
                    continue
                if total_bins is not None:
                    window_bins = min(int(crop_size), int(total_bins) - int(ws))
                else:
                    window_bins = int(crop_size)

                expected_start = float(ws) * hop + offset
                if expected_start < 0:
                    expected_start = 0.0
                expected_end = expected_start + (window_bins - 1) * hop + float(win_dur)
                if abs(float(wts) - expected_start) > args.tolerance:
                    bad_time_calc += 1
                if abs(float(wte) - expected_end) > args.tolerance:
                    bad_time_calc += 1

            # Start index coverage check
            if total_bins is not None:
                expected = set(compute_window_positions(int(total_bins), int(crop_size)))
                observed = {int(c.get('window_start')) for c in chs if c.get('window_start') is not None}
                if expected != observed:
                    start_mismatches += 1

            if total_bins is not None:
                clip_dur = float(win_dur) + (float(total_bins) - 1) * hop
                clip_durations.append(clip_dur)

    print(f"  Sources checked: {len(by_source)}")
    print(f"  Chunk shape mismatches: {bad_shapes}")
    print(f"  Original freq bin mismatches: {bad_orig_freq}")
    print(f"  Window start coverage mismatches: {start_mismatches}")
    print(f"  Window timing calc mismatches: {bad_time_calc}")
    print(f"  Window timing bounds violations: {bad_time_bounds}")
    if win_dur is not None and overlap is not None and crop_size is not None:
        if offset_notes:
            # Round to milliseconds for readability
            offs = sorted(set(round(o, 3) for o in offset_notes))
            print(f"  Detected window_time_start offsets (s): {offs}")
        if clip_durations:
            clip_durations.sort()
            mid = clip_durations[len(clip_durations) // 2]
            clip_min = clip_durations[0]
            clip_max = clip_durations[-1]
            print(f"  Inference clip duration (median/min/max, s): {mid:.2f} / {clip_min:.2f} / {clip_max:.2f}")

            ctx = train_cfg.get('context_duration')
            if ctx:
                try:
                    ctx = float(ctx)
                    if ctx > 0 and mid / ctx >= 2.0:
                        print(
                            f"  WARNING: training context is {ctx:.2f}s but inference spectrograms are ~{mid:.2f}s.\n"
                            "           Spectrogram normalization is per-clip, so this changes dynamic range."
                        )
                except Exception:
                    pass

    if bad_shapes or bad_orig_freq or start_mismatches or bad_time_calc or bad_time_bounds:
        print("\nRESULT: Issues detected in test window consistency checks.")
        return 2

    print("\nRESULT: Metadata-level consistency checks passed.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
