#!/usr/bin/env python3
"""
Prepare train-style spectrograms for comparison against existing training MATs.

- Call-centered mode: 40s context around annotated calls (training pipeline)
- Sliding-window mode: 40s windows across a clip (no call info)

Outputs MATs and an optional comparison report vs an existing MAT directory.
"""
from __future__ import annotations

import argparse
import sys
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
from matplotlib import pyplot as plt
from tqdm import tqdm

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Prefer local onc-hydrophone-data repo if present
ONC_REPO = Path("/home/sbialek/ONC/onc-hydrophone-data")
if ONC_REPO.exists() and str(ONC_REPO) not in sys.path:
    sys.path.insert(0, str(ONC_REPO))

from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator
from src.data.sequential_prep import load_dataset_documentation, get_processing_params, crop_to_freq_lims
from scripts.diagnostics.compare_train_newprep import (
    CallRow,
    parse_clip_ts,
    build_audio_index,
    _find_audio_file_with_index,
    _find_adjacent_file_with_index,
    load_training_power,
    normalize_db_to_unit,
    power_to_db_norm,
    center_crop_with_pad,
)


@dataclass
class WorkItem:
    call: CallRow
    out_name: str


def _to_mono_audio(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=1)
    return arr.reshape(arr.shape[0], -1).mean(axis=1)


def _parse_mat_list(mat_list: Path) -> List[WorkItem]:
    pattern = re.compile(r'^(?P<clip>.+)_(?P<begin>-?\d+(?:\.\d+)?)s_(?P<end>-?\d+(?:\.\d+)?)s')
    rows: List[WorkItem] = []
    with open(mat_list, 'r') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if not name.lower().endswith('.mat'):
                name = name + '.mat'
            p = Path(name)
            m = pattern.match(p.stem)
            if not m:
                continue
            clip = m.group('clip')
            begin_s = float(m.group('begin'))
            end_s = float(m.group('end'))
            clip_dt = parse_clip_ts(clip)
            call_dt = clip_dt + timedelta(seconds=begin_s) if clip_dt else pd.NaT
            call = CallRow(clip, begin_s, end_s, call_dt, p)
            rows.append(WorkItem(call, p.name))
    return rows


def _parse_calls_csv(calls_csv: Path) -> List[WorkItem]:
    df = pd.read_csv(calls_csv)
    cols = {c.lower(): c for c in df.columns}
    if 'clip' not in cols or 'begin_s' not in cols or 'end_s' not in cols:
        raise ValueError("calls-csv must include columns: clip, begin_s, end_s")
    df = df.rename(columns={cols['clip']: 'clip', cols['begin_s']: 'begin_s', cols['end_s']: 'end_s'})
    rows: List[WorkItem] = []
    for _, row in df.iterrows():
        clip = str(row['clip'])
        begin_s = float(row['begin_s'])
        end_s = float(row['end_s'])
        clip_dt = parse_clip_ts(clip)
        call_dt = clip_dt + timedelta(seconds=begin_s) if clip_dt else pd.NaT
        call = CallRow(clip, begin_s, end_s, call_dt, None)
        out_name = f"{clip}_{begin_s:.1f}s_{end_s:.1f}s_trainstyle.mat"
        rows.append(WorkItem(call, out_name))
    return rows


def _parse_clip_list(clip_list: Path, window_s: float, step_s: float) -> List[WorkItem]:
    rows: List[WorkItem] = []
    with open(clip_list, 'r') as f:
        for line in f:
            clip = line.strip()
            if not clip:
                continue
            start = 0.0
            while start + window_s <= 300.0 + 1e-6:
                end = start + window_s
                clip_dt = parse_clip_ts(clip)
                call_dt = clip_dt + timedelta(seconds=start) if clip_dt else pd.NaT
                call = CallRow(clip, start, end, call_dt, None)
                out_name = f"{clip}_{start:.1f}s_{end:.1f}s_window.mat"
                rows.append(WorkItem(call, out_name))
                start += step_s
    return rows


def _load_context_audio(
    audio_dir: Path,
    clip: str,
    start_s: float,
    end_s: float,
    context_s: float,
    audio_index: Dict[str, Path],
    audio_index_by_second: Dict[str, Path],
) -> Tuple[np.ndarray, int]:
    cur_file = _find_audio_file_with_index(audio_dir, clip, audio_index, audio_index_by_second)
    if cur_file is None:
        raise FileNotFoundError(f"Audio file not found for {clip}")

    data, fs = sf.read(str(cur_file))
    data = _to_mono_audio(data)
    total_s = len(data) / fs

    def _slice_from_file(arr: np.ndarray, s0: float, s1: float) -> np.ndarray:
        s0 = max(0.0, s0)
        s1 = min(total_s, s1)
        i0 = int(round(s0 * fs))
        i1 = int(round(s1 * fs))
        if i1 <= i0:
            return np.zeros(0, dtype=arr.dtype)
        return arr[i0:i1]

    chunks = []
    if start_s < 0:
        clip_dt = parse_clip_ts(clip)
        prev_file = _find_adjacent_file_with_index(audio_dir, clip.split('_')[0], clip_dt - timedelta(minutes=5), audio_index_by_second) if clip_dt else None
        if prev_file and prev_file.exists():
            prev_data, prev_fs = sf.read(str(prev_file))
            prev_data = _to_mono_audio(prev_data)
            if prev_fs == fs:
                need = -start_s
                take = int(round(need * fs))
                chunks.append(prev_data[-take:])
            else:
                chunks.append(np.zeros(int(round(-start_s * fs)), dtype=data.dtype))
        else:
            chunks.append(np.zeros(int(round(-start_s * fs)), dtype=data.dtype))

    # main segment
    chunks.append(_slice_from_file(data, start_s, end_s))

    if end_s > total_s:
        clip_dt = parse_clip_ts(clip)
        next_file = _find_adjacent_file_with_index(audio_dir, clip.split('_')[0], clip_dt + timedelta(minutes=5), audio_index_by_second) if clip_dt else None
        if next_file and next_file.exists():
            next_data, next_fs = sf.read(str(next_file))
            next_data = _to_mono_audio(next_data)
            if next_fs == fs:
                need = end_s - total_s
                take = int(round(need * fs))
                chunks.append(next_data[:take])
            else:
                chunks.append(np.zeros(int(round((end_s - total_s) * fs)), dtype=data.dtype))
        else:
            chunks.append(np.zeros(int(round((end_s - total_s) * fs)), dtype=data.dtype))

    full = np.concatenate(chunks) if chunks else np.zeros(0, dtype=data.dtype)
    target = int(round(context_s * fs))
    if len(full) > target:
        full = full[:target]
    elif len(full) < target:
        full = np.pad(full, (0, target - len(full)))
    return full, fs


def _save_db_image(db: np.ndarray, path: Path, vmin: float, vmax: float, cmap: str = "magma") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.2, 2.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(db, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create train-style MATs for testing")
    ap.add_argument('--mat-list', type=str, default=None, help='List of existing MAT filenames (call-centered)')
    ap.add_argument('--calls-csv', type=str, default=None, help='CSV with clip, begin_s, end_s')
    ap.add_argument('--slide', action='store_true', help='Generate sliding 40s windows per clip')
    ap.add_argument('--clip-list', type=str, default=None, help='List of clip IDs for sliding windows')
    ap.add_argument('--audio-dir', type=str, required=True)
    ap.add_argument('--dataset-doc', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--compare-dir', type=str, default=None, help='Existing training MAT dir for comparison')
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--device', type=str, default='ICLISTENHF1353')
    ap.add_argument('--spec-backend', type=str, default='auto', choices=['auto', 'scipy', 'torch'])
    ap.add_argument('--window-s', type=float, default=40.0)
    ap.add_argument('--step-s', type=float, default=40.0)
    ap.add_argument('--save-images', action='store_true')
    args = ap.parse_args()

    if not args.mat_list and not args.calls_csv and not args.slide:
        raise RuntimeError("Provide --mat-list, --calls-csv, or --slide with --clip-list")

    dataset_doc = load_dataset_documentation(args.dataset_doc)
    proc = get_processing_params(dataset_doc=dataset_doc, model_path=None)
    freq_min, freq_max = proc['freq_lims']
    win_dur = proc['win_dur']
    overlap = proc['overlap']
    clim_min, clim_max = proc['clim']
    min_db, max_db = -80.0, 0.0

    spec_gen = SpectrogramGenerator(
        win_dur=win_dur,
        overlap=overlap,
        freq_lims=(freq_min, freq_max),
        clim=(clim_min, clim_max),
        log_freq=False,
        crop_freq_lims=False,
        backend=args.spec_backend,
        quiet=True,
    )

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"

    compare_dir = Path(args.compare_dir) if args.compare_dir else None

    items: List[WorkItem] = []
    if args.mat_list:
        items.extend(_parse_mat_list(Path(args.mat_list)))
    if args.calls_csv:
        items.extend(_parse_calls_csv(Path(args.calls_csv)))
    if args.slide:
        if not args.clip_list:
            raise RuntimeError("--slide requires --clip-list")
        items.extend(_parse_clip_list(Path(args.clip_list), args.window_s, args.step_s))

    if args.device:
        items = [it for it in items if args.device in it.call.clip]

    if args.n:
        items = items[:args.n]

    audio_index, audio_index_by_second = build_audio_index(audio_dir)

    rows = []
    for it in tqdm(items, desc="Generating", unit="mat"):
        call = it.call
        # Determine 40s context window for call-centered mode
        if args.slide:
            start_s = call.begin_s
            end_s = call.end_s
            context_s = args.window_s
        else:
            padding = (args.window_s - (call.end_s - call.begin_s)) / 2.0
            start_s = call.begin_s - padding
            end_s = call.end_s + padding
            context_s = args.window_s

        audio_seg, fs = _load_context_audio(
            audio_dir, call.clip, start_s, end_s, context_s, audio_index, audio_index_by_second
        )

        freqs, times, Sxx, PdB = spec_gen.compute_spectrogram(audio_seg, fs)
        freqs_c, PdB_c = crop_to_freq_lims(freqs, PdB, freq_min, freq_max)
        _, Sxx_c = crop_to_freq_lims(freqs, Sxx, freq_min, freq_max)

        out_path = out_dir / it.out_name
        scipy.io.savemat(str(out_path), {
            'F': freqs_c,
            'T': times,
            'P': Sxx_c,
            'PdB_norm': PdB_c,
            'freq_min': freq_min,
            'freq_max': freq_max,
            'window_s': context_s,
            'backend': args.spec_backend,
        })

        row = {
            'clip': call.clip,
            'begin_s': call.begin_s,
            'end_s': call.end_s,
            'out_mat': str(out_path),
            'backend': args.spec_backend,
        }

        if compare_dir and not args.slide:
            ref_path = compare_dir / it.out_name
            if ref_path.exists():
                ref_spec = load_training_power(ref_path)
                gen_spec = load_training_power(out_path)
                # Convert power to dB if needed
                ref_is_db = np.nanmax(ref_spec) <= 10 and np.nanmin(ref_spec) < 0
                gen_is_db = np.nanmax(gen_spec) <= 10 and np.nanmin(gen_spec) < 0
                if not ref_is_db:
                    ref_spec = power_to_db_norm(ref_spec)
                if not gen_is_db:
                    gen_spec = power_to_db_norm(gen_spec)
                # Normalize to unit
                ref_norm = normalize_db_to_unit(ref_spec, min_db, max_db)
                gen_norm = normalize_db_to_unit(gen_spec, min_db, max_db)
                # Align dims
                if ref_norm.shape != gen_norm.shape:
                    # center crop to smallest common
                    f = min(ref_norm.shape[0], gen_norm.shape[0])
                    t = min(ref_norm.shape[1], gen_norm.shape[1])
                    ref_norm = center_crop_with_pad(ref_norm, f, t)
                    gen_norm = center_crop_with_pad(gen_norm, f, t)
                diff = np.abs(ref_norm - gen_norm)
                row['abs_diff_mean'] = float(np.mean(diff))
                row['abs_diff_max'] = float(np.max(diff))
                row['ref_mat'] = str(ref_path)

                if args.save_images:
                    img_id = out_path.stem
                    ref_img = img_dir / f"{img_id}_ref.png"
                    gen_img = img_dir / f"{img_id}_gen.png"
                    diff_img = img_dir / f"{img_id}_diff.png"
                    _save_db_image(ref_spec, ref_img, min_db, max_db, cmap="magma")
                    _save_db_image(gen_spec, gen_img, min_db, max_db, cmap="magma")
                    v = max(1.0, float(np.max(np.abs(gen_spec - ref_spec))))
                    _save_db_image(gen_spec - ref_spec, diff_img, -v, v, cmap="coolwarm")
                    row['ref_img'] = str(ref_img.relative_to(out_dir))
                    row['gen_img'] = str(gen_img.relative_to(out_dir))
                    row['diff_img'] = str(diff_img.relative_to(out_dir))
            else:
                row['ref_mat'] = str(ref_path)
                row['abs_diff_mean'] = None
                row['abs_diff_max'] = None

        rows.append(row)

    csv_path = out_dir / "report.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
