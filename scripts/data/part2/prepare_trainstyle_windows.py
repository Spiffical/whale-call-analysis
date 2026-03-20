#!/usr/bin/env python3
"""
Prepare spectrogram MATs for train-style comparison or Part 2 full-clip inference.

- Call-centered mode: 40s context around annotated calls (training pipeline)
- Sliding-window mode: fixed-duration MATs across a clip (used by Part 2 VM prep)

Outputs MATs and an optional comparison report vs an existing MAT directory.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from src.data.sequential_prep import crop_to_freq_lims, get_processing_params, load_dataset_documentation


@dataclass
class CallRow:
    clip: str
    begin_s: float
    end_s: float
    call_dt: datetime
    mat_path: Optional[Path] = None


def parse_clip_ts(clip: str) -> Optional[datetime]:
    match = re.search(r"_(\d{8}T\d{6})", str(clip))
    if not match:
        return None
    ts = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    return ts.replace(tzinfo=timezone.utc)


def build_audio_index(audio_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    by_stem: Dict[str, Path] = {}
    by_second: Dict[str, Path] = {}
    pattern = re.compile(r"^(?P<device>[^_]+)_(?P<ts>\d{8}T\d{6})")
    for path in audio_dir.rglob("*"):
        try:
            if not path.is_file() or path.suffix.lower() not in (".flac", ".wav"):
                continue
            stem = path.stem
            by_stem.setdefault(stem, path)
            match = pattern.match(stem)
            if match:
                key = f"{match.group('device')}_{match.group('ts')}"
                by_second.setdefault(key, path)
        except FileNotFoundError:
            continue
    return by_stem, by_second


def _find_audio_file_with_index(
    audio_dir: Path,
    clip_name: str,
    audio_index: Optional[Dict[str, Path]],
    audio_index_by_second: Optional[Dict[str, Path]],
) -> Optional[Path]:
    if audio_index is not None:
        stem = Path(clip_name).stem if clip_name.endswith((".flac", ".wav")) else clip_name
        if stem in audio_index:
            return audio_index[stem]
    if audio_index_by_second is not None:
        match = re.search(r"^(?P<device>[^_]+)_(?P<ts>\d{8}T\d{6})", clip_name)
        if match:
            key = f"{match.group('device')}_{match.group('ts')}"
            if key in audio_index_by_second:
                return audio_index_by_second[key]
    if clip_name.endswith((".flac", ".wav")):
        direct = audio_dir / clip_name
        if direct.exists():
            return direct
    for ext in (".flac", ".wav"):
        direct = audio_dir / f"{clip_name}{ext}"
        if direct.exists():
            return direct
    stem = clip_name.replace(".wav", "")
    matches = list(audio_dir.rglob(f"{stem}*.flac")) + list(audio_dir.rglob(f"{stem}*.wav"))
    return matches[0] if matches else None


def _find_adjacent_file_with_index(
    audio_dir: Path,
    device: str,
    ts: datetime,
    audio_index_by_second: Optional[Dict[str, Path]],
) -> Optional[Path]:
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    if audio_index_by_second is not None:
        key = f"{device}_{stamp}"
        if key in audio_index_by_second:
            return audio_index_by_second[key]
    for ext in (".flac", ".wav"):
        matches = list(audio_dir.rglob(f"{device}_{stamp}*{ext}"))
        if matches:
            return matches[0]
    return None


def power_to_db_norm(power: np.ndarray) -> np.ndarray:
    power = np.abs(power.astype(np.float32))
    max_power = float(np.max(power)) if power.size else 0.0
    if max_power > 0:
        normalized = power / max_power
        normalized = np.maximum(normalized, 1e-10)
        return 10.0 * np.log10(normalized)
    return np.full_like(power, -100.0, dtype=np.float32)


def normalize_db_to_unit(db: np.ndarray, min_db: float, max_db: float) -> np.ndarray:
    db = db.astype(np.float32)
    db = np.clip(db, min_db, max_db)
    return (db - min_db) / (max_db - min_db)


def center_crop_with_pad(spec: np.ndarray, target_f: int, target_t: int, center_t: Optional[int] = None) -> np.ndarray:
    freq_bins, time_bins = spec.shape
    if freq_bins < target_f:
        spec = np.pad(spec, ((0, target_f - freq_bins), (0, 0)), mode="edge")
        freq_bins = target_f
    elif freq_bins > target_f:
        start_f = (freq_bins - target_f) // 2
        spec = spec[start_f : start_f + target_f, :]
        freq_bins = target_f

    if center_t is None:
        center_t = time_bins // 2
    start_t = int(center_t - target_t // 2)
    end_t = start_t + target_t
    pad_left = max(0, -start_t)
    pad_right = max(0, end_t - time_bins)
    if pad_left or pad_right:
        spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode="edge")
        start_t += pad_left
        end_t += pad_left
    return spec[:, start_t:end_t]


def load_training_power(mat_path: Path) -> np.ndarray:
    data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    for key in ("P", "Sxx", "PSD", "psd", "power_spectrogram"):
        if key in data:
            spec = np.asarray(data[key])
            break
    else:
        for key in ("PdB_norm", "power_db_norm", "PdB", "P_db", "spectrogram"):
            if key in data:
                return np.asarray(data[key])
        raise KeyError(f"No spectrogram-like key found in {mat_path.name}")

    freq_key = next((key for key in ("F", "freqs", "frequencies") if key in data), None)
    time_key = next((key for key in ("T", "times", "time") if key in data), None)
    if freq_key and time_key:
        freq_len = int(np.asarray(data[freq_key]).ravel().shape[0])
        time_len = int(np.asarray(data[time_key]).ravel().shape[0])
        rows, cols = spec.shape[:2]
        if (rows, cols) == (time_len, freq_len):
            spec = spec.T
    return spec


@dataclass
class WorkItem:
    call: CallRow
    out_name: str


def _parse_mat_list(mat_list: Path) -> List[WorkItem]:
    pattern = re.compile(r"^(?P<clip>.+)_(?P<begin>-?\d+(?:\.\d+)?)s_(?P<end>-?\d+(?:\.\d+)?)s")
    rows: List[WorkItem] = []
    with open(mat_list, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name:
                continue
            if not name.lower().endswith(".mat"):
                name = name + ".mat"
            path = Path(name)
            match = pattern.match(path.stem)
            if not match:
                continue
            clip = match.group("clip")
            begin_s = float(match.group("begin"))
            end_s = float(match.group("end"))
            clip_dt = parse_clip_ts(clip)
            call_dt = clip_dt + timedelta(seconds=begin_s) if clip_dt else pd.NaT
            call = CallRow(clip, begin_s, end_s, call_dt, path)
            rows.append(WorkItem(call, path.name))
    return rows


def _parse_calls_csv(calls_csv: Path) -> List[WorkItem]:
    df = pd.read_csv(calls_csv)
    cols = {c.lower(): c for c in df.columns}
    if "clip" not in cols or "begin_s" not in cols or "end_s" not in cols:
        raise ValueError("calls-csv must include columns: clip, begin_s, end_s")
    df = df.rename(columns={cols["clip"]: "clip", cols["begin_s"]: "begin_s", cols["end_s"]: "end_s"})
    rows: List[WorkItem] = []
    for _, row in df.iterrows():
        clip = str(row["clip"])
        begin_s = float(row["begin_s"])
        end_s = float(row["end_s"])
        clip_dt = parse_clip_ts(clip)
        call_dt = clip_dt + timedelta(seconds=begin_s) if clip_dt else pd.NaT
        call = CallRow(clip, begin_s, end_s, call_dt, None)
        out_name = f"{clip}_{begin_s:.1f}s_{end_s:.1f}s_trainstyle.mat"
        rows.append(WorkItem(call, out_name))
    return rows


def _parse_clip_list(clip_list: Path, window_s: float, step_s: float) -> List[WorkItem]:
    rows: List[WorkItem] = []
    with open(clip_list, "r", encoding="utf-8") as handle:
        for line in handle:
            clip = line.strip()
            if not clip:
                continue
            clip_duration_s = 300.0
            max_start = max(0.0, clip_duration_s - window_s)
            if step_s <= 0:
                raise ValueError("--step-s must be > 0")
            starts: List[float] = []
            start = 0.0
            while start <= max_start + 1e-6:
                starts.append(float(start))
                start += step_s
            if not starts:
                starts = [0.0]
            if starts[-1] < max_start - 1e-6:
                starts.append(float(max_start))
            for start in starts:
                end = min(start + window_s, clip_duration_s)
                clip_dt = parse_clip_ts(clip)
                call_dt = clip_dt + timedelta(seconds=start) if clip_dt else pd.NaT
                call = CallRow(clip, start, end, call_dt, None)
                out_name = f"{clip}_{start:.1f}s_{end:.1f}s_window.mat"
                rows.append(WorkItem(call, out_name))
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
        prev_file = (
            _find_adjacent_file_with_index(
                audio_dir,
                clip.split("_")[0],
                clip_dt - timedelta(minutes=5),
                audio_index_by_second,
            )
            if clip_dt
            else None
        )
        if prev_file and prev_file.exists():
            prev_data, prev_fs = sf.read(str(prev_file))
            if prev_fs == fs:
                need = -start_s
                take = int(round(need * fs))
                chunks.append(prev_data[-take:])
            else:
                chunks.append(np.zeros(int(round(-start_s * fs)), dtype=data.dtype))
        else:
            chunks.append(np.zeros(int(round(-start_s * fs)), dtype=data.dtype))

    chunks.append(_slice_from_file(data, start_s, end_s))

    if end_s > total_s:
        clip_dt = parse_clip_ts(clip)
        next_file = (
            _find_adjacent_file_with_index(
                audio_dir,
                clip.split("_")[0],
                clip_dt + timedelta(minutes=5),
                audio_index_by_second,
            )
            if clip_dt
            else None
        )
        if next_file and next_file.exists():
            next_data, next_fs = sf.read(str(next_file))
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
    ax.imshow(db, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _trim_edge_context(
    times: np.ndarray,
    power: np.ndarray,
    db: np.ndarray,
    edge_context_s: float,
    segment_duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edge_context_s <= 0:
        return times, power, db

    trim_start = float(edge_context_s)
    trim_end = trim_start + float(segment_duration_s)
    mask = (times >= trim_start) & (times <= trim_end)

    if not np.any(mask):
        start_idx = int(np.searchsorted(times, trim_start, side="left"))
        end_idx = int(np.searchsorted(times, trim_end, side="right"))
        start_idx = max(0, min(start_idx, max(0, len(times) - 1)))
        end_idx = max(start_idx + 1, min(end_idx, len(times)))
        mask = np.zeros_like(times, dtype=bool)
        mask[start_idx:end_idx] = True

    return times[mask] - trim_start, power[:, mask], db[:, mask]


def main() -> None:
    ap = argparse.ArgumentParser(description="Create train-style MATs for testing")
    ap.add_argument("--mat-list", type=str, default=None, help="List of existing MAT filenames (call-centered)")
    ap.add_argument("--calls-csv", type=str, default=None, help="CSV with clip, begin_s, end_s")
    ap.add_argument("--slide", action="store_true", help="Generate sliding 40s windows per clip")
    ap.add_argument("--clip-list", type=str, default=None, help="List of clip IDs for sliding windows")
    ap.add_argument("--audio-dir", type=str, required=True)
    ap.add_argument("--dataset-doc", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--compare-dir", type=str, default=None, help="Existing training MAT dir for comparison")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, help="Optional substring filter for clip names")
    ap.add_argument("--spec-backend", type=str, default="auto", choices=["auto", "scipy", "torch"])
    ap.add_argument("--window-s", type=float, default=40.0)
    ap.add_argument("--step-s", type=float, default=40.0)
    ap.add_argument("--edge-context-s", type=float, default=2.0, help="Seconds of extra audio on both sides before spectrogram generation")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    if not args.mat_list and not args.calls_csv and not args.slide:
        raise RuntimeError("Provide --mat-list, --calls-csv, or --slide with --clip-list")

    dataset_doc = load_dataset_documentation(args.dataset_doc)
    proc = get_processing_params(dataset_doc=dataset_doc, model_path=None)
    freq_min, freq_max = proc["freq_lims"]
    win_dur = proc["win_dur"]
    overlap = proc["overlap"]
    clim_min, clim_max = proc["clim"]
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

    if not items:
        raise RuntimeError("No work items matched the requested inputs. Check --clip-list/--device filtering.")

    audio_index, audio_index_by_second = build_audio_index(audio_dir)

    rows = []
    for it in tqdm(items, desc="Generating", unit="mat"):
        call = it.call
        desired_duration = float(call.end_s - call.begin_s)
        if args.slide:
            start_s = call.begin_s - float(args.edge_context_s)
            end_s = call.end_s + float(args.edge_context_s)
            context_s = desired_duration + (2.0 * float(args.edge_context_s))
        else:
            padding = (args.window_s - (call.end_s - call.begin_s)) / 2.0
            start_s = call.begin_s - padding - float(args.edge_context_s)
            end_s = call.end_s + padding + float(args.edge_context_s)
            context_s = args.window_s + (2.0 * float(args.edge_context_s))

        audio_seg, fs = _load_context_audio(
            audio_dir, call.clip, start_s, end_s, context_s, audio_index, audio_index_by_second
        )

        freqs, times, sxx, pdb = spec_gen.compute_spectrogram(audio_seg, fs)
        freqs_c, pdb_c = crop_to_freq_lims(freqs, pdb, freq_min, freq_max)
        _, sxx_c = crop_to_freq_lims(freqs, sxx, freq_min, freq_max)
        times_c, sxx_c, pdb_c = _trim_edge_context(
            times,
            sxx_c,
            pdb_c,
            float(args.edge_context_s),
            desired_duration if args.slide else float(args.window_s),
        )

        out_path = out_dir / it.out_name
        scipy.io.savemat(
            str(out_path),
            {
                "F": freqs_c,
                "T": times_c,
                "P": sxx_c,
                "PdB_norm": pdb_c,
                "freq_min": freq_min,
                "freq_max": freq_max,
                "window_s": desired_duration if args.slide else float(args.window_s),
                "edge_context_s": float(args.edge_context_s),
                "backend": args.spec_backend,
            },
        )

        row = {
            "clip": call.clip,
            "begin_s": call.begin_s,
            "end_s": call.end_s,
            "out_mat": str(out_path),
            "backend": args.spec_backend,
        }

        if compare_dir and not args.slide:
            ref_path = compare_dir / it.out_name
            if ref_path.exists():
                ref_spec = load_training_power(ref_path)
                gen_spec = load_training_power(out_path)
                ref_is_db = np.nanmax(ref_spec) <= 10 and np.nanmin(ref_spec) < 0
                gen_is_db = np.nanmax(gen_spec) <= 10 and np.nanmin(gen_spec) < 0
                if not ref_is_db:
                    ref_spec = power_to_db_norm(ref_spec)
                if not gen_is_db:
                    gen_spec = power_to_db_norm(gen_spec)
                ref_norm = normalize_db_to_unit(ref_spec, min_db, max_db)
                gen_norm = normalize_db_to_unit(gen_spec, min_db, max_db)
                if ref_norm.shape != gen_norm.shape:
                    f = min(ref_norm.shape[0], gen_norm.shape[0])
                    t = min(ref_norm.shape[1], gen_norm.shape[1])
                    ref_norm = center_crop_with_pad(ref_norm, f, t)
                    gen_norm = center_crop_with_pad(gen_norm, f, t)
                diff = np.abs(ref_norm - gen_norm)
                row["abs_diff_mean"] = float(np.mean(diff))
                row["abs_diff_max"] = float(np.max(diff))
                row["ref_mat"] = str(ref_path)

                if args.save_images:
                    img_id = out_path.stem
                    ref_img = img_dir / f"{img_id}_ref.png"
                    gen_img = img_dir / f"{img_id}_gen.png"
                    diff_img = img_dir / f"{img_id}_diff.png"
                    _save_db_image(ref_spec, ref_img, min_db, max_db, cmap="magma")
                    _save_db_image(gen_spec, gen_img, min_db, max_db, cmap="magma")
                    vmax = max(1.0, float(np.max(np.abs(gen_spec - ref_spec))))
                    _save_db_image(gen_spec - ref_spec, diff_img, -vmax, vmax, cmap="coolwarm")
                    row["ref_img"] = str(ref_img.relative_to(out_dir))
                    row["gen_img"] = str(gen_img.relative_to(out_dir))
                    row["diff_img"] = str(diff_img.relative_to(out_dir))
            else:
                row["ref_mat"] = str(ref_path)
                row["abs_diff_mean"] = None
                row["abs_diff_max"] = None

        rows.append(row)

    if not rows:
        raise RuntimeError("Generated 0 MAT files. Aborting because this would produce an unusable bundle.")

    csv_path = out_dir / "report.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
