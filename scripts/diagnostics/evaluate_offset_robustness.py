#!/usr/bin/env python3
"""
Evaluate model robustness to call decentering by sweeping fixed crop offsets.

For each positive MAT spectrogram, this script extracts multiple time-crops where the
call center is forced to different fractional positions inside the crop (e.g. 0.05..0.95),
runs inference, and reports confidence/recall vs offset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"scipy is required: {exc}")

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.fin_models import create_model
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
    _start_from_fraction,
    parse_crop_size,
)


def _parse_crop_size_arg(raw: Optional[str]) -> Optional[Any]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) != 2:
            raise SystemExit(f"Invalid --crop-size '{raw}': expected int or freq,time")
        return [int(parts[0]), int(parts[1])]
    return int(text)


def _parse_offset_fracs(raw: Optional[str]) -> List[float]:
    if raw is None or not str(raw).strip():
        # Center + near-edge grid
        return [0.05, 0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95]
    vals: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        v = float(token)
        if not (0.0 <= v <= 1.0):
            raise SystemExit(f"Offset fraction must be in [0,1], got {v}")
        vals.append(v)
    if not vals:
        raise SystemExit("No valid --offset-fracs values parsed")
    return sorted(vals)


def _list_mat_files(folder: Path) -> List[Path]:
    out: List[Path] = []
    for entry in os.scandir(folder):
        try:
            if entry.is_file() and entry.name.lower().endswith(".mat"):
                out.append(Path(entry.path))
        except FileNotFoundError:
            continue
    out.sort()
    return out


def _load_spec(path: Path) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    data = sio.loadmat(str(path), simplify_cells=True)
    k = _find_key(data, POWER_KEYS)
    kind = "power"
    if k is None:
        k = _find_key(data, DB_KEYS) or _find_key(data, SPECTRO_KEYS)
        kind = "db"
    if k is None:
        raise KeyError(f"No spectrogram-like key in {path.name}")

    spec = np.asarray(data[k])
    if spec.ndim != 2:
        raise ValueError(f"Unexpected spectrogram ndim={spec.ndim} in {path.name}")

    fk = _find_key(data, FREQ_KEYS)
    tk = _find_key(data, TIME_KEYS)
    freqs = np.asarray(data[fk]).squeeze() if fk in data else None
    times = np.asarray(data[tk]).squeeze() if tk in data else None

    if freqs is not None and times is not None:
        f_len = int(np.asarray(freqs).ravel().shape[0])
        t_len = int(np.asarray(times).ravel().shape[0])
        r, c = spec.shape[:2]
        if (r, c) == (t_len, f_len):
            spec = spec.T

    return spec, kind, freqs, times


def _apply_physical_freq_crop(
    spec: np.ndarray,
    freqs: Optional[np.ndarray],
    crop_freq_range_hz: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if crop_freq_range_hz is None or freqs is None:
        return spec, freqs
    f_arr = np.asarray(freqs).ravel()
    if f_arr.shape[0] != spec.shape[0]:
        return spec, freqs
    fmin, fmax = crop_freq_range_hz
    mask = (f_arr >= fmin) & (f_arr <= fmax)
    if not np.any(mask):
        return spec, freqs
    idx = np.where(mask)[0]
    f0, f1 = int(idx[0]), int(idx[-1]) + 1
    return spec[f0:f1, :], f_arr[f0:f1]


def _resolve_target_dims(
    spec: np.ndarray,
    crop_size: Optional[Any],
    crop_time_seconds: Optional[float],
    times: Optional[np.ndarray],
) -> Tuple[int, int]:
    freq_crop, time_crop = parse_crop_size(crop_size)
    F, _ = spec.shape
    target_f = int(freq_crop) if freq_crop is not None else int(F)
    if crop_time_seconds is not None:
        dt = _infer_time_bin_seconds(times)
        if dt is not None and dt > 0:
            target_t = max(1, int(round(float(crop_time_seconds) / dt)))
        elif time_crop is not None:
            target_t = int(time_crop)
        else:
            target_t = target_f
    else:
        target_t = int(time_crop) if time_crop is not None else target_f
    return int(target_f), int(target_t)


def _center_crop_or_pad_freq(spec: np.ndarray, target_f: int) -> np.ndarray:
    F, _ = spec.shape
    if F < target_f:
        return np.pad(spec, ((0, target_f - F), (0, 0)), mode="edge")
    if F > target_f:
        f_start = max(0, (F - target_f) // 2)
        return spec[f_start:f_start + target_f, :]
    return spec


def _time_crop_at_fraction(spec: np.ndarray, target_t: int, frac: float) -> Tuple[np.ndarray, int]:
    _, T = spec.shape
    start = _start_from_fraction(T, target_t, frac)
    if T < target_t:
        cropped = np.pad(spec, ((0, 0), (0, target_t - T)), mode="edge")
        return cropped, 0
    return spec[:, start:start + target_t], int(start)


def _normalize_for_model(spec: np.ndarray, spec_kind: str, min_db: float, max_db: float) -> np.ndarray:
    if spec_kind == "power":
        spec = _power_to_db_norm(spec)
    return _normalize_db_to_unit(spec, min_db=min_db, max_db=max_db)


def _infer_model_name_from_sidecar(checkpoint_path: Path) -> Optional[str]:
    sidecar = checkpoint_path.parent / "args.pkl"
    if not sidecar.exists():
        return None
    try:
        with open(sidecar, "rb") as f:
            args_obj = pickle.load(f)
        if hasattr(args_obj, "model"):
            return str(getattr(args_obj, "model"))
        if isinstance(args_obj, dict) and "model" in args_obj:
            return str(args_obj["model"])
    except Exception:
        return None
    return None


def _infer_crop_from_sidecar(checkpoint_path: Path) -> Optional[Any]:
    sidecar = checkpoint_path.parent / "args.pkl"
    if not sidecar.exists():
        return None
    try:
        with open(sidecar, "rb") as f:
            args_obj = pickle.load(f)
        crop = None
        if hasattr(args_obj, "crop_size"):
            crop = getattr(args_obj, "crop_size")
        elif isinstance(args_obj, dict):
            crop = args_obj.get("crop_size")
        if crop is None:
            return None
        if isinstance(crop, str):
            return _parse_crop_size_arg(crop)
        if isinstance(crop, (list, tuple)) and len(crop) == 2:
            return [int(crop[0]), int(crop[1])]
        return int(crop)
    except Exception:
        return None


def _load_model(checkpoint_path: Path, device: torch.device, explicit_model: Optional[str]) -> Tuple[torch.nn.Module, str]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model_name = explicit_model or _infer_model_name_from_sidecar(checkpoint_path) or "SmallCNN"
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        model_name = str(ckpt["args"].get("model", model_name))
    model = create_model(model_name, num_classes=2, in_ch=1).to(device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, model_name


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_positive_split_names(split_file: Path) -> Optional[set]:
    if not split_file.exists():
        raise SystemExit(f"split file does not exist: {split_file}")
    names = set()
    with open(split_file, "r") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if len(parts) < 2:
                continue
            path_raw, label_raw = parts[0], parts[1]
            try:
                label = int(label_raw)
            except Exception:
                continue
            if label != 1:
                continue
            names.add(Path(path_raw).name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic offset robustness evaluation on positive MATs")
    ap.add_argument("--pos-dir", type=str, required=True, help="Directory with positive MAT files")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--split-file", type=str, default=None,
                    help="Optional split txt (train/val/test.txt). If set, evaluate only positive files listed there.")
    ap.add_argument("--model", type=str, default=None, help="Force model architecture (default: infer)")
    ap.add_argument("--crop-size", type=str, default=None, help='Crop bins: int or "freq,time" (default: infer from checkpoint)')
    ap.add_argument("--crop-time-seconds", type=float, default=None, help="Physical crop duration in seconds")
    ap.add_argument("--crop-freq-range-hz", type=float, nargs=2, default=None, metavar=("MIN_HZ", "MAX_HZ"),
                    help="Physical crop frequency range")
    ap.add_argument("--min-db", type=float, default=-80.0)
    ap.add_argument("--max-db", type=float, default=0.0)
    ap.add_argument("--offset-fracs", type=str, default=None,
                    help="Comma list in [0,1], e.g. 0.05,0.15,...,0.95")
    ap.add_argument("--threshold", type=float, default=0.5, help="Positive threshold for recall summaries")
    ap.add_argument("--threshold-high", type=float, default=0.7, help="Second threshold for stricter recall")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on number of positive MAT files (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_dir = Path(args.pos_dir)
    if not pos_dir.exists():
        raise SystemExit(f"pos-dir does not exist: {pos_dir}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("Warning: CUDA requested but unavailable; using CPU.")

    crop_size = _parse_crop_size_arg(args.crop_size)
    if crop_size is None:
        crop_size = _infer_crop_from_sidecar(checkpoint)
    if crop_size is None and args.crop_time_seconds is None:
        raise SystemExit("Unable to resolve crop dimensions. Provide --crop-size or --crop-time-seconds.")

    crop_freq_range_hz = tuple(args.crop_freq_range_hz) if args.crop_freq_range_hz is not None else None
    offsets = _parse_offset_fracs(args.offset_fracs)

    model, model_name = _load_model(checkpoint, device=device, explicit_model=args.model)
    print(f"Model: {model_name} | Device: {device}")
    print(f"Crop config: crop_size={crop_size}, crop_time_seconds={args.crop_time_seconds}, crop_freq_range_hz={crop_freq_range_hz}")
    print(f"Offsets: {offsets}")

    mats = _list_mat_files(pos_dir)
    if not mats:
        raise SystemExit(f"No .mat files in {pos_dir}")

    if args.split_file:
        split_names = _load_positive_split_names(Path(args.split_file))
        mats = [p for p in mats if p.name in split_names]
        if not mats:
            raise SystemExit(f"No positive MATs from split file matched in {pos_dir}: {args.split_file}")

    rng = np.random.default_rng(args.seed)
    if args.max_samples and args.max_samples > 0 and len(mats) > args.max_samples:
        idx = rng.choice(len(mats), size=int(args.max_samples), replace=False)
        idx.sort()
        mats = [mats[int(i)] for i in idx]

    rows: List[Dict[str, Any]] = []
    total = len(mats)
    for i, mat_path in enumerate(mats, start=1):
        if i % 100 == 0 or i == 1 or i == total:
            print(f"Processing {i}/{total}: {mat_path.name}")

        try:
            spec, kind, freqs, times = _load_spec(mat_path)
            spec, freqs = _apply_physical_freq_crop(spec, freqs, crop_freq_range_hz)
            target_f, target_t = _resolve_target_dims(spec, crop_size=crop_size, crop_time_seconds=args.crop_time_seconds, times=times)
            spec = _center_crop_or_pad_freq(spec, target_f=target_f)
        except Exception as exc:
            for frac in offsets:
                rows.append({
                    "file": mat_path.name,
                    "offset_frac": float(frac),
                    "confidence": np.nan,
                    "pred_pos": np.nan,
                    "crop_start": np.nan,
                    "status": f"load_error:{exc}",
                })
            continue

        crops = []
        starts: List[int] = []
        for frac in offsets:
            c, st = _time_crop_at_fraction(spec, target_t=target_t, frac=float(frac))
            c = _normalize_for_model(c, spec_kind=kind, min_db=float(args.min_db), max_db=float(args.max_db))
            crops.append(torch.from_numpy(c).unsqueeze(0).float())
            starts.append(int(st))

        xb = torch.stack(crops, dim=0).to(device, non_blocking=True)  # [N,1,F,T]
        with torch.no_grad():
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= float(args.threshold)).astype(np.int32)

        for frac, conf, pred, st in zip(offsets, probs, preds, starts):
            rows.append({
                "file": mat_path.name,
                "offset_frac": float(frac),
                "confidence": float(conf),
                "pred_pos": int(pred),
                "crop_start": int(st),
                "status": "ok",
            })

    # Save per-sample rows
    detail_rows = [r for r in rows if r.get("status") == "ok"]
    _write_csv(out_dir / "offset_detail.csv", detail_rows)

    # Aggregate by offset
    offset_summary: List[Dict[str, Any]] = []
    for frac in offsets:
        vals = [r for r in detail_rows if abs(float(r["offset_frac"]) - float(frac)) < 1e-9]
        confs = np.array([float(r["confidence"]) for r in vals], dtype=float) if vals else np.array([], dtype=float)
        preds = np.array([int(r["pred_pos"]) for r in vals], dtype=int) if vals else np.array([], dtype=int)
        rec = float(np.mean(preds >= 1)) if preds.size else float("nan")
        rec_hi = float(np.mean(confs >= float(args.threshold_high))) if confs.size else float("nan")
        offset_summary.append({
            "offset_frac": float(frac),
            "n": int(len(vals)),
            "mean_conf": float(np.mean(confs)) if confs.size else np.nan,
            "median_conf": float(np.median(confs)) if confs.size else np.nan,
            "p10_conf": float(np.percentile(confs, 10)) if confs.size else np.nan,
            "p90_conf": float(np.percentile(confs, 90)) if confs.size else np.nan,
            f"recall_at_{args.threshold:.2f}": rec,
            f"recall_at_{args.threshold_high:.2f}": rec_hi,
        })

    _write_csv(out_dir / "offset_summary.csv", offset_summary)

    # Plot confidence + recall vs offset
    xs = np.array([r["offset_frac"] for r in offset_summary], dtype=float)
    mean_conf = np.array([r["mean_conf"] for r in offset_summary], dtype=float)
    p10 = np.array([r["p10_conf"] for r in offset_summary], dtype=float)
    p90 = np.array([r["p90_conf"] for r in offset_summary], dtype=float)
    rec_a = np.array([r[f"recall_at_{args.threshold:.2f}"] for r in offset_summary], dtype=float)
    rec_b = np.array([r[f"recall_at_{args.threshold_high:.2f}"] for r in offset_summary], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, mean_conf, marker="o", label="mean confidence")
    ax.fill_between(xs, p10, p90, alpha=0.2, label="p10-p90 confidence")
    ax.set_xlabel("Call position inside crop (fraction)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_vs_offset.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, rec_a, marker="o", label=f"recall @ {args.threshold:.2f}")
    ax.plot(xs, rec_b, marker="o", label=f"recall @ {args.threshold_high:.2f}")
    ax.set_xlabel("Call position inside crop (fraction)")
    ax.set_ylabel("Recall on positives")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "recall_vs_offset.png", dpi=150)
    plt.close(fig)

    # Edge-vs-center summary
    center = [r for r in offset_summary if 0.45 <= float(r["offset_frac"]) <= 0.55]
    edge = [r for r in offset_summary if float(r["offset_frac"]) <= 0.15 or float(r["offset_frac"]) >= 0.85]

    def _safe_mean(vals: Sequence[float]) -> float:
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        return float(np.mean(arr)) if arr.size else float("nan")

    center_conf = _safe_mean([float(r["mean_conf"]) for r in center])
    edge_conf = _safe_mean([float(r["mean_conf"]) for r in edge])
    center_rec = _safe_mean([float(r[f"recall_at_{args.threshold:.2f}"]) for r in center])
    edge_rec = _safe_mean([float(r[f"recall_at_{args.threshold:.2f}"]) for r in edge])

    summary = {
        "checkpoint": str(checkpoint),
        "model": model_name,
        "device": str(device),
        "n_positive_files": len({r["file"] for r in detail_rows}),
        "n_offset_samples": len(detail_rows),
        "offsets": offsets,
        "threshold": float(args.threshold),
        "threshold_high": float(args.threshold_high),
        "center_mean_conf": center_conf,
        "edge_mean_conf": edge_conf,
        "edge_minus_center_conf": edge_conf - center_conf if np.isfinite(center_conf) and np.isfinite(edge_conf) else np.nan,
        "center_recall": center_rec,
        "edge_recall": edge_rec,
        "edge_minus_center_recall": edge_rec - center_rec if np.isfinite(center_rec) and np.isfinite(edge_rec) else np.nan,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# Offset Robustness Summary",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- model: `{model_name}`",
        f"- n positive files: `{summary['n_positive_files']}`",
        f"- offsets evaluated: `{len(offsets)}`",
        f"- threshold: `{args.threshold:.2f}`",
        f"- center mean confidence: `{center_conf:.4f}`",
        f"- edge mean confidence: `{edge_conf:.4f}`",
        f"- edge-center confidence delta: `{summary['edge_minus_center_conf']:.4f}`",
        f"- center recall @ {args.threshold:.2f}: `{center_rec:.4f}`",
        f"- edge recall @ {args.threshold:.2f}: `{edge_rec:.4f}`",
        f"- edge-center recall delta: `{summary['edge_minus_center_recall']:.4f}`",
        "",
        "## Artifacts",
        "",
        "- `offset_detail.csv`",
        "- `offset_summary.csv`",
        "- `confidence_vs_offset.png`",
        "- `recall_vs_offset.png`",
        "- `summary.json`",
    ]
    (out_dir / "offset_robustness.md").write_text("\n".join(md_lines) + "\n")

    print(f"Wrote: {out_dir / 'offset_robustness.md'}")
    print(f"Wrote: {out_dir / 'offset_summary.csv'}")
    print(f"Wrote: {out_dir / 'confidence_vs_offset.png'}")
    print(f"Wrote: {out_dir / 'recall_vs_offset.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
