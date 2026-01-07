#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict
import random
import sys

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

try:
    from src.training.mat_utils import iter_mat_files, parse_mat_filename
except ModuleNotFoundError:
    from src.training.mat_utils import iter_mat_files, parse_mat_filename


def sample_files(dir_path: str, limit: int) -> List[Path]:
    out: List[Path] = []
    for p in iter_mat_files(dir_path):
        out.append(p)
        if len(out) >= limit:
            break
    return out


def print_parsed(files: List[Path], label: str) -> None:
    print(f"\n[{label}] Parsed filename fields (showing up to {len(files)}):")
    for p in files:
        src, start, dur = parse_mat_filename(p.name)
        print(f"- {p.name} -> src={src} start={start} dur={dur}")


def group_by_source(files: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in files:
        src, _, _ = parse_mat_filename(p.name)
        groups.setdefault(src, []).append(p)
    return groups


def tiny_time_separated_split(files: List[Path], train_ratio: float, val_ratio: float, seed: int, min_gap_s: float):
    rng = random.Random(seed)
    entries = []
    for p in files:
        src, start, dur = parse_mat_filename(p.name)
        entries.append({'path': p, 'src': src, 'start': start, 'dur': dur})
    rng.shuffle(entries)

    split_map = {'train': [], 'val': [], 'test': []}
    n = len(entries)
    t_train = int(n * train_ratio)
    t_val = int(n * val_ratio)

    def violates(target: str, e: dict) -> bool:
        if e['start'] is None:
            return False
        center = float(e['start']) + float(e['dur'] or 0.0) * 0.5
        for split, items in split_map.items():
            if split == target:
                continue
            for it in items:
                if it['src'] != e['src'] or it['start'] is None:
                    continue
                oc = float(it['start']) + float(it['dur'] or 0.0) * 0.5
                if abs(center - oc) < min_gap_s:
                    return True
        return False

    for e in entries:
        order = ['train', 'val', 'test']
        order.sort(key=lambda s: (len(split_map[s]) / max(1, {'train': t_train, 'val': t_val, 'test': n - t_train - t_val}[s])))
        placed = False
        for s in order:
            if not violates(s, e):
                split_map[s].append(e)
                placed = True
                break
        if not placed:
            sizes = sorted(['train', 'val', 'test'], key=lambda s: len(split_map[s]))
            split_map[sizes[0]].append(e)

    return {k: [d['path'] for d in v] for k, v in split_map.items()}


def main():
    ap = argparse.ArgumentParser(description='Quick sanity checks for FinWhale MAT filenames and splitting')
    ap.add_argument('--pos-dir', required=True)
    ap.add_argument('--neg-dir', required=True)
    ap.add_argument('--limit', type=int, default=30, help='Number of files to sample per class')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--min-gap-seconds', type=float, default=120.0)
    args = ap.parse_args()

    print(f"Sampling up to {args.limit} positives and {args.limit} negatives...")
    pos_files = sample_files(args.pos_dir, args.limit)
    neg_files = sample_files(args.neg_dir, args.limit)

    print_parsed(pos_files[:10], 'positives (first 10)')
    print_parsed(neg_files[:10], 'negatives (first 10)')

    # Group-by-source counts
    pos_groups = group_by_source(pos_files)
    neg_groups = group_by_source(neg_files)
    print(f"\nGroup-by-source summary (subset):")
    print(f"  positives: {len(pos_groups)} sources across {len(pos_files)} files")
    print(f"  negatives: {len(neg_groups)} sources across {len(neg_files)} files")

    # Tiny time-separated split on subset
    subset_all = pos_files + neg_files
    tiny_split = tiny_time_separated_split(subset_all, args.train_ratio, args.val_ratio, args.seed, args.min_gap_seconds)
    print("\nTiny time-separated split sizes (subset):")
    for k in ['train', 'val', 'test']:
        print(f"  {k}: {len(tiny_split[k])}")

    print("\nDone.")


if __name__ == '__main__':
    main()
