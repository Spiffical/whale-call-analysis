#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random
import json
import sys

# Ensure repo root on sys.path so `scripts` imports work when running as a file
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from src.training.mat_utils import iter_mat_files, parse_mat_filename
except ModuleNotFoundError:
    from src.training.mat_utils import iter_mat_files, parse_mat_filename


def build_combined_entries(pos_dir: str, neg_dir: str) -> List[dict]:
    """Stream directories and build a combined list of entries with labels.
    label: 1 for positive, 0 for negative
    """
    entries: List[dict] = []
    for p in tqdm(iter_mat_files(pos_dir), desc='Indexing positives'):
        src, start, dur = parse_mat_filename(p.name)
        entries.append({'path': p, 'src': src, 'start': start, 'dur': dur, 'label': 1})
    for p in tqdm(iter_mat_files(neg_dir), desc='Indexing negatives'):
        src, start, dur = parse_mat_filename(p.name)
        entries.append({'path': p, 'src': src, 'start': start, 'dur': dur, 'label': 0})
    return entries


def group_by_source_combined(entries: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for e in entries:
        groups.setdefault(e['src'], []).append(e)
    return groups


def stratified_group_split_combined(groups: Dict[str, List[dict]],
                                    train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    src_ids = list(groups.keys())
    rng.shuffle(src_ids)
    n = len(src_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    split_srcs = {
        'train': src_ids[:n_train],
        'val': src_ids[n_train:n_train + n_val],
        'test': src_ids[n_train + n_val:]
    }
    split = {k: sum([groups[sid] for sid in v], []) for k, v in split_srcs.items()}
    return split


def time_separated_split_combined(entries: List[dict],
                                  train_ratio: float, val_ratio: float, seed: int,
                                  min_gap_seconds: float) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    entries = list(entries)  # shallow copy
    rng.shuffle(entries)

    split_map: Dict[str, List[dict]] = {'train': [], 'val': [], 'test': []}
    n = len(entries)
    t_train = int(n * train_ratio)
    t_val = int(n * val_ratio)

    def violates(target: str, e: dict) -> bool:
        if e['start'] is None:
            return False
        center = float(e['start']) + float(e['dur'] or 0.0) * 0.5
        src = e['src']
        for split, items in split_map.items():
            if split == target:
                continue
            for it in items:
                if it['src'] != src or it['start'] is None:
                    continue
                oc = float(it['start']) + float(it['dur'] or 0.0) * 0.5
                if abs(center - oc) < min_gap_seconds:
                    return True
        return False

    for e in tqdm(entries, desc='Allocating with min-gap (combined)'):
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
    return split_map


def summarise_counts_by_label(split: Dict[str, List[dict]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k, lst in split.items():
        pos = sum(1 for e in lst if e['label'] == 1)
        neg = sum(1 for e in lst if e['label'] == 0)
        out[k] = {'pos': pos, 'neg': neg, 'total': len(lst)}
    return out


def leak_check_by_source(s1: List[dict], s2: List[dict]) -> int:
    src1 = set(e['src'] for e in s1)
    src2 = set(e['src'] for e in s2)
    return len(src1 & src2)


def main():
    ap = argparse.ArgumentParser(description='Analyze FinWhale split strategies (leakage-safe)')
    ap.add_argument('--pos-dir', required=True)
    ap.add_argument('--neg-dir', required=True)
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-gap-seconds', type=float, default=60.0)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = build_combined_entries(args.pos_dir, args.neg_dir)
    groups = group_by_source_combined(entries)

    # Strategy A: group-by-source combined
    splitA = stratified_group_split_combined(groups, args.train_ratio, args.val_ratio, args.seed)

    # Strategy B: time-separated combined
    splitB = time_separated_split_combined(entries, args.train_ratio, args.val_ratio, args.seed, args.min_gap_seconds)

    def counts_summary(split_dict: Dict[str, List[dict]]):
        return {
            'pos_counts': {k: v['pos'] for k, v in summarise_counts_by_label(split_dict).items()},
            'neg_counts': {k: v['neg'] for k, v in summarise_counts_by_label(split_dict).items()},
            'all_counts': {k: v['total'] for k, v in summarise_counts_by_label(split_dict).items()},
            'leak_train_val_src_overlap': leak_check_by_source(split_dict['train'], split_dict['val']),
            'leak_train_test_src_overlap': leak_check_by_source(split_dict['train'], split_dict['test']),
            'leak_val_test_src_overlap': leak_check_by_source(split_dict['val'], split_dict['test']),
        }

    summary = {
        'A_group_by_source_combined': counts_summary(splitA),
        'B_time_separated_combined': counts_summary(splitB)
    }

    with open(out_dir / 'split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save split lists with labels
    def save_lists(tag: str, split_dict: Dict[str, List[dict]]):
        base = out_dir / f'splits_{tag}'
        base.mkdir(exist_ok=True)
        for k in ['train', 'val', 'test']:
            with open(base / f'{k}.txt', 'w') as f:
                for e in split_dict[k]:
                    f.write(f"{e['path']}\t{e['label']}\n")
    save_lists('group_by_source', splitA)
    save_lists('time_separated', splitB)

    # Plot counts
    def plot_counts(tag: str, cnts: Dict[str, int]):
        ks = ['train', 'val', 'test']
        vs = [cnts[k] for k in ks]
        plt.figure()
        plt.bar(ks, vs)
        plt.title(f'{tag} counts')
        plt.tight_layout()
        plt.savefig(out_dir / f'{tag}_counts.png', dpi=150)
        plt.close()

    plot_counts('A_group_by_source_all', summary['A_group_by_source_combined']['all_counts'])
    plot_counts('B_time_separated_all', summary['B_time_separated_combined']['all_counts'])

    print('Wrote:', out_dir / 'split_summary.json')


if __name__ == '__main__':
    main()
