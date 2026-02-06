import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.training.mat_utils import iter_mat_files, parse_mat_filename


def build_entries(pos_dir: str, neg_dir: str) -> List[dict]:
    entries: List[dict] = []
    for p in iter_mat_files(pos_dir):
        src, start, dur = parse_mat_filename(Path(p).name)
        entries.append({'path': Path(p), 'src': src, 'start': start, 'dur': dur, 'label': 1})
    for p in iter_mat_files(neg_dir):
        src, start, dur = parse_mat_filename(Path(p).name)
        entries.append({'path': Path(p), 'src': src, 'start': start, 'dur': dur, 'label': 0})
    return entries


def split_group_by_source(entries: List[dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for e in entries:
        groups.setdefault(e['src'], []).append(e)
    src_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(src_ids)
    n = len(src_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    split_srcs = {
        'train': src_ids[:n_train],
        'val': src_ids[n_train:n_train + n_val],
        'test': src_ids[n_train + n_val:]
    }
    return {k: sum([groups[sid] for sid in v], []) for k, v in split_srcs.items()}


def split_time_separated(entries: List[dict], train_ratio: float, val_ratio: float, seed: int, min_gap_seconds: float) -> Dict[str, List[dict]]:
    rng = random.Random(seed)

    split_names = ['train', 'val', 'test']
    split_map: Dict[str, List[dict]] = {k: [] for k in split_names}

    # Separate entries by label/time metadata.
    pos_timed: List[dict] = []
    pos_untimed: List[dict] = []
    neg_entries: List[dict] = []
    for e in entries:
        if int(e.get('label', 0)) == 1:
            if e.get('start') is None:
                pos_untimed.append(e)
            else:
                pos_timed.append(e)
        else:
            neg_entries.append(e)

    # Targets are stratified by class to avoid extreme skew in val/test.
    n_pos = len(pos_timed) + len(pos_untimed)
    n_neg = len(neg_entries)
    pos_targets = {
        'train': int(n_pos * train_ratio),
        'val': int(n_pos * val_ratio),
        'test': max(0, n_pos - int(n_pos * train_ratio) - int(n_pos * val_ratio)),
    }
    neg_targets = {
        'train': int(n_neg * train_ratio),
        'val': int(n_neg * val_ratio),
        'test': max(0, n_neg - int(n_neg * train_ratio) - int(n_neg * val_ratio)),
    }
    pos_counts = {k: 0 for k in split_names}
    neg_counts = {k: 0 for k in split_names}

    def _center(e: dict) -> float:
        return float(e['start']) + 0.5 * float(e.get('dur') or 0.0)

    def _choose_split(counts: Dict[str, int], targets: Dict[str, int]) -> str:
        # Prefer the split with the largest remaining deficit.
        deficits = [(int(targets[s]) - int(counts[s]), -int(counts[s]), rng.random(), s) for s in split_names]
        deficits.sort(reverse=True)
        return deficits[0][3]

    # Cluster positives per source so centers within min_gap_seconds stay together.
    by_src: Dict[str, List[dict]] = defaultdict(list)
    for e in pos_timed:
        by_src[str(e['src'])].append(e)

    pos_clusters: List[List[dict]] = []
    for src, src_entries in by_src.items():
        src_entries = sorted(src_entries, key=_center)
        if not src_entries:
            continue
        cluster: List[dict] = [src_entries[0]]
        prev_c = _center(src_entries[0])
        for e in src_entries[1:]:
            c = _center(e)
            if abs(c - prev_c) < float(min_gap_seconds):
                cluster.append(e)
            else:
                pos_clusters.append(cluster)
                cluster = [e]
            prev_c = c
        if cluster:
            pos_clusters.append(cluster)

    rng.shuffle(pos_clusters)
    for cluster in pos_clusters:
        split = _choose_split(pos_counts, pos_targets)
        split_map[split].extend(cluster)
        pos_counts[split] += len(cluster)

    # Untimed positives: keep same-source entries together and assign by deficit.
    pos_untimed_by_src: Dict[str, List[dict]] = defaultdict(list)
    for e in pos_untimed:
        pos_untimed_by_src[str(e['src'])].append(e)
    untimed_groups = list(pos_untimed_by_src.values())
    rng.shuffle(untimed_groups)
    for group in untimed_groups:
        split = _choose_split(pos_counts, pos_targets)
        split_map[split].extend(group)
        pos_counts[split] += len(group)

    # Negatives: assign grouped by source to reduce cross-split background leakage.
    neg_by_src: Dict[str, List[dict]] = defaultdict(list)
    for e in neg_entries:
        neg_by_src[str(e['src'])].append(e)
    neg_groups = list(neg_by_src.values())
    rng.shuffle(neg_groups)
    for group in neg_groups:
        split = _choose_split(neg_counts, neg_targets)
        split_map[split].extend(group)
        neg_counts[split] += len(group)

    return split_map


def summarise_counts(split: Dict[str, List[dict]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k, lst in split.items():
        pos = sum(1 for e in lst if e['label'] == 1)
        neg = sum(1 for e in lst if e['label'] == 0)
        out[k] = {'pos': pos, 'neg': neg, 'total': len(lst)}
    return out
