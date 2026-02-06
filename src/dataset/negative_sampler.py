#!/usr/bin/env python3
import random
from typing import List, Tuple, Dict, Optional

def compute_free_intervals(
    occupied_intervals: List[Tuple[float, float]], 
    file_duration: float, 
    margin: float = 0.0
) -> List[Tuple[float, float]]:
    """Compute free intervals in [0, file_duration] given occupied intervals, with optional margins."""
    if not occupied_intervals:
        return [(0.0, file_duration)]
    
    # Sort and add margins
    sorted_occ = sorted(occupied_intervals)
    margined = []
    for start, end in sorted_occ:
        margined.append((max(0, start - margin), min(file_duration, end + margin)))
    
    # Merge overlapping margined intervals
    if not margined: return [(0.0, file_duration)]
    merged = []
    curr_start, curr_end = margined[0]
    for next_start, next_end in margined[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    
    # Compute gaps
    free = []
    last_end = 0.0
    for start, end in merged:
        if start > last_end:
            free.append((last_end, start))
        last_end = end
    if last_end < file_duration:
        free.append((last_end, file_duration))
        
    return free


def enumerate_negative_windows_for_file(
    clip_id: str,
    duration: float,
    context_duration: float,
    calls_by_file: Dict[str, List[Tuple[float, float]]],
    margin: float = 2.0,
    step_seconds: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Enumerate candidate negative windows from free intervals.

    By default, uses non-overlapping windows (step = context_duration). Set
    step_seconds smaller for denser candidates.
    """
    occupied = calls_by_file.get(clip_id, [])
    free_intervals = compute_free_intervals(occupied, duration, margin=margin)
    candidates = [i for i in free_intervals if (i[1] - i[0]) >= context_duration]
    if not candidates:
        return []

    step = float(step_seconds) if step_seconds is not None else float(context_duration)
    if step <= 0:
        step = float(context_duration)

    windows: List[Tuple[float, float]] = []
    for start_i, end_i in candidates:
        max_start = end_i - context_duration
        if max_start < start_i:
            continue
        pos = float(start_i)
        while pos <= max_start + 1e-9:
            windows.append((pos, pos + context_duration))
            pos += step
        # Always include the rightmost valid window to cover interval edge.
        if windows and windows[-1][0] < max_start - 1e-9:
            windows.append((max_start, max_start + context_duration))
    return windows

def sample_negative_windows_for_file(
    clip_id: str,
    duration: float,
    context_duration: float,
    calls_by_file: Dict[str, List[Tuple[float, float]]],
    max_windows: int,
    margin: float = 2.0,
    strategy: str = "random",
    step_seconds: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Sample up to max_windows negative [start, end] pairs that avoid calls.

    strategy:
        - random: historical random sampling behavior
        - tiled: enumerate windows on free intervals then downsample if needed
    """
    if max_windows <= 0:
        return []

    strategy = str(strategy).lower().strip()
    if strategy == "tiled":
        all_windows = enumerate_negative_windows_for_file(
            clip_id=clip_id,
            duration=duration,
            context_duration=context_duration,
            calls_by_file=calls_by_file,
            margin=margin,
            step_seconds=step_seconds,
        )
        if len(all_windows) <= max_windows:
            return sorted(all_windows)
        # Evenly subsample candidates to keep broad temporal coverage.
        if max_windows == 1:
            return [all_windows[len(all_windows) // 2]]
        step = (len(all_windows) - 1) / float(max_windows - 1)
        idx = sorted(set(int(round(i * step)) for i in range(max_windows)))
        return sorted(all_windows[i] for i in idx)

    occupied = calls_by_file.get(clip_id, [])
    free_intervals = compute_free_intervals(occupied, duration, margin=margin)
    candidates = [i for i in free_intervals if (i[1] - i[0]) >= context_duration]

    if not candidates:
        return []

    negative_windows = []
    attempts = 0
    max_attempts = max_windows * 5

    while len(negative_windows) < max_windows and attempts < max_attempts:
        attempts += 1
        lengths = [i[1] - i[0] for i in candidates]
        interval = random.choices(candidates, weights=lengths)[0]

        max_start = interval[1] - context_duration
        win_start = random.uniform(interval[0], max_start)
        win_end = win_start + context_duration

        overlap_threshold = 0.2 * context_duration
        too_much_overlap = False
        for s, e in negative_windows:
            overlap = min(win_end, e) - max(win_start, s)
            if overlap > overlap_threshold:
                too_much_overlap = True
                break

        if not too_much_overlap:
            negative_windows.append((win_start, win_end))

    return sorted(negative_windows)
