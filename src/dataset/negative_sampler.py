#!/usr/bin/env python3
import random
from typing import List, Tuple, Dict
from src.dataset.reporting import print_status

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

def sample_negative_windows_for_file(
    clip_id: str,
    duration: float,
    context_duration: float,
    calls_by_file: Dict[str, List[Tuple[float, float]]],
    max_windows: int,
    margin: float = 2.0
) -> List[Tuple[float, float]]:
    """Sample up to max_windows negative [start, end] pairs that avoid any annotated calls."""
    occupied = calls_by_file.get(clip_id, [])
    free_intervals = compute_free_intervals(occupied, duration, margin=margin)
    
    # Simple sampling strategy:
    # 1. Filter intervals shorter than context
    candidates = [i for i in free_intervals if (i[1] - i[0]) >= context_duration]
    
    if not candidates:
        return []
        
    negative_windows = []
    
    # Randomly sample windows within candidates
    # We try to distribute them across available free space
    attempts = 0
    max_attempts = max_windows * 5
    
    while len(negative_windows) < max_windows and attempts < max_attempts:
        attempts += 1
        # Pick a random candidate weighted by length
        lengths = [i[1] - i[0] for i in candidates]
        interval = random.choices(candidates, weights=lengths)[0]
        
        # Pick a random start time within that interval
        max_start = interval[1] - context_duration
        win_start = random.uniform(interval[0], max_start)
        win_end = win_start + context_duration
        
        # Check for overlap with already picked negatives to avoid redundancy
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
