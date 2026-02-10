#!/usr/bin/env python3
"""
Post-process inference predictions with temporal clustering + hysteresis filtering.

This script is intended for sliding-window inference outputs (UnifiedPredictionTracker v2 JSON).
It reduces isolated false positives by keeping only event-like clusters:
1) candidate windows must exceed a low threshold
2) each kept cluster must contain at least one high-threshold window
3) each kept cluster must contain at least N windows
4) optional minimum cluster duration
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _basename(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    return Path(path_value).name


def _base_id_from_item_id(item_id: Optional[str]) -> Optional[str]:
    if not item_id:
        return None
    text = str(item_id)
    if "_win" in text:
        return text.rsplit("_win", 1)[0]
    return text


def _extract_score(item: Dict[str, Any], class_hierarchy: Optional[str]) -> Optional[float]:
    outputs = item.get("model_outputs")
    if not isinstance(outputs, list):
        return None
    for output in outputs:
        if not isinstance(output, dict):
            continue
        if class_hierarchy is not None and output.get("class_hierarchy") != class_hierarchy:
            continue
        score = _safe_float(output.get("score"))
        if score is not None:
            return score
    if class_hierarchy is None:
        for output in outputs:
            if isinstance(output, dict):
                score = _safe_float(output.get("score"))
                if score is not None:
                    return score
    return None


def _extract_time_bounds(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    start = (
        _safe_float(item.get("window_time_start"))
        if item.get("window_time_start") is not None
        else _safe_float(item.get("segment_start_sec"))
    )
    end = (
        _safe_float(item.get("window_time_end"))
        if item.get("window_time_end") is not None
        else _safe_float(item.get("segment_end_sec"))
    )
    duration = _safe_float(item.get("duration_sec"))

    if start is not None and end is None and duration is not None:
        end = start + max(duration, 0.0)
    if start is None and end is not None and duration is not None:
        start = end - max(duration, 0.0)
    return start, end


def _group_key(item: Dict[str, Any]) -> str:
    source_audio = item.get("source_audio")
    if isinstance(source_audio, str) and source_audio:
        return source_audio
    audio_path_name = _basename(item.get("audio_path"))
    if audio_path_name:
        return audio_path_name
    base = _base_id_from_item_id(item.get("item_id"))
    if base:
        return base
    return "unknown_group"


@dataclass
class Candidate:
    idx: int
    group: str
    score: float
    start_sec: Optional[float]
    end_sec: Optional[float]


@dataclass
class Event:
    event_id: str
    group: str
    member_indices: List[int]
    start_sec: Optional[float]
    end_sec: Optional[float]
    duration_sec: Optional[float]
    max_score: float
    mean_score: float
    n_members: int
    n_high: int
    inferred_gap_sec: Optional[float]


def _sort_key(c: Candidate) -> Tuple[int, float]:
    # Unknown times get pushed to the end while preserving deterministic order by idx.
    if c.start_sec is None:
        return (1, float(c.idx))
    return (0, float(c.start_sec))


def _infer_gap_seconds(cands: Sequence[Candidate]) -> Optional[float]:
    starts = [c.start_sec for c in cands if c.start_sec is not None]
    if len(starts) < 2:
        return None
    starts = sorted(starts)
    diffs: List[float] = []
    for i in range(1, len(starts)):
        d = starts[i] - starts[i - 1]
        if d > 0 and math.isfinite(d):
            diffs.append(d)
    if not diffs:
        return None
    return float(median(diffs))


def _cluster_candidates(
    candidates: Sequence[Candidate],
    low_threshold: float,
    high_threshold: float,
    min_members: int,
    min_duration_sec: float,
    max_gap_seconds: Optional[float],
) -> Tuple[List[Event], Dict[str, Any]]:
    by_group: Dict[str, List[Candidate]] = {}
    for c in candidates:
        if c.score >= low_threshold:
            by_group.setdefault(c.group, []).append(c)

    events: List[Event] = []
    event_idx = 0
    debug: Dict[str, Any] = {"groups": {}}

    for group, members in by_group.items():
        members_sorted = sorted(members, key=_sort_key)
        inferred_step = _infer_gap_seconds(members_sorted)
        effective_gap = max_gap_seconds
        if effective_gap is None and inferred_step is not None:
            effective_gap = max(inferred_step * 1.5, inferred_step + 1e-6)
        if effective_gap is None:
            effective_gap = 0.0

        clusters: List[List[Candidate]] = []
        current: List[Candidate] = []
        prev = None
        for c in members_sorted:
            if not current:
                current = [c]
                prev = c
                continue
            contiguous = False
            if prev is not None and prev.end_sec is not None and c.start_sec is not None:
                contiguous = (c.start_sec - prev.end_sec) <= float(effective_gap)
            elif prev is not None and prev.start_sec is not None and c.start_sec is not None:
                contiguous = (c.start_sec - prev.start_sec) <= float(effective_gap)
            else:
                contiguous = (c.idx == prev.idx + 1) if prev is not None else False

            if contiguous:
                current.append(c)
            else:
                clusters.append(current)
                current = [c]
            prev = c
        if current:
            clusters.append(current)

        kept = 0
        dropped = 0
        debug_clusters: List[Dict[str, Any]] = []
        for cluster in clusters:
            scores = [c.score for c in cluster]
            n_high = sum(1 for s in scores if s >= high_threshold)
            n_members = len(cluster)
            starts = [c.start_sec for c in cluster if c.start_sec is not None]
            ends = [c.end_sec for c in cluster if c.end_sec is not None]
            start_sec = min(starts) if starts else None
            end_sec = max(ends) if ends else None
            duration = (end_sec - start_sec) if (start_sec is not None and end_sec is not None) else None
            passes = (
                n_high >= 1
                and n_members >= min_members
                and ((duration is None) or (duration >= min_duration_sec))
            )
            if passes:
                event_idx += 1
                events.append(
                    Event(
                        event_id=f"evt_{event_idx:06d}",
                        group=group,
                        member_indices=[c.idx for c in cluster],
                        start_sec=start_sec,
                        end_sec=end_sec,
                        duration_sec=duration,
                        max_score=max(scores),
                        mean_score=float(sum(scores) / len(scores)),
                        n_members=n_members,
                        n_high=n_high,
                        inferred_gap_sec=float(effective_gap),
                    )
                )
                kept += 1
            else:
                dropped += 1
            debug_clusters.append(
                {
                    "n_members": n_members,
                    "n_high": n_high,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration,
                    "max_score": max(scores) if scores else None,
                    "kept": bool(passes),
                }
            )

        debug["groups"][group] = {
            "n_candidates": len(members_sorted),
            "n_clusters": len(clusters),
            "n_kept_clusters": kept,
            "n_dropped_clusters": dropped,
            "inferred_step_sec": inferred_step,
            "effective_gap_sec": effective_gap,
            "clusters": debug_clusters,
        }

    return events, debug


def _write_events_csv(
    path: Path,
    events: Sequence[Event],
    *,
    idx_to_item_id: Optional[Dict[int, str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "group",
                "start_sec",
                "end_sec",
                "duration_sec",
                "n_members",
                "n_high",
                "max_score",
                "mean_score",
                "member_item_ids",
            ],
        )
        w.writeheader()
        for e in events:
            w.writerow(
                {
                    "event_id": e.event_id,
                    "group": e.group,
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "duration_sec": e.duration_sec,
                    "n_members": e.n_members,
                    "n_high": e.n_high,
                    "max_score": e.max_score,
                    "mean_score": e.mean_score,
                    "member_item_ids": ",".join(
                        idx_to_item_id.get(i, str(i)) if idx_to_item_id is not None else str(i)
                        for i in e.member_indices
                    ),
                }
            )


def _write_summary_md(
    path: Path,
    *,
    input_json: Path,
    output_json: Path,
    low_threshold: float,
    high_threshold: float,
    min_members: int,
    min_duration_sec: float,
    max_gap_seconds: Optional[float],
    total_items: int,
    candidate_items: int,
    kept_items: int,
    n_events: int,
    by_group_counts: Dict[str, int],
) -> None:
    lines: List[str] = [
        "# Prediction Post-processing Summary",
        "",
        f"- input: `{input_json}`",
        f"- output: `{output_json}`",
        f"- generated_at: `{_iso_now()}`",
        f"- low_threshold: `{low_threshold:.4f}`",
        f"- high_threshold: `{high_threshold:.4f}`",
        f"- min_members: `{min_members}`",
        f"- min_duration_sec: `{min_duration_sec:.2f}`",
        f"- max_gap_seconds: `{max_gap_seconds if max_gap_seconds is not None else 'auto'}`",
        f"- total_input_items: `{total_items}`",
        f"- candidate_items(score>=low): `{candidate_items}`",
        f"- kept_items: `{kept_items}`",
        f"- kept_events: `{n_events}`",
        "",
        "## Kept Events Per Group",
        "",
    ]
    if by_group_counts:
        for group, count in sorted(by_group_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{group}`: `{count}`")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Post-process sliding-window predictions via temporal clustering + hysteresis."
    )
    ap.add_argument("--input-json", type=str, required=True, help="Input predictions JSON from run_inference.py")
    ap.add_argument("--output-json", type=str, required=True, help="Filtered predictions JSON output")
    ap.add_argument(
        "--class-hierarchy",
        type=str,
        default=None,
        help="Class hierarchy to score/filter. Default: first model output per item.",
    )
    ap.add_argument(
        "--low-threshold",
        type=float,
        default=0.70,
        help="Low threshold for candidate windows (cluster membership).",
    )
    ap.add_argument(
        "--high-threshold",
        type=float,
        default=0.82,
        help="High threshold required at least once in each kept cluster.",
    )
    ap.add_argument(
        "--min-members",
        type=int,
        default=2,
        help="Minimum number of windows in a kept cluster.",
    )
    ap.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Minimum cluster duration in seconds.",
    )
    ap.add_argument(
        "--max-gap-seconds",
        type=float,
        default=None,
        help="Maximum allowed inter-window gap within a cluster. Default: auto from median step.",
    )
    ap.add_argument(
        "--events-csv",
        type=str,
        default=None,
        help="Optional event summary CSV path (default: <output-json stem>_events.csv).",
    )
    ap.add_argument(
        "--summary-md",
        type=str,
        default=None,
        help="Optional markdown summary path (default: <output-json stem>_summary.md).",
    )
    ap.add_argument(
        "--debug-json",
        type=str,
        default=None,
        help="Optional detailed debug JSON (cluster diagnostics).",
    )
    args = ap.parse_args()

    if not (0.0 <= args.low_threshold <= 1.0):
        raise SystemExit("--low-threshold must be in [0,1]")
    if not (0.0 <= args.high_threshold <= 1.0):
        raise SystemExit("--high-threshold must be in [0,1]")
    if args.high_threshold < args.low_threshold:
        raise SystemExit("--high-threshold must be >= --low-threshold")
    if args.min_members < 1:
        raise SystemExit("--min-members must be >= 1")
    if args.min_duration_sec < 0:
        raise SystemExit("--min-duration-sec must be >= 0")
    if args.max_gap_seconds is not None and args.max_gap_seconds < 0:
        raise SystemExit("--max-gap-seconds must be >= 0 when provided")

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    if not input_json.exists():
        raise SystemExit(f"Input JSON not found: {input_json}")

    with open(input_json, "r") as f:
        data = json.load(f)
    items = data.get("items")
    if not isinstance(items, list):
        raise SystemExit("Invalid predictions JSON: missing list field 'items'")

    candidates: List[Candidate] = []
    total_items = len(items)
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        score = _extract_score(item, class_hierarchy=args.class_hierarchy)
        if score is None:
            continue
        start_sec, end_sec = _extract_time_bounds(item)
        candidates.append(
            Candidate(
                idx=idx,
                group=_group_key(item),
                score=float(score),
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    events, debug = _cluster_candidates(
        candidates=candidates,
        low_threshold=float(args.low_threshold),
        high_threshold=float(args.high_threshold),
        min_members=int(args.min_members),
        min_duration_sec=float(args.min_duration_sec),
        max_gap_seconds=args.max_gap_seconds,
    )

    keep_idx_to_event: Dict[int, Event] = {}
    for event in events:
        for idx in event.member_indices:
            keep_idx_to_event[idx] = event

    kept_items: List[Dict[str, Any]] = []
    by_group_counts: Dict[str, int] = {}
    for idx, item in enumerate(items):
        event = keep_idx_to_event.get(idx)
        if event is None:
            continue
        new_item = dict(item)
        new_item["postprocess_event_id"] = event.event_id
        new_item["postprocess_group"] = event.group
        new_item["postprocess_event_max_score"] = float(event.max_score)
        new_item["postprocess_event_mean_score"] = float(event.mean_score)
        new_item["postprocess_event_n_members"] = int(event.n_members)
        new_item["postprocess_event_n_high"] = int(event.n_high)
        kept_items.append(new_item)
        by_group_counts[event.group] = by_group_counts.get(event.group, 0) + 1

    output_data = dict(data)
    output_data["items"] = kept_items
    output_data["updated_at"] = _iso_now()
    output_data["postprocessing"] = {
        "method": "temporal_cluster_hysteresis_v1",
        "input_json": str(input_json),
        "generated_at": _iso_now(),
        "class_hierarchy": args.class_hierarchy,
        "low_threshold": float(args.low_threshold),
        "high_threshold": float(args.high_threshold),
        "min_members": int(args.min_members),
        "min_duration_sec": float(args.min_duration_sec),
        "max_gap_seconds": float(args.max_gap_seconds) if args.max_gap_seconds is not None else None,
        "total_items_in": total_items,
        "total_items_scored": len(candidates),
        "candidate_items": sum(1 for c in candidates if c.score >= args.low_threshold),
        "events_kept": len(events),
        "items_kept": len(kept_items),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    events_csv = Path(args.events_csv) if args.events_csv else output_json.with_name(f"{output_json.stem}_events.csv")
    summary_md = Path(args.summary_md) if args.summary_md else output_json.with_name(f"{output_json.stem}_summary.md")
    idx_to_item_id = {
        idx: str(item.get("item_id", idx))
        for idx, item in enumerate(items)
        if isinstance(item, dict)
    }
    _write_events_csv(events_csv, events, idx_to_item_id=idx_to_item_id)
    _write_summary_md(
        summary_md,
        input_json=input_json,
        output_json=output_json,
        low_threshold=float(args.low_threshold),
        high_threshold=float(args.high_threshold),
        min_members=int(args.min_members),
        min_duration_sec=float(args.min_duration_sec),
        max_gap_seconds=args.max_gap_seconds,
        total_items=total_items,
        candidate_items=sum(1 for c in candidates if c.score >= args.low_threshold),
        kept_items=len(kept_items),
        n_events=len(events),
        by_group_counts=by_group_counts,
    )

    if args.debug_json:
        debug_path = Path(args.debug_json)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w") as f:
            json.dump(debug, f, indent=2)

    print("Post-processing complete")
    print(f"  Input items: {total_items}")
    print(f"  Output items: {len(kept_items)}")
    print(f"  Events kept: {len(events)}")
    print(f"  Output JSON: {output_json}")
    print(f"  Events CSV: {events_csv}")
    print(f"  Summary MD: {summary_md}")
    if args.debug_json:
        print(f"  Debug JSON: {args.debug_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
