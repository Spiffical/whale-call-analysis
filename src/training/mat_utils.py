#!/usr/bin/env python3
import re
import os
from pathlib import Path
from typing import Optional, Tuple, List, Iterator

# Positive samples are encoded as: <clip>_<start>s_<end>s[_suffix].mat
# (legacy comma-separated form is also supported).
FILENAME_RE_COMMA = re.compile(r"^(?P<id>[^,]+),(?P<start>[\d\.]+)s,(?P<end>[\d\.]+)s(?:,.*)?$")
FILENAME_RE_UNDERSCORE = re.compile(r"^(?P<id>.+?)_(?P<start>[\d\.]+)s_(?P<end>[\d\.]+)s(?:_.*)?$")
NEGATIVE_RE = re.compile(r"^(?P<id>.+?)\.wav_neg_(?P<idx>\d+)(?:_.*)?$")


def parse_mat_filename(filename: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Parse a MAT filename to extract source audio id and time info.

    Supports:
    - Underscore form: ICLISTENHF1353_20180705T050128.948Z_104.8s_105.3s_40Hz_custom.mat
    - Comma form:      ICLISTENHF1353_20180705T050128.948Z,104.8s,105.3s,_40Hz_custom.mat

    Returns: (source_id, start_seconds, duration_seconds)
    If time parsing fails, returns None for those fields.
    """
    name = filename
    if name.lower().endswith('.mat'):
        name = name[:-4]

    neg = NEGATIVE_RE.match(name)
    if neg:
        return neg.group("id"), None, None

    m = FILENAME_RE_UNDERSCORE.match(name)
    if not m:
        m = FILENAME_RE_COMMA.match(name)

    if not m:
        # Fallback: return the stem as id
        stem = Path(filename).stem
        return stem, None, None

    src_id = m.group('id')
    try:
        start_s = float(m.group('start'))
    except Exception:
        start_s = None
    try:
        end_s = float(m.group('end'))
    except Exception:
        end_s = None

    # Most MAT names store start/end. Keep backward compatibility with any
    # duration-like second field by falling back when end < start.
    dur_s: Optional[float]
    if start_s is None or end_s is None:
        dur_s = None
    elif end_s >= start_s:
        dur_s = end_s - start_s
    else:
        dur_s = end_s
    return src_id, start_s, dur_s


def list_mat_files(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() == '.mat'])


def iter_mat_files(dir_path: str) -> Iterator[Path]:
    """Efficiently iterate .mat files in a directory using os.scandir without building a huge list."""
    for entry in os.scandir(dir_path):
        try:
            if entry.is_file() and entry.name.lower().endswith('.mat'):
                yield Path(entry.path)
        except FileNotFoundError:
            continue
