#!/usr/bin/env python3
"""Write the tar members needed to extract bbox raw-audio clips for a split set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build raw-audio tar member list for fin-whale bbox jobs")
    ap.add_argument("--clip-manifest", type=str, required=True)
    ap.add_argument("--split-assignments", type=str, required=True)
    ap.add_argument("--output-path", type=str, required=True)
    ap.add_argument("--tar-prefix", type=str, default="raw_audio")
    args = ap.parse_args()

    clip_df = pd.read_csv(args.clip_manifest, low_memory=False)
    assignments_df = pd.read_csv(args.split_assignments, low_memory=False)

    assigned = set(assignments_df["filename"].astype(str).tolist())
    filenames = sorted(set(clip_df[clip_df["filename"].astype(str).isin(assigned)]["filename"].astype(str).tolist()))
    members = [f"{args.tar_prefix.rstrip('/')}/{name}" for name in filenames]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(members) + ("\n" if members else ""), encoding="utf-8")

    print(f"Wrote {len(members)} tar members to {output_path}")


if __name__ == "__main__":
    main()
