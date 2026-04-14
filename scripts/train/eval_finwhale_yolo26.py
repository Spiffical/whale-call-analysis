#!/usr/bin/env python3
"""Evaluate a YOLO26 fin-whale detector across one or more prepared eval YAMLs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.finwhale_yolo import ultralytics_metrics_to_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate YOLO26 fin-whale detector")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument(
        "--eval-yamls",
        type=str,
        required=True,
        help="Comma-separated list of split=yaml_path entries",
    )
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(Path(args.weights).resolve()))

    summary: dict[str, object] = {
        "weights": str(Path(args.weights).resolve()),
        "output_dir": str(output_dir),
        "splits": {},
    }

    pairs = [item.strip() for item in args.eval_yamls.split(",") if item.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Invalid --eval-yamls entry: {pair}")
        split_name, yaml_path = pair.split("=", 1)
        metrics = model.val(
            data=str(Path(yaml_path).resolve()),
            split="val",
            device=str(args.device),
            batch=int(args.batch_size),
            imgsz=int(args.imgsz),
            project=str(output_dir),
            name=f"eval_{split_name}",
            exist_ok=True,
            save_json=True,
            plots=False,
        )
        summary["splits"][split_name] = ultralytics_metrics_to_dict(metrics)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
