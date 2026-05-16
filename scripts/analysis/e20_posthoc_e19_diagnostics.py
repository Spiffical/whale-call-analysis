#!/usr/bin/env python3
"""Post-hoc E19 diagnostics for the E20 experiment batch.

This produces two lightweight checks before spending more H100 time:

* per-label AP/AUPRC summaries for every E19 rung with prediction CSVs;
* a simple single-target ensemble that merges the best E19 single-species
  probe scores into one validation/test table and runs the ONC-calibrated
  evaluator on the union of ONC rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.summarize_multilabel_predictions import summarize  # noqa: E402


PRIMARY_LABELS = ("species:Bp", "species:Bm", "species:Mn", "species:Oo")


E19_RUNS = {
    "bp_low_probe": {
        "labels": ["species:Bp"],
        "run": "E19_bp_low_probe_r18_noposw_20260516T000809Z",
    },
    "bm_low_probe": {
        "labels": ["species:Bm"],
        "run": "E19_bm_low_probe_r18_noposw_20260516T000811Z",
    },
    "bp_bm_low_cumulative": {
        "labels": ["species:Bp", "species:Bm"],
        "run": "E19_bp_bm_low_cumulative_r18_noposw_20260516T000811Z",
    },
    "mn_lowmid_probe": {
        "labels": ["species:Mn"],
        "run": "E19_mn_lowmid_probe_r18_noposw_20260516T000812Z",
    },
    "bp_bm_mn_lowmid_cumulative": {
        "labels": ["species:Bp", "species:Bm", "species:Mn"],
        "run": "E19_bp_bm_mn_lowmid_cumulative_r18_noposw_20260516T000813Z",
    },
    "oo_midhigh_probe": {
        "labels": ["species:Oo"],
        "run": "E19_oo_midhigh_probe_r18_noposw_20260516T000813Z",
    },
    "full_routed_noposw": {
        "labels": list(PRIMARY_LABELS),
        "run": "E19_full_routed_allbands_r18_noposw_20260516T000814Z",
    },
    "full_routed_posw": {
        "labels": list(PRIMARY_LABELS),
        "run": "E19_full_routed_allbands_r18_posw_20260516T000814Z",
    },
}


ENSEMBLE_MEMBERS = {
    "species:Bp": "bp_low_probe",
    "species:Bm": "bm_low_probe",
    "species:Mn": "mn_lowmid_probe",
    "species:Oo": "oo_midhigh_probe",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def split_labels(value: Any) -> List[str]:
    out: List[str] = []
    for token in clean(value).replace(";", "|").replace(",", "|").split("|"):
        token = token.strip()
        if token:
            out.append(token)
    return out


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int((y_true >= 0.5).sum())
    if positives <= 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = (y_true[order] >= 0.5).astype(np.float64)
    tp = np.cumsum(sorted_true)
    rank = np.arange(1, len(sorted_true) + 1, dtype=np.float64)
    precision_at_k = tp / rank
    return float((precision_at_k * sorted_true).sum() / positives)


def score_labels(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    labels: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("score__"):
                label = key.removeprefix("score__")
                if label not in labels:
                    labels.append(label)
    return labels


def ap_rows(
    *,
    run_name: str,
    split: str,
    rows: Sequence[Mapping[str, str]],
    label_ids: Sequence[str],
    source_kind: str,
) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if not source_kind or clean(row.get("source_kind")) == source_kind]
    out: List[Dict[str, Any]] = []
    for label in label_ids:
        if not filtered or f"score__{label}" not in filtered[0]:
            continue
        y_true = np.array([1.0 if label in split_labels(row.get("target_label_ids")) else 0.0 for row in filtered], dtype=np.float32)
        y_score = np.array([float(row.get(f"score__{label}") or 0.0) for row in filtered], dtype=np.float32)
        out.append(
            {
                "run": run_name,
                "split": split,
                "source_kind": source_kind or "<all>",
                "label_id": label,
                "rows": len(filtered),
                "support": int(y_true.sum()),
                "average_precision": average_precision(y_true, y_score),
                "score_mean_positive": float(y_score[y_true >= 0.5].mean()) if int(y_true.sum()) else float("nan"),
                "score_mean_negative": float(y_score[y_true < 0.5].mean()) if int((y_true < 0.5).sum()) else float("nan"),
            }
        )
    return out


def base_key(row: Mapping[str, str]) -> str:
    return clean(row.get("item_id")) or "|".join(
        [
            clean(row.get("source_dataset")),
            clean(row.get("source_audio")),
            clean(row.get("begin_s")),
            clean(row.get("end_s")),
            clean(row.get("split")),
        ]
    )


def merge_target_labels(existing: str, new: str) -> str:
    labels = []
    for label in [*split_labels(existing), *split_labels(new)]:
        if label not in labels:
            labels.append(label)
    return "|".join(labels)


def build_ensemble_rows(
    *,
    weekend_root: Path,
    split: str,
    source_kind: str = "ONC",
) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for label, run_key in ENSEMBLE_MEMBERS.items():
        run_dir = weekend_root / "runs" / E19_RUNS[run_key]["run"]
        csv_path = run_dir / "train" / f"{split}_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        for row in read_csv(csv_path):
            if source_kind and clean(row.get("source_kind")) != source_kind:
                continue
            key = base_key(row)
            out = merged.get(key)
            if out is None:
                out = {
                    "item_id": key,
                    "source_dataset": clean(row.get("source_dataset")),
                    "source_kind": clean(row.get("source_kind")),
                    "source_audio": clean(row.get("source_audio")),
                    "mat_path": clean(row.get("mat_path")),
                    "source_label_ids": clean(row.get("source_label_ids")),
                    "canonical_label_ids": clean(row.get("canonical_label_ids")),
                    "analysis_label_ids": clean(row.get("analysis_label_ids")),
                    "negative_bucket": clean(row.get("negative_bucket")),
                    "split": split.replace("validation", "val"),
                    "is_background": clean(row.get("is_background")),
                    "review_status": clean(row.get("review_status")),
                    "context_tags": clean(row.get("context_tags")),
                    "begin_s": clean(row.get("begin_s")),
                    "end_s": clean(row.get("end_s")),
                    "event_group": clean(row.get("event_group")),
                    "target_label_ids": "",
                    "pred_label_ids": "",
                }
                for primary in PRIMARY_LABELS:
                    out[f"score__{primary}"] = "0.00000000"
                merged[key] = out
            out["target_label_ids"] = merge_target_labels(out.get("target_label_ids", ""), row.get("target_label_ids", ""))
            if f"score__{label}" in row:
                out[f"score__{label}"] = clean(row.get(f"score__{label}"))
    return list(merged.values())


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [safe_json(val) for val in value]
    return value


def run_posthoc(*, weekend_root: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ap_summary: List[Dict[str, Any]] = []
    for run_name, info in E19_RUNS.items():
        run_dir = weekend_root / "runs" / info["run"]
        for split in ("validation", "test"):
            csv_path = run_dir / "train" / f"{split}_predictions.csv"
            if not csv_path.exists():
                continue
            rows = read_csv(csv_path)
            for source_kind in ("ONC", "BioDCASE", "DCLDE", ""):
                ap_summary.extend(
                    ap_rows(
                        run_name=run_name,
                        split=split,
                        rows=rows,
                        label_ids=info["labels"],
                        source_kind=source_kind,
                    )
                )
    write_csv(output_dir / "e19_pr_ap_summary.csv", ap_summary)

    val_rows = build_ensemble_rows(weekend_root=weekend_root, split="validation", source_kind="ONC")
    test_rows = build_ensemble_rows(weekend_root=weekend_root, split="test", source_kind="ONC")
    ensemble_dir = output_dir / "single_target_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "source_dataset",
        "source_kind",
        "source_audio",
        "mat_path",
        "source_label_ids",
        "canonical_label_ids",
        "analysis_label_ids",
        "negative_bucket",
        "split",
        "is_background",
        "review_status",
        "context_tags",
        "begin_s",
        "end_s",
        "event_group",
        "target_label_ids",
        "pred_label_ids",
        *[f"score__{label}" for label in PRIMARY_LABELS],
    ]
    validation_csv = ensemble_dir / "validation_predictions.csv"
    test_csv = ensemble_dir / "test_predictions.csv"
    write_csv(validation_csv, val_rows, fieldnames=fieldnames)
    write_csv(test_csv, test_rows, fieldnames=fieldnames)
    ensemble_summary = summarize(
        validation_csv=validation_csv,
        test_csv=test_csv,
        output_dir=ensemble_dir / "onc_calibrated_eval",
        calibration_source_kind="ONC",
        eval_source_kind="ONC",
        label_ids=list(PRIMARY_LABELS),
    )
    payload = {
        "weekend_root": str(weekend_root),
        "ap_summary_csv": str(output_dir / "e19_pr_ap_summary.csv"),
        "ensemble_validation_csv": str(validation_csv),
        "ensemble_test_csv": str(test_csv),
        "ensemble_summary": ensemble_summary,
        "ensemble_validation_rows": len(val_rows),
        "ensemble_test_rows": len(test_rows),
    }
    (output_dir / "e20_posthoc_summary.json").write_text(
        json.dumps(safe_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekend-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = run_posthoc(weekend_root=args.weekend_root, output_dir=args.output_dir)
    print(json.dumps(safe_json(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
