#!/usr/bin/env python3
"""Collect E22 metrics, expert ensembles, and visual audit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.summarize_multilabel_predictions import summarize  # noqa: E402
from src.dataset.multilabel import write_csv_rows  # noqa: E402


LABEL_NAMES = {
    "species:Bp": "fin whale",
    "species:Bm": "blue whale",
    "species:Mn": "humpback whale",
    "species:Oo": "killer whale",
}
THREE_SPECIES = ("species:Bp", "species:Bm", "species:Mn")
FOUR_SPECIES = ("species:Bp", "species:Bm", "species:Mn", "species:Oo")
EXPERT_MEMBERS = {
    "species:Bp": "E22_fin_whale_low_expert_balanced",
    "species:Bm": "E22_blue_whale_low_expert_balanced",
    "species:Mn": "E22_humpback_whale_lowmid_expert_balanced",
    "species:Oo": "E22_killer_whale_onc_only_midhigh_expert_balanced",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def split_labels(value: Any) -> List[str]:
    return [token.strip() for token in clean(value).replace(",", "|").replace(";", "|").split("|") if token.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
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


def read_submitted(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
    labels: List[str] = []
    for label in [*split_labels(existing), *split_labels(new)]:
        if label not in labels:
            labels.append(label)
    return "|".join(labels)


def score_labels(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    labels: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("score__"):
                label = key.removeprefix("score__")
                if label not in labels:
                    labels.append(label)
    return labels


def run_dir_by_experiment(submitted_rows: Sequence[Mapping[str, str]]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for row in submitted_rows:
        exp = clean(row.get("experiment"))
        run_dir = clean(row.get("run_dir"))
        if exp and run_dir and exp != "DRY_RUN":
            out[exp] = Path(run_dir)
    return out


def metrics_rows(submitted_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in submitted_rows:
        experiment = clean(row.get("experiment"))
        run_dir = Path(clean(row.get("run_dir")))
        if not experiment or not run_dir:
            continue
        metrics_path = run_dir / "train" / "onc_calibrated_eval" / "onc_calibrated_metrics_summary.json"
        if not metrics_path.exists():
            out.append({"job_id": clean(row.get("job_id")), "experiment": experiment, "status": "missing_metrics", "metrics_path": str(metrics_path)})
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = payload.get("onc_test_metrics", {})
        hard_rows = payload.get("onc_test_hard_negative_fp_rows", [])
        hard_fp = sum(int(item.get("any_primary_fp") or 0) for item in hard_rows)
        hard_total = sum(int(item.get("rows") or 0) for item in hard_rows)
        out.append(
            {
                "job_id": clean(row.get("job_id")),
                "experiment": experiment,
                "status": "complete",
                "macro_f1": test.get("macro_f1_supported", 0.0),
                "micro_f1": test.get("micro_f1", 0.0),
                "precision": test.get("micro_precision", 0.0),
                "recall": test.get("micro_recall", 0.0),
                "tp": test.get("tp", 0),
                "fp": test.get("fp", 0),
                "fn": test.get("fn", 0),
                "hard_fp": hard_fp,
                "hard_total": hard_total,
                "hard_fp_rate": hard_fp / max(hard_total, 1) if hard_total else "",
                "label_ids": ",".join(payload.get("label_ids", [])),
                "calibration_source_kind": payload.get("calibration_source_kind", ""),
                "eval_source_kind": payload.get("eval_source_kind", ""),
                "metrics_path": str(metrics_path),
            }
        )
    return out


def build_ensemble_rows(
    *,
    run_dirs: Mapping[str, Path],
    split: str,
    label_ids: Sequence[str],
    source_kind: str,
) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for label in label_ids:
        exp_name = EXPERT_MEMBERS[label]
        run_dir = run_dirs.get(exp_name)
        if run_dir is None:
            raise FileNotFoundError(f"Missing run dir for {exp_name}")
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
                    "low_mat_path": clean(row.get("low_mat_path")),
                    "mid_mat_path": clean(row.get("mid_mat_path")),
                    "high_mat_path": clean(row.get("high_mat_path")),
                    "source_label_ids": clean(row.get("source_label_ids")),
                    "canonical_label_ids": clean(row.get("canonical_label_ids")),
                    "analysis_label_ids": clean(row.get("analysis_label_ids")),
                    "negative_bucket": clean(row.get("negative_bucket")),
                    "split": "val" if split == "validation" else split,
                    "is_background": clean(row.get("is_background")),
                    "review_status": clean(row.get("review_status")),
                    "context_tags": clean(row.get("context_tags")),
                    "begin_s": clean(row.get("begin_s")),
                    "end_s": clean(row.get("end_s")),
                    "event_group": clean(row.get("event_group")),
                    "target_label_ids": "",
                    "pred_label_ids": "",
                }
                for primary in FOUR_SPECIES:
                    out[f"score__{primary}"] = "0.00000000"
                merged[key] = out
            out["target_label_ids"] = merge_target_labels(out.get("target_label_ids", ""), row.get("target_label_ids", ""))
            if f"score__{label}" in row:
                out[f"score__{label}"] = clean(row.get(f"score__{label}"))
    return list(merged.values())


def evaluate_ensemble(
    *,
    run_dirs: Mapping[str, Path],
    output_dir: Path,
    name: str,
    label_ids: Sequence[str],
) -> Dict[str, Any]:
    ensemble_dir = output_dir / name
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "source_dataset",
        "source_kind",
        "source_audio",
        "mat_path",
        "low_mat_path",
        "mid_mat_path",
        "high_mat_path",
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
        *[f"score__{label}" for label in FOUR_SPECIES],
    ]
    val_rows = build_ensemble_rows(run_dirs=run_dirs, split="validation", label_ids=label_ids, source_kind="ONC")
    test_rows = build_ensemble_rows(run_dirs=run_dirs, split="test", label_ids=label_ids, source_kind="ONC")
    validation_csv = ensemble_dir / "validation_predictions.csv"
    test_csv = ensemble_dir / "test_predictions.csv"
    write_csv(validation_csv, val_rows, fieldnames=fieldnames)
    write_csv(test_csv, test_rows, fieldnames=fieldnames)
    summary = summarize(
        validation_csv=validation_csv,
        test_csv=test_csv,
        output_dir=ensemble_dir / "onc_calibrated_eval",
        calibration_source_kind="ONC",
        eval_source_kind="ONC",
        label_ids=label_ids,
    )
    return {
        "name": name,
        "label_ids": list(label_ids),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "summary": summary,
        "output_dir": str(ensemble_dir),
    }


def month_bin(row: Mapping[str, Any]) -> str:
    text = " ".join(clean(row.get(key)) for key in ("item_id", "source_audio", "mat_path", "low_mat_path"))
    match = re.search(r"(20\d{2})[-_]?([01]\d)", text)
    return f"{match.group(1)}-{match.group(2)}" if match else "<unknown>"


def thresholds_from_metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("onc_validation_thresholds", {})
    return {label: float(info.get("threshold", 0.5)) for label, info in raw.items()}


def prediction_audit_rows(
    *,
    experiment: str,
    predictions_csv: Path,
    metrics_path: Path,
    label_ids: Sequence[str],
    eval_source_kind: str,
    max_rows_per_kind: int,
) -> List[Dict[str, Any]]:
    rows = [row for row in read_csv(predictions_csv) if clean(row.get("source_kind")) == eval_source_kind]
    thresholds = thresholds_from_metrics(metrics_path)
    candidates: List[Dict[str, Any]] = []
    rng = random.Random(2026 + sum(ord(ch) for ch in experiment))
    for label in label_ids:
        score_key = f"score__{label}"
        threshold = float(thresholds.get(label, 0.5))
        false_pos: List[Dict[str, Any]] = []
        false_neg: List[Dict[str, Any]] = []
        for row in rows:
            if score_key not in row:
                continue
            score = safe_float(row.get(score_key))
            targets = set(split_labels(row.get("target_label_ids")))
            item = {
                **row,
                "experiment": experiment,
                "label_id": label,
                "label_name": LABEL_NAMES.get(label, label),
                "score": f"{score:.8f}",
                "threshold": f"{threshold:.2f}",
                "margin": f"{score - threshold:.8f}",
                "review_label": "",
                "review_options": "missed true call|adjacent-call leakage|non-target biological transient|noise/artifact|unclear",
            }
            if label not in targets and score >= threshold:
                false_pos.append({**item, "audit_kind": "false_positive"})
            if label in targets and score < threshold:
                false_neg.append({**item, "audit_kind": "false_negative"})
        for kind_rows, kind in ((false_pos, "false_positive"), (false_neg, "false_negative")):
            top = sorted(kind_rows, key=lambda row: safe_float(row.get("margin")), reverse=(kind == "false_positive"))[:max_rows_per_kind]
            near = sorted(kind_rows, key=lambda row: abs(safe_float(row.get("margin"))))[:max_rows_per_kind]
            shuffled = list(kind_rows)
            rng.shuffle(shuffled)
            random_rows = shuffled[:max_rows_per_kind]
            for sample_name, sample_rows in (("top_score", top), ("near_threshold", near), ("random", random_rows)):
                for out_row in sample_rows:
                    candidates.append({**out_row, "sample_strategy": sample_name})
    return candidates


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_band_image(path_text: str, band: str) -> Optional[np.ndarray]:
    path = Path(clean(path_text))
    if not path.exists():
        return None
    try:
        import scipy.io as sio
        from src.dataset.multiband import _extract_spectrogram_raw
        from src.training.mat_dataset import _normalize_db_to_unit, _power_to_db_norm
    except Exception:
        return None
    try:
        payload = sio.loadmat(str(path), simplify_cells=True)
        spec, kind, _, _ = _extract_spectrogram_raw(payload, path, band=band)
        if kind == "power":
            spec = _power_to_db_norm(spec)
        else:
            spec = _normalize_db_to_unit(np.asarray(spec, dtype=np.float32), -80.0, 0.0)
        image = np.nan_to_num(np.asarray(spec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(image, [2, 98])
        if hi > lo:
            image = (image - lo) / (hi - lo)
        return np.clip(image, 0.0, 1.0)
    except Exception:
        return None


def make_multiband_contact_sheet(rows: Sequence[Mapping[str, Any]], out_path: Path, *, title: str, max_rows: int = 12) -> bool:
    selected = list(rows)[:max_rows]
    if not selected:
        return False
    plt = ensure_matplotlib()
    bands = ("low", "mid", "high")
    fig, axes = plt.subplots(len(selected), len(bands), figsize=(len(bands) * 4.2, max(2.4, len(selected) * 2.0)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_idx, row in enumerate(selected):
        for band_idx, band in enumerate(bands):
            ax = axes[row_idx, band_idx]
            image = load_band_image(clean(row.get(f"{band}_mat_path") or row.get("mat_path")), band)
            if image is not None:
                ax.imshow(image, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, f"missing {band}", ha="center", va="center", fontsize=7)
            ax.set_title(band, fontsize=8)
            ax.axis("off")
        item = clean(row.get("item_id"))[:46]
        label = clean(row.get("label_name") or row.get("label_id"))
        score = clean(row.get("score"))
        threshold = clean(row.get("threshold"))
        bucket = clean(row.get("negative_bucket"))
        axes[row_idx, 0].set_ylabel(f"{label}\n{score}/{threshold}\n{bucket}\n{item}", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def build_prediction_audit(
    *,
    submitted_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    max_rows_per_kind: int,
) -> Dict[str, Any]:
    audit_dir = output_dir / "prediction_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    rendered: List[str] = []
    for row in submitted_rows:
        experiment = clean(row.get("experiment"))
        run_dir = Path(clean(row.get("run_dir")))
        metrics_path = run_dir / "train" / "onc_calibrated_eval" / "onc_calibrated_metrics_summary.json"
        predictions_csv = run_dir / "train" / "test_predictions.csv"
        if not metrics_path.exists() or not predictions_csv.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        label_ids = payload.get("label_ids", [])
        eval_source_kind = clean(payload.get("eval_source_kind")) or "ONC"
        rows = prediction_audit_rows(
            experiment=experiment,
            predictions_csv=predictions_csv,
            metrics_path=metrics_path,
            label_ids=label_ids,
            eval_source_kind=eval_source_kind,
            max_rows_per_kind=max_rows_per_kind,
        )
        all_rows.extend(rows)
        for (kind, strategy), group in _group_rows(rows, ["audit_kind", "sample_strategy"]).items():
            out_png = audit_dir / f"{experiment}_{kind}_{strategy}.png"
            if make_multiband_contact_sheet(group, out_png, title=f"{experiment} {kind} {strategy}"):
                rendered.append(str(out_png))
    write_csv(audit_dir / "prediction_review_queue.csv", all_rows)
    return {"review_queue_csv": str(audit_dir / "prediction_review_queue.csv"), "row_count": len(all_rows), "contact_sheets": rendered}


def _group_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(clean(row.get(key)) for key in keys)].append(row)
    return grouped


def killer_whale_domain_audit(
    *,
    run_dirs: Mapping[str, Path],
    output_dir: Path,
) -> Dict[str, Any]:
    audit_dir = output_dir / "killer_whale_domain_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows_out: List[Dict[str, Any]] = []
    contact_rows: List[Dict[str, Any]] = []
    for experiment in (
        "E22_killer_whale_onc_only_midhigh_expert_balanced",
        "E22_killer_whale_dclde_only_midhigh_expert_balanced",
        "E22_killer_whale_onc_dclde_midhigh_sourcecap_balanced",
    ):
        run_dir = run_dirs.get(experiment)
        if run_dir is None:
            continue
        for split in ("validation", "test"):
            csv_path = run_dir / "train" / f"{split}_predictions.csv"
            if not csv_path.exists():
                continue
            for row in read_csv(csv_path):
                score = safe_float(row.get("score__species:Oo"))
                target = "species:Oo" in set(split_labels(row.get("target_label_ids")))
                rows_out.append(
                    {
                        "experiment": experiment,
                        "split": split,
                        "source_kind": clean(row.get("source_kind")),
                        "source_dataset": clean(row.get("source_dataset")),
                        "month": month_bin(row),
                        "is_killer_whale_target": int(target),
                        "score": score,
                    }
                )
                if target:
                    contact_rows.append({**row, "experiment": experiment, "label_name": "killer whale", "score": f"{score:.8f}", "threshold": ""})
    dist_rows: List[Dict[str, Any]] = []
    for key, group in _group_rows(rows_out, ["experiment", "split", "source_kind", "is_killer_whale_target"]).items():
        scores = np.asarray([safe_float(row.get("score")) for row in group], dtype=np.float32)
        dist_rows.append(
            {
                "experiment": key[0],
                "split": key[1],
                "source_kind": key[2],
                "is_killer_whale_target": key[3],
                "rows": len(group),
                "score_mean": float(scores.mean()) if scores.size else "",
                "score_q05": float(np.quantile(scores, 0.05)) if scores.size else "",
                "score_q50": float(np.quantile(scores, 0.50)) if scores.size else "",
                "score_q95": float(np.quantile(scores, 0.95)) if scores.size else "",
            }
        )
    month_rows = [
        {"experiment": exp, "split": split, "source_kind": src, "month": month, "rows": count}
        for (exp, split, src, month), count in Counter(
            (clean(row.get("experiment")), clean(row.get("split")), clean(row.get("source_kind")), clean(row.get("month")))
            for row in rows_out
        ).most_common()
    ]
    write_csv(audit_dir / "killer_whale_score_distributions.csv", dist_rows)
    write_csv(audit_dir / "killer_whale_month_counts.csv", month_rows)
    contact_rows = sorted(contact_rows, key=lambda row: (clean(row.get("source_kind")), -safe_float(row.get("score"))))
    contact_png = audit_dir / "killer_whale_onc_dclde_positive_examples.png"
    make_multiband_contact_sheet(contact_rows, contact_png, title="Killer whale ONC and DCLDE positive examples", max_rows=18)
    return {
        "score_distribution_csv": str(audit_dir / "killer_whale_score_distributions.csv"),
        "month_counts_csv": str(audit_dir / "killer_whale_month_counts.csv"),
        "contact_sheet": str(contact_png),
    }


def write_report(
    *,
    output_dir: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    ensemble_payloads: Sequence[Mapping[str, Any]],
    prediction_audit: Mapping[str, Any],
    killer_audit: Mapping[str, Any],
) -> Path:
    report = output_dir / "e22_expert_multihead_report.md"
    lines = [
        "# E22 Expert/Multi-Head Diagnostic Report",
        "",
        "This report is generated on Nibi and should not be committed unless explicitly requested.",
        "",
        "## Metrics",
        "",
        "| Experiment | Macro F1 | Micro F1 | Hard-negative FP | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in sorted(metric_rows, key=lambda item: safe_float(item.get("macro_f1")), reverse=True):
        hard = ""
        if clean(row.get("hard_total")):
            hard = f"{row.get('hard_fp')}/{row.get('hard_total')}={safe_float(row.get('hard_fp_rate')):.4f}"
        lines.append(
            f"| {row.get('experiment')} | {safe_float(row.get('macro_f1')):.4f} | "
            f"{safe_float(row.get('micro_f1')):.4f} | {hard} | {row.get('status')} |"
        )
    lines.extend(["", "## Expert Ensembles", ""])
    for payload in ensemble_payloads:
        summary = payload["summary"]["onc_test_metrics"]
        hard_rows = payload["summary"].get("onc_test_hard_negative_fp_rows", [])
        hard_fp = sum(int(item.get("any_primary_fp") or 0) for item in hard_rows)
        hard_total = sum(int(item.get("rows") or 0) for item in hard_rows)
        lines.append(
            f"- `{payload['name']}`: macro F1 `{summary.get('macro_f1_supported', 0.0):.4f}`, "
            f"micro F1 `{summary.get('micro_f1', 0.0):.4f}`, hard-negative FP "
            f"`{hard_fp}/{hard_total}={hard_fp / max(hard_total, 1):.4f}`."
        )
    lines.extend(
        [
            "",
            "## Audit Artifacts",
            "",
            f"- Prediction review queue: `{prediction_audit.get('review_queue_csv', '')}`.",
            f"- Prediction contact sheets: `{len(prediction_audit.get('contact_sheets', []))}` files.",
            f"- Killer whale score distributions: `{killer_audit.get('score_distribution_csv', '')}`.",
            f"- Killer whale contact sheet: `{killer_audit.get('contact_sheet', '')}`.",
            "",
            "## Gate Notes",
            "",
            "- Three-species expert ensemble target: macro F1 at or above `0.7063` and hard-negative FP at or below `0.20`.",
            "- Multi-head three-species models are promising only if they approach the expert ensemble and beat E21 shared three-species models.",
            "- Killer whale remains diagnostic unless ONC support improves without a false-positive blowup.",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def smoke_test(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    val = output_dir / "validation_predictions.csv"
    test = output_dir / "test_predictions.csv"
    rows = [
        {"item_id": "a", "source_kind": "ONC", "target_label_ids": "species:Bp", "score__species:Bp": "0.9", "negative_bucket": ""},
        {"item_id": "b", "source_kind": "ONC", "target_label_ids": "", "score__species:Bp": "0.1", "negative_bucket": "primary_adjacent_gap"},
    ]
    write_csv(val, rows)
    write_csv(test, rows)
    summarize(
        validation_csv=val,
        test_csv=test,
        output_dir=output_dir / "eval",
        calibration_source_kind="ONC",
        eval_source_kind="ONC",
        label_ids=("species:Bp",),
    )


def run_collect(*, submitted_tsv: Path, output_dir: Path, max_rows_per_kind: int) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    submitted = read_submitted(submitted_tsv)
    run_dirs = run_dir_by_experiment(submitted)
    metric_rows = metrics_rows(submitted)
    write_csv(output_dir / "e22_metrics_summary.csv", metric_rows)
    ensemble_payloads = [
        evaluate_ensemble(run_dirs=run_dirs, output_dir=output_dir, name="three_species_expert_ensemble", label_ids=THREE_SPECIES),
        evaluate_ensemble(run_dirs=run_dirs, output_dir=output_dir, name="four_species_diagnostic_expert_ensemble", label_ids=FOUR_SPECIES),
    ]
    (output_dir / "e22_ensemble_summary.json").write_text(json.dumps(ensemble_payloads, indent=2, sort_keys=True), encoding="utf-8")
    prediction_audit = build_prediction_audit(submitted_rows=submitted, output_dir=output_dir, max_rows_per_kind=max_rows_per_kind)
    killer_audit = killer_whale_domain_audit(run_dirs=run_dirs, output_dir=output_dir)
    report_path = write_report(
        output_dir=output_dir,
        metric_rows=metric_rows,
        ensemble_payloads=ensemble_payloads,
        prediction_audit=prediction_audit,
        killer_audit=killer_audit,
    )
    payload = {
        "submitted_tsv": str(submitted_tsv),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "metrics_csv": str(output_dir / "e22_metrics_summary.csv"),
        "prediction_audit": prediction_audit,
        "killer_whale_domain_audit": killer_audit,
    }
    (output_dir / "e22_collect_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submitted-tsv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-rows-per-kind", type=int, default=12)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test(args.output_dir)
        return 0
    if args.submitted_tsv is None:
        raise SystemExit("--submitted-tsv is required unless --smoke-test is set")
    payload = run_collect(
        submitted_tsv=args.submitted_tsv,
        output_dir=args.output_dir,
        max_rows_per_kind=int(args.max_rows_per_kind),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
