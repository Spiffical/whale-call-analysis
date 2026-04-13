#!/usr/bin/env python3
"""Run CAM / attribution localization experiments for fin-whale spectrograms."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.analysis.attention_localization import (
    CAM_METHODS,
    FIN_BUCKET_FREQ_PRIORS,
    aggregate_metric_rows,
    build_annotation_crop,
    build_mat_lookup,
    build_negative_crop_from_annotation,
    derive_annotation_frequency_bounds,
    generate_attention_artifacts,
    load_localized_annotations,
    load_model_checkpoint,
    render_attention_panel,
    resolve_target_layers,
    save_attention_arrays,
    summarize_localization,
    write_csv,
    write_json,
)


PILOT_CONTEXTS = ("vessel_or_masking", "mixed_species", "faint")


def _parse_checkpoint_spec(raw: str) -> Tuple[str, str]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), path.strip()
    path = raw.strip()
    return Path(path).parent.parent.name or Path(path).stem, path


def _parse_methods(raw: str) -> List[str]:
    methods = [token.strip().lower() for token in str(raw or "").split(",") if token.strip()]
    if not methods:
        raise ValueError("Expected at least one attention method")
    valid = set(CAM_METHODS) | {"integrated_gradients", "occlusion"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def _annotation_sort_key(row) -> Tuple[Any, ...]:
    return (row.filename, row.begin_time_s, row.end_time_s, row.annotation_id)


def _sample_rows(rows: Sequence[Any], limit: int, rng: random.Random) -> List[Any]:
    rows = list(rows)
    if len(rows) <= limit:
        return rows
    return rng.sample(rows, limit)


def select_pilot_positive_annotations(
    annotations: Sequence[Any],
    *,
    per_bucket: int,
    per_context: int,
    remainder_limit: int,
    seed: int,
) -> List[Any]:
    rng = random.Random(seed)
    selected: Dict[str, Any] = {}
    positives = [row for row in annotations if row.species == "Bp"]
    for bucket in ("20Hz", "40Hz", "other_fin"):
        bucket_rows = sorted([row for row in positives if row.call_type_bucket == bucket], key=_annotation_sort_key)
        for row in _sample_rows(bucket_rows, per_bucket, rng):
            selected[row.annotation_id] = row
    for context in PILOT_CONTEXTS:
        ctx_rows = sorted([row for row in positives if context in row.context_tags], key=_annotation_sort_key)
        for row in _sample_rows(ctx_rows, per_context, rng):
            selected[row.annotation_id] = row
    leftovers = [row for row in positives if row.annotation_id not in selected]
    for row in _sample_rows(leftovers, remainder_limit, rng):
        selected[row.annotation_id] = row
    return sorted(selected.values(), key=_annotation_sort_key)


def select_negative_annotations(
    all_annotations: Sequence[Any],
    *,
    limit: int,
    seed: int,
) -> List[Any]:
    rng = random.Random(seed)
    negatives = [
        row
        for row in all_annotations
        if row.species and row.species != "Bp" and row.end_time_s > row.begin_time_s
    ]
    return sorted(_sample_rows(negatives, limit, rng), key=_annotation_sort_key)


def _find_mat_path(annotation, mat_lookup: Mapping[str, Path]) -> Optional[Path]:
    if annotation.filename in mat_lookup:
        return mat_lookup[annotation.filename]
    for key, value in mat_lookup.items():
        if annotation.filename in key:
            return value
    return None


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _method_ranking_rows(overall_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in overall_rows:
        pointing = row.get("pointing_hit_mean")
        box_iou = row.get("box_iou_mean")
        mask_cov = row.get("mask_coverage_mean")
        score = 0.0
        parts = []
        for value, weight in ((pointing, 0.45), (box_iou, 0.35), (mask_cov, 0.20)):
            if value is not None:
                score += float(value) * weight
                parts.append(weight)
        if not parts:
            combined = None
        else:
            combined = score / sum(parts)
        out.append(
            {
                "model_label": row.get("model_label"),
                "method": row.get("method"),
                "count": row.get("count"),
                "combined_localization_score": combined,
                "pointing_hit_mean": pointing,
                "box_iou_mean": box_iou,
                "mask_coverage_mean": mask_cov,
                "temporal_iou_mean": row.get("temporal_iou_mean"),
                "frequency_iou_mean": row.get("frequency_iou_mean"),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            row["model_label"],
            -(row["combined_localization_score"] if row["combined_localization_score"] is not None else -1.0),
            row["method"],
        ),
    )


def _bucket_failure_mode(row: Mapping[str, Any], crop_area: int) -> Optional[str]:
    box_iou = row.get("box_iou")
    temporal = row.get("temporal_iou")
    frequency = row.get("frequency_iou")
    if box_iou is None:
        return None
    t0, t1 = row.get("pred_box_t0"), row.get("pred_box_t1")
    f0, f1 = row.get("pred_box_f0"), row.get("pred_box_f1")
    if None not in (t0, t1, f0, f1):
        pred_area = max(0, int(t1) - int(t0)) * max(0, int(f1) - int(f0))
        if crop_area > 0 and (pred_area / crop_area) > 0.45 and float(box_iou) < 0.25:
            return "diffuse_activation"
        if temporal is not None and float(temporal) < 0.2 and float(frequency or 0.0) >= 0.4:
            return "time_shifted"
        if frequency is not None and float(frequency) < 0.2 and float(temporal or 0.0) >= 0.4:
            return "wrong_noise_band"
    if float(box_iou) < 0.1:
        return "missed_call_region"
    return None


def _draft_recommendation(ranking_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    best_rows = [row for row in ranking_rows if row.get("combined_localization_score") is not None]
    if not best_rows:
        return [
            "# Recommendation",
            "",
            "No valid localization rows were produced, so CAM-style localization is not yet supported by evidence.",
            "",
            "Recommended fallback: train a dedicated detector, starting with `RT-DETR` for box prediction.",
        ]
    best = best_rows[0]
    score = float(best["combined_localization_score"])
    method = best["method"]
    model_label = best["model_label"]
    lines = [
        "# Recommendation",
        "",
        f"Best observed pilot/localization method: `{method}` on `{model_label}`.",
        "",
    ]
    if score >= 0.55 and (best.get("box_iou_mean") or 0.0) >= 0.35 and (best.get("pointing_hit_mean") or 0.0) >= 0.75:
        lines.append(
            "Current evidence suggests CAM-style localization is good enough to use as a proposal generator for fin-call masks/boxes, with manual review or a lightweight post-filter."
        )
        lines.append("")
        lines.append(
            "Next step: keep the classifier, use the CAM map as the proposal stage, and only train a detector later if downstream proposal precision is still too low."
        )
    else:
        lines.append(
            "Current evidence suggests CAM-style localization is not strong enough to rely on as the main box generator without a dedicated detector."
        )
        lines.append("")
        lines.append(
            "Recommended fallback: train a dedicated detector next, starting with `RT-DETR`; if masks matter more than boxes, try a lightweight segmentation model after that."
        )
    return lines


def _write_markdown_summary(
    *,
    output_path: Path,
    title: str,
    overall_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [f"# {title}", "", "## Overall", ""]
    lines.append("| model | method | count | box IoU | temporal IoU | frequency IoU | pointing | mask coverage |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in overall_rows:
        lines.append(
            "| {model} | {method} | {count} | {box} | {time} | {freq} | {point} | {cov} |".format(
                model=row.get("model_label"),
                method=row.get("method"),
                count=row.get("count"),
                box=_format_metric(row.get("box_iou_mean")),
                time=_format_metric(row.get("temporal_iou_mean")),
                freq=_format_metric(row.get("frequency_iou_mean")),
                point=_format_metric(row.get("pointing_hit_mean")),
                cov=_format_metric(row.get("mask_coverage_mean")),
            )
        )
    lines.extend(["", "## Ranking", "", "| model | method | combined score | box IoU | pointing |", "| --- | --- | ---: | ---: | ---: |"])
    for row in ranking_rows:
        lines.append(
            "| {model} | {method} | {score} | {box} | {point} |".format(
                model=row.get("model_label"),
                method=row.get("method"),
                score=_format_metric(row.get("combined_localization_score")),
                box=_format_metric(row.get("box_iou_mean")),
                point=_format_metric(row.get("pointing_hit_mean")),
            )
        )
    _write_lines(output_path, lines)


def _gallery_groups(
    rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    *,
    prediction_threshold: float,
    limit_per_group: int,
) -> Dict[str, List[str]]:
    groups: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for row in rows:
        key = f"{row['model_label']}::{row['method']}"
        box_iou = row.get("box_iou")
        score = float(row.get("score") or 0.0)
        png_path = str(row.get("panel_png") or "")
        if not png_path:
            continue
        if box_iou is not None and float(box_iou) >= 0.45:
            groups[f"{key}::good"].append((float(box_iou), png_path))
        if row.get("pointing_hit") == 0.0 or (box_iou is not None and float(box_iou) < 0.15):
            groups[f"{key}::bad"].append((-float(box_iou or 0.0), png_path))
        failure = row.get("failure_mode")
        if failure:
            groups[f"{key}::{failure}"].append((-score, png_path))
    for row in negative_rows:
        score = float(row.get("score") or 0.0)
        png_path = str(row.get("panel_png") or "")
        if score >= float(prediction_threshold) and png_path:
            key = f"{row['model_label']}::{row['method']}::false_positive"
            groups[key].append((-score, png_path))
    out: Dict[str, List[str]] = {}
    for key, items in groups.items():
        items.sort(key=lambda item: item[0], reverse=True)
        out[key] = [path for _, path in items[:limit_per_group]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fin-whale attention localization experiments")
    ap.add_argument("--checkpoint", action="append", required=True, help="Repeatable `label=/path/to/best.pt`")
    ap.add_argument("--methods", type=str, default="gradcampp,hirescam,layercam,scorecam,integrated_gradients")
    ap.add_argument("--mode", type=str, choices=["pilot", "quant"], default="pilot")
    ap.add_argument("--mat-dir", type=str, required=True)
    ap.add_argument("--fin-annotations-csv", type=str, required=True)
    ap.add_argument("--all-annotations-csv", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--layer-preset", type=str, choices=["last", "late", "hierarchical"], default="last")
    ap.add_argument("--prediction-threshold", type=float, default=0.5)
    ap.add_argument("--cam-threshold", type=float, default=0.6)
    ap.add_argument("--integrated-gradients-steps", type=int, default=32)
    ap.add_argument("--occlusion-window", type=str, default="8,8")
    ap.add_argument("--occlusion-stride", type=str, default="4,4")
    ap.add_argument("--pilot-per-bucket", type=int, default=10)
    ap.add_argument("--pilot-per-context", type=int, default=8)
    ap.add_argument("--pilot-random-positive", type=int, default=12)
    ap.add_argument("--pilot-negative-limit", type=int, default=24)
    ap.add_argument("--quant-max-positive", type=int, default=0, help="0 means use all positives")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gallery-limit-per-group", type=int, default=8)
    args = ap.parse_args()

    methods = _parse_methods(args.methods)
    checkpoint_specs = [_parse_checkpoint_spec(raw) for raw in args.checkpoint]
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mat_lookup = build_mat_lookup(args.mat_dir)
    fin_annotations = load_localized_annotations(args.fin_annotations_csv)
    if args.mode == "pilot":
        positive_annotations = select_pilot_positive_annotations(
            fin_annotations,
            per_bucket=int(args.pilot_per_bucket),
            per_context=int(args.pilot_per_context),
            remainder_limit=int(args.pilot_random_positive),
            seed=int(args.seed),
        )
    else:
        positives = [row for row in fin_annotations if row.species == "Bp"]
        if int(args.quant_max_positive) > 0:
            positive_annotations = sorted(_sample_rows(positives, int(args.quant_max_positive), random.Random(args.seed)), key=_annotation_sort_key)
        else:
            positive_annotations = sorted(positives, key=_annotation_sort_key)

    negative_annotations: List[Any] = []
    if args.all_annotations_csv:
        all_annotations = load_localized_annotations(args.all_annotations_csv)
        if args.mode == "pilot":
            negative_annotations = select_negative_annotations(all_annotations, limit=int(args.pilot_negative_limit), seed=int(args.seed))

    overall_rows: List[Dict[str, Any]] = []
    by_bucket_rows: List[Dict[str, Any]] = []
    by_context_rows: List[Dict[str, Any]] = []
    localization_rows: List[Dict[str, Any]] = []
    negative_rows: List[Dict[str, Any]] = []

    occ_window = tuple(int(token) for token in args.occlusion_window.split(","))
    occ_stride = tuple(int(token) for token in args.occlusion_stride.split(","))

    run_manifest: Dict[str, Any] = {
        "mode": args.mode,
        "methods": methods,
        "mat_dir": str(Path(args.mat_dir).resolve()),
        "fin_annotations_csv": str(Path(args.fin_annotations_csv).resolve()),
        "all_annotations_csv": str(Path(args.all_annotations_csv).resolve()) if args.all_annotations_csv else None,
        "models": [],
    }

    for model_label, checkpoint_path in checkpoint_specs:
        model, checkpoint_meta = load_model_checkpoint(checkpoint_path, device=device)
        training_args = checkpoint_meta.get("training_args", {}) or {}
        target_layers = resolve_target_layers(model, layer_preset=args.layer_preset)
        run_manifest["models"].append(
            {
                "label": model_label,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "architecture": checkpoint_meta.get("architecture"),
                "val_metrics": checkpoint_meta.get("val_metrics"),
            }
        )

        for annotation in positive_annotations:
            mat_path = _find_mat_path(annotation, mat_lookup)
            if mat_path is None:
                continue
            input_tensor, crop_spec, crop_image = build_annotation_crop(annotation, mat_path=mat_path, training_args=training_args)
            input_tensor = input_tensor.to(device)
            crop_area = int(crop_image.shape[0] * crop_image.shape[1])
            for method in methods:
                artifacts = generate_attention_artifacts(
                    method_name=method,
                    model=model,
                    input_tensor=input_tensor,
                    target_layers=target_layers,
                    target_class=1,
                    threshold_rel=float(args.cam_threshold),
                    integrated_gradients_steps=int(args.integrated_gradients_steps),
                    occlusion_window=occ_window,
                    occlusion_stride=occ_stride,
                )
                stem = f"{model_label}_{method}_{annotation.annotation_id}"
                arrays = save_attention_arrays(
                    artifacts=artifacts,
                    output_root=output_dir / "arrays" / model_label / method,
                    stem=stem,
                )
                panel_path = output_dir / "gallery" / model_label / method / f"{stem}.png"
                render_attention_panel(
                    crop_image=crop_image,
                    artifacts=artifacts,
                    crop_spec=crop_spec,
                    method_name=method,
                    model_label=model_label,
                    output_path=panel_path,
                )
                row = summarize_localization(crop_spec, artifacts, method_name=method, model_label=model_label)
                row.update(
                    {
                        "panel_png": str(panel_path),
                        "heatmap_npy": arrays["heatmap_npy"],
                        "mask_npy": arrays["mask_npy"],
                    }
                )
                row["failure_mode"] = _bucket_failure_mode(row, crop_area)
                localization_rows.append(row)

        for annotation in negative_annotations:
            mat_path = _find_mat_path(annotation, mat_lookup)
            if mat_path is None:
                continue
            input_tensor, crop_spec, crop_image = build_negative_crop_from_annotation(annotation, mat_path=mat_path, training_args=training_args)
            input_tensor = input_tensor.to(device)
            for method in methods:
                artifacts = generate_attention_artifacts(
                    method_name=method,
                    model=model,
                    input_tensor=input_tensor,
                    target_layers=target_layers,
                    target_class=1,
                    threshold_rel=float(args.cam_threshold),
                    integrated_gradients_steps=int(args.integrated_gradients_steps),
                    occlusion_window=occ_window,
                    occlusion_stride=occ_stride,
                )
                stem = f"{model_label}_{method}_{annotation.annotation_id}"
                panel_path = output_dir / "gallery_negative" / model_label / method / f"{stem}.png"
                render_attention_panel(
                    crop_image=crop_image,
                    artifacts=artifacts,
                    crop_spec=crop_spec,
                    method_name=method,
                    model_label=model_label,
                    output_path=panel_path,
                )
                negative_rows.append(
                    {
                        "model_label": model_label,
                        "method": method,
                        "annotation_id": annotation.annotation_id,
                        "filename": annotation.filename,
                        "species": annotation.species,
                        "call_type_bucket": annotation.call_type_bucket,
                        "context_tags": "|".join(annotation.context_tags),
                        "score": artifacts.score,
                        "panel_png": str(panel_path),
                    }
                )

    write_csv(output_dir / "localization_rows.csv", localization_rows)
    if negative_rows:
        write_csv(output_dir / "negative_rows.csv", negative_rows)

    grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in localization_rows:
        grouped_rows[(row["model_label"], row["method"])].append(row)

    for (model_label, method), rows in sorted(grouped_rows.items()):
        overall = aggregate_metric_rows(rows, "method")[0]
        overall["model_label"] = model_label
        overall["method"] = method
        overall_rows.append(overall)
        for bucket_row in aggregate_metric_rows(rows, "call_type_bucket"):
            bucket_row["model_label"] = model_label
            bucket_row["method"] = method
            by_bucket_rows.append(bucket_row)
        expanded_rows: List[Dict[str, Any]] = []
        for row in rows:
            tags = [tag for tag in str(row.get("context_tags", "")).split("|") if tag]
            if not tags:
                tags = ["unknown_other"]
            for tag in tags:
                expanded = dict(row)
                expanded["context_tag"] = tag
                expanded_rows.append(expanded)
        for ctx_row in aggregate_metric_rows(expanded_rows, "context_tag"):
            ctx_row["model_label"] = model_label
            ctx_row["method"] = method
            by_context_rows.append(ctx_row)

    ranking_rows = _method_ranking_rows(overall_rows)
    gallery_groups = _gallery_groups(
        localization_rows,
        negative_rows,
        prediction_threshold=float(args.prediction_threshold),
        limit_per_group=int(args.gallery_limit_per_group),
    )

    write_csv(output_dir / "localization_overall.csv", overall_rows)
    write_csv(output_dir / "localization_by_bucket.csv", by_bucket_rows)
    write_csv(output_dir / "localization_by_context.csv", by_context_rows)
    write_csv(output_dir / "method_ranking.csv", ranking_rows)
    write_json(output_dir / "gallery_manifest.json", gallery_groups)
    write_json(output_dir / "run_manifest.json", run_manifest)
    _write_markdown_summary(
        output_path=output_dir / f"{args.mode}_summary.md",
        title=f"Fin Whale Attention {args.mode.title()} Summary",
        overall_rows=overall_rows,
        ranking_rows=ranking_rows,
    )
    _write_lines(output_dir / "recommendation_draft.md", _draft_recommendation(ranking_rows))

    failure_rows: List[Dict[str, Any]] = []
    for row in localization_rows:
        if row.get("failure_mode"):
            failure_rows.append(
                {
                    "model_label": row["model_label"],
                    "method": row["method"],
                    "annotation_id": row["annotation_id"],
                    "filename": row["filename"],
                    "call_type_bucket": row["call_type_bucket"],
                    "context_tags": row["context_tags"],
                    "failure_mode": row["failure_mode"],
                    "score": row["score"],
                    "box_iou": row["box_iou"],
                    "temporal_iou": row["temporal_iou"],
                    "frequency_iou": row["frequency_iou"],
                    "panel_png": row["panel_png"],
                }
            )
    if failure_rows:
        write_csv(output_dir / "failure_analysis.csv", failure_rows)

    methods_note = [
        "# Methods Note",
        "",
        f"- Mode: `{args.mode}`",
        f"- Models: {', '.join(label for label, _ in checkpoint_specs)}",
        f"- Methods: {', '.join(methods)}",
        f"- Target layers: `{args.layer_preset}`",
        f"- Positive annotations evaluated: {len(positive_annotations)}",
        f"- Negative annotations evaluated: {len(negative_annotations)}",
        "",
        "Localization targets use annotation start/end times plus frequency bounds when present.",
        f"Fallback bucket priors: `{json.dumps(FIN_BUCKET_FREQ_PRIORS, sort_keys=True)}`",
        "",
        "Large-scale quantitative ranking uses box IoU, pointing accuracy, and mask coverage.",
    ]
    _write_lines(output_dir / "methods_note.md", methods_note)


if __name__ == "__main__":
    main()
