#!/usr/bin/env python3
"""Build a supervised Part 2 fine-tuning dataset from a prepared VM bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import soundfile as sf
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sequential_prep import get_processing_params, load_dataset_documentation
from src.dataset.audio import stitch_audio_files
from src.dataset.generator import SpectrogramDatasetGenerator
from src.dataset.negative_sampler import sample_negative_windows_for_file
from src.dataset.part2_annotations import FIN_SPECIES_CODE, parse_filename_timestamp
from src.dataset.reporting import configure_output, print_header, print_status
from src.training.mat_utils import parse_mat_filename


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_selected_clip_names(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    clip_path = Path(path)
    names: set[str] = set()
    with open(clip_path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                names.add(value)
    return names


def _existing_output_path(call_id: str, png_dir: Path, mat_dir: Path) -> Optional[Path]:
    png_path = png_dir / f"{call_id}.png"
    mat_path = mat_dir / f"{call_id}.mat"
    if mat_path.exists():
        return mat_path
    if png_path.exists():
        return png_path
    return None


def _fin_annotations_dataframe(
    *,
    fin_annotations_csv: Path,
    selected_clips: Optional[set[str]],
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    rows = _read_csv(fin_annotations_csv)
    selected_rows: List[Dict[str, object]] = []
    call_inventory: List[Dict[str, object]] = []
    for row in rows:
        if str(row.get("species", "")).strip() != FIN_SPECIES_CODE:
            continue
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        if selected_clips is not None and filename not in selected_clips:
            continue
        clip_ts = parse_filename_timestamp(filename)
        if clip_ts is None:
            continue
        begin = float(row.get("begin_time_s", 0.0))
        end = float(row.get("end_time_s", 0.0))
        selected_rows.append(
            {
                "clip id": filename,
                "begin time (s)": begin,
                "end time (s)": end,
                "date (utc)": clip_ts.isoformat(),
                "Date (UTC)": clip_ts,
                "device_code": filename.split("_")[0],
                "call_type": str(row.get("call_type_bucket", "")).strip(),
                "comments": str(row.get("comments", "")).strip(),
                "context_tags": str(row.get("context_tags", "")).strip(),
            }
        )
        call_inventory.append(
            {
                "filename": filename,
                "begin_time_s": begin,
                "end_time_s": end,
                "call_type_bucket": str(row.get("call_type_bucket", "")).strip(),
                "call_type_raw": str(row.get("call_type_raw", "")).strip(),
                "context_tags": str(row.get("context_tags", "")).strip(),
                "comments": str(row.get("comments", "")).strip(),
            }
        )
    if not selected_rows:
        raise SystemExit("No fin-whale annotations selected for dataset generation.")
    whale_calls = pd.DataFrame(selected_rows)
    whale_calls = whale_calls.sort_values(["Date (UTC)", "clip id", "begin time (s)"]).reset_index(drop=True)
    return whale_calls, call_inventory


def _sample_inventory_rows(
    *,
    dataset_dir: Path,
    clip_manifest_csv: Path,
    call_inventory: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    clip_manifest_rows = {str(row.get("filename", "")).strip(): row for row in _read_csv(clip_manifest_csv)}
    call_lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in call_inventory:
        call_lookup[
            (
                str(row["filename"]),
                f"{float(row['begin_time_s']):.1f}",
                f"{float(row['end_time_s']):.1f}",
            )
        ] = dict(row)

    rows: List[Dict[str, object]] = []
    for rel_dir, label, kind in [("mat_files", 1, "positive"), ("neg_mat_files", 0, "negative")]:
        abs_dir = dataset_dir / rel_dir
        if not abs_dir.exists():
            continue
        for mat_path in sorted(abs_dir.glob("*.mat")):
            src, start_s, dur_s = parse_mat_filename(mat_path.name)
            if not src:
                continue
            clip_row = clip_manifest_rows.get(src, {})
            start_str = f"{float(start_s):.1f}" if start_s is not None else ""
            end_str = f"{float(start_s + dur_s):.1f}" if start_s is not None and dur_s is not None else ""
            call_row = call_lookup.get((src, start_str, end_str))
            clip_ts = parse_filename_timestamp(src)
            rows.append(
                {
                    "relative_path": f"{rel_dir}/{mat_path.name}",
                    "label": int(label),
                    "kind": kind,
                    "source_audio": src,
                    "timestamp": clip_ts.isoformat() if clip_ts is not None else "",
                    "month": clip_ts.strftime("%Y%m") if clip_ts is not None else "",
                    "start_time_s": "" if start_s is None else float(start_s),
                    "duration_s": "" if dur_s is None else float(dur_s),
                    "call_type_bucket": str(call_row.get("call_type_bucket", "")) if call_row else "",
                    "context_tags": str(call_row.get("context_tags", "")) if call_row else str(clip_row.get("context_tags", "")),
                    "fin_call_type_buckets": str(clip_row.get("fin_call_type_buckets", "")),
                    "is_fin_positive": str(clip_row.get("is_fin_positive", "")),
                    "is_annotated_non_fin": str(clip_row.get("is_annotated_non_fin", "")),
                }
            )
    return rows


def _pure_nonfin_clip_rows(
    *,
    clip_manifest_csv: Path,
    selected_clips: Optional[set[str]],
) -> List[Dict[str, str]]:
    rows = _read_csv(clip_manifest_csv)
    selected: List[Dict[str, str]] = []
    for row in rows:
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        if selected_clips is not None and filename not in selected_clips:
            continue
        is_fin_positive = str(row.get("is_fin_positive", "0")).strip() == "1"
        is_annotated_non_fin = str(row.get("is_annotated_non_fin", "0")).strip() == "1"
        if is_annotated_non_fin and not is_fin_positive:
            selected.append(dict(row))
    return selected


def _generate_pure_nonfin_negatives(
    *,
    generator: SpectrogramDatasetGenerator,
    pure_nonfin_rows: Sequence[Dict[str, str]],
    raw_audio_dir: Path,
    output_dir: Path,
    neg_context: float,
    edge_context: float,
    neg_margin: float,
    neg_strategy: str,
    neg_step_seconds: Optional[float],
    max_negatives_per_clip: Optional[int],
    existing_policy: str,
    png_style: str = "test",
    png_scale: int = 3,
    png_cmap: str = "inferno",
    png_pmin: float = 2.0,
    png_pmax: float = 98.0,
) -> Dict[str, Any]:
    neg_png_dir = output_dir / "neg_png_files"
    neg_mat_dir = output_dir / "neg_mat_files"
    neg_png_dir.mkdir(parents=True, exist_ok=True)
    neg_mat_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    failures: List[Dict[str, str]] = []

    for row in pure_nonfin_rows:
        clip_id = str(row.get("filename", "")).strip()
        if not clip_id:
            continue
        device_code = clip_id.split("_")[0]
        audio_path = raw_audio_dir / clip_id
        if not audio_path.exists():
            failures.append({"clip_id": clip_id, "reason": "missing_raw_audio"})
            continue

        try:
            with sf.SoundFile(audio_path) as handle:
                sample_rate = handle.samplerate
                clip_duration = float(len(handle) / sample_rate)
        except Exception as exc:
            failures.append({"clip_id": clip_id, "reason": f"audio_probe_failed: {exc}"})
            continue

        requested = max_negatives_per_clip
        if requested is None:
            requested = max(1, int(clip_duration // max(neg_context, 1.0)))
        windows = sample_negative_windows_for_file(
            clip_id,
            clip_duration,
            neg_context,
            calls_by_file={},
            max_windows=int(requested),
            margin=float(neg_margin),
            strategy=neg_strategy,
            step_seconds=neg_step_seconds,
        )
        for n_idx, (start, end) in enumerate(windows):
            neg_id = f"{clip_id}_neg_purenonfin_{n_idx}"
            if existing_policy == "skip":
                existing = _existing_output_path(neg_id, neg_png_dir, neg_mat_dir)
                if existing is not None:
                    skipped += 1
                    continue
            try:
                ext_context = float(neg_context) + (2.0 * float(edge_context))
                audio_data = stitch_audio_files(
                    generator.onc_token,
                    clip_id,
                    device_code,
                    float(start) - float(edge_context),
                    float(end) + float(edge_context),
                    ext_context,
                    raw_audio_dir,
                    show_onc_warnings=generator.show_onc_warnings,
                    allow_downloads=False,
                )
                if audio_data is None:
                    failures.append({"clip_id": clip_id, "reason": f"stitch_failed:{start:.1f}-{end:.1f}"})
                    continue
                res_path, _ = generator._generate_and_save(
                    audio_data,
                    sample_rate,
                    neg_id,
                    neg_png_dir,
                    neg_mat_dir,
                    edge_context=float(edge_context),
                    target_duration=float(neg_context),
                    png_style=png_style,
                    png_scale=int(png_scale),
                    png_cmap=png_cmap,
                    png_pmin=float(png_pmin),
                    png_pmax=float(png_pmax),
                )
                if res_path is not None:
                    generated += 1
            except Exception as exc:
                failures.append({"clip_id": clip_id, "reason": f"generate_failed:{exc}"})

    return {
        "generated_count": generated,
        "skipped_existing_count": skipped,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Part 2 fine-tuning MAT dataset from a VM bundle")
    ap.add_argument("--bundle-dir", type=str, required=True, help="Prepared Part 2 VM bundle directory")
    ap.add_argument("--dataset-doc", type=str, required=True, help="dataset_documentation.json from the original training dataset")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory for the fine-tune dataset")
    ap.add_argument("--config", type=str, default=str(REPO_ROOT / "config" / "dataset_config.yaml"), help="Dataset config path")
    ap.add_argument("--selected-clips-file", type=str, default=None, help="Optional text file of clip names to include")
    ap.add_argument(
        "--pure-nonfin-clips-file",
        type=str,
        default=None,
        help="Optional text file of pure non-fin clip names to use for extra negative crops",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--negatives-per-call", type=int, default=1)
    ap.add_argument("--neg-margin", type=float, default=2.0)
    ap.add_argument("--neg-strategy", type=str, default="tiled", choices=["random", "tiled"])
    ap.add_argument("--neg-step-seconds", type=float, default=None)
    ap.add_argument("--max-negatives-per-file", type=int, default=None)
    ap.add_argument(
        "--max-negatives-per-nonfin-clip",
        type=int,
        default=None,
        help="Optional cap for extra negatives from pure annotated non-fin clips",
    )
    ap.add_argument("--edge-context", type=float, default=2.0)
    ap.add_argument("--existing-policy", type=str, default="skip", choices=["overwrite", "skip"])
    ap.add_argument(
        "--no-pure-nonfin-negatives",
        dest="include_pure_nonfin_negatives",
        action="store_false",
        help="Skip extra negative crop generation from pure annotated non-fin clips",
    )
    ap.add_argument("--cleanup-audio", action="store_true")
    ap.add_argument("--tar-output", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.set_defaults(include_pure_nonfin_negatives=True)
    args = ap.parse_args()

    configure_output(verbose=True, use_tqdm=not args.no_progress)

    bundle_dir = Path(args.bundle_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifests_dir = bundle_dir / "manifests"
    fin_annotations_csv = manifests_dir / "fin_annotations.csv"
    clip_manifest_csv = manifests_dir / "clip_manifest.csv"
    raw_audio_dir = bundle_dir / "raw_audio"
    if not fin_annotations_csv.exists():
        raise SystemExit(f"Missing fin annotations: {fin_annotations_csv}")
    if not clip_manifest_csv.exists():
        raise SystemExit(f"Missing clip manifest: {clip_manifest_csv}")
    if not raw_audio_dir.exists():
        raise SystemExit(f"Missing raw audio directory: {raw_audio_dir}")

    load_dotenv()
    onc_token = os.getenv("ONC_TOKEN")
    if not onc_token:
        raise SystemExit("ONC_TOKEN is required in the environment or .env for dataset generation.")

    selected_clips = _load_selected_clip_names(args.selected_clips_file)
    selected_pure_nonfin_clips = _load_selected_clip_names(args.pure_nonfin_clips_file)
    whale_calls, call_inventory = _fin_annotations_dataframe(
        fin_annotations_csv=fin_annotations_csv,
        selected_clips=selected_clips,
    )
    pure_nonfin_rows = _pure_nonfin_clip_rows(
        clip_manifest_csv=clip_manifest_csv,
        selected_clips=selected_pure_nonfin_clips if selected_pure_nonfin_clips is not None else selected_clips,
    )

    dataset_doc = load_dataset_documentation(args.dataset_doc)
    proc = get_processing_params(dataset_doc=dataset_doc)
    context_duration = (
        (dataset_doc.get("processing_parameters", {}) or {})
        .get("temporal_context", {})
        .get("context_duration_s", 40.0)
    )

    print_header("BUILDING PART 2 FINE-TUNE DATASET")
    print_status(f"Selected fin calls: {len(whale_calls):,}", "INFO", force=True)
    print_status(f"Selected clips: {whale_calls['clip id'].nunique():,}", "INFO", force=True)
    print_status(f"Pure non-fin clips for extra negatives: {len(pure_nonfin_rows):,}", "INFO", force=True)
    print_status(f"Context duration: {float(context_duration):.1f}s", "INFO", force=True)
    print_status(
        f"Spectrogram params: win_dur={proc['win_dur']} overlap={proc['overlap']} freq={proc['freq_lims']}",
        "INFO",
        force=True,
    )

    generator = SpectrogramDatasetGenerator(
        onc_token=onc_token,
        excel_files=[],
        config_path=args.config,
        show_onc_warnings=False,
    )
    generator.whale_data = whale_calls
    generator.apply_overrides(
        win_dur=float(proc["win_dur"]),
        overlap=float(proc["overlap"]),
        freq_range=tuple(proc["freq_lims"]),
        ml_context=float(context_duration),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    specs, failed, dims = generator.generate_spectrograms(
        whale_calls,
        output_dir,
        show_progress=not args.no_progress,
        max_workers=int(args.workers),
        cleanup_audio=bool(args.cleanup_audio),
        generate_negatives=True,
        negatives_per_call=int(args.negatives_per_call),
        neg_margin=float(args.neg_margin),
        neg_strategy=args.neg_strategy,
        neg_step_seconds=args.neg_step_seconds,
        max_negatives_per_file=args.max_negatives_per_file,
        audio_cache_dir=raw_audio_dir,
        allow_audio_download=False,
        edge_context=float(args.edge_context),
        existing_policy=args.existing_policy,
        ml_context=float(context_duration),
        neg_context=float(context_duration),
    )

    pure_nonfin_neg_summary = {
        "generated_count": 0,
        "skipped_existing_count": 0,
        "failure_count": 0,
        "failures": [],
    }
    if args.include_pure_nonfin_negatives and pure_nonfin_rows:
        print_status("Generating extra negatives from pure annotated non-fin clips", "PROGRESS", force=True)
        pure_nonfin_neg_summary = _generate_pure_nonfin_negatives(
            generator=generator,
            pure_nonfin_rows=pure_nonfin_rows,
            raw_audio_dir=raw_audio_dir,
            output_dir=output_dir,
            neg_context=float(context_duration),
            edge_context=float(args.edge_context),
            neg_margin=float(args.neg_margin),
            neg_strategy=args.neg_strategy,
            neg_step_seconds=args.neg_step_seconds,
            max_negatives_per_clip=args.max_negatives_per_nonfin_clip,
            existing_policy=args.existing_policy,
        )
        if pure_nonfin_neg_summary["failure_count"]:
            print_status(
                f"Pure non-fin negative generation failures: {pure_nonfin_neg_summary['failure_count']}",
                "WARNING",
                force=True,
            )

    sample_inventory = _sample_inventory_rows(
        dataset_dir=output_dir,
        clip_manifest_csv=clip_manifest_csv,
        call_inventory=call_inventory,
    )
    _write_csv(output_dir / "sample_inventory.csv", sample_inventory)
    _write_csv(output_dir / "call_inventory.csv", call_inventory)

    summary = {
        "bundle_dir": str(bundle_dir),
        "dataset_doc": str(Path(args.dataset_doc).resolve()),
        "selected_clip_count": int(whale_calls["clip id"].nunique()),
        "selected_call_count": int(len(whale_calls)),
        "positive_mat_count": sum(1 for row in sample_inventory if int(row["label"]) == 1),
        "negative_mat_count": sum(1 for row in sample_inventory if int(row["label"]) == 0),
        "failed_count": len(failed),
        "pure_nonfin_negative_summary": pure_nonfin_neg_summary,
        "actual_dimensions": list(dims) if dims else None,
        "context_duration_s": float(context_duration),
        "processing_params": proc,
        "selected_clips_file": str(Path(args.selected_clips_file).resolve()) if args.selected_clips_file else None,
        "pure_nonfin_clips_file": str(Path(args.pure_nonfin_clips_file).resolve()) if args.pure_nonfin_clips_file else None,
    }
    with open(output_dir / "fine_tune_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if failed:
        with open(output_dir / "fine_tune_failures.json", "w", encoding="utf-8") as handle:
            json.dump(failed, handle, indent=2, sort_keys=True)
    if pure_nonfin_neg_summary["failure_count"]:
        with open(output_dir / "fine_tune_nonfin_failures.json", "w", encoding="utf-8") as handle:
            json.dump(pure_nonfin_neg_summary["failures"], handle, indent=2, sort_keys=True)

    print_status(f"Positive MATs: {summary['positive_mat_count']:,}", "SUCCESS", force=True)
    print_status(f"Negative MATs: {summary['negative_mat_count']:,}", "SUCCESS", force=True)
    print_status(f"Sample inventory: {output_dir / 'sample_inventory.csv'}", "SUCCESS", force=True)
    if failed:
        print_status(f"Failures logged: {output_dir / 'fine_tune_failures.json'}", "WARNING", force=True)

    if args.tar_output:
        tar_path = output_dir / "all_mat_files.tar"
        import tarfile

        print_status(f"Creating tar archive: {tar_path}", "PROGRESS", force=True)
        with tarfile.open(tar_path, "w") as tar:
            for dirname in ["mat_files", "neg_mat_files"]:
                dir_path = output_dir / dirname
                if dir_path.exists():
                    tar.add(str(dir_path), arcname=dirname)
        print_status(f"Archive ready: {tar_path}", "SUCCESS", force=True)


if __name__ == "__main__":
    main()
