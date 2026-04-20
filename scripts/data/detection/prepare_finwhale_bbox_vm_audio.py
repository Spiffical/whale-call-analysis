#!/usr/bin/env python3
"""Prepare bbox-required raw audio caches on the ONC VM.

This script is intended for execution on `oncvm` near the mounted Clayoquot
audio share. It:

1. Builds the unified bbox manifests and split assignments.
2. Audits the exact raw-audio requirements for the current bbox export policy.
3. Materializes separate historical and 2025 raw-audio trees on the mounted
   drive using hardlinks or copies from the local cache when possible.
4. Optionally downloads any remaining missing clips from ONC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.finwhale_bbox import (
    ANNOTATIONS_2025_WORKBOOK_DEFAULT,
    HISTORICAL_WORKBOOK_DEFAULT,
    MAR18_WORKBOOK_DEFAULT,
    MAR26_WORKBOOK_DEFAULT,
    build_bbox_splits,
    build_joint_bbox_manifests,
    write_bbox_splits,
    write_joint_bbox_manifests,
)
from src.dataset.finwhale_bbox_audio_audit import (
    AudioAuditConfig,
    COHORT_2025,
    COHORT_HISTORICAL,
    audit_audio_requirements,
    write_audio_audit,
)
from src.dataset.finwhale_bbox_vm_audio import (
    COHORT_STAGE_DIRS,
    REQUIRED_AUDIO_POLICIES,
    download_audio_subset,
    load_env_file,
    materialize_audio_subset,
    select_missing_required_audio_filenames,
    select_required_audio_filenames,
    summarize_stage_availability,
)


DEFAULT_VM_HIST_AUDIO_DIR = "/home/sbialek/whalestor_mount/FinWhalesProject/data/audio"
DEFAULT_VM_2025_AUDIO_DIR = "/home/sbialek/whalestor_mount/FinWhalesProject/data/finwhale_part2_bundle/raw_audio"
DEFAULT_VM_OUTPUT_ROOT = "/home/sbialek/whalestor_mount/FinWhalesProject/data/finwhale_bbox_audio"


def _log(message: str, status: str = "INFO") -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [{status}] {message}", flush=True)


def _write_name_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _cohort_source_dir(cohort: str, *, historical_audio_dir: Path, audio_2025_dir: Path) -> Path:
    if str(cohort) == COHORT_HISTORICAL:
        return historical_audio_dir
    return audio_2025_dir


def _cohort_download_roles(cohort: str) -> list[str]:
    if str(cohort) == COHORT_HISTORICAL:
        return ["main"]
    return ["main", "prev", "next"]


def main() -> None:
    load_env_file(REPO_ROOT / ".env")

    ap = argparse.ArgumentParser(description="Prepare bbox-required raw audio caches on the ONC VM")
    ap.add_argument("--historical-workbook", type=str, default=HISTORICAL_WORKBOOK_DEFAULT)
    ap.add_argument(
        "--annotations-2025-workbook",
        "--species-temporal-workbook",
        dest="annotations_2025_workbook",
        type=str,
        default=ANNOTATIONS_2025_WORKBOOK_DEFAULT,
    )
    ap.add_argument("--mar26-workbook", type=str, default=MAR26_WORKBOOK_DEFAULT)
    ap.add_argument("--mar18-workbook", type=str, default=MAR18_WORKBOOK_DEFAULT)
    ap.add_argument("--historical-source-audio-dir", type=str, default=DEFAULT_VM_HIST_AUDIO_DIR)
    ap.add_argument("--audio-2025-source-dir", type=str, default=DEFAULT_VM_2025_AUDIO_DIR)
    ap.add_argument("--output-root", type=str, default=DEFAULT_VM_OUTPUT_ROOT)
    ap.add_argument(
        "--required-policies",
        nargs="+",
        default=["current_export_render"],
        choices=list(REQUIRED_AUDIO_POLICIES),
        help="Audio requirement policies to materialize into the staged raw-audio caches",
    )
    ap.add_argument(
        "--materialization-mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy"],
        help="How to populate staged raw-audio trees from the mounted source cache",
    )
    ap.add_argument(
        "--download-missing-audio",
        action="store_true",
        help="Download remaining missing files from ONC into the staged raw-audio directories",
    )
    ap.add_argument("--onc-token-env", type=str, default="ONC_TOKEN")
    ap.add_argument("--show-onc-warnings", action="store_true")
    ap.add_argument("--context-duration-s", type=float, default=40.0)
    ap.add_argument("--clip-duration-s", type=float, default=300.0)
    ap.add_argument("--edge-buffer-s", type=float, default=2.0)
    ap.add_argument("--pure-zero-ratio", type=float, default=0.5)
    ap.add_argument("--negative-margin-s", type=float, default=2.0)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    manifests_dir = output_root / "manifests" / "joint_v1"
    splits_dir = output_root / "splits" / "joint_v1"
    audit_dir = output_root / "audit"
    historical_audio_dir = Path(args.historical_source_audio_dir)
    audio_2025_dir = Path(args.audio_2025_source_dir)

    _log(f"Historical source audio dir: {historical_audio_dir}")
    _log(f"2025 source audio dir: {audio_2025_dir}")
    _log(f"Output root: {output_root}")
    _log(f"Required policies: {', '.join(args.required_policies)}")
    _log(f"Materialization mode: {args.materialization_mode}")
    _log(f"Download missing audio: {bool(args.download_missing_audio)}")

    manifests = build_joint_bbox_manifests(
        historical_workbook=args.historical_workbook,
        species_temporal_workbook=args.annotations_2025_workbook,
        mar26_workbook=args.mar26_workbook,
        mar18_workbook=args.mar18_workbook,
    )
    written_manifests = write_joint_bbox_manifests(manifests_dir, manifests)
    _log(f"Wrote manifests to {manifests_dir}", "SUCCESS")

    split_data = build_bbox_splits(
        manifests["annotations"],
        manifests["clip_manifest"],
    )
    written_splits = write_bbox_splits(splits_dir, split_data)
    _log(f"Wrote splits to {splits_dir}", "SUCCESS")

    audit = audit_audio_requirements(
        annotation_manifest_csv=written_manifests["annotations"],
        clip_manifest_csv=written_manifests["clip_manifest"],
        split_assignments_csv=written_splits["assignments"],
        config=AudioAuditConfig(
            historical_audio_dir=historical_audio_dir,
            audio_2025_dir=audio_2025_dir,
            context_duration_s=float(args.context_duration_s),
            clip_duration_s=float(args.clip_duration_s),
            edge_buffer_s=float(args.edge_buffer_s),
            pure_zero_ratio=float(args.pure_zero_ratio),
            negative_margin_s=float(args.negative_margin_s),
        ),
    )
    written_audit = write_audio_audit(audit_dir, audit)
    _log(f"Wrote audio audit to {audit_dir}", "SUCCESS")

    requirement_df = audit["requirement_df"]
    overall_summary: dict[str, object] = {
        "required_policies": list(args.required_policies),
        "materialization_mode": str(args.materialization_mode),
        "historical_source_audio_dir": str(historical_audio_dir),
        "audio_2025_source_dir": str(audio_2025_dir),
        "output_root": str(output_root),
        "manifest_summary": manifests["summary"],
        "split_summary": split_data["summary"],
        "audit_summary": audit["summary"],
        "cohorts": {},
    }

    onc_token = os.getenv(args.onc_token_env, "").strip()
    if args.download_missing_audio and not onc_token:
        raise SystemExit(
            f"{args.onc_token_env} is required when --download-missing-audio is used. "
            "Set it in the environment or repo .env on oncvm."
        )

    for cohort in (COHORT_HISTORICAL, COHORT_2025):
        stage_dir = output_root / COHORT_STAGE_DIRS[cohort]
        raw_audio_dir = stage_dir / "raw_audio"
        lists_dir = stage_dir / "lists"
        cohort_required = select_required_audio_filenames(
            requirement_df,
            cohort=cohort,
            policies=args.required_policies,
        )
        _write_name_list(lists_dir / "required_filenames.txt", cohort_required)
        _log(
            f"{cohort}: {len(cohort_required):,} unique required files selected for staging into {raw_audio_dir}",
            "PROGRESS",
        )

        materialize_result = materialize_audio_subset(
            cohort_required,
            source_root=_cohort_source_dir(
                cohort,
                historical_audio_dir=historical_audio_dir,
                audio_2025_dir=audio_2025_dir,
            ),
            target_dir=raw_audio_dir,
            mode=args.materialization_mode,
        )
        _write_name_list(
            lists_dir / "materialized_from_source.txt",
            materialize_result["materialized_from_source"],
        )
        _write_name_list(
            lists_dir / "missing_from_source.txt",
            materialize_result["missing_source_names"],
        )

        download_candidate_names = select_missing_required_audio_filenames(
            requirement_df,
            cohort=cohort,
            policies=args.required_policies,
            roles=_cohort_download_roles(cohort),
        )
        skipped_missing_names = sorted(
            set(materialize_result["missing_source_names"]) - set(download_candidate_names)
        )
        _write_name_list(lists_dir / "download_candidates.txt", download_candidate_names)
        _write_name_list(lists_dir / "skipped_missing_download_candidates.txt", skipped_missing_names)

        download_result = {
            "requested_count": 0,
            "downloaded_count": 0,
            "failed_count": 0,
            "downloaded_names": [],
            "failed_names": [],
        }
        if args.download_missing_audio and download_candidate_names:
            _log(
                f"{cohort}: downloading {len(download_candidate_names):,} missing files from ONC",
                "PROGRESS",
            )
            download_result = download_audio_subset(
                download_candidate_names,
                target_dir=raw_audio_dir,
                onc_token=onc_token,
                show_onc_warnings=bool(args.show_onc_warnings),
            )
            _write_name_list(lists_dir / "downloaded_from_onc.txt", download_result["downloaded_names"])
            _write_name_list(lists_dir / "download_failed.txt", download_result["failed_names"])
        else:
            _write_name_list(lists_dir / "downloaded_from_onc.txt", [])
            _write_name_list(lists_dir / "download_failed.txt", [])

        post_stage_summary = summarize_stage_availability(
            requirement_df,
            cohort=cohort,
            policies=args.required_policies,
            target_dir=raw_audio_dir,
        )
        cohort_summary = {
            "stage_dir": str(stage_dir),
            "raw_audio_dir": str(raw_audio_dir),
            "required_file_count": int(len(cohort_required)),
            "materialize_result": materialize_result,
            "download_candidate_count": int(len(download_candidate_names)),
            "download_roles": _cohort_download_roles(cohort),
            "skipped_missing_download_candidate_count": int(len(skipped_missing_names)),
            "download_result": download_result,
            "post_stage_summary": post_stage_summary,
        }
        _write_json(stage_dir / "summary.json", cohort_summary)
        overall_summary["cohorts"][cohort] = cohort_summary
        _log(
            f"{cohort}: post-stage missing requirement rows = "
            f"{post_stage_summary['missing_requirement_count']:,}",
            "SUCCESS" if post_stage_summary["missing_requirement_count"] == 0 else "WARNING",
        )

    _write_json(output_root / "summary.json", overall_summary)
    _log("Prepared fin-whale bbox VM audio staging outputs:", "SUCCESS")
    _log(f"  manifests: {manifests_dir}", "INFO")
    _log(f"  splits: {splits_dir}", "INFO")
    _log(f"  audit: {audit_dir}", "INFO")
    for cohort in (COHORT_HISTORICAL, COHORT_2025):
        _log(f"  {cohort}: {output_root / COHORT_STAGE_DIRS[cohort]}", "INFO")


if __name__ == "__main__":
    main()
