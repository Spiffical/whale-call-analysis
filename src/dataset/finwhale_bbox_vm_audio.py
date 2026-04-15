"""Helpers for staging bbox-required raw audio on the ONC VM."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

from .finwhale_bbox_audio_audit import COHORT_2025, COHORT_HISTORICAL


REQUIRED_AUDIO_POLICIES = (
    "current_export_render",
    "centered_40s_event_context",
)

COHORT_STAGE_DIRS = {
    COHORT_HISTORICAL: "clayoquot_2018_2019",
    COHORT_2025: "clayoquot_2025",
}


def load_env_file(path: Path | str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _audio_candidates(audio_root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(audio_root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in {".wav", ".flac"}:
                yield path


def index_audio(audio_root: Path | str) -> Dict[str, Path]:
    root = Path(audio_root)
    if not root.exists():
        return {}
    index: Dict[str, Path] = {}
    for path in _audio_candidates(root):
        index.setdefault(path.name, path)
    return index


def select_required_audio_filenames(
    requirement_df: pd.DataFrame,
    *,
    cohort: str,
    policies: Sequence[str],
) -> list[str]:
    if requirement_df.empty:
        return []
    selected = requirement_df[
        (requirement_df["cohort"].astype(str) == str(cohort))
        & (requirement_df["policy"].astype(str).isin([str(policy) for policy in policies]))
    ].copy()
    if selected.empty:
        return []
    names = {
        str(value).strip()
        for value in selected["required_filename"].astype(str).tolist()
        if str(value).strip()
    }
    return sorted(names)


def _materialize_one_file(source: Path, target: Path, *, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "reused_existing"
    if str(mode) == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy_fallback"
    if str(mode) == "copy":
        shutil.copy2(source, target)
        return "copy"
    raise ValueError(f"Unsupported materialization mode: {mode}")


def materialize_audio_subset(
    filenames: Sequence[str],
    *,
    source_root: Path | str,
    target_dir: Path | str,
    mode: str = "hardlink",
) -> Dict[str, Any]:
    source_index = index_audio(source_root)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    method_counts: Counter[str] = Counter()
    missing_source_names: list[str] = []
    available_names: list[str] = []
    materialized_from_source: list[str] = []

    for filename in sorted({str(name).strip() for name in filenames if str(name).strip()}):
        target_file = target_path / filename
        if target_file.exists() and target_file.stat().st_size > 0:
            method_counts["reused_existing"] += 1
            available_names.append(filename)
            continue

        source_file = source_index.get(filename)
        if source_file is None or not source_file.exists():
            missing_source_names.append(filename)
            continue

        method = _materialize_one_file(source_file, target_file, mode=mode)
        method_counts[method] += 1
        available_names.append(filename)
        materialized_from_source.append(filename)

    return {
        "requested_count": int(len(sorted({str(name).strip() for name in filenames if str(name).strip()}))),
        "available_names": sorted(available_names),
        "available_count": int(len(available_names)),
        "materialized_from_source": sorted(materialized_from_source),
        "materialized_from_source_count": int(len(materialized_from_source)),
        "missing_source_names": sorted(missing_source_names),
        "missing_source_count": int(len(missing_source_names)),
        "method_counts": dict(method_counts),
    }


def download_audio_subset(
    filenames: Sequence[str],
    *,
    target_dir: Path | str,
    onc_token: str,
    show_onc_warnings: bool = False,
) -> Dict[str, Any]:
    try:
        from onc import ONC
    except Exception as exc:  # pragma: no cover - exercised only on the VM.
        raise RuntimeError(
            "Downloading bbox-required audio requires the ONC Python client. "
            f"Import error: {exc}"
        ) from exc

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    client = ONC(onc_token, showWarning=show_onc_warnings)
    client.outPath = str(target_path)

    downloaded: list[str] = []
    failed: list[str] = []
    for filename in sorted({str(name).strip() for name in filenames if str(name).strip()}):
        target_file = target_path / filename
        if target_file.exists() and target_file.stat().st_size > 0:
            downloaded.append(filename)
            continue
        try:
            client.getFile(filename)
        except Exception:
            failed.append(filename)
            continue
        if target_file.exists() and target_file.stat().st_size > 0:
            downloaded.append(filename)
        else:
            failed.append(filename)

    return {
        "requested_count": int(len(sorted({str(name).strip() for name in filenames if str(name).strip()}))),
        "downloaded_names": sorted(downloaded),
        "downloaded_count": int(len(downloaded)),
        "failed_names": sorted(failed),
        "failed_count": int(len(failed)),
    }


def summarize_stage_availability(
    requirement_df: pd.DataFrame,
    *,
    cohort: str,
    policies: Sequence[str],
    target_dir: Path | str,
) -> Dict[str, Any]:
    if requirement_df.empty:
        return {
            "cohort": str(cohort),
            "policies": list(policies),
            "requirement_row_count": 0,
            "missing_requirement_count": 0,
            "unique_required_file_count": 0,
            "unique_available_file_count": 0,
            "missing_by_role": {},
        }

    selected = requirement_df[
        (requirement_df["cohort"].astype(str) == str(cohort))
        & (requirement_df["policy"].astype(str).isin([str(policy) for policy in policies]))
    ].copy()
    if selected.empty:
        return {
            "cohort": str(cohort),
            "policies": list(policies),
            "requirement_row_count": 0,
            "missing_requirement_count": 0,
            "unique_required_file_count": 0,
            "unique_available_file_count": 0,
            "missing_by_role": {},
        }

    stage_root = Path(target_dir)
    selected["exists_in_stage"] = [
        int(bool(str(name).strip()) and (stage_root / str(name).strip()).exists())
        for name in selected["required_filename"].astype(str).tolist()
    ]
    missing_rows = selected[selected["exists_in_stage"] == 0]
    return {
        "cohort": str(cohort),
        "policies": list(policies),
        "requirement_row_count": int(len(selected)),
        "missing_requirement_count": int(len(missing_rows)),
        "unique_required_file_count": int(
            len({str(value).strip() for value in selected["required_filename"].astype(str).tolist() if str(value).strip()})
        ),
        "unique_available_file_count": int(
            len(
                {
                    str(value).strip()
                    for value in selected.loc[selected["exists_in_stage"] == 1, "required_filename"].astype(str).tolist()
                    if str(value).strip()
                }
            )
        ),
        "missing_by_role": dict(Counter(missing_rows["role"].astype(str).tolist())),
    }
