#!/usr/bin/env python3
"""Unified prediction tracker for O3 schema-compatible JSON output.

This module provides :class:`UnifiedPredictionTracker`, a small persistence
layer for building predictions and verification files that match the
OCEANS3 unified JSON format (`schema_version: "2.1"`).

Design goals:
1. Produce one canonical JSON shape for both model-assisted and manual flows.
2. Accept legacy producer inputs without breaking existing pipelines.
3. Normalize and sanitize output on save so emitted JSON stays schema-friendly.

Typical usage:
    tracker = UnifiedPredictionTracker("predictions.json")
    tracker.set_task_type("whale_detection")
    tracker.set_model_info(model_id="sha256-abc123", architecture="resnet18")
    tracker.add_data_source(
        data_source_id="ICLISTENHF1353_CLAYO_2019",
        device_code="ICLISTENHF1353",
        location_name="Clayoquot Slope",
    )
    tracker.add_item(
        item_id="ICLISTENHF1353_20190630T000458.000Z",
        data_source_id="ICLISTENHF1353_CLAYO_2019",
        audio_start_time="2019-06-30T00:04:58Z",
        audio_end_time="2019-06-30T00:05:38Z",
        model_outputs=[{
            "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
            "score": 0.87,
        }],
    )
    tracker.save()
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string, accepting both `Z` and `+00:00`.

    Args:
        value: Input datetime string.

    Returns:
        A timezone-aware ``datetime`` when parsing succeeds, otherwise ``None``.
    """
    if not value:
        return None
    try:
        # Handle both "...Z" and "+00:00" style strings.
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_str(value: Any) -> Optional[str]:
    """Convert a value to stripped string, returning ``None`` for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_float(value: Any) -> Optional[float]:
    """Best-effort float conversion, returning ``None`` on invalid input."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_annotation_extent(extent: Any) -> Optional[Dict[str, Any]]:
    """Validate and normalize an ``annotation_extent`` object.

    The result is constrained to O3-supported extent types:
    ``clip``, ``time_range``, ``freq_range``, and ``time_freq_box``.
    """
    if not isinstance(extent, dict):
        return None
    extent_type = _as_str(extent.get("type"))
    if extent_type not in {"clip", "time_range", "freq_range", "time_freq_box"}:
        return None
    cleaned: Dict[str, Any] = {"type": extent_type}
    for key in ("time_start_sec", "time_end_sec", "freq_min_hz", "freq_max_hz"):
        value = _as_float(extent.get(key))
        if value is not None:
            cleaned[key] = value
    if extent_type == "time_range" and not {"time_start_sec", "time_end_sec"}.issubset(cleaned):
        return None
    if extent_type == "freq_range" and not {"freq_min_hz", "freq_max_hz"}.issubset(cleaned):
        return None
    if extent_type == "time_freq_box" and not {
        "time_start_sec",
        "time_end_sec",
        "freq_min_hz",
        "freq_max_hz",
    }.issubset(cleaned):
        return None
    return cleaned


def _clean_source_audio(source_audio: Any) -> Optional[Dict[str, Any]]:
    """Normalize source audio to canonical dict form.

    Accepts either:
    - a plain string filename/path, or
    - a dict containing at least ``file_name``.
    """
    if isinstance(source_audio, str):
        source_name = source_audio.strip()
        if source_name:
            return {"file_name": Path(source_name).name}
        return None
    if not isinstance(source_audio, dict):
        return None
    file_name = _as_str(source_audio.get("file_name"))
    if not file_name:
        return None
    cleaned: Dict[str, Any] = {"file_name": file_name}
    for key in ("format", "uri", "recording_start_time", "recording_end_time", "checksum_sha256"):
        value = _as_str(source_audio.get(key))
        if value is not None:
            cleaned[key] = value
    return cleaned


class UnifiedPredictionTracker:
    """Build, normalize, and persist O3 unified prediction/verification JSON.

    The tracker intentionally accepts some legacy arguments for compatibility.
    Regardless of input shape, ``save()`` emits a normalized structure aligned
    with the O3 v2.1 schema used by the verification app.
    """

    VERSION = "2.1"

    def __init__(self, output_path: Union[str, Path]):
        """Initialize an empty tracker bound to an output file path.

        Args:
            output_path: Destination JSON path for ``save()`` and ``load()``.
        """
        self.output_path = Path(output_path)
        self.data: Dict[str, Any] = {
            "schema_version": self.VERSION,
            "created_at": None,
            "updated_at": None,
            "task_type": None,
            "model": {},
            "data_sources": [],
            "spectrogram_config": {},
            "pipeline": {},
            "items": [],
        }

    def set_model_info(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        architecture: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_url: Optional[str] = None,
        trained_at: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        training_dataset_id: Optional[str] = None,
        training_dataset_version: Optional[str] = None,
        training_dataset_url: Optional[str] = None,
        training_data_time_range: Optional[str] = None,
        input_shape: Optional[List[int]] = None,
        output_classes: Optional[List[str]] = None,
    ) -> None:
        """Set model metadata at the root ``model`` object.

        Args:
            model_id: Stable model identifier (typically weight hash).
            model_version: Optional human-readable model version tag.
            architecture: Model architecture name.
            checkpoint_path: Optional path to model checkpoint.
            checkpoint_url: Optional URL for model artifact retrieval.
            trained_at: Optional model training timestamp.
            wandb_run_id: Optional W&B run identifier.
            training_dataset_id: Optional training dataset identifier.
            training_dataset_version: Optional training dataset version.
            training_dataset_url: Optional URL to the training dataset.
            training_data_time_range: Optional ISO-8601 interval describing
                source-data time coverage.
            input_shape: Optional expected input shape.
            output_classes: Optional list of model output label paths.
        """
        model: Dict[str, Any] = {"model_id": model_id}
        optional = {
            "model_version": model_version,
            "architecture": architecture,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_url": checkpoint_url,
            "trained_at": trained_at,
            "wandb_run_id": wandb_run_id,
            "training_dataset_id": training_dataset_id,
            "training_dataset_version": training_dataset_version,
            "training_dataset_url": training_dataset_url,
            "training_data_time_range": training_data_time_range,
            "input_shape": input_shape,
            "output_classes": output_classes,
        }
        for key, value in optional.items():
            if value is not None:
                model[key] = value
        self.data["model"] = model

    def add_data_source(
        self,
        data_source_id: str,
        device_code: str,
        deployment_id: Optional[str] = None,
        location_name: Optional[str] = None,
        site_code: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        depth_m: Optional[float] = None,
        channel: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sample_rate: Optional[float] = None,
        is_calibrated: Optional[bool] = None,
        calibration_reference: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Append one data source entry under ``data_sources``.

        Args:
            data_source_id: Unique data source key in this file.
            device_code: Device identifier (for example ONC hydrophone code).
            deployment_id: Optional deployment identifier.
            location_name: Optional human-readable location.
            site_code: Optional site code.
            latitude: Optional latitude (decimal degrees).
            longitude: Optional longitude (decimal degrees).
            depth_m: Optional depth in meters.
            channel: Optional channel identifier.
            date_from: Optional source availability start (ISO datetime).
            date_to: Optional source availability end (ISO datetime).
            sample_rate: Optional sample rate in Hz.
            is_calibrated: Optional calibration state flag.
            calibration_reference: Optional calibration reference text.
            **kwargs: Additional forward-compatible source fields.
        """
        source: Dict[str, Any] = {
            "data_source_id": data_source_id,
            "device_code": device_code,
        }
        optional = {
            "deployment_id": deployment_id,
            "location_name": location_name,
            "site_code": site_code,
            "latitude": latitude,
            "longitude": longitude,
            "depth_m": depth_m,
            "channel": channel,
            "date_from": date_from,
            "date_to": date_to,
            "sample_rate": sample_rate,
            "is_calibrated": is_calibrated,
            "calibration_reference": calibration_reference,
        }
        for key, value in optional.items():
            if value is not None:
                source[key] = value
        for key, value in kwargs.items():
            if value is not None:
                source[key] = value
        self.data["data_sources"].append(source)

    def set_data_source(
        self,
        device_code: str,
        location: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Compatibility wrapper for older single-data-source call sites.

        This resets ``data_sources`` and inserts one source entry.
        """
        data_source_id = kwargs.pop("data_source_id", device_code)
        self.data["data_sources"] = []
        self.add_data_source(
            data_source_id=data_source_id,
            device_code=device_code,
            location_name=location,
            date_from=date_from,
            date_to=date_to,
            sample_rate=sample_rate,
            **kwargs,
        )

    def set_spectrogram_config(self, config: Dict[str, Any]) -> None:
        """Set root-level ``spectrogram_config`` metadata."""
        self.data["spectrogram_config"] = config

    def set_pipeline_info(
        self,
        pipeline_version: Optional[str] = None,
        pipeline_commit: Optional[str] = None,
        pipeline_repo: Optional[str] = None,
    ) -> None:
        """Set root-level inference pipeline provenance metadata."""
        pipeline: Dict[str, Any] = {}
        if pipeline_version is not None:
            pipeline["pipeline_version"] = pipeline_version
        if pipeline_commit is not None:
            pipeline["pipeline_commit"] = pipeline_commit
        if pipeline_repo is not None:
            pipeline["pipeline_repo"] = pipeline_repo
        self.data["pipeline"] = pipeline

    def set_task_type(self, task_type: str) -> None:
        """Set root-level task type.

        Args:
            task_type: Free-form task identifier (e.g., 'whale_detection',
                'anomaly_detection', 'classification', or custom values)
        """
        self.data["task_type"] = task_type

    def add_item(
        self,
        item_id: str,
        model_outputs: List[Dict[str, Any]],
        data_source_id: Optional[str] = None,
        audio_start_time: Optional[str] = None,
        audio_end_time: Optional[str] = None,
        segment_index: Optional[int] = None,
        mat_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        spectrogram_path: Optional[str] = None,
        spectrogram_mat_path: Optional[str] = None,
        spectrogram_png_path: Optional[str] = None,
        audio_timestamp: Optional[str] = None,
        duration_sec: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Add one prediction item.

        This method accepts both canonical O3 fields and selected legacy
        producer arguments. Legacy inputs are mapped into canonical fields
        where possible.

        Notes:
            - ``segment_index`` is accepted for compatibility but not emitted.
            - Unknown ``kwargs`` are ignored by design to keep output strict.
        """
        _ = segment_index
        if data_source_id is None and len(self.data.get("data_sources", [])) == 1:
            data_source_id = self.data["data_sources"][0].get("data_source_id")

        if spectrogram_mat_path is None:
            spectrogram_mat_path = kwargs.pop("spectrogram_mat_path", None) or mat_path
        else:
            kwargs.pop("spectrogram_mat_path", None)
        if spectrogram_png_path is None:
            spectrogram_png_path = kwargs.pop("spectrogram_png_path", None) or spectrogram_path
        else:
            kwargs.pop("spectrogram_png_path", None)

        # Legacy inputs -> schema fields
        if audio_start_time is None:
            audio_start_time = str(audio_timestamp) if audio_timestamp else _as_str(kwargs.pop("t0", None))
        if audio_end_time is None:
            audio_end_time = _as_str(kwargs.pop("t1", None))
        if audio_end_time is None and audio_start_time and duration_sec is not None:
            start_dt = _parse_iso_datetime(audio_start_time)
            if start_dt is not None:
                try:
                    audio_end_time = (start_dt + timedelta(seconds=float(duration_sec))).isoformat()
                except (TypeError, ValueError):
                    audio_end_time = None

        item: Dict[str, Any] = {
            "item_id": item_id,
            "model_outputs": model_outputs or [],
            "verifications": [],
        }
        if data_source_id:
            item["data_source_id"] = data_source_id
        if audio_start_time:
            item["audio_start_time"] = str(audio_start_time)
        if audio_end_time:
            item["audio_end_time"] = str(audio_end_time)

        paths: Dict[str, str] = {}
        if spectrogram_mat_path:
            paths["spectrogram_mat_path"] = str(spectrogram_mat_path)
        if spectrogram_png_path:
            paths["spectrogram_png_path"] = str(spectrogram_png_path)
        if audio_path:
            paths["audio_path"] = str(audio_path)
        if paths:
            item["paths"] = paths

        source_audio = kwargs.pop("source_audio", None)
        if source_audio is not None:
            cleaned_source = _clean_source_audio(source_audio)
            if cleaned_source:
                item["source_audio"] = cleaned_source

        self.data["items"].append(item)

    def add_verification(
        self,
        item_id: str,
        labels: List[str],
        verified_by: str,
        threshold_used: Optional[float] = None,
        confidence: Optional[str] = None,
        notes: str = "",
    ) -> bool:
        """Append a simple accepted-label verification record for an item.

        This helper creates one verification round where all provided labels are
        marked ``decision="accepted"``.

        Args:
            item_id: Target item ID.
            labels: Labels to mark as accepted in this verification round.
            verified_by: Reviewer identifier (recommended: ``"Name <email>"``).
            threshold_used: Threshold associated with decisions, if applicable.
            confidence: Optional reviewer confidence.
            notes: Optional free-form notes.

        Returns:
            ``True`` if the item was found and updated, else ``False``.
        """
        for item in self.data["items"]:
            if item["item_id"] == item_id:
                verification_round = len(item["verifications"]) + 1
                label_decisions = [
                    {
                        "label": str(label),
                        "decision": "accepted",
                        "threshold_used": threshold_used,
                    }
                    for label in (labels or [])
                ]
                verification = {
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                    "verified_by": verified_by,
                    "verification_round": verification_round,
                    "verification_status": "verified",
                    "label_decisions": label_decisions,
                    "confidence": confidence,
                    "notes": notes,
                    "label_source": "expert",
                }
                item["verifications"].append(verification)
                return True
        return False

    def get_items_by_score_threshold(
        self,
        class_hierarchy: str,
        threshold: float,
        above: bool = True,
    ) -> List[Dict]:
        """Filter items by class score threshold.

        Args:
            class_hierarchy: Target class path.
            threshold: Score threshold.
            above: When ``True``, keep items with score >= threshold.
                When ``False``, keep items with score < threshold.
        """
        matches: List[Dict] = []
        for item in self.data["items"]:
            for output in item.get("model_outputs", []):
                if output.get("class_hierarchy") != class_hierarchy:
                    continue
                score = output.get("score", 0)
                if (above and score >= threshold) or (not above and score < threshold):
                    matches.append(item)
                    break
        return matches

    def get_unverified_items(self) -> List[Dict]:
        """Return items that have no verification rounds yet."""
        return [item for item in self.data["items"] if not item.get("verifications")]

    def _normalize_loaded_data(self) -> None:
        """Normalize in-memory data and enforce strict O3 v2.1 field sets.

        This routine:
        - upgrades selected legacy keys,
        - canonicalizes nested structures,
        - drops unsupported/deprecated fields,
        - ensures predictable key layout in ``self.data``.
        """
        if "schema_version" not in self.data and "version" in self.data:
            self.data["schema_version"] = self.data.pop("version")
        if "task_type" not in self.data:
            self.data["task_type"] = None
        if "data_sources" not in self.data:
            old_ds = self.data.pop("data_source", {}) if isinstance(self.data.get("data_source"), dict) else {}
            if old_ds:
                ds_id = old_ds.get("data_source_id") or old_ds.get("device_code") or "default_data_source"
                old_ds["data_source_id"] = ds_id
                self.data["data_sources"] = [old_ds]
            else:
                self.data["data_sources"] = []
        if "pipeline" not in self.data:
            self.data["pipeline"] = {}

        model_allowed = {
            "model_id",
            "model_version",
            "architecture",
            "checkpoint_path",
            "checkpoint_url",
            "trained_at",
            "wandb_run_id",
            "training_dataset_id",
            "training_dataset_version",
            "training_dataset_url",
            "training_data_time_range",
            "input_shape",
            "output_classes",
        }
        data_source_allowed = {
            "data_source_id",
            "device_code",
            "deployment_id",
            "location_name",
            "site_code",
            "latitude",
            "longitude",
            "depth_m",
            "channel",
            "sample_rate",
            "is_calibrated",
            "calibration_reference",
            "date_from",
            "date_to",
        }
        pipeline_allowed = {"pipeline_version", "pipeline_commit", "pipeline_repo"}
        item_allowed = {
            "item_id",
            "data_source_id",
            "audio_start_time",
            "audio_end_time",
            "model_outputs",
            "verifications",
            "source_audio",
            "paths",
        }

        model_obj = self.data.get("model")
        model_clean: Dict[str, Any] = {}
        if isinstance(model_obj, dict):
            for key in model_allowed:
                if key in model_obj and model_obj[key] is not None:
                    model_clean[key] = model_obj[key]
            if not model_clean.get("model_id"):
                model_clean = {}

        data_sources_clean: List[Dict[str, Any]] = []
        raw_data_sources = self.data.get("data_sources")
        if isinstance(raw_data_sources, list):
            for ds in raw_data_sources:
                if not isinstance(ds, dict):
                    continue
                cleaned = {k: ds.get(k) for k in data_source_allowed if ds.get(k) is not None}
                if cleaned.get("data_source_id") and cleaned.get("device_code"):
                    data_sources_clean.append(cleaned)

        spectrogram_clean: Dict[str, Any] = {}
        spectrogram_obj = self.data.get("spectrogram_config")
        if isinstance(spectrogram_obj, dict):
            for key in (
                "nfft",
                "window_function",
                "window_duration_sec",
                "hop_length",
                "overlap",
                "context_duration_sec",
                "crop_size",
            ):
                value = spectrogram_obj.get(key)
                if value is not None:
                    spectrogram_clean[key] = value

            frequency_limits = spectrogram_obj.get("frequency_limits")
            if isinstance(frequency_limits, dict):
                freq_min = _as_float(frequency_limits.get("min"))
                freq_max = _as_float(frequency_limits.get("max"))
                if freq_min is not None and freq_max is not None:
                    spectrogram_clean["frequency_limits"] = {"min": freq_min, "max": freq_max}

            source_obj = spectrogram_obj.get("source")
            if isinstance(source_obj, dict):
                source_clean = {}
                for key in (
                    "type",
                    "generator",
                    "backend",
                    "onc_data_product_code",
                    "onc_data_product_options",
                ):
                    value = source_obj.get(key)
                    if value is not None:
                        source_clean[key] = value
                if source_clean:
                    spectrogram_clean["source"] = source_clean

            audio_source = spectrogram_obj.get("audio_source")
            if isinstance(audio_source, dict):
                audio_source_clean = {}
                for key in ("type", "onc_data_product_code", "format"):
                    value = audio_source.get(key)
                    if value is not None:
                        audio_source_clean[key] = value
                if audio_source_clean:
                    spectrogram_clean["audio_source"] = audio_source_clean

        pipeline_obj = self.data.get("pipeline")
        pipeline_clean = {}
        if isinstance(pipeline_obj, dict):
            for key in pipeline_allowed:
                value = pipeline_obj.get(key)
                if value is not None:
                    pipeline_clean[key] = value

        items_clean: List[Dict[str, Any]] = []
        raw_items = self.data.get("items")
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue

                item_id = _as_str(raw_item.get("item_id"))
                if not item_id:
                    continue

                audio_start_time = _as_str(raw_item.get("audio_start_time")) or _as_str(raw_item.get("audio_timestamp"))
                audio_end_time = _as_str(raw_item.get("audio_end_time"))
                if audio_end_time is None and audio_start_time and raw_item.get("duration_sec") is not None:
                    start_dt = _parse_iso_datetime(audio_start_time)
                    if start_dt is not None:
                        try:
                            audio_end_time = (start_dt + timedelta(seconds=float(raw_item.get("duration_sec")))).isoformat()
                        except (TypeError, ValueError):
                            audio_end_time = None

                model_outputs_clean: List[Dict[str, Any]] = []
                model_outputs = raw_item.get("model_outputs")
                if isinstance(model_outputs, list):
                    for output in model_outputs:
                        if not isinstance(output, dict):
                            continue
                        class_hierarchy = _as_str(output.get("class_hierarchy"))
                        score = _as_float(output.get("score"))
                        if class_hierarchy is None or score is None:
                            continue
                        cleaned_output: Dict[str, Any] = {
                            "class_hierarchy": class_hierarchy,
                            "score": score,
                        }
                        class_id = _as_str(output.get("class_id"))
                        if class_id is not None:
                            cleaned_output["class_id"] = class_id
                        extent = _clean_annotation_extent(output.get("annotation_extent"))
                        if extent is not None:
                            cleaned_output["annotation_extent"] = extent
                        model_outputs_clean.append(cleaned_output)

                verifications_clean: List[Dict[str, Any]] = []
                verifications = raw_item.get("verifications")
                if isinstance(verifications, list):
                    for verification in verifications:
                        if not isinstance(verification, dict):
                            continue
                        decisions_clean: List[Dict[str, Any]] = []
                        decisions = verification.get("label_decisions")
                        if isinstance(decisions, list):
                            for decision in decisions:
                                if not isinstance(decision, dict):
                                    continue
                                label = _as_str(decision.get("label"))
                                decision_type = _as_str(decision.get("decision"))
                                if label is None or decision_type not in {"accepted", "rejected", "added"}:
                                    continue
                                threshold_used = decision.get("threshold_used")
                                if threshold_used is not None:
                                    threshold_used = _as_float(threshold_used)
                                cleaned_decision: Dict[str, Any] = {
                                    "label": label,
                                    "decision": decision_type,
                                    "threshold_used": threshold_used,
                                }
                                decision_extent = _clean_annotation_extent(decision.get("annotation_extent"))
                                if decision_extent is not None:
                                    cleaned_decision["annotation_extent"] = decision_extent
                                decisions_clean.append(cleaned_decision)

                        cleaned_verification: Dict[str, Any] = {
                            "verified_at": _as_str(verification.get("verified_at"))
                            or datetime.now(timezone.utc).isoformat(),
                            "verified_by": _as_str(verification.get("verified_by")) or "unknown",
                            "verification_round": verification.get("verification_round")
                            if isinstance(verification.get("verification_round"), int)
                            else (len(verifications_clean) + 1),
                            "label_decisions": decisions_clean,
                        }
                        for key in (
                            "reviewer_affiliation",
                            "verification_status",
                            "confidence",
                            "notes",
                            "label_source",
                            "taxonomy_version",
                        ):
                            value = verification.get(key)
                            if value is not None:
                                cleaned_verification[key] = value
                        verifications_clean.append(cleaned_verification)

                source_audio_clean = _clean_source_audio(raw_item.get("source_audio"))

                paths_clean = {}
                paths_obj = raw_item.get("paths")
                if isinstance(paths_obj, dict):
                    for key in ("spectrogram_mat_path", "spectrogram_png_path", "audio_path"):
                        value = _as_str(paths_obj.get(key))
                        if value is not None:
                            paths_clean[key] = value
                if not paths_clean:
                    for key, fallback in (
                        ("spectrogram_mat_path", raw_item.get("spectrogram_mat_path") or raw_item.get("mat_path")),
                        ("spectrogram_png_path", raw_item.get("spectrogram_png_path") or raw_item.get("spectrogram_path")),
                        ("audio_path", raw_item.get("audio_path")),
                    ):
                        value = _as_str(fallback)
                        if value is not None:
                            paths_clean[key] = value

                item_clean: Dict[str, Any] = {"item_id": item_id}
                data_source_id = _as_str(raw_item.get("data_source_id"))
                if data_source_id is not None:
                    item_clean["data_source_id"] = data_source_id
                if audio_start_time is not None:
                    item_clean["audio_start_time"] = audio_start_time
                if audio_end_time is not None:
                    item_clean["audio_end_time"] = audio_end_time
                item_clean["model_outputs"] = model_outputs_clean
                item_clean["verifications"] = verifications_clean
                if source_audio_clean is not None:
                    item_clean["source_audio"] = source_audio_clean
                if paths_clean:
                    item_clean["paths"] = paths_clean

                item_clean = {k: item_clean[k] for k in item_allowed if k in item_clean}
                items_clean.append(item_clean)

        self.data = {
            "schema_version": _as_str(self.data.get("schema_version")) or self.VERSION,
            "created_at": _as_str(self.data.get("created_at")),
            "updated_at": _as_str(self.data.get("updated_at")),
            "task_type": _as_str(self.data.get("task_type")),
            "model": model_clean,
            "data_sources": data_sources_clean,
            "spectrogram_config": spectrogram_clean,
            "pipeline": pipeline_clean,
            "items": items_clean,
        }

    def save(self) -> None:
        """Normalize and write the tracker JSON to disk."""
        self._normalize_loaded_data()
        now = datetime.now(timezone.utc).isoformat()
        if self.data["created_at"] is None:
            self.data["created_at"] = now
        self.data["updated_at"] = now

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def load(self) -> None:
        """Load tracker JSON from disk and normalize it in memory."""
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                self.data = json.load(f)
            self._normalize_loaded_data()

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "UnifiedPredictionTracker":
        """Construct a tracker and immediately load existing JSON from disk."""
        tracker = cls(path)
        tracker.load()
        return tracker

    def __len__(self) -> int:
        """Return number of tracked items."""
        return len(self.data["items"])

    def summary(self) -> Dict[str, Any]:
        """Return a compact numeric summary of tracked predictions."""
        items = self.data["items"]
        if not items:
            return {"total": 0}

        all_scores: List[float] = []
        for item in items:
            for output in item.get("model_outputs", []):
                if "score" in output:
                    all_scores.append(output["score"])

        verified = sum(1 for item in items if item.get("verifications"))
        summary = {
            "total_items": len(items),
            "verified": verified,
            "unverified": len(items) - verified,
        }
        if all_scores:
            summary.update(
                {
                    "mean_score": sum(all_scores) / len(all_scores),
                    "min_score": min(all_scores),
                    "max_score": max(all_scores),
                }
            )
        return summary
