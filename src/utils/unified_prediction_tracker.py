#!/usr/bin/env python3
"""
Unified Predictions Tracker

Outputs predictions in the same unified v2 format used by
`labeling-verification-app`:
- root: `schema_version`, `task_type`, `model`, `data_sources`, ...
- item: `item_id`, `data_source_id`, `audio_start_time`, `audio_end_time`,
  `model_outputs`, `verifications`, `paths`.

This class keeps compatibility helpers (`set_data_source`, legacy path args) so
existing inference code can migrate without breaking.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Handle both "...Z" and "+00:00" style strings.
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class UnifiedPredictionTracker:
    """Manages predictions in the unified JSON schema."""

    VERSION = "2.1"

    def __init__(self, output_path: Union[str, Path]):
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
        architecture: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        trained_at: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        input_shape: Optional[List[int]] = None,
        output_classes: Optional[List[str]] = None,
    ) -> None:
        model: Dict[str, Any] = {"model_id": model_id}
        optional = {
            "architecture": architecture,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "trained_at": trained_at,
            "wandb_run_id": wandb_run_id,
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
        location_name: Optional[str] = None,
        site_code: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sample_rate: Optional[float] = None,
        **kwargs,
    ) -> None:
        source: Dict[str, Any] = {
            "data_source_id": data_source_id,
            "device_code": device_code,
        }
        optional = {
            "location_name": location_name,
            "site_code": site_code,
            "date_from": date_from,
            "date_to": date_to,
            "sample_rate": sample_rate,
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
        """Compatibility wrapper for older call sites."""
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
        self.data["spectrogram_config"] = config

    def set_pipeline_info(
        self,
        pipeline_version: Optional[str] = None,
        pipeline_commit: Optional[str] = None,
        pipeline_repo: Optional[str] = None,
    ) -> None:
        pipeline: Dict[str, Any] = {}
        if pipeline_version is not None:
            pipeline["pipeline_version"] = pipeline_version
        if pipeline_commit is not None:
            pipeline["pipeline_commit"] = pipeline_commit
        if pipeline_repo is not None:
            pipeline["pipeline_repo"] = pipeline_repo
        self.data["pipeline"] = pipeline

    def set_task_type(self, task_type: str) -> None:
        """Set task type.

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
        """Add a prediction item, accepting both new and legacy arguments."""
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
        if audio_start_time is None and audio_timestamp:
            audio_start_time = str(audio_timestamp)
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
        if segment_index is not None:
            item["segment_index"] = int(segment_index)

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
            if isinstance(source_audio, dict):
                item["source_audio"] = source_audio
            else:
                source_name = str(source_audio).strip()
                if source_name:
                    item["source_audio"] = {"file_name": Path(source_name).name}

        for key, value in kwargs.items():
            if value is not None:
                item[key] = value

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
        return [item for item in self.data["items"] if not item.get("verifications")]

    def _normalize_loaded_data(self) -> None:
        """Normalize legacy keys when loading older prediction files."""
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

        items = self.data.get("items")
        if not isinstance(items, list):
            self.data["items"] = []
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            if "model_outputs" not in item:
                item["model_outputs"] = []
            if "verifications" not in item:
                item["verifications"] = []

            # Convert old flat paths into `paths`.
            if "paths" not in item or not isinstance(item.get("paths"), dict):
                paths: Dict[str, str] = {}
                if item.get("spectrogram_mat_path"):
                    paths["spectrogram_mat_path"] = item.get("spectrogram_mat_path")
                elif item.get("mat_path"):
                    paths["spectrogram_mat_path"] = item.get("mat_path")
                if item.get("spectrogram_png_path"):
                    paths["spectrogram_png_path"] = item.get("spectrogram_png_path")
                elif item.get("spectrogram_path"):
                    paths["spectrogram_png_path"] = item.get("spectrogram_path")
                if item.get("audio_path"):
                    paths["audio_path"] = item.get("audio_path")
                if paths:
                    item["paths"] = paths

            # Convert old audio timestamp fields.
            if "audio_start_time" not in item and item.get("audio_timestamp"):
                item["audio_start_time"] = item.get("audio_timestamp")

            # Normalize legacy string source_audio -> canonical object.
            source_audio = item.get("source_audio")
            if isinstance(source_audio, str) and source_audio:
                item["source_audio"] = {"file_name": Path(source_audio).name}

        # Keep canonical ordering for output.
        canonical = {
            "schema_version": self.data.get("schema_version", self.VERSION),
            "created_at": self.data.get("created_at"),
            "updated_at": self.data.get("updated_at"),
            "task_type": self.data.get("task_type"),
            "model": self.data.get("model", {}),
            "data_sources": self.data.get("data_sources", []),
            "spectrogram_config": self.data.get("spectrogram_config", {}),
            "pipeline": self.data.get("pipeline", {}),
            "items": self.data.get("items", []),
        }
        self.data = canonical

    def save(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if self.data["created_at"] is None:
            self.data["created_at"] = now
        self.data["updated_at"] = now

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def load(self) -> None:
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                self.data = json.load(f)
            self._normalize_loaded_data()

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "UnifiedPredictionTracker":
        tracker = cls(path)
        tracker.load()
        return tracker

    def __len__(self) -> int:
        return len(self.data["items"])

    def summary(self) -> Dict[str, Any]:
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
