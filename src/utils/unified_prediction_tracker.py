#!/usr/bin/env python3
"""
Unified Predictions Tracker

Standardized JSON format for model predictions across whale detection
and anomaly detection projects. Matches specification in:
labeling-verification-app/docs/predictions_json_format.md

Usage:
    tracker = UnifiedPredictionTracker(output_path='predictions.json')
    tracker.set_model_info(model_id='sha256-abc123', architecture='resnet18')
    
    # For single-class detector (e.g., whale)
    tracker.add_item(
        item_id='seg_000',
        model_outputs=[{
            'class_hierarchy': 'Biophony > Marine mammal > ... > Fin whale',
            'score': 0.87
        }]
    )
    
    # For multi-class detector (e.g., anomaly)
    tracker.add_item(
        item_id='spec_001',
        model_outputs=[
            {'class_hierarchy': 'Anthropophony > Vessel', 'score': 0.45},
            {'class_hierarchy': 'Other > Ambient sound', 'score': 0.89}
        ]
    )
    
    tracker.save()
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class UnifiedPredictionTracker:
    """Manages predictions in unified format for expert verification.
    
    Follows specification from predictions_json_format.md with:
    - Model metadata (model_id hash, architecture, training info)
    - Data source info (device, date range, location)
    - Raw model scores (not thresholded)
    - Hierarchical labels from taxonomy
    - Multi-round verification support
    - Spectrogram provenance (local/custom vs ONC download)
    """
    
    VERSION = "2.0"
    
    def __init__(self, output_path: Union[str, Path]):
        """Initialize tracker.
        
        Args:
            output_path: Path to output JSON file
        """
        self.output_path = Path(output_path)
        self.data: Dict[str, Any] = {
            "version": self.VERSION,
            "created_at": None,
            "updated_at": None,
            "model": {},
            "data_source": {},
            "spectrogram_config": {},
            "task_type": None,  # 'whale_detection' | 'anomaly_detection' | 'classification'
            "items": []
        }
    
    def set_model_info(
        self,
        model_id: str,
        architecture: str,
        checkpoint_path: Optional[str] = None,
        trained_at: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        input_shape: Optional[List[int]] = None,
        output_classes: Optional[List[str]] = None
    ) -> None:
        """Set model metadata.
        
        Args:
            model_id: SHA256 hash of model weights (e.g., 'sha256-abc123')
            architecture: Model architecture name
            checkpoint_path: Path to model checkpoint
            trained_at: ISO timestamp of when model was trained
            wandb_run_id: Optional WandB run ID
            input_shape: Expected input dimensions [freq, time]
            output_classes: List of hierarchical class names the model predicts
        """
        self.data["model"] = {
            "model_id": model_id,
            "architecture": architecture,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "trained_at": trained_at,
            "wandb_run_id": wandb_run_id,
        }
        if input_shape is not None:
            self.data["model"]["input_shape"] = input_shape
        if output_classes is not None:
            self.data["model"]["output_classes"] = output_classes
    
    def set_data_source(
        self,
        device_code: str,
        location: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> None:
        """Set data source information.
        
        Args:
            device_code: ONC device code
            location: Location description
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            sample_rate: Audio sample rate
            **kwargs: Additional metadata
        """
        self.data["data_source"] = {
            "device_code": device_code,
        }
        if location:
            self.data["data_source"]["location"] = location
        if date_from:
            self.data["data_source"]["date_from"] = date_from
        if date_to:
            self.data["data_source"]["date_to"] = date_to
        if sample_rate:
            self.data["data_source"]["sample_rate"] = sample_rate
        if kwargs:
            self.data["data_source"].update(kwargs)
    
    def set_spectrogram_config(self, config: Dict[str, Any]) -> None:
        """Set spectrogram generation configuration.
        
        Args:
            config: Dict with parameters (window_duration, overlap, frequency_limits, etc.).
                Recommended schema supports both local/custom and ONC-downloaded spectrograms:
                {
                  "window_duration": 1.0,
                  "overlap": 0.9,
                  "frequency_limits": {"min": 5, "max": 100},
                  "color_limits": {"min": -60, "max": 0},
                  "crop_size": 96,
                  "source": {
                    "type": "computed" | "onc_download",
                    "provider": "ONC",            # if onc_download
                    "generator": "SpectrogramGenerator",  # if computed
                    "backend": "scipy" | "torch",         # if computed
                    "plot_res": 2,                        # if onc_download (plotRes)
                    "data_product_options": {...}         # optional
                  }
                }
        """
        self.data["spectrogram_config"] = config
    
    def set_task_type(self, task_type: str) -> None:
        """Set task type.
        
        Args:
            task_type: One of 'whale_detection', 'anomaly_detection', 'classification'
        """
        self.data["task_type"] = task_type
    
    def add_item(
        self,
        item_id: str,
        model_outputs: List[Dict[str, Any]],
        mat_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        spectrogram_path: Optional[str] = None,
        audio_timestamp: Optional[str] = None,
        duration_sec: Optional[float] = None,
        **kwargs
    ) -> None:
        """Add a prediction item.
        
        Args:
            item_id: Unique identifier for this item
            model_outputs: List of {class_hierarchy, score, ...} dicts
            mat_path: Path to MAT file
            audio_path: Path to audio file
            spectrogram_path: Path to PNG spectrogram
            audio_timestamp: ISO timestamp
            duration_sec: Duration of audio segment
            **kwargs: Additional metadata
        """
        item = {
            "item_id": item_id,
            "mat_path": mat_path,
            "audio_path": audio_path,
            "spectrogram_path": spectrogram_path,
            "audio_timestamp": audio_timestamp,
            "duration_sec": duration_sec,
            "model_outputs": model_outputs,
            "verifications": []
        }
        # Add any additional metadata
        if kwargs:
            item.update(kwargs)
        
        self.data["items"].append(item)
    
    def add_verification(
        self,
        item_id: str,
        labels: List[str],
        verified_by: str,
        threshold_used: Optional[float] = None,
        confidence: Optional[str] = None,
        notes: str = ""
    ) -> bool:
        """Add expert verification to an item.
        
        Args:
            item_id: ID of item to verify
            labels: List of hierarchical labels (e.g., ['Biophony > ... > Fin whale'])
            verified_by: Email or name of verifier
            threshold_used: Threshold applied to model scores
            confidence: Verifier confidence ('high'|'medium'|'low')
            notes: Optional notes
            
        Returns:
            True if item found and updated, False otherwise
        """
        for item in self.data["items"]:
            if item["item_id"] == item_id:
                verification_round = len(item["verifications"]) + 1
                verification = {
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                    "verified_by": verified_by,
                    "threshold_used": threshold_used,
                    "labels": labels,
                    "confidence": confidence,
                    "notes": notes,
                    "verification_round": verification_round
                }
                item["verifications"].append(verification)
                return True
        return False
    
    def get_items_by_score_threshold(
        self,
        class_hierarchy: str,
        threshold: float,
        above: bool = True
    ) -> List[Dict]:
        """Get items filtered by score threshold for a specific class.
        
        Args:
            class_hierarchy: Full hierarchical class name
            threshold: Score threshold (0-1)
            above: If True, return items >= threshold; else < threshold
            
        Returns:
            List of matching items
        """
        matches = []
        for item in self.data["items"]:
            for output in item.get("model_outputs", []):
                if output.get("class_hierarchy") == class_hierarchy:
                    score = output.get("score", 0)
                    if (above and score >= threshold) or (not above and score < threshold):
                        matches.append(item)
                        break
        return matches
    
    def get_unverified_items(self) -> List[Dict]:
        """Get all items without any verifications."""
        return [item for item in self.data["items"] if not item.get("verifications")]
    
    def save(self) -> None:
        """Save data to JSON file."""
        now = datetime.now(timezone.utc).isoformat()
        if self.data["created_at"] is None:
            self.data["created_at"] = now
        self.data["updated_at"] = now
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load(self) -> None:
        """Load data from JSON file."""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                self.data = json.load(f)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "UnifiedPredictionTracker":
        """Create tracker from existing JSON file.
        
        Args:
            path: Path to existing predictions JSON
            
        Returns:
            UnifiedPredictionTracker with loaded data
        """
        tracker = cls(path)
        tracker.load()
        return tracker
    
    def __len__(self) -> int:
        """Return number of items."""
        return len(self.data["items"])
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        items = self.data["items"]
        if not items:
            return {"total": 0}
        
        # Collect all scores across all classes
        all_scores = []
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
            summary.update({
                "mean_score": sum(all_scores) / len(all_scores),
                "min_score": min(all_scores),
                "max_score": max(all_scores),
            })
        
        return summary
