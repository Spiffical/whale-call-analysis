#!/usr/bin/env python3
"""
Prediction Tracker

JSON-based prediction and metadata management for expert review dashboard.
Tracks model versioning, data source info, spectrogram config, and predictions.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PredictionTracker:
    """Manages predictions and metadata for expert review.
    
    Maintains a standardized JSON structure with:
    - Model versioning info (model_id, architecture, wandb_run_id)
    - Data source info (device_code, date range)
    - Spectrogram configuration parameters
    - List of predictions with confidence scores and paths
    """
    
    VERSION = "1.0"
    
    def __init__(self, output_path: Union[str, Path]):
        """Initialize the prediction tracker.
        
        Args:
            output_path: Path to the output JSON file
        """
        self.output_path = Path(output_path)
        self.data: Dict[str, Any] = {
            "version": self.VERSION,
            "created_at": None,
            "model": {},
            "data_source": {},
            "spectrogram_config": {},
            "predictions": []
        }
    
    def set_model_info(
        self,
        checkpoint_path: str,
        model_id: str,
        architecture: str,
        trained_at: Optional[str] = None,
        wandb_run_id: Optional[str] = None
    ) -> None:
        """Set model versioning information.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            model_id: SHA256 hash of model weights
            architecture: Model architecture name
            trained_at: ISO timestamp of when model was trained
            wandb_run_id: Optional WandB run ID
        """
        self.data["model"] = {
            "model_id": model_id,
            "checkpoint_path": str(checkpoint_path),
            "architecture": architecture,
            "trained_at": trained_at,
            "wandb_run_id": wandb_run_id
        }
    
    def set_data_source(
        self,
        device_code: str,
        date_from: str,
        date_to: str,
        additional_info: Optional[Dict] = None
    ) -> None:
        """Set data source information.
        
        Args:
            device_code: ONC device code
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            additional_info: Optional additional metadata
        """
        self.data["data_source"] = {
            "device_code": device_code,
            "date_from": date_from,
            "date_to": date_to
        }
        if additional_info:
            self.data["data_source"].update(additional_info)
    
    def set_spectrogram_config(self, config: Dict[str, Any]) -> None:
        """Set spectrogram generation configuration.
        
        Args:
            config: Dict with spectrogram parameters (window_duration, overlap, etc.)
        """
        self.data["spectrogram_config"] = config
    
    def add_prediction(
        self,
        file_id: str,
        audio_timestamp: str,
        confidence: float,
        mat_path: str,
        spectrogram_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        additional_meta: Optional[Dict] = None
    ) -> None:
        """Add a prediction entry.
        
        Args:
            file_id: Unique identifier for this spectrogram
            audio_timestamp: ISO timestamp of the audio segment
            confidence: Model confidence score (0-1)
            mat_path: Path to MAT file (always saved)
            spectrogram_path: Optional path to PNG spectrogram
            audio_path: Optional path to audio file
            additional_meta: Optional additional metadata
        """
        prediction = {
            "file_id": file_id,
            "audio_timestamp": audio_timestamp,
            "confidence": float(confidence),
            "mat_path": str(mat_path),
            "spectrogram_path": str(spectrogram_path) if spectrogram_path else None,
            "audio_path": str(audio_path) if audio_path else None,
            "expert_label": None,
            "expert_reviewer": None,
            "review_timestamp": None
        }
        if additional_meta:
            prediction.update(additional_meta)
        self.data["predictions"].append(prediction)
    
    def update_expert_label(
        self,
        file_id: str,
        label: str,
        reviewer: Optional[str] = None
    ) -> bool:
        """Update expert label for a prediction.
        
        Args:
            file_id: ID of the prediction to update
            label: Expert-assigned label
            reviewer: Optional reviewer name/ID
            
        Returns:
            True if prediction was found and updated, False otherwise
        """
        for pred in self.data["predictions"]:
            if pred["file_id"] == file_id:
                pred["expert_label"] = label
                pred["expert_reviewer"] = reviewer
                pred["review_timestamp"] = datetime.now(timezone.utc).isoformat()
                return True
        return False
    
    def get_predictions_by_threshold(
        self,
        threshold: float,
        above: bool = True
    ) -> List[Dict]:
        """Get predictions filtered by confidence threshold.
        
        Args:
            threshold: Confidence threshold (0-1)
            above: If True, return predictions >= threshold; else < threshold
            
        Returns:
            List of matching predictions
        """
        if above:
            return [p for p in self.data["predictions"] if p["confidence"] >= threshold]
        return [p for p in self.data["predictions"] if p["confidence"] < threshold]
    
    def get_unreviewed_predictions(self) -> List[Dict]:
        """Get all predictions without expert labels."""
        return [p for p in self.data["predictions"] if p["expert_label"] is None]
    
    def save(self) -> None:
        """Save data to JSON file."""
        self.data["created_at"] = datetime.now(timezone.utc).isoformat()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load(self) -> None:
        """Load data from JSON file."""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                self.data = json.load(f)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PredictionTracker":
        """Create a PredictionTracker from an existing JSON file.
        
        Args:
            path: Path to existing predictions JSON
            
        Returns:
            PredictionTracker instance with loaded data
        """
        tracker = cls(path)
        tracker.load()
        return tracker
    
    def __len__(self) -> int:
        """Return number of predictions."""
        return len(self.data["predictions"])
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        predictions = self.data["predictions"]
        if not predictions:
            return {"total": 0}
        
        confidences = [p["confidence"] for p in predictions]
        reviewed = sum(1 for p in predictions if p["expert_label"] is not None)
        
        return {
            "total": len(predictions),
            "reviewed": reviewed,
            "unreviewed": len(predictions) - reviewed,
            "mean_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        }
