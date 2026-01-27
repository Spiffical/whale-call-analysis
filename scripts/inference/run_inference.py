#!/usr/bin/env python3
"""
Run Model Inference on Sequential Spectrograms

Runs a trained model on processed spectrogram data and saves predictions
to JSON with full versioning metadata for expert review.

Usage:
    python scripts/inference/run_inference.py \
        --mat-dir output/test_windows/spectrograms/2024-01-01/ICLISTENHF1951 \
        --checkpoint checkpoints/best.pt \
        --output-json output/test_windows/predictions.json \
        --dataset-metadata output/test_windows/metadata.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io

from src.models.fin_models import create_model
from src.utils.model_utils import extract_model_info, verify_model_hash, compute_model_hash
from src.utils.unified_prediction_tracker import UnifiedPredictionTracker
from src.dataset.reporting import print_status, print_header


class InferenceDataset(torch.utils.data.Dataset):
    """Dataset for inference on MAT spectrograms with optional sliding window.
    
    Matches the data preparation from FinWhaleMatDataset in training.
    Supports sliding window mode for exhaustive scanning of spectrograms.
    """
    
    # Same keys as training dataset
    SPECTRO_KEYS = ('spectrogram', 'PdB_norm', 'power_db_norm', 'PdB', 'P_db',
                    'P', 'PSD', 'psd', 'Sxx', 'S', 'spec', 'power_spectrogram')
    POWER_KEYS = ('P', 'Sxx', 'PSD', 'psd', 'power_spectrogram')
    DB_KEYS = ('PdB_norm', 'power_db_norm', 'PdB', 'P_db')
    FREQ_KEYS = ('frequencies', 'F', 'freqs', 'freq', 'f')
    TIME_KEYS = ('times', 'T', 'time', 't')
    
    def __init__(
        self,
        mat_dir: str,
        crop_size: Optional[int] = None,
        min_db: float = -80.0,
        max_db: float = 0.0,
        sliding_window: bool = False,
        window_step: Optional[int] = None,  # None = same as crop_size (no overlap)
    ):
        """Initialize the inference dataset.
        
        Args:
            mat_dir: Directory containing MAT files
            crop_size: Crop size (should match training). If None, no cropping.
            min_db: Minimum dB for normalization
            max_db: Maximum dB for normalization
            sliding_window: If True, slide window across time axis
            window_step: Step size for sliding window. None = crop_size (no overlap)
        """
        self.mat_dir = Path(mat_dir)
        self.crop_size = crop_size
        self.min_db = min_db
        self.max_db = max_db
        self.sliding_window = sliding_window
        self.window_step = window_step if window_step is not None else crop_size
        
        # Find all MAT files
        self.mat_files = sorted(list(self.mat_dir.glob("*.mat")))
        if not self.mat_files:
            raise ValueError(f"No MAT files found in {mat_dir}")
        
        # Build index: list of (file_idx, window_start) tuples
        self.samples = []
        if sliding_window and crop_size is not None:
            # Pre-scan files to build window indices
            for file_idx, mat_path in enumerate(self.mat_files):
                spec, _ = self._load_spectrogram_raw(mat_path)
                F_dim, T_dim = spec.shape
                
                if T_dim <= crop_size:
                    # Single window if spectrogram is smaller than crop
                    self.samples.append((file_idx, 0))
                else:
                    # Calculate minimum number of windows needed
                    # and distribute them evenly with equal overlap
                    n_windows = int(np.ceil((T_dim - crop_size) / (self.window_step or crop_size))) + 1
                    
                    # Calculate actual step to distribute windows evenly
                    # n_windows covers: first at 0, last at (T_dim - crop_size)
                    # step = (T_dim - crop_size) / (n_windows - 1)
                    if n_windows > 1:
                        even_step = (T_dim - crop_size) / (n_windows - 1)
                    else:
                        even_step = 0
                    
                    for i in range(n_windows):
                        win_start = int(round(i * even_step))
                        self.samples.append((file_idx, win_start))
        else:
            # One sample per file (center crop)
            self.samples = [(i, None) for i in range(len(self.mat_files))]
    
    def _find_key(self, data: dict, keys: tuple) -> Optional[str]:
        """Find matching key in data dict (same as training)."""
        for k in keys:
            if k in data:
                return k
        # Case-insensitive fallback
        lowered = {k.lower(): k for k in data.keys()}
        for k in keys:
            if k.lower() in lowered:
                return lowered[k.lower()]
        return None
    
    def _load_spectrogram_raw(self, mat_path: Path) -> Tuple[np.ndarray, str]:
        """Load raw spectrogram from MAT file without normalization.

        Returns:
            Tuple of (spec, spec_kind) where spec_kind is 'power' or 'db'.
        """
        data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
        
        k = self._find_key(data, self.POWER_KEYS)
        spec_kind = 'power'
        if k is None:
            k = self._find_key(data, self.DB_KEYS) or self._find_key(data, self.SPECTRO_KEYS)
            spec_kind = 'db'
        if k is None:
            raise KeyError(f"No spectrogram-like key found in {mat_path.name}")
        
        spec = np.asarray(data[k])
        if spec.ndim != 2:
            raise ValueError(f"Unexpected spectrogram ndim {spec.ndim} in {mat_path.name}")
        
        # Check orientation using freq/time vectors if available
        fk = self._find_key(data, self.FREQ_KEYS)
        tk = self._find_key(data, self.TIME_KEYS)
        if fk in data and tk in data:
            f_len = int(np.asarray(data[fk]).ravel().shape[0])
            t_len = int(np.asarray(data[tk]).ravel().shape[0])
            r, c = spec.shape[:2]
            if (r, c) == (t_len, f_len):
                spec = spec.T  # now (F, T)
        
        return spec, spec_kind
    
    def _normalize_db_to_unit(self, x: np.ndarray) -> np.ndarray:
        """Normalize dB to [0, 1] (exactly like training: clip then normalize)."""
        x = x.astype(np.float32)
        x = np.clip(x, self.min_db, self.max_db)
        return (x - self.min_db) / (self.max_db - self.min_db)

    def _power_to_db_norm(self, power: np.ndarray) -> np.ndarray:
        power = np.abs(power.astype(np.float32))
        max_power = float(np.max(power)) if power.size else 0.0
        if max_power > 0:
            normalized = power / max_power
            normalized = np.maximum(normalized, 1e-10)
            return 10.0 * np.log10(normalized)
        return np.full_like(power, -100.0, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict]:
        file_idx, window_start = self.samples[idx]
        mat_path = self.mat_files[file_idx]
        file_id = mat_path.stem
        
        # Load spectrogram
        spec, spec_kind = self._load_spectrogram_raw(mat_path)
        F_dim, T_dim = spec.shape
        
        meta = {
            'original_shape': [F_dim, T_dim],
            'crop_size': [self.crop_size, self.crop_size] if self.crop_size else None,
            'sliding_window': self.sliding_window,
            'window_start': window_start,
            'window_step': self.window_step,
        }
        
        if self.crop_size is not None:
            crop_f, crop_t = self.crop_size, self.crop_size
            
            # Frequency axis: center crop
            if F_dim > crop_f:
                start_f = (F_dim - crop_f) // 2
                spec = spec[start_f:start_f + crop_f, :]
            
            # Time axis: sliding window or center crop
            if self.sliding_window and window_start is not None:
                # Use specified window position
                spec = spec[:, window_start:window_start + crop_t]
                meta['crop_type'] = 'sliding_window'
                meta['window_time_start'] = window_start
                meta['window_time_end'] = window_start + crop_t
            else:
                # Center crop
                if T_dim > crop_t:
                    start_t = (T_dim - crop_t) // 2
                    spec = spec[:, start_t:start_t + crop_t]
                    meta['window_time_start'] = start_t
                    meta['window_time_end'] = start_t + crop_t
                meta['crop_type'] = 'center_crop'
            
            meta['output_shape'] = list(spec.shape)
            meta['crop_applied'] = True
        else:
            meta['output_shape'] = [F_dim, T_dim]
            meta['crop_applied'] = False
            meta['crop_type'] = None

        # Normalize after cropping for consistent context
        if spec_kind == 'power':
            spec = self._power_to_db_norm(spec)
        spec = self._normalize_db_to_unit(spec)
        
        # Create unique file_id for sliding windows
        if self.sliding_window and window_start is not None:
            file_id = f"{file_id}_win{window_start}"
        
        # Convert to tensor [1, F, T]
        tensor = torch.from_numpy(spec).unsqueeze(0).float()
        
        return tensor, file_id, meta


def extract_crop_size_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """Extract crop_size from checkpoint's training args.
    
    Checks both the checkpoint dict and args.pkl in the same directory.
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Try args.pkl first (more complete)
    args_pkl = checkpoint_path.parent / 'args.pkl'
    if args_pkl.exists():
        try:
            import pickle
            with open(args_pkl, 'rb') as f:
                args = pickle.load(f)
            if hasattr(args, 'crop_size') and args.crop_size is not None:
                return int(args.crop_size) if isinstance(args.crop_size, (int, float)) else args.crop_size
        except Exception:
            pass
    
    # Try checkpoint dict
    try:
        import torch
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        if 'training_args' in ckpt:
            args = ckpt['training_args']
            if isinstance(args, dict) and 'crop_size' in args:
                return args['crop_size']
        if 'args' in ckpt:
            args = ckpt['args']
            if isinstance(args, dict) and 'crop_size' in args:
                return args['crop_size']
    except Exception:
        pass
    
    return None


def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run inference on all samples in dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with inference samples
        device: Device to run on
        
    Returns:
        List of {file_id, confidence, meta} dicts
    """
    model.eval()
    results = []
    
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                x, file_ids, metas = batch
            else:
                x, file_ids = batch
                metas = [{}] * len(file_ids)
            
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            # Get positive class probability
            pos_probs = probs[:, 1].cpu().numpy()
            
            for i, (file_id, prob) in enumerate(zip(file_ids, pos_probs)):
                meta = {}
                if isinstance(metas, dict):
                    # Batch collation turned it into a dict of tensors
                    for k, v in metas.items():
                        try:
                            meta[k] = v[i].item() if hasattr(v[i], 'item') else v[i]
                        except:
                            meta[k] = None
                elif isinstance(metas, (list, tuple)) and i < len(metas):
                    meta = metas[i] if isinstance(metas[i], dict) else {}
                
                results.append({
                    'file_id': file_id,
                    'confidence': float(prob),
                    'meta': meta
                })
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                print(f"  Batch {batch_idx + 1}/{total_batches}", end='\r')
    
    print()  # newline after progress
    return results


def _normalize_spec_config_from_test_metadata(proc_params: Dict[str, Any]) -> Dict[str, Any]:
    # Allow caller to inject provenance; default to computed if missing
    source_override = proc_params.get("spectrogram_source")
    freq_lims = proc_params.get('freq_lims_hz')
    freq_limits = None
    if isinstance(freq_lims, (list, tuple)) and len(freq_lims) >= 2:
        freq_limits = {"min": freq_lims[0], "max": freq_lims[1]}
    elif isinstance(freq_lims, dict):
        freq_limits = {"min": freq_lims.get("min"), "max": freq_lims.get("max")}

    clim = proc_params.get('clim_db')
    color_limits = None
    if isinstance(clim, (list, tuple)) and len(clim) >= 2:
        color_limits = {"min": clim[0], "max": clim[1]}
    elif isinstance(clim, dict):
        color_limits = {"min": clim.get("min"), "max": clim.get("max")}

    spec_config = {
        "window_duration": proc_params.get("win_dur_s"),
        "overlap": proc_params.get("overlap"),
        "frequency_limits": freq_limits,
        "color_limits": color_limits,
        "crop_size": proc_params.get("crop_size"),
        "pipeline": "test_windows",
    }
    if source_override:
        if isinstance(source_override, dict):
            spec_config["source"] = source_override
        else:
            spec_config["source"] = {"type": str(source_override)}
    else:
        spec_config["source"] = {
            "type": "computed",
            "generator": "onc_hydrophone_data.SpectrogramGenerator",
        }
    # Remove empty keys
    return {k: v for k, v in spec_config.items() if v is not None}


def _infer_spectrogram_source(dataset_meta: Dict[str, Any], default_type: str) -> Dict[str, Any]:
    """Infer spectrogram source/provenance metadata when not explicitly provided."""
    source = {}
    if "spectrogram_source" in dataset_meta:
        value = dataset_meta.get("spectrogram_source")
        if isinstance(value, dict):
            return value
        return {"type": str(value)}
    if "spectrogram_download" in dataset_meta and isinstance(dataset_meta.get("spectrogram_download"), dict):
        return dataset_meta["spectrogram_download"]

    plot_res = dataset_meta.get("plot_res", dataset_meta.get("plotRes"))
    if plot_res is not None:
        source = {"type": "onc_download", "provider": "ONC", "plot_res": plot_res}
        return source

    return {"type": default_type}


def load_inference_metadata(path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    """Load metadata and normalize to unified tracker fields.

    Returns (data_source, spectrogram_config, file_info_map, metadata_type)
    """
    data_source: Dict[str, Any] = {}
    spec_config: Dict[str, Any] = {}
    file_info_map: Dict[str, Any] = {}
    metadata_type = "unknown"

    if not path or not Path(path).exists():
        return data_source, spec_config, file_info_map, metadata_type

    with open(path, 'r') as f:
        dataset_meta = json.load(f)

    # Legacy sequential pipeline metadata.json
    if "data_source" in dataset_meta and "files" in dataset_meta:
        metadata_type = "legacy_segments"
        data_source = dataset_meta.get("data_source", {})
        spec_config = dataset_meta.get("spectrogram_config", {})
        if "source" not in spec_config:
            spec_config["source"] = _infer_spectrogram_source(dataset_meta, "computed")
        for file_info in dataset_meta.get("files", []):
            file_info_map[file_info.get("file_id")] = file_info
        return data_source, spec_config, file_info_map, metadata_type

    # Test windows metadata.json (prepare_test_windows.py)
    if "chunks" in dataset_meta and "processing_parameters" in dataset_meta:
        metadata_type = "test_windows"
        data_source = {
            "device_code": dataset_meta.get("device_code", "unknown"),
            "date_from": dataset_meta.get("start_date", ""),
            "date_to": dataset_meta.get("end_date", ""),
        }
        proc_params = dataset_meta.get("processing_parameters", {})
        if "spectrogram_source" not in proc_params:
            proc_params["spectrogram_source"] = _infer_spectrogram_source(dataset_meta, "computed")
        spec_config = _normalize_spec_config_from_test_metadata(proc_params)
        for chunk in dataset_meta.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            file_info_map[chunk_id] = {
                "mat_path": chunk.get("mat_path"),
                "audio_path": chunk.get("audio_path"),
                "audio_timestamp": chunk.get("timestamp"),
                "chunk_shape": chunk.get("chunk_shape"),
                "original_shape": chunk.get("original_shape"),
                "window_index": chunk.get("window_index"),
                "window_start": chunk.get("window_start"),
                "window_time_start": chunk.get("window_time_start"),
                "window_time_end": chunk.get("window_time_end"),
                "source_audio": chunk.get("source_audio"),
                "date": chunk.get("date"),
            }
        return data_source, spec_config, file_info_map, metadata_type

    return data_source, spec_config, file_info_map, metadata_type


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on sequential spectrograms"
    )
    parser.add_argument('--mat-dir', type=str, required=True,
                        help='Directory with MAT spectrograms')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-json', type=str, required=True,
                        help='Output predictions JSON path')
    parser.add_argument('--dataset-metadata', type=str, default=None,
                        help='Path to metadata JSON (auto-detects legacy vs test windows)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--crop-size', type=str, default=None,
                        help='Crop size (int). Auto-detected from checkpoint if not specified.')
    parser.add_argument('--sliding-window', action='store_true',
                        help='Use sliding window to scan entire spectrogram')
    parser.add_argument('--window-step', type=int, default=None,
                        help='Step size for sliding window (default: crop_size = no overlap)')
    parser.add_argument('--min-db', type=float, default=-80.0,
                        help='Min dB for normalization')
    parser.add_argument('--max-db', type=float, default=0.0,
                        help='Max dB for normalization')
    parser.add_argument('--verify-hash', action='store_true',
                        help='Verify model hash matches checkpoint')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print_header("MODEL INFERENCE ON SEQUENTIAL DATA")
    print(f"MAT directory: {args.mat_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Parse crop size - auto-detect from checkpoint if not specified
    crop_size = None
    if args.crop_size:
        if ',' in args.crop_size:
            parts = args.crop_size.split(',')
            crop_size = [int(p.strip()) for p in parts]
        else:
            crop_size = int(args.crop_size)
    else:
        # Try to auto-detect from checkpoint
        crop_size = extract_crop_size_from_checkpoint(args.checkpoint)
        if crop_size:
            print(f"Auto-detected crop_size from checkpoint: {crop_size}")
        else:
            print("Warning: crop_size not specified and could not be auto-detected")
    
    # Load checkpoint
    print_status("Loading checkpoint...", "PROGRESS")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract model info
    model_info = extract_model_info(checkpoint)
    model_info['checkpoint_path'] = str(Path(args.checkpoint).resolve())
    
    print(f"  Model ID: {model_info['model_id']}")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Trained at: {model_info['trained_at']}")
    
    # Verify hash if requested
    if args.verify_hash:
        if verify_model_hash(checkpoint):
            print_status("Model hash verified ✓", "SUCCESS")
        else:
            print_status("WARNING: Model hash mismatch!", "WARNING")
    
    # Create model
    architecture = model_info['architecture']
    model = create_model(architecture, num_classes=2, in_ch=1).to(device)
    
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    print_status(f"Model loaded: {architecture}", "SUCCESS")
    
    # Create dataset
    print_status("Loading dataset...", "PROGRESS")
    dataset = InferenceDataset(
        mat_dir=args.mat_dir,
        crop_size=crop_size,
        min_db=args.min_db,
        max_db=args.max_db,
        sliding_window=args.sliding_window,
        window_step=args.window_step,
    )
    
    n_files = len(dataset.mat_files)
    n_samples = len(dataset)
    mode = "sliding window" if args.sliding_window else "center crop"
    print(f"  Found {n_files} MAT files -> {n_samples} samples ({mode})")
    
    # Create dataloader with custom collate for variable metadata
    def collate_fn(batch):
        tensors = torch.stack([b[0] for b in batch])
        file_ids = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        return tensors, file_ids, metas
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Run inference
    print_status("Running inference...", "PROGRESS")
    results = run_inference(model, dataloader, device)
    print_status(f"Inference complete: {len(results)} predictions", "SUCCESS")
    
    # Load dataset metadata if provided (auto-detect format)
    data_source, spec_config, file_info_map, metadata_type = load_inference_metadata(args.dataset_metadata)
    if args.dataset_metadata:
        print(f"Metadata type: {metadata_type}")
    
    # Create prediction tracker
    tracker = UnifiedPredictionTracker(args.output_json)
    
    # Set model info
    tracker.set_model_info(
        model_id=model_info['model_id'],
        architecture=model_info['architecture'],
        checkpoint_path=model_info['checkpoint_path'],
        trained_at=model_info['trained_at'],
        wandb_run_id=model_info['wandb_run_id'],
        input_shape=[crop_size, crop_size] if crop_size and isinstance(crop_size, int) else None,
        output_classes=["Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"]
    )
    
    # Set task type
    tracker.set_task_type('whale_detection')
    
    # Set data source and config
    if data_source:
        tracker.set_data_source(
            device_code=data_source.get('device_code', 'unknown'),
            date_from=data_source.get('date_from', ''),
            date_to=data_source.get('date_to', ''),
            sample_rate=data_source.get('sample_rate'),
        )
    
    if spec_config:
        tracker.set_spectrogram_config(spec_config)
    
    # Add predictions
    for result in results:
        file_id = result['file_id']
        base_id = file_id.rsplit('_win', 1)[0] if '_win' in file_id else file_id
        file_info = file_info_map.get(file_id, file_info_map.get(base_id, {}))
        meta = result.get('meta', {})
        
        # Build model_outputs in unified format
        model_outputs = [{
            "class_hierarchy": "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale",
            "score": result['confidence'],  # Store raw score (not thresholded)
        }]
        
        # Add item with unified format
        mat_path_default = str(Path("spectrograms") / f"{base_id}.mat")
        audio_path_default = str(Path("audio") / f"{base_id}.wav")
        spectrogram_path_default = None
        if "spectrogram_path" in file_info:
            spectrogram_path_default = file_info.get("spectrogram_path")
        # Prefer richer metadata from file_info_map when available (e.g., test windows)
        window_start = meta.get('window_start')
        window_time_start = meta.get('window_time_start')
        window_time_end = meta.get('window_time_end')
        original_shape = meta.get('original_shape')
        if file_info:
            window_start = file_info.get('window_start', window_start)
            window_time_start = file_info.get('window_time_start', window_time_start)
            window_time_end = file_info.get('window_time_end', window_time_end)
            original_shape = file_info.get('original_shape', original_shape)

        duration_sec = spec_config.get('context_duration') if spec_config else None
        if duration_sec is None and window_time_start is not None and window_time_end is not None:
            duration_sec = max(0.0, float(window_time_end) - float(window_time_start))

        spectrogram_mat_path = file_info.get('mat_path', mat_path_default)
        spectrogram_png_path = spectrogram_path_default

        tracker.add_item(
            item_id=file_id,
            model_outputs=model_outputs,
            mat_path=spectrogram_mat_path,
            audio_path=file_info.get('audio_path', audio_path_default),
            spectrogram_path=spectrogram_png_path,
            audio_timestamp=file_info.get('audio_timestamp', ''),
            duration_sec=duration_sec,
            # Additional metadata
            source_audio=file_info.get('source_audio'),
            segment_start_sec=file_info.get('segment_start_sec', window_time_start),
            segment_end_sec=file_info.get('segment_end_sec', window_time_end),
            segment_index=file_info.get('segment_index', file_info.get('window_index')),
            chunk_shape=file_info.get('chunk_shape'),
            # Descriptive aliases for spectrogram paths
            spectrogram_mat_path=spectrogram_mat_path,
            spectrogram_png_path=spectrogram_png_path,
            # Crop/window metadata
            original_shape=original_shape,
            crop_size=meta.get('crop_size'),
            crop_applied=meta.get('crop_applied'),
            crop_type=meta.get('crop_type'),
            window_start=window_start,
            window_time_start=window_time_start,
            window_time_end=window_time_end,
        )
    
    # Save predictions
    tracker.save()
    
    # Print summary
    print_header("RESULTS")
    summary = tracker.summary()
    print(f"Total items: {summary.get('total_items', 0)}")
    if 'mean_score' in summary:
        print(f"Mean confidence: {summary['mean_score']:.4f}")
        print(f"Min confidence: {summary['min_score']:.4f}")
        print(f"Max confidence: {summary['max_score']:.4f}")
    
    # Show threshold-based counts as preview
    class_name = "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        above = len(tracker.get_items_by_score_threshold(class_name, threshold, above=True))
        total_items = max(summary.get('total_items', 0), 1)
        print(f"  >= {threshold:.1f}: {above} ({100*above/total_items:.1f}%)")
    
    print(f"\nPredictions saved to: {args.output_json}")
    print_status("Inference complete!", "SUCCESS")


if __name__ == "__main__":
    main()
