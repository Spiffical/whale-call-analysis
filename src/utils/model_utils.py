#!/usr/bin/env python3
"""
Model Versioning Utilities

Provides functions for computing model hashes and creating standardized
checkpoint metadata for reproducibility and tracking.
"""

import hashlib
import io
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch


def compute_model_hash(state_dict: Dict[str, torch.Tensor], length: int = 12) -> str:
    """Compute SHA256 hash of model weights for unique identification.
    
    Args:
        state_dict: Model state dictionary containing weights
        length: Number of characters to include in hash (default: 12)
        
    Returns:
        String in format 'sha256-{first N chars of hash}'
    """
    # Serialize weights deterministically to bytes
    buffer = io.BytesIO()
    # Sort keys for deterministic ordering
    sorted_dict = {k: state_dict[k] for k in sorted(state_dict.keys())}
    torch.save(sorted_dict, buffer)
    buffer.seek(0)
    
    # Compute SHA256 hash
    hasher = hashlib.sha256()
    hasher.update(buffer.read())
    full_hash = hasher.hexdigest()
    
    return f"sha256-{full_hash[:length]}"


def create_checkpoint_metadata(
    model: torch.nn.Module,
    args: Any,
    wandb_run_id: Optional[str] = None,
    additional_info: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create standardized checkpoint metadata with versioning info.
    
    Args:
        model: The PyTorch model
        args: Training arguments (argparse.Namespace or dict)
        wandb_run_id: Optional WandB run ID
        additional_info: Optional dict of additional metadata
        
    Returns:
        Dict with standardized checkpoint metadata
    """
    state_dict = model.state_dict()
    model_id = compute_model_hash(state_dict)
    
    # Extract model name from args
    if hasattr(args, 'model'):
        architecture = args.model
    elif isinstance(args, dict) and 'model' in args:
        architecture = args['model']
    else:
        architecture = model.__class__.__name__
    
    # Convert args to dict if needed
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        args_dict = {}
    
    metadata = {
        'model_id': model_id,
        'architecture': architecture,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'wandb_run_id': wandb_run_id,
        'training_args': args_dict,
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    return metadata


def verify_model_hash(checkpoint: Dict[str, Any], expected_hash: Optional[str] = None) -> bool:
    """Verify that model weights match the stored hash.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        expected_hash: Optional hash to verify against (uses stored hash if None)
        
    Returns:
        True if hash matches, False otherwise
    """
    if 'model_state' not in checkpoint:
        return False
    
    stored_hash = expected_hash or checkpoint.get('model_id')
    if not stored_hash:
        return False
    
    computed_hash = compute_model_hash(checkpoint['model_state'])
    return computed_hash == stored_hash


def extract_model_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model versioning info from checkpoint.
    
    Automatically computes model_id hash if not present in checkpoint
    for backwards compatibility with older models.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Dict with model_id, architecture, trained_at, wandb_run_id
    """
    # Get model_id, compute if not present
    model_id = checkpoint.get('model_id')
    if not model_id or model_id == 'unknown':
        # Compute hash from model weights for backwards compatibility
        if 'model_state' in checkpoint:
            model_id = compute_model_hash(checkpoint['model_state'])
            print(f"  Computed model_id from weights: {model_id}")
        else:
            model_id = 'unknown'
    
    return {
        'model_id': model_id,
        'architecture': checkpoint.get('architecture', 
                                       checkpoint.get('args', {}).get('model', 'unknown')),
        'trained_at': checkpoint.get('trained_at', None),
        'wandb_run_id': checkpoint.get('wandb_run_id', None),
        'checkpoint_path': None,  # To be filled by caller
    }
