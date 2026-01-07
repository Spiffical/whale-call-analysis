"""
Utilities for Weights & Biases (wandb) integration.
Supports both training and testing workflows.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import wandb
import numpy as np
import torch


def init_wandb(args, project_name="whale-call-analysis", entity=None, group=None, run_id=None):
    """Initialize a wandb run for training."""
    if not hasattr(args, 'use_wandb') or not args.use_wandb:
        return None
        
    config = {
        "architecture": getattr(args, 'model', None),
        "batch_size": getattr(args, 'batch_size', None),
        "learning_rate": getattr(args, 'lr', None),
        "epochs": getattr(args, 'epochs', None) or getattr(args, 'n_epochs', None),
    }
    
    run = wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=os.path.basename(args.exp_dir) if hasattr(args, 'exp_dir') else None,
        dir=getattr(args, 'exp_dir', None),
        group=group,
        id=run_id
    )
    
    return run


def init_wandb_test(
    project_name: str = "whale-call-analysis",
    entity: Optional[str] = None,
    group: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    out_dir: Optional[str] = None,
) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize a wandb run for testing/evaluation.
    
    Args:
        project_name: WandB project name
        entity: WandB entity (username or team)
        group: Group name for organizing runs
        run_name: Name for this test run
        config: Configuration dict to log
        out_dir: Directory for wandb files
        
    Returns:
        wandb Run object or None if initialization fails
    """
    try:
        run = wandb.init(
            project=project_name,
            entity=entity,
            config=config or {},
            name=run_name,
            dir=out_dir,
            group=group,
            job_type="evaluation",
        )
        return run
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        return None


def log_training_metrics(metrics_dict, use_wandb=True):
    """Log training metrics to wandb."""
    if not use_wandb or wandb.run is None:
        return
    wandb.log(metrics_dict)


def log_validation_metrics(metrics, task, epoch, use_wandb=True):
    """Log validation metrics to wandb."""
    if not use_wandb or wandb.run is None:
        return
    
    log_dict = {"epoch": epoch}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.number)):
            log_dict[f"val/{k}"] = v
            
    wandb.log(log_dict)


def log_test_metrics(
    model_label: str,
    metrics: Dict[str, Any],
    probs: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Log test metrics and plots for a single model to wandb.
    
    Args:
        model_label: Label/name for this model
        metrics: Dict of metrics (accuracy, precision, recall, f1, auc)
        probs: Predicted probabilities for positive class
        y_true: Ground truth labels
    """
    if wandb.run is None:
        return
    
    # Log scalar metrics with model prefix
    log_dict = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.number)):
            log_dict[f"test/{model_label}/{k}"] = float(v)
    
    # Log ROC curve
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        
        # Create ROC data for wandb
        roc_data = [[x, y] for x, y in zip(fpr, tpr)]
        log_dict[f"test/{model_label}/roc"] = wandb.plot.line_series(
            xs=[fpr.tolist()],
            ys=[tpr.tolist()],
            keys=[f"{model_label} (AUC={roc_auc:.3f})"],
            title=f"ROC Curve - {model_label}",
            xname="False Positive Rate",
        )
        
        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, probs)
        log_dict[f"test/{model_label}/pr"] = wandb.plot.line_series(
            xs=[rec.tolist()],
            ys=[prec.tolist()],
            keys=[model_label],
            title=f"Precision-Recall - {model_label}",
            xname="Recall",
        )
    except Exception as e:
        print(f"Warning: Could not create wandb plots: {e}")
    
    wandb.log(log_dict)


def log_test_comparison(
    all_results: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
) -> None:
    """Log comparison plots for multiple models to wandb.
    
    Args:
        all_results: Dict mapping model_label to {'probs': array, 'y_true': array}
        out_dir: Directory containing the saved plot images
    """
    if wandb.run is None:
        return
    
    # Log combined plot images as artifacts
    plot_files = [
        'roc_curve_all.png',
        'pr_curve_all.png',
        'precision_recall_vs_threshold_all.png',
        'accuracy_vs_center_offset_all.png',
    ]
    
    for fname in plot_files:
        fpath = out_dir / fname
        if fpath.exists():
            wandb.log({f"plots/{fname.replace('.png', '')}": wandb.Image(str(fpath))})
    
    # Create a summary table
    try:
        from sklearn.metrics import roc_curve, auc
        
        columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
        data = []
        
        for label, res in all_results.items():
            probs = res['probs']
            y_true = res['y_true']
            preds = (probs >= 0.5).astype(int)
            
            tp = ((preds == 1) & (y_true == 1)).sum()
            tn = ((preds == 0) & (y_true == 0)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            
            data.append([label, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"])
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"test/model_comparison": table})
    except Exception as e:
        print(f"Warning: Could not create comparison table: {e}")


def finish_run():
    """Finish the current wandb run."""
    if wandb.run is not None:
        wandb.finish()
