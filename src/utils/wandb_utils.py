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


def log_test_example_images(
    model_label: str,
    tp_images: list,
    fp_images: list,
    fn_images: list,
    max_images: int = 8,
) -> None:
    """Log example prediction images to wandb.
    
    Args:
        model_label: Label/name for this model
        tp_images: List of (image_array, caption) tuples for true positives
        fp_images: List of (image_array, caption) tuples for false positives  
        fn_images: List of (image_array, caption) tuples for false negatives
        max_images: Maximum images per category
    """
    if wandb.run is None:
        return
    
    def make_wandb_images(img_list, max_n):
        return [wandb.Image(img, caption=cap) for img, cap in img_list[:max_n]]
    
    if tp_images:
        wandb.log({f"{model_label}/true_positives": make_wandb_images(tp_images, max_images)})
    if fp_images:
        wandb.log({f"{model_label}/false_positives": make_wandb_images(fp_images, max_images)})
    if fn_images:
        wandb.log({f"{model_label}/false_negatives": make_wandb_images(fn_images, max_images)})


def log_test_metrics(
    model_label: str,
    metrics: Dict[str, Any],
    probs: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Log test metrics and individual plots for a single model to wandb.
    
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
            log_dict[f"{model_label}/{k}"] = float(v)
    
    # Log individual ROC and PR curves
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        
        # ROC curve plot
        log_dict[f"{model_label}/roc"] = wandb.plot.line_series(
            xs=[fpr.tolist()],
            ys=[tpr.tolist()],
            keys=[f"{model_label} (AUC={roc_auc:.3f})"],
            title=f"ROC Curve - {model_label}",
            xname="False Positive Rate",
            yname="True Positive Rate"
        )
        
        # PR curve plot
        prec, rec, _ = precision_recall_curve(y_true, probs)
        log_dict[f"{model_label}/pr"] = wandb.plot.line_series(
            xs=[rec.tolist()],
            ys=[prec.tolist()],
            keys=[model_label],
            title=f"Precision-Recall - {model_label}",
            xname="Recall",
            yname="Precision"
        )
    except Exception as e:
        print(f"Warning: Could not create wandb plots for {model_label}: {e}")
    
    wandb.log(log_dict)


def log_test_comparison(
    all_results: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
) -> None:
    """Log combined comparison plots for multiple models to wandb.
    
    Logs:
    - Combined ROC curve (all models on same plot)
    - Combined PR curve (all models on same plot)
    - Summary table with key metrics
    
    Args:
        all_results: Dict mapping model_label to {'probs': array, 'y_true': array}
        out_dir: Directory containing the saved plot images
    """
    if wandb.run is None:
        return
    
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # Log the combined plot images (generated by test_cnn.py)
    plot_files = [
        ('roc_curve_all.png', 'ROC Curves (All Models)'),
        ('pr_curve_all.png', 'Precision-Recall Curves (All Models)'),
    ]
    
    for fname, title in plot_files:
        fpath = out_dir / fname
        if fpath.exists():
            wandb.log({f"comparison/{fname.replace('.png', '')}": wandb.Image(str(fpath), caption=title)})
    
    # Create summary metrics table
    columns = ["Model", "AUC", "F1", "Precision", "Recall", "Accuracy"]
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
        
        data.append([label, f"{roc_auc:.4f}", f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{acc:.4f}"])
    
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"comparison/metrics_table": table})


def finish_run():
    """Finish the current wandb run."""
    if wandb.run is not None:
        wandb.finish()
