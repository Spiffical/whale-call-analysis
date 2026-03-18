# Model training utilities for whale call analysis
# Used by scripts/train/train_cnn.py and scripts/train/test_cnn.py

from .splits import build_entries, split_group_by_source, split_time_separated
from .mat_utils import parse_mat_filename, list_mat_files

# Optional: CNN dataset helpers require torch and should not block
# embedding-only workflows that only need split utilities.
try:
    from .mat_dataset import FinWhaleMatDataset, make_dataloaders  # noqa: F401
except Exception:
    FinWhaleMatDataset = None
    make_dataloaders = None
