# Model training utilities for whale call analysis
# Used by scripts/train_cnn.py and scripts/test_cnn.py

from .mat_dataset import FinWhaleMatDataset, make_dataloaders
from .splits import build_entries, split_group_by_source, split_time_separated
from .mat_utils import parse_mat_filename, list_mat_files
