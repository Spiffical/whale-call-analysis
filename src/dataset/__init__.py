# Dataset creation utilities for whale call analysis
# Used by scripts/data/train/create_training_dataset.py

try:
    from .call_catalog import load_whale_data, sample_calls  # noqa: F401
except Exception:
    load_whale_data = None
    sample_calls = None

from .negative_sampler import sample_negative_windows_for_file  # noqa: F401
from .part2_annotations import build_part2_manifests, write_part2_manifests  # noqa: F401
from .part2_finetune import (  # noqa: F401
    assign_time_pools,
    build_learning_curve_plan,
    load_finetune_clip_records,
    order_train_pool,
    select_budget_clips,
)

try:
    from .reporting import print_status, print_header, create_analysis_report  # noqa: F401
except Exception:
    print_status = None
    print_header = None
    create_analysis_report = None

# Optional: ONC-backed audio helpers are not required for prebuilt-context
# workflows and should not block imports in cluster training environments.
try:
    from .audio import stitch_audio_files, cleanup_audio_files  # noqa: F401
except Exception:
    stitch_audio_files = None
    cleanup_audio_files = None

# Optional: spectrogram utilities depend on the ONC-backed audio module.
try:
    from .spectrogram import create_custom_spectrograms, download_onc_spectrograms  # noqa: F401
except Exception:
    create_custom_spectrograms = None
    download_onc_spectrograms = None

# Optional: avoid hard import of SpectrogramDatasetGenerator (pulls torchaudio)
# so inference-only workflows don't fail on torchaudio/torch binary mismatches.
try:
    from .generator import SpectrogramDatasetGenerator  # noqa: F401
except Exception:
    SpectrogramDatasetGenerator = None
