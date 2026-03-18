# Dataset creation utilities for whale call analysis
# Used by scripts/data/train/create_training_dataset.py

from .call_catalog import load_whale_data, sample_calls
from .negative_sampler import sample_negative_windows_for_file
from .reporting import print_status, print_header, create_analysis_report

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
