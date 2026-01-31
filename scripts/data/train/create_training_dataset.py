#!/usr/bin/env python3
"""
Fin Whale Training Dataset Creation Script

Creates training datasets from whale call annotations by:
1. Loading annotations from Excel files
2. Downloading ONC audio files
3. Generating spectrograms (MAT files for training, PNGs for visualization)
4. Optionally generating negative (no-call) samples
5. Creating an analysis report

Usage:
    python scripts/data/train/create_training_dataset.py \
        --excel-file data/finwhales/calls.xlsx \
        --output-dir output/
"""

import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

# Ensure repo root is on sys.path so `src` is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.generator import SpectrogramDatasetGenerator
from src.dataset.call_catalog import load_whale_data, sample_calls
from src.dataset.spectrogram import download_onc_spectrograms
from src.dataset.reporting import print_status, create_analysis_report


def main():
    parser = argparse.ArgumentParser(
        description="Create spectrogram dataset from whale call annotations"
    )
    parser.add_argument('--excel-file', type=str, nargs='+', required=True,
                        help='Path(s) to Excel file(s) with whale call annotations')
    parser.add_argument('--output-dir', type=str, default='whale_dataset',
                        help='Output directory for spectrograms')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of calls to sample (default: all)')
    parser.add_argument('--process-all', action='store_true',
                        help='Process all calls (ignore --sample-size)')

    filter_group = parser.add_argument_group("Sampling filters")
    filter_group.add_argument('--min-duration', type=float, default=1.0,
                              help='Minimum call duration in seconds (default: 1.0)')
    filter_group.add_argument('--max-duration', type=float, default=30.0,
                              help='Maximum call duration in seconds (default: 30.0)')

    config_group = parser.add_argument_group("Config overrides")
    config_group.add_argument('--win-dur', type=float, default=None,
                              help='Override config custom_spectrograms.window_duration (seconds)')
    config_group.add_argument('--overlap', type=float, default=None,
                              help='Override config custom_spectrograms.overlap')
    config_group.add_argument('--ml-context', type=float, default=None,
                              help='Override config temporal_context.context_duration (seconds)')
    config_group.add_argument('--freq-range', type=float, nargs=2, default=None,
                              metavar=('MIN_HZ', 'MAX_HZ'),
                              help='Override config frequency limits and filter calls by [min max] Hz')

    parser.add_argument('--generate-negatives', action='store_true',
                        help='Generate negative (no-call) samples')
    parser.add_argument('--negatives-per-call', type=int, default=1,
                        help='Number of negative windows per call (default: 1)')
    parser.add_argument('--neg-context', type=float, default=None,
                        help='Context duration for negatives in seconds (default: --ml-context)')
    parser.add_argument('--neg-margin', type=float, default=2.0,
                        help='Safety margin around calls when sampling negatives (seconds)')

    parser.add_argument('--cleanup-audio', action='store_true',
                        help='Delete audio files after processing to save space')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel workers')
    parser.add_argument('--config', type=str, default='./config/dataset_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--skip-onc-spectrograms', action='store_true',
                        help='Skip downloading ONC reference spectrograms')
    parser.add_argument('--tar-output', action='store_true',
                        help='Create a tar archive of mat_files and neg_mat_files after processing')

    args = parser.parse_args()
    
    # Load environment variables (for ONC_TOKEN)
    load_dotenv()
    onc_token = os.getenv('ONC_TOKEN')
    if not onc_token:
        print_status("Error: ONC_TOKEN not found in .env file.", "ERROR")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create generator (supports multiple Excel files)
    generator = SpectrogramDatasetGenerator(
        onc_token=onc_token,
        excel_files=args.excel_file,
        config_path=args.config
    )
    
    # 2. Load whale call data
    whale_data = load_whale_data(generator.excel_files)
    generator.whale_data = whale_data
    
    # 3. Sample calls
    sample_size = None if args.process_all else args.sample_size
    freq_range = tuple(args.freq_range) if args.freq_range else None
    sampled_calls = sample_calls(
        whale_data,
        sample_size=sample_size,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        freq_range=freq_range,
    )

    # 4. Apply config overrides (if provided)
    generator.apply_overrides(
        win_dur=args.win_dur,
        overlap=args.overlap,
        freq_range=freq_range,
        ml_context=args.ml_context,
    )
    
    # 5. Generate spectrograms
    gen_kwargs = {
        "max_workers": args.workers,
        "cleanup_audio": args.cleanup_audio,
        "generate_negatives": args.generate_negatives,
        "negatives_per_call": args.negatives_per_call,
        "neg_margin": args.neg_margin,
    }
    if args.ml_context is not None:
        gen_kwargs["ml_context"] = args.ml_context
    if args.neg_context is not None:
        gen_kwargs["neg_context"] = args.neg_context

    specs, failed, dims = generator.generate_spectrograms(
        sampled_calls, 
        output_dir,
        **gen_kwargs
    )
    
    # 6. Optionally download ONC reference spectrograms
    onc_specs = {}
    if not args.skip_onc_spectrograms:
        onc_specs = download_onc_spectrograms(generator.onc, sampled_calls, output_dir)
    
    # 7. Create analysis report
    create_analysis_report(
        output_dir, 
        generator.excel_files, 
        sampled_calls,
        {cid: "processed" for cid in sampled_calls['clip id'].unique()},
        specs, 
        onc_specs, 
        generator.spectrogram_generator, 
        generator.config,
        failed_calls=failed, 
        actual_dimensions=dims, 
        audio_cleaned_up=args.cleanup_audio
    )
    
    # 8. Optionally tar up MAT files
    if args.tar_output:
        import tarfile
        tar_path = output_dir / "all_mat_files.tar"
        print_status(f"Creating tar archive: {tar_path}", "PROGRESS")
        with tarfile.open(tar_path, 'w') as tar:
            for dirname in ['mat_files', 'neg_mat_files']:
                dir_path = output_dir / dirname
                if dir_path.exists():
                    tar.add(str(dir_path), arcname=dirname)
                    print_status(f"  Added {dirname}/", "SUCCESS")
        print_status(f"Tar archive created: {tar_path} ({tar_path.stat().st_size / (1024**3):.1f} GB)", "SUCCESS")

    print_status(f"Dataset created successfully in {output_dir}", "SUCCESS")


if __name__ == "__main__":
    main()
