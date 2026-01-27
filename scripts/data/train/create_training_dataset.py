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
    parser.add_argument('--excel-file', type=str, required=True, 
                        help='Path to Excel file with whale call annotations')
    parser.add_argument('--output-dir', type=str, default='whale_dataset', 
                        help='Output directory for spectrograms')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of calls to sample (default: all)')
    parser.add_argument('--process-all', action='store_true', 
                        help='Process all calls (ignore --sample-size)')
    parser.add_argument('--generate-negatives', action='store_true', 
                        help='Generate negative (no-call) samples')
    parser.add_argument('--cleanup-audio', action='store_true', 
                        help='Delete audio files after processing to save space')
    parser.add_argument('--workers', type=int, default=2, 
                        help='Number of parallel workers')
    parser.add_argument('--config', type=str, default='./config/dataset_config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--skip-onc-spectrograms', action='store_true',
                        help='Skip downloading ONC reference spectrograms')
    
    args = parser.parse_args()
    
    # Load environment variables (for ONC_TOKEN)
    load_dotenv()
    onc_token = os.getenv('ONC_TOKEN')
    if not onc_token:
        print_status("Error: ONC_TOKEN not found in .env file.", "ERROR")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create generator
    generator = SpectrogramDatasetGenerator(
        onc_token=onc_token,
        excel_file=args.excel_file,
        config_path=args.config
    )
    
    # 2. Load whale call data
    whale_data = load_whale_data(generator.excel_files)
    generator.whale_data = whale_data
    
    # 3. Sample calls
    sample_size = None if args.process_all else args.sample_size
    sampled_calls = sample_calls(whale_data, sample_size=sample_size)
    
    # 4. Generate spectrograms
    specs, failed, dims = generator.generate_spectrograms(
        sampled_calls, 
        output_dir,
        max_workers=args.workers,
        cleanup_audio=args.cleanup_audio,
        generate_negatives=args.generate_negatives
    )
    
    # 5. Optionally download ONC reference spectrograms
    onc_specs = {}
    if not args.skip_onc_spectrograms:
        onc_specs = download_onc_spectrograms(generator.onc, sampled_calls, output_dir)
    
    # 6. Create analysis report
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
    
    print_status(f"Dataset created successfully in {output_dir}", "SUCCESS")


if __name__ == "__main__":
    main()
