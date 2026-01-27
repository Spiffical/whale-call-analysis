#!/usr/bin/env python3
"""
DEPRECATED: Download Sequential Audio Data and Generate Custom Spectrograms

Downloads raw audio files between two dates for a specific hydrophone,
splits them into context_duration segments, and generates custom spectrograms
using the project's spectrogram configuration.

This script reflects an older segmentation workflow that does not match the
current test-data preparation pipeline. Use:
    python scripts/data/test/prepare_test_windows.py ...
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
import scipy.io
import soundfile as sf
from dotenv import load_dotenv

# External package
from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader
from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator

from src.dataset.reporting import print_status, print_header


def parse_datetime(dt_str: str) -> datetime:
    """Parse ISO format datetime string to timezone-aware datetime."""
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def compute_segment_windows(
    audio_duration: float,
    context_duration: float,
    min_overlap: float = 0.5
) -> List[Tuple[float, float]]:
    """Compute segment windows that cleanly fit in audio file with minimal overlap.
    
    Given a 5-minute (300s) audio file and context_duration (e.g., 40s),
    this computes segments that:
    1. Each segment is exactly context_duration seconds
    2. Segments have minimal overlap to cleanly cover the file
    3. All segments are the same size
    
    Args:
        audio_duration: Total audio duration in seconds
        context_duration: Desired segment duration in seconds
        min_overlap: Minimum overlap in seconds
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    windows = []
    
    # Calculate how many full segments fit
    if context_duration >= audio_duration:
        # Single segment covering entire file
        return [(0.0, min(context_duration, audio_duration))]
    
    # Calculate number of segments and required overlap
    # n segments, (n-1) overlaps
    # n * context - (n-1) * overlap = duration
    # overlap = (n * context - duration) / (n - 1)
    
    # Start with minimum segments needed
    n_segments = int(np.ceil(audio_duration / context_duration))
    
    # Calculate overlap to make segments fit exactly
    if n_segments > 1:
        total_segment_time = n_segments * context_duration
        total_overlap_needed = total_segment_time - audio_duration
        overlap_per_gap = total_overlap_needed / (n_segments - 1)
        
        # Ensure minimum overlap
        overlap = max(overlap_per_gap, min_overlap)
        
        # Recalculate with actual overlap
        step = context_duration - overlap
        
        current = 0.0
        while current + context_duration <= audio_duration + 0.001:  # small tolerance
            windows.append((current, current + context_duration))
            current += step
            
            # Safety limit
            if len(windows) > 100:
                break
    else:
        windows.append((0.0, context_duration))
    
    return windows


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from ONC audio filename.
    
    Format: DEVICE_YYYYMMDDTHHMMSSmmm.flac or .wav
    """
    try:
        base = Path(filename).stem
        parts = base.split('_')
        if len(parts) >= 2:
            ts_str = parts[1].replace('Z', '')
            # Handle milliseconds
            if '.' in ts_str:
                ts_str = ts_str.split('.')[0]
            dt = datetime.strptime(ts_str[:15], '%Y%m%dT%H%M%S')
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download sequential audio and generate custom spectrograms"
    )
    # Required arguments
    parser.add_argument('--device-code', type=str, required=True,
                        help='ONC device code (e.g., ICLISTENHF1951)')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date (ISO format, e.g., 2025-01-01T00:00:00Z)')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date (ISO format)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for spectrograms and metadata')
    
    # Config and output options
    parser.add_argument('--config', type=str, default='./config/dataset_config.yaml',
                        help='Path to spectrogram configuration YAML')
    parser.add_argument('--save-audio', action='store_true',
                        help='Save audio segment files')
    parser.add_argument('--save-png', action='store_true',
                        help='Save PNG spectrogram images')
    
    # Custom spectrogram parameters (override config)
    parser.add_argument('--context-duration', type=float, default=None,
                        help='Segment duration in seconds (default: from config)')
    parser.add_argument('--temporal-padding', type=float, default=2.0,
                        help='Seconds of padding before/after segment to eliminate edge effects (default: 2.0)')
    parser.add_argument('--window-duration', type=float, default=None,
                        help='Spectrogram window duration in seconds')
    parser.add_argument('--overlap', type=float, default=None,
                        help='Spectrogram overlap ratio (0-1)')
    parser.add_argument('--freq-min', type=float, default=None,
                        help='Minimum frequency in Hz')
    parser.add_argument('--freq-max', type=float, default=None,
                        help='Maximum frequency in Hz')
    parser.add_argument('--colormap', type=str, default=None,
                        help='Colormap for spectrograms')
    parser.add_argument('--segment-overlap', type=float, default=0.5,
                        help='Minimum overlap between segments in seconds')
    
    # Processing options
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel download workers')
    
    args = parser.parse_args()

    print_header("DEPRECATED SCRIPT")
    print_status(
        "download_sequential_audio.py is deprecated and reflects an older, "
        "incorrect preprocessing path. Use scripts/data/test/prepare_test_windows.py instead.",
        "WARNING",
    )
    sys.exit(1)
    
    # Load environment
    load_dotenv()
    onc_token = os.getenv('ONC_TOKEN')
    if not onc_token:
        print_status("Error: ONC_TOKEN not found in .env file", "ERROR")
        sys.exit(1)
    
    # Parse dates
    start_dt = parse_datetime(args.start_date)
    end_dt = parse_datetime(args.end_date)
    
    print_header("SEQUENTIAL AUDIO DOWNLOAD & SPECTROGRAM GENERATION")
    print(f"Device: {args.device_code}")
    print(f"Date range: {start_dt.isoformat()} to {end_dt.isoformat()}")
    print(f"Output: {args.output_dir}")
    print(f"Save audio: {args.save_audio}, Save PNG: {args.save_png}")
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    mat_dir = output_dir / "mat_files"
    mat_dir.mkdir(parents=True, exist_ok=True)
    
    png_dir = output_dir / "spectrograms" if args.save_png else None
    audio_dir = output_dir / "audio" if args.save_audio else None
    
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and apply CLI overrides
    config = load_config(args.config)
    spec_cfg = config.get('custom_spectrograms', {})
    ctx_cfg = config.get('temporal_context', {})
    
    # Context duration for segments
    context_duration = args.context_duration or ctx_cfg.get('context_duration', 40.0)
    
    # Spectrogram parameters (CLI overrides config)
    freq_lims = spec_cfg.get('frequency_limits', {'min': 5, 'max': 100})
    freq_min = args.freq_min if args.freq_min is not None else freq_lims.get('min', 5)
    freq_max = args.freq_max if args.freq_max is not None else freq_lims.get('max', 100)
    
    color_lims = spec_cfg.get('color_limits', {'min': -60, 'max': 0})
    
    window_duration = args.window_duration if args.window_duration is not None else spec_cfg.get('window_duration', 1.0)
    overlap = args.overlap if args.overlap is not None else spec_cfg.get('overlap', 0.9)
    colormap = args.colormap if args.colormap is not None else spec_cfg.get('colormap', 'viridis')
    
    print(f"\nSpectrogram config:")
    print(f"  Context duration: {context_duration}s")
    print(f"  Temporal padding: {args.temporal_padding}s (before/after each segment)")
    print(f"  Window duration: {window_duration}s, Overlap: {overlap}")
    print(f"  Frequency range: {freq_min}-{freq_max} Hz")
    
    # Initialize spectrogram generator with final parameters
    spec_gen = SpectrogramGenerator(
        win_dur=window_duration,
        overlap=overlap,
        freq_lims=(freq_min, freq_max),
        log_freq=spec_cfg.get('log_frequency', False),
        clim=(color_lims.get('min', -60), color_lims.get('max', 0)),
        colormap=colormap
    )
    
    # Initialize downloader
    downloader = HydrophoneDownloader(onc_token, str(output_dir))
    
    # Download audio files
    print_header("DOWNLOADING AUDIO FILES")
    
    try:
        # ONC API expects ISO format: YYYY-MM-DDTHH:MM:SS.000Z
        start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        end_str = end_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # Try FLAC first, fall back to WAV if no FLAC files found
        downloader.download_flac_files(
            args.device_code,
            start_str,
            end_str
        )
        
        # Check if any files were downloaded
        import glob
        flac_files = glob.glob(str(output_dir / '**/*.flac'), recursive=True)
        
        if not flac_files:
            print_status("No FLAC files found, trying WAV format...", "PROGRESS")
            
            # Download WAV files using ONC client directly
            from onc import ONC
            onc_client = ONC(onc_token)
            
            filters = {
                'deviceCode': args.device_code,
                'dateFrom': start_str,
                'dateTo': end_str,
                'extension': 'wav'
            }
            
            wav_result = onc_client.getListByDevice(filters)
            wav_files = [f for f in wav_result.get('files', []) if f.endswith('.wav')]
            
            if wav_files:
                print(f"Found {len(wav_files)} WAV files to download")
                wav_dir = output_dir / 'wav'
                wav_dir.mkdir(parents=True, exist_ok=True)
                onc_client.outPath = str(wav_dir)
                
                for wav_file in wav_files:
                    try:
                        onc_client.getFile(wav_file)
                    except Exception as e:
                        print_status(f"Failed to download {wav_file}: {e}", "WARNING")
                
                print_status(f"Downloaded WAV files to {wav_dir}", "SUCCESS")
            else:
                print_status("No WAV files found either", "WARNING")
        
        print_status("Audio download complete", "SUCCESS")
    except Exception as e:
        print_status(f"Download error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Find downloaded audio files
    import glob
    audio_patterns = [
        str(output_dir / '**/*.flac'),
        str(output_dir / '**/*.wav'),
    ]
    audio_files = []
    for pattern in audio_patterns:
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    # Exclude our output audio dir
    audio_files = [f for f in audio_files if '/audio/' not in f or not args.save_audio]
    audio_files = sorted(audio_files)
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Track processed files
    processed_files = []
    failed_files = []
    
    print_header("GENERATING SPECTROGRAMS")
    
    for audio_idx, audio_path in enumerate(audio_files):
        audio_path = Path(audio_path)
        print_status(f"Processing {audio_idx + 1}/{len(audio_files)}: {audio_path.name}", "PROGRESS")
        
        try:
            # Load audio
            audio_data, sample_rate = sf.read(str(audio_path))
            audio_duration = len(audio_data) / sample_rate
            
            # Extract timestamp from filename
            file_timestamp = extract_timestamp_from_filename(audio_path.name)
            if file_timestamp is None:
                file_timestamp = datetime.now(timezone.utc)
            
            # Compute segment windows
            windows = compute_segment_windows(
                audio_duration, 
                context_duration,
                min_overlap=args.segment_overlap
            )
            
            print(f"  Audio: {audio_duration:.1f}s @ {sample_rate}Hz, {len(windows)} segments")
            
            for seg_idx, (start_sec, end_sec) in enumerate(windows):
                # Calculate padded extraction window to eliminate edge effects
                padding = args.temporal_padding
                padded_start_sec = max(0, start_sec - padding)
                padded_end_sec = min(audio_duration, end_sec + padding)
                
                # Extract segment WITH padding
                padded_start_sample = int(padded_start_sec * sample_rate)
                padded_end_sample = int(padded_end_sec * sample_rate)
                padded_segment = audio_data[padded_start_sample:padded_end_sample]
                
                # Track actual padding applied (may be less at boundaries)
                actual_padding_before = start_sec - padded_start_sec
                actual_padding_after = padded_end_sec - end_sec
                
                # Generate spectrogram on PADDED segment
                freqs, times, Sxx, power_db = spec_gen.compute_spectrogram(padded_segment, sample_rate)
                
                # Trim spectrogram to remove padding (keep only target time range)
                # Find time indices that fall within [actual_padding_before, actual_padding_before + context_duration]
                target_start_time = actual_padding_before
                target_end_time = actual_padding_before + context_duration
                
                # Create mask for target time range
                target_mask = (times >= target_start_time) & (times <= target_end_time)
                time_indices = np.where(target_mask)[0]
                
                if len(time_indices) == 0:
                    print(f"    Warning: No valid time indices for segment {seg_idx}, skipping")
                    continue
                
                # Trim arrays to target range
                times_trimmed = times[time_indices] - actual_padding_before  # Shift to start at 0
                Sxx_trimmed = Sxx[:, time_indices]
                power_db_trimmed = power_db[:, time_indices]
                
                # Crop to frequency range
                freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
                freq_indices = np.where(freq_mask)[0]
                if len(freq_indices) > 0:
                    f_start = freq_indices[0]
                    f_end = freq_indices[-1] + 1
                    freqs_cropped = freqs[f_start:f_end]
                    Sxx_cropped = Sxx_trimmed[f_start:f_end, :]
                    power_db_cropped = power_db_trimmed[f_start:f_end, :]
                else:
                    freqs_cropped = freqs
                    Sxx_cropped = Sxx_trimmed
                    power_db_cropped = power_db_trimmed
                
                # Create unique file ID
                file_id = f"{audio_path.stem}_seg{seg_idx:03d}"
                
                # Compute segment timestamp
                seg_timestamp = file_timestamp + timedelta(seconds=start_sec)
                
                # Save MAT file with temporal padding metadata
                mat_path = mat_dir / f"{file_id}.mat"
                scipy.io.savemat(str(mat_path), {
                    'F': freqs_cropped,
                    'T': times_trimmed,
                    'P': Sxx_cropped,
                    'PdB_norm': power_db_cropped,
                    'freq_min': freq_min,
                    'freq_max': freq_max,
                    'sample_rate': sample_rate,
                    'context_duration': context_duration,
                    'temporal_padding_used': padding,
                })
                
                # Save PNG if requested
                png_path = None
                if args.save_png:
                    png_path = png_dir / f"{file_id}.png"
                    spec_gen.plot_spectrogram(
                        freqs_cropped, times_trimmed, power_db_cropped,
                        title=f"{args.device_code}: {seg_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        save_path=png_path
                    )
                    import matplotlib.pyplot as plt
                    plt.close('all')  # Prevent memory buildup from unclosed figures
                
                # Save audio segment if requested (target segment, not padded)
                seg_audio_path = None
                if args.save_audio:
                    # Extract the actual target segment for audio saving
                    target_start_sample = int(start_sec * sample_rate)
                    target_end_sample = int(end_sec * sample_rate)
                    target_segment = audio_data[target_start_sample:target_end_sample]
                    
                    # Ensure exact length
                    expected_samples = int(context_duration * sample_rate)
                    if len(target_segment) < expected_samples:
                        target_segment = np.pad(target_segment, (0, expected_samples - len(target_segment)))
                    elif len(target_segment) > expected_samples:
                        target_segment = target_segment[:expected_samples]
                    
                    seg_audio_path = audio_dir / f"{file_id}.wav"
                    sf.write(str(seg_audio_path), target_segment, sample_rate)
                
                processed_files.append({
                    "file_id": file_id,
                    "source_audio": audio_path.name,
                    "segment_index": seg_idx,
                    "segment_start_sec": start_sec,
                    "segment_end_sec": end_sec,
                    "audio_timestamp": seg_timestamp.isoformat(),
                    "mat_path": str(mat_path.relative_to(output_dir)),
                    "spectrogram_path": str(png_path.relative_to(output_dir)) if png_path else None,
                    "audio_path": str(seg_audio_path.relative_to(output_dir)) if seg_audio_path else None,
                })
                
        except Exception as e:
            failed_files.append({"file": str(audio_path), "error": str(e)})
            print_status(f"Failed: {e}", "WARNING")
            import traceback
            traceback.print_exc()
    
    # Save metadata JSON
    metadata = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": {
            "device_code": args.device_code,
            "date_from": start_dt.isoformat(),
            "date_to": end_dt.isoformat(),
        },
        "spectrogram_config": {
            "window_duration": window_duration,
            "overlap": overlap,
            "frequency_limits": {"min": freq_min, "max": freq_max},
            "context_duration": context_duration,
            "segment_overlap": args.segment_overlap,
            "colormap": colormap,
            "color_limits": color_lims,
        },
        "processing": {
            "total_files": len(processed_files),
            "failed_files": len(failed_files),
            "audio_files_processed": len(audio_files),
            "save_audio": args.save_audio,
            "save_png": args.save_png,
        },
        "files": processed_files,
        "errors": failed_files if failed_files else None,
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print_header("COMPLETE")
    print(f"Processed: {len(processed_files)} segments from {len(audio_files)} audio files")
    print(f"Failed: {len(failed_files)} files")
    print(f"Metadata saved to: {metadata_path}")
    print_status("Dataset generation complete!", "SUCCESS")


if __name__ == "__main__":
    main()
