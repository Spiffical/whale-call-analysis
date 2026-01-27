#!/usr/bin/env python3
"""
Prepare Test Windows: Download and Preprocess Sequential Audio
- Downloads all 5-minute clips between start and end dates.
- Generates 5-minute spectrograms with edge context padding (from adjacent files).
- Chunks spectrograms and audio into CNN-compatible windows for inference.
- Efficient: each clip downloaded once.

Usage:
    python scripts/data/test/prepare_test_windows.py \
        --device-code ICLISTENHF1951 \
        --start-date 2025-01-01T00:00:00Z \
        --end-date 2025-01-01T06:00:00Z \
        --output-dir output/test_windows/ \
        --dataset-documentation /path/to/dataset_documentation.json \
        --model-path /path/to/trained-models/finwhale-cnn-... \
        --save-chunk-audio
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
import scipy.io
import soundfile as sf
from dotenv import load_dotenv

from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader
from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator
from src.dataset.reporting import print_status, print_header
from src.data.sequential_prep import (
    parse_datetime,
    extract_timestamp_from_filename,
    compute_window_positions,
    crop_to_freq_lims,
    load_dataset_documentation,
    get_processing_params,
)

def main():
    parser = argparse.ArgumentParser(description="Prepare inference data from ONC")
    parser.add_argument('--device-code', type=str, required=True, help='ONC device code')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (ISO)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (ISO)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    # Model/Dataset configuration
    parser.add_argument('--model-path', type=str, help='Path to trained model dir containing args.pkl')
    parser.add_argument('--dataset-documentation', type=str, 
                        help='Path to dataset_documentation.json or directory containing it')
    
    # Parameter overrides (highest precedence)
    parser.add_argument('--crop-size', type=int, help='Override crop size (pixels, for square chunks)')
    parser.add_argument('--freq-lims', type=str, help='Override frequency limits as "min,max" Hz (e.g., "5,100")')
    parser.add_argument('--win-dur', type=float, help='Override window duration (seconds)')
    parser.add_argument('--overlap', type=float, help='Override overlap ratio (0-1)')
    
    # Options
    parser.add_argument('--save-raw-audio', action='store_true', help='Save original 5-min clips')
    parser.add_argument('--save-full-spectrogram', action='store_true', help='Save full 5-min spectrograms')
    parser.add_argument('--save-chunk-audio', action='store_true',
                        default=True, help='Save audio clips for each chunk (default: true)')
    parser.add_argument('--no-save-chunk-audio', dest='save_chunk_audio', action='store_false',
                        help='Disable saving audio clips for each chunk')
    parser.add_argument('--edge-padding', type=float, default=2.0, help='Seconds of edge padding')
    parser.add_argument('--config', type=str, default='./config/dataset_config.yaml', help='Config path (fallback)')
    parser.add_argument('--workers', type=int, default=4, help='Download workers')

    args = parser.parse_args()
    load_dotenv()
    onc_token = os.getenv('ONC_TOKEN')
    if not onc_token:
        print_status("Error: ONC_TOKEN not found", "ERROR")
        sys.exit(1)

    start_dt = parse_datetime(args.start_date)
    end_dt = parse_datetime(args.end_date)
    
    # Parse freq-lims if provided
    freq_lims_override = None
    if args.freq_lims:
        try:
            parts = args.freq_lims.split(',')
            freq_lims_override = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            print_status(f"Invalid --freq-lims format: {args.freq_lims}. Use 'min,max' e.g., '5,100'", "ERROR")
            sys.exit(1)
    
    # Load dataset documentation if provided
    dataset_doc = None
    if args.dataset_documentation:
        dataset_doc = load_dataset_documentation(args.dataset_documentation)
        if not dataset_doc:
            print_status(f"Warning: Could not load dataset documentation from {args.dataset_documentation}", "WARNING")
    
    # Get processing parameters with proper precedence
    proc_params = get_processing_params(
        dataset_doc=dataset_doc,
        model_path=args.model_path,
        crop_size_override=args.crop_size,
        freq_lims_override=freq_lims_override,
        win_dur_override=args.win_dur,
        overlap_override=args.overlap,
    )
    
    crop_size = proc_params['crop_size']
    freq_lims = proc_params['freq_lims']
    win_dur = proc_params['win_dur']
    overlap = proc_params['overlap']
    clim = proc_params['clim']

    # Load spectrogram config from yaml as fallback for colormap, log_freq
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)
            spec_cfg = full_config.get('custom_spectrograms', {})
    else:
        spec_cfg = {}

    print_header("PREPARING INFERENCE DATA")
    print(f"Device: {args.device_code}")
    print(f"Date range: {start_dt.isoformat()} to {end_dt.isoformat()}")
    print(f"Crop size: {crop_size} x {crop_size} (square)")
    print(f"Frequency limits: {freq_lims[0]}-{freq_lims[1]} Hz")
    print(f"Window duration: {win_dur}s, Overlap: {overlap}")
    print(f"Color limits: {clim[0]} to {clim[1]} dB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Internal temp download dir (will be cleaned if not save-raw-audio)
    raw_audio_dir = output_dir / "raw_audio"
    raw_audio_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories based on {date}/{device}/{spectrograms|audio}
    def get_device_dir(dt: datetime, device: str) -> Path:
        date_str = dt.strftime('%Y-%m-%d')
        path = output_dir / date_str / device
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_structure_path(dt: datetime, device: str, subfolder: str) -> Path:
        base = get_device_dir(dt, device)
        path = base / subfolder
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Phase 1: Download all audio
    print_header("PHASE 1: DOWNLOADING AUDIO")
    downloader = HydrophoneDownloader(onc_token, str(raw_audio_dir))
    start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_str = end_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    try:
        downloader.download_flac_files(args.device_code, start_str, end_str)
    except Exception as e:
        print_status(f"Download error: {e}", "WARNING")

    # Find and sort downloaded files
    audio_files = sorted(list(raw_audio_dir.glob("**/*.flac")) + list(raw_audio_dir.glob("**/*.wav")))
    if not audio_files:
        print_status("No audio files found!", "ERROR")
        sys.exit(1)
    
    print_status(f"Downloaded {len(audio_files)} files.", "SUCCESS")

    # Phase 2: Process Spectrograms
    print_header("PHASE 2: GENERATING SPECTROGRAMS & CHUNKS")
    
    # Use dynamically loaded parameters
    spec_gen = SpectrogramGenerator(
        win_dur=win_dur,
        overlap=overlap,
        freq_lims=freq_lims,  # Used for plotting but we'll manually crop
        clim=clim,
        colormap=spec_cfg.get('colormap', 'viridis'),
        log_freq=spec_cfg.get('log_frequency', False),
        crop_freq_lims=False,  # We'll do manual cropping for precise control
        quiet=True
    )
    
    processed_chunks = []

    for i, audio_path in enumerate(audio_files):
        print_status(f"Processing {i+1}/{len(audio_files)}: {audio_path.name}", "PROGRESS")
        try:
            # Metadata for current file
            file_ts = extract_timestamp_from_filename(audio_path.name) or datetime.now(timezone.utc)
            
            # Load current, prev, next
            data, fs = sf.read(str(audio_path))
            pad_samples = int(args.edge_padding * fs)
            
            # Stitch with neighbors if available
            buffer = [data]
            offset_seconds = 0.0
            
            if i > 0:
                prev_data, prev_fs = sf.read(str(audio_files[i-1]))
                if prev_fs == fs:
                    buffer.insert(0, prev_data[-pad_samples:])
                    offset_seconds = args.edge_padding
                else:
                    print_status(f"Sample rate mismatch with prev file: {prev_fs} vs {fs}", "WARNING")
            
            if i < len(audio_files) - 1:
                next_data, next_fs = sf.read(str(audio_files[i+1]))
                if next_fs == fs:
                    buffer.append(next_data[:pad_samples])
                else:
                    print_status(f"Sample rate mismatch with next file: {next_fs} vs {fs}", "WARNING")
            
            full_audio = np.concatenate(buffer)
            
            # Generate full 5-min spectrogram (full frequency range)
            freqs, times, Sxx, PdB = spec_gen.compute_spectrogram(
                full_audio, fs, 
                clip_meta={'clip_offset_seconds': offset_seconds, 'clip_duration_seconds': len(data)/fs}
            )
            
            # Apply frequency cropping to match training data
            cropped_freqs, cropped_Sxx = crop_to_freq_lims(freqs, Sxx, freq_lims[0], freq_lims[1])
            _, cropped_PdB = crop_to_freq_lims(freqs, PdB, freq_lims[0], freq_lims[1])
            
            # Log dimensions for first file
            if i == 0:
                print(f"  Full spectrogram: {PdB.shape[0]} freq bins × {PdB.shape[1]} time bins")
                print(f"  After freq crop ({freq_lims[0]}-{freq_lims[1]} Hz): {cropped_PdB.shape[0]} freq bins × {cropped_PdB.shape[1]} time bins")
            
            n_freq_bins = cropped_PdB.shape[0]
            n_time_bins = cropped_PdB.shape[1]
            
            # Verify frequency dimension matches crop_size (for square chunks)
            if n_freq_bins != crop_size:
                print_status(f"Warning: freq bins ({n_freq_bins}) != crop_size ({crop_size}). Adjusting via center crop.", "WARNING")
                # Center crop in frequency dimension if larger, or pad if smaller
                if n_freq_bins > crop_size:
                    start_f = (n_freq_bins - crop_size) // 2
                    cropped_freqs = cropped_freqs[start_f:start_f + crop_size]
                    cropped_Sxx = cropped_Sxx[start_f:start_f + crop_size, :]
                    cropped_PdB = cropped_PdB[start_f:start_f + crop_size, :]
                    n_freq_bins = crop_size
                # If smaller, we can't produce square chunks - warn and continue
            
            # Save full if requested (before chunking)
            if args.save_full_spectrogram:
                full_spec_dir = get_structure_path(file_ts, args.device_code, "full_spectrograms")
                scipy.io.savemat(full_spec_dir / f"{audio_path.stem}.mat", {
                    'F': cropped_freqs, 'T': times, 'P': cropped_Sxx, 'PdB_norm': cropped_PdB, 'fs': fs
                })

            # Tiling / Chunking: create crop_size x crop_size square chunks
            # Tile along time axis
            time_starts = compute_window_positions(n_time_bins, crop_size)
            
            chunk_spec_dir = get_structure_path(file_ts, args.device_code, "spectrograms")
            chunk_audio_dir = get_structure_path(file_ts, args.device_code, "audio") if args.save_chunk_audio else None
            device_dir = get_device_dir(file_ts, args.device_code)

            for win_idx, start_idx in enumerate(time_starts):
                end_idx = min(start_idx + crop_size, n_time_bins)
                
                # Extract square chunk: crop_size (freq) x crop_size (time)
                chunk_PdB = cropped_PdB[:crop_size, start_idx:end_idx]
                chunk_Sxx = cropped_Sxx[:crop_size, start_idx:end_idx]
                chunk_times = times[start_idx:end_idx]
                chunk_freqs = cropped_freqs[:crop_size]
                
                # Ensure time dimension is correct (pad if at edge)
                if chunk_PdB.shape[1] < crop_size:
                    pad_width = crop_size - chunk_PdB.shape[1]
                    chunk_PdB = np.pad(chunk_PdB, ((0, 0), (0, pad_width)), mode='edge')
                    chunk_Sxx = np.pad(chunk_Sxx, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                
                # Create unique chunk ID
                chunk_id = f"{audio_path.stem}_w{win_idx:02d}"
                
                # Save chunked MAT (square: crop_size x crop_size)
                mat_path = chunk_spec_dir / f"{chunk_id}.mat"
                scipy.io.savemat(mat_path, {
                    'F': chunk_freqs, 'T': chunk_times, 'P': chunk_Sxx, 'PdB_norm': chunk_PdB,
                    'fs': fs, 'chunk_start_bin': start_idx,
                    'original_shape': [n_freq_bins, n_time_bins],
                    'chunk_shape': [chunk_PdB.shape[0], chunk_PdB.shape[1]]
                })
                
                # Save chunked audio if requested
                wav_path = None
                if chunk_audio_dir:
                    wav_path = chunk_audio_dir / f"{chunk_id}.wav"
                    # Time to samples
                    start_sec = times[start_idx]
                    end_sec = times[min(end_idx, len(times)-1)] + (times[1] - times[0]) if len(times) > 1 else start_sec + 1
                    s_start = int(start_sec * fs)
                    s_end = int(end_sec * fs)
                    # Extract from the original 'data' (which is exactly the 5-min part)
                    s_start = max(0, min(s_start, len(data)))
                    s_end = max(s_start, min(s_end, len(data)))
                    if s_end > s_start:
                        sf.write(str(wav_path), data[s_start:s_end], fs)

                # Window timing (seconds) relative to the 5-min clip
                # Use edge-based times so the first window starts at 0.0s.
                window_time_start = None
                window_time_end = None
                if len(times) > 0 and len(times) > start_idx:
                    if len(times) > 1:
                        hop_sec = float(times[1] - times[0])
                    else:
                        hop_sec = 0.0
                    window_bins = int(end_idx - start_idx)
                    center_start = float(times[start_idx])
                    window_time_start = max(0.0, center_start - (win_dur / 2.0))
                    window_time_end = window_time_start + max(0, window_bins - 1) * hop_sec + win_dur

                processed_chunks.append({
                    "chunk_id": chunk_id,
                    "date": file_ts.strftime('%Y-%m-%d'),
                    "timestamp": (file_ts + timedelta(seconds=times[start_idx])).isoformat(),
                    "mat_path": str(mat_path.relative_to(device_dir)),
                    "audio_path": str(wav_path.relative_to(device_dir)) if wav_path else None,
                    "chunk_shape": [chunk_PdB.shape[0], chunk_PdB.shape[1]],
                    "original_shape": [int(n_freq_bins), int(n_time_bins)],
                    "window_index": int(win_idx),
                    "window_start": int(start_idx),
                    "window_time_start": window_time_start,
                    "window_time_end": window_time_end,
                    "source_audio": audio_path.name,
                })

        except Exception as e:
            print_status(f"Failed to process {audio_path.name}: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    # Save metadata with full processing parameters
    metadata = {
        "device_code": args.device_code,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "processing_parameters": {
            "crop_size": crop_size,
            "freq_lims_hz": list(freq_lims),
            "win_dur_s": win_dur,
            "overlap": overlap,
            "clim_db": list(clim),
        },
        "spectrogram_source": {
            "type": "computed",
            "generator": "onc_hydrophone_data.SpectrogramGenerator",
        },
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_documentation_source": args.dataset_documentation,
        "model_path": args.model_path,
        "chunks": processed_chunks
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Cleanup raw audio if not requested
    if not args.save_raw_audio:
        print_status("Cleaning up raw audio files...", "PROGRESS")
        import shutil
        shutil.rmtree(raw_audio_dir)

    print_header("PREPARATION COMPLETE")
    print(f"Generated {len(processed_chunks)} chunks for inference.")
    print_status(f"Project directory: {args.output_dir}", "SUCCESS")

if __name__ == "__main__":
    main()
