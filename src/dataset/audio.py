#!/usr/bin/env python3
import os
import threading
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from onc import ONC
from src.dataset.reporting import print_status

def cleanup_audio_files(downloaded_files: Dict[str, str]) -> int:
    """Clean up downloaded audio files to save disk space."""
    deleted_count = 0
    total_size_mb = 0
    
    for clip_id, file_path in downloaded_files.items():
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
                total_size_mb += file_size_mb
                file_path_obj.unlink()
                deleted_count += 1
            else:
                print_status(f"⚠️ File not found for cleanup: {clip_id}", "WARNING")
        except Exception as e:
            print_status(f"❌ Error deleting {clip_id}: {e}", "ERROR")
    
    if deleted_count > 0:
        print_status(f"🗑️ Cleaned up {deleted_count} audio files, freed {total_size_mb:.1f} MB", "SUCCESS")
    return deleted_count

def get_adjacent_filenames(clip_id: str, device_code: str) -> Tuple[Optional[str], Optional[str]]:
    """Estimate previous and next filenames based on the 5-minute naming convention."""
    try:
        # Expected format: DEVICE_YYYYMMDDTHHMMSS.mmmZ.wav
        base_name = clip_id.replace('.wav', '')
        parts = base_name.split('_')
        if len(parts) < 2:
            return None, None
            
        time_str = parts[1]
        # Remove 'Z' for parsing
        dt = datetime.strptime(time_str.replace('Z', ''), '%Y%m%dT%H%M%S.%f')
        
        # ONC files are typically 5 minutes (300 seconds)
        prev_dt = dt - timedelta(minutes=5)
        next_dt = dt + timedelta(minutes=5)
        
        prev_filename = f"{device_code}_{prev_dt.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
        next_filename = f"{device_code}_{next_dt.strftime('%Y%m%dT%H%M%S.%f')[:-3]}Z.wav"
        
        return prev_filename, next_filename
    except Exception:
        return None, None

def download_adjacent_file(onc: ONC, device_code: str, timestamp: datetime, audio_dir: Path) -> Optional[Path]:
    """Download an adjacent audio file if needed."""
    try:
        filters = {
            'deviceCode': device_code,
            'dateFrom': (timestamp - timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'dateTo': (timestamp + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'extension': 'wav'
        }
        
        result = onc.getListByDevice(filters)
        if 'files' in result and result['files']:
            filename = result['files'][0]
            target_path = audio_dir / filename
            if not target_path.exists():
                onc.getFile(filename)
            return target_path
    except Exception as e:
        print_status(f"Error downloading adjacent file for {timestamp}: {e}", "WARNING")
    return None

def stitch_audio_files(
    onc_token: str,
    clip_id: str,
    device_code: str,
    desired_start: float,
    desired_end: float,
    context_duration: float,
    audio_dir: Path
) -> Optional[np.ndarray]:
    """Stitch audio files when context window spans multiple files."""
    try:
        main_file_path = audio_dir / clip_id
        if not main_file_path.exists():
            return None
            
        with sf.SoundFile(main_file_path) as f:
            sample_rate = f.samplerate
            total_frames = len(f)
            main_duration = total_frames / sample_rate
            
        # Case 1: Simple within-file window (with safety margins)
        if desired_start >= 0 and desired_end <= main_duration:
            with sf.SoundFile(main_file_path) as f:
                start_frame = int(desired_start * sample_rate)
                end_frame = int(desired_end * sample_rate)
                
                # Double check bounds
                start_frame = max(0, min(start_frame, total_frames - 1))
                count = max(0, min(end_frame - start_frame, total_frames - start_frame))
                
                f.seek(start_frame)
                return f.read(count)
                
        # Case 2: Spans across files
        thread_onc = ONC(onc_token)
        thread_onc.outPath = str(audio_dir)
        
        # Get timestamp of current file to find neighbors
        time_str = clip_id.replace('.wav', '').split('_')[1].replace('Z', '')
        dt = datetime.strptime(time_str, '%Y%m%dT%H%M%S.%f')
        
        full_audio_list = []
        
        # Handle prefix (previous file)
        if desired_start < 0:
            prev_path = download_adjacent_file(thread_onc, device_code, dt - timedelta(minutes=5), audio_dir)
            if prev_path and prev_path.exists():
                with sf.SoundFile(prev_path) as f:
                    p_fs = f.samplerate
                    p_frames = len(f)
                    p_dur = p_frames / p_fs
                    needed_prev = abs(desired_start)
                    p_start_frame = int(max(0, p_dur - needed_prev) * p_fs)
                    p_start_frame = min(p_start_frame, p_frames - 1)
                    f.seek(p_start_frame)
                    full_audio_list.append(f.read())
            else:
                full_audio_list.append(np.zeros(int(abs(desired_start) * sample_rate)))
                
        # Add main file content
        start_in_main = max(0, desired_start)
        end_in_main = min(main_duration, desired_end)
        with sf.SoundFile(main_file_path) as f:
            m_start_frame = int(start_in_main * sample_rate)
            m_start_frame = max(0, min(m_start_frame, total_frames - 1))
            m_end_frame = int(end_in_main * sample_rate)
            m_count = max(0, min(m_end_frame - m_start_frame, total_frames - m_start_frame))
            f.seek(m_start_frame)
            full_audio_list.append(f.read(m_count))
                
        # Handle suffix (next file)
        if desired_end > main_duration:
            next_path = download_adjacent_file(thread_onc, device_code, dt + timedelta(minutes=5), audio_dir)
            if next_path and next_path.exists():
                with sf.SoundFile(next_path) as f:
                    needed_next = desired_end - main_duration
                    n_fs = f.samplerate
                    n_count = int(needed_next * n_fs)
                    n_count = min(n_count, len(f))
                    full_audio_list.append(f.read(n_count))
            else:
                needed_next = desired_end - main_duration
                full_audio_list.append(np.zeros(int(needed_next * sample_rate)))
                
        # Combine all chunks
        if not full_audio_list:
            return None
        full_audio = np.concatenate(full_audio_list)
                
        # Final length check and trim/pad to exact duration
        expected_samples = int(context_duration * sample_rate)
        if len(full_audio) > expected_samples:
            full_audio = full_audio[:expected_samples]
        elif len(full_audio) < expected_samples:
            full_audio = np.pad(full_audio, (0, expected_samples - len(full_audio)))
            
        return full_audio
        
    except Exception as e:
        print_status(f"Stitching failed for {clip_id}: {e}", "WARNING")
        # Log more details for debugging
        import traceback
        traceback.print_exc()
        return None
