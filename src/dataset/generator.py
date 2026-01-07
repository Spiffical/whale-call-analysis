"""
Spectrogram Dataset Generator

Generates training datasets from whale call annotations by:
1. Loading configuration from YAML
2. Downloading ONC audio files
3. Generating frequency-cropped spectrograms (MAT files for training, PNGs for visualization)
4. Optionally generating negative (no-call) samples
"""

import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
import yaml
from onc import ONC
import soundfile as sf

from src.dataset.reporting import print_status, print_header
from src.dataset.audio import stitch_audio_files
from src.dataset.negative_sampler import sample_negative_windows_for_file

# External dependencies
from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader
from onc_hydrophone_data.audio.spectrogram_generator import SpectrogramGenerator


class SpectrogramDatasetGenerator:
    """
    Generates spectrogram datasets from whale call annotations.
    
    Handles:
    - Configuration loading from YAML
    - ONC API connection for audio downloads
    - Spectrogram computation and frequency cropping
    - MAT file generation for training
    - Optional negative sample generation
    """
    
    def __init__(self, 
                 onc_token: str, 
                 excel_file: Optional[str] = None,
                 config_path: str = "./config/dataset_config.yaml",
                 excel_files: Optional[List[str]] = None):
        """Initialize the generator with ONC credentials and configuration.
        
        Args:
            onc_token: ONC API token for data downloads
            excel_file: Single Excel file with whale call annotations
            config_path: Path to YAML configuration file
            excel_files: List of Excel files (alternative to excel_file)
        """
        self.onc = ONC(onc_token)
        self.onc_token = onc_token
        
        # Determine Excel files to process
        if excel_files:
            self.excel_files = excel_files
        elif excel_file:
            self.excel_files = [excel_file]
        else:
            self.excel_files = []
            
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize sub-modules from onc_hydrophone_data
        self.downloader = HydrophoneDownloader(onc_token, ".")
        
        # Setup spectrogram generator using config
        spec_cfg = self.config.get('custom_spectrograms', {})
        freq_lims = spec_cfg.get('frequency_limits', {'min': 5, 'max': 100})
        color_lims = spec_cfg.get('color_limits', {'min': -60, 'max': 0})
        
        self.spectrogram_generator = SpectrogramGenerator(
            win_dur=spec_cfg.get('window_duration', 0.1),
            overlap=spec_cfg.get('overlap', 0.9),
            freq_lims=(freq_lims.get('min', 5), freq_lims.get('max', 100)),
            log_freq=spec_cfg.get('log_frequency', False),
            clim=(color_lims.get('min', -60), color_lims.get('max', 0)),
            colormap=spec_cfg.get('colormap', 'viridis')
        )
        
        self.whale_data = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            print_status(f"Warning: Could not load config from {config_path}: {e}. Using defaults.", "WARNING")
            return {}

    def _create_safe_call_id(self, clip_id: str, call: pd.Series) -> str:
        """Create a safe call ID for filenames."""
        try:
            begin = float(call['begin time (s)'])
            end = float(call['end time (s)'])
            return f"{clip_id}_{begin:.1f}s_{end:.1f}s".replace('.wav', '').replace(':', '-').replace(' ', '_')
        except Exception:
            return None

    def generate_spectrograms(self,
                              whale_calls: pd.DataFrame,
                              output_dir: Path,
                              **kwargs) -> Tuple[Dict[str, str], List[Dict], Optional[Tuple[int, int]]]:
        """
        Generate spectrograms for whale calls.
        
        Args:
            whale_calls: DataFrame with whale call annotations
            output_dir: Output directory for spectrograms
            **kwargs: Additional options:
                - max_workers: Number of parallel workers (default: 2)
                - cleanup_audio: Delete audio after processing (default: False)
                - ml_context: Context duration in seconds (default: from config)
                - generate_positives: Generate positive samples (default: True)
                - generate_negatives: Generate negative samples (default: False)
                - negatives_per_call: Number of negatives per call (default: 1)
                - neg_margin: Margin around calls for negatives (default: 2.0)
        
        Returns:
            Tuple of (spectrogram_files dict, failed_calls list, dimensions tuple)
        """
        print_header("GENERATING SPECTROGRAMS")
        
        # Processing parameters from config or kwargs
        max_workers = kwargs.get('max_workers', 2)
        cleanup_audio = kwargs.get('cleanup_audio', False)
        
        ctx_cfg = self.config.get('temporal_context', {})
        ml_context = kwargs.get('ml_context', ctx_cfg.get('context_duration', 40.0))
        
        generate_positives = kwargs.get('generate_positives', True)
        generate_negatives = kwargs.get('generate_negatives', False)
        negatives_per_call = kwargs.get('negatives_per_call', 1)
        neg_margin = kwargs.get('neg_margin', 2.0)
        
        spectrogram_files = {}
        failed_calls = []
        actual_dimensions = None
        
        output_dir = Path(output_dir)
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Group calls by audio file
        file_groups = list(whale_calls.groupby('clip id'))
        total_files = len(file_groups)
        
        # Build calls-by-file map for negative sampling logic
        calls_by_file = {clip: list(zip(df['begin time (s)'], df['end time (s)'])) for clip, df in file_groups}

        # Directories for outputs
        png_dir = output_dir / "png_files"
        mat_dir = output_dir / "mat_files"
        neg_png_dir = output_dir / "neg_png_files"
        neg_mat_dir = output_dir / "neg_mat_files"
        
        for d in [png_dir, mat_dir, neg_png_dir, neg_mat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        def _process_file(clip_id, calls_in_file, idx):
            thread_id = threading.current_thread().name
            print_status(f"[{thread_id}] File {idx}/{total_files}: {clip_id}", "PROGRESS")
            
            local_failed = []
            local_specs = {}
            local_dims = None
            
            try:
                # 1. Ensure audio is downloaded
                audio_path = audio_dir / clip_id
                if not audio_path.exists():
                    # Thread-safe download using a local ONC client
                    local_onc = ONC(self.onc_token)
                    local_onc.outPath = str(audio_dir)
                    local_onc.getFile(clip_id)
                
                if not audio_path.exists():
                    raise FileNotFoundError(f"Failed to download {clip_id}")

                # Use original sample rate from file
                with sf.SoundFile(audio_path) as f:
                    fs = f.samplerate

                # 2. Process Positive Detections
                if generate_positives:
                    for _, call in calls_in_file.iterrows():
                        call_id = self._create_safe_call_id(clip_id, call)
                        try:
                            # Context window calculation
                            begin = call['begin time (s)']
                            end = call['end time (s)']
                            padding = (ml_context - (end - begin)) / 2
                            
                            # Retrieve stitched audio
                            audio_data = stitch_audio_files(
                                self.onc_token, clip_id, call['device_code'],
                                begin - padding, end + padding, ml_context, audio_dir
                            )
                            
                            if audio_data is not None:
                                # Generate and save
                                res = self._generate_and_save(
                                    audio_data, fs, call_id, png_dir, mat_dir
                                )
                                if res:
                                    local_specs[call_id] = str(res)
                                    if local_dims is None:
                                        # Compute dims once for the report
                                        f_bins, t_bins, _, _ = self.spectrogram_generator.compute_spectrogram(audio_data, fs)
                                        local_dims = (len(f_bins), len(t_bins))
                        except Exception as e:
                            local_failed.append({'call_id': call_id, 'clip_id': clip_id, 'reason': str(e)})

                # 3. Process Negative Samples
                if generate_negatives:
                    neg_windows = sample_negative_windows_for_file(
                        clip_id, 300.0, ml_context, calls_by_file, 
                        len(calls_in_file) * negatives_per_call, margin=neg_margin
                    )
                    for n_idx, (start, end) in enumerate(neg_windows):
                        neg_id = f"{clip_id}_neg_{n_idx}"
                        try:
                            audio_data = stitch_audio_files(
                                self.onc_token, clip_id, calls_in_file.iloc[0]['device_code'],
                                start, end, ml_context, audio_dir
                            )
                            if audio_data is not None:
                                res = self._generate_and_save(
                                    audio_data, fs, neg_id, neg_png_dir, neg_mat_dir
                                )
                                if res:
                                    local_specs[neg_id] = str(res)
                        except Exception as e:
                            local_failed.append({'call_id': neg_id, 'clip_id': clip_id, 'reason': str(e)})

            except Exception as e:
                print_status(f"Error processing {clip_id}: {e}", "ERROR")
            finally:
                if cleanup_audio:
                    try: 
                        audio_path = audio_dir / clip_id
                        if audio_path.exists(): audio_path.unlink()
                    except: pass
            
            return local_specs, local_failed, local_dims

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_file, cid, df, i+1) for i, (cid, df) in enumerate(file_groups)]
            for future in concurrent.futures.as_completed(futures):
                s, f, d = future.result()
                spectrogram_files.update(s)
                failed_calls.extend(f)
                if d: actual_dimensions = d
                
        return spectrogram_files, failed_calls, actual_dimensions

    def _generate_and_save(self, 
                          audio_data: np.ndarray, 
                          fs: float, 
                          call_id: str, 
                          png_dir: Path, 
                          mat_dir: Path) -> Optional[Path]:
        """Generate and save spectrogram.
        
        The MAT file is saved with frequency-cropped data so training doesn't need to re-crop.
        """
        try:
            import scipy.io
            
            # 1. Compute full spectrogram
            freqs, times, Sxx, power_db_norm = self.spectrogram_generator.compute_spectrogram(audio_data, fs)
            
            # 2. Crop to frequency range from config
            spec_cfg = self.config.get('custom_spectrograms', {})
            freq_lims = spec_cfg.get('frequency_limits', {'min': 5, 'max': 100})
            freq_min = freq_lims.get('min', 5)
            freq_max = freq_lims.get('max', 100)
            
            # Find frequency indices
            freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
            freq_indices = np.where(freq_mask)[0]
            
            if len(freq_indices) > 0:
                f_start = freq_indices[0]
                f_end = freq_indices[-1] + 1
                freqs_cropped = freqs[f_start:f_end]
                Sxx_cropped = Sxx[f_start:f_end, :]
                power_db_cropped = power_db_norm[f_start:f_end, :]
            else:
                # Fallback to full range if no bins match
                freqs_cropped = freqs
                Sxx_cropped = Sxx
                power_db_cropped = power_db_norm
            
            # 3. Save PNG if enabled (use cropped data)
            if spec_cfg.get('output_formats', {}).get('plots', True):
                png_path = png_dir / f"{call_id}.png"
                self.spectrogram_generator.plot_spectrogram(
                    freqs_cropped, times, power_db_cropped, title=f"Whale Call: {call_id}", save_path=png_path
                )
            
            # 4. Save MAT with CROPPED data so training data is already frequency-limited
            if spec_cfg.get('output_formats', {}).get('matlab', True):
                mat_path = mat_dir / f"{call_id}.mat"
                scipy.io.savemat(str(mat_path), {
                    'F': freqs_cropped,
                    'T': times,
                    'P': Sxx_cropped,
                    'PdB_norm': power_db_cropped,
                    'freq_min': freq_min,
                    'freq_max': freq_max,
                })
            
            # Return path to PNG if it exists, else MAT
            if (png_dir / f"{call_id}.png").exists():
                return png_dir / f"{call_id}.png"
            return mat_dir / f"{call_id}.mat"
        except Exception as e:
            print_status(f"Generation failed for {call_id}: {e}", "WARNING")
            return None
