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
from PIL import Image
import matplotlib
import matplotlib.cm as cm

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
                 excel_files: Optional[List[str]] = None,
                 show_onc_warnings: bool = False):
        """Initialize the generator with ONC credentials and configuration.
        
        Args:
            onc_token: ONC API token for data downloads
            excel_file: Single Excel file with whale call annotations
            config_path: Path to YAML configuration file
            excel_files: List of Excel files (alternative to excel_file)
        """
        self.show_onc_warnings = show_onc_warnings
        self.onc = ONC(onc_token, showWarning=show_onc_warnings)
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
        try:
            self.downloader.onc.showWarning = show_onc_warnings
        except Exception:
            pass
        try:
            self.downloader.request_manager.onc.showWarning = show_onc_warnings
        except Exception:
            pass
        
        # Setup spectrogram generator using config
        self._init_spectrogram_generator()
        
        self.whale_data = None
        self.last_clip_id = None
        self.last_file_index = None
        
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

    def _init_spectrogram_generator(self) -> None:
        """Initialize spectrogram generator from current config."""
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

    def probe_spectrogram_backend(self, sample_rate: int = 64000) -> Dict[str, Any]:
        """Probe which spectrogram backend (torch vs scipy) is actually used."""
        sg = self.spectrogram_generator
        info: Dict[str, Any] = {
            "backend_requested": getattr(sg, "backend", None),
            "backend_used": None,
            "backend_device": None,
            "scaling_used": None,
            "probe_sample_rate_hz": sample_rate,
        }
        try:
            win_length, nfft, _, _ = sg._resolve_fft_params(sample_rate)
            audio_len = max(int(win_length), int(nfft), 1)
            dummy_audio = np.zeros(audio_len, dtype=np.float32)
            sg.compute_spectrogram(dummy_audio, sample_rate)
            info["backend_used"] = getattr(sg, "_last_backend", None)
            info["backend_device"] = getattr(sg, "_last_device", None)
            info["scaling_used"] = getattr(sg, "_last_scaling", None)
        except Exception as exc:
            info["backend_error"] = str(exc)
        return info

    def apply_overrides(
        self,
        win_dur: Optional[float] = None,
        overlap: Optional[float] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        ml_context: Optional[float] = None
    ) -> None:
        """Apply config overrides and rebuild the spectrogram generator if needed."""
        updated_spec = False
        if win_dur is not None:
            self.config.setdefault('custom_spectrograms', {})['window_duration'] = float(win_dur)
            updated_spec = True
        if overlap is not None:
            self.config.setdefault('custom_spectrograms', {})['overlap'] = float(overlap)
            updated_spec = True
        if freq_range is not None:
            if len(freq_range) != 2:
                raise ValueError("freq_range must be (min, max)")
            freq_min, freq_max = float(freq_range[0]), float(freq_range[1])
            if freq_min >= freq_max:
                raise ValueError("freq_range min must be < max")
            self.config.setdefault('custom_spectrograms', {})['frequency_limits'] = {
                'min': freq_min,
                'max': freq_max,
            }
            updated_spec = True
        if ml_context is not None:
            self.config.setdefault('temporal_context', {})['context_duration'] = float(ml_context)

        if updated_spec:
            self._init_spectrogram_generator()

    def _create_safe_call_id(self, clip_id: str, call: pd.Series) -> str:
        """Create a safe call ID for filenames."""
        try:
            begin = float(call['begin time (s)'])
            end = float(call['end time (s)'])
            return f"{clip_id}_{begin:.1f}s_{end:.1f}s".replace('.wav', '').replace(':', '-').replace(' ', '_')
        except Exception:
            return None

    @staticmethod
    def _apply_contrast(x01: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
        """Stretch contrast using percentile clipping."""
        lo = np.percentile(x01, pmin)
        hi = np.percentile(x01, pmax)
        if hi <= lo:
            return x01
        y = (x01 - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    @staticmethod
    def _to_colormap_rgb(x01: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
        """Map a normalized array to RGB using a matplotlib colormap."""
        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = cm.get_cmap(cmap_name)
        return (cmap(x01)[..., :3] * 255.0).astype(np.uint8)

    def _save_png_test_style(
        self,
        power_db: np.ndarray,
        save_path: Path,
        scale: int = 3,
        cmap: str = "inferno",
        pmin: float = 2.0,
        pmax: float = 98.0,
    ) -> None:
        """Save PNG in the same visual style as scripts/train/test_cnn.py."""
        arr = np.asarray(power_db, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr_min = float(arr.min()) if arr.size else 0.0
        arr_max = float(arr.max()) if arr.size else 1.0
        if arr_max > arr_min:
            arr01 = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr01 = np.zeros_like(arr, dtype=np.float32)
        arr01 = self._apply_contrast(arr01, pmin=pmin, pmax=pmax)
        rgb = self._to_colormap_rgb(arr01, cmap_name=cmap)
        img = Image.fromarray(rgb)
        if scale > 1:
            width, height = img.size
            img = img.resize((width * int(scale), height * int(scale)), resample=Image.BICUBIC)
        img.save(str(save_path))

    def generate_spectrograms(self,
                              whale_calls: pd.DataFrame,
                              output_dir: Path,
                              show_progress: bool = True,
                              edge_context: float = 0.0,
                              audio_cache_dir: Optional[Path] = None,
                              **kwargs) -> Tuple[Dict[str, str], List[Dict], Optional[Tuple[int, int]]]:
        """
        Generate spectrograms for whale calls.
        
        Args:
            whale_calls: DataFrame with whale call annotations
            output_dir: Output directory for spectrograms
            audio_cache_dir: Optional directory to cache downloaded audio files
            **kwargs: Additional options:
                - max_workers: Number of parallel workers (default: 2)
                - cleanup_audio: Delete audio after processing (default: False)
                - ml_context: Context duration in seconds (default: from config)
                - show_progress: Display a progress bar (default: True)
                - edge_context: Seconds of extra context on each side (trimmed after spectrogram)
                - generate_positives: Generate positive samples (default: True)
                - generate_negatives: Generate negative samples (default: False)
                - negatives_per_call: Number of negatives per call (default: 1)
                - neg_margin: Margin around calls for negatives (default: 2.0)
                - neg_context: Context duration for negatives (default: ml_context)
                - neg_strategy: 'random' or 'tiled' (default: random)
                - neg_step_seconds: Step for tiled negatives (default: context duration)
                - max_negatives_per_file: Optional cap per source clip
                - existing_policy: 'overwrite' (default) or 'skip'
        
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
        neg_context = kwargs.get('neg_context', None)
        neg_strategy = kwargs.get('neg_strategy', 'random')
        neg_step_seconds = kwargs.get('neg_step_seconds', None)
        max_negatives_per_file = kwargs.get('max_negatives_per_file', None)
        allow_audio_download = kwargs.get('allow_audio_download', True)
        png_style = kwargs.get('png_style', 'test')
        png_scale = kwargs.get('png_scale', 3)
        png_cmap = kwargs.get('png_cmap', 'inferno')
        png_pmin = kwargs.get('png_pmin', 2.0)
        png_pmax = kwargs.get('png_pmax', 98.0)
        existing_policy = str(kwargs.get('existing_policy', 'overwrite')).strip().lower()
        if existing_policy not in {"overwrite", "skip"}:
            raise ValueError("existing_policy must be 'overwrite' or 'skip'")
        if neg_context is None:
            neg_context = ml_context
        
        spectrogram_files = {}
        failed_calls = []
        actual_dimensions = None
        
        output_dir = Path(output_dir)
        audio_dir = Path(audio_cache_dir) if audio_cache_dir else (output_dir / "audio")
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

        spec_cfg = self.config.get('custom_spectrograms', {})
        save_png = bool(spec_cfg.get('output_formats', {}).get('plots', True))
        save_mat = bool(spec_cfg.get('output_formats', {}).get('matlab', True))
        skipped_existing_total = 0
        skipped_lock = threading.Lock()

        def _existing_output_path(call_id: str, out_png_dir: Path, out_mat_dir: Path) -> Optional[Path]:
            png_path = out_png_dir / f"{call_id}.png"
            mat_path = out_mat_dir / f"{call_id}.mat"
            # Prefer the format that is enabled by config, but still accept either existing file.
            if save_mat and mat_path.exists():
                return mat_path
            if save_png and png_path.exists():
                return png_path
            if mat_path.exists():
                return mat_path
            if png_path.exists():
                return png_path
            return None

        def _process_file(clip_id, calls_in_file, idx):
            nonlocal skipped_existing_total
            self.last_clip_id = clip_id
            self.last_file_index = idx
            thread_id = threading.current_thread().name
            print_status(f"[{thread_id}] File {idx}/{total_files}: {clip_id}", "PROGRESS")
            
            local_failed = []
            local_specs = {}
            local_dims = None
            
            try:
                # 1. Ensure audio is downloaded
                audio_path = audio_dir / clip_id
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    print_status(f"[{thread_id}] Using cached audio: {clip_id}", "INFO")
                else:
                    if allow_audio_download:
                        # Thread-safe download using a local ONC client
                        local_onc = ONC(self.onc_token, showWarning=self.show_onc_warnings)
                        local_onc.outPath = str(audio_dir)
                        local_onc.getFile(clip_id)
                    else:
                        raise FileNotFoundError(
                            f"Missing local audio {clip_id} and downloads are disabled (--no-audio-download)"
                        )
                
                if not audio_path.exists():
                    raise FileNotFoundError(f"Failed to download {clip_id}")

                # Use original sample rate from file
                with sf.SoundFile(audio_path) as f:
                    fs = f.samplerate

                # 2. Process Positive Detections
                if generate_positives:
                    for _, call in calls_in_file.iterrows():
                        call_id = self._create_safe_call_id(clip_id, call)
                        if existing_policy == "skip":
                            existing_path = _existing_output_path(call_id, png_dir, mat_dir)
                            if existing_path is not None:
                                local_specs[call_id] = str(existing_path)
                                with skipped_lock:
                                    skipped_existing_total += 1
                                continue
                        try:
                            # Context window calculation
                            begin = call['begin time (s)']
                            end = call['end time (s)']
                            padding = (ml_context - (end - begin)) / 2
                            ext_context = ml_context + (2 * edge_context)
                            desired_start = (begin - padding) - edge_context
                            desired_end = (end + padding) + edge_context
                            
                            # Retrieve stitched audio
                            audio_data = stitch_audio_files(
                                self.onc_token, clip_id, call['device_code'],
                                desired_start, desired_end, ext_context, audio_dir,
                                show_onc_warnings=self.show_onc_warnings,
                                allow_downloads=allow_audio_download,
                            )
                            
                            if audio_data is not None:
                                # Generate and save
                                res_path, res_dims = self._generate_and_save(
                                    audio_data, fs, call_id, png_dir, mat_dir,
                                    edge_context=edge_context,
                                    target_duration=ml_context,
                                    png_style=png_style,
                                    png_scale=png_scale,
                                    png_cmap=png_cmap,
                                    png_pmin=png_pmin,
                                    png_pmax=png_pmax,
                                )
                                if res_path:
                                    local_specs[call_id] = str(res_path)
                                    if local_dims is None and res_dims:
                                        local_dims = res_dims
                        except Exception as e:
                            local_failed.append({'call_id': call_id, 'clip_id': clip_id, 'reason': str(e)})

                # 3. Process Negative Samples
                if generate_negatives:
                    requested_negatives = len(calls_in_file) * negatives_per_call
                    if max_negatives_per_file is not None:
                        requested_negatives = min(int(max_negatives_per_file), int(requested_negatives))
                    neg_windows = sample_negative_windows_for_file(
                        clip_id, 300.0, neg_context, calls_by_file, 
                        requested_negatives,
                        margin=neg_margin,
                        strategy=neg_strategy,
                        step_seconds=neg_step_seconds,
                    )
                    for n_idx, (start, end) in enumerate(neg_windows):
                        neg_id = f"{clip_id}_neg_{n_idx}"
                        if existing_policy == "skip":
                            existing_path = _existing_output_path(neg_id, neg_png_dir, neg_mat_dir)
                            if existing_path is not None:
                                local_specs[neg_id] = str(existing_path)
                                with skipped_lock:
                                    skipped_existing_total += 1
                                continue
                        try:
                            ext_context = neg_context + (2 * edge_context)
                            desired_start = start - edge_context
                            desired_end = end + edge_context
                            audio_data = stitch_audio_files(
                                self.onc_token, clip_id, calls_in_file.iloc[0]['device_code'],
                                desired_start, desired_end, ext_context, audio_dir,
                                show_onc_warnings=self.show_onc_warnings,
                                allow_downloads=allow_audio_download,
                            )
                            if audio_data is not None:
                                res_path, res_dims = self._generate_and_save(
                                    audio_data, fs, neg_id, neg_png_dir, neg_mat_dir,
                                    edge_context=edge_context,
                                    target_duration=neg_context,
                                    png_style=png_style,
                                    png_scale=png_scale,
                                    png_cmap=png_cmap,
                                    png_pmin=png_pmin,
                                    png_pmax=png_pmax,
                                )
                                if res_path:
                                    local_specs[neg_id] = str(res_path)
                                    if local_dims is None and res_dims:
                                        local_dims = res_dims
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

        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_files, desc="Processing audio files", unit="file")
            except Exception:
                pbar = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_file, cid, df, i+1): cid for i, (cid, df) in enumerate(file_groups)}
            for future in concurrent.futures.as_completed(futures):
                s, f, d = future.result()
                spectrogram_files.update(s)
                failed_calls.extend(f)
                if d: actual_dimensions = d
                if pbar:
                    clip_id = futures.get(future, "")
                    if clip_id:
                        pbar.set_postfix_str(clip_id)
                    pbar.update(1)

        if pbar:
            pbar.close()
        if existing_policy == "skip" and skipped_existing_total:
            print_status(f"Reused {skipped_existing_total} existing spectrogram files (existing_policy=skip)", "INFO")
                
        return spectrogram_files, failed_calls, actual_dimensions

    def _generate_and_save(self, 
                          audio_data: np.ndarray, 
                          fs: float, 
                          call_id: str, 
                          png_dir: Path, 
                          mat_dir: Path,
                          edge_context: float = 0.0,
                          target_duration: Optional[float] = None,
                          png_style: str = "test",
                          png_scale: int = 3,
                          png_cmap: str = "inferno",
                          png_pmin: float = 2.0,
                          png_pmax: float = 98.0) -> Tuple[Optional[Path], Optional[Tuple[int, int]]]:
        """Generate and save spectrogram.
        
        The MAT file is saved with frequency-cropped data so training doesn't need to re-crop.
        Returns the saved path and (freq_bins, time_bins) dimensions when available.
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
            
            # 3. Optionally trim edge context in time domain
            if edge_context and target_duration:
                t_start = float(edge_context)
                t_end = t_start + float(target_duration)
                time_mask = (times >= t_start) & (times <= t_end)
                if not np.any(time_mask):
                    t0 = int(np.searchsorted(times, t_start, side="left"))
                    t1 = int(np.searchsorted(times, t_end, side="right"))
                    t0 = max(0, min(t0, len(times) - 1))
                    t1 = max(t0 + 1, min(t1, len(times)))
                    time_mask = np.zeros_like(times, dtype=bool)
                    time_mask[t0:t1] = True
                times = times[time_mask] - t_start
                Sxx_cropped = Sxx_cropped[:, time_mask]
                power_db_cropped = power_db_cropped[:, time_mask]

            # 4. Save PNG if enabled (use cropped data)
            if spec_cfg.get('output_formats', {}).get('plots', True):
                png_path = png_dir / f"{call_id}.png"
                style = str(png_style).lower().strip()
                if style == "test":
                    self._save_png_test_style(
                        power_db_cropped,
                        png_path,
                        scale=int(png_scale),
                        cmap=str(png_cmap),
                        pmin=float(png_pmin),
                        pmax=float(png_pmax),
                    )
                else:
                    self.spectrogram_generator.plot_spectrogram(
                        freqs_cropped, times, power_db_cropped, title=f"Whale Call: {call_id}", save_path=png_path
                    )
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                except Exception:
                    pass
            
            # 5. Save MAT with CROPPED data so training data is already frequency-limited
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
            
            dims = (len(freqs_cropped), len(times))

            # Return path to PNG if it exists, else MAT
            if (png_dir / f"{call_id}.png").exists():
                return png_dir / f"{call_id}.png", dims
            return mat_dir / f"{call_id}.mat", dims
        except Exception as e:
            print_status(f"Generation failed for {call_id}: {e}", "WARNING")
            return None, None
