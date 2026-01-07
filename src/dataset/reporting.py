#!/usr/bin/env python3
import logging
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_status(message: str, status: str = "INFO"):
    """Print formatted status messages"""
    colors = {
        "INFO": "\033[94m",       # Blue
        "SUCCESS": "\033[92m",    # Green
        "WARNING": "\033[93m",    # Yellow
        "ERROR": "\033[91m",      # Red
        "PROGRESS": "\033[96m",   # Cyan
        "RESET": "\033[0m"
    }
    
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌ ",
        "PROGRESS": "🔄 "
    }.get(status, "")
    
    color = colors.get(status, colors["INFO"])
    print(f"{color}{prefix}{message}{colors['RESET']}")

def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def create_analysis_report(
    output_dir: Path,
    excel_files: List[str],
    whale_calls: 'pd.DataFrame',
    downloaded_files: Dict[str, str],
    custom_spectrograms: Dict[str, str],
    onc_spectrograms: Dict[str, str],
    spectrogram_generator: any,
    config: Dict,
    failed_calls: List[Dict] = None,
    actual_dimensions: Optional[Tuple[int, int]] = None,
    audio_cleaned_up: bool = False
):
    """Create a comprehensive analysis report in JSON format"""
    print_header("CREATING ANALYSIS REPORT")
    
    # Separate positive/negative counts
    total_specs = len(custom_spectrograms)
    negative_count = sum(1 for k in custom_spectrograms.keys() if "_neg_" in k)
    positive_count = total_specs - negative_count

    report = {
        "dataset_metadata": {
            "creation_date": datetime.now().isoformat(),
            "source_libraries": list(excel_files),
            "total_calls_analyzed": len(whale_calls),
            "successful_spectrograms": total_specs,
            "positive_spectrograms": positive_count,
            "negative_spectrograms": negative_count,
            "failed_spectrograms": len(failed_calls) if failed_calls else 0,
            "unique_audio_files": len(downloaded_files),
            "onc_spectrograms_downloaded": len(onc_spectrograms)
        },
        "processing_parameters": {
            "spectrogram_generation": {
                "window_duration_s": spectrogram_generator.win_dur,
                "overlap_ratio": spectrogram_generator.overlap,
                "frequency_limits_hz": {
                    "min": spectrogram_generator.freq_lims[0],
                    "max": spectrogram_generator.freq_lims[1]
                },
                "colormap": spectrogram_generator.colormap,
                "color_limits_db": {
                    "min": spectrogram_generator.clim[0],
                    "max": spectrogram_generator.clim[1]
                },
                "log_frequency_scale": spectrogram_generator.log_freq,
                "fft_method": "scipy.signal.spectrogram with Hann window",
                "scaling": "power spectral density (PSD)",
                "normalization": "10*log10(abs(P/max(P)))"
            },
            "temporal_context": {
                "context_duration_s": config.get('temporal_context', {}).get('context_duration', 40.0),
                "padding_method": config.get('temporal_context', {}).get('padding_method', 'centered'),
                "multi_file_stitching": config.get('temporal_context', {}).get('multi_file_stitching', True),
                "exact_duration_enforcement": config.get('temporal_context', {}).get('exact_duration_enforcement', True)
            },
            "frequency_filtering": {
                "whale_call_range_hz": [5, 100],
                "post_processing_crop": "applied after spectrogram generation",
                "actual_freq_bins": f"{actual_dimensions[0]} bins" if actual_dimensions else "varies per spectrogram",
                "actual_freq_resolution_hz": f"~{95/actual_dimensions[0]:.2f} Hz per bin" if actual_dimensions else "varies per spectrogram"
            },
            "spectrogram_dimensions": {
                "actual_dimensions": f"{actual_dimensions[0]} x {actual_dimensions[1]} (freq x time)" if actual_dimensions else "varies per spectrogram",
                "actual_time_resolution_ms": f"~{40000/actual_dimensions[1]:.1f} ms per bin" if actual_dimensions else "varies per spectrogram",
                "frequency_range_hz": [5, 100],
                "temporal_context_s": 40.0,
                "augmentation_ready": "centered context allows sliding window cropping"
            }
        },
        "technical_specifications": {
            "audio_format": "WAV files from Ocean Networks Canada",
            "sample_rate_hz": "varies by file (typically 64kHz)",
            "bit_depth": "varies by file",
            "file_duration_s": 300,
            "device_codes": list(set(whale_calls['device_code'].tolist())),
            "date_range": {
                "start": whale_calls['Date (UTC)'].min().isoformat(),
                "end": whale_calls['Date (UTC)'].max().isoformat()
            }
        },
        "output_locations": {
            "audio_directory": str(output_dir / "audio") if not audio_cleaned_up else f"{output_dir / 'audio'} (cleaned up)",
            "mat_files_directory": str(output_dir / "mat_files") if config.get('custom_spectrograms', {}).get('output_formats', {}).get('matlab', False) else None,
            "png_files_directory": str(output_dir / "png_files") if config.get('custom_spectrograms', {}).get('output_formats', {}).get('plots', True) else None,
            "neg_mat_files_directory": str(output_dir / "neg_mat_files") if config.get('custom_spectrograms', {}).get('output_formats', {}).get('matlab', False) else None,
            "neg_png_files_directory": str(output_dir / "neg_png_files") if config.get('custom_spectrograms', {}).get('output_formats', {}).get('plots', True) else None,
            "onc_spectrograms_directory": str(output_dir / "onc_spectrograms"),
            "audio_files_cleaned_up": audio_cleaned_up
        },
        "reproduction_instructions": {
            "required_libraries": [
                "pandas", "numpy", "matplotlib", "scipy", "soundfile", "onc"
            ],
            "key_parameters": {
                "win_dur": spectrogram_generator.win_dur,
                "overlap": spectrogram_generator.overlap,
                "freq_crop": [5, 100],
                "context_duration": config.get('temporal_context', {}).get('context_duration', 40.0),
                "normalization": "10*log10(abs(spectrogram/max(spectrogram)))"
            }
        }
    }
    
    # Save main dataset report
    dataset_report_file = output_dir / "dataset_documentation.json"
    with open(dataset_report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    print_status(f"Dataset documentation saved: {dataset_report_file}", "SUCCESS")
    
    # Save failed calls report
    if failed_calls:
        failed_report = {
            "failed_spectrograms": {
                "total_failed": len(failed_calls),
                "analysis_date": datetime.now().isoformat(),
                "failures": failed_calls
            }
        }
        
        failed_file = output_dir / "failed_spectrograms.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_report, f, indent=2, default=str)
            
        print_status(f"Failed spectrograms report saved: {failed_file}", "SUCCESS")
    
    # Print summary
    print_status(f"📊 Analyzed {len(whale_calls)} fin whale calls")
    print_status(f"🎵 Downloaded {len(downloaded_files)} audio files")
    print_status(f"📈 Created {len(custom_spectrograms)} custom spectrograms")
    print_status(f"🌊 Downloaded {len(onc_spectrograms)} ONC spectrograms")
    print_status(f"📁 Results saved to: {output_dir}")
