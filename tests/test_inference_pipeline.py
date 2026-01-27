import unittest
import numpy as np
import math
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Reference logic from src/data/sequential_prep.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sequential_prep import compute_window_positions, crop_to_freq_lims


class TestWindowTiling(unittest.TestCase):
    """Tests for the compute_window_positions function."""
    
    def test_perfect_fit(self):
        """Windows divide evenly into total bins."""
        self.assertEqual(compute_window_positions(300, 100), [0, 100, 200])
        
    def test_uneven_fit_with_overlap(self):
        """Uneven fit should produce minimal, evenly distributed overlap."""
        # 300 bins, 96 window size
        # n_windows = ceil(300/96) = 4
        # step = (300-96)/(4-1) = 204/3 = 68
        positions = compute_window_positions(300, 96)
        self.assertEqual(positions, [0, 68, 136, 204])
        # Verify last window covers the end: 204 + 96 = 300
        self.assertEqual(positions[-1] + 96, 300)
        
    def test_smaller_than_window(self):
        """If total bins smaller than window, return single position at 0."""
        self.assertEqual(compute_window_positions(50, 100), [0])
        self.assertEqual(compute_window_positions(100, 100), [0])
        
    def test_realistic_spectrogram_size(self):
        """Test with realistic 5-minute spectrogram dimensions."""
        # 5-min @ win_dur=1.0, overlap=0.9 → ~2996 time bins
        # Crop size of 96 should yield ~32 windows
        positions = compute_window_positions(2996, 96)
        n_expected = math.ceil(2996 / 96)  # 32
        self.assertEqual(len(positions), n_expected)
        # Verify coverage: last window should reach the end
        self.assertGreaterEqual(positions[-1] + 96, 2996)
        
    def test_single_window_exact(self):
        """Exactly one window that fits perfectly."""
        self.assertEqual(compute_window_positions(96, 96), [0])


class TestAudioStitching(unittest.TestCase):
    """Tests for audio stitching logic used for edge artifact handling."""
    
    def test_stitching_middle_clip(self):
        """Stitching a middle clip with context from neighbors."""
        fs = 1000
        clip_len = 300 * fs  # 300 seconds
        c0 = np.zeros(clip_len) + 0
        c1 = np.zeros(clip_len) + 1
        c2 = np.zeros(clip_len) + 2
        
        pad = 2 * fs  # 2 seconds
        
        # Stitching c1 with context from c0 and c2
        stitched = np.concatenate([c0[-pad:], c1, c2[:pad]])
        
        self.assertEqual(len(stitched), (300 + 4) * fs)
        self.assertTrue(np.all(stitched[:pad] == 0))
        self.assertTrue(np.all(stitched[pad:-pad] == 1))
        self.assertTrue(np.all(stitched[-pad:] == 2))
        
    def test_stitching_first_clip(self):
        """First clip has no previous context."""
        fs = 1000
        clip_len = 300 * fs
        c0 = np.ones(clip_len) * 0
        c1 = np.ones(clip_len) * 1
        pad = 2 * fs
        
        # First clip: no prev, only next
        stitched = np.concatenate([c0, c1[:pad]])
        self.assertEqual(len(stitched), (300 + 2) * fs)
        
    def test_stitching_last_clip(self):
        """Last clip has no next context."""
        fs = 1000
        clip_len = 300 * fs
        c0 = np.ones(clip_len) * 0
        c1 = np.ones(clip_len) * 1
        pad = 2 * fs
        
        # Last clip: prev but no next
        stitched = np.concatenate([c0[-pad:], c1])
        self.assertEqual(len(stitched), (300 + 2) * fs)


class TestFrequencyCropping(unittest.TestCase):
    """Tests for frequency dimension cropping."""
    
    def test_crop_to_freq_range(self):
        """Crop spectrogram to specific frequency range."""
        # Simulate a spectrogram with 0-100 Hz in 101 bins
        freqs = np.linspace(0, 100, 101)
        data = np.random.rand(101, 50)
        
        cropped_freqs, cropped_data = crop_to_freq_lims(freqs, data, 10, 50)
        
        self.assertEqual(cropped_freqs[0], 10)
        self.assertEqual(cropped_freqs[-1], 50)
        self.assertEqual(len(cropped_freqs), 41)  # 10 to 50 inclusive
        self.assertEqual(cropped_data.shape[0], 41)
        self.assertEqual(cropped_data.shape[1], 50)
        
    def test_crop_whale_call_range(self):
        """Test cropping to typical whale call frequency range (5-100 Hz)."""
        # Simulate full frequency range up to 32 kHz
        freqs = np.linspace(0, 32000, 32001)
        data = np.random.rand(32001, 100)
        
        cropped_freqs, cropped_data = crop_to_freq_lims(freqs, data, 5, 100)
        
        # Should get approximately 96 bins (5 Hz to 100 Hz at 1 Hz resolution)
        self.assertGreaterEqual(len(cropped_freqs), 95)
        self.assertLessEqual(len(cropped_freqs), 97)


class TestSquareChunkGeneration(unittest.TestCase):
    """Tests for generating square CNN-compatible chunks."""
    
    def test_chunk_dimensions(self):
        """Verify chunks are square with correct dimensions."""
        crop_size = 96
        n_freq_bins = 96
        n_time_bins = 2996
        
        # Simulate cropped spectrogram
        PdB = np.random.rand(n_freq_bins, n_time_bins)
        
        # Get window positions
        positions = compute_window_positions(n_time_bins, crop_size)
        
        # Extract chunks and verify dimensions
        for start_idx in positions:
            end_idx = min(start_idx + crop_size, n_time_bins)
            chunk = PdB[:crop_size, start_idx:end_idx]
            
            # Pad if at edge
            if chunk.shape[1] < crop_size:
                pad_width = crop_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='edge')
            
            self.assertEqual(chunk.shape, (crop_size, crop_size))
            
    def test_frequency_mismatch_handling(self):
        """Handle case where freq bins don't match crop size."""
        crop_size = 96
        n_freq_bins = 100  # More than crop_size
        n_time_bins = 200
        
        PdB = np.random.rand(n_freq_bins, n_time_bins)
        
        # Center crop in frequency if larger
        if n_freq_bins > crop_size:
            start_f = (n_freq_bins - crop_size) // 2
            PdB = PdB[start_f:start_f + crop_size, :]
        
        self.assertEqual(PdB.shape[0], crop_size)


class TestDatasetDocumentationLoading(unittest.TestCase):
    """Tests for loading parameters from dataset_documentation.json."""
    
    def test_load_from_json(self):
        """Test loading parameters from dataset documentation."""
        doc = {
            "processing_parameters": {
                "spectrogram_generation": {
                    "window_duration_s": 1.0,
                    "overlap_ratio": 0.9,
                    "frequency_limits_hz": {"min": 5, "max": 100},
                    "color_limits_db": {"min": -40, "max": 0}
                },
                "frequency_filtering": {
                    "actual_freq_bins": "96 bins"
                }
            }
        }
        
        # Simulate parameter extraction
        proc_params = doc.get('processing_parameters', {})
        spec_gen = proc_params.get('spectrogram_generation', {})
        freq_filt = proc_params.get('frequency_filtering', {})
        
        freq_limits = spec_gen.get('frequency_limits_hz', {})
        freq_lims = (freq_limits.get('min', 5), freq_limits.get('max', 100))
        
        win_dur = spec_gen.get('window_duration_s', 1.0)
        overlap = spec_gen.get('overlap_ratio', 0.9)
        
        actual_bins = freq_filt.get('actual_freq_bins', '')
        crop_size = int(actual_bins.split()[0]) if actual_bins else 96
        
        self.assertEqual(freq_lims, (5, 100))
        self.assertEqual(win_dur, 1.0)
        self.assertEqual(overlap, 0.9)
        self.assertEqual(crop_size, 96)
        
    def test_parameter_precedence(self):
        """CLI overrides should take precedence over documentation."""
        # Simulate precedence logic
        doc_crop_size = 96
        cli_crop_size = 128
        
        # CLI takes precedence
        final_crop_size = cli_crop_size if cli_crop_size is not None else doc_crop_size
        self.assertEqual(final_crop_size, 128)
        
        # When CLI is None, doc takes precedence
        cli_crop_size = None
        final_crop_size = cli_crop_size if cli_crop_size is not None else doc_crop_size
        self.assertEqual(final_crop_size, 96)


class TestChunkCoverage(unittest.TestCase):
    """Tests to verify complete coverage of spectrograms."""
    
    def test_full_coverage(self):
        """Ensure all time bins are covered by at least one window."""
        total_bins = 2996
        window_size = 96
        positions = compute_window_positions(total_bins, window_size)
        
        covered = set()
        for start in positions:
            for i in range(start, min(start + window_size, total_bins)):
                covered.add(i)
        
        self.assertEqual(len(covered), total_bins)
        
    def test_overlap_distribution(self):
        """Verify overlap is evenly distributed."""
        total_bins = 300
        window_size = 96
        positions = compute_window_positions(total_bins, window_size)
        
        # Calculate gaps between consecutive windows
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        
        # Gaps should be approximately equal
        if len(gaps) > 1:
            gap_std = np.std(gaps)
            self.assertLess(gap_std, 2)  # Very small variance


if __name__ == '__main__':
    unittest.main()
