import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.part2.prepare_trainstyle_windows import frame_centre_times

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestFrameCentreTimes(unittest.TestCase):
    def test_frame_starts_move_to_centres(self):
        times = frame_centre_times(np.arange(5) * 0.1, 1.0)
        np.testing.assert_allclose(times, [0.5, 0.6, 0.7, 0.8, 0.9], atol=1e-6)
        self.assertEqual(times.dtype, np.float32)

    def test_centred_axis_is_not_shifted_again(self):
        times = frame_centre_times(0.5 + np.arange(5) * 0.1, 1.0)
        np.testing.assert_allclose(times, [0.5, 0.6, 0.7, 0.8, 0.9], atol=1e-6)

    def test_empty_axis(self):
        self.assertEqual(frame_centre_times(np.array([]), 1.0).size, 0)


class TestSlidingWindowMatTimeAxis(unittest.TestCase):
    def test_time_axis_matches_audio_with_edge_context(self):
        fs = 1000
        clip = "ICLISTENHF6016_20250401T000000.000Z"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            rng = np.random.default_rng(0)
            for name in ("ICLISTENHF6016_20250331T235500.000Z", clip, "ICLISTENHF6016_20250401T000500.000Z"):
                audio = (1e-3 * rng.standard_normal(300 * fs)).astype(np.float32)
                if name == clip:
                    audio[1 * fs] = 1.0  # impulse 1 s into the clip
                sf.write(str(audio_dir / f"{name}.wav"), audio, fs)
            (root / "clips.txt").write_text(f"{clip}\n")

            subprocess.run(
                [
                    sys.executable, str(REPO_ROOT / "scripts" / "data" / "part2" / "prepare_trainstyle_windows.py"),
                    "--slide", "--clip-list", str(root / "clips.txt"),
                    "--audio-dir", str(audio_dir),
                    "--dataset-doc", str(root / "missing_dataset_documentation.json"),
                    "--out-dir", str(root / "mats"),
                    "--window-s", "300", "--step-s", "300",
                ],
                check=True,
                capture_output=True,
            )
            mat = scipy.io.loadmat(str(root / "mats" / f"{clip}_0.0s_300.0s_window.mat"), simplify_cells=True)

        times = np.asarray(mat["T"], dtype=np.float64).ravel()
        # 10.5 s of edge context on each side, 1 s frames every 0.1 s, centre times.
        self.assertEqual(times.size, 3201)
        self.assertAlmostEqual(times[0], -10.0, places=3)
        self.assertAlmostEqual(times[-1], 310.0, places=3)
        # The frame centred on the impulse has the most power.
        power = np.asarray(mat["P"], dtype=np.float64)
        self.assertAlmostEqual(times[int(np.argmax(power.sum(axis=0)))], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
