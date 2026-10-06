import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.postprocess_predictions import (
    _event_time_bounds_from_parent_mat,
    _extract_event_audio_from_parent,
    _extract_event_spectrogram_from_parent,
    _spectrogram_duration_seconds,
)

POSTPROCESS = REPO_ROOT / "scripts" / "inference" / "postprocess_predictions.py"
FIN_WHALE = "Biophony > Marine mammal > Cetacean > Baleen whale > Fin whale"
DEVICE = "ICLISTENHF6016"
DAY = datetime(2025, 4, 1, tzinfo=timezone.utc)
FILE_A = "ICLISTENHF6016_20250401T000000.000Z.flac"
FILE_B = "ICLISTENHF6016_20250401T000500.000Z.flac"
FS = 1000
HOP_S = 0.1
WINDOW_S = 1.0
FIRST_CENTRE_S = -10.0
N_FRAMES = 3201
CROP_BINS = 96
MARKER_POWER = 1000.0


def _write_edge_context_mat(path: Path, *, markers=(), silent_before=None, silent_after=None) -> None:
    """Write a MAT shaped like prepare_trainstyle_windows.py --slide output.

    A 300 s file with 10.5 s of edge context: 1 s frames every 0.1 s whose
    centres run from -10.0 to 310.0 s. Each marker time gets one loud frame.
    Edge-context frames centred before ``silent_before`` or after
    ``silent_after`` are silent, as when the neighbouring file was missing
    while the MAT was prepared.
    """
    rng = np.random.default_rng(0)
    times = (FIRST_CENTRE_S + HOP_S * np.arange(N_FRAMES)).astype(np.float32)
    power = rng.uniform(0.5, 1.0, (96, N_FRAMES)).astype(np.float32)
    for marker_s in markers:
        power[:, int(round((marker_s - FIRST_CENTRE_S) / HOP_S))] = MARKER_POWER
    if silent_before is not None:
        power[:, times < silent_before] = 0.0
    if silent_after is not None:
        power[:, times > silent_after] = 0.0
    scipy.io.savemat(
        str(path),
        {
            "F": np.linspace(5.0, 100.0, 96).astype(np.float32),
            "T": times,
            "P": power,
            "PdB_norm": 10.0 * np.log10(np.maximum(power / power.max(), 1e-10)),
            "window_s": 300.0,
            "analysis_window_s": WINDOW_S,
            "edge_context_s": 10.5,
            "backend": "torch",
            "time_axis_reference": "window_center",
        },
    )


def _write_raw_audio(path: Path, marker_times_s) -> None:
    """300 s of silence with a single full-scale sample at each marker time."""
    audio = np.zeros(300 * FS, dtype=np.float32)
    for marker_s in marker_times_s:
        audio[int(round(marker_s * FS))] = 1.0
    sf.write(str(path), audio, FS)


def _window_span(start_bin: int):
    """Seconds from the file start covered by the window starting at ``start_bin``."""
    start_s = FIRST_CENTRE_S - 0.5 * WINDOW_S + HOP_S * start_bin
    return start_s, start_s + (CROP_BINS - 1) * HOP_S + WINDOW_S


def _window_item(file_name: str, mat_rel: str, start_bin: int, score: float) -> dict:
    """A strict O3 window item as run_inference.py writes it without --export-crops."""
    file_start = DAY + timedelta(minutes=5 * int(file_name == FILE_B))
    start_s, end_s = _window_span(start_bin)
    return {
        "item_id": f"{Path(file_name).stem}-w{start_bin:06d}",
        "model_outputs": [{"class_hierarchy": FIN_WHALE, "score": score}],
        "verifications": [],
        "data_source_id": DEVICE,
        "audio_start_time": (file_start + timedelta(seconds=start_s)).isoformat(),
        "audio_end_time": (file_start + timedelta(seconds=end_s)).isoformat(),
        "paths": {"spectrogram_mat_path": mat_rel},
        "source_audio": {"file_name": file_name},
    }


class TestEventMediaFromParentMats(unittest.TestCase):
    """Event media for sliding windows that reference full-clip parent MATs."""

    # Window start bins (0.1 s each) per file; the rest of each file scores low.
    EDGE_EVENT = {FILE_A: (0, 48, 96)}  # file A from -10.5 s; nothing before it
    INSIDE_EVENT = {FILE_A: (1056, 1104, 1152, 1200)}  # file A 95.1 .. 120.0 s
    BOUNDARY_EVENT = {FILE_A: (3024, 3072), FILE_B: (48, 96, 144)}  # 291.9 .. 314.4 s after A starts

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        mat_dir = self.root / "mat_files"
        mat_dir.mkdir()
        self.mat_rel = {name: f"mat_files/{name}_0.0s_300.0s_window.mat" for name in (FILE_A, FILE_B)}
        # A's edge context after 300 s and B's before 0 s are silent, so the
        # markers near the boundary only show when frames come from the file
        # that owns them.
        _write_edge_context_mat(
            self.root / self.mat_rel[FILE_A], markers=(2.0, 112.3, 296.2), silent_before=0.0, silent_after=300.0
        )
        _write_edge_context_mat(self.root / self.mat_rel[FILE_B], markers=(1.7,), silent_before=0.0)
        self.raw_dir = self.root / "raw_audio"
        self.raw_dir.mkdir()
        _write_raw_audio(self.raw_dir / FILE_A, (2.0, 112.3, 296.2))
        # B is staged as WAV although its windows name the FLAC.
        _write_raw_audio(self.raw_dir / FILE_B.replace(".flac", ".wav"), (1.7,))

        high = {}
        for event in (self.EDGE_EVENT, self.INSIDE_EVENT, self.BOUNDARY_EVENT):
            for file_name, bins in event.items():
                high.update({(file_name, b): 0.95 for b in bins})
        start_bins = list(range(0, 3073, 48)) + [3105]
        items = [
            _window_item(file_name, self.mat_rel[file_name], b, high.get((file_name, b), 0.1))
            for file_name in (FILE_A, FILE_B)
            for b in start_bins
        ]
        self.input_json = self.root / "predictions_window.json"
        self.input_json.write_text(json.dumps({"schema_version": "2.1", "items": items}))

    def tearDown(self):
        self._tmp.cleanup()

    def _postprocess(self, *extra_args):
        output_json = self.root / "out" / "predictions_postprocessed.json"
        subprocess.run(
            [
                sys.executable, str(POSTPROCESS),
                "--input-json", str(self.input_json),
                "--output-json", str(output_json),
                "--low-threshold", "0.7", "--high-threshold", "0.9",
                "--min-members", "3", "--max-gap-seconds", "15",
                "--merge-event-media", "--replace-items-with-events", "--merge-across-source-audio",
                *extra_args,
            ],
            check=True,
            capture_output=True,
        )
        items = json.loads(output_json.read_text())["items"]
        return output_json, sorted(items, key=lambda item: item["audio_start_time"])

    @staticmethod
    def _seconds_after_a(iso_value: str) -> float:
        return (datetime.fromisoformat(iso_value) - DAY).total_seconds()

    def _load_event_media(self, output_json: Path, item: dict):
        mat_path = output_json.parent / item["paths"]["spectrogram_mat_path"]
        mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
        audio = None
        if "audio_path" in item["paths"]:
            audio, fs = sf.read(str(output_json.parent / item["paths"]["audio_path"]))
            self.assertEqual(fs, FS)
        return mat_path, mat, audio

    def _assert_event(self, output_json, item, *, span, marker_times, silent_until=None):
        start_s = self._seconds_after_a(item["audio_start_time"])
        end_s = self._seconds_after_a(item["audio_end_time"])
        self.assertAlmostEqual(start_s, span[0], places=3)
        self.assertAlmostEqual(end_s, span[1], places=3)
        duration = span[1] - span[0]
        mat_path, mat, audio = self._load_event_media(output_json, item)

        # The spectrogram covers exactly the event: frame centres from half a
        # window after its start to half a window before its end.
        times = np.asarray(mat["T"], dtype=np.float64)
        power = np.asarray(mat["P"])
        self.assertEqual(power.shape, (96, int(round((duration - WINDOW_S) / HOP_S)) + 1))
        self.assertAlmostEqual(times[0], 0.5 * WINDOW_S, places=6)
        self.assertAlmostEqual(times[-1] + 0.5 * WINDOW_S, duration, places=4)
        self.assertAlmostEqual(_spectrogram_duration_seconds(mat_path), duration, places=4)
        loud = times[power[0] >= MARKER_POWER]
        np.testing.assert_allclose(loud, [t - span[0] for t in marker_times], atol=1e-4)

        # The audio covers the same span with the markers at the same times.
        self.assertIsNotNone(audio)
        self.assertEqual(len(audio), int(round(duration * FS)))
        np.testing.assert_allclose(
            np.flatnonzero(np.abs(audio) > 0.5) / FS, [t - span[0] for t in marker_times], atol=1.5 / FS
        )
        if silent_until is not None:
            self.assertFalse(np.any(power[:, times < silent_until - span[0] - 0.05]))
            self.assertFalse(np.any(audio[: int(round((silent_until - span[0]) * FS))]))

    def test_event_media_cover_event_span_inside_and_across_files(self):
        output_json, items = self._postprocess("--raw-audio-dir", str(self.raw_dir))
        self.assertEqual(len(items), 3)
        edge, inside, boundary = items
        # Reaches 10.5 s into A's edge context with no previous file: those
        # frames and samples are silent instead of the clip starting late.
        self._assert_event(output_json, edge, span=(-10.5, 9.6), marker_times=(2.0,), silent_until=0.0)
        self._assert_event(output_json, inside, span=(95.1, 120.0), marker_times=(112.3,))
        # Crosses from file A into file B: frames come from the file that owns
        # them (each MAT's edge context there is silent) and audio from both files.
        self._assert_event(output_json, boundary, span=(291.9, 314.4), marker_times=(296.2, 301.7))

    def test_event_spectrogram_without_raw_audio(self):
        output_json, items = self._postprocess()
        self.assertEqual(len(items), 3)
        inside = items[1]
        self.assertNotIn("audio_path", inside["paths"])
        mat_path, mat, _ = self._load_event_media(output_json, inside)
        # Before the fix the 24.9 s event got A's whole 320 s parent plus tails.
        self.assertEqual(np.asarray(mat["P"]).shape, (96, 240))
        self.assertAlmostEqual(_spectrogram_duration_seconds(mat_path), 24.9, places=4)

    def test_merge_min_score_moves_event_times_with_the_clip(self):
        payload = json.loads(self.input_json.read_text())
        for item in payload["items"]:
            if item["item_id"].endswith("000Z-w001056"):
                item["model_outputs"][0]["score"] = 0.92
        self.input_json.write_text(json.dumps(payload))
        output_json, items = self._postprocess("--raw-audio-dir", str(self.raw_dir), "--merge-min-score", "0.94")
        # The 95.1 s window scores below the floor, so the clip and the event
        # times both start with the next window.
        self._assert_event(output_json, items[1], span=(99.9, 120.0), marker_times=(112.3,))

    def test_exported_crops_are_placed_by_time(self):
        # With --export-crops the windows reference crops of the parent MAT
        # (T kept relative to the file start) and their own audio clips.
        parent = scipy.io.loadmat(str(self.root / self.mat_rel[FILE_A]), simplify_cells=True)
        raw_a, _ = sf.read(str(self.raw_dir / FILE_A))
        crop_dir = self.root / "exported"
        crop_dir.mkdir()
        items = json.loads(self.input_json.read_text())["items"]
        for item in items:
            if item["source_audio"]["file_name"] != FILE_A:
                continue
            start_bin = int(item["item_id"].rsplit("-w", 1)[1])
            if start_bin not in self.INSIDE_EVENT[FILE_A]:
                continue
            cols = slice(start_bin, start_bin + CROP_BINS)
            crop_rel = f"exported/{item['item_id']}.mat"
            scipy.io.savemat(
                str(self.root / crop_rel),
                {"F": parent["F"], "T": parent["T"][cols], "P": parent["P"][:, cols], "PdB_norm": parent["PdB_norm"][:, cols]},
            )
            start_s, end_s = _window_span(start_bin)
            audio_rel = f"exported/{item['item_id']}.wav"
            sf.write(str(self.root / audio_rel), raw_a[int(round(start_s * FS)) : int(round(end_s * FS))], FS)
            item["paths"] = {"spectrogram_mat_path": crop_rel, "audio_path": audio_rel}
        self.input_json.write_text(
            json.dumps({"schema_version": "2.1", "spectrogram_config": {"overlap": 0.9}, "items": items})
        )

        output_json, events = self._postprocess()
        self._assert_event(output_json, events[1], span=(95.1, 120.0), marker_times=(112.3,))


class TestParentBinEventMedia(unittest.TestCase):
    """Older window items that reference parent media by bin indices."""

    def test_edge_context_clip_keeps_its_negative_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_edge_context_mat(root / "parent.mat", markers=(2.0,))
            _write_raw_audio(root / "parent.wav", (2.0,))
            members = [
                {
                    "parent_spectrogram_mat_path": "parent.mat",
                    "parent_audio_path": "parent.wav",
                    "parent_time_bin_start": start_bin,
                    "parent_time_bin_end": start_bin + CROP_BINS,
                }
                for start_bin in (0, 48)
            ]
            input_json = root / "predictions.json"
            # Frames centred on -10.0 .. -0.5 s and -5.2 .. 4.3 s, each 1 s long.
            start, end = _event_time_bounds_from_parent_mat(members, input_json)
            self.assertAlmostEqual(start, -10.5, places=4)
            self.assertAlmostEqual(end, 4.8, places=4)

            audio_rel = _extract_event_audio_from_parent("evt", members, input_json, root / "media", input_json)
            audio, fs = sf.read(str(root / audio_rel))
            self.assertEqual(len(audio), int(round(15.3 * fs)))
            self.assertAlmostEqual(int(np.argmax(audio)) / fs, 12.5, places=2)

            mat_rel = _extract_event_spectrogram_from_parent("evt", members, input_json, root / "media", input_json)
            mat = scipy.io.loadmat(str(root / mat_rel), simplify_cells=True)
            times = np.asarray(mat["T"], dtype=np.float64)
            self.assertAlmostEqual(times[0], 0.5, places=4)
            self.assertAlmostEqual(_spectrogram_duration_seconds(root / mat_rel), 15.3, places=4)
            self.assertAlmostEqual(times[int(np.argmax(mat["P"][0]))], 12.5, places=4)


if __name__ == "__main__":
    unittest.main()
