import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.finwhale_bbox import FIN_SPECIES_CODE, HISTORICAL_DATASET, PURE_NEGATIVE_DATASET
from src.dataset.finwhale_bbox_audio_audit import COHORT_2025, COHORT_HISTORICAL
from src.dataset.finwhale_bbox_vm_audio import (
    build_export_required_audio_filenames,
    materialize_audio_subset,
    select_missing_required_audio_filenames,
    select_required_audio_filenames,
    summarize_stage_availability,
)


class TestFinwhaleBboxVmAudio(unittest.TestCase):
    def test_select_required_audio_filenames_filters_by_cohort_and_policy(self) -> None:
        requirement_df = pd.DataFrame(
            [
                {
                    "cohort": COHORT_HISTORICAL,
                    "policy": "current_export_render",
                    "required_filename": "hist_a.wav",
                },
                {
                    "cohort": COHORT_HISTORICAL,
                    "policy": "centered_40s_event_context",
                    "required_filename": "hist_b.wav",
                },
                {
                    "cohort": COHORT_2025,
                    "policy": "current_export_render",
                    "required_filename": "clip_2025.flac",
                },
                {
                    "cohort": COHORT_2025,
                    "policy": "current_export_render",
                    "required_filename": "clip_2025.flac",
                },
            ]
        )

        selected = select_required_audio_filenames(
            requirement_df,
            cohort=COHORT_2025,
            policies=["current_export_render"],
        )

        self.assertEqual(selected, ["clip_2025.flac"])

    def test_select_missing_required_audio_filenames_can_filter_roles(self) -> None:
        requirement_df = pd.DataFrame(
            [
                {
                    "cohort": COHORT_HISTORICAL,
                    "policy": "current_export_render",
                    "required_filename": "hist_main.wav",
                    "role": "main",
                    "exists": 0,
                },
                {
                    "cohort": COHORT_HISTORICAL,
                    "policy": "current_export_render",
                    "required_filename": "hist_prev.wav",
                    "role": "prev",
                    "exists": 0,
                },
                {
                    "cohort": COHORT_HISTORICAL,
                    "policy": "current_export_render",
                    "required_filename": "hist_present.wav",
                    "role": "main",
                    "exists": 1,
                },
            ]
        )

        selected = select_missing_required_audio_filenames(
            requirement_df,
            cohort=COHORT_HISTORICAL,
            policies=["current_export_render"],
            roles=["main"],
        )

        self.assertEqual(selected, ["hist_main.wav"])

    def test_materialize_audio_subset_hardlinks_existing_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            target_dir = tmp / "target"
            source_dir.mkdir()
            target_dir.mkdir()

            source_file = source_dir / "ICLISTENHF1353_20180701T000000.000Z.wav"
            source_file.write_bytes(b"abc123")

            result = materialize_audio_subset(
                [source_file.name, "missing.wav"],
                source_root=source_dir,
                target_dir=target_dir,
                mode="hardlink",
            )

            target_file = target_dir / source_file.name
            self.assertTrue(target_file.exists())
            self.assertEqual(result["requested_count"], 2)
            self.assertEqual(result["available_count"], 1)
            self.assertEqual(result["missing_source_count"], 1)
            self.assertIn(source_file.name, result["materialized_from_source"])
            self.assertIn("missing.wav", result["missing_source_names"])
            self.assertEqual(os.stat(source_file).st_ino, os.stat(target_file).st_ino)

    def test_summarize_stage_availability_counts_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir)
            (stage_dir / "present.wav").write_bytes(b"data")
            requirement_df = pd.DataFrame(
                [
                    {
                        "cohort": COHORT_HISTORICAL,
                        "policy": "current_export_render",
                        "required_filename": "present.wav",
                        "role": "main",
                    },
                    {
                        "cohort": COHORT_HISTORICAL,
                        "policy": "current_export_render",
                        "required_filename": "missing.wav",
                        "role": "next",
                    },
                ]
            )

            summary = summarize_stage_availability(
                requirement_df,
                cohort=COHORT_HISTORICAL,
                policies=["current_export_render"],
                target_dir=stage_dir,
            )

            self.assertEqual(summary["requirement_row_count"], 2)
            self.assertEqual(summary["missing_requirement_count"], 1)
            self.assertEqual(summary["unique_required_file_count"], 2)
            self.assertEqual(summary["unique_available_file_count"], 1)
            self.assertEqual(summary["missing_by_role"], {"next": 1})

    def test_build_export_required_audio_filenames_matches_context_policy(self) -> None:
        annotation_df = pd.DataFrame(
            [
                {
                    "annotation_id": "ann_hist_edge",
                    "source_dataset": HISTORICAL_DATASET,
                    "filename": "ICLISTENHF1353_20180701T000000.000Z.wav",
                    "recording_day_utc": "2018-07-01",
                    "species_code": FIN_SPECIES_CODE,
                    "call_type_std": "20Hz",
                    "begin_time_s": 5.0,
                    "end_time_s": 6.0,
                    "low_freq_hz": 18.0,
                    "high_freq_hz": 24.0,
                }
            ]
        )
        clip_df = pd.DataFrame(
            [
                {
                    "source_dataset": HISTORICAL_DATASET,
                    "filename": "ICLISTENHF1353_20180701T000000.000Z.wav",
                    "recording_day_utc": "2018-07-01",
                    "is_pure_negative_candidate": 0,
                },
                {
                    "source_dataset": PURE_NEGATIVE_DATASET,
                    "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                    "recording_day_utc": "2025-01-05",
                    "is_pure_negative_candidate": 1,
                },
            ]
        )
        assignments_df = pd.DataFrame(
            [
                {
                    "source_dataset": HISTORICAL_DATASET,
                    "filename": "ICLISTENHF1353_20180701T000000.000Z.wav",
                    "split_name": "train",
                },
                {
                    "source_dataset": PURE_NEGATIVE_DATASET,
                    "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                    "split_name": "train",
                },
            ]
        )

        result = build_export_required_audio_filenames(
            annotation_df,
            clip_df,
            assignments_df,
            context_duration_s=40.0,
            clip_duration_s=300.0,
            edge_buffer_s=2.0,
            pure_zero_ratio=2.0,
            negative_margin_s=2.0,
        )

        required = set(result["required_filenames"])
        self.assertIn("ICLISTENHF1353_20180701T000000.000Z.wav", required)
        self.assertIn("ICLISTENHF1353_20180630T235500.000Z.wav", required)
        self.assertIn("ICLISTENHF6016_20250105T000000.000Z.flac", required)
        self.assertGreaterEqual(result["summary"]["context_summary"]["context_count"], 2)
        self.assertGreaterEqual(result["summary"]["requirement_role_counts"]["main"], 2)
        self.assertGreaterEqual(result["summary"]["requirement_role_counts"]["prev"], 1)


if __name__ == "__main__":
    unittest.main()
