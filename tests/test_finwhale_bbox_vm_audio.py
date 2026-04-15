import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.finwhale_bbox_audio_audit import COHORT_2025, COHORT_HISTORICAL
from src.dataset.finwhale_bbox_vm_audio import (
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


if __name__ == "__main__":
    unittest.main()
