import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.multilabel.build_multispecies_prep_manifest import build_prep_manifest


def _write_csv(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BuildMultispeciesPrepManifestTest(unittest.TestCase):
    def test_builds_positive_background_and_required_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            clips = root / "clip_manifest.csv"
            out = root / "out"

            _write_csv(
                annotations,
                [
                    {
                        "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                        "species": "Mn",
                        "call_type_raw": "S",
                        "call_type_bucket": "",
                        "begin_time_s": "2.0",
                        "end_time_s": "3.0",
                        "verified_flag": "1",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250105T000500.000Z.flac",
                        "species": "Bp",
                        "call_type_raw": "20 Hz",
                        "call_type_bucket": "20Hz",
                        "begin_time_s": "150.0",
                        "end_time_s": "151.0",
                        "verified_flag": "1",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250105T001000.000Z.flac",
                        "species": "EQ",
                        "call_type_raw": "earthquake",
                        "call_type_bucket": "",
                        "begin_time_s": "100.0",
                        "end_time_s": "120.0",
                        "verified_flag": "1",
                    },
                ],
            )
            _write_csv(
                clips,
                [
                    {
                        "filename": "ICLISTENHF6016_20250105T002000.000Z.flac",
                        "is_pure_negative_candidate": "1",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "0",
                        "context_tags": "pure_negative",
                    }
                ],
            )

            summary = build_prep_manifest(
                annotations_csv=annotations,
                clip_manifest_csv=clips,
                output_dir=out,
                dataset_name="unit",
                species=("Mn",),
                include_fin=True,
                include_nonbiological=False,
                max_per_species=0,
                max_fin=1,
                max_background=1,
                context_s=40.0,
                edge_context_s=10.5,
                clip_duration_s=300.0,
                background_window_s=40.0,
                background_windows_per_clip=1,
            )

            self.assertEqual(summary["positive_count"], 2)
            self.assertEqual(summary["background_count"], 1)
            self.assertEqual(summary["label_counts"]["species:Mn"], 1)
            self.assertEqual(summary["label_counts"]["species:Bp"], 1)
            self.assertNotIn("species:EQ", summary["label_counts"])

            with (out / "selected_calls.csv").open(newline="", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual(len(selected), 3)
            background = [row for row in selected if row["is_background"] == "1"]
            self.assertEqual(len(background), 1)
            self.assertEqual(background[0]["begin_s"], "130.000000")
            self.assertEqual(background[0]["label_ids"], "")
            self.assertEqual(background[0]["review_status"], "pure_negative_candidate")
            self.assertEqual(background[0]["context_tags"], "pure_negative")

            required = (out / "required_audio_filenames.txt").read_text(encoding="utf-8").splitlines()
            self.assertIn("ICLISTENHF6016_20250105T000000.000Z.flac", required)
            self.assertIn("ICLISTENHF6016_20250104T235500.000Z.flac", required)

            vocab = json.loads((out / "label_vocabulary.json").read_text(encoding="utf-8"))
            self.assertEqual([label["id"] for label in vocab["labels"]], ["species:Bp", "species:Mn", "call:20Hz", "call:song"])

    def test_filters_rows_that_require_missing_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            clips = root / "clip_manifest.csv"
            audio = root / "audio"
            out = root / "out"
            audio.mkdir()

            _write_csv(
                annotations,
                [
                    {
                        "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                        "species": "Mn",
                        "call_type_raw": "S",
                        "call_type_bucket": "",
                        "begin_time_s": "2.0",
                        "end_time_s": "3.0",
                        "verified_flag": "1",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250105T000500.000Z.flac",
                        "species": "Bp",
                        "call_type_raw": "20 Hz",
                        "call_type_bucket": "20Hz",
                        "begin_time_s": "150.0",
                        "end_time_s": "151.0",
                        "verified_flag": "1",
                    },
                ],
            )
            _write_csv(
                clips,
                [
                    {
                        "filename": "ICLISTENHF6016_20250105T001000.000Z.flac",
                        "is_pure_negative_candidate": "1",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "0",
                    },
                    {
                        "filename": "ICLISTENHF6016_20250105T001500.000Z.flac",
                        "is_pure_negative_candidate": "1",
                        "is_fin_positive": "0",
                        "is_annotated_non_fin": "0",
                    },
                ],
            )
            for name in [
                "ICLISTENHF6016_20250105T000000.000Z.flac",
                "ICLISTENHF6016_20250105T000500.000Z.flac",
                "ICLISTENHF6016_20250105T001000.000Z.flac",
            ]:
                (audio / name).touch()

            summary = build_prep_manifest(
                annotations_csv=annotations,
                clip_manifest_csv=clips,
                output_dir=out,
                dataset_name="unit",
                species=("Mn",),
                include_fin=True,
                include_nonbiological=False,
                max_per_species=0,
                max_fin=1,
                max_background=2,
                context_s=40.0,
                edge_context_s=10.5,
                clip_duration_s=300.0,
                background_window_s=40.0,
                background_windows_per_clip=1,
                available_audio_dir=audio,
            )

            self.assertEqual(summary["positive_count"], 1)
            self.assertEqual(summary["background_count"], 1)
            self.assertEqual(summary["skipped_annotation_counts"]["missing_required_audio"], 1)
            self.assertEqual(summary["skipped_annotation_counts"]["background_missing_audio"], 1)
            self.assertIn("ICLISTENHF6016_20250104T235500.000Z.flac", summary["missing_required_audio_top"])

            with (out / "selected_calls.csv").open(newline="", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual({row["clip"] for row in selected}, {
                "ICLISTENHF6016_20250105T000500.000Z.flac",
                "ICLISTENHF6016_20250105T001000.000Z.flac",
            })


if __name__ == "__main__":
    unittest.main()
