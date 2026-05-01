import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.multilabel.build_biodcase_task2_manifest import build_biodcase_manifest
from src.dataset.multilabel import normalize_call_type


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


class BuildBiodcaseTask2ManifestTest(unittest.TestCase):
    def test_builds_blue_fin_rows_from_absolute_datetimes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            out = root / "out"
            _write_csv(
                annotations,
                [
                    {
                        "dataset": "ballenyislands2015",
                        "filename": "2015-02-04T03-00-00_000.wav",
                        "annotation": "bma",
                        "annotator": "unit",
                        "low_frequency": "21.9",
                        "high_frequency": "28.4",
                        "start_datetime": "2015-02-04T03:27:32.053000+00:00",
                        "end_datetime": "2015-02-04T03:27:43.709000+00:00",
                    },
                    {
                        "dataset": "casey2014",
                        "filename": "2014-09-01T00-00-00_000.wav",
                        "annotation": "bp20plus",
                        "annotator": "unit",
                        "low_frequency": "20.0",
                        "high_frequency": "120.0",
                        "start_datetime": "2014-09-01T00:00:10+00:00",
                        "end_datetime": "2014-09-01T00:00:13+00:00",
                    },
                ],
            )

            summary = build_biodcase_manifest(annotations_csvs=[annotations], output_dir=out)

            self.assertEqual(summary["positive_count"], 2)
            self.assertEqual(summary["background_count"], 0)
            self.assertEqual(summary["label_counts"]["species:Bm"], 1)
            self.assertEqual(summary["label_counts"]["species:Bp"], 1)
            self.assertEqual(summary["label_counts"]["call:BmA"], 1)
            self.assertEqual(summary["label_counts"]["call:Bp20plus"], 1)

            with (out / "selected_calls.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            by_call = {row["call_type"]: row for row in rows}
            self.assertEqual(by_call["BmA"]["species"], "Bm")
            self.assertEqual(by_call["BmA"]["begin_s"], "1652.053000")
            self.assertEqual(by_call["Bp20plus"]["begin_s"], "10.000000")
            self.assertTrue(by_call["Bp20plus"]["mat_path"].endswith("_10.0s_13.0s_trainstyle.mat"))

            vocab = json.loads((out / "label_vocabulary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [label["id"] for label in vocab["labels"]],
                ["species:Bm", "species:Bp", "call:BmA", "call:Bp20plus"],
            )

    def test_supports_relative_times_audio_filter_and_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            audio = root / "audio"
            audio.mkdir()
            (audio / "clip-present.wav").touch()
            (audio / "background-only.wav").touch()
            audio_list = root / "audio_list.txt"
            audio_list.write_text("clip-present.wav\nbackground-only.wav\n", encoding="utf-8")
            out = root / "out"
            _write_csv(
                annotations,
                [
                    {
                        "dataset": "unit",
                        "filename": "clip-present.wav",
                        "annotation": "bpd",
                        "start_s": "4.5",
                        "duration_s": "2.0",
                    },
                    {
                        "dataset": "unit",
                        "filename": "clip-missing.wav",
                        "annotation": "bmb",
                        "start_s": "7.0",
                        "end_s": "9.0",
                    },
                ],
            )

            summary = build_biodcase_manifest(
                annotations_csvs=[annotations],
                output_dir=out,
                audio_root=audio,
                audio_lists=[audio_list],
                require_existing_audio=True,
                max_background=1,
            )

            self.assertEqual(summary["positive_count"], 1)
            self.assertEqual(summary["background_count"], 1)
            self.assertEqual(summary["skipped_counts"]["missing_audio"], 1)
            self.assertEqual(summary["label_counts"]["call:BpD"], 1)
            self.assertEqual(summary["label_counts"]["<background>"], 1)

            with (out / "selected_calls.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            background = [row for row in rows if row["is_background"] == "1"]
            self.assertEqual(background[0]["filename"], "background-only.wav")
            self.assertEqual(background[0]["label_ids"], "")
            positive = [row for row in rows if row["is_background"] == "0"][0]
            self.assertEqual(positive["begin_s"], "4.500000")
            self.assertEqual(positive["end_s"], "6.500000")

    def test_dataset_prefixed_clip_names_for_flat_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            out = root / "out"
            _write_csv(
                annotations,
                [
                    {
                        "dataset": "casey2014",
                        "filename": "2014-09-01T00-00-00_000.wav",
                        "annotation": "bp20",
                        "start_s": "10",
                        "end_s": "12",
                    }
                ],
            )

            summary = build_biodcase_manifest(
                annotations_csvs=[annotations],
                output_dir=out,
                clip_name_mode="dataset_prefix",
            )

            self.assertEqual(summary["positive_count"], 1)
            with (out / "selected_calls.csv").open(newline="", encoding="utf-8") as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertEqual(row["clip"], "casey2014__2014-09-01T00-00-00_000.wav")
            self.assertEqual(row["source_audio"], "2014-09-01T00-00-00_000.wav")
            self.assertTrue(row["mat_path"].endswith("casey2014__2014-09-01T00-00-00_000.wav_10.0s_12.0s_trainstyle.mat"))

            with (out / "required_audio_sources.csv").open(newline="", encoding="utf-8") as handle:
                source_row = list(csv.DictReader(handle))[0]
            self.assertEqual(source_row["clip"], row["clip"])
            self.assertEqual(source_row["source_dataset"], "casey2014")
            self.assertEqual(source_row["source_audio"], row["source_audio"])

    def test_biodcase_label_aliases(self):
        self.assertEqual(normalize_call_type("bma"), "BmA")
        self.assertEqual(normalize_call_type("bp20plus"), "Bp20plus")
        self.assertEqual(normalize_call_type("bp20p"), "Bp20plus")
        self.assertEqual(normalize_call_type("bpd"), "BpD")


if __name__ == "__main__":
    unittest.main()
