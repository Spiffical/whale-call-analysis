import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.multilabel.build_dclde_killer_whale_manifest import build_dclde_manifest


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


class BuildDcldeKillerWhaleManifestTest(unittest.TestCase):
    def test_maps_kw_and_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "Annotations.csv"
            objects = root / "gcs_objects.txt"
            out = root / "out"
            _write_csv(
                annotations,
                [
                    {
                        "Soundfile": "kw.wav",
                        "Dataset": "BarkleyCanyon",
                        "LowFreqHz": "1500",
                        "HighFreqHz": "5000",
                        "FileBeginSec": "10.0",
                        "FileEndSec": "12.5",
                        "ClassSpecies": "KW",
                        "KW": "1",
                        "KW_certain": "1",
                        "Ecotype": "SRKW",
                        "Provider": "ONC",
                        "AnnotationLevel": "Call",
                        "FileOk": "TRUE",
                    },
                    {
                        "Soundfile": "hw.wav",
                        "Dataset": "BarkleyCanyon",
                        "LowFreqHz": "500",
                        "HighFreqHz": "1500",
                        "FileBeginSec": "30.0",
                        "FileEndSec": "35.0",
                        "ClassSpecies": "HW",
                        "KW": "0",
                        "KW_certain": "NA",
                        "Ecotype": "NA",
                        "Provider": "ONC",
                        "AnnotationLevel": "Detection",
                        "FileOk": "TRUE",
                    },
                    {
                        "Soundfile": "ab.wav",
                        "Dataset": "BarkleyCanyon",
                        "LowFreqHz": "10",
                        "HighFreqHz": "100",
                        "FileBeginSec": "40.0",
                        "FileEndSec": "45.0",
                        "ClassSpecies": "AB",
                        "KW": "0",
                        "KW_certain": "NA",
                        "Ecotype": "NA",
                        "Provider": "ONC",
                        "AnnotationLevel": "Detection",
                        "FileOk": "TRUE",
                    },
                ],
            )
            objects.write_text(
                "\n".join(
                    [
                        "dclde/2027/dclde_2027_killer_whales/onc/audio/barkleycanyon/kw.wav",
                        "dclde/2027/dclde_2027_killer_whales/onc/audio/barkleycanyon/hw.wav",
                        "dclde/2027/dclde_2027_killer_whales/onc/audio/barkleycanyon/ab.wav",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_dclde_manifest(
                annotations_csv=annotations,
                output_dir=out,
                gcs_object_lists=[objects],
                require_gcs_audio=True,
                max_positive=10,
                max_hard_negative=10,
            )

            self.assertEqual(summary["positive_count"], 2)
            self.assertEqual(summary["hard_negative_count"], 1)
            self.assertEqual(summary["label_counts"]["species:Oo"], 1)
            self.assertEqual(summary["label_counts"]["species:Mn"], 1)
            self.assertEqual(summary["label_counts"]["call:orca_call"], 1)
            self.assertEqual(summary["label_counts"]["<background>"], 1)

            with (out / "selected_calls.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            kw = [row for row in rows if row["source_class_species"] == "KW"][0]
            self.assertEqual(kw["source_provider"], "ONC")
            self.assertEqual(kw["source_kind"], "DCLDE")
            self.assertEqual(kw["source_dataset_raw"], "BarkleyCanyon")
            self.assertEqual(kw["dclde_ecotype"], "SRKW")
            self.assertEqual(kw["label_ids"], "call:orca_call|species:Oo")
            self.assertEqual(kw["canonical_label_ids"], "call:orca_call|species:Oo")
            self.assertEqual(kw["canonical_species"], "Oo")
            self.assertEqual(kw["canonical_call_type"], "orca_call")
            self.assertEqual(kw["source_row_id"], "1")
            self.assertEqual(kw["event_group"], "ONC:BarkleyCanyon:kw.wav")
            self.assertIn("/onc/audio/barkleycanyon/kw.wav", kw["https_url"])

            hw = [row for row in rows if row["source_class_species"] == "HW"][0]
            self.assertEqual(hw["label_ids"], "species:Mn")
            self.assertEqual(hw["canonical_label_ids"], "species:Mn")
            self.assertEqual(hw["analysis_label_ids"], "")
            self.assertEqual(hw["canonical_species"], "Mn")
            self.assertEqual(hw["is_background"], "0")

            ab = [row for row in rows if row["source_class_species"] == "AB"][0]
            self.assertEqual(ab["label_ids"], "")
            self.assertEqual(ab["analysis_label_ids"], "confounder:abiotic")
            self.assertEqual(ab["is_background"], "1")
            self.assertEqual(ab["negative_bucket"], "nonbiological_signal")

            with (out / "required_audio_sources.csv").open(newline="", encoding="utf-8") as handle:
                required = list(csv.DictReader(handle))
            self.assertEqual(len(required), 3)
            self.assertTrue(required[0]["clip"].startswith("dclde_ONC_BarkleyCanyon__"))

    def test_source_balanced_caps_and_missing_audio_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "Annotations.csv"
            objects = root / "gcs_objects.txt"
            out = root / "out"
            rows = []
            object_lines = []
            for idx, provider in enumerate(["ONC", "OrcaSound", "DFO_CRP", "ONC"]):
                soundfile = f"kw-{idx}.wav"
                rows.append(
                    {
                        "Soundfile": soundfile,
                        "Dataset": "BarkleyCanyon" if provider == "ONC" else "NorthBc",
                        "FileBeginSec": str(10 + idx),
                        "FileEndSec": str(11 + idx),
                        "ClassSpecies": "KW",
                        "KW": "1",
                        "KW_certain": "1",
                        "Ecotype": "SRKW",
                        "Provider": provider,
                        "AnnotationLevel": "Call",
                        "FileOk": "TRUE",
                    }
                )
                slug = {"ONC": "onc", "OrcaSound": "orcasound", "DFO_CRP": "dfo_crp"}[provider]
                dataset = "barkleycanyon" if provider == "ONC" else "northbc"
                if idx != 3:
                    object_lines.append(f"dclde/2027/dclde_2027_killer_whales/{slug}/audio/{dataset}/{soundfile}")
            _write_csv(annotations, rows)
            objects.write_text("\n".join(object_lines) + "\n", encoding="utf-8")

            summary = build_dclde_manifest(
                annotations_csv=annotations,
                output_dir=out,
                gcs_object_lists=[objects],
                require_gcs_audio=True,
                max_positive=2,
                max_hard_negative=0,
            )

            self.assertEqual(summary["positive_count"], 2)
            self.assertEqual(summary["skipped_counts"]["missing_gcs_audio"], 1)
            with (out / "selected_calls.csv").open(newline="", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle))
            groups = {row["event_group"] for row in selected}
            self.assertEqual(len(groups), len(selected))
            self.assertLessEqual(len(selected), 2)

    def test_positive_cap_balances_kw_and_hw_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "Annotations.csv"
            out = root / "out"
            rows = []
            for idx, class_species in enumerate(["KW", "KW", "KW", "HW", "HW", "HW"]):
                rows.append(
                    {
                        "Soundfile": f"{class_species.lower()}-{idx}.wav",
                        "Dataset": "BarkleyCanyon",
                        "FileBeginSec": str(10 + idx),
                        "FileEndSec": str(11 + idx),
                        "ClassSpecies": class_species,
                        "KW": "1" if class_species == "KW" else "0",
                        "KW_certain": "1" if class_species == "KW" else "NA",
                        "Ecotype": "SRKW" if class_species == "KW" else "NA",
                        "Provider": "ONC",
                        "AnnotationLevel": "Call",
                        "FileOk": "TRUE",
                    }
                )
            _write_csv(annotations, rows)

            summary = build_dclde_manifest(
                annotations_csv=annotations,
                output_dir=out,
                max_positive=2,
                max_hard_negative=0,
            )

            self.assertEqual(summary["positive_count"], 2)
            self.assertEqual(summary["source_class_counts"]["KW"], 1)
            self.assertEqual(summary["source_class_counts"]["HW"], 1)
            self.assertEqual(summary["label_counts"]["species:Oo"], 1)
            self.assertEqual(summary["label_counts"]["species:Mn"], 1)

    def test_gcs_inventory_overrides_misleading_dataset_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "Annotations.csv"
            objects = root / "gcs_objects.txt"
            out = root / "out"
            _write_csv(
                annotations,
                [
                    {
                        "Soundfile": "AMAR779.20210904T191552Z.wav",
                        "Dataset": "StrGeoS2",
                        "FileBeginSec": "75.0",
                        "FileEndSec": "76.0",
                        "ClassSpecies": "AB",
                        "KW": "0",
                        "KW_certain": "NA",
                        "Ecotype": "NA",
                        "Provider": "DFO_WDLP",
                        "AnnotationLevel": "Detection",
                        "FileOk": "TRUE",
                    }
                ],
            )
            objects.write_text(
                "dclde/2027/dclde_2027_killer_whales/dfo_wdlp/audio/strgeos1/AMAR779.20210904T191552Z.wav\n",
                encoding="utf-8",
            )

            build_dclde_manifest(
                annotations_csv=annotations,
                output_dir=out,
                gcs_object_lists=[objects],
                require_gcs_audio=True,
                max_positive=0,
                max_hard_negative=10,
            )

            with (out / "hard_negative_windows.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertIn("/dfo_wdlp/audio/strgeos1/", rows[0]["https_url"])
            self.assertNotIn("/strgeos2/", rows[0]["https_url"])


if __name__ == "__main__":
    unittest.main()
