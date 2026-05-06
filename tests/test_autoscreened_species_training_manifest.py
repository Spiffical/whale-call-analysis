import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.build_autoscreened_species_training_manifest import build_manifests


def _write_csv(path: Path, rows):
    if not rows:
        raise ValueError("rows required")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class AutoscreenedSpeciesTrainingManifestTest(unittest.TestCase):
    def test_maps_dclde_hw_to_mn_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onc = root / "onc.csv"
            dclde = root / "dclde.csv"
            neg = root / "neg.csv"
            labels = root / "labels.csv"
            report = root / "report.csv"
            _write_csv(
                onc,
                [
                    {
                        "item_id": "onc-pos",
                        "mat_path": "/tmp/onc.mat",
                        "source_audio": "ICLISTENHF6016_20250101T000000.000Z.flac",
                        "label_ids": "species:Bp",
                        "split": "train",
                    }
                ],
            )
            _write_csv(
                dclde,
                [
                    {
                        "item_id": "hw-pos",
                        "mat_path": "mat_files/hw.mat",
                        "source_audio": "hw.wav",
                        "source_class_species": "HW",
                        "source_label_ids": "confounder:humpback",
                        "label_ids": "",
                        "split": "train",
                    },
                    {
                        "item_id": "ab-neg",
                        "mat_path": "mat_files/ab.mat",
                        "source_audio": "ab.wav",
                        "source_class_species": "AB",
                        "source_label_ids": "confounder:abiotic",
                        "label_ids": "",
                        "split": "train",
                    },
                ],
            )
            _write_csv(
                neg,
                [
                    {
                        "item_id": "gap-keep",
                        "source_audio": "/audio/clip.flac",
                        "filename": "clip.flac",
                        "begin_s": "0",
                        "end_s": "10",
                        "negative_bucket": "primary_adjacent_gap",
                        "label_ids": "",
                        "split": "train",
                    }
                ],
            )
            _write_csv(
                labels,
                [
                    {
                        "item_id": "gap-keep",
                        "model_assisted_review_label": "candidate_clean_background",
                        "visual_notes": "diffuse",
                    }
                ],
            )
            _write_csv(report, [{"clip": "clip.flac", "begin_s": "0", "end_s": "10", "out_mat": "/tmp/gap.mat"}])

            build_manifests(
                output_dir=root / "out",
                onc_csv=onc,
                biodcase_csv=None,
                dclde_csv=dclde,
                negative_csv=neg,
                model_labels_csv=labels,
                gap_report_csv=report,
            )
            with (root / "out/tables/dclde_kw_hw_primary_positive_manifest.csv").open(newline="") as handle:
                dclde_rows = list(csv.DictReader(handle))
            self.assertEqual(len(dclde_rows), 1)
            self.assertEqual(dclde_rows[0]["label_ids"], "species:Mn")
            self.assertEqual(dclde_rows[0]["species_code"], "Mn")

    def test_excludes_obvious_signal_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onc = root / "onc.csv"
            dclde = root / "dclde.csv"
            neg = root / "neg.csv"
            labels = root / "labels.csv"
            _write_csv(
                onc,
                [
                    {
                        "item_id": "onc-pos",
                        "mat_path": "/tmp/onc.mat",
                        "source_audio": "ICLISTENHF6016_20250101T000000.000Z.flac",
                        "label_ids": "species:Bp",
                        "split": "train",
                    }
                ],
            )
            _write_csv(
                dclde,
                [
                    {
                        "item_id": "kw-pos",
                        "mat_path": "mat_files/kw.mat",
                        "source_audio": "kw.wav",
                        "source_class_species": "KW",
                        "source_label_ids": "species:Oo",
                        "label_ids": "species:Oo",
                        "split": "train",
                    }
                ],
            )
            _write_csv(
                neg,
                [
                    {
                        "item_id": "gap-bad",
                        "source_audio": "/audio/clip.flac",
                        "filename": "clip.flac",
                        "begin_s": "0",
                        "end_s": "10",
                        "negative_bucket": "primary_adjacent_gap",
                        "label_ids": "",
                        "split": "train",
                    }
                ],
            )
            _write_csv(
                labels,
                [
                    {
                        "item_id": "gap-bad",
                        "model_assisted_review_label": "unlabeled_signal_suspect",
                        "visual_notes": "obvious downsweep",
                    }
                ],
            )

            build_manifests(
                output_dir=root / "out",
                onc_csv=onc,
                biodcase_csv=None,
                dclde_csv=dclde,
                negative_csv=neg,
                model_labels_csv=labels,
                gap_report_csv=None,
            )
            with (root / "out/tables/autoscreened_negative_manifest.csv").open(newline="") as handle:
                kept = list(csv.DictReader(handle))
            with (root / "out/tables/autoscreened_negative_excluded_rows.csv").open(newline="") as handle:
                excluded = list(csv.DictReader(handle))
            self.assertEqual(kept, [])
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["auto_screen_decision"], "excluded_obvious_signal")


if __name__ == "__main__":
    unittest.main()
