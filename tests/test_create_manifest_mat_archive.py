import csv
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.data.multilabel.create_manifest_mat_archive import create_mat_archive


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


class CreateManifestMatArchiveTest(unittest.TestCase):
    def test_archives_unique_mats_and_writes_remapped_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            (data / "a").mkdir(parents=True)
            (data / "b").mkdir(parents=True)
            mat_a = data / "a" / "same.mat"
            mat_b = data / "b" / "same.mat"
            mat_a.write_bytes(b"mat-a")
            mat_b.write_bytes(b"mat-b")
            manifest = root / "manifest.csv"
            _write_csv(
                manifest,
                [
                    {"item_id": "a1", "mat_path": "a/same.mat", "label_ids": "species:Bp"},
                    {"item_id": "a2", "mat_path": "a/same.mat", "label_ids": "species:Bp"},
                    {"item_id": "b1", "mat_path": "b/same.mat", "label_ids": ""},
                ],
            )

            out = root / "archive"
            archive_path = out / "cache.tar"
            summary = create_mat_archive(
                manifest_csv=manifest,
                output_dir=out,
                archive_path=archive_path,
                dataset_root=data,
            )

            self.assertEqual(summary["input_row_count"], 3)
            self.assertEqual(summary["output_row_count"], 3)
            self.assertEqual(summary["unique_mat_count"], 2)
            self.assertEqual(summary["duplicate_mat_reference_count"], 1)
            with (out / "archive_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["mat_path"], rows[1]["mat_path"])
            self.assertNotEqual(rows[0]["mat_path"], rows[2]["mat_path"])
            self.assertTrue(rows[0]["mat_path"].startswith("mat_files/"))

            with tarfile.open(archive_path) as tar:
                members = sorted(tar.getnames())
            self.assertIn("archive_manifest.csv", members)
            self.assertIn("archive_summary.json", members)
            self.assertEqual(len([name for name in members if name.startswith("mat_files/")]), 2)

    def test_missing_mat_is_an_error_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            _write_csv(manifest, [{"item_id": "missing", "mat_path": "missing.mat"}])
            with self.assertRaises(FileNotFoundError):
                create_mat_archive(
                    manifest_csv=manifest,
                    output_dir=root / "out",
                    archive_path=root / "out" / "cache.tar",
                    dataset_root=root,
                )


if __name__ == "__main__":
    unittest.main()
