import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.train.create_perch2_context_dataset import (  # noqa: E402
    _create_archive,
    _list_archive_members,
)


class TestPerch2ContextDatasetArchive(unittest.TestCase):
    def test_list_archive_members_uses_relative_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            (dataset_dir / "context_audio").mkdir(parents=True)
            (dataset_dir / "context_audio" / "clip_a.wav").write_bytes(b"a")
            (dataset_dir / "summary.json").write_text("{}", encoding="utf-8")

            members = _list_archive_members(dataset_dir)

            self.assertEqual(members, ["context_audio/clip_a.wav", "summary.json"])
            self.assertNotIn(".", members)
            self.assertNotIn("context_audio", members)

    def test_create_archive_tar_can_write_inside_dataset_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            (dataset_dir / "context_audio").mkdir(parents=True)
            (dataset_dir / "context_audio" / "clip_a.wav").write_bytes(b"a")
            (dataset_dir / "context_window_manifest.csv").write_text(
                "clip_id,label\nclip_a.wav,1\n",
                encoding="utf-8",
            )
            archive_path = dataset_dir / "context_dataset.tar"

            _create_archive(
                dataset_dir=dataset_dir,
                output_path=archive_path,
                fmt="tar",
                threads=1,
                zstd_level=3,
                gzip_level=3,
            )

            result = subprocess.run(
                ["tar", "-tf", str(archive_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertEqual(
                result.stdout.splitlines(),
                ["context_audio/clip_a.wav", "context_window_manifest.csv"],
            )


if __name__ == "__main__":
    unittest.main()
