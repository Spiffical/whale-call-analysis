import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.part2_annotations import (
    FIN_BUCKET_20,
    FIN_BUCKET_OTHER,
    adjacent_clip_filename,
    build_part2_manifests,
    normalize_audio_filename,
)


def _col_name(index: int) -> str:
    out = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _sheet_xml(rows):
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for row_idx, row in enumerate(rows, start=1):
        xml_lines.append(f'<row r="{row_idx}">')
        for col_idx, value in enumerate(row, start=1):
            if value is None:
                continue
            cell_ref = f"{_col_name(col_idx)}{row_idx}"
            if isinstance(value, (int, float)):
                xml_lines.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = (
                    str(value)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                xml_lines.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        xml_lines.append("</row>")
    xml_lines.extend(["</sheetData>", "</worksheet>"])
    return "".join(xml_lines)


def _write_minimal_workbook(path: Path) -> None:
    sheets = {
        "READ ME": [["filename", "species"], ["ignore", "ignore"]],
        "file_list": [
            ["filename", "date_utc"],
            ["ICLISTENHF6016_20250104T235500.000Z.flac", "20250104"],
            ["ICLISTENHF6016_20250105T000000.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T000500.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250205T000000.000Z.flac", "20250205"],
            ["ICLISTENHF6016_20250205T000500.000Z.flac", "20250205"],
        ],
        "Jan 2025": [
            [
                "filename",
                "begin_time",
                "end_time",
                "species",
                "call_type",
                "comments",
                "vessel_presence",
            ],
            ["note row", "", "", "", "", "", ""],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac\t2025-01-05\t12:00:00 AM",
                7.5,
                8.0,
                "Bp",
                "20 Hz",
                "vessel masking low frequencies",
                1,
            ],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac",
                298.5,
                299.2,
                "Bp",
                "20 Hz",
                "tail-end call",
                0,
            ],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac",
                45.0,
                49.0,
                "OD",
                "CK",
                "click train",
                0,
            ],
        ],
        "Feb 2025": [
            [
                "filename",
                "begin_time",
                "end_time",
                "species",
                "call_type",
                "comments",
                "vessel_presence",
            ],
            [
                "ICLISTENHF6016_20250205T000000.000Z.flac",
                20.0,
                21.0,
                "Bp",
                "S",
                "Bp AB song",
                0,
            ],
        ],
    }

    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<sheets>",
    ]
    rels_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]

    with ZipFile(path, "w") as zf:
        for idx, (sheet_name, rows) in enumerate(sheets.items(), start=1):
            workbook_xml.append(
                f'<sheet name="{sheet_name}" sheetId="{idx}" '
                f'r:id="rId{idx}"/>'
            )
            rels_xml.append(
                f'<Relationship Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{idx}.xml"/>'
            )
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))

        workbook_xml.extend(["</sheets>", "</workbook>"])
        rels_xml.append("</Relationships>")
        zf.writestr("xl/workbook.xml", "".join(workbook_xml))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(rels_xml))


class TestPart2Annotations(unittest.TestCase):
    def test_build_manifests_from_multisheet_workbook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "part2.xlsx"
            _write_minimal_workbook(workbook_path)

            manifests = build_part2_manifests(
                workbook_path,
                smoke_per_bucket=1,
                smoke_non_fin=1,
                adjacent_boundary_seconds=10.0,
                seed=1,
            )

            self.assertEqual(manifests["summary"]["inventory_clip_count"], 5)
            self.assertEqual(manifests["summary"]["annotated_row_count"], 4)
            self.assertEqual(manifests["summary"]["fin_annotation_count"], 3)
            self.assertEqual(manifests["summary"]["annotated_non_fin_clip_count"], 0)
            self.assertEqual(manifests["summary"]["adjacent_context_clip_count"], 2)
            self.assertEqual(manifests["summary"]["download_clip_count"], 4)
            self.assertEqual(manifests["summary"]["prep_clip_count"], 2)

            fin_rows = manifests["fin_annotations"]
            self.assertEqual(fin_rows[0]["call_type_bucket"], FIN_BUCKET_20)
            self.assertEqual(fin_rows[-1]["call_type_bucket"], FIN_BUCKET_OTHER)

            clip_manifest = manifests["clip_manifest_by_name"]["ICLISTENHF6016_20250105T000000.000Z.flac"]
            self.assertEqual(clip_manifest["is_fin_positive"], "1")
            self.assertIn("mixed_species", clip_manifest["context_tags"])
            self.assertIn("vessel_or_masking", clip_manifest["context_tags"])

            adjacent_names = {row["filename"] for row in manifests["adjacent_context_clips"]}
            self.assertEqual(
                adjacent_names,
                {
                    "ICLISTENHF6016_20250104T235500.000Z.flac",
                    "ICLISTENHF6016_20250105T000500.000Z.flac",
                },
            )

            smoke_names = {row["filename"] for row in manifests["smoke_clips"]}
            self.assertTrue(smoke_names)

            manifests_with_adjacent_prep = build_part2_manifests(
                workbook_path,
                smoke_per_bucket=1,
                smoke_non_fin=1,
                adjacent_boundary_seconds=10.0,
                include_adjacent_in_prep=True,
                seed=1,
            )
            prep_names = {row["filename"] for row in manifests_with_adjacent_prep["prep_clips"]}
            self.assertEqual(
                prep_names,
                {
                    "ICLISTENHF6016_20250104T235500.000Z.flac",
                    "ICLISTENHF6016_20250105T000000.000Z.flac",
                    "ICLISTENHF6016_20250105T000500.000Z.flac",
                    "ICLISTENHF6016_20250205T000000.000Z.flac",
                },
            )

    def test_adjacent_clip_filename_preserves_extension(self):
        self.assertEqual(
            adjacent_clip_filename("ICLISTENHF6016_20250105T000000.000Z.flac", clip_delta=-1),
            "ICLISTENHF6016_20250104T235500.000Z.flac",
        )
        self.assertEqual(
            adjacent_clip_filename("ICLISTENHF6016_20250105T000000.000Z.wav", clip_delta=1),
            "ICLISTENHF6016_20250105T000500.000Z.wav",
        )

    def test_normalize_audio_filename_extracts_embedded_filename(self):
        self.assertEqual(
            normalize_audio_filename("ICLISTENHF6016_20251105T020000.000Z.flac\t2025-11-05\t2:00:00 AM"),
            "ICLISTENHF6016_20251105T020000.000Z.flac",
        )


if __name__ == "__main__":
    unittest.main()
