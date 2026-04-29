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
    FIN_SPECIES_CODE,
    INSTRUMENT_SIGNAL_CODE,
    SONAR_SIGNAL_CODE,
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
                "low_freq",
                "high_freq",
                "species",
                "call_type",
                "comments",
                "vessel_presence",
            ],
            ["note row", "", "", "", "", "", "", ""],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac\t2025-01-05\t12:00:00 AM",
                7.5,
                8.0,
                15.0,
                25.0,
                "Bp",
                "20 Hz",
                "vessel masking low frequencies",
                1,
            ],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac",
                298.5,
                299.2,
                15.0,
                25.0,
                "Bp",
                "20 Hz",
                "tail-end call",
                0,
            ],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac",
                45.0,
                49.0,
                1000.0,
                1200.0,
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
                "low_freq",
                "high_freq",
                "species",
                "call_type",
                "comments",
                "vessel_presence",
            ],
            [
                "ICLISTENHF6016_20250205T000000.000Z.flac",
                20.0,
                21.0,
                15.0,
                25.0,
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


def _write_final_style_workbook(path: Path) -> None:
    sheets = {
        "READ ME": [["filename", "species"], ["ignore", "ignore"]],
        "file_list": [
            ["filename", "date_utc"],
            ["ICLISTENHF6016_20250105T000000.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T000500.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T001000.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T001500.000Z.flac", "20250105"],
        ],
        "Cetaceans": [
            [
                "filename",
                "begin_time",
                "end_time",
                "low_freq",
                "high_freq",
                "species",
                "call_type",
                "comment",
                "vessel_presence",
            ],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac",
                5.0,
                6.0,
                15.0,
                25.0,
                "BP",
                "20 Hz",
                "vessel masking low frequencies",
                1,
            ],
            [
                "ICLISTENHF6016_20250105T000500.000Z.flac",
                20.0,
                24.0,
                200.0,
                400.0,
                "CE",
                "",
                "too high for Mn?",
                0,
            ],
        ],
        "hydrophone_thuds": [
            [
                "filename",
                "begin_time",
                "end_time",
                "low_freq",
                "high_freq",
                "peak_freq",
                "peak_power",
                "signal_source",
                "signal_type",
                "comment",
            ],
            [
                "ICLISTENHF6016_20250105T001000.000Z.flac",
                30.0,
                31.0,
                0.0,
                200.0,
                80.0,
                -30.0,
                "",
                "",
                "hydrophone thud",
            ],
        ],
        "sonar": [
            [
                "filename",
                "begin_time",
                "end_time",
                "low_freq",
                "high_freq",
                "peak_freq",
                "peak_power",
                "signal_source",
                "signal_type",
                "comment",
            ],
            [
                "ICLISTENHF6016_20250105T001500.000Z.flac",
                40.0,
                44.0,
                900.0,
                1000.0,
                960.0,
                -40.0,
                "sonar",
                "upsweep",
                "active sonar",
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


def _write_final_style_workbook_without_inventory(path: Path) -> None:
    sheets = {
        "Cetaceans": [
            [
                "filename",
                "begin_time",
                "end_time",
                "low_freq",
                "high_freq",
                "species",
                "call_type",
                "comment",
                "vessel_presence",
            ],
            [
                "ICLISTENHF6016_20250105T000000.000Z.flac",
                5.0,
                6.0,
                15.0,
                25.0,
                "BP",
                "20 Hz",
                "vessel masking low frequencies",
                1,
            ],
        ],
        "hydrophone_thuds": [
            [
                "filename",
                "begin_time",
                "end_time",
                "low_freq",
                "high_freq",
                "peak_freq",
                "peak_power",
                "signal_source",
                "signal_type",
                "comment",
            ],
            [
                "ICLISTENHF6016_20250105T001000.000Z.flac",
                30.0,
                31.0,
                0.0,
                200.0,
                80.0,
                -30.0,
                "",
                "",
                "hydrophone thud",
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


def _write_mar26_supplemental_workbook(path: Path) -> None:
    sheets = {
        "file_list": [
            ["filename", "date_utc"],
            ["ICLISTENHF6016_20250105T000000.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T001000.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T002000.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T003000.000Z.flac", "20250105"],
            ["ICLISTENHF6016_20250105T004000.000Z.flac", "20250105"],
        ],
        "Jan 2025": [
            ["filename", "verified", "Bp", "Bm", "Mn", "Bb", "OD", "CE_unknown"],
            ["ICLISTENHF6016_20250105T000000.000Z.flac", 1, 0, 0, 0, 0, 0, 0],
            ["ICLISTENHF6016_20250105T002000.000Z.flac", 1, 0, 0, 0, 0, 0, 0],
            ["ICLISTENHF6016_20250105T003000.000Z.flac", 1, 0, 0, 0, 0, 0, 0],
            ["ICLISTENHF6016_20250105T004000.000Z.flac", 1, 1, 0, 0, 0, 0, 0],
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


def _write_mar18_guardrail_workbook(path: Path) -> None:
    sheets = {
        "file_list": [
            ["filename", "date_utc"],
            ["ICLISTENHF6016_20250105T002000.000Z.flac", "20250105"],
        ],
        "Jan 2025": [
            ["filename", "species", "begin_time", "end_time", "low_freq", "high_freq"],
            ["ICLISTENHF6016_20250105T002000.000Z.flac", "Bp", 10.0, 11.0, 15.0, 25.0],
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

    def test_build_manifests_from_final_multisheet_workbook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "final_2025.xlsx"
            _write_final_style_workbook(workbook_path)

            manifests = build_part2_manifests(
                workbook_path,
                smoke_per_bucket=1,
                smoke_non_fin=1,
                adjacent_boundary_seconds=0.0,
                seed=1,
            )

            self.assertEqual(manifests["summary"]["annotated_row_count"], 4)
            self.assertEqual(manifests["summary"]["fin_annotation_count"], 1)
            self.assertEqual(manifests["summary"]["annotated_non_fin_clip_count"], 3)
            self.assertEqual(manifests["summary"]["sheet_counts"]["Cetaceans"], 2)
            self.assertEqual(manifests["summary"]["sheet_counts"]["hydrophone_thuds"], 1)
            self.assertEqual(manifests["summary"]["sheet_counts"]["sonar"], 1)
            self.assertEqual(manifests["summary"]["species_counts"][FIN_SPECIES_CODE], 1)
            self.assertEqual(manifests["summary"]["species_counts"][INSTRUMENT_SIGNAL_CODE], 1)
            self.assertEqual(manifests["summary"]["species_counts"][SONAR_SIGNAL_CODE], 1)

            fin_row = manifests["fin_annotations"][0]
            self.assertEqual(fin_row["species"], FIN_SPECIES_CODE)
            self.assertEqual(fin_row["call_type_bucket"], FIN_BUCKET_20)

            clip_manifest = manifests["clip_manifest_by_name"]["ICLISTENHF6016_20250105T001000.000Z.flac"]
            self.assertEqual(clip_manifest["is_fin_positive"], "0")
            self.assertEqual(clip_manifest["is_annotated_non_fin"], "1")
            self.assertIn("non_biological", clip_manifest["context_tags"])
            self.assertIn("instrument_noise", clip_manifest["context_tags"])

    def test_build_manifests_without_inventory_uses_supplemental_workbooks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workbook_path = tmp / "final_2025_no_inventory.xlsx"
            mar26_path = tmp / "mar26.xlsx"
            mar18_path = tmp / "mar18.xlsx"
            _write_final_style_workbook_without_inventory(workbook_path)
            _write_mar26_supplemental_workbook(mar26_path)
            _write_mar18_guardrail_workbook(mar18_path)

            manifests = build_part2_manifests(
                workbook_path,
                mar26_workbook=mar26_path,
                mar18_workbook=mar18_path,
                smoke_per_bucket=1,
                smoke_non_fin=1,
                adjacent_boundary_seconds=0.0,
                seed=1,
            )

            self.assertEqual(manifests["summary"]["inventory_clip_count"], 5)
            self.assertEqual(manifests["summary"]["inventory_source"], str(mar26_path.resolve()))
            self.assertEqual(manifests["summary"]["pure_negative_clip_count"], 1)
            self.assertEqual(
                [row["filename"] for row in manifests["pure_negative_clips"]],
                ["ICLISTENHF6016_20250105T003000.000Z.flac"],
            )
            self.assertIn(
                "ICLISTENHF6016_20250105T003000.000Z.flac",
                {row["filename"] for row in manifests["download_clips"]},
            )
            self.assertNotIn(
                "ICLISTENHF6016_20250105T003000.000Z.flac",
                {row["filename"] for row in manifests["prep_clips"]},
            )
            pure_negative_row = manifests["clip_manifest_by_name"]["ICLISTENHF6016_20250105T003000.000Z.flac"]
            self.assertEqual(pure_negative_row["is_pure_negative_candidate"], "1")
            self.assertEqual(
                manifests["summary"]["mar26_pure_negative_summary"]["drop_reasons"]["excluded_by_annotation_guardrail"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
