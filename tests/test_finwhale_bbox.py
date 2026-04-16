import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.finwhale_bbox import (
    ANNOTATION_COLUMNS,
    CLIP_COLUMNS,
    FIN_LABEL_20,
    FIN_LABEL_30,
    FIN_LABEL_40,
    FIN_LABEL_OTHER,
    FIN_LABEL_SONG,
    FIN_SPECIES_CODE,
    HISTORICAL_DATASET,
    PURE_NEGATIVE_DATASET,
    SPECIES_TEMPORAL_DATASET,
    build_bbox_splits,
    build_joint_bbox_manifests,
    parse_historical_workbook,
    standardize_fin_call_type,
)
from src.dataset.finwhale_bbox_export import project_fin_boxes_to_crop


def _write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


class TestFinwhaleBbox(unittest.TestCase):
    def test_standardize_fin_call_type_aliases(self) -> None:
        self.assertEqual(standardize_fin_call_type("20 Hz", FIN_SPECIES_CODE), FIN_LABEL_20)
        self.assertEqual(standardize_fin_call_type("20HZ", FIN_SPECIES_CODE), FIN_LABEL_20)
        self.assertEqual(standardize_fin_call_type("30 Hz note", FIN_SPECIES_CODE), FIN_LABEL_30)
        self.assertEqual(standardize_fin_call_type("40Hz", FIN_SPECIES_CODE), FIN_LABEL_40)
        self.assertEqual(standardize_fin_call_type("S", FIN_SPECIES_CODE), FIN_LABEL_SONG)
        self.assertEqual(standardize_fin_call_type("song pattern", FIN_SPECIES_CODE), FIN_LABEL_SONG)
        self.assertEqual(standardize_fin_call_type("downsweep", FIN_SPECIES_CODE), FIN_LABEL_OTHER)
        self.assertEqual(standardize_fin_call_type("CK", "OD"), "CK")

    def test_parse_historical_workbook_repairs_timestamp_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "historical.xlsx"
            base_cols = {
                "Clip ID": [
                    "ICLISTENHF1353_20190210T032000.068Z.wav",
                    "ICLISTENHF1353_20190210T032000.068Z.wav",
                    "ICLISTENHF1353_20190210T032000.068Z.wav",
                ],
                "begin time (s)": [305.0, 299.0, 70.0],
                "end time (s)": [306.0, 301.0, 69.0],
                "low freq": [17.0, 32.0, 18.0],
                "high freq": [25.0, 44.0, 24.0],
                "peak freq": [21.0, 36.0, 20.0],
                "Peak Power Density (dB FS)": [-18.0, -22.0, -25.0],
                "call type": ["20 Hz", "30Hz", "20Hz"],
                "Note or pattern analysis": ["note", "", ""],
                "Individual note comments ": ["", "boundary", ""],
                "Comments": ["", "", "drop me"],
            }
            sei_cols = {
                "Clip ID": ["ICLISTENHF1353_20190210T033000.068Z.wav"],
                "begin time (s)": [10.0],
                "end time (s)": [12.0],
                "low freq": [45.0],
                "high freq": [60.0],
                "peak freq": [52.0],
                "Peak Power Density (dB FS)": [-17.0],
                "Note or pattern analysis": [""],
                "Comments": ["sei"],
            }
            _write_workbook(
                workbook_path,
                {
                    "July 2018": pd.DataFrame(base_cols),
                    "Sei Whale Calls": pd.DataFrame(sei_cols),
                },
            )

            parsed, summary = parse_historical_workbook(workbook_path)

            self.assertEqual(len(parsed), 3)
            self.assertEqual(summary["drop_reasons"]["nonpositive_duration"], 1)
            self.assertEqual(summary["timestamp_fix_counts"]["minus_300s"], 1)
            self.assertEqual(summary["timestamp_fix_counts"]["clip_end_to_300s"], 1)

            repaired = parsed.loc[parsed["timestamp_fix"] == "minus_300s"].iloc[0]
            self.assertAlmostEqual(float(repaired["begin_time_s"]), 5.0)
            self.assertAlmostEqual(float(repaired["end_time_s"]), 6.0)
            self.assertEqual(repaired["call_type_std"], FIN_LABEL_20)

            boundary = parsed.loc[parsed["timestamp_fix"] == "clip_end_to_300s"].iloc[0]
            self.assertAlmostEqual(float(boundary["begin_time_s"]), 299.0)
            self.assertAlmostEqual(float(boundary["end_time_s"]), 300.0)
            self.assertEqual(boundary["call_type_std"], FIN_LABEL_30)

            sei_row = parsed.loc[parsed["source_sheet"] == "Sei Whale Calls"].iloc[0]
            self.assertEqual(sei_row["species_code"], "Bb")
            self.assertEqual(sei_row["call_type_std"], "")

    def test_parse_historical_workbook_drops_invalid_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "historical_invalid.xlsx"
            _write_workbook(
                workbook_path,
                {
                    "July 2018": pd.DataFrame(
                        {
                            "Clip ID": [
                                "ICLISTENHF1353_20180701T000000.000Z.wav",
                                "random+M614+614:636+614:A638631+614:632",
                            ],
                            "begin time (s)": [10.0, 12.0],
                            "end time (s)": [11.0, 13.0],
                            "low freq": [18.0, 19.0],
                            "high freq": [24.0, 25.0],
                            "call type": ["20Hz", "20Hz"],
                        }
                    )
                },
            )

            parsed, summary = parse_historical_workbook(workbook_path)

            self.assertEqual(len(parsed), 1)
            self.assertEqual(summary["drop_reasons"]["invalid_filename"], 1)
            self.assertEqual(parsed.iloc[0]["filename"], "ICLISTENHF1353_20180701T000000.000Z.wav")

    def test_build_joint_manifests_excludes_guardrailed_pure_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            historical_path = tmpdir_path / "historical.xlsx"
            _write_workbook(
                historical_path,
                {
                    "July 2018": pd.DataFrame(
                        {
                            "Clip ID": ["ICLISTENHF1353_20180701T000000.000Z.wav"],
                            "begin time (s)": [10.0],
                            "end time (s)": [11.0],
                            "low freq": [18.0],
                            "high freq": [25.0],
                            "call type": ["20Hz"],
                        }
                    ),
                    "Sei Whale Calls": pd.DataFrame(
                        {
                            "Clip ID": [],
                            "begin time (s)": [],
                            "end time (s)": [],
                            "low freq": [],
                            "high freq": [],
                        }
                    ),
                },
            )

            species_temporal_path = tmpdir_path / "species_temporal.xlsx"
            _write_workbook(
                species_temporal_path,
                {
                    "Bp_all": pd.DataFrame(
                        {
                            "filename": ["ICLISTENHF6016_20250105T000000.000Z.flac"],
                            "begin_time": [5.0],
                            "end_time": [6.0],
                            "low_freq": [17.0],
                            "high_freq": [24.0],
                            "peak_freq": [20.0],
                            "peak_power": [-19.0],
                            "species": ["Bp"],
                            "call_type": ["20 Hz"],
                            "comments": [""],
                            "annotator": ["a"],
                            "granularity": ["call"],
                        }
                    ),
                    "Bm": pd.DataFrame(
                        {
                            "filename": ["ICLISTENHF6016_20250105T000500.000Z.flac"],
                            "begin_time": [20.0],
                            "end_time": [22.0],
                            "low_freq": [55.0],
                            "high_freq": [90.0],
                            "peak_freq": [70.0],
                            "peak_power": [-15.0],
                            "species": ["Bm"],
                            "call_type": ["Bm"],
                            "comments": [""],
                            "annotator": ["a"],
                            "granularity": ["call"],
                        }
                    ),
                },
            )

            mar18_path = tmpdir_path / "mar18.xlsx"
            _write_workbook(
                mar18_path,
                {
                    "READ ME": pd.DataFrame({"filename": []}),
                    "file_list": pd.DataFrame({"filename": []}),
                    "Jan 2025": pd.DataFrame(
                        {
                            "filename": ["ICLISTENHF6016_20250105T001500.000Z.flac"],
                            "begin_time": [12.0],
                            "end_time": [13.0],
                            "low_freq": [40.0],
                            "high_freq": [60.0],
                            "peak_freq": [50.0],
                            "peak_power": [-20.0],
                            "species": ["OD"],
                            "call_type": ["CK"],
                            "comments": [""],
                            "granularity": ["call"],
                        }
                    ),
                },
            )

            mar26_path = tmpdir_path / "mar26.xlsx"
            mar26_month = pd.DataFrame(
                {
                    "filename": [
                        "ICLISTENHF6016_20250105T000000.000Z.flac",
                        "ICLISTENHF6016_20250105T001500.000Z.flac",
                        "ICLISTENHF6016_20250105T002000.000Z.flac",
                        "ICLISTENHF6016_20250105T002500.000Z.flac",
                        "ICLISTENHF6016_20250105T003000.000Z.flac",
                    ],
                    "verified": [1, 1, 1, 1, 0],
                    "Bp": [0, 0, 1, 0, 0],
                    "Bm": [0, 0, 0, 0, 0],
                    "Mn": [0, 0, 0, 0, 0],
                    "Bb": [0, 0, 0, 0, 0],
                    "OD": [0, 0, 0, 0, 0],
                    "OD_CK": [0, 0, 0, 0, 0],
                    "OD_CK_low": [0, 0, 0, 0, 0],
                    "OD_CK_high": [0, 0, 0, 0, 0],
                    "OD_W": [0, 0, 0, 0, 0],
                    "OD_BP": [0, 0, 0, 0, 0],
                    "CE_unknown": [0, 0, 0, 0, 0],
                }
            )
            _write_workbook(
                mar26_path,
                {
                    "READ ME": pd.DataFrame({"filename": []}),
                    "file_list": pd.DataFrame({"filename": []}),
                    "Jan 2025": mar26_month,
                },
            )

            manifests = build_joint_bbox_manifests(
                historical_workbook=historical_path,
                species_temporal_workbook=species_temporal_path,
                mar26_workbook=mar26_path,
                mar18_workbook=mar18_path,
            )

            pure_neg = manifests["pure_negative_clips"]
            self.assertEqual(len(pure_neg), 1)
            self.assertEqual(
                pure_neg.iloc[0]["filename"],
                "ICLISTENHF6016_20250105T002500.000Z.flac",
            )
            self.assertEqual(manifests["summary"]["pure_negative_clip_count"], 1)
            self.assertEqual(
                manifests["summary"]["mar26_pure_negative_summary"]["drop_reasons"]["excluded_by_annotation_guardrail"],
                2,
            )
            self.assertEqual(
                manifests["summary"]["mar26_pure_negative_summary"]["drop_reasons"]["species_flag_present"],
                1,
            )

    def test_build_bbox_splits_is_day_safe_and_pure_negatives_inherit(self) -> None:
        annotation_rows = [
            {
                "annotation_id": "hist_a1",
                "source_dataset": HISTORICAL_DATASET,
                "source_workbook": "hist.xlsx",
                "source_sheet": "July 2018",
                "source_row_index": 2,
                "filename": "ICLISTENHF1353_20180701T000000.000Z.wav",
                "device_code": "ICLISTENHF1353",
                "clip_start_utc": "2018-07-01T00:00:00+00:00",
                "recording_day_utc": "2018-07-01",
                "species_code": "Bp",
                "is_target_species": 1,
                "call_type_raw": "20Hz",
                "call_type_std": "20Hz",
                "begin_time_s": 10.0,
                "end_time_s": 11.0,
                "duration_s": 1.0,
                "low_freq_hz": 17.0,
                "high_freq_hz": 24.0,
                "peak_freq_hz": 20.0,
                "peak_power_dbfs": -20.0,
                "annotator": "",
                "verified_flag": 0,
                "vessel_flag": 0,
                "granularity": "",
                "comments": "",
                "context_tags": "",
                "timestamp_fix": "none",
                "quality_flags": "",
            },
            {
                "annotation_id": "hist_b1",
                "source_dataset": HISTORICAL_DATASET,
                "source_workbook": "hist.xlsx",
                "source_sheet": "July 2018",
                "source_row_index": 3,
                "filename": "ICLISTENHF1353_20180701T000500.000Z.wav",
                "device_code": "ICLISTENHF1353",
                "clip_start_utc": "2018-07-01T00:05:00+00:00",
                "recording_day_utc": "2018-07-01",
                "species_code": "Bp",
                "is_target_species": 1,
                "call_type_raw": "40Hz",
                "call_type_std": "40Hz",
                "begin_time_s": 20.0,
                "end_time_s": 21.0,
                "duration_s": 1.0,
                "low_freq_hz": 40.0,
                "high_freq_hz": 60.0,
                "peak_freq_hz": 50.0,
                "peak_power_dbfs": -21.0,
                "annotator": "",
                "verified_flag": 0,
                "vessel_flag": 0,
                "granularity": "",
                "comments": "",
                "context_tags": "",
                "timestamp_fix": "none",
                "quality_flags": "",
            },
            {
                "annotation_id": "sp_a1",
                "source_dataset": SPECIES_TEMPORAL_DATASET,
                "source_workbook": "2025.xlsx",
                "source_sheet": "Bp_all",
                "source_row_index": 2,
                "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                "device_code": "ICLISTENHF6016",
                "clip_start_utc": "2025-01-05T00:00:00+00:00",
                "recording_day_utc": "2025-01-05",
                "species_code": "Bp",
                "is_target_species": 1,
                "call_type_raw": "20Hz",
                "call_type_std": "20Hz",
                "begin_time_s": 5.0,
                "end_time_s": 6.0,
                "duration_s": 1.0,
                "low_freq_hz": 17.0,
                "high_freq_hz": 24.0,
                "peak_freq_hz": 20.0,
                "peak_power_dbfs": -20.0,
                "annotator": "a",
                "verified_flag": 0,
                "vessel_flag": 0,
                "granularity": "call",
                "comments": "",
                "context_tags": "",
                "timestamp_fix": "none",
                "quality_flags": "",
            },
        ]
        annotation_df = pd.DataFrame(annotation_rows, columns=ANNOTATION_COLUMNS)
        clip_df = pd.DataFrame(
            [
                {
                    "source_dataset": HISTORICAL_DATASET,
                    "inventory_source": "hist.xlsx",
                    "filename": "ICLISTENHF1353_20180701T000000.000Z.wav",
                    "device_code": "ICLISTENHF1353",
                    "clip_start_utc": "2018-07-01T00:00:00+00:00",
                    "recording_day_utc": "2018-07-01",
                    "is_fin_positive": 1,
                    "is_annotated_non_fin": 0,
                    "is_pure_negative_candidate": 0,
                    "annotation_count": 1,
                    "fin_annotation_count": 1,
                    "non_fin_annotation_count": 0,
                    "species_codes": "Bp",
                    "fin_call_type_stds": "20Hz",
                    "source_workbooks": "hist.xlsx",
                    "verified_flag": 0,
                },
                {
                    "source_dataset": HISTORICAL_DATASET,
                    "inventory_source": "hist.xlsx",
                    "filename": "ICLISTENHF1353_20180701T000500.000Z.wav",
                    "device_code": "ICLISTENHF1353",
                    "clip_start_utc": "2018-07-01T00:05:00+00:00",
                    "recording_day_utc": "2018-07-01",
                    "is_fin_positive": 1,
                    "is_annotated_non_fin": 0,
                    "is_pure_negative_candidate": 0,
                    "annotation_count": 1,
                    "fin_annotation_count": 1,
                    "non_fin_annotation_count": 0,
                    "species_codes": "Bp",
                    "fin_call_type_stds": "40Hz",
                    "source_workbooks": "hist.xlsx",
                    "verified_flag": 0,
                },
                {
                    "source_dataset": SPECIES_TEMPORAL_DATASET,
                    "inventory_source": "2025.xlsx",
                    "filename": "ICLISTENHF6016_20250105T000000.000Z.flac",
                    "device_code": "ICLISTENHF6016",
                    "clip_start_utc": "2025-01-05T00:00:00+00:00",
                    "recording_day_utc": "2025-01-05",
                    "is_fin_positive": 1,
                    "is_annotated_non_fin": 0,
                    "is_pure_negative_candidate": 0,
                    "annotation_count": 1,
                    "fin_annotation_count": 1,
                    "non_fin_annotation_count": 0,
                    "species_codes": "Bp",
                    "fin_call_type_stds": "20Hz",
                    "source_workbooks": "2025.xlsx",
                    "verified_flag": 0,
                },
                {
                    "source_dataset": PURE_NEGATIVE_DATASET,
                    "inventory_source": "mar26.xlsx",
                    "filename": "ICLISTENHF6016_20250105T000500.000Z.flac",
                    "device_code": "ICLISTENHF6016",
                    "clip_start_utc": "2025-01-05T00:05:00+00:00",
                    "recording_day_utc": "2025-01-05",
                    "is_fin_positive": 0,
                    "is_annotated_non_fin": 0,
                    "is_pure_negative_candidate": 1,
                    "annotation_count": 0,
                    "fin_annotation_count": 0,
                    "non_fin_annotation_count": 0,
                    "species_codes": "",
                    "fin_call_type_stds": "",
                    "source_workbooks": "mar26.xlsx",
                    "verified_flag": 1,
                },
                {
                    "source_dataset": PURE_NEGATIVE_DATASET,
                    "inventory_source": "mar26.xlsx",
                    "filename": "ICLISTENHF6016_20250106T000000.000Z.flac",
                    "device_code": "ICLISTENHF6016",
                    "clip_start_utc": "2025-01-06T00:00:00+00:00",
                    "recording_day_utc": "2025-01-06",
                    "is_fin_positive": 0,
                    "is_annotated_non_fin": 0,
                    "is_pure_negative_candidate": 1,
                    "annotation_count": 0,
                    "fin_annotation_count": 0,
                    "non_fin_annotation_count": 0,
                    "species_codes": "",
                    "fin_call_type_stds": "",
                    "source_workbooks": "mar26.xlsx",
                    "verified_flag": 1,
                },
            ],
            columns=CLIP_COLUMNS,
        )

        split_data = build_bbox_splits(annotation_df, clip_df)
        assignments = split_data["assignments"]

        hist_splits = assignments.loc[
            assignments["source_dataset"] == HISTORICAL_DATASET, "split_name"
        ].unique()
        self.assertEqual(len(hist_splits), 1)

        ann_split = assignments.loc[
            assignments["filename"] == "ICLISTENHF6016_20250105T000000.000Z.flac", "split_name"
        ].iloc[0]
        inherited_split = assignments.loc[
            assignments["filename"] == "ICLISTENHF6016_20250105T000500.000Z.flac", "split_name"
        ].iloc[0]
        self.assertEqual(inherited_split, ann_split)

    def test_project_fin_boxes_to_crop_clips_to_time_and_frequency_bounds(self) -> None:
        boxes = project_fin_boxes_to_crop(
            fin_rows=[
                {
                    "annotation_id": "box_a",
                    "begin_time_s": 5.0,
                    "end_time_s": 12.0,
                    "low_freq_hz": 10.0,
                    "high_freq_hz": 50.0,
                }
            ],
            crop_start_s=0.0,
            crop_end_s=10.0,
            freq_min_hz=1.0,
            freq_max_hz=200.0,
            image_width=100,
            image_height=100,
        )

        self.assertEqual(len(boxes), 1)
        x, y, w, h = boxes[0]["bbox"]
        self.assertAlmostEqual(x, 50.0, places=3)
        self.assertAlmostEqual(w, 50.0, places=3)
        self.assertAlmostEqual(y, (9.0 / 199.0) * 100.0, places=3)
        self.assertAlmostEqual(h, (40.0 / 199.0) * 100.0, places=3)


if __name__ == "__main__":
    unittest.main()
