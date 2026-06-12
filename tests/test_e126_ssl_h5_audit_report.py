import json
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
except Exception:
    h5py = None

try:
    import numpy as np
except Exception:
    np = None

from scripts.analysis import e126_ssl_h5_audit_report as audit


class TestE126SslH5AuditReport(unittest.TestCase):
    def test_extract_month_from_common_timestamp_forms(self):
        self.assertEqual(audit.extract_month("ICLISTENHF1234_20250403T120000.wav"), "2025-04")
        self.assertEqual(audit.extract_month("/path/2025-11-20/file.mat"), "2025-11")
        self.assertEqual(audit.extract_month("clip_2025_12_31_235959"), "2025-12")
        self.assertEqual(audit.extract_month("no date here"), "unknown")

    def test_summarize_rows_and_quality_checks(self):
        summary = audit.summarize_rows(
            label_strings=["normal", "normal", "Bm", "Bp", "Mn", "Bm;Mn"],
            splits=["train", "val", "train", "train", "val", "train"],
            source_kinds=["ONC", "ONC", "BioDCASE", "ONC", "DCLDE", "ONC"],
            item_ids=[
                "bg_20250101T000000",
                "bg_20250201T000000",
                "bm_20250301T000000",
                "bp_20250401T000000",
                "mn_20250501T000000",
                "multi_20250601T000000",
            ],
            sources=[""] * 6,
            target_labels=["Bm", "Bp", "Mn"],
            spectrogram_shape=[6, 16, 16, 1],
        )
        self.assertEqual(summary["rows"], 6)
        self.assertEqual(summary["normal_rows"], 2)
        self.assertEqual(summary["normal_train_rows"], 1)
        self.assertEqual(summary["normal_months"], 2)
        self.assertEqual(summary["months"], 6)
        self.assertEqual(summary["target_label_counts"], {"Bm": 2, "Bp": 1, "Mn": 2})
        checks = audit.quality_checks(
            summary,
            min_normal_rows=2,
            min_normal_months=3,
            target_labels=["Bm", "Bp", "Mn"],
        )
        by_name = {row["check"]: row for row in checks}
        self.assertTrue(by_name["normal_rows"]["passed"])
        self.assertFalse(by_name["normal_months"]["passed"])
        self.assertTrue(by_name["target_rows:Bm"]["passed"])

    @unittest.skipIf(h5py is None or np is None, "h5py/numpy are required for H5 audit round-trip")
    def test_run_audit_writes_report_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "dataset.h5"
            string_dtype = h5py.string_dtype(encoding="utf-8")
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("spectrograms", data=np.zeros((3, 4, 4, 1), dtype=np.float32))
                h5.create_dataset("label_strings", data=np.asarray(["normal", "Bm", "Mn"], dtype=object), dtype=string_dtype)
                h5.create_dataset("splits", data=np.asarray(["train", "train", "val"], dtype=object), dtype=string_dtype)
                h5.create_dataset("source_kinds", data=np.asarray(["ONC", "ONC", "ONC"], dtype=object), dtype=string_dtype)
                h5.create_dataset(
                    "item_ids",
                    data=np.asarray(["bg_20250101T000000", "bm_20250201T000000", "mn_20250301T000000"], dtype=object),
                    dtype=string_dtype,
                )
                h5.create_dataset("sources", data=np.asarray([""] * 3, dtype=object), dtype=string_dtype)
            out = root / "out"
            ledger = root / "ledger.md"
            payload = audit.run_audit(
                input_h5=h5_path,
                output_dir=out,
                builder_summary_json=None,
                target_labels=["Bm", "Bp", "Mn"],
                min_normal_rows=1,
                min_normal_months=1,
                ledger_path=ledger,
            )
            self.assertTrue((out / "e126_ssl_h5_audit_report.md").is_file())
            self.assertTrue((out / "e126_ssl_h5_quality_checks.csv").is_file())
            self.assertEqual(payload["outputs"]["ledger"], str(ledger))
            self.assertIn("E126 SSL H5 Coverage Audit", ledger.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
