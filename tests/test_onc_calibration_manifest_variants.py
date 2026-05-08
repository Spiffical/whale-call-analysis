import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.multilabel.build_onc_calibration_manifest_variants import build_variant
from src.dataset.multilabel import write_csv_rows


def _row(item: str, split: str, source: str, label: str) -> dict:
    return {
        "item_id": item,
        "event_group": item,
        "split": split,
        "source_kind": source,
        "label_ids": label,
        "mat_path": f"mat_files/{item}.mat",
    }


class OncCalibrationManifestVariantsTest(unittest.TestCase):
    def test_caps_external_train_rows_and_oversamples_only_train_onc_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            rows = [
                _row("onc-oo-train-a", "train", "ONC", "species:Oo"),
                _row("onc-oo-train-b", "train", "ONC", "species:Oo"),
                _row("onc-oo-val", "val", "ONC", "species:Oo"),
                _row("onc-bg-train", "train", "ONC", ""),
            ]
            rows.extend(_row(f"dclde-mn-{idx}", "train", "DCLDE", "species:Mn") for idx in range(5))
            rows.extend(_row(f"dclde-mn-val-{idx}", "val", "DCLDE", "species:Mn") for idx in range(2))
            write_csv_rows(manifest, rows)
            vocab = root / "label_vocabulary.json"
            vocab.write_text(json.dumps({"labels": [{"id": "species:Oo"}, {"id": "species:Mn"}]}), encoding="utf-8")

            out = root / "variant"
            summary = build_variant(
                manifest_csv=manifest,
                output_dir=out,
                variant_name="tiny",
                train_caps={("DCLDE", "species:Mn"): 3},
                oversample_targets={("ONC", "species:Oo"): 6},
                seed=11,
                vocab_json=vocab,
            )

            with (out / "standardized_manifest.csv").open(newline="", encoding="utf-8") as handle:
                out_rows = list(csv.DictReader(handle))

            self.assertEqual(sum(1 for row in out_rows if row["split"] == "train" and row["source_kind"] == "DCLDE"), 3)
            self.assertEqual(sum(1 for row in out_rows if row["split"] == "val" and row["source_kind"] == "DCLDE"), 2)
            self.assertEqual(
                sum(1 for row in out_rows if row["split"] == "train" and row["source_kind"] == "ONC" and row["label_ids"] == "species:Oo"),
                6,
            )
            self.assertEqual(
                sum(1 for row in out_rows if row["split"] == "val" and row["source_kind"] == "ONC" and row["label_ids"] == "species:Oo"),
                1,
            )
            self.assertEqual(summary["cap_summary"]["dropped_row_count"], 2)
            self.assertEqual(summary["oversample_summary"]["added_row_count"], 4)
            self.assertTrue((out / "label_vocabulary.json").exists())


if __name__ == "__main__":
    unittest.main()
