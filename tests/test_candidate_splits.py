import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.multilabel.build_candidate_splits import (  # noqa: E402
    source_label_balanced_grouped_split,
    write_split_outputs,
)
from src.dataset.multilabel import write_csv_rows  # noqa: E402


def _row(source_kind: str, label_id: str, idx: int):
    return {
        "item_id": f"{source_kind}-{label_id}-{idx}",
        "event_group": f"{source_kind}-{label_id}-{idx}",
        "source_kind": source_kind,
        "label_ids": label_id,
    }


class TestCandidateSplits(unittest.TestCase):
    def test_source_label_balanced_keeps_onc_oo_support_in_holdouts(self):
        rows = []
        for label in ("species:Bp", "species:Bm", "species:Mn", "species:Oo"):
            for idx in range(6):
                rows.append(_row("ONC", label, idx))
                rows.append(_row("DCLDE", label, idx))
        for idx in range(12):
            rows.append(
                {
                    "item_id": f"ONC-bg-{idx}",
                    "event_group": f"ONC-bg-{idx}",
                    "source_kind": "ONC",
                    "label_ids": "",
                }
            )

        split_rows = source_label_balanced_grouped_split(rows, seed=17)
        for split in ("val", "test"):
            onc_labels = {
                row["label_ids"]
                for row in split_rows[split]
                if row.get("source_kind") == "ONC" and row.get("label_ids")
            }
            self.assertIn("species:Oo", onc_labels)
            self.assertIn("species:Mn", onc_labels)

    def test_write_split_summary_includes_source_split_label_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            rows = [_row("ONC", "species:Oo", idx) for idx in range(6)]
            rows += [_row("DCLDE", "species:Oo", idx) for idx in range(6)]
            write_csv_rows(manifest, rows)

            summary = write_split_outputs(
                rows,
                root / "splits",
                train_ratio=0.5,
                val_ratio=0.25,
                strategy="source_label_balanced",
                seed=3,
            )

            self.assertEqual(summary["config"]["strategy"], "source_label_balanced")
            self.assertIn("source_split_label_counts", summary)
            self.assertIn("ONC", summary["source_split_label_counts"]["val"])


if __name__ == "__main__":
    unittest.main()
