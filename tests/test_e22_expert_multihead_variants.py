import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.multilabel.build_e22_expert_multihead_variants import main as _unused  # noqa: F401
from scripts.data.multilabel.build_e22_expert_multihead_variants import E22_VARIANTS
from scripts.data.multilabel import build_e20_diagnostic_variants as e20


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


class E22ExpertMultiheadVariantsTest(unittest.TestCase):
    def test_builds_expert_and_multihead_variant_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "input.csv"
            vocab = root / "vocab.json"
            rows = [
                {"item_id": "onc-fin", "split": "train", "source_kind": "ONC", "label_ids": "species:Bp", "low_mat_path": "a.mat", "mid_mat_path": "a.mat", "high_mat_path": "a.mat", "negative_bucket": ""},
                {"item_id": "bio-blue", "split": "train", "source_kind": "BioDCASE", "label_ids": "species:Bm", "low_mat_path": "b.mat", "mid_mat_path": "b.mat", "high_mat_path": "b.mat", "negative_bucket": ""},
                {"item_id": "dclde-hump", "split": "test", "source_kind": "DCLDE", "label_ids": "species:Mn", "low_mat_path": "c.mat", "mid_mat_path": "c.mat", "high_mat_path": "c.mat", "negative_bucket": ""},
                {"item_id": "onc-killer", "split": "val", "source_kind": "ONC", "label_ids": "species:Oo", "low_mat_path": "d.mat", "mid_mat_path": "d.mat", "high_mat_path": "d.mat", "negative_bucket": ""},
                {"item_id": "dclde-killer", "split": "train", "source_kind": "DCLDE", "label_ids": "species:Oo", "low_mat_path": "f.mat", "mid_mat_path": "f.mat", "high_mat_path": "f.mat", "negative_bucket": ""},
                {"item_id": "onc-bg", "split": "test", "source_kind": "ONC", "label_ids": "", "low_mat_path": "e.mat", "mid_mat_path": "e.mat", "high_mat_path": "e.mat", "negative_bucket": "primary_adjacent_gap"},
            ]
            _write_csv(manifest, rows)
            vocab.write_text(
                json.dumps(
                    {
                        "schema_version": "multilabel-v1",
                        "labels": [
                            {"id": "species:Bp"},
                            {"id": "species:Bm"},
                            {"id": "species:Mn"},
                            {"id": "species:Oo"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            original = e20.VARIANTS
            try:
                e20.VARIANTS = E22_VARIANTS
                index = e20.build_variants(
                    input_manifest=manifest,
                    input_vocab=vocab,
                    output_root=root / "out",
                    seed=2026,
                    dry_run=False,
                )
            finally:
                e20.VARIANTS = original

            names = {row["variant_name"] for row in index}
            self.assertIn("E22_fin_whale_low_expert", names)
            self.assertIn("E22_three_species_multihead_lowmid", names)
            killer = json.loads(
                (root / "out" / "E22_killer_whale_onc_only_midhigh_expert" / "manifest_variant_summary.json").read_text()
            )
            self.assertEqual(killer["active_label_ids"], ["species:Oo"])
            self.assertEqual(killer["sources"], ["ONC"])
            dclde_killer = json.loads(
                (root / "out" / "E22_killer_whale_dclde_only_midhigh_expert" / "manifest_variant_summary.json").read_text()
            )
            self.assertEqual(dclde_killer["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
