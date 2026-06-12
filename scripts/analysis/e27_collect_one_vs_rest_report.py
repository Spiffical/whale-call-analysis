#!/usr/bin/env python3
"""Collect E27 ONC-only one-vs-rest expert metrics and ensemble rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import e24_collect_expert_hparam_report as e24  # noqa: E402


def _float_text(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}"


def markdown_report(
    *,
    individual: Sequence[Mapping[str, Any]],
    ensembles: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    lines = [
        "# E27 ONC-Only One-vs-Rest Expert Report",
        "",
        "E27 repeats the E24/E26 per-species expert architecture with ONC-only training, but retains non-active ONC labels as explicit negatives for each expert. Metrics are calibrated on ONC validation rows and evaluated on ONC test rows.",
        "",
        "## Best Ensembles",
        "",
        "| rank | macro F1 | micro F1 | precision | recall | hard FP | fin whale | blue whale | humpback whale |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for rank, row in enumerate(ensembles[:20], start=1):
        lines.append(
            "| {rank} | {macro} | {micro} | {precision} | {recall} | {hard} | {fin} | {blue} | {hump} |".format(
                rank=rank,
                macro=_float_text(row.get("macro_f1")),
                micro=_float_text(row.get("micro_f1")),
                precision=_float_text(row.get("precision")),
                recall=_float_text(row.get("recall")),
                hard=_float_text(row.get("hard_fp_rate")),
                fin=row.get("fin_whale_experiment", ""),
                blue=row.get("blue_whale_experiment", ""),
                hump=row.get("humpback_whale_experiment", ""),
            )
        )

    lines.extend(["", "## Individual Experts", ""])
    for label in e24.THREE_SPECIES:
        label_rows = [row for row in individual if e24.clean(row.get("label_id")) == label and row.get("status") == "complete"]
        label_rows = sorted(label_rows, key=lambda row: float(row.get("macro_f1") or 0.0), reverse=True)
        lines.append(f"### {e24.LABEL_NAMES[label].title()}")
        lines.append("")
        lines.append("| rank | F1 | precision | recall | hard FP | experiment |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for rank, row in enumerate(label_rows, start=1):
            lines.append(
                f"| {rank} | {_float_text(row.get('macro_f1'))} | {_float_text(row.get('precision'))} | {_float_text(row.get('recall'))} | {_float_text(row.get('hard_fp_rate'))} | {row.get('experiment', '')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Variant Checks",
            "",
            "| variant | rows | train | val | test | sources | cap |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in variants:
        lines.append(
            f"| {row.get('variant')} | {row.get('rows')} | {row.get('train_rows')} | {row.get('val_rows')} | {row.get('test_rows')} | {row.get('sources')} | {row.get('cap_strategy')} |"
        )

    lines.extend(
        [
            "",
            "## Comparison Targets",
            "",
            "- E24 best ensemble with external annotations: macro F1 0.9442, micro F1 0.9691, precision 0.9594, recall 0.9790, hard-negative FP 0.1846.",
            "- E26 is the ONC-only filtered-label ablation; compare E27 against E26 to isolate explicit non-target species negatives.",
            "",
            f"Individual metrics CSV: `{output_dir / 'e27_individual_metrics.csv'}`",
            f"Ensemble rankings CSV: `{output_dir / 'e27_ensemble_rankings.csv'}`",
            f"Selected example CSVs: `{output_dir / 'ensembles'}/<ensemble>/selected_examples.csv` for retained ensembles",
            f"Variant CSV: `{output_dir / 'e27_variant_summary.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submitted-tsv", required=True, type=Path)
    parser.add_argument("--plan-tsv", required=True, type=Path)
    parser.add_argument("--variant-root", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submitted = e24.read_tsv(args.submitted_tsv)
    plan = e24.read_tsv(args.plan_tsv)
    individual = e24.run_rows(submitted, plan)
    variants = e24.variant_rows(args.variant_root)
    ensembles = e24.evaluate_ensembles(individual, args.output_dir)
    e24.write_csv(args.output_dir / "e27_individual_metrics.csv", individual)
    e24.write_csv(args.output_dir / "e27_ensemble_rankings.csv", ensembles)
    e24.write_csv(args.output_dir / "e27_variant_summary.csv", variants)
    report = markdown_report(individual=individual, ensembles=ensembles, variants=variants, output_dir=args.output_dir)
    report_path = args.output_dir / "e27_one_vs_rest_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "individual_runs": len(individual), "ensembles": len(ensembles)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
