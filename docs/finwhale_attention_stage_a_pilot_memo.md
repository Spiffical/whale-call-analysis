# Fin Whale Attention Stage A Pilot Memo

## Question

Can CAM-style explainability maps from our existing ResNet fin-whale classifiers localize the actual call region tightly enough to propose usable masks or boxes, without training a dedicated detector?

## Pilot Setup

- Data: stratified annotated 2025 Clayoquot subset from the Part 2 bundle
- Models:
  - `baseline`
  - `balanced`
  - `highperf`
- Methods:
  - `gradcampp`
  - `hirescam`
  - `layercam`
  - `scorecam`
  - `integrated_gradients`
- Output bundle:
  - `/scratch/merileo/finwhale_attention_experiment/finwhale_attention_pilot_20260409_194330`

## Main Result

The pilot does not support using CAMs as the main fin-whale box generator.

Top-line localization metrics were weak across all three models:

| model | method | box IoU | temporal IoU | frequency IoU | pointing | mask coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced | layercam | 0.019 | 0.087 | 0.215 | 0.058 | 0.890 |
| balanced | gradcampp | 0.019 | 0.087 | 0.216 | 0.058 | 0.887 |
| baseline | gradcampp | 0.020 | 0.086 | 0.233 | 0.038 | 0.863 |
| highperf | layercam | 0.018 | 0.087 | 0.197 | 0.038 | 0.886 |
| highperf | scorecam | 0.032 | 0.135 | 0.187 | 0.058 | 0.669 |

Interpretation:

- Box IoU stayed extremely low, usually around `0.02`
- Pointing accuracy was weak, usually `0.02` to `0.08`
- `gradcampp`, `layercam`, and `hirescam` often covered most of the crop rather than isolating the call
- `integrated_gradients` had the opposite failure mode: very sparse maps with poor overlap
- `scorecam` occasionally improved the high-performance model, but not enough to rescue box quality

## Failure Modes

The dominant failure patterns were:

- `diffuse_activation`
- `missed_call_region`
- `time_shifted`

These were especially common in:

- `mixed_species`
- `vessel_or_masking`
- `faint`

The maps looked more like broad evidence regions than useful call-localization masks.

## Practical Recommendation

Recommendation after Stage A:

- Use `layercam` and `gradcampp` only as the held-out confirmation methods for Stage B
- Do not treat CAM outputs as the primary localization product
- Move toward a dedicated detector if Stage B confirms the same pattern on held-out 2025 data

Recommended fallback model sequence:

1. `RT-DETR` for box detection
2. a lightweight segmentation model if masks become more important than boxes

## Held-Out Stage B Status

A full held-out `quant` run was launched on `part2_eval_test` using `gradcampp` and `layercam` across the `baseline`, `balanced`, and `highperf` models.

- Job: `12017914`
- Status: `TIMEOUT`
- Elapsed: about `8` hours
- Output root:
  - `/scratch/merileo/finwhale_attention_experiment/finwhale_attention_quant_20260410_141832`

That run produced only partial `arrays/` and `gallery/` artifacts before the wall-clock limit, and did not write aggregate summary tables.

This means the clean held-out quant confirmation is still incomplete, but it does not overturn the practical conclusion from Stage A: the current CAM maps are already too diffuse and weakly aligned to justify using them as the main localization method.

## Files To Inspect

- Summary table:
  - `/scratch/merileo/finwhale_attention_experiment/finwhale_attention_pilot_20260409_194330/pilot_summary.md`
- Method ranking:
  - `/scratch/merileo/finwhale_attention_experiment/finwhale_attention_pilot_20260409_194330/method_ranking.csv`
- Failure analysis:
  - `/scratch/merileo/finwhale_attention_experiment/finwhale_attention_pilot_20260409_194330/failure_analysis.csv`
- Example galleries:
  - `/scratch/merileo/finwhale_attention_experiment/finwhale_attention_pilot_20260409_194330/gallery`
  - `/scratch/merileo/finwhale_attention_experiment/finwhale_attention_pilot_20260409_194330/gallery_negative`
