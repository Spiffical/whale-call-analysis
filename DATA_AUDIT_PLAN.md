# Data Audit Plan: Multi-Species + Multi-Call-Type

## Purpose

Extend the final2025 fin-whale ResNet work into a broader acoustic classifier without assuming mutually exclusive labels. The first phase is an audit and smoke-test scaffold only: no expensive full training, no overwrite of existing final2025 outputs, and no mutation of the dirty `/home/sbialek/ONC/whale-call-analysis` worktree.

## Starting Point

- Clean worktree: `/home/sbialek/ONC/whale-call-analysis-multispecies`
- Base branch: `origin/main`
- Base commit: `34b4288 Add final 2025 ResNet training utilities`
- Existing dirty worktree to avoid: `/home/sbialek/ONC/whale-call-analysis`
- OCEANS3 schema reference: `/home/sbialek/ONC/labeling-verification-app/OCEANS3_JSON_SCHEMA.md`
- Preferred experiment root on Nibi:
  `/project/6070467/merileo/data/finwhales/final2025_resnet_20260423/multispecies_calltype_experiments/`

## Source Inventory

Audit these sources first:

- Local final2025-style manifests:
  - `/home/sbialek/ONC/whale-call-analysis/tmp/final2025_manifest_check`
  - `/home/sbialek/ONC/whale-call-analysis/tmp/audio_audit_run/manifests`
- Local Part 2 bundles with MATs for smoke tests:
  - `/home/sbialek/ONC/whale-call-analysis/data/finwhale_part2_smoke_bundle`
  - `/home/sbialek/ONC/whale-call-analysis/data/cleanholdout/bundle`
- Workbook inputs:
  - `/home/sbialek/ONC/whale-call-analysis/data/finwhales/Clayoquot_Call_Library_copy.xlsx`
  - `/home/sbialek/ONC/whale-call-analysis/data/finwhales/Clayoquot_2025_SpeciesTemporalAnalysis.xlsx`
  - `/home/sbialek/ONC/whale-call-analysis/data/finwhales/ONC_ClayoquotSlope2025_Annotations_Cetaceans_Instrument_EQ_Sonar_Unknown.xlsx`
  - `/home/sbialek/ONC/whale-call-analysis/data/finwhales/Clayoquot_2025_Analysis_Mar26_Final.xlsx`
- ONC VM review package references, read-only:
  - `/home/sbialek/ONC/finwhale_review_packages`
  - `/home/sbialek/whalestor_mount/finwhale_review_packages`

Use Nibi or the ONC VM only for read-only path checks and large-file inspection until this audit and recommendation set is reviewed.

## Audit Dimensions

For each source, report:

- species code and normalized species label
- call type raw value and normalized call type
- species-call-type pair
- annotation date, month, and year
- device/hydrophone code
- site/deployment if available
- source dataset/workbook/sheet
- reviewed vs unreviewed status
- positive vs background/negative clip status
- clip/window duration and annotation duration
- label confidence/source when available
- duplicate or near-duplicate event groups
- missing audio and missing spectrogram/MAT files when media roots are available

## Leakage And Confounding Checks

Flag:

- adjacent windows from the same source clip or event that could leak across splits
- high-density event clusters within a configurable temporal gap
- labels occurring in only one date/month/device/source dataset
- rare species and call types below the v1 trainability threshold
- device/site/time confounding, especially historical `ICLISTENHF1353` vs 2025 `ICLISTENHF6016`
- background/negative clips drawn from a narrow time range

## V1 Label Scope

The first smoke-trainable model should use clean species and call-type outputs only:

- species labels are multi-label indicators per window/clip
- call-type labels are multi-label indicators per window/clip
- non-biological/context labels such as vessel/masking, sonar, earthquake, and instrument events are audited and preserved as metadata, but not first-class v1 training targets

## Deliverables

- `DATA_AUDIT.md`: observed counts, distributions, rare labels, missing media, and leakage/confounding risks
- `MODELING_RECOMMENDATION.md`: model framing, initialization, heads, loss, thresholds, and excluded labels
- `SPLIT_AND_SAMPLING_PLAN.md`: candidate split policy, grouping, imbalance handling, and smoke/full job progression
- Minimal code for:
  - loading multi-label annotation manifests
  - building candidate grouped splits
  - running a tiny BCE-with-logits smoke training job
  - evaluating a tiny validation subset with multi-label metrics

## Guardrails

- Do not launch full Nibi training in this phase.
- Do not relaunch old sparse April job `12802118` or old all-April job `12801785`.
- Do not overwrite final2025 fin-whale outputs or ONC VM review packages.
- Preserve audio quality; do not downsample audio unless a later reviewed recommendation explicitly justifies it.
- Keep generated audit/training outputs under ignored `output/` locally or the dedicated multispecies experiment root on Nibi.
