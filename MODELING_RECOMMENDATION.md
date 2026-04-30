# Modeling Recommendation

## Recommendation

Use a multi-label formulation for v1. Initialize from the current final2025 ResNet backbone when the input representation matches, replace the binary `num_classes=2` head with a multi-label head, and train with `BCEWithLogitsLoss`.

Do not use plain softmax cross-entropy except for strictly mutually exclusive subproblems. The audited manifests already contain windows with multiple species and multiple call types.

## Output Structure

Use one shared model with grouped label metadata:

- species logits, for example `species:Bp`, `species:Mn`, `species:OD`
- call-type logits, for example `call:20Hz`, `call:40Hz`, `call:other_fin`

The first implementation uses one vector for both groups plus a vocabulary JSON containing group, code, display name, and OCEANS3-compatible `class_hierarchy`. This keeps the training code simple while preserving the option to report species and call-type metrics separately or weight their losses separately.

Avoid species-call-type pair-only labels for v1. Pair labels are useful for analysis, but pair-only training would explode the class space and lose the ability to represent labels like `Fin whale + 20Hz` and `Humpback whale + song` independently in the same window.

## Initialization

Recommended sequence:

1. Smoke test mechanics with the current ResNet architecture and a new multi-label head.
2. Fine-tune from a final2025 ResNet checkpoint by loading matching backbone tensors and skipping the old binary head.
3. Fine-tune the full model with a low learning rate once splits are reviewed.
4. Compare against a general pretrained vision/audio backbone after the data audit is deduplicated.
5. Train from scratch only if the canonical training set has enough non-fin diversity.

The current model is fin-whale-specialized in labels and training distribution, but its spectrogram backbone is still the lowest-risk starting point for inputs matching the final2025 MAT pipeline.

## Trainable V1 Labels

For the smoke path, promote clean biological species labels and fin call-type labels only. Non-biological/context labels stay in the audit and manifest metadata unless later promoted.

Initial trainable labels from the local candidate vocabulary:

- Species: `BA`, `Bm`, `Bp`, `Mn`, `OD`, `P`, `UN`
- Call types: `20Hz`, `40Hz`, `other_fin`

Before full training, raise a minimum-count threshold and likely exclude `BA`, `P`, and other labels with only a handful of clean positives from aggregate metrics. Keep them in the manifest for active learning and future review.

## Ambiguity And Missing Labels

- Store `source`, `review_status`, and optional confidence per label.
- Treat reviewed background/negative clips as "no positive trainable labels present."
- Do not assume every unlabeled class is truly absent in partially annotated fin-whale manifests.
- Add masked-loss support before training on sources where review coverage is class-specific or uncertain.

## Loss And Thresholds

Start with:

- `BCEWithLogitsLoss`
- per-class threshold search on validation
- per-group reporting for species and call types

Add class-balanced BCE, focal loss, or asymmetric loss only after deduplication and split review confirm the rare-label imbalance in the canonical dataset. The smoke trainer includes optional positive class weighting but leaves it off by default.

## Evaluation

Report:

- per-class precision, recall, and F1
- macro F1 and micro F1
- per-class average precision / PR-AUC when enough positives exist
- threshold curves
- performance by year, month, device, and source dataset
- rare-label metrics separately
- pair-level analysis for species-call-type combinations
- event-level performance after adjacent-window postprocessing is defined

Accuracy alone is not useful for this task.

## OCEANS3 Compatibility

Use one item per audio/spectrogram window and write one `model_outputs[]` entry per class path. Human labels/reviews should map into `verifications[].label_decisions[]` when producing strict OCEANS3 packages. The smoke trainer exports `validation_predictions.o3_compatible.json` with raw per-class scores and class hierarchies as a minimal compatibility check.
