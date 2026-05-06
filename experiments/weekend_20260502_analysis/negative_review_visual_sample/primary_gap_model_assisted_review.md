# ONC Primary-Adjacent Gap Model-Assisted Review

Generated from `onc_primary_adjacent_gap_review_contact_sheet.png` as a conservative helper table, not as final human review truth.

Artifacts:

- `tables/onc_primary_adjacent_gap_model_assisted_labels.csv`
- `figures/onc_primary_adjacent_gap_review_contact_sheet.png`

Summary:

- `candidate_clean_background`: 10 panels
- `needs_human_review`: 12 panels
- `unlabeled_signal_suspect`: 10 panels

Important interpretation:

- None of these rows should be promoted directly to `reviewed_background`.
- `candidate_clean_background` means the panel looks plausibly clean at contact-sheet scale and is worth human confirmation.
- `unlabeled_signal_suspect` means the panel has localized bright, tonal, blob-like, or downsweep-like structure and should stay out of clean background gates.
- `needs_human_review` means the panel is too ambiguous from the contact sheet because of structured low-frequency energy, tonal bands, or possible weak acoustic events.

Panel guidance:

| model-assisted label | panels | suggested use |
| --- | --- | --- |
| `candidate_clean_background` | 10, 12, 14, 15, 17, 22, 23, 26, 30, 31 | Human-confirm before using as clean reviewed background. |
| `unlabeled_signal_suspect` | 7, 11, 18, 19, 21, 24, 25, 28, 29, 32 | Exclude from clean background gates; consider as hard negative or suspect signal after review. |
| `needs_human_review` | 1, 2, 3, 4, 5, 6, 8, 9, 13, 16, 20, 27 | Keep in review queue; do not use for deployment background metrics yet. |

Decision:

The visual sample supports the current block on full-scale training. The primary-adjacent gap bucket is useful, but it is not clean enough to treat as background wholesale. The next useful step is a human-confirmed review pass that writes explicit labels such as `reviewed_background`, `ambiguous_hard_negative`, `unlabeled_signal_suspect`, or `demoted_nonprimary_signal`.
