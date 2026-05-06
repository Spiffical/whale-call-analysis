# ONC Background Human Review Template

These templates are for converting the primary-adjacent gap visual sample into explicit human labels. They are intentionally separate from the model-assisted labels so no row becomes training or calibration background without review.

Templates:

- `tables/onc_primary_adjacent_gap_human_review_template.csv`: all 32 sampled primary-adjacent panels.
- `tables/onc_primary_adjacent_gap_candidate_clean_human_review_template.csv`: the 10 panels that looked most plausibly clean in the model-assisted pass.

Recommended `human_review_label` values:

- `reviewed_background`: no visible whale call or other biological/acoustic event that should count as signal; OK for clean background calibration.
- `ambiguous_hard_negative`: no primary-species call is confirmed, but the panel contains structured energy that should not be used as clean background.
- `unlabeled_signal_suspect`: visible downsweep, tonal blob, pulse, or other signal-like feature that may indicate a missed label.
- `demoted_nonprimary_signal`: clear non-primary biological/source signal; useful as hard negative, not clean background.
- `needs_more_context`: contact-sheet view is insufficient; inspect the surrounding audio/spectrogram before deciding.

Gate rule:

Only rows with `human_review_label=reviewed_background` and `use_as_reviewed_background=yes` should enter deployment background metrics or calibration. Model-assisted `candidate_clean_background` rows are still just candidates until those fields are filled by a human reviewer.

Suggested first pass:

Review the candidate-clean template first. If those 10 rows are mostly accepted as `reviewed_background`, expand to a larger random sample of primary-adjacent gaps. If several are rejected, tighten the gap-selection rules before generating more clean-background candidates.
