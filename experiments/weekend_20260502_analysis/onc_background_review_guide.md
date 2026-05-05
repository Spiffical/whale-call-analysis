# ONC Background Review Guide

This guide turns the ONC no-primary review queue into a concrete labeling pass.
Use it with `figures/onc_background_review_queue_contact_sheet.png` and
`tables/onc_background_review_queue.csv`.

## Label Choices

- `reviewed_background`: clean negative audio after visual review; safe for deployment background gates.
- `ambiguous_hard_negative`: not a primary species label, but visually energetic or signal-like enough that it should not define clean background.
- `unlabeled_signal_suspect`: likely biological/acoustic signal with incomplete or missing label.
- `demoted_nonprimary_signal`: known non-primary or demoted signal, such as OD; evaluate separately from background.

## First-Pass Recommendation

Do not mark any queued row as `reviewed_background` yet. The candidate-background rows are visually energetic or signal-like, and the demoted rows already represent known signal. Treat the queue as review/calibration work, not training clearance.

Recommended first pass:

- Accept all `demoted_nonprimary_signal` suggestions for the 14 demoted rows.
- Mark 14 of 15 candidate-background rows as `ambiguous_hard_negative`.
- Leave the one low-priority candidate row undecided until direct visual/audio review.

## Review Queue

| # | Priority | Proposed label | Bucket | Top label | Max score | Triggered runs | Clip |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Mn | 0.978689 | 5 | `ICLISTENHF6016_20250215T010000.000Z.flac_276.9s_282.1s_trainstyle` |
| 2 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.967529 | 5 | `ICLISTENHF6016_20250415T130000.000Z-LPF.flac_130.0s_170.0s_trainstyle` |
| 3 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Mn | 0.957438 | 5 | `ICLISTENHF6016_20251010T000000.000Z.flac_34.1s_38.8s_trainstyle` |
| 4 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Mn | 0.947266 | 5 | `ICLISTENHF6016_20250105T063000.000Z.flac_8.4s_9.6s_trainstyle` |
| 5 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.935169 | 3 | `ICLISTENHF6016_20250525T010000.000Z.flac_28.7s_40.3s_trainstyle` |
| 6 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Mn | 0.932215 | 5 | `ICLISTENHF6016_20251020T070000.000Z.flac_52.5s_55.9s_trainstyle` |
| 7 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.916460 | 5 | `ICLISTENHF6016_20250405T120000.000Z.flac_130.0s_170.0s_trainstyle` |
| 8 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.906413 | 4 | `ICLISTENHF6016_20250715T150000.000Z.flac_130.0s_170.0s_trainstyle` |
| 9 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.901446 | 5 | `ICLISTENHF6016_20250425T090000.000Z.flac_130.0s_170.0s_trainstyle` |
| 10 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.897934 | 2 | `ICLISTENHF6016_20250720T040000.000Z.flac_3.0s_3.4s_trainstyle` |
| 11 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.894142 | 3 | `ICLISTENHF6016_20250610T080000.000Z.flac_130.0s_170.0s_trainstyle` |
| 12 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.889873 | 2 | `ICLISTENHF6016_20250330T100000.000Z.flac_130.0s_170.0s_trainstyle` |
| 13 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.873592 | 2 | `ICLISTENHF6016_20250815T060000.000Z.flac_44.2s_47.6s_trainstyle` |
| 14 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.873393 | 2 | `ICLISTENHF6016_20250420T080000.000Z.flac_130.0s_170.0s_trainstyle` |
| 15 | high | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.872987 | 4 | `ICLISTENHF6016_20250720T190000.000Z.flac_187.3s_188.5s_trainstyle` |
| 16 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.865207 | 4 | `ICLISTENHF6016_20250410T130000.000Z.flac_130.0s_170.0s_trainstyle` |
| 17 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.859280 | 5 | `ICLISTENHF6016_20250620T230000.000Z-LPF.flac_130.0s_170.0s_trainstyle` |
| 18 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.858897 | 4 | `ICLISTENHF6016_20250430T220000.000Z.flac_130.0s_170.0s_trainstyle` |
| 19 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.858758 | 3 | `ICLISTENHF6016_20250115T010000.000Z.flac_130.0s_170.0s_trainstyle` |
| 20 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.834188 | 4 | `ICLISTENHF6016_20250620T000000.000Z-LPF.flac_130.0s_170.0s_trainstyle` |
| 21 | high | `ambiguous_hard_negative` | candidate_background | Oo | 0.776706 | 4 | `ICLISTENHF6016_20250730T080000.000Z.flac_130.0s_170.0s_trainstyle` |
| 22 | medium | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.838582 | 1 | `ICLISTENHF6016_20250905T190000.000Z.flac_248.9s_263.7s_trainstyle` |
| 23 | medium | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.835062 | 2 | `ICLISTENHF6016_20250710T070000.000Z.flac_270.3s_270.5s_trainstyle` |
| 24 | medium | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.833916 | 3 | `ICLISTENHF6016_20250830T110000.000Z.flac_293.5s_297.8s_trainstyle` |
| 25 | medium | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.745505 | 0 | `ICLISTENHF6016_20250510T040000.000Z.flac_165.5s_170.5s_trainstyle` |
| 26 | medium | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Oo | 0.725330 | 2 | `ICLISTENHF6016_20250705T180000.000Z.flac_154.6s_155.3s_trainstyle` |
| 27 | medium | `ambiguous_hard_negative` | candidate_background | Oo | 0.716938 | 2 | `ICLISTENHF6016_20250815T170000.000Z.flac_130.0s_170.0s_trainstyle` |
| 28 | medium | `demoted_nonprimary_signal` | demoted_nonprimary_signal | Mn | 0.673533 | 1 | `ICLISTENHF6016_20250820T140000.000Z.flac_230.5s_230.9s_trainstyle` |
| 29 | low | needs visual review | candidate_background | Oo | 0.649572 | 0 | `ICLISTENHF6016_20250605T030000.000Z.flac_130.0s_170.0s_trainstyle` |

## Next Calibration Step

After review labels are accepted or corrected, write them back to `tables/onc_background_review_queue.csv`. The analysis gate should then compute clean background false-positive rate only from rows labeled `reviewed_background`, while reporting `ambiguous_hard_negative` and `demoted_nonprimary_signal` separately.
