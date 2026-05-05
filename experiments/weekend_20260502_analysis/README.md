# Weekend Multi-Species Dataset Analysis

## Short Diagnosis

The external datasets are not failing because they lack signal. They are failing because the model is learning source-specific decision boundaries that do not transfer cleanly back to ONC deployment audio. BioDCASE and DCLDE add real whale examples, but they also shift the model toward higher primary-species scores on ONC background. That shows up as much higher background false positives and weaker ONC Oo/Mn precision.

The species-only E08/E09 retries show that call-type complexity was not the only issue. Removing call labels did not recover deployability: E08 lost macro F1 and still raised background false positives, while E09 roughly matched macro F1 only by accepting a much higher ONC background false-positive rate. E01 remains the best deployable baseline.

## ONC-Gated Metrics

| run | ONC macro F1 | ONC micro F1 | ONC reviewed bg FP | ONC no-primary FP | Bm F1 | Bp F1 | Mn F1 | Oo F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01 ONC control | 0.6372 | 0.6207 | NA | 0.4138 | 0.8889 | 0.5957 | 0.5455 | 0.5185 |
| E04 ONC+BioDCASE species+call | 0.6549 | 0.6289 | NA | 0.6897 | 0.9333 | 0.6316 | 0.5283 | 0.5263 |
| E06 ONC+BioDCASE+DCLDE | 0.5944 | 0.5660 | NA | 0.7241 | 0.9286 | 0.5909 | 0.4898 | 0.3684 |
| E08 ONC+DCLDE species-only | 0.5976 | 0.5890 | NA | 0.5172 | 0.9032 | 0.5789 | 0.5333 | 0.3750 |
| E09 ONC+BioDCASE+DCLDE species-only | 0.6386 | 0.6118 | NA | 0.8276 | 0.9333 | 0.6522 | 0.5600 | 0.4091 |

## What Looks Wrong

- The bucketed parsing audit found no ONC `reviewed_background` rows in these validation artifacts. The earlier "background FP" metric is therefore a `no-primary-label` metric, not a clean-background deployment metric.
- Visual inspection of the ONC `candidate_background` false-positive contact sheets across E01/E04/E06/E08/E09 shows many high-energy vertical broadband events, tonal-looking structure, and ambiguous signal-like clips. These rows may be useful as review candidates or hard negatives, but they are not safe to treat as clean reviewed background.
- Visual inspection of the ONC `demoted_nonprimary_signal` false-positive sheets confirms that these rows contain acoustic events and should be evaluated separately from background. A model firing on these rows is not the same failure mode as firing on clean background.
- E06 added DCLDE killer-whale positives and hard negatives, but ONC Oo precision dropped instead of improving. That means the DCLDE examples are not teaching the model an ONC-compatible killer-whale boundary.
- The biggest deployment problem is calibration on rows without primary species labels. Some of these are true reviewed background, but others are demoted OD, known signal, or pure-negative candidates that still need visual audit; they should not all be interpreted as silent background.
- BioDCASE appears to add useful Bm/Bp signal and raises species recall, but it also makes ONC background look whale-like to the model. That is why its macro F1 can look acceptable while deployment risk increases.
- DCLDE cap200 did not repair ONC Oo. The model learned extra Oo sensitivity, but the DCLDE Oo boundary does not transfer cleanly to ONC Oo versus ONC background.
- Mn and Oo are the fragile labels. Their recall can look acceptable, but precision collapses because the model starts assigning these labels to ONC background or other-species rows.
- Call labels probably made the initial external-data problem harder, but the deeper issue is source/domain calibration. Species-only training alone is not enough.

## Dataset/Training Hypotheses

- Source mismatch: BioDCASE and DCLDE have different hydrophones, annotation styles, event durations, frequency ranges, and background scenes than the ONC held-out target.
- Background definition mismatch: DCLDE hard negatives are selected confounders, while ONC pure-negative candidates include local noise and possibly unlabeled low-frequency events. A negative from one source or workbook bucket is not automatically a clean negative for another.
- Label granularity mismatch: ONC OD was demoted correctly, but DCLDE adds explicit Oo. That helps ontology, yet it changes the class boundary unless ONC-like Oo/background examples anchor it.
- Threshold transfer: thresholds optimized on the mixed validation set do not necessarily produce good ONC deployment thresholds.
- Pos-weight and source imbalance likely encourage sensitivity over specificity, which worsens background false positives. The next ResNet ablation should only happen if it directly tests ONC-specific calibration or source balancing.
- External validation rows are too easy for the external sources relative to ONC background. Mixed validation can pick thresholds that look good globally while failing the ONC deployment distribution.

## Figures

- ![onc_primary_f1_by_model](figures/onc_primary_f1_by_model.png)
- ![onc_background_max_primary_scores](figures/onc_background_max_primary_scores.png)
- ![onc_primary_false_positives](figures/onc_primary_false_positives.png)
- ![onc_background_false_positive_top_labels](figures/onc_background_false_positive_top_labels.png)
- ![source_background_score_distributions](figures/source_background_score_distributions.png)
- ![manifest_source_composition](figures/manifest_source_composition.png)
- ![onc_background_review_queue_contact_sheet](figures/onc_background_review_queue_contact_sheet.png)
- ![e01_onc_control_onc_demoted_nonprimary_signal_fp_contact_sheet](figures/e01_onc_control_onc_demoted_nonprimary_signal_fp_contact_sheet.png)
- ![e01_onc_control_onc_candidate_background_fp_contact_sheet](figures/e01_onc_control_onc_candidate_background_fp_contact_sheet.png)
- ![e04_oncplusbiodcase_speciespluscall_onc_demoted_nonprimary_signal_fp_contact_sheet](figures/e04_oncplusbiodcase_speciespluscall_onc_demoted_nonprimary_signal_fp_contact_sheet.png)
- ![e04_oncplusbiodcase_speciespluscall_onc_candidate_background_fp_contact_sheet](figures/e04_oncplusbiodcase_speciespluscall_onc_candidate_background_fp_contact_sheet.png)
- ![e06_oncplusbiodcaseplusdclde_onc_demoted_nonprimary_signal_fp_contact_sheet](figures/e06_oncplusbiodcaseplusdclde_onc_demoted_nonprimary_signal_fp_contact_sheet.png)
- ![e06_oncplusbiodcaseplusdclde_onc_candidate_background_fp_contact_sheet](figures/e06_oncplusbiodcaseplusdclde_onc_candidate_background_fp_contact_sheet.png)
- ![e08_oncplusdclde_species-only_onc_demoted_nonprimary_signal_fp_contact_sheet](figures/e08_oncplusdclde_species-only_onc_demoted_nonprimary_signal_fp_contact_sheet.png)
- ![e08_oncplusdclde_species-only_onc_candidate_background_fp_contact_sheet](figures/e08_oncplusdclde_species-only_onc_candidate_background_fp_contact_sheet.png)
- ![e09_oncplusbiodcaseplusdclde_species-only_onc_demoted_nonprimary_signal_fp_contact_sheet](figures/e09_oncplusbiodcaseplusdclde_species-only_onc_demoted_nonprimary_signal_fp_contact_sheet.png)
- ![e09_oncplusbiodcaseplusdclde_species-only_onc_candidate_background_fp_contact_sheet](figures/e09_oncplusbiodcaseplusdclde_species-only_onc_candidate_background_fp_contact_sheet.png)
- ![e08_oncplusdclde_species-only_example_images_contact_sheet](figures/e08_oncplusdclde_species-only_example_images_contact_sheet.png)
- ![e09_oncplusbiodcaseplusdclde_species-only_example_images_contact_sheet](figures/e09_oncplusbiodcaseplusdclde_species-only_example_images_contact_sheet.png)

## Reviewed Background Workflow

- Review queue: `tables/onc_background_review_queue.csv`.
- Review contact sheet: `figures/onc_background_review_queue_contact_sheet.png`.
- Review guide: `onc_background_review_guide.md`.
- Fill the `review_label` column with one of `reviewed_background`, `ambiguous_hard_negative`, `unlabeled_signal_suspect`, or `demoted_nonprimary_signal` before using these rows as a deployment background gate.
- Gate future training only on rows marked `reviewed_background`; keep `ambiguous_hard_negative` rows for hard-negative experiments and keep signal rows out of clean-background metrics.

## Recommended Next Experiments

1. Stop broad ResNet scaling for now. E08/E09 show that the issue is not solved by species-only training, and the current ONC validation artifacts do not contain a clean reviewed-background bucket.
2. Build or label a reviewed ONC background/calibration bucket before any full-scale training gate. The current `candidate_background` bucket should be treated as `needs_review` or `ambiguous_hard_negative`, not as clean background.
3. Run ONC-calibrated post-hoc analysis first: per-source thresholds, source-normalized score calibration, and ONC-background hard-negative mining after the reviewed bucket exists.
4. If we run one more ResNet job, make it narrow: species-only ONC+DCLDE or ONC+BioDCASE+DCLDE with source-balanced batches and an ONC-background-heavy validation/calibration split. Do not reintroduce call types yet.
5. Add explicit ONC-like hard negatives for Oo/Mn/background before scaling DCLDE.
6. Prioritize the embedding branch: extract Perch/other foundation embeddings for ONC/BioDCASE/DCLDE caps, train linear/MLP probes, and compare source-separable clusters. If embeddings separate source more strongly than label, that confirms domain shift and suggests adaptation/calibration work before more ResNet training.

## Manifest Composition

### E01 ONC control

- Rows: `582`
- Source counts: `{"ONC": 582}`
- Primary species counts: `{"<background>": 200, "species:Bm": 100, "species:Bp": 101, "species:Mn": 100, "species:Oo": 82}`
- Top call counts: `{"<no-call-label>": 582}`

### E04 ONC+BioDCASE species+call

- Rows: `1032`
- Source counts: `{"BioDCASE": 450, "ONC": 582}`
- Primary species counts: `{"<background>": 300, "species:Bm": 300, "species:Bp": 251, "species:Mn": 100, "species:Oo": 82}`
- Top call counts: `{"<no-call-label>": 522, "call:blue_A": 50, "call:blue_B": 50, "call:blue_D": 50, "call:blue_Z": 50, "call:fin_20hz": 143, "call:fin_20hz_plus": 50, "call:fin_downsweep": 50, "call:fin_other": 5, "call:humpback_song": 59}`

### E06 ONC+BioDCASE+DCLDE

- Rows: `1424`
- Source counts: `{"BioDCASE": 450, "DCLDE": 392, "ONC": 582}`
- Primary species counts: `{"<background>": 498, "species:Bm": 300, "species:Bp": 251, "species:Mn": 100, "species:Oo": 276}`
- Top call counts: `{"<no-call-label>": 720, "call:blue_A": 50, "call:blue_B": 50, "call:blue_D": 50, "call:blue_Z": 50, "call:fin_20hz": 143, "call:fin_20hz_plus": 50, "call:fin_downsweep": 50, "call:humpback_song": 59, "call:orca_call": 194}`

### E08 ONC+DCLDE species-only

- Rows: `974`
- Source counts: `{"DCLDE": 392, "ONC": 582}`
- Primary species counts: `{"<background>": 398, "species:Bm": 100, "species:Bp": 101, "species:Mn": 100, "species:Oo": 276}`
- Top call counts: `{"<no-call-label>": 974}`

### E09 ONC+BioDCASE+DCLDE species-only

- Rows: `1424`
- Source counts: `{"BioDCASE": 450, "DCLDE": 392, "ONC": 582}`
- Primary species counts: `{"<background>": 498, "species:Bm": 300, "species:Bp": 251, "species:Mn": 100, "species:Oo": 276}`
- Top call counts: `{"<no-call-label>": 1424}`
