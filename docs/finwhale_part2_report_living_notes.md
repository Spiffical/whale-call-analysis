# Fin Whale Part 2 Report Living Notes

Last updated: 2026-03-25

This file is the living specification for the final Part 2 report.
It should be updated whenever new feedback is given about:

- report audience
- report structure
- wording and tone
- which metrics to emphasize or remove
- figures and captions
- training provenance
- testing rationale
- annotation-team recommendations
- rapid-review deliverables

## Audience And Tone

The report should be:

- easy for a layperson to understand
- comprehensive, but concise
- written in plain English
- careful about jargon
- visually clear and publication-ready

Rules:

- Every technical term must be explained in plain language the first time it appears.
- Every acronym must be expanded on first use.
- If we keep shorthand after first use, we should still keep the wording readable.
- Do not assume the reader knows machine-learning terms.
- Do not leave internal tuning terms unexplained. For example:
  - `window step`
  - `threshold`
  - `minimum members`
  - `maximum gap`

Examples:

- `True positive (TP)`: a model result that correctly points us to a time region that contains annotated fin-whale calls.
- `False positive (FP)`: a model result that points us to a region that does not contain annotated fin-whale calls.
- `False negative (FN)`: an annotated fin-whale call that the model did not recover.
- `Confusion matrix`: a table that summarizes how often the model was correct or incorrect.
- `ResNet`: a convolutional neural network architecture used for image-like inputs such as spectrograms.

## Final Report Goals

The final report must clearly answer:

1. How well does the current fin-whale ResNet model work on the new Part 2 dataset?
2. What kinds of calls or sound conditions are still difficult?
3. What should the annotation team prioritize to improve robustness?
4. What high-confidence candidate fin-whale calls should be rapidly reviewed in the unannotated portions of the data?
5. How does current Part 2 performance compare with the model's original held-out baseline on the historical training-era dataset?

## Required Report Content

The final report must include:

- plain-English explanation of the testing strategy and why we used it
- plain-English explanation of the two evaluation views we care about
- performance metrics of the current ResNet model on Part 2
- confusion matrices for merged-region coverage and raw-window detection
- example images for correct and incorrect behavior
- recommendations for the annotation team
- a timestamped rapid-review list from non-annotated regions
- historical baseline results from the original held-out test set
- short explanation of how the current model was originally selected

## Metrics To Include

The final report should emphasize **two** evaluation views only:

### 1. Raw-Window Detection

Definition:

- A `raw window` is one detector input context passed directly to the ResNet.
- In the current pipeline, this is effectively the overlapping detector window used before event merging.
- A raw-window prediction counts as correct if:
  - its score is above the chosen threshold, and
  - the window overlaps at least one annotated fin-whale call

What this tells the reader:

- whether the ResNet itself is “lighting up” on real fin-whale calls before any postprocessing

Recommended plain-English wording:

- “This view asks whether the model notices fin-whale calls when shown short overlapping spectrogram excerpts.”

### 2. Merged-Region Coverage

Definition:

- Nearby positive windows are merged into longer review regions.
- A merged prediction counts as useful if it overlaps annotated fin-whale calls.
- Any annotated fin-whale calls inside a merged predicted region should be counted as successfully found.

What this tells the reader:

- whether the review workflow would bring a human to the right portions of the recording

Recommended plain-English wording:

- “This view asks whether the final review clips produced by the system actually contain real fin-whale calls.”

## Metrics To Exclude From Final Main Narrative

Do **not** feature these as primary report metrics:

- clip recall
- strict one-to-one single-call extraction metric
- clip-level confusion matrices

Notes:

- The strict metric can remain available internally for diagnostics, but it should not be part of the main layperson-facing story unless there is a strong reason later.
- Clip recall is not important for the report audience and should not take space in the main text.
- If confusion matrices are shown, they should be the merged-region and raw-window views only.
- Do not show clip-level confusion matrices in the layperson-facing report.
- Use all four confusion-matrix cells.
- To do that cleanly, define both confusion matrices on the same short detector windows:
  - raw-window matrix: positive if the detector window score is above threshold
  - merged-region matrix: positive if the detector window falls inside a merged review clip
  - actual positive: the detector window overlaps an annotated fin-whale call
- Explain clearly that these confusion matrices are **window-level summaries**, while the main headline metrics are **call-coverage summaries**.

## Testing Strategy To Explain Clearly

The report should explain the testing strategy in simple terms:

### Historical Baseline

Explain that:

- the model was originally trained on earlier fin-whale data from the historical dataset
- we re-ran the saved best checkpoint on the original held-out test split
- this gives a baseline for how well the model performs on data similar to what it saw during development

### Part 2 Evaluation

Explain that:

- the new 2025 annotation workbook contains annotated calls from a new location/time period
- we first built full 5-minute spectrogram inputs from the Part 2 audio
- we then ran the ResNet across overlapping windows
- we combined nearby positive windows into merged review regions
- we evaluated performance in two ways:
  - raw-window detection
  - merged-region coverage

### Why We Used Two Views

Explain plainly:

- raw-window detection tells us how well the detector itself reacts to real calls
- merged-region coverage tells us how useful the final review clips are for human review
- both are important because this project currently accepts merged review clips, not just perfectly isolated calls

### Parameter Explanations

If we mention tuning parameters in the report, they must be translated into plain English.

Examples:

- `window step`: how far the model moves forward between one detector window and the next
- `threshold`: how confident the model must be before we count a window as positive
- `minimum members`: how many nearby positive windows we require before we keep a merged review region
- `maximum gap`: the largest time gap allowed between nearby positive windows when combining them into one review clip

Where possible, timing parameters should be expressed in **seconds**, not bins or frames.

## Model Provenance Section

The report needs a short “How this model was chosen” section.

Current known details:

- current selected model:
  - `finwhale-resnet18-b64-lr3e-4_-tr0.8-none-time_separated-gap120-cbs0p25-seed1337-...`
- architecture:
  - `ResNet-18`
- training dataset source:
  - `wd1.0_ov0.9/new_v1/all_mat_files.tar`
- training split strategy:
  - time-separated
- train / validation / test proportions:
  - 80% / 10% / 10%
- key selected hyperparameters:
  - learning rate `3e-4`
  - batch size `64`
  - no class rebalancing
  - center-bias sigma fraction `0.25`
  - minimum time gap `120 s`
  - seed `1337`
- model selection metric:
  - validation F1 score

Training sweep context:

- the chosen checkpoint came from a structured comparison sweep between at least:
  - overlap `0.9`
  - overlap `0.95`
- the sweep varied:
  - architecture
  - learning rate
  - class balancing strategy
  - center-bias setting
  - minimum gap
  - random seed

Historical held-out baseline for the selected model:

- accuracy `0.9433`
- precision `0.9708`
- recall `0.9662`
- F1 `0.9685`
- AUC `0.9464`

## Figures And Captions

All report figures must:

- be publication-ready quality
- have clear axis labels
- use simple captions
- explain in plain language what the figure shows and why it matters

The final markdown should embed images directly and give each figure a caption.

Very important:

- the main report should **not** make the reader click through links to understand the result
- wherever possible, results should appear directly in the report as:
  - markdown tables
  - embedded images
  - short captions
- links can still exist in internal working notes, but the layperson-facing report should present the content directly
- embedded images inside the markdown file should use **report-relative paths** so the markdown renders correctly when opened normally

Caption style:

- captions should appear **under** the image
- captions should be centered
- captions should be written as `Figure 1: ...`
- do not write `Caption:` above or below the figure

Preferred figures:

1. Historical baseline confusion matrix
2. Part 2 overall metrics comparison figure
   - raw-window detection vs merged-region coverage
3. Subtype comparison figure
   - `20Hz`, `40Hz`, `other_fin`
4. Example contact sheets or selected example images
   - merged true positives
   - merged false positives
   - merged false negatives
   - raw-window true positives
   - raw-window false positives
   - raw-window false negatives
   - raw-window true negatives
5. Rapid-review examples
   - a few top-ranked unmatched predictions

Preferred presentation rule:

- if a figure exists and matters for interpretation, show it directly
- if a metric table is small enough to fit on the page, show it directly
- avoid “see file X” style writing in the final report

Do not use clip-level confusion-matrix figures in the main report.

If we keep any confusion-style summary, it should be directly tied to the two report views:

- raw-window detection
- merged-region coverage

Caption style:

- short
- plain English
- no unexplained abbreviations

## Example Images Requirements

For image examples:

- show only local window or local merged-region context
- only show tags if those tags are present in the actual displayed window/region
- for positive and missed-call examples, include annotation markers above the spectrogram
- keep the current cleaner layout:
  - no spectrogram grid
  - clean colorbar
  - reduced whitespace
  - lower panel shows score trace

Current lower score panel behavior:

- raw-window examples show overlap-maximum score over time, not overlap-average
- the orange line shows the focal raw-window score used for the raw-window TP/FP/TN/FN decision

## Recommendations Section

The recommendation section should not be generic.

It should clearly say things like:

- we need more examples of ship-noise masking of 20 Hz calls
- we need more examples of mixed-species overlap
- we need more examples of faint calls
- we need more examples of 40 Hz calls in masking conditions
- we need more examples of `other_fin` call types in difficult backgrounds

Each recommendation should be traceable to measured errors.

## Rapid Review Section

The report must include:

- a short plain-English explanation of what the rapid-review list is
- a table of timestamped high-confidence predictions from regions without matching fin-whale annotations
- a note that these are candidate calls for fast human review, not confirmed new annotations yet

Recommended columns:

- clip filename
- time within clip
- score
- duration
- local context tags
- local species context if available

## Smoke-Test Status

Current smoke-test conclusion:

- the pipeline is working
- the report structure is now close to what we want
- the two primary metrics to use in the final report are:
  - raw-window detection
  - merged-region coverage
- the smoke subset is too small to use as the final performance estimate
- we can proceed to the full Nibi run

## Current Presentation Preference

Based on user feedback, the current preferred final-report style is:

- fewer file-path references
- more direct presentation
- figures embedded inline
- captions immediately below figures
- key metrics shown as simple markdown tables

Smoke-test caveats:

- only `22` clips in the current smoke subset
- no true-negative clips in the clip-level smoke confusion table
- subtype numbers are too small for stable final conclusions

## Current Open Decisions

These points should be revisited after the full Nibi run:

- which operating point should be the headline setting in the final report
- whether to include one or more secondary operating points
- how many rapid-review examples to embed directly in the markdown
- whether to keep any internal-only diagnostic appendix

## Update Protocol

When the user gives new feedback, update this file rather than scattering decisions across chat history.

New feedback should usually be added to:

- `Audience And Tone`
- `Required Report Content`
- `Metrics To Include`
- `Figures And Captions`
- `Recommendations Section`
- `Current Open Decisions`
