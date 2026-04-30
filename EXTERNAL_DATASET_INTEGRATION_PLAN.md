# External Dataset Integration Plan

Date: 2026-04-30

Goal: add public acoustic datasets that improve the multi-species classifier without breaking the current ONC train-style pipeline. External data should enter through the same path as local Part 2 data:

1. Build a call/window manifest with `clip`, `begin_s`, `end_s`, `labels_json`, and `label_ids`.
2. Generate the usual larger context window and 40 s train-style MATs with `scripts/data/part2/prepare_trainstyle_windows.py`.
3. Build leak-aware grouped splits with `scripts/data/multilabel/build_candidate_splits.py`.
4. Train on GPU with `scripts/train/train_multilabel_resnet_smoke.py`, preserving source-dataset fields for per-source evaluation.

## Priority Sources

### 1. BioDCASE 2026 Task 2 / AcousticTrends BlueFin Library

Source:
- Challenge page: https://biodcase.github.io/challenge2026/task2
- Development data: linked Zenodo record from the challenge page
- Underlying annotated library paper: https://www.nature.com/articles/s41598-020-78995-8
- Underlying AAD dataset DOI noted by the paper: https://doi.org/10.26179/5e6056035c01b

Why it is first priority:
- Strong labels include event start/end times, not just clip tags.
- The task is explicitly multi-class and multi-label sound event detection.
- The call set is close to our low-frequency spectrogram pipeline: Antarctic blue whale `BmA`, `BmB`, `BmZ`, `BmD`, and fin whale `Bp20`, `Bp20plus`, `BpD`.
- It adds much more blue-whale diversity than the current ONC balanced splits, while still retaining fin-whale compatibility.
- The original library was built across multiple Antarctic sites, years, instruments, and environmental conditions, so it is useful for testing whether the model generalizes rather than memorizing one ONC deployment.

Current repo support:
- `src/dataset/multilabel.py` now recognizes BioDCASE call aliases:
  `bma`, `bmb`, `bmz`, `bmd`, `bp20`, `bp20plus`, `bp20p`, `bpd`.
- `scripts/data/multilabel/build_biodcase_task2_manifest.py` converts BioDCASE/ATBFL annotations into `selected_calls.csv`, `expected_multilabel_manifest.csv`, `required_audio_filenames.txt`, `label_vocabulary.json`, and `prep_summary.json`.

Recommended first Nibi dry run:

```bash
python scripts/data/multilabel/build_biodcase_task2_manifest.py \
  --annotations-csv /path/to/biodcase/dev/*/annotations.csv \
  --audio-root /path/to/biodcase/dev/audio \
  --require-existing-audio \
  --max-per-label 100 \
  --max-background 200 \
  --output-dir "$EXP_ROOT/external_prep/biodcase_task2_manifest_smoke"

python scripts/data/part2/prepare_trainstyle_windows.py \
  --calls-csv "$EXP_ROOT/external_prep/biodcase_task2_manifest_smoke/selected_calls.csv" \
  --audio-dir /path/to/biodcase/dev/audio \
  --dataset-doc "$FINAL2025_ROOT/historical/training_dataset/dataset_documentation.json" \
  --out-dir "$EXP_ROOT/external_prep/biodcase_task2_manifest_smoke/mat_files" \
  --window-s 40 \
  --edge-context-s 10.5 \
  --spec-backend torch
```

Next small code step after downloading/inspecting the real Zenodo tree:
- Add a submit script modeled on `drac/scripts/submit_multispecies_prep_tiny.sh`, with BioDCASE-specific audio staging.
- Confirm whether challenge files are hour-long or 5-minute windows; absolute datetimes are already supported, so either is acceptable.
- Keep BioDCASE site-year folds separate enough to measure domain transfer.

### 2. DCLDE 2026 killer-whale dataset

Source:
- Scientific Data article: https://www.nature.com/articles/s41597-025-05281-5
- Dataset DOI from the article: https://doi.org/10.25921/15EY-MH50

Why it matters:
- The current ONC species model is weakest on `OD` and `Oo`.
- This corpus is directly targeted at killer whale detection/ecotype classification, with more than 225,000 bounding-box annotations across Northeast Pacific locations.
- It includes ONC-related data and acoustically similar confounders, which is exactly where our OD/Oo/background false positives are showing up.

Integration stance:
- Treat `Oo` as a first-class species label.
- Preserve ecotype labels as metadata or optional future heads, not as V1 species labels.
- Map broad biologic/anthropogenic labels cautiously. Some non-target annotations are weak or incomplete; they should support hard-negative mining only after source-level QA.
- Use a dedicated converter after inspecting `Annotations.csv` columns and provider folder layout.

### 3. NOAA NEFSC / DCLDE 2013 baleen whale and right-whale annotations

Source:
- NOAA Fisheries overview: https://www.fisheries.noaa.gov/resource/data/noaa-nefsc-north-atlantic-right-whale-acoustic-data-and-annotations
- NCEI metadata pages for DCLDE 2013 and NEFSC baleen/right whale annotations.

Why it matters:
- Adds North Atlantic right whale upcalls and additional baleen whale call types.
- The DCLDE 2013 baleen set includes right whale upcalls, humpback song, sei whale downsweeps, fin 20 Hz pulses, minke pulse trains, and blue whale A/B/AB calls.
- Useful for broad species discrimination and low-frequency baleen confusion analysis.

Integration stance:
- Add after BioDCASE and DCLDE killer whale because the geographic domain is farther from ONC.
- Keep as an external validation or auxiliary training source until we confirm audio/annotation licensing and Raven table conventions.

### 4. Watkins Marine Mammal Sound Database

Source:
- WHOI database: https://cis.whoi.edu/science/B/whalesounds/index.cfm

Why it matters:
- Species-rich public reference data, with recordings for many marine mammal species.
- Useful for weak-label pretraining or species embedding sanity checks.

Integration stance:
- Do not mix directly into the strongly labeled event detector as if it were equivalent to call-centered annotations.
- Use only as weak clip-level auxiliary data, contrastive/pretraining data, or curated reference examples after license review.

### 5. Right-whale clip datasets

Sources:
- Kaggle Marinexplore/Cornell Whale Detection Challenge: https://www.kaggle.com/competitions/whale-detection-challenge
- Time Series Classification `RightWhaleCalls`: https://www.timeseriesclassification.com/description.php?Dataset=RightWhaleCalls

Why it matters:
- Adds binary right-whale upcall examples and negatives.

Integration stance:
- Treat as weak, short-window data. It is useful for auxiliary right-whale experiments, not for the first unified 40 s strongly labeled event pipeline.

## Ontology Notes

- Keep species labels and call-type labels separate in one shared multi-label model.
- Do not collapse BioDCASE `Bp20` into local `20Hz` silently; keep the external label as `call:Bp20` and map it to display text "Fin whale 20 Hz pulse". This preserves provenance and lets us later decide whether ONC `20Hz` and BioDCASE `Bp20` should merge.
- Avoid training `OD` and `Oo` as independent positives for the same event unless the source explicitly supports hierarchy. `OD` is a broad toothed-whale bucket; `Oo` is species-level.
- Keep `source_dataset`, deployment/site, annotator, and confidence fields in every manifest row for source-stratified metrics.

## Evaluation Requirements

- Every external-data run must report metrics by source dataset as well as global metrics.
- Use source-aware splits: do not let adjacent windows, source recordings, or site-year groups leak across train/validation/test.
- Keep at least one ONC-only validation slice. External data should improve ONC performance, not just public-benchmark performance.
- Continue using calibrated threshold sweeps, because the current species runs show that threshold choice materially changes macro F1.

## Immediate Next Steps

1. Download or stage the BioDCASE Task 2 development set on Nibi under the multispecies experiment root or a scratch-backed external-data root.
2. Run the new BioDCASE manifest converter with a small per-label cap and inspect `prep_summary.json`.
3. Generate a small MAT dry run with the existing 40 s context machinery.
4. Build label-balanced grouped splits that preserve site-year/source groups.
5. Launch one bounded GPU smoke train that combines ONC balanced100/balanced200 with BioDCASE blue/fin examples, then report source-stratified metrics.
