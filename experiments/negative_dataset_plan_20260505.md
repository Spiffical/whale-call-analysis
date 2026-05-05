# Negative Dataset Plan

Date: 2026-05-05

Goal: build a much better primary-species dataset for `species:Bp`, `species:Bm`, `species:Mn`, and `species:Oo`, while treating non-primary signatures and source-specific gaps as explicit negative buckets rather than a single background class.

## Label Policy

Primary positives:

- ONC `Bp` -> `species:Bp`.
- ONC `Bm` -> `species:Bm`.
- ONC `Mn` -> `species:Mn`.
- ONC `Oo` -> `species:Oo`.
- BioDCASE `bma/bmb/bmz/bmd` -> `species:Bm`.
- BioDCASE `bp20/bp20plus/bpd` -> `species:Bp`.
- DCLDE `KW` -> `species:Oo`.
- DCLDE `HW` -> `species:Mn`.

Non-primary / negative buckets:

- ONC `OD`, `UN`, `CE`, `MA`, `P`, `Bb`, `BA`, `Pm`, `Lo`, `UNKNOWN` -> non-primary signal metadata; negative for the four primary species unless manually promoted later.
- ONC `INSTRUMENT`, `EQ`, `SONAR`, vessel/instrument/non-biological context -> non-biological hard negatives.
- DCLDE `AB` -> abiotic hard negative.
- DCLDE `UndBio` -> undetermined-biological hard negative.
- BioDCASE gaps between annotated calls -> candidate source-background negatives, not true reviewed clean background.

## Raw Event Counts

Event rows are annotation rows, not unique model windows.

| Dataset | Bp | Bm | Mn | Oo | Non-primary / confounder |
| --- | ---: | ---: | ---: | ---: | --- |
| ONC raw annotations | 31,231 | 134 | 1,319 | 88 | OD 1,567; instrument 96; UN 56; CE 34; MA 25; EQ 14; SONAR 14; UNKNOWN 9; P 8; Bb 7; BA 6; Pm 3; Lo 1 |
| BioDCASE train | 21,079 | 37,489 | 0 | 0 | no explicit non-primary labels |
| DCLDE 2027 audit | 0 | 0 | 124,977 HW | 59,107 KW | UndBio 11,821; AB 11,669 |
| Total raw primary events | 52,310 | 37,623 | 126,296 | 59,195 | explicit non-primary/confounder events >= 26,? depending which ONC rare labels are included |

ONC fin call-type note: raw ONC has `30 Hz` fin annotations, but the current old bucket summary folds them into `other_fin`. Before call-type training, preserve `30Hz` as `call:fin_30hz`.

## Estimated 10-Second Window Counts

Counts are unique 10-second bins, which are closer to the number of possible spectrogram/audio chunks. Multi-label overlap means per-species counts can sum above the "any primary" count.

### ONC

Source basis: `1,629` reviewed/annotated 5-minute clips in `clip_manifest.csv`, assuming `30` non-overlapping 10-second bins per clip.

| Bucket | 10-second bins |
| --- | ---: |
| Total bins | 48,870 |
| Any primary species | 17,129 |
| Bp | 16,279 |
| Bm | 387 |
| Mn | 1,958 |
| Oo | 165 |
| Non-primary signal with no primary overlap | 1,695 |
| Pure-negative-candidate clip bins | 8,580 |
| No-primary bins across all ONC clips | 31,741 |
| Empty/unannotated gap bins across all ONC clips | 30,046 |
| No-primary gap bins inside primary-positive clips | 18,211 |

Interpretation: ONC can give us many negative windows, especially gaps between calls, but these should be reviewed or at least bucketed. Our current audit showed that some pure-negative candidates are visually energetic/signal-like.

### BioDCASE

Source basis: `4,266` train recordings with annotations, treated as 1-hour files for 10-second bin estimates.

| Bucket | 10-second bins |
| --- | ---: |
| Total bins | 1,535,760 |
| Any primary species | 67,561 |
| Bm | 50,604 |
| Bp | 20,092 |
| Candidate no-event/no-primary gaps | 1,468,199 |

Interpretation: BioDCASE has an enormous source-background gap pool, but it is Antarctic/source-specific and not a reviewed ONC deployment background. Sample it, stratify it by site/year, and keep it as source-background rather than clean ONC background.

### DCLDE 2027

Source basis: `13,084` annotated soundfiles, counted up to each file's max annotated end time. This is conservative relative to full file duration.

| Bucket | 10-second bins |
| --- | ---: |
| Total bins to max annotated end | 270,688 |
| Any primary species after HW->Mn and KW->Oo | 82,776 |
| Mn from HW | 54,517 |
| Oo from KW | 28,386 |
| Explicit non-primary signal, no primary overlap | 14,330 |
| AB/abiotic | 9,571 |
| UndBio | 5,720 |
| Candidate no-event gaps | 173,582 |
| Candidate no-primary bins | 187,912 |

Interpretation: DCLDE is not just an Oo repair set. If we include HW as `species:Mn`, it becomes the dominant Mn source. AB/UndBio are valuable hard negatives; DCLDE gaps are useful candidate negatives but should be source-balanced and not treated as ONC clean background.

## Combined Capacity

Approximate unique 10-second windows available:

| Group | Count |
| --- | ---: |
| Positive, any primary | ~167,466 |
| Positive Bp windows | ~36,371 |
| Positive Bm windows | ~50,991 |
| Positive Mn windows | ~56,475 |
| Positive Oo windows | ~28,551 |
| Explicit non-primary hard negatives | ~16,025 |
| Candidate no-primary/gap negatives across all sources | ~1,687,852 |

These are capacity estimates, not the proposed training distribution. We should not train on all ~1.7M candidate negatives blindly.

## Recommended Negative Dataset Design

Create explicit negative buckets:

1. `reviewed_background`: visually reviewed clean background. This should be the only bucket used for deployment background FP gates.
2. `primary_adjacent_gap`: no-primary windows between primary calls inside reviewed ONC clips, with a buffer around primary events.
3. `nonprimary_biological_signal`: OD, unknown biological, DCLDE UndBio, and other non-primary species-like signals.
4. `nonbiological_signal`: instrument, sonar, earthquake, vessel/masking, and DCLDE AB.
5. `external_source_gap`: BioDCASE/DCLDE no-event gaps. Useful for source robustness, not a substitute for ONC reviewed background.
6. `ambiguous_hard_negative`: visually energetic or model-high-scoring no-primary clips that should be separated from clean background.

Windowing proposal:

- Use 10-second non-overlapping windows as the canonical unit.
- Label a positive window if it overlaps a primary event by at least a small threshold, e.g. >=0.25s or event center inside the window.
- Apply an exclusion buffer around primary events, e.g. +/-2s or +/-5s, before sampling clean or gap negatives.
- Mark windows that overlap both primary and non-primary events as multi-label positive plus metadata, not negative.
- Group splits by source file and site/year/provider to avoid leakage from adjacent 10-second windows.

Practical starting training set:

- Use all or capped positives per source/species, with source-aware balancing.
- Use ONC negatives heavily for calibration, but split them into reviewed clean, adjacent-gap, and hard-negative buckets.
- Use DCLDE AB/UndBio as hard negatives and HW/KW as positives.
- Use BioDCASE/DCLDE gap negatives sparingly and source-balanced, because there are far too many and they are source-specific.

Suggested first full-scale species-only dataset target:

- Positives: up to `25k-50k` windows per primary species where available, with all scarce ONC Bm/Oo kept.
- Negatives: roughly match positives per epoch via sampling rather than materializing all negatives.
- Negative sampler proportions: 40% ONC reviewed/adjacent gaps, 25% ONC/non-primary hard negatives, 20% DCLDE AB/UndBio/gaps, 15% BioDCASE gaps.
- Validation: ONC-only deployment validation must include a manually reviewed clean-background bucket; external-source validation is reported separately.

## Blockers Before Training

- Need to create or confirm a clean ONC `reviewed_background` bucket. Current ONC review queue has no row that is safe to mark clean background without more review.
- Need to update DCLDE conversion so `HW` maps to `species:Mn` for the primary species task, instead of being treated only as a confounder.
- Need to preserve ONC `30Hz` as `call:fin_30hz` before any call-type run.
- Need to decide whether the model input should be true 10-second spectrograms, or 40-second context windows with a central 10-second label. The latter is compatible with the current train-style MAT approach; the former is cleaner for the requested chunk definition.
