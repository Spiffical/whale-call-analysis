# E127 Synthetic SSL Suite

E127 tests whether GAVDNet-inspired synthetic augmentation helps scarce blue
whale (`Bm`) and humpback (`Mn`) classes when paired with the E123/E126
SSAMBA-style H5 workflow.

## Source Idea

The Scientific Reports paper "Automated detection of stereotyped animal sounds
using data augmentation and transfer learning" uses a semi-synthetic training
set built from exemplar target calls plus physically motivated audio-domain
augmentation, then fine-tunes a pretrained detector. The authors report strong
blue-whale-song detector performance and release MATLAB code:

- Paper: https://www.nature.com/articles/s41598-026-48308-6
- GAVDNet: https://github.com/b-jancovich/GAVDNet
- customAudioAugmenter: https://github.com/b-jancovich/customAudioAugmenter

Their code is audio-domain MATLAB. Our current E127 implementation is a
spectrogram-space approximation so it can run directly on the E123/E126 H5
datasets. It should be treated as a hypothesis test, not a claim that we have
fully ported GAVDNet.

Implementation notes from the public code inspection:

| GAVDNet/customAudioAugmenter idea | E127 approximation |
| --- | --- |
| Mix exemplar target calls with target-absent background at `snrRange = [-10, 10]`. | Mix target and normal/background spectrograms with default `--snr-db-min -10 --snr-db-max 10`. |
| Mild speed perturbation with `speedup_factor_range = [0.97, 1.03]`. | Time-axis stretch/compression with default `--time-stretch-min 0.97 --time-stretch-max 1.03`. |
| Transmission-loss simulation with strength range `[0.1, 0.75]`. | Smooth time-axis envelope with default `--transmission-loss-min 0.10 --transmission-loss-max 0.75`. |
| Reverberation via `simpleVerb` and `decayTimeRange`. | Optional causal spectrogram reverb smear using `--reverb-smear-strength-*` and `--reverb-smear-decay-*`; default is off for the first controlled suite. |
| Doppler/source-velocity simulation. | Coarse frequency-bin translation; this is not a physical Doppler port. |
| Audio-domain high/low-pass filtering, nonlinear distortion, compression, chorus, and long synthetic sequences. | Not yet ported. These should be follow-up variants only if the first synthetic suite helps. |

## What E127 Varies

The synthetic H5 builder appends synthetic rows only to the training split. It
leaves validation and test rows unchanged so metrics remain real-data metrics.

Default suite variants:

| Variant | Training H5 | Synthetic labels |
| --- | --- | --- |
| `baseline` | original E123/E126 H5 | none |
| `bm` | augmented H5 | `Bm` |
| `mn` | augmented H5 | `Mn` |
| `bm_mn` | augmented H5 | `Bm`, `Mn` |

Synthetic perturbations:

- target spectrogram exemplar sampled from the training split
- normal/background spectrogram sampled from the training split
- frequency translation without circular wrapping
- time-axis stretch/compression
- smooth transmission-loss-like envelope
- optional causal reverb-like time smear
- controlled signal-to-noise mixing
- small Gaussian noise

## Launch

Once the E126 H5 exists:

```bash
bash drac/scripts/submit_multispecies_e127_synthetic_ssl_suite.sh \
  --base-h5 /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/datasets/e126_ssl_e16_low_bgall_target3000_20260612T031656Z.h5 \
  --repo-root /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/repo_e24_expert_hparam_68be99f \
  --ssl-repo-root /scratch/merileo/whale-call-analysis/multispecies_weekend_20260502/selfsupervision_anomalies_onc \
  --synthetic-per-target 1000 \
  --num-pretrain-jobs 2 \
  --num-finetune-jobs 1 \
  --time 03:00:00 \
  --gres gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
```

The submitter writes:

- `e127_synthetic_ssl_suite_plan.tsv`
- one augmentation job script per synthetic variant
- one E123 SSAMBA run root per variant
- one E123 submit log per variant

## Evaluation Requirement

The suite only prepares and trains the variants. A variant does not count as a
completed experiment until it is scored on the same production-style common-row
ONC test set used for E99/E115/E116, with:

- macro/micro F1, precision, recall
- per-species precision/recall/F1
- cross-species false positives
- background false positives
- species-as-background false negatives
- false-positive and false-negative examples

After scoring, build an E124 leaderboard with `--ledger-path
docs/multispecies_experiment_results.md` so the living ledger receives the final
metrics. Review both `e124_candidate_leaderboard.csv` and
`e124_candidate_examples.csv`; if the examples file reports missing or
directory-only examples for the winning candidate, export row-level examples
before treating the variant as reviewed.
