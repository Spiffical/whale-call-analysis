# Fin Whale Attention / Weak-Localization Note

## Practical question

Can our existing ResNet-style fin-whale classifiers localize the actual call region in a spectrogram well enough to propose usable masks or boxes, or are CAM-style maps too diffuse and unstable?

## Shortlist

### Primary methods to compare

1. `HiResCAM`
   Why: best current first choice for faithfulness on standard CNNs. The official `pytorch-grad-cam` implementation describes it as elementwise activation-gradient weighting with a faithfulness guarantee for certain CNN settings.

2. `LayerCAM`
   Why: strongest candidate for sharper time-frequency structure because it uses spatially positive gradients and is reported to work better in lower layers, which is useful when a final-layer CAM is too blob-like.

3. `Grad-CAM++`
   Why: still the most useful reference baseline for CAM-style localization on ResNet classifiers. If a newer method cannot beat this clearly, it is hard to justify extra complexity.

4. `Score-CAM`
   Why: good gradient-free comparator. It is much slower, but useful in a pilot to test whether gradient noise is the main reason a CAM is mislocalized.

### Sanity-check methods

5. `Integrated Gradients`
   Why: strong attribution baseline with better axiomatic grounding than many saliency methods. I do not expect it to produce the cleanest masks, but it is valuable for checking whether evidence is concentrated at all.

6. `Occlusion`
   Why: expensive but useful as a small-scale faithfulness check. If masking the top highlighted region barely drops the score, the visualization is not very trustworthy.

### Not prioritized

- `EigenCAM`
  It often looks clean, but the official implementation notes that it is not class-discriminative. That makes it weaker for the question "did the model highlight the fin call?" rather than "where is there strong structure?".

- ViTs or new classifier training
  Not needed for the first pass. CAM tooling is most mature for CNNs, and the current checkpoints are ResNet-based.

## Source-backed takeaways

### Grad-CAM family

- `Grad-CAM` was introduced as a coarse localization map from gradients flowing into the final convolutional layer.
- `Grad-CAM++` extends the weighting with higher-order gradient terms and is commonly used when multiple instances or finer localization are needed.
- `HiResCAM` is the most attractive newer CAM variant for this experiment because it was designed specifically to improve faithfulness rather than just visual appeal.
- `LayerCAM` is attractive when the target occupies a narrow time-frequency band and final-layer CAMs become too broad.

### Integrated Gradients / perturbation methods

- `Integrated Gradients` is still the cleanest principled attribution baseline because it was designed to satisfy Sensitivity and Implementation Invariance.
- Perturbation methods such as occlusion are slower, but they are valuable because they test whether removing highlighted evidence actually changes the model score.

### What the audio literature suggests

- Weakly supervised sound-event localization papers often add architecture support for localization rather than relying only on post-hoc CAM.
- The DCASE 2017 winning weak-label system used temporal attention for localization on log-mel spectrograms.
- A later Interspeech 2020 system used a DenseNet with global average pooling to produce frame-level labels from weak labels.

Practical implication: if post-hoc CAM works well enough here, that is a pleasant simplification. But the audio literature does not suggest we should expect post-hoc CAM on a plain classifier to be the strongest possible localization approach.

## Recommendation for the experiment

### Stage A pilot

Compare:

- `HiResCAM`
- `LayerCAM`
- `Grad-CAM++`
- `Score-CAM`
- `Integrated Gradients`

Use `Occlusion` only on a smaller subset as a faithfulness sanity check because of runtime.

### Stage B quantitative follow-up

Promote the best 1-2 methods from the pilot and score them against annotation-derived time-frequency boxes using:

- box IoU
- temporal IoU
- frequency IoU
- pointing-game hit rate
- mask coverage / precision
- confidence drop when the top activated region is masked

## Working hypothesis

- Best bet for usable proposals: `HiResCAM`, then `LayerCAM`.
- Best legacy baseline: `Grad-CAM++`.
- Useful but probably too expensive for large runs: `Score-CAM`.
- Useful for sanity checking but probably not the final proposal method: `Integrated Gradients`.
- `EigenCAM` is not a good primary choice for this task because class specificity matters.

## Sources

- Grad-CAM, Selvaraju et al. 2016: [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Grad-CAM++, Chattopadhyay et al. 2017: [arXiv:1710.11063](https://arxiv.org/abs/1710.11063)
- HiResCAM, Draelos and Carin 2020: [arXiv:2011.08891](https://arxiv.org/abs/2011.08891)
- Score-CAM, Wang et al. 2019: [arXiv:1910.01279](https://arxiv.org/abs/1910.01279)
- Integrated Gradients, Sundararajan et al. 2017: [PMLR 70](https://proceedings.mlr.press/v70/sundararajan17a.html)
- Weakly supervised audio classification with temporal attention, Xu et al. 2017: [arXiv:1710.00343](https://arxiv.org/abs/1710.00343)
- Weakly supervised acoustic event detection with DenseNet + GAP, Kao et al. 2020: [arXiv:2008.03350](https://arxiv.org/abs/2008.03350)
- Official CAM implementation notes and method list: [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
