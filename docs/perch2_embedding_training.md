# Perch 2.0 Embedding Training

This repo now includes an embedding-based training script:

- `scripts/train/train_perch2_embeddings.py`

It trains a binary call/background classifier using Perch 2.0 embeddings from
raw audio windows (no spectrogram image input), following the same two-stage
idea as the spectrogram pipeline:

- Build base context windows (default `40s`)
- Sample train/eval subclips from those contexts (default `10s`)

Training augmentation:
- Train positives are decentered by default, mirroring the CNN
  `center_bias_sigma_frac` logic.
- Train negatives are randomly positioned within the context by default.
- Controls:
  - `--context-seconds` (default `40`)
  - `--train-clip-seconds` / `--eval-clip-seconds` (default `10` / `10`)
  - `--train-pos-augment-copies` (default `1`)
  - `--train-neg-augment-copies` (default `1`)
  - `--center-bias-sigma-frac` (default `0.25`)

## Install

```bash
/home/sbialek/ONC/whale-call-analysis/.venv/bin/pip install -r requirements-perch.txt
```

Notes:
- First run downloads the Perch checkpoint from Kaggle through `kagglehub`.
- If you want CPU-only TensorFlow execution, add `--disable-gpu`.
- Perch v2 uses 5s internal model windows at 32 kHz; longer subclips are
  framed internally and pooled to one embedding vector per subclip.
- The trainer now runs an audio preflight and fails early with clear clip-id
  examples if manifest clip IDs do not map to files in `--audio-dir`.

## Example (small/quick smoke run)

```bash
/home/sbialek/ONC/whale-call-analysis/.venv/bin/python scripts/train/train_perch2_embeddings.py \
  --excel-files \
    /home/sbialek/ONC/whale-call-analysis/data/finwhales/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx \
    /home/sbialek/ONC/whale-call-analysis/data/finwhales/Clayoquot_40Hz_Annotations_Rannankari.xlsx \
  --audio-dir /mnt/z/FinWhalesProject/data/audio \
  --perch-model perch_v2_cpu \
  --disable-gpu \
  --max-positives 100 \
  --max-audio-files 40 \
  --negatives-per-positive 1 \
  --batch-size 8 \
  --context-seconds 40 \
  --train-clip-seconds 10 \
  --eval-clip-seconds 10 \
  --train-pos-augment-copies 1 \
  --train-neg-augment-copies 1 \
  --center-bias-sigma-frac 0.25
```

## Full run template

```bash
/home/sbialek/ONC/whale-call-analysis/.venv/bin/python scripts/train/train_perch2_embeddings.py \
  --excel-files \
    /home/sbialek/ONC/whale-call-analysis/data/finwhales/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx \
    /home/sbialek/ONC/whale-call-analysis/data/finwhales/Clayoquot_40Hz_Annotations_Rannankari.xlsx \
  --audio-dir /mnt/z/FinWhalesProject/data/audio \
  --perch-model perch_v2_cpu \
  --disable-gpu \
  --negatives-per-positive 1 \
  --batch-size 16 \
  --context-seconds 40 \
  --train-clip-seconds 10 \
  --eval-clip-seconds 10 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --min-gap-seconds 120 \
  --train-pos-augment-copies 1 \
  --train-neg-augment-copies 1 \
  --center-bias-sigma-frac 0.25
```

## DRAC

Build an audio archive once in project storage (recommended for fast node-local
extract per job):

```bash
bash drac/scripts/create_finwhale_audio_archive.sh \
  --audio-dir /mnt/z/FinWhalesProject/data/audio \
  --output-path "$PROJECT/whale-call-analysis/data/archives/finwhale_audio_20260303.tar.zst" \
  --format tar.zst \
  --threads 16
```

Submit one DRAC training job:

```bash
sbatch drac/scripts/submit_finwhale_perch2_embeddings.sh \
  --excel-file /path/to/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx \
  --excel-file /path/to/Clayoquot_40Hz_Annotations_Rannankari.xlsx \
  --audio-tar-path "$PROJECT/whale-call-analysis/data/archives/finwhale_audio_20260303.tar.zst" \
  --perch-model perch_v2_gpu \
  --batch-size 16 \
  --context-seconds 40 \
  --train-clip-seconds 10 \
  --eval-clip-seconds 10 \
  --train-pos-augment-copies 1 \
  --train-neg-augment-copies 1 \
  --center-bias-sigma-frac 0.25 \
  --min-gap-seconds 120 \
  --seed 42
```

Launch a DRAC sweep (with dry-run first):

```bash
bash drac/scripts/launch_finwhale_perch2_sweep.sh \
  --excel-file /path/to/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx \
  --excel-file /path/to/Clayoquot_40Hz_Annotations_Rannankari.xlsx \
  --audio-tar-path "$PROJECT/whale-call-analysis/data/archives/finwhale_audio_20260303.tar.zst" \
  --seeds 42,1337 \
  --logreg-c-list 0.5,1.0,2.0 \
  --center-bias-list 0.25,0.45 \
  --min-gap-list 120,180 \
  --dry-run
```

Then remove `--dry-run` to submit.

Local orchestration smoke test (no `sbatch`):

```bash
bash drac/scripts/submit_finwhale_perch2_embeddings.sh \
  --local-test-mode \
  --excel-file /path/to/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx \
  --excel-file /path/to/Clayoquot_40Hz_Annotations_Rannankari.xlsx \
  --audio-tar-path /path/to/finwhale_audio.tar.zst \
  --perch-model perch_v2_cpu \
  --disable-gpu \
  --max-positives 8 \
  --skip-save-embeddings \
  --exp-dir output/perch2_drac_local_test
```

Notes for DRAC:
- First-time Perch checkpoint fetch uses Kaggle (`kagglehub`), so configure
  `~/.kaggle/kaggle.json` (or `KAGGLE_CONFIG_DIR`) on the cluster.
- Job logs write to `$SCRATCH/whale-call-analysis/perch2_training_logs/`.
- `--audio-tar-path` is preferred over `--copy-audio-to-tmp` for large datasets.
- Submit script defaults still preserve the `40s` context and train-time
  de-centering augmentation (`center_bias_sigma_frac`).

Augmentation strategy recommendation:
- For Perch embeddings, generate train augmentations once at job start
  (current pipeline behavior) and embed them once on GPU.
- Do not re-sample augmentations every epoch when Perch embedding extraction is
  in-loop; that multiplies expensive forward passes and usually hurts throughput.
- If you want more augmentation diversity, increase
  `--train-pos-augment-copies` / `--train-neg-augment-copies` in one pass, or
  run multiple seeds/sweeps.

## Outputs

Each run writes to:

- `output/perch2_embedding_training/perch2_<UTC_TIMESTAMP>/`

Artifacts include:

- `context_window_manifest.csv`
- `subclip_manifest.csv`
- `used_subclips.csv`
- `skipped_subclips.csv` (if any)
- `train_subclips.csv`, `val_subclips.csv`, `test_subclips.csv`
- `perch2_logreg.joblib`
- `embeddings.npz` (unless `--skip-save-embeddings`)
- `summary.json`
