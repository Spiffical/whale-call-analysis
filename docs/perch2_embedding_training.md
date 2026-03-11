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

### 1) Build a 40s context dataset (positive + negative) from 5-minute audio

```bash
/home/sbialek/ONC/whale-call-analysis/.venv/bin/python scripts/data/train/create_perch2_context_dataset.py \
  --excel-files \
    /home/sbialek/ONC/whale-call-analysis/data/finwhales/FinWhale20Hz_CallLibrary_Rannankari_patched.xlsx \
    /home/sbialek/ONC/whale-call-analysis/data/finwhales/Clayoquot_40Hz_Annotations_Rannankari.xlsx \
  --audio-dir /mnt/z/FinWhalesProject/data/audio \
  --context-seconds 40 \
  --negatives-per-positive 1 \
  --output-dir output/perch2_context_dataset \
  --create-archive \
  --archive-format tar.zst \
  --archive-threads 16
```

The generated dataset contains:
- `context_window_manifest.csv`
- `context_audio/*.wav` (40s clips)

### 2) (Optional) Re-archive an existing prepared dataset directory

```bash
bash drac/scripts/create_finwhale_audio_archive.sh \
  --dataset-dir /path/to/perch2_context_dataset_YYYYMMDDTHHMMSSZ \
  --output-path "$PROJECT/whale-call-analysis/data/archives/perch2_context_dataset_YYYYMMDDTHHMMSSZ.tar.zst" \
  --format tar.zst \
  --threads 16
```

### 3) Submit one DRAC training job from context dataset archive

```bash
sbatch drac/scripts/submit_finwhale_perch2_embeddings.sh \
  --context-dataset-tar "$PROJECT/whale-call-analysis/data/archives/perch2_context_dataset_YYYYMMDDTHHMMSSZ.tar.zst" \
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

### 4) Launch a DRAC sweep (dry-run first)

```bash
bash drac/scripts/launch_finwhale_perch2_sweep.sh \
  --context-dataset-tar "$PROJECT/whale-call-analysis/data/archives/perch2_context_dataset_YYYYMMDDTHHMMSSZ.tar.zst" \
  --seeds 42,1337 \
  --logreg-c-list 0.5,1.0,2.0 \
  --center-bias-list 0.25,0.45 \
  --min-gap-list 120,180 \
  --dry-run
```

Then remove `--dry-run` to submit.

Notes for DRAC:
- First-time Perch checkpoint fetch uses Kaggle (`kagglehub`), so configure
  `~/.kaggle/kaggle.json` (or `KAGGLE_CONFIG_DIR`) on the cluster.
- Job logs write to `$SCRATCH/whale-call-analysis/perch2_training_logs/`.
- DRAC training now expects a prebuilt context dataset (`context_window_manifest.csv` + `context_audio/`).
- The submit script extracts `--context-dataset-tar` into `$SLURM_TMPDIR/perch2_context_dataset/`.
- Training outputs default to `$SCRATCH/whale-call-analysis/perch2_training_runs/finwhale/perch2/...`.
- Submit script still preserves `40s` context with train-time de-centering
  (`center_bias_sigma_frac`) when generating 10s train clips.

Augmentation strategy recommendation:
- Generate the 40s context dataset once and transfer/archive it.
- In SLURM jobs, generate decentered 10s train subclips once per run and embed once on GPU.
- Avoid re-sampling augmentations every epoch when embedding in-loop; it multiplies Perch cost.

## Outputs

Each run writes to:

- Direct Python runs: `output/perch2_embedding_training/perch2_<UTC_TIMESTAMP>/`
- SLURM submit script default: `$SCRATCH/whale-call-analysis/perch2_training_runs/finwhale/perch2/.../perch2_<UTC_TIMESTAMP>/`

Artifacts include:

- `context_window_manifest.csv`
- `subclip_manifest.csv`
- `used_subclips.csv`
- `skipped_subclips.csv` (if any)
- `train_subclips.csv`, `val_subclips.csv`, `test_subclips.csv`
- `perch2_logreg.joblib`
- `embeddings.npz` (unless `--skip-save-embeddings`)
- `summary.json`
