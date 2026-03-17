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

### Nibi / DRAC install

The Alliance wheelhouse currently tops out at `tensorflow 2.19.1`, but Perch
v2 runtime needs `tensorflow >= 2.20.0`. The repo also does not need the
optional `pandas[gcp]` dependency chain that `perch-hoplite` normally pulls
in, so the clean cluster install path is:

```bash
source .venv/bin/activate
bash drac/scripts/install_perch_nibi.sh
```

What that helper does:
- replaces the Alliance `setuptools` build with upstream `setuptools<81` so
  `pkg_resources` is available for `tensorflow_hub`
- installs `simsimd` from source so `usearch` can link against a compatible
  shared library
- builds `usearch` from source because the published wheel is not compatible
  with the cluster glibc
- installs upstream `tensorflow==2.20.0` with isolated pip so the Perch v2
  XLA runtime ops are available
- installs `tensorflow-hub` and the remaining runtime dependencies
- installs the Perch runtime dependencies needed by this repo
- installs `perch-hoplite` without the optional `pandas[gcp]` dependency chain
  that pulls in the Alliance dummy `pyarrow` wheel
- verifies `perch_hoplite.zoo.model_configs` imports successfully

If you must stay on the Alliance-provided `tensorflow 2.19.1` stack, Perch v2
will fail at runtime with:

```text
XlaCallModuleOp with version 10 is not supported by this build. Must be <= 9
```

In that case, the practical fallback is to train with `--perch-model perch_8`
instead of a Perch v2 preset. `perch_8` is an older Perch release with a
different embedding size, so do not compare those embeddings directly to
Perch v2 runs.

### Perch v2 on Nibi via Apptainer

For actual Perch v2 runs on Nibi, prefer a containerized TensorFlow `2.20.0`
runtime instead of the Alliance wheelhouse TensorFlow build.

1. Prepare the container-backed runtime once on the login node:

```bash
git pull
bash drac/scripts/prepare_perch2_apptainer_env.sh
```

That script pulls `docker://tensorflow/tensorflow:2.20.0-gpu`, creates a
virtualenv inside the container runtime under `$SCRATCH`, installs Perch
dependencies there, and runs a small `perch_v2_cpu` smoke test.

2. Submit the job using the prepared image + venv:

```bash
sbatch drac/scripts/submit_finwhale_perch2_embeddings.sh \
  --context-dataset-tar /project/.../perch2_context_dataset_YYYYMMDDTHHMMSSZ.tar.zst \
  --container-image "$SCRATCH/whale-call-analysis/containers/tensorflow_2.20.0_gpu.sif" \
  --container-venv-path "$SCRATCH/whale-call-analysis/venvs/perch2_tf220" \
  --apptainer-module apptainer \
  --perch-model perch_v2_gpu \
  --batch-size 16 \
  --context-seconds 40 \
  --train-clip-seconds 10 \
  --eval-clip-seconds 10 \
  --train-pos-augment-copies 1 \
  --train-neg-augment-copies 1 \
  --center-bias-sigma-frac 0.25 \
  --min-gap-seconds 120 \
  --seed 42 \
  --use-wandb \
  --wandb-project finwhale_perch2
```

The submit script still extracts the dataset archive into `$SLURM_TMPDIR` and
writes run artifacts under `$SCRATCH`; only the Python runtime changes.

Run that once on the login node for your training venv. On Nibi, prefer this
over `--install-perch-deps` inside the SLURM job.

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
  --seed 42 \
  --use-wandb \
  --wandb-project finwhale_perch2 \
  --wandb-group finwhale-perch2-manual
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
- W&B is optional; when enabled, set `WANDB_API_KEY` or create `~/.wandb_api_key` on the cluster.
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
