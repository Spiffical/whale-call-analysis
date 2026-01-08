# Whale Call Analysis

Tools for creating and training CNN classifiers on fin whale call spectrograms from ONC (Ocean Networks Canada) hydrophone data.

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Create Dataset](#1-create-dataset)
  - [2. Train Model](#2-train-model)
  - [3. Evaluate Model](#3-evaluate-model)
- [Configuration](#configuration)
- [DRAC Cluster](#drac-cluster)

## Setup

```bash
# Clone and setup
git clone https://github.com/Spiffical/whale-call-analysis.git
cd whale-call-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure ONC API token
echo "ONC_TOKEN=your_token_here" > .env
```

## Project Structure

```
scripts/
├── create_dataset.py       # Generate spectrogram dataset from annotations
├── train_cnn.py            # Train CNN classifier
└── test_cnn.py             # Evaluate trained models

src/
├── dataset/                # Dataset creation utilities
│   ├── generator.py        # SpectrogramDatasetGenerator class
│   ├── call_catalog.py     # Excel annotation parsing
│   ├── audio.py            # Audio download/stitching
│   ├── spectrogram.py      # Spectrogram generation
│   └── ...
├── training/               # Model training utilities
│   ├── mat_dataset.py      # PyTorch Dataset for MAT files
│   ├── splits.py           # Train/val/test splitting
│   └── ...
└── models/
    └── fin_models.py       # CNN architectures (SmallCNN, ResNet, etc.)

config/
└── dataset_config.yaml     # Spectrogram & training configuration

data/finwhales/             # Excel files with whale call annotations
```

## Usage

### 1. Create Dataset

Generate spectrograms from whale call annotations:

```bash
python scripts/create_dataset.py \
    --excel-file data/finwhales/FinWhale20Hz_CallLibrary.xlsx \
    --output-dir output/dataset \
    --sample-size 100 \
    --generate-negatives
```

Options:
- `--sample-size N` — Process N calls (omit for all)
- `--generate-negatives` — Create negative (no-call) samples
- `--cleanup-audio` — Delete audio after processing
- `--workers N` — Parallel workers (default: 2)

### 2. Train Model

Train a CNN on the generated MAT files:

```bash
python scripts/train_cnn.py \
    --pos-dir output/dataset/mat_files \
    --neg-dir output/dataset/neg_mat_files \
    --epochs 50 \
    --model resnet18 \
    --use_wandb \                              # optional
    --wandb_project finwhale-cnn               # optional
```

Options:
- `--model` — `SmallCNN`, `DeepCNN`, `resnet18`, `resnet34`, `resnet50`
- `--crop-size` — Spectrogram crop (e.g., `96` or `96,200`)
- `--balance` — Class balancing: `none`, `weighted`, `oversample`
- `--use_wandb` — (optional) Log to Weights & Biases
- `--wandb_project`, `--wandb_entity`, `--wandb_group` — (optional) WandB settings

### 3. Evaluate Model

Test trained model(s):

```bash
python scripts/test_cnn.py \
    --pos-dir output/dataset/mat_files \
    --neg-dir output/dataset/neg_mat_files \
    --checkpoints exp/resnet18/best.pt exp/smallcnn/best.pt \
    --labels ResNet18 SmallCNN \
    --out-dir output/test_results \
    --use-wandb \                              # optional
    --wandb-project finwhale-cnn               # optional
```

Options:
- `--checkpoints` — One or more model checkpoint paths
- `--labels` — Labels for each checkpoint (for plots)
- `--use-wandb` — (optional) Log results and plots to WandB
- `--wandb-project`, `--wandb-entity`, `--wandb-group` — (optional) WandB settings

## Configuration

Edit `config/dataset_config.yaml` to customize:

```yaml
custom_spectrograms:
  window_duration: 1.0      # FFT window (seconds)
  overlap: 0.9              # Window overlap
  frequency_limits:
    min: 5                  # Hz (fin whale range)
    max: 100

temporal_context:
  context_duration: 40.0    # Seconds around each call
```

## DRAC Cluster

Submit jobs to Digital Research Alliance of Canada:

```bash
# Training
bash drac/scripts/submit_finwhale_cnn.sh \
    --pos-dir /path/to/pos \
    --neg-dir /path/to/neg \
    --epochs 100 \
    --model resnet18 \
    --use_wandb \                              # optional
    --wandb_project finwhale-cnn               # optional

# Testing
bash drac/scripts/submit_finwhale_test.sh \
    --checkpoints /path/to/model1.pt /path/to/model2.pt \
    --labels Model1 Model2 \
    --out-dir /path/to/results \
    --use-wandb \                              # optional
    --wandb-project finwhale-cnn               # optional
```

**WandB on DRAC**: Since compute nodes can't run interactive login, set your API key:
```bash
# Option 1: Create a file (recommended)
echo "your_wandb_api_key" > ~/.wandb_api_key

# Option 2: Export in your ~/.bashrc
export WANDB_API_KEY=your_wandb_api_key
```
Get your API key from: https://wandb.ai/authorize
