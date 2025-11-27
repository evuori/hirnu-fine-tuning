# Model Training - Hirnu

## Overview

This project is a model training pipeline for the Hirnu dataset. The dataset is a about imaginary language called "Hirnu". Selected model is trained to generate Hirnu language text, answer questions in English and translate between English and Hirnu.

## Tools

Apple MLX Framework
Python libraries: mlx-lm, pandas, huggingface-hub, pyyaml
Model: mlx-community/Ministral-8B-Instruct-2410-4bit (configurable in training_config.yaml)

Note: MLX requires data to be in specific formats. Three main formats are discussed within MLX: chat, completion, and text.

Note: The MLX requires three sets of datasets: train, test, and valid for fine-tuning. The data files should be in JSONL format.

## Hirnu

### History

Hirnu is ancient scandinavian language with roots in ancient germanic languages. It is based on old Norse language. 

### Language

Hirnu is a simple, poetic and natural language that offers opportunities for a wide range of modern applications. Its simple structure and vocabulary make it easy to learn, but at the same time it contains a phonetic depth and beauty that inspires Hirnun speakers.

The foundation of Hirnu is clear, and it can be developed in many ways, for example by adding cultural aspects to it, by utilizing technology in language development, and by expanding its artistic expressive power. Whether it is telling a story, writing a poem, or creating a game world, Hirnu offers a rich soil for expressing emotions and thoughts.

The beauty of Hirnu lies in the simplicity of its structure and the possibility of describing the world phonetically and rhythmically. Hirnu is not only a means of communication, but also a way to describe the ancient connection between man and nature. Hirnu inspires creation and exploration, and encourages finding new ways to express contemporary reality.

## Project Structure

```
hirnu/
├── configs/              # Configuration files
├── data/                 # Dataset storage
│   ├── raw/             # Original Hirnu data (grammar, vocabulary, texts)
│   ├── processed/       # Processed JSONL datasets
│   └── samples/         # Sample data for testing
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── training/        # Training modules
│   ├── evaluation/      # Evaluation modules
│   └── utils/           # Utility functions
├── scripts/             # Executable scripts
├── models/              # Saved models
├── outputs/             # Training outputs
└── docs/                # Documentation
```

## Getting Started

### 1. Setup

Install dependencies using UV:

```bash
# Install main dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

See [docs/SETUP.md](docs/SETUP.md) for detailed setup instructions.

### 2. Download Base Model

```bash
uv run python scripts/download_model.py
```

### 3. Prepare Your Data

Add your Hirnu language data to:
- `data/raw/grammar/` - Grammar rules in Q&A format
- `data/raw/vocabulary/` - Word translations in EN:/HI: format
- `data/raw/texts/` - Parallel texts in HI:/EN: format

Then run data preparation:

```bash
uv run python scripts/prepare_data.py
```

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for detailed instructions.

### 4. Train the Model

```bash
uv run python scripts/train.py
```

See [docs/TRAINING.md](docs/TRAINING.md) for training guide.

### 5. Evaluate and Use

```bash
# Evaluate model
uv run python scripts/evaluate.py --model-path models/hirnu-finetuned

# Interactive inference
uv run python scripts/inference.py --model-path models/hirnu-finetuned
```

### 6. Fuse Adapters (Optional)

For faster inference and simpler deployment, fuse the LoRA adapters into a standalone model:

```bash
# Basic fusion (keeps quantization)
uv run python scripts/fuse.py --adapter-path models/hirnu-finetuned --output models/hirnu-fused

# De-quantize and fuse (full precision)
uv run python scripts/fuse.py --adapter-path models/hirnu-finetuned --output models/hirnu-fused --de-quantize

# Export to GGUF format (for Ollama, llama.cpp, etc.)
# Note: May encounter array format issues with some models
uv run python scripts/fuse.py --adapter-path models/hirnu-finetuned --output models/hirnu-fused --de-quantize --export-gguf

# Alternative: Use mlx_lm CLI directly for GGUF export
mlx_lm.fuse --model mlx-community/Ministral-8B-Instruct-2410-4bit \
  --adapter-path models/hirnu-finetuned \
  --save-path models/hirnu-fused-gguf \
  --de-quantize --export-gguf
```

This creates a single model that doesn't require loading adapters separately.

**Note on GGUF Export:** Due to MLX's array format requirements, GGUF export may fail with "row-major arrays" error. If this occurs, use the `mlx_lm.fuse` CLI tool directly (shown above) or convert the fused model using llama.cpp's conversion tools.

## Quick Commands

**Important:** Always use `uv run python` to ensure correct environment and dependencies.

```bash
# Prepare data
uv run python scripts/prepare_data.py

# Validate existing datasets
uv run python scripts/prepare_data.py --validate-only

# Test configuration without training (dry run - safe, no GPU required)
uv run python scripts/train.py --dry-run

# Train with default config
uv run python scripts/train.py

# Train with custom config
uv run python scripts/train.py --config my_config.yaml

# Evaluate model
uv run python scripts/evaluate.py --model-path models/hirnu-finetuned

# Translate text
uv run python scripts/inference.py --model-path models/hirnu-finetuned --translate "Hello"

# Interactive mode
uv run python scripts/inference.py --model-path models/hirnu-finetuned

# Fuse LoRA adapters into standalone model
uv run python scripts/fuse.py --adapter-path models/hirnu-finetuned --output models/hirnu-fused

# Use fused model (faster inference)
uv run python scripts/inference.py --model-path models/hirnu-fused
```

## Configuration

- **Training Config**: [configs/training_config.yaml](configs/training_config.yaml)
  - Model settings, hyperparameters, LoRA config
- **Data Config**: [configs/data_config.yaml](configs/data_config.yaml)
  - Data paths, format settings, preprocessing options

## Documentation

- [Setup Guide](docs/SETUP.md) - Installation and environment setup
- [Data Preparation](docs/DATA_PREPARATION.md) - How to prepare training data
- [Training Guide](docs/TRAINING.md) - Training, evaluation, and inference

## Data Format

The pipeline processes three types of raw data and automatically converts them into MLX-compatible chat format:

### Grammar Files (`data/raw/grammar/*.txt`)
Q&A format for grammar rules:
```
Q: What is the basic form of nouns in Hirnu?
A: Nouns are simple in their basic form. They do not have inflections for case or gender in the singular.
```

### Vocabulary Files (`data/raw/vocabulary/*.txt`)
EN:/HI: format for translations:
```
EN: where / in / at
HI: var

EN: now
HI: nu
```

### Text Files (`data/raw/texts/*.txt`)
Parallel HI:/EN: format for stories and examples:
```
HI: Barn lugnir himrin var vono.
EN: A child looks at the sky in the night.
```

The preprocessing creates bidirectional translation examples (EN→HI and HI→EN) automatically.


