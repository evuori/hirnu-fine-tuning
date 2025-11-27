# Model Training - Hirnu

## Overview

This project is a model training pipeline for the Hirnu dataset. The dataset is a about imaginary language called "Hirnu". Selected model is trained to generate Hirnu language text, answer questions in English and translate between English and Hirnu.

## Tools

Apple MLX Framework
Python libraries: mlx-lm, pandas, huggingface-hub
Model: mlx-community/Llama-3.2-3B-Instruct-4bit

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
python scripts/download_model.py
```

### 3. Prepare Your Data

Add your Hirnu language data to:
- `data/raw/grammar/` - Grammar rules and examples
- `data/raw/vocabulary/` - Word lists and translations
- `data/raw/texts/` - Hirnu texts and stories

Then run data preparation:

```bash
python scripts/prepare_data.py
```

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for detailed instructions.

### 4. Train the Model

```bash
python scripts/train.py
```

See [docs/TRAINING.md](docs/TRAINING.md) for training guide.

### 5. Evaluate and Use

```bash
# Evaluate model
python scripts/evaluate.py --model-path models/hirnu-finetuned

# Interactive inference
python scripts/inference.py --model-path models/hirnu-finetuned
```

## Quick Commands

```bash
# Prepare data
python scripts/prepare_data.py

# Validate existing datasets
python scripts/prepare_data.py --validate-only

# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py --config my_config.yaml

# Test configuration without training
python scripts/train.py --dry-run

# Evaluate model
python scripts/evaluate.py --model-path models/hirnu-finetuned

# Translate text
python scripts/inference.py --model-path models/hirnu-finetuned --translate "Hello"

# Interactive mode
python scripts/inference.py --model-path models/hirnu-finetuned
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

## Training

### Available Data Types

- Grammar
- Vocabulary
- Texts


