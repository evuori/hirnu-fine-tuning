# Hirnu Model Training - Setup Guide

This guide will help you set up the Hirnu model training environment.

## Prerequisites

- macOS with Apple Silicon (for MLX support)
- Python 3.13+
- UV package manager
- Git

## Installation

### 1. Clone the Repository

```bash
cd /path/to/hirnu
```

### 2. Install Dependencies

Using UV (recommended):

```bash
# Install main dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

Alternatively, using pip:

```bash
pip install -e .
pip install -e ".[dev]"
```

### 3. Verify Installation

Check that MLX is installed correctly:

```bash
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
```

Check MLX-LM:

```bash
python -c "import mlx_lm; print('MLX-LM installed successfully')"
```

## Project Structure

```
hirnu/
├── configs/              # Configuration files
│   ├── training_config.yaml
│   └── data_config.yaml
├── data/                 # Dataset storage
│   ├── raw/             # Original Hirnu data
│   ├── processed/       # Processed JSONL datasets
│   └── samples/         # Sample data
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── training/        # Training modules
│   ├── evaluation/      # Evaluation modules
│   └── utils/           # Utility functions
├── scripts/             # Executable scripts
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── download_model.py
├── models/              # Saved models
├── outputs/             # Training outputs
└── notebooks/           # Jupyter notebooks
```

## Directory Setup

Create necessary directory structure (already created during project setup):

```bash
mkdir -p data/raw/{grammar,vocabulary,texts}
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p outputs/{logs,results}
```

## Configuration

### Training Configuration

Edit `configs/training_config.yaml` to customize:
- Model parameters
- Training hyperparameters
- LoRA settings
- Checkpointing options
- Logging settings

### Data Configuration

Edit `configs/data_config.yaml` to customize:
- Data paths
- Dataset split ratios
- MLX format settings (chat, completion, or text)
- Preprocessing options

## Download Base Model

Download the base Llama model:

```bash
python scripts/download_model.py --model-id mlx-community/Llama-3.2-3B-Instruct-4bit
```

Or specify a custom output directory:

```bash
python scripts/download_model.py \
  --model-id mlx-community/Llama-3.2-3B-Instruct-4bit \
  --output-dir models/base
```

## Verify Setup

Run a dry-run of the training script to verify configuration:

```bash
python scripts/train.py --dry-run
```

## Next Steps

Once setup is complete:

1. **Prepare your data** - See [DATA_PREPARATION.md](DATA_PREPARATION.md)
2. **Train your model** - See [TRAINING.md](TRAINING.md)
3. **Evaluate results** - See [TRAINING.md](TRAINING.md#evaluation)

## Troubleshooting

### MLX Installation Issues

If MLX installation fails:
- Ensure you're on macOS with Apple Silicon
- Update to the latest macOS version
- Try installing from source: `pip install git+https://github.com/ml-explore/mlx.git`

### Import Errors

If you get import errors:
- Make sure you're in the project root directory
- Verify virtual environment is activated
- Run `uv pip install -e .` again

### Memory Issues

If you encounter memory issues:
- Reduce batch size in `configs/training_config.yaml`
- Enable gradient accumulation
- Use smaller sequence length

## Support

For issues or questions:
- Check the documentation in `docs/`
- Review configuration files in `configs/`
- Examine example scripts in `scripts/`
