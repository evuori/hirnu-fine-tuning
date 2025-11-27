"""
Hirnu Model Training Pipeline

This is the main entry point for the Hirnu project.
For specific tasks, use the scripts in the scripts/ directory.
"""


def main():
    """Display project information and usage instructions."""
    print("=" * 70)
    print("Hirnu Model Training Pipeline")
    print("=" * 70)
    print()
    print("An MLX-based training pipeline for the Hirnu language model.")
    print()
    print("Quick Start:")
    print("  1. Install dependencies:  uv pip install -e .")
    print("  2. Download base model:   python scripts/download_model.py")
    print("  3. Prepare data:          python scripts/prepare_data.py")
    print("  4. Train model:           python scripts/train.py")
    print("  5. Evaluate model:        python scripts/evaluate.py --model-path models/hirnu-finetuned")
    print()
    print("Available Scripts:")
    print("  scripts/download_model.py  - Download base model from Hugging Face")
    print("  scripts/prepare_data.py    - Prepare training data")
    print("  scripts/train.py           - Train the model")
    print("  scripts/evaluate.py        - Evaluate trained model")
    print("  scripts/inference.py       - Interactive inference")
    print()
    print("Documentation:")
    print("  docs/SETUP.md              - Setup and installation guide")
    print("  docs/DATA_PREPARATION.md   - Data preparation guide")
    print("  docs/TRAINING.md           - Training and evaluation guide")
    print("  README.md                  - Project overview")
    print()
    print("Configuration:")
    print("  configs/training_config.yaml - Training configuration")
    print("  configs/data_config.yaml     - Data processing configuration")
    print()
    print("=" * 70)
    print()
    print("For detailed instructions, see README.md and docs/")
    print()


if __name__ == "__main__":
    main()
