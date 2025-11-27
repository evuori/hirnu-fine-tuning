#!/usr/bin/env python3
"""Script to train Hirnu model."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import HirnuTrainingConfig
from src.training.trainer import HirnuTrainer


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Hirnu language model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        default="data/processed/valid.jsonl",
        help="Path to validation data",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test configuration without training",
    )

    args = parser.parse_args()

    # Validate data files exist
    train_path = Path(args.train_data)
    valid_path = Path(args.valid_data)

    if not train_path.exists():
        print(f"✗ Training data not found: {train_path}")
        print("\nPlease run data preparation first:")
        print("  python scripts/prepare_data.py")
        return 1

    if not valid_path.exists():
        print(f"✗ Validation data not found: {valid_path}")
        print("\nPlease run data preparation first:")
        print("  python scripts/prepare_data.py")
        return 1

    print("=" * 60)
    print("Hirnu Model Training")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = HirnuTrainingConfig.from_yaml(args.config)

    # Print configuration summary
    print("\nTraining Configuration:")
    print(f"  Model: {config.model.name}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  LoRA enabled: {config.lora.use_lora}")
    if config.lora.use_lora:
        print(f"    Rank: {config.lora.lora_rank}")
        print(f"    Alpha: {config.lora.lora_alpha}")

    print("\nData:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {valid_path}")

    if args.dry_run:
        print("\n✓ Dry run complete - configuration is valid")
        return 0

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = HirnuTrainer(
        config=config, train_data=str(train_path), valid_data=str(valid_path)
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        # TODO: Implement checkpoint loading

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    try:
        trainer.train()

        print("\n" + "=" * 60)
        print("✓ Training complete!")
        print("=" * 60)
        print(f"\nModel saved to: {config.model.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Checkpoints are saved in:", config.checkpointing.checkpoint_dir)
        return 1

    except Exception as e:
        print(f"\n✗ Training failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
