#!/usr/bin/env python3
"""Script to prepare Hirnu training data."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import HirnuPreprocessor
from src.data.converter import MLXFormatConverter
from src.data.dataset_builder import DatasetBuilder
from src.data.validator import DatasetValidator


def main():
    """Main data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare Hirnu data for model training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing datasets without processing",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step",
    )

    args = parser.parse_args()

    # Validate existing datasets if requested
    if args.validate_only:
        print("Validating existing datasets...")
        builder = DatasetBuilder(args.config)
        train_path, test_path, valid_path = builder.get_output_paths()

        # Determine format from config
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        format_type = config["format"]["type"]

        validator = DatasetValidator(format_type)
        report = validator.validate_dataset_splits(train_path, test_path, valid_path)

        if report["all_valid"]:
            print("\n✓ All datasets are valid!")
            return 0
        else:
            print("\n✗ Validation failed. See errors above.")
            return 1

    print("=" * 60)
    print("Hirnu Data Preparation Pipeline")
    print("=" * 60)

    # Step 1: Preprocess raw data
    print("\n[Step 1/5] Preprocessing raw data...")
    preprocessor = HirnuPreprocessor(args.config)
    examples = preprocessor.process_all()
    print(f"  Processed {len(examples)} examples")

    if not examples:
        print("\n✗ No examples found! Please add data to data/raw directories.")
        print("  Expected directories:")
        print("    - data/raw/grammar/")
        print("    - data/raw/vocabulary/")
        print("    - data/raw/texts/")
        return 1

    # Step 2: Convert to MLX format
    print("\n[Step 2/5] Converting to MLX format...")
    converter = MLXFormatConverter(args.config)

    # Step 3: Create train/test/valid splits
    print("\n[Step 3/5] Creating dataset splits...")
    builder = DatasetBuilder(args.config)
    train_examples, test_examples, valid_examples = builder.create_splits(examples)

    # Validate splits
    if not builder.validate_splits(train_examples, test_examples, valid_examples):
        print("\n✗ Dataset splits validation failed!")
        return 1

    # Step 4: Save datasets
    print("\n[Step 4/5] Saving datasets...")
    train_path, test_path, valid_path = builder.get_output_paths()

    converter.convert_and_save(train_examples, train_path)
    converter.convert_and_save(test_examples, test_path)
    converter.convert_and_save(valid_examples, valid_path)

    # Step 5: Validate output datasets
    if not args.skip_validation:
        print("\n[Step 5/5] Validating output datasets...")
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        format_type = config["format"]["type"]

        validator = DatasetValidator(format_type)
        report = validator.validate_dataset_splits(train_path, test_path, valid_path)

        if not report["all_valid"]:
            print("\n✗ Dataset validation failed!")
            return 1
    else:
        print("\n[Step 5/5] Skipping validation...")

    print("\n" + "=" * 60)
    print("✓ Data preparation complete!")
    print("=" * 60)
    print("\nDataset files created:")
    print(f"  Training:   {train_path}")
    print(f"  Test:       {test_path}")
    print(f"  Validation: {valid_path}")
    print("\nYou can now run training with:")
    print("  python scripts/train.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
