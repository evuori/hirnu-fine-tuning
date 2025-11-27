"""Builds train/test/valid dataset splits for MLX fine-tuning."""

import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml


class DatasetBuilder:
    """Creates train/test/valid splits from processed data."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize dataset builder with configuration.

        Args:
            config_path: Path to data configuration file
        """
        self.config = self._load_config(config_path)
        self.split_ratios = self.config["splits"]
        self.random_seed = self.split_ratios.get("random_seed", 42)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def create_splits(
        self, examples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split examples into train, test, and validation sets.

        Args:
            examples: List of all examples

        Returns:
            Tuple of (train_examples, test_examples, valid_examples)
        """
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        random.shuffle(examples)

        # Calculate split indices
        total = len(examples)
        train_ratio = self.split_ratios["train"]
        test_ratio = self.split_ratios["test"]
        valid_ratio = self.split_ratios["valid"]

        # Validate ratios sum to 1.0
        if not abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + test_ratio + valid_ratio}"
            )

        train_end = int(total * train_ratio)
        test_end = train_end + int(total * test_ratio)

        # Create splits
        train_examples = examples[:train_end]
        test_examples = examples[train_end:test_end]
        valid_examples = examples[test_end:]

        print(f"Dataset splits created:")
        print(f"  Train: {len(train_examples)} examples ({train_ratio*100:.1f}%)")
        print(f"  Test:  {len(test_examples)} examples ({test_ratio*100:.1f}%)")
        print(f"  Valid: {len(valid_examples)} examples ({valid_ratio*100:.1f}%)")

        return train_examples, test_examples, valid_examples

    def validate_splits(
        self,
        train_examples: List[Dict[str, Any]],
        test_examples: List[Dict[str, Any]],
        valid_examples: List[Dict[str, Any]],
    ) -> bool:
        """Validate that dataset splits are reasonable.

        Args:
            train_examples: Training examples
            test_examples: Test examples
            valid_examples: Validation examples

        Returns:
            True if splits are valid
        """
        # Check that we have data in each split
        if not train_examples:
            print("WARNING: No training examples!")
            return False

        if not test_examples:
            print("WARNING: No test examples!")
            return False

        if not valid_examples:
            print("WARNING: No validation examples!")
            return False

        # Check that splits are reasonable sizes
        total = len(train_examples) + len(test_examples) + len(valid_examples)
        if total < 10:
            print(f"WARNING: Very small dataset ({total} examples)")
            return False

        return True

    def get_output_paths(self) -> Tuple[Path, Path, Path]:
        """Get output paths for train, test, and valid datasets.

        Returns:
            Tuple of (train_path, test_path, valid_path)
        """
        processed_dir = Path(self.config["data"]["processed_data_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)

        train_path = processed_dir / "train.jsonl"
        test_path = processed_dir / "test.jsonl"
        valid_path = processed_dir / "valid.jsonl"

        return train_path, test_path, valid_path
