"""Validates dataset files and formats for MLX compatibility."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class DatasetValidator:
    """Validates MLX dataset files."""

    def __init__(self, format_type: str = "chat"):
        """Initialize validator.

        Args:
            format_type: Expected format type (chat, completion, or text)
        """
        self.format_type = format_type

    def validate_chat_format(self, example: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate chat format example.

        Args:
            example: Example to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if "messages" not in example:
            return False, "Missing 'messages' field"

        if not isinstance(example["messages"], list):
            return False, "'messages' must be a list"

        for i, msg in enumerate(example["messages"]):
            if not isinstance(msg, dict):
                return False, f"Message {i} is not a dictionary"

            if "role" not in msg:
                return False, f"Message {i} missing 'role' field"

            if "content" not in msg:
                return False, f"Message {i} missing 'content' field"

            if msg["role"] not in ["system", "user", "assistant"]:
                return False, f"Message {i} has invalid role: {msg['role']}"

        return True, None

    def validate_completion_format(
        self, example: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate completion format example.

        Args:
            example: Example to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if "prompt" not in example:
            return False, "Missing 'prompt' field"

        if "completion" not in example:
            return False, "Missing 'completion' field"

        return True, None

    def validate_text_format(self, example: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate text format example.

        Args:
            example: Example to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if "text" not in example:
            return False, "Missing 'text' field"

        if not isinstance(example["text"], str):
            return False, "'text' must be a string"

        return True, None

    def validate_example(self, example: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a single example based on format type.

        Args:
            example: Example to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.format_type == "chat":
            return self.validate_chat_format(example)
        elif self.format_type == "completion":
            return self.validate_completion_format(example)
        elif self.format_type == "text":
            return self.validate_text_format(example)
        else:
            return False, f"Unknown format type: {self.format_type}"

    def validate_jsonl_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            Validation report dictionary
        """
        if not file_path.exists():
            return {
                "valid": False,
                "error": f"File not found: {file_path}",
                "total": 0,
                "valid_examples": 0,
                "invalid_examples": 0,
            }

        total = 0
        valid_examples = 0
        invalid_examples = 0
        errors: List[str] = []

        with open(file_path, "r") as f:
            for i, line in enumerate(f, 1):
                total += 1
                try:
                    example = json.loads(line)
                    is_valid, error = self.validate_example(example)

                    if is_valid:
                        valid_examples += 1
                    else:
                        invalid_examples += 1
                        errors.append(f"Line {i}: {error}")

                except json.JSONDecodeError as e:
                    invalid_examples += 1
                    errors.append(f"Line {i}: Invalid JSON - {str(e)}")

        is_valid = invalid_examples == 0

        report = {
            "valid": is_valid,
            "file": str(file_path),
            "total": total,
            "valid_examples": valid_examples,
            "invalid_examples": invalid_examples,
            "errors": errors[:10],  # Limit to first 10 errors
        }

        return report

    def validate_dataset_splits(
        self, train_path: Path, test_path: Path, valid_path: Path
    ) -> Dict[str, Any]:
        """Validate all dataset split files.

        Args:
            train_path: Path to training data
            test_path: Path to test data
            valid_path: Path to validation data

        Returns:
            Combined validation report
        """
        print("Validating dataset splits...")

        train_report = self.validate_jsonl_file(train_path)
        test_report = self.validate_jsonl_file(test_path)
        valid_report = self.validate_jsonl_file(valid_path)

        all_valid = (
            train_report["valid"] and test_report["valid"] and valid_report["valid"]
        )

        report = {
            "all_valid": all_valid,
            "train": train_report,
            "test": test_report,
            "valid": valid_report,
        }

        # Print summary
        self.print_validation_report(report)

        return report

    def print_validation_report(self, report: Dict[str, Any]) -> None:
        """Print validation report.

        Args:
            report: Validation report dictionary
        """
        print("\nValidation Report")
        print("=" * 60)

        for split_name in ["train", "test", "valid"]:
            split_report = report[split_name]
            status = "✓ VALID" if split_report["valid"] else "✗ INVALID"

            print(f"\n{split_name.upper()}: {status}")
            print(f"  Total examples: {split_report['total']}")
            print(f"  Valid: {split_report['valid_examples']}")
            print(f"  Invalid: {split_report['invalid_examples']}")

            if split_report.get("errors"):
                print(f"  Errors (showing first 10):")
                for error in split_report["errors"]:
                    print(f"    - {error}")

        print("\n" + "=" * 60)
        if report["all_valid"]:
            print("✓ All datasets are valid!")
        else:
            print("✗ Some datasets have validation errors")
