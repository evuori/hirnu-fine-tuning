"""Converts preprocessed data to MLX-compatible formats."""

import json
from pathlib import Path
from typing import List, Dict, Any
import yaml


class MLXFormatConverter:
    """Converts data to MLX-compatible JSONL format."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize converter with configuration.

        Args:
            config_path: Path to data configuration file
        """
        self.config = self._load_config(config_path)
        self.format_type = self.config["format"]["type"]
        self.chat_template = self.config["format"].get("chat_template", {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def to_chat_format(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert example to chat format.

        Chat format structure for MLX:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

        Args:
            example: Preprocessed example

        Returns:
            Example in chat format
        """
        # If the example already has messages in the correct format, return as-is
        if "messages" in example:
            return example

        # Otherwise, construct the chat format
        system_msg = self.chat_template.get("system", "")
        messages = [{"role": "system", "content": system_msg}]

        # Add user and assistant messages if present
        if "user" in example and "assistant" in example:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})

        return {"messages": messages}

    def to_completion_format(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert example to completion format.

        Completion format structure for MLX:
        {
            "prompt": "...",
            "completion": "..."
        }

        Args:
            example: Preprocessed example

        Returns:
            Example in completion format
        """
        # TODO: Implement conversion based on your data structure
        return {"prompt": "", "completion": ""}

    def to_text_format(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert example to simple text format.

        Text format structure for MLX:
        {
            "text": "..."
        }

        Args:
            example: Preprocessed example

        Returns:
            Example in text format
        """
        # TODO: Implement conversion based on your data structure
        return {"text": example.get("content", "")}

    def convert_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single example to the configured MLX format.

        Args:
            example: Preprocessed example

        Returns:
            Example in MLX format
        """
        if self.format_type == "chat":
            return self.to_chat_format(example)
        elif self.format_type == "completion":
            return self.to_completion_format(example)
        elif self.format_type == "text":
            return self.to_text_format(example)
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")

    def convert_and_save(
        self, examples: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """Convert examples and save to JSONL file.

        Args:
            examples: List of preprocessed examples
            output_path: Path to output JSONL file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in examples:
                converted = self.convert_example(example)
                f.write(json.dumps(converted) + "\n")

        print(f"Saved {len(examples)} examples to {output_path}")
