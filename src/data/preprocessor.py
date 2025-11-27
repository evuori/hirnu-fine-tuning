"""Data preprocessing module for Hirnu dataset."""

import re
from pathlib import Path
from typing import Dict, List, Any
import yaml


class HirnuPreprocessor:
    """Preprocesses raw Hirnu data for model training."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize preprocessor with configuration.

        Args:
            config_path: Path to data configuration file
        """
        self.config = self._load_config(config_path)
        self.preprocessing_opts = self.config.get("preprocessing", {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        if not self.preprocessing_opts.get("normalize_whitespace", True):
            return text
        return re.sub(r"\s+", " ", text).strip()

    def validate_length(self, text: str) -> bool:
        """Check if text length is within acceptable range.

        Args:
            text: Input text

        Returns:
            True if text length is valid
        """
        min_length = self.preprocessing_opts.get("min_text_length", 10)
        max_length = self.preprocessing_opts.get("max_text_length", 4096)
        return min_length <= len(text) <= max_length

    def preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps to text.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        # Normalize whitespace
        text = self.normalize_whitespace(text)

        # Add more preprocessing steps here as needed
        # - Lowercase (if configured)
        # - Remove special characters (if configured)
        # - etc.

        return text

    def preprocess_grammar_data(self, grammar_dir: Path) -> List[Dict[str, str]]:
        """Process grammar data into training examples.

        Args:
            grammar_dir: Directory containing grammar files

        Returns:
            List of processed grammar examples
        """
        examples = []
        system_msg = self.config["format"]["chat_template"]["system"]

        for grammar_file in grammar_dir.glob("*.txt"):
            try:
                content = grammar_file.read_text(encoding="utf-8")
                lines = content.strip().split("\n")

                i = 0
                while i < len(lines):
                    line = lines[i].strip()

                    # Parse Q: and A: format
                    if line.startswith("Q:"):
                        question = line[2:].strip()
                        i += 1

                        # Look for the answer on the next line
                        if i < len(lines) and lines[i].strip().startswith("A:"):
                            answer = lines[i].strip()[2:].strip()

                            # Create chat format example
                            if question and answer and self.validate_length(question + answer):
                                examples.append({
                                    "messages": [
                                        {"role": "system", "content": system_msg},
                                        {"role": "user", "content": question},
                                        {"role": "assistant", "content": answer}
                                    ]
                                })
                    i += 1

            except Exception as e:
                print(f"Warning: Error processing {grammar_file}: {e}")

        return examples

    def preprocess_vocabulary_data(self, vocab_dir: Path) -> List[Dict[str, str]]:
        """Process vocabulary data into training examples.

        Args:
            vocab_dir: Directory containing vocabulary files

        Returns:
            List of processed vocabulary examples
        """
        examples = []
        system_msg = self.config["format"]["chat_template"]["system"]

        for vocab_file in vocab_dir.glob("*.txt"):
            try:
                content = vocab_file.read_text(encoding="utf-8")
                lines = content.strip().split("\n")

                i = 0
                while i < len(lines):
                    line = lines[i].strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        i += 1
                        continue

                    # Parse EN: HI: format
                    if line.startswith("EN:"):
                        english = line[3:].strip()
                        i += 1

                        # Look for Hirnu translation on the next line
                        if i < len(lines) and lines[i].strip().startswith("HI:"):
                            hirnu = lines[i].strip()[3:].strip()

                            if english and hirnu:
                                # Create both directions of translation
                                # English to Hirnu
                                examples.append({
                                    "messages": [
                                        {"role": "system", "content": system_msg},
                                        {"role": "user", "content": f"Translate to Hirnu: {english}"},
                                        {"role": "assistant", "content": hirnu}
                                    ]
                                })

                                # Hirnu to English
                                examples.append({
                                    "messages": [
                                        {"role": "system", "content": system_msg},
                                        {"role": "user", "content": f"Translate to English: {hirnu}"},
                                        {"role": "assistant", "content": english}
                                    ]
                                })
                    i += 1

            except Exception as e:
                print(f"Warning: Error processing {vocab_file}: {e}")

        return examples

    def preprocess_text_data(self, texts_dir: Path) -> List[Dict[str, str]]:
        """Process text data into training examples.

        Args:
            texts_dir: Directory containing text files

        Returns:
            List of processed text examples
        """
        examples = []
        system_msg = self.config["format"]["chat_template"]["system"]

        for text_file in texts_dir.glob("*.txt"):
            try:
                content = text_file.read_text(encoding="utf-8")
                lines = content.strip().split("\n")

                i = 0
                while i < len(lines):
                    line = lines[i].strip()

                    # Skip empty lines, comments, and separator lines
                    if not line or line.startswith("#") or line.startswith("---"):
                        i += 1
                        continue

                    # Parse HI: EN: parallel text format
                    if line.startswith("HI:"):
                        hirnu_text = line[3:].strip()
                        i += 1

                        # Look for English translation on the next line
                        if i < len(lines) and lines[i].strip().startswith("EN:"):
                            english_text = lines[i].strip()[3:].strip()

                            if hirnu_text and english_text:
                                # Create both directions of translation
                                # English to Hirnu
                                examples.append({
                                    "messages": [
                                        {"role": "system", "content": system_msg},
                                        {"role": "user", "content": f"Translate to Hirnu: {english_text}"},
                                        {"role": "assistant", "content": hirnu_text}
                                    ]
                                })

                                # Hirnu to English
                                examples.append({
                                    "messages": [
                                        {"role": "system", "content": system_msg},
                                        {"role": "user", "content": f"Translate to English: {hirnu_text}"},
                                        {"role": "assistant", "content": english_text}
                                    ]
                                })

                                # Also create examples for questions if the text contains quotes
                                if '"' in hirnu_text and '"' in english_text:
                                    examples.append({
                                        "messages": [
                                            {"role": "system", "content": system_msg},
                                            {"role": "user", "content": f"What does this mean in English: {hirnu_text}"},
                                            {"role": "assistant", "content": english_text}
                                        ]
                                    })
                    i += 1

            except Exception as e:
                print(f"Warning: Error processing {text_file}: {e}")

        return examples

    def process_all(self) -> List[Dict[str, Any]]:
        """Process all data sources.

        Returns:
            Combined list of all processed examples
        """
        all_examples = []

        sources = self.config["data"]["sources"]
        grammar_dir = Path(sources["grammar"])
        vocab_dir = Path(sources["vocabulary"])
        texts_dir = Path(sources["texts"])

        # Process each data source
        if grammar_dir.exists():
            all_examples.extend(self.preprocess_grammar_data(grammar_dir))

        if vocab_dir.exists():
            all_examples.extend(self.preprocess_vocabulary_data(vocab_dir))

        if texts_dir.exists():
            all_examples.extend(self.preprocess_text_data(texts_dir))

        return all_examples
