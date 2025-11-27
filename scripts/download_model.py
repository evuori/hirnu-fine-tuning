#!/usr/bin/env python3
"""Script to download base model from Hugging Face."""

import argparse
from pathlib import Path
import sys
import yaml

from huggingface_hub import snapshot_download


def load_model_from_config():
    """Load model name from training config."""
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("model", {}).get("name", "mlx-community/Llama-3.2-3B-Instruct-4bit")
    except Exception:
        return "mlx-community/Llama-3.2-3B-Instruct-4bit"


def main():
    """Download model from Hugging Face Hub."""
    parser = argparse.ArgumentParser(description="Download base model from Hugging Face")
    parser.add_argument(
        "--model-id",
        type=str,
        default=load_model_from_config(),
        help="Hugging Face model ID (defaults to model from training_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/base",
        help="Directory to save downloaded model",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face authentication token (for private models)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Download Base Model")
    print("=" * 60)
    print(f"\nModel ID: {args.model_id}")
    print(f"Output directory: {args.output_dir}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print("\nDownloading model...")
        snapshot_download(
            repo_id=args.model_id,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            token=args.token,
        )

        print("\n" + "=" * 60)
        print("✓ Model downloaded successfully!")
        print("=" * 60)
        print(f"\nModel saved to: {output_path.absolute()}")
        print("\nYou can now use this model for training:")
        print(f'  Update configs/training_config.yaml with model path: "{output_path}"')

        return 0

    except Exception as e:
        print(f"\n✗ Download failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
