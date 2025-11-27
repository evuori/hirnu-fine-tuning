#!/usr/bin/env python3
"""Script for interactive inference with trained Hirnu model."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load, generate
from src.evaluation.evaluator import TranslationEvaluator


def interactive_mode(model, tokenizer):
    """Run interactive inference loop.

    Args:
        model: Loaded model
        tokenizer: Model tokenizer
    """
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  /translate <text>  - Translate English to Hirnu")
    print("  /generate <prompt> - Generate text from prompt")
    print("  /quit or /exit     - Exit interactive mode")
    print("\n" + "=" * 60)

    translator = TranslationEvaluator(model, tokenizer)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input in ["/quit", "/exit"]:
                print("Goodbye!")
                break

            elif user_input.startswith("/translate "):
                text = user_input[11:].strip()
                if text:
                    print("\nTranslating to Hirnu...")
                    translation = translator.translate(
                        text, source_lang="english", target_lang="hirnu"
                    )
                    print(f"Hirnu: {translation}")
                else:
                    print("Please provide text to translate")

            elif user_input.startswith("/generate "):
                prompt = user_input[10:].strip()
                if prompt:
                    print("\nGenerating...")
                    # TODO: Implement generation
                    response = ""
                    print(f"Generated: {response}")
                else:
                    print("Please provide a prompt")

            else:
                # Default to translation
                print("\nTranslating to Hirnu...")
                translation = translator.translate(
                    user_input, source_lang="english", target_lang="hirnu"
                )
                print(f"Hirnu: {translation}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(
        description="Interactive inference with Hirnu model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model or model ID",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to LoRA adapter weights (if using adapters)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for one-time inference",
    )
    parser.add_argument(
        "--translate",
        type=str,
        help="Translate text from English to Hirnu",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Hirnu Model Inference")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    try:
        model, tokenizer = load(args.model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        return 1

    # Load adapter weights if specified
    if args.adapter_path:
        print(f"\nLoading adapter weights from {args.adapter_path}...")
        from src.utils.mlx_helpers import load_adapter_weights

        try:
            model = load_adapter_weights(model, args.adapter_path)
            print("✓ Adapter weights loaded")
        except Exception as e:
            print(f"✗ Failed to load adapters: {str(e)}")
            return 1

    # Single prompt mode
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        # TODO: Implement generation
        response = ""
        print(f"\nResponse: {response}")
        return 0

    # Translation mode
    if args.translate:
        print(f"\nEnglish: {args.translate}")
        print("\nTranslating to Hirnu...")
        translator = TranslationEvaluator(model, tokenizer)
        translation = translator.translate(
            args.translate, source_lang="english", target_lang="hirnu"
        )
        print(f"Hirnu: {translation}")
        return 0

    # Interactive mode
    interactive_mode(model, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
