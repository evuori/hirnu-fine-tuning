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

    # Check if model_path is actually an adapter directory (missing config.json)
    model_path = args.model_path
    adapter_path = args.adapter_path
    
    path_obj = Path(model_path)
    if path_obj.is_dir() and not (path_obj / "config.json").exists():
        if (path_obj / "adapters.safetensors").exists():
            print(f"\nNote: '{model_path}' appears to be an adapter directory (no config.json).")
            if Path("models/base").exists():
                print("Using 'models/base' as base model.")
                adapter_path = model_path
                model_path = "models/base"
            else:
                print("Warning: Could not find 'models/base'. Please specify base model path.")

    # Load model
    print(f"\nLoading model from {model_path}...")
    if adapter_path:
        print(f"Loading with adapters from {adapter_path}...")

    try:
        if adapter_path:
            model, tokenizer = load(
                model_path, 
                adapter_path=adapter_path,
                tokenizer_config={"fix_mistral_regex": True}
            )
        else:
            model, tokenizer = load(
                model_path,
                tokenizer_config={"fix_mistral_regex": True}
            )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
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
