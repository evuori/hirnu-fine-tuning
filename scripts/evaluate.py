#!/usr/bin/env python3
"""Script to evaluate trained Hirnu model."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load
from src.evaluation.evaluator import HirnuEvaluator
from src.utils.mlx_helpers import print_model_info


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Hirnu language model")
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
        "--test-data",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results/evaluation.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Hirnu Model Evaluation")
    print("=" * 60)

    # Validate test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"\n✗ Test data not found: {test_path}")
        return 1

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

    # Print model info
    print_model_info(model, tokenizer)

    # Initialize evaluator
    evaluator = HirnuEvaluator(model, tokenizer)

    # Run evaluation
    print(f"\nEvaluating on {test_path}...")
    print("-" * 60)

    try:
        metrics = evaluator.evaluate_on_dataset(
            eval_data_path=str(test_path), output_path=args.output
        )

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        print(f"\nDetailed results saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\n✗ Evaluation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
