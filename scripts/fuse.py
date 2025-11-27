#!/usr/bin/env python3
"""Script to fuse LoRA adapters with base model.

This script takes trained LoRA adapters and merges them into the base model,
creating a standalone fused model that can be used without loading adapters separately.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_lm import load
from mlx_lm.utils import save
from mlx_lm.fuse import dequantize, convert_to_gguf
from mlx.utils import tree_flatten, tree_unflatten


def fuse_lora_adapters(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    de_quantize: bool = False,
    export_gguf: bool = False
) -> None:
    """Fuse LoRA adapters with base model.

    Args:
        base_model_path: Path to base model or model ID
        adapter_path: Path to directory containing adapters.safetensors
        output_path: Where to save the fused model
        de_quantize: Whether to de-quantize the model
        export_gguf: Whether to export to GGUF format
    """
    print("=" * 60)
    print("LoRA Adapter Fusion")
    print("=" * 60)

    print(f"\nBase model: {base_model_path}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    print()

    # Validate paths
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        print(f"✗ Error: Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    adapter_file = adapter_dir / "adapters.safetensors"
    if not adapter_file.exists():
        print(f"✗ Error: No adapters.safetensors found in {adapter_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model with adapters...")
    try:
        model, tokenizer, config = load(
            base_model_path,
            adapter_path=str(adapter_dir),
            return_config=True,
            tokenizer_config={"fix_mistral_regex": True}
        )
        print("✓ Model and adapters loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nFusing adapters into base model...")
    try:
        # Fuse all modules that have a fuse method (LoRA layers)
        fused_linears = [
            (n, m.fuse(de_quantize=de_quantize))
            for n, m in model.named_modules()
            if hasattr(m, "fuse")
        ]

        if fused_linears:
            model.update_modules(tree_unflatten(fused_linears))
            print(f"✓ Fused {len(fused_linears)} LoRA layers")
        else:
            print("⚠ Warning: No LoRA layers found to fuse")

        if de_quantize:
            print("De-quantizing model...")
            model = dequantize(model)
            config.pop("quantization", None)
            print("✓ Model de-quantized")

    except Exception as e:
        print(f"✗ Failed to fuse adapters: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nSaving fused model to {output_dir}...")
    try:
        save(
            output_dir,
            base_model_path,
            model,
            tokenizer,
            config,
            donate_model=False,
        )
        print("✓ Fused model saved successfully")

        if export_gguf:
            print("\nExporting to GGUF format...")

            # Check if model is quantized
            if config.get("quantization") and not de_quantize:
                print("✗ Error: Cannot export quantized models to GGUF")
                print("  Solution: Use --de-quantize flag to de-quantize before exporting")
                print("  Example: uv run python scripts/fuse.py --adapter-path models/hirnu-finetuned --output models/hirnu-fused --de-quantize --export-gguf")
                sys.exit(1)

            model_type = config["model_type"]
            if model_type not in ["llama", "mixtral", "mistral"]:
                raise ValueError(
                    f"Model type {model_type} not supported for GGUF conversion."
                )

            try:
                import mlx.core as mx

                # Get weights
                print("Preparing weights for GGUF conversion...")
                weights = dict(tree_flatten(model.parameters()))

                gguf_path = output_dir / "ggml-model-f16.gguf"
                convert_to_gguf(output_dir, weights, config, str(gguf_path))
                print(f"✓ GGUF model saved to {gguf_path}")

            except ValueError as e:
                if "row-major arrays" in str(e):
                    print("\n⚠ GGUF export failed due to array format issue")
                    print("  This is a known limitation with MLX's GGUF export")
                    print("\n  Alternative approach:")
                    print("  1. Use the mlx_lm.fuse CLI tool directly:")
                    print(f"     mlx_lm.fuse --model {base_model_path} \\")
                    print(f"       --adapter-path {adapter_path} \\")
                    print(f"       --save-path {output_path}-gguf \\")
                    print("       --de-quantize --export-gguf")
                    print("\n  2. Or use llama.cpp's convert.py to convert the fused model:")
                    print(f"     python llama.cpp/convert.py {output_dir}")
                    print("\n  The fused MLX model was saved successfully and can be used with MLX.")
                else:
                    raise

        print("\n" + "=" * 60)
        print("Fusion Complete!")
        print("=" * 60)
        print(f"\nFused model saved to: {output_dir}")
        print("\nYou can now use this model without loading adapters:")
        print(f"  uv run python scripts/inference.py --model-path {output_path}")

    except Exception as e:
        print(f"✗ Failed to save fused model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main fusion script."""
    parser = argparse.ArgumentParser(
        description="Fuse LoRA adapters with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fuse adapters from default output directory
  python scripts/fuse.py --adapter-path models/hirnu-finetuned --output models/hirnu-fused

  # Specify custom base model
  python scripts/fuse.py \\
    --base-model mlx-community/Ministral-8B-Instruct-2410-4bit \\
    --adapter-path models/hirnu-finetuned \\
    --output models/hirnu-fused
        """
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="mlx-community/Ministral-8B-Instruct-2410-4bit",
        help="Base model path or HuggingFace model ID (default: mlx-community/Ministral-8B-Instruct-2410-4bit)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to directory containing adapters.safetensors",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for fused model",
    )
    parser.add_argument(
        "--de-quantize",
        action="store_true",
        help="De-quantize the model during fusion (required for GGUF export)",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export model to GGUF format (requires --de-quantize for quantized models)",
    )

    args = parser.parse_args()

    fuse_lora_adapters(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output,
        de_quantize=args.de_quantize,
        export_gguf=args.export_gguf
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
