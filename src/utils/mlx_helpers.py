"""MLX-specific utility functions."""

from pathlib import Path
from typing import Tuple, Optional
import mlx.core as mx


def count_parameters(model) -> Tuple[int, int]:
    """Count total and trainable parameters in model.

    Args:
        model: MLX model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    # TODO: Implement based on MLX model API
    # This is a placeholder
    total_params = 0
    trainable_params = 0

    return total_params, trainable_params


def get_model_size_mb(model) -> float:
    """Estimate model size in megabytes.

    Args:
        model: MLX model

    Returns:
        Model size in MB
    """
    # TODO: Implement based on MLX model API
    # This is a placeholder
    total_params, _ = count_parameters(model)
    # Rough estimate: 4 bytes per parameter (float32)
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb


def print_model_info(model, tokenizer=None):
    """Print model information.

    Args:
        model: MLX model
        tokenizer: Optional tokenizer
    """
    total_params, trainable_params = count_parameters(model)
    size_mb = get_model_size_mb(model)

    print("\nModel Information")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated size: {size_mb:.2f} MB")

    if tokenizer:
        print(f"Vocabulary size: {len(tokenizer)}")

    print("=" * 60)


def save_adapter_weights(
    model, output_path: str, metadata: Optional[dict] = None
):
    """Save LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        output_path: Path to save weights
        metadata: Optional metadata to save with weights
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # TODO: Implement based on MLX-LM API
    # This is a placeholder
    print(f"Adapter weights saved to {output_path}")


def load_adapter_weights(model, adapter_path: str):
    """Load LoRA adapter weights into model.

    Args:
        model: Base model
        adapter_path: Path to adapter weights

    Returns:
        Model with loaded adapters
    """
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_path}")

    # TODO: Implement based on MLX-LM API
    # This is a placeholder
    print(f"Adapter weights loaded from {adapter_path}")

    return model


def merge_lora_adapters(model):
    """Merge LoRA adapters into base model weights.

    Args:
        model: Model with LoRA adapters

    Returns:
        Model with merged weights
    """
    # TODO: Implement based on MLX-LM API
    # This is a placeholder
    print("LoRA adapters merged into base model")

    return model


def estimate_memory_usage(
    model, batch_size: int, seq_length: int
) -> dict:
    """Estimate memory usage for training.

    Args:
        model: MLX model
        batch_size: Training batch size
        seq_length: Sequence length

    Returns:
        Dictionary with memory estimates
    """
    model_size = get_model_size_mb(model)

    # Rough estimates (these are approximate)
    # Activations: batch_size * seq_length * hidden_dim * 4 bytes
    # Gradients: ~2x model size for optimizer states
    # Temporary buffers: ~1.5x model size

    activations_mb = batch_size * seq_length * 0.001  # Very rough estimate
    gradients_mb = model_size * 2
    buffers_mb = model_size * 1.5

    total_mb = model_size + activations_mb + gradients_mb + buffers_mb

    return {
        "model_mb": model_size,
        "activations_mb": activations_mb,
        "gradients_mb": gradients_mb,
        "buffers_mb": buffers_mb,
        "total_mb": total_mb,
        "total_gb": total_mb / 1024,
    }
