"""Main training module for Hirnu model using MLX."""

from pathlib import Path
from typing import Optional
import logging

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.lora import run as mlx_lora_run
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm import models

from src.training.config import HirnuTrainingConfig
from src.utils.logging_utils import setup_logging


class HirnuTrainer:
    """Trainer for fine-tuning Hirnu language model."""

    def __init__(
        self, config: HirnuTrainingConfig, train_data: str, valid_data: str
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            train_data: Path to training data (JSONL)
            valid_data: Path to validation data (JSONL)
        """
        self.config = config
        self.train_data = train_data
        self.valid_data = valid_data
        self.logger = setup_logging(
            config.logging.log_level, config.logging.log_dir
        )

    def prepare_model(self):
        """Load and prepare the base model for fine-tuning.

        Returns:
            Loaded model and tokenizer
        """
        self.logger.info(f"Loading base model: {self.config.model.name}")

        try:
            model, tokenizer = load(
                self.config.model.name,
                tokenizer_config={"fix_mistral_regex": True}
            )
            self.logger.info("Model loaded successfully")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def setup_lora(self, model):
        """Setup LoRA layers for efficient fine-tuning.

        Args:
            model: Base model

        Returns:
            Model with LoRA layers
        """
        if not self.config.lora.use_lora:
            self.logger.info("LoRA disabled, using full fine-tuning")
            return model

        self.logger.info("Setting up LoRA layers")
        self.logger.info(f"  Rank: {self.config.lora.lora_rank}")
        self.logger.info(f"  Alpha: {self.config.lora.lora_alpha}")
        self.logger.info(f"  Dropout: {self.config.lora.lora_dropout}")

        # Apply LoRA to specified layers
        # Build LoRA config dictionary
        lora_config = {
            "rank": self.config.lora.lora_rank,
            "alpha": self.config.lora.lora_alpha,
            "dropout": self.config.lora.lora_dropout,
            "scale": self.config.lora.lora_alpha / self.config.lora.lora_rank,
        }

        linear_to_lora_layers(
            model,
            num_layers=16,  # Number of layers to apply LoRA to
            config=lora_config
        )

        return model

    def train(self):
        """Execute the training loop.

        Returns:
            Training results/metrics
        """
        self.logger.info("Starting Hirnu model training")
        self.logger.info(f"Training data: {self.train_data}")
        self.logger.info(f"Validation data: {self.valid_data}")

        # Create checkpoint directory
        Path(self.config.checkpointing.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        adapter_path = Path(self.config.checkpointing.checkpoint_dir)

        # Build args object for mlx_lm.lora.run
        class TrainingArgs:
            pass

        args = TrainingArgs()
        args.model = self.config.model.name
        args.train = True
        args.data = str(Path(self.train_data).parent)  # Directory containing train.jsonl and valid.jsonl
        args.batch_size = self.config.training.batch_size
        args.iters = self.calculate_total_iterations()
        args.learning_rate = self.config.training.learning_rate
        args.steps_per_report = self.config.logging.log_steps
        args.steps_per_eval = self.config.evaluation.eval_steps
        args.save_every = self.config.checkpointing.save_steps
        args.adapter_path = str(adapter_path)
        args.max_seq_length = self.config.training.max_seq_length
        args.grad_accumulation_steps = self.config.training.gradient_accumulation_steps
        args.num_layers = 16
        args.seed = 42
        args.val_batches = 25
        args.grad_checkpoint = False

        # Additional required fields
        args.report_to = None
        args.project_name = None
        args.test = False
        args.test_batches = 500
        args.resume_adapter_file = None
        args.lr_schedule = None
        args.mask_prompt = False
        args.fine_tune_type = "lora"
        args.optimizer = "adamw"
        args.optimizer_config = {}

        # LoRA config
        args.lora_parameters = {
            "rank": self.config.lora.lora_rank,
            "dropout": self.config.lora.lora_dropout,
            "scale": float(self.config.lora.lora_alpha),
        }

        self.logger.info(f"Starting training for {args.iters} iterations...")
        self.logger.info(f"LoRA config: rank={args.lora_parameters['rank']}, alpha={self.config.lora.lora_alpha}")

        # Run training using MLX-LM's high-level interface
        mlx_lora_run(args)

        self.logger.info("Training completed successfully")

        # The adapter file will be saved by MLX-LM to adapter_path/adapters.safetensors
        adapter_file = adapter_path / "adapters.safetensors"

        # Copy to final output directory
        output_dir = Path(self.config.model.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if adapter_file.exists():
            import shutil
            shutil.copy(adapter_file, output_dir / "adapters.safetensors")
            self.logger.info(f"Copied adapters to {output_dir / 'adapters.safetensors'}")

        adapter_config_file = adapter_path / "adapter_config.json"
        if adapter_config_file.exists():
            import shutil
            shutil.copy(adapter_config_file, output_dir / "adapter_config.json")
            self.logger.info(f"Copied adapter config to {output_dir / 'adapter_config.json'}")

        return str(adapter_file)

    def calculate_total_iterations(self) -> int:
        """Calculate total training iterations.

        Returns:
            Total number of iterations
        """
        import json

        # Count training examples
        try:
            with open(self.train_data, 'r') as f:
                num_examples = sum(1 for _ in f)
        except Exception:
            # Fallback to a reasonable default
            num_examples = 220  # Based on our dataset

        # Calculate iterations per epoch
        iters_per_epoch = max(1, num_examples // self.config.training.batch_size)

        # Total iterations = iterations per epoch * number of epochs
        total_iters = iters_per_epoch * self.config.training.num_epochs

        self.logger.info(f"Calculated {total_iters} iterations ({num_examples} examples, {iters_per_epoch} iters/epoch, {self.config.training.num_epochs} epochs)")

        return total_iters

    def save_model(self, model, tokenizer, adapter_file: str):
        """Save trained model and tokenizer.

        Args:
            model: Trained model
            tokenizer: Tokenizer
            adapter_file: Path to adapter weights file
        """
        from mlx_lm import utils
        import shutil

        output_dir = Path(self.config.model.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving model to {output_dir}")

        # Save tokenizer
        tokenizer.save_pretrained(str(output_dir))

        # Save model weights (if using LoRA, this saves the adapters)
        if self.config.lora.use_lora:
            # Copy adapter file to output directory
            adapter_path = Path(adapter_file)
            if adapter_path.exists():
                shutil.copy(adapter_path, output_dir / "adapters.safetensors")
                self.logger.info(f"Saved LoRA adapters to {output_dir / 'adapters.safetensors'}")
        else:
            # Save full model weights
            mx.save_safetensors(str(output_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))

        # Save config
        utils.upload_to_hub.__globals__['save_config'](
            config=model.config if hasattr(model, 'config') else {},
            config_path=str(output_dir)
        )

        self.logger.info("Model saved successfully")

    def evaluate(self, model, tokenizer, eval_data: str):
        """Evaluate model on evaluation dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            eval_data: Path to evaluation data

        Returns:
            Evaluation metrics
        """
        self.logger.info(f"Evaluating model on {eval_data}")

        # TODO: Implement evaluation based on MLX-LM API
        # This is a placeholder

        metrics = {}

        self.logger.info(f"Evaluation results: {metrics}")
        return metrics
