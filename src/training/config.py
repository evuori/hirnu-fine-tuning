"""Training configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    output_dir: str = "models/hirnu-finetuned"


@dataclass
class LoRAConfig:
    """LoRA/PEFT configuration."""

    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_seq_length: int = 2048
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    save_steps: int = 500
    checkpoint_dir: str = "models/checkpoints"
    keep_last_n: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_level: str = "INFO"
    log_dir: str = "outputs/logs"
    log_steps: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    eval_steps: int = 500
    eval_batch_size: int = 4


@dataclass
class HirnuTrainingConfig:
    """Complete training configuration for Hirnu model."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "HirnuTrainingConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            HirnuTrainingConfig instance
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract training config
        training_dict = config_dict.get("training", {})

        # Separate LoRA settings from training settings
        lora_keys = {"use_lora", "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules"}
        lora_dict = {k: v for k, v in training_dict.items() if k in lora_keys}
        training_only_dict = {k: v for k, v in training_dict.items() if k not in lora_keys}

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**training_only_dict),
            lora=LoRAConfig(**lora_dict),
            checkpointing=CheckpointConfig(**config_dict.get("checkpointing", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "lora": self.lora.__dict__,
            "checkpointing": self.checkpointing.__dict__,
            "logging": self.logging.__dict__,
            "evaluation": self.evaluation.__dict__,
        }

    def save(self, output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            output_path: Path to output YAML file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

        print(f"Configuration saved to {output_path}")
