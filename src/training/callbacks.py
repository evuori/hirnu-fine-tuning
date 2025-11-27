"""Training callbacks for monitoring and checkpointing."""

from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Dict[str, Any] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of each batch."""
        pass


class MetricsLogger(TrainingCallback):
    """Logs training metrics to file."""

    def __init__(self, log_dir: str):
        """Initialize metrics logger.

        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.start_time = None

    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Record training start time."""
        self.start_time = datetime.now()
        self._log_event("train_begin", logs)

    def on_train_end(self, logs: Dict[str, Any] = None):
        """Record training end time and duration."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if logs is None:
                logs = {}
            logs["training_duration_seconds"] = duration
        self._log_event("train_end", logs)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Log epoch metrics."""
        if logs is None:
            logs = {}
        logs["epoch"] = epoch
        self._log_event("epoch_end", logs)

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Log batch metrics."""
        if logs is None:
            logs = {}
        logs["batch"] = batch
        self._log_event("batch_end", logs)

    def _log_event(self, event_type: str, logs: Dict[str, Any] = None):
        """Log an event with timestamp.

        Args:
            event_type: Type of event
            logs: Event data
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": logs or {},
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(event) + "\n")


class CheckpointCallback(TrainingCallback):
    """Saves model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 500,
        keep_last_n: int = 3,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps
            keep_last_n: Keep only last N checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.checkpoints = []

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Save checkpoint if needed.

        Args:
            batch: Batch number
            logs: Batch logs
        """
        if batch % self.save_every == 0:
            self._save_checkpoint(batch, logs)
            self._cleanup_old_checkpoints()

    def _save_checkpoint(self, step: int, logs: Dict[str, Any] = None):
        """Save a checkpoint.

        Args:
            step: Training step
            logs: Training logs
        """
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # TODO: Implement actual model checkpoint saving
        # This is a placeholder

        self.checkpoints.append(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_name}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoints = self.checkpoints[: -self.keep_last_n]
            for checkpoint in old_checkpoints:
                if checkpoint.exists():
                    # TODO: Implement checkpoint deletion
                    pass
            self.checkpoints = self.checkpoints[-self.keep_last_n :]


class EarlyStoppingCallback(TrainingCallback):
    """Stops training early if validation metric doesn't improve."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        metric: str = "loss",
        mode: str = "min",
    ):
        """Initialize early stopping callback.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Check if training should stop.

        Args:
            epoch: Current epoch
            logs: Epoch logs
        """
        if logs is None or self.metric not in logs:
            return

        current_value = logs[self.metric]

        if self._is_improvement(current_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.metric}: {self.best_value}"
                )
                # TODO: Implement actual training stop signal

    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value is an improvement.

        Args:
            current_value: Current metric value

        Returns:
            True if improved
        """
        if self.mode == "min":
            return current_value < (self.best_value - self.min_delta)
        else:
            return current_value > (self.best_value + self.min_delta)
