"""
Training Script for PaddleOCR-VL Fine-tuning
DocuSpend Project - PaddleOCR-VL Fine-tuning

This module provides training utilities for fine-tuning PaddleOCR-VL on the DocuSpend dataset.
Designed to run on Google Colab Free (T4 GPU, 16GB VRAM) with mixed precision training.

Note: Actual training loop to be implemented in notebook 05_fine_tuning.ipynb
This module provides setup and configuration utilities.
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration manager."""

    def __init__(self, config_path: str):
        """
        Load training configuration from YAML file.

        Args:
            config_path: Path to training_config.yaml
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded training config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load training config: {str(e)}")
            return {}

    def get(self, key: str, default=None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)

    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = [
            "training",
            "optimization",
            "hardware",
            "logging",
        ]

        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required config key: {key}")
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config


class TrainingSetup:
    """Training pipeline setup."""

    def __init__(self, config: TrainingConfig, output_dir: str = "outputs"):
        """
        Initialize training setup.

        Args:
            config: TrainingConfig instance
            output_dir: Directory for outputs (logs, checkpoints)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.tensorboard_dir = self.log_dir / "tensorboard"

        self._create_directories()

    def _create_directories(self):
        """Create required directories."""
        for directory in [
            self.checkpoint_dir,
            self.log_dir,
            self.tensorboard_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        log_file = self.log_dir / "training.log"

        # Configure root logger
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

        return logger

    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        info = {
            "device": self.config.get("hardware", {}).get("device", "cpu"),
            "gpu_id": self.config.get("hardware", {}).get("gpu_id", 0),
            "max_vram_gb": self.config.get("hardware", {}).get("max_vram_gb", 16),
        }

        try:
            import torch
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        except ImportError:
            logger.warning("PyTorch not available for device info")

        return info

    def get_training_info(self) -> Dict[str, Any]:
        """Get training configuration summary."""
        training_cfg = self.config.get("training", {})
        opt_cfg = self.config.get("optimization", {})

        return {
            "epochs": training_cfg.get("epochs", 10),
            "batch_size": training_cfg.get("batch_size", 4),
            "gradient_accumulation_steps": training_cfg.get(
                "gradient_accumulation_steps", 4
            ),
            "effective_batch_size": (
                training_cfg.get("batch_size", 4) *
                training_cfg.get("gradient_accumulation_steps", 4)
            ),
            "optimizer": opt_cfg.get("optimizer", "AdamW"),
            "learning_rate": training_cfg.get("learning_rate", {}).get("initial", 1e-4),
            "scheduler": training_cfg.get("learning_rate", {}).get("scheduler", "linear"),
            "mixed_precision": opt_cfg.get("mix_precision", True),
        }

    def verify_setup(self) -> Dict[str, Any]:
        """Verify training setup is correct."""
        checks = {
            "timestamp": datetime.now().isoformat(),
            "config_valid": self.config.validate(),
            "directories_created": True,
            "device_info": self.get_device_info(),
            "training_info": self.get_training_info(),
            "issues": [],
        }

        # Check for potential issues
        if not checks["config_valid"]:
            checks["issues"].append("Configuration invalid")

        device_info = checks["device_info"]
        if device_info.get("device") == "cuda" and not device_info.get(
            "cuda_available"
        ):
            checks["issues"].append(
                "CUDA requested but not available"
            )

        return checks


class TrainingLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str):
        """
        Initialize training logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "training_metrics.json"
        self.metrics = {
            "training": [],
            "validation": [],
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float] = None,
        val_metrics: Dict[str, float] = None,
    ):
        """Log metrics for an epoch."""
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        if train_metrics:
            entry["train_metrics"] = train_metrics
        if val_metrics:
            entry["val_metrics"] = val_metrics

        self.metrics["training"].append(entry)

        self._save_metrics()

    def log_batch(
        self,
        epoch: int,
        batch: int,
        loss: float,
        learning_rate: float = None,
    ):
        """Log metrics for a training batch."""
        entry = {
            "epoch": epoch,
            "batch": batch,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }

        if learning_rate is not None:
            entry["learning_rate"] = learning_rate

        self.metrics["training"].append(entry)

    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.metrics["training"]:
            return {}

        return {
            "total_epochs": len(
                set(e["epoch"] for e in self.metrics["training"] if "epoch" in e)
            ),
            "total_batches": len(
                [e for e in self.metrics["training"] if "batch" in e]
            ),
            "start_time": self.metrics["training"][0].get("timestamp"),
            "end_time": self.metrics["training"][-1].get("timestamp"),
        }


def create_training_checkpoint(
    model_state: Any,
    optimizer_state: Any,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str,
) -> bool:
    """
    Create and save training checkpoint.

    Args:
        model_state: Model state dict
        optimizer_state: Optimizer state dict
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint

    Returns:
        bool: True if successful
    """
    try:
        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # For paddle models, use torch/pickle as fallback
        import pickle
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        return False


def load_training_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint dictionary or None if failed
    """
    try:
        import pickle
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return None


def get_training_summary(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Get summary of all training checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Summary dictionary
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_path.glob("*.pt"))

    summary = {
        "total_checkpoints": len(checkpoints),
        "checkpoints": [],
    }

    for checkpoint_file in checkpoints:
        checkpoint = load_training_checkpoint(str(checkpoint_file))
        if checkpoint:
            summary["checkpoints"].append({
                "file": checkpoint_file.name,
                "epoch": checkpoint.get("epoch"),
                "timestamp": checkpoint.get("timestamp"),
                "metrics": checkpoint.get("metrics"),
            })

    return summary
