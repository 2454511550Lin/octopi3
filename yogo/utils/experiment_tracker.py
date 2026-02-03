"""
Experiment tracking system for YOGO training.

Creates organized experiment folders with timestamps containing:
- Dataset splits
- Hyperparameters
- Training logs
- Model checkpoints
- Evaluation metrics
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class ExperimentTracker:
    """Manages experiment folders and logging."""

    def __init__(self, results_dir: Path = Path("results"), experiment_name: Optional[str] = None):
        """
        Initialize experiment tracker.

        Args:
            results_dir: Root directory for all experiments
            experiment_name: Optional name for the experiment. If None, uses timestamp only.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Create timestamped experiment folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            folder_name = f"{timestamp}_{experiment_name}"
        else:
            folder_name = timestamp

        self.experiment_dir = self.results_dir / folder_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.config_dir = self.experiment_dir / "config"
        self.splits_dir = self.experiment_dir / "splits"
        self.metrics_dir = self.experiment_dir / "metrics"

        for d in [self.checkpoints_dir, self.logs_dir, self.config_dir,
                  self.splits_dir, self.metrics_dir]:
            d.mkdir(exist_ok=True)

        print(f"Experiment directory: {self.experiment_dir}")

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """Save training configuration."""
        config_path = self.config_dir / filename

        # Convert non-serializable types
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, Path):
                serializable_config[k] = str(v)
            elif hasattr(v, 'item'):  # torch.Tensor or numpy
                serializable_config[k] = v.item()
            elif isinstance(v, (list, tuple)):
                serializable_config[k] = [x.item() if hasattr(x, 'item') else x for x in v]
            else:
                try:
                    json.dumps(v)  # Test if serializable
                    serializable_config[k] = v
                except (TypeError, ValueError):
                    serializable_config[k] = str(v)

        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2, sort_keys=True)

        print(f"Config saved: {config_path}")
        return config_path

    def save_dataset_splits(
        self,
        train_samples: List[str],
        val_samples: List[str],
        test_samples: Optional[List[str]] = None
    ):
        """Save dataset split information."""
        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples if test_samples else []
        }

        splits_path = self.splits_dir / "dataset_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)

        # Also save as readable text
        splits_txt = self.splits_dir / "dataset_splits.txt"
        with open(splits_txt, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATASET SPLITS\n")
            f.write("="*80 + "\n\n")

            f.write(f"TRAIN ({len(train_samples)} samples):\n")
            for s in train_samples:
                f.write(f"  - {s}\n")
            f.write("\n")

            f.write(f"VALIDATION ({len(val_samples)} samples):\n")
            for s in val_samples:
                f.write(f"  - {s}\n")
            f.write("\n")

            if test_samples:
                f.write(f"TEST ({len(test_samples)} samples):\n")
                for s in test_samples:
                    f.write(f"  - {s}\n")

        print(f"Dataset splits saved: {splits_path}")
        return splits_path

    def get_checkpoint_path(self, name: str = "model.pth") -> Path:
        """Get path for saving checkpoint."""
        return self.checkpoints_dir / name

    def get_log_path(self, name: str = "training.log") -> Path:
        """Get path for log file."""
        return self.logs_dir / name

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        """Save evaluation metrics."""
        metrics_path = self.metrics_dir / filename

        # Convert tensors to Python types
        serializable_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                serializable_metrics[k] = v.item()
            elif hasattr(v, 'tolist'):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v

        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Metrics saved: {metrics_path}")
        return metrics_path

    def copy_dataset_definition(self, dataset_defn_path: Path):
        """Copy dataset definition file to experiment folder."""
        dest = self.config_dir / "dataset_definition.yaml"
        shutil.copy(dataset_defn_path, dest)
        print(f"Dataset definition copied: {dest}")
        return dest

    def log(self, message: str, filename: str = "training.log"):
        """Append message to log file."""
        log_path = self.logs_dir / filename
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def save_summary(self, summary: Dict[str, Any]):
        """Save experiment summary."""
        summary_path = self.experiment_dir / "SUMMARY.txt"

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n\n")

            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

            f.write("\n" + "="*80 + "\n")
            f.write(f"Experiment directory: {self.experiment_dir}\n")
            f.write("="*80 + "\n")

        print(f"Summary saved: {summary_path}")
        return summary_path

    def __str__(self) -> str:
        return f"ExperimentTracker({self.experiment_dir})"

    def __repr__(self) -> str:
        return self.__str__()
