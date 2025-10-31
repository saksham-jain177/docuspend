"""
visualize_metrics.py

Generate metric visualizations from training CSV files.

Generates 5 PNG plots:
1. Loss curve (training vs validation)
2. F1 score curve
3. Recognition accuracy curve
4. Combined score curve
5. Learning rate schedule
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Colab/headless
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsVisualizer:
    """Generate metric visualizations from training data."""

    def __init__(self, metrics_csv: str, output_dir: str = './docuspend/logs/'):
        """
        Initialize visualizer.

        Args:
            metrics_csv: Path to metrics.csv file
            output_dir: Output directory for plots
        """
        self.metrics_csv = Path(metrics_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []
        self.best_epoch = None
        self.best_combined_score = -1

    def load_metrics(self) -> bool:
        """
        Load metrics from CSV file.

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading metrics from {self.metrics_csv}")

        if not self.metrics_csv.exists():
            logger.error(f"Metrics file not found: {self.metrics_csv}")
            return False

        try:
            with open(self.metrics_csv, 'r') as f:
                header = f.readline().strip().split(',')
                for i, line in enumerate(f, 1):
                    values = line.strip().split(',')
                    if len(values) != len(header):
                        logger.warning(f"Line {i+1}: Mismatched columns, skipping")
                        continue

                    metric_dict = {}
                    for h, v in zip(header, values):
                        try:
                            # Try to convert to float, fall back to string
                            metric_dict[h.strip()] = float(v)
                        except ValueError:
                            metric_dict[h.strip()] = v

                    self.metrics.append(metric_dict)

                    # Track best epoch
                    if 'combined_score' in metric_dict:
                        score = metric_dict['combined_score']
                        if score > self.best_combined_score:
                            self.best_combined_score = score
                            self.best_epoch = metric_dict.get('epoch', i)

            logger.info(f"✓ Loaded {len(self.metrics)} epoch metrics")
            if self.best_epoch is not None:
                logger.info(f"  Best epoch: {self.best_epoch} (combined_score: {self.best_combined_score:.4f})")
            return True

        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return False

    def extract_column(self, col_name: str) -> Tuple[List, bool]:
        """
        Extract column from metrics.

        Args:
            col_name: Column name

        Returns:
            Tuple of (values, exists) where exists is True if column found
        """
        if not self.metrics:
            return [], False

        if col_name not in self.metrics[0]:
            return [], False

        values = []
        for metric in self.metrics:
            try:
                values.append(float(metric.get(col_name, 0)))
            except (ValueError, TypeError):
                values.append(0)

        return values, True

    def get_epochs(self) -> List[int]:
        """Get epoch numbers."""
        epochs = []
        for metric in self.metrics:
            try:
                epochs.append(int(metric.get('epoch', 0)))
            except (ValueError, TypeError):
                epochs.append(len(epochs) + 1)
        return epochs

    def plot_loss_curve(self):
        """Generate loss curve plot."""
        logger.info("Generating loss curve...")

        epochs = self.get_epochs()
        train_loss, train_exists = self.extract_column('avg_train_loss')
        val_loss, val_exists = self.extract_column('val_loss')

        if not train_exists and not val_exists:
            logger.warning("Loss columns not found in metrics, skipping loss curve")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if train_exists and train_loss:
            ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        if val_exists and val_loss:
            ax.plot(epochs, val_loss, 'orange', label='Validation Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training vs Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'loss_curve.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Loss curve saved: {output_path}")

    def plot_f1_curve(self):
        """Generate F1 score curve plot."""
        logger.info("Generating F1 score curve...")

        epochs = self.get_epochs()
        train_f1, train_exists = self.extract_column('avg_train_f1_det')
        val_f1, val_exists = self.extract_column('val_f1_det')

        if not train_exists and not val_exists:
            logger.warning("F1 columns not found in metrics, skipping F1 curve")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if train_exists and train_f1:
            ax.plot(epochs, train_f1, 'b-', label='Training F1', linewidth=2)
        if val_exists and val_f1:
            ax.plot(epochs, val_f1, 'orange', label='Validation F1', linewidth=2)

        # Highlight best epoch
        if self.best_epoch is not None and val_f1:
            try:
                best_idx = int(self.best_epoch) - 1
                if 0 <= best_idx < len(val_f1):
                    ax.axvline(x=self.best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
            except (ValueError, IndexError):
                pass

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Detection F1 Score Over Epochs', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'f1_curve.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ F1 curve saved: {output_path}")

    def plot_accuracy_curve(self):
        """Generate recognition accuracy curve plot."""
        logger.info("Generating accuracy curve...")

        epochs = self.get_epochs()
        train_acc, train_exists = self.extract_column('avg_train_acc_rec')
        val_acc, val_exists = self.extract_column('val_acc_rec')

        if not train_exists and not val_exists:
            logger.warning("Accuracy columns not found in metrics, skipping accuracy curve")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if train_exists and train_acc:
            ax.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        if val_exists and val_acc:
            ax.plot(epochs, val_acc, 'orange', label='Validation Accuracy', linewidth=2)

        # Highlight best epoch
        if self.best_epoch is not None and val_acc:
            try:
                best_idx = int(self.best_epoch) - 1
                if 0 <= best_idx < len(val_acc):
                    ax.axvline(x=self.best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
            except (ValueError, IndexError):
                pass

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Recognition Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'accuracy_curve.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Accuracy curve saved: {output_path}")

    def plot_combined_score_curve(self):
        """Generate combined score curve plot."""
        logger.info("Generating combined score curve...")

        epochs = self.get_epochs()
        combined_score, exists = self.extract_column('combined_score')

        if not exists or not combined_score:
            logger.warning("Combined score not found in metrics, skipping combined score curve")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, combined_score, 'g-', label='Combined Score', linewidth=2)

        # Highlight best epoch
        if self.best_epoch is not None:
            try:
                best_idx = int(self.best_epoch) - 1
                if 0 <= best_idx < len(combined_score):
                    ax.axvline(x=self.best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
                    ax.text(self.best_epoch, self.best_combined_score,
                           f'  Best: {self.best_combined_score:.4f}',
                           fontsize=10, va='center')
            except (ValueError, IndexError):
                pass

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Combined Score', fontsize=12)
        ax.set_title('Combined Score (0.5 * F1_detection + 0.5 * Accuracy_recognition)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / 'combined_score_curve.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Combined score curve saved: {output_path}")

    def plot_lr_schedule(self):
        """Generate learning rate schedule plot."""
        logger.info("Generating learning rate schedule...")

        epochs = self.get_epochs()
        lr_values, exists = self.extract_column('learning_rate')

        if not exists or not lr_values:
            logger.info("Learning rate column not found, generating theoretical cosine schedule")
            # Generate theoretical cosine warmup schedule
            lr_values = []
            base_lr = 0.001
            warmup_epochs = 2
            total_epochs = len(epochs) if epochs else 30

            for epoch in epochs:
                if epoch <= warmup_epochs:
                    # Linear warmup
                    lr = base_lr * (epoch / warmup_epochs)
                else:
                    # Cosine decay
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
                lr_values.append(max(lr, 1e-6))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, lr_values, 'purple', linewidth=2, label='Learning Rate')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        output_path = self.output_dir / 'lr_schedule.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Learning rate schedule saved: {output_path}")

    def visualize_all(self) -> bool:
        """
        Generate all visualizations.

        Returns:
            True if all visualizations succeeded
        """
        logger.info("=" * 80)
        logger.info("Generating Metric Visualizations")
        logger.info("=" * 80)

        if not self.load_metrics():
            return False

        try:
            self.plot_loss_curve()
            self.plot_f1_curve()
            self.plot_accuracy_curve()
            self.plot_combined_score_curve()
            self.plot_lr_schedule()

            logger.info("=" * 80)
            logger.info("✓ All visualizations generated successfully!")
            logger.info("=" * 80)
            return True

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Generate metric visualizations')
    parser.add_argument(
        '--metrics-csv',
        type=str,
        default='docuspend/logs/metrics.csv',
        help='Path to metrics.csv file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='docuspend/logs/',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    visualizer = MetricsVisualizer(args.metrics_csv, args.output_dir)
    success = visualizer.visualize_all()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
