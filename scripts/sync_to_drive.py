"""
sync_to_drive.py

Synchronize training checkpoints and logs to Google Drive.

This script is designed for Colab notebooks and copies:
- Epoch checkpoints
- Best model checkpoint
- Final checkpoint
- Metrics CSV files
- Training logs
- Generated plots

To mount Google Drive in Colab:
    from google.colab import drive
    drive.mount('/content/drive')

Usage:
    python docuspend/scripts/sync_to_drive.py \\
      --checkpoint-dir ./output/ \\
      --logs-dir docuspend/logs/
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoogleDriveSync:
    """Synchronize checkpoints and logs to Google Drive."""

    # Colab Google Drive mount point
    DRIVE_MOUNT = Path('/content/drive')
    DRIVE_BACKUP_DIR = DRIVE_MOUNT / 'My Drive' / 'docuspend_checkpoints'

    def __init__(self, checkpoint_dir: str, logs_dir: str):
        """
        Initialize sync manager.

        Args:
            checkpoint_dir: Path to checkpoint directory
            logs_dir: Path to logs directory
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.is_colab = self.detect_colab()
        self.files_synced = 0
        self.bytes_synced = 0

    def detect_colab(self) -> bool:
        """
        Detect if running in Colab environment.

        Returns:
            True if in Colab, False otherwise
        """
        try:
            import google.colab
            logger.info("✓ Detected Colab environment")
            return True
        except ImportError:
            logger.warning("⚠ Not running in Colab environment")
            logger.info("  Google Drive sync will be skipped (only works in Colab)")
            return False

    def verify_drive_mount(self) -> bool:
        """
        Verify Google Drive is mounted.

        Returns:
            True if mounted, False otherwise
        """
        if not self.is_colab:
            return False

        if not self.DRIVE_MOUNT.exists():
            logger.error(f"Google Drive not mounted at {self.DRIVE_MOUNT}")
            logger.info("Please run in Colab: drive.mount('/content/drive')")
            return False

        logger.info(f"✓ Google Drive mounted at {self.DRIVE_MOUNT}")
        return True

    def create_backup_dir(self) -> bool:
        """
        Create backup directory on Google Drive.

        Returns:
            True if successful or already exists
        """
        if not self.is_colab:
            return False

        try:
            self.DRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Backup directory ready: {self.DRIVE_BACKUP_DIR}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup directory: {e}")
            return False

    def copy_file(self, source: Path, destination: Path, description: str = "") -> bool:
        """
        Copy file with error handling.

        Args:
            source: Source file path
            destination: Destination file path
            description: Description for logging

        Returns:
            True if successful
        """
        if not source.exists():
            logger.warning(f"Source not found: {source}")
            return False

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            file_size = source.stat().st_size
            self.files_synced += 1
            self.bytes_synced += file_size
            logger.info(f"  ✓ {description or source.name} ({file_size/1024:.1f}KB)")
            return True
        except Exception as e:
            logger.error(f"Failed to copy {source}: {e}")
            return False

    def copy_directory(self, source: Path, destination: Path, description: str = "") -> bool:
        """
        Copy entire directory with error handling.

        Args:
            source: Source directory path
            destination: Destination directory path
            description: Description for logging

        Returns:
            True if successful
        """
        if not source.exists():
            logger.warning(f"Source directory not found: {source}")
            return False

        try:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            logger.info(f"  ✓ {description or source.name}/ (directory)")
            self.files_synced += 1
            return True
        except Exception as e:
            logger.error(f"Failed to copy directory {source}: {e}")
            return False

    def sync_checkpoints(self) -> bool:
        """
        Sync checkpoint files to Google Drive.

        Returns:
            True if sync succeeded or was skipped
        """
        if not self.is_colab:
            logger.info("Skipping checkpoint sync (not in Colab)")
            return True

        if not self.verify_drive_mount():
            return False

        if not self.create_backup_dir():
            return False

        logger.info("Syncing checkpoints...")

        success = True

        # Copy best model directory
        best_model = self.checkpoint_dir / 'best_accuracy'
        if best_model.exists():
            logger.info("Backing up best model...")
            self.copy_directory(
                best_model,
                self.DRIVE_BACKUP_DIR / 'best_model',
                'best_model'
            )

        # Copy epoch checkpoints
        epoch_pattern = '*.pth'
        epoch_files = list(self.checkpoint_dir.glob(epoch_pattern))
        if epoch_files:
            logger.info(f"Backing up {len(epoch_files)} epoch checkpoint(s)...")
            for pth_file in epoch_files:
                self.copy_file(
                    pth_file,
                    self.DRIVE_BACKUP_DIR / pth_file.name,
                    pth_file.name
                )

        # Copy final checkpoint if exists
        final_checkpoint = self.checkpoint_dir / 'final.pth'
        if final_checkpoint.exists():
            self.copy_file(
                final_checkpoint,
                self.DRIVE_BACKUP_DIR / 'final.pth',
                'final.pth'
            )

        return success

    def sync_logs_and_metrics(self) -> bool:
        """
        Sync log files and metric CSVs to Google Drive.

        Returns:
            True if sync succeeded or was skipped
        """
        if not self.is_colab:
            logger.info("Skipping logs sync (not in Colab)")
            return True

        if not self.verify_drive_mount():
            return False

        if not self.create_backup_dir():
            return False

        logger.info("Syncing logs and metrics...")

        success = True

        # Copy metrics CSV
        metrics_csv = self.logs_dir / 'metrics.csv'
        if metrics_csv.exists():
            self.copy_file(
                metrics_csv,
                self.DRIVE_BACKUP_DIR / 'metrics.csv',
                'metrics.csv'
            )

        # Copy raw metrics CSV
        raw_metrics_csv = self.logs_dir / 'raw_metrics.csv'
        if raw_metrics_csv.exists():
            self.copy_file(
                raw_metrics_csv,
                self.DRIVE_BACKUP_DIR / 'raw_metrics.csv',
                'raw_metrics.csv'
            )

        # Copy training log
        training_log = self.logs_dir / 'training_log.txt'
        if training_log.exists():
            self.copy_file(
                training_log,
                self.DRIVE_BACKUP_DIR / 'training_log.txt',
                'training_log.txt'
            )

        # Copy PNG plots
        plot_files = list(self.logs_dir.glob('*.png'))
        if plot_files:
            logger.info(f"Backing up {len(plot_files)} plot file(s)...")
            for plot in plot_files:
                self.copy_file(
                    plot,
                    self.DRIVE_BACKUP_DIR / plot.name,
                    plot.name
                )

        return success

    def sync_training_summary(self) -> bool:
        """
        Sync training summary markdown to Google Drive.

        Returns:
            True if sync succeeded or was skipped
        """
        if not self.is_colab:
            return True

        if not self.verify_drive_mount():
            return False

        summary_file = Path('docuspend/TRAINING_SUMMARY.md')
        if summary_file.exists():
            logger.info("Backing up training summary...")
            self.copy_file(
                summary_file,
                self.DRIVE_BACKUP_DIR / 'TRAINING_SUMMARY.md',
                'TRAINING_SUMMARY.md'
            )
            return True

        return True

    def generate_sync_report(self) -> bool:
        """
        Generate sync report.

        Returns:
            True if report generated
        """
        report = f"""# Google Drive Sync Report

## Summary
- **Sync time:** {datetime.now().isoformat()}
- **Files synced:** {self.files_synced}
- **Data synced:** {self.bytes_synced / 1024 / 1024:.2f} MB
- **Destination:** {self.DRIVE_BACKUP_DIR if self.is_colab else 'N/A (not in Colab)'}

## Status
{'✓ Sync completed successfully' if self.is_colab else '⚠ Skipped (not running in Colab)'}

## Synced Files
- Checkpoints (best, epoch, final)
- Metrics CSV (metrics.csv, raw_metrics.csv)
- Training logs (training_log.txt)
- Plots (loss_curve.png, f1_curve.png, accuracy_curve.png, combined_score_curve.png, lr_schedule.png)
- Training summary (TRAINING_SUMMARY.md)
"""

        try:
            report_path = Path('docuspend/logs/sync_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Sync report saved: {report_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save sync report: {e}")
            return False

    def run(self) -> int:
        """
        Run complete sync pipeline.

        Returns:
            0 if successful, 1 if failed
        """
        try:
            logger.info("=" * 80)
            logger.info("Google Drive Checkpoint Sync")
            logger.info("=" * 80)

            if not self.is_colab:
                logger.info("Not in Colab environment. Sync skipped.")
                logger.info("To enable sync in Colab, run: drive.mount('/content/drive')")
                return 0

            # Verify directories exist
            if not self.checkpoint_dir.exists():
                logger.error(f"Checkpoint directory not found: {self.checkpoint_dir}")
                return 1

            if not self.logs_dir.exists():
                logger.error(f"Logs directory not found: {self.logs_dir}")
                return 1

            # Verify Drive is mounted
            if not self.verify_drive_mount():
                logger.error("Google Drive not properly mounted")
                return 1

            # Run sync operations
            self.sync_checkpoints()
            self.sync_logs_and_metrics()
            self.sync_training_summary()
            self.generate_sync_report()

            logger.info("=" * 80)
            if self.files_synced > 0:
                logger.info(f"✓ Sync completed! ({self.files_synced} files, {self.bytes_synced/1024/1024:.2f}MB)")
            else:
                logger.info("✓ Sync completed (no files to sync)")
            logger.info("=" * 80)

            return 0

        except Exception as e:
            logger.error(f"Unexpected error during sync: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(
        description='Sync training artifacts to Google Drive'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        required=True,
        help='Path to logs directory'
    )

    args = parser.parse_args()

    syncer = GoogleDriveSync(args.checkpoint_dir, args.logs_dir)
    return syncer.run()


if __name__ == '__main__':
    sys.exit(main())
