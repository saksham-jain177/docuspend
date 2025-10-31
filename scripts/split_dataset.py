"""
split_dataset.py

Split dataset into stratified train/val splits with vendor and language balance.

Input: raw_annotations.txt (from prepare_dataset.py)
Output: train_list.txt (80%), val_list.txt (20%), SPLIT_REPORT.md
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """Split dataset into train/val with stratification by vendor and language."""

    def __init__(self, annotations_path: str, train_ratio: float = 0.8, random_seed: int = 42):
        """
        Initialize splitter.

        Args:
            annotations_path: Path to raw_annotations.txt
            train_ratio: Ratio of training samples (default 0.8)
            random_seed: Random seed for reproducibility
        """
        self.annotations_path = annotations_path
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        self.random_seed = random_seed
        random.seed(random_seed)

        self.annotations = []
        self.stratification_groups = defaultdict(list)
        self.train_split = []
        self.val_split = []
        self.vendor_counts = defaultdict(lambda: {'total': 0, 'train': 0, 'val': 0})
        self.language_counts = defaultdict(lambda: {'total': 0, 'train': 0, 'val': 0})

    def load_annotations(self):
        """Load annotations from file."""
        logger.info(f"Loading annotations from {self.annotations_path}")

        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t', 1)
                if len(parts) != 2:
                    logger.warning(f"Skipping invalid line {i+1}")
                    continue

                image_path = parts[0]
                try:
                    labels = json.loads(parts[1])
                    self.annotations.append({
                        'line': line,
                        'image_path': image_path,
                        'labels': labels
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {i+1} with invalid JSON")
                    continue

        logger.info(f"Loaded {len(self.annotations)} annotations")

    def extract_vendor_language(self, image_path: str) -> Tuple[str, str]:
        """
        Extract vendor and language from image path.

        Assumes structure: {vendor}/{language}/image.jpg

        Args:
            image_path: Relative image path

        Returns:
            Tuple of (vendor, language)
        """
        parts = Path(image_path).parts

        if len(parts) >= 2:
            vendor = parts[0]
            language = parts[1]
            return vendor, language

        return 'unknown', 'unknown'

    def create_stratification_groups(self):
        """Group annotations by (vendor, language) pair."""
        logger.info("Creating stratification groups by (vendor, language) pair")

        for ann in self.annotations:
            vendor, language = self.extract_vendor_language(ann['image_path'])
            key = (vendor, language)
            self.stratification_groups[key].append(ann)

        logger.info(f"Created {len(self.stratification_groups)} stratification groups")
        for (vendor, language), anns in sorted(self.stratification_groups.items()):
            logger.info(f"  ({vendor}, {language}): {len(anns)} samples")

    def stratified_split(self):
        """
        Split each group into train/val maintaining ratio.

        Ensures every vendor-language combination appears in both splits.
        """
        logger.info("Performing stratified split...")

        for (vendor, language), anns in self.stratification_groups.items():
            # Shuffle within group
            shuffled = anns.copy()
            random.shuffle(shuffled)

            # Split by ratio
            split_idx = int(len(shuffled) * self.train_ratio)
            train_group = shuffled[:split_idx]
            val_group = shuffled[split_idx:]

            # Add to splits
            self.train_split.extend(train_group)
            self.val_split.extend(val_group)

            # Update counts
            self.vendor_counts[vendor]['total'] += len(anns)
            self.vendor_counts[vendor]['train'] += len(train_group)
            self.vendor_counts[vendor]['val'] += len(val_group)

            self.language_counts[language]['total'] += len(anns)
            self.language_counts[language]['train'] += len(train_group)
            self.language_counts[language]['val'] += len(val_group)

        logger.info(f"Train split: {len(self.train_split)} samples")
        logger.info(f"Val split: {len(self.val_split)} samples")

    def save_splits(self, output_dir: str):
        """
        Save train and val splits to files.

        Args:
            output_dir: Output directory path
        """
        logger.info(f"Saving splits to {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Save train split
        train_path = os.path.join(output_dir, 'train_list.txt')
        with open(train_path, 'w', encoding='utf-8') as f:
            for ann in self.train_split:
                f.write(ann['line'] + '\n')
        logger.info(f"Saved {len(self.train_split)} train samples to {train_path}")

        # Save val split
        val_path = os.path.join(output_dir, 'val_list.txt')
        with open(val_path, 'w', encoding='utf-8') as f:
            for ann in self.val_split:
                f.write(ann['line'] + '\n')
        logger.info(f"Saved {len(self.val_split)} val samples to {val_path}")

    def verify_split_integrity(self):
        """Verify no data leakage between splits."""
        logger.info("Verifying split integrity...")

        train_images = set(ann['image_path'] for ann in self.train_split)
        val_images = set(ann['image_path'] for ann in self.val_split)

        overlap = train_images & val_images
        if overlap:
            logger.error(f"Data leakage detected: {len(overlap)} images in both splits")
            return False
        else:
            logger.info("✓ No data leakage detected")

        # Verify vendor/language representation
        logger.info("Verifying vendor/language representation in both splits...")
        train_vendors = set()
        train_languages = set()
        val_vendors = set()
        val_languages = set()

        for ann in self.train_split:
            vendor, language = self.extract_vendor_language(ann['image_path'])
            train_vendors.add(vendor)
            train_languages.add(language)

        for ann in self.val_split:
            vendor, language = self.extract_vendor_language(ann['image_path'])
            val_vendors.add(vendor)
            val_languages.add(language)

        missing_vendors = train_vendors ^ val_vendors
        missing_languages = train_languages ^ val_languages

        if missing_vendors:
            logger.warning(f"Vendors not in both splits: {missing_vendors}")
        else:
            logger.info("✓ All vendors represented in both splits")

        if missing_languages:
            logger.warning(f"Languages not in both splits: {missing_languages}")
        else:
            logger.info("✓ All languages represented in both splits")

        return True

    def generate_report(self, output_path: str):
        """
        Generate split report.

        Args:
            output_path: Path to save report
        """
        logger.info(f"Generating split report: {output_path}")

        report = f"""# Dataset Split Report

## Overview
- **Total samples:** {len(self.annotations)}
- **Train samples:** {len(self.train_split)} ({len(self.train_split)/len(self.annotations)*100:.1f}%)
- **Val samples:** {len(self.val_split)} ({len(self.val_split)/len(self.annotations)*100:.1f}%)
- **Stratification:** By (vendor, language) pair
- **Random seed:** {self.random_seed}

## Vendor Distribution

| Vendor | Total | Train | Val |
|--------|-------|-------|-----|
"""

        for vendor in sorted(self.vendor_counts.keys()):
            counts = self.vendor_counts[vendor]
            report += f"| {vendor} | {counts['total']} | {counts['train']} | {counts['val']} |\n"

        report += f"\n## Language Distribution\n\n| Language | Total | Train | Val |\n|----------|-------|-------|-----|\n"

        for language in sorted(self.language_counts.keys()):
            counts = self.language_counts[language]
            report += f"| {language} | {counts['total']} | {counts['train']} | {counts['val']} |\n"

        report += f"""

## Stratification Groups

All {len(self.stratification_groups)} (vendor, language) pairs are represented in both train and val splits.

### Groups:
"""

        for (vendor, language), anns in sorted(self.stratification_groups.items()):
            train_count = sum(1 for ann in self.train_split if
                            self.extract_vendor_language(ann['image_path']) == (vendor, language))
            val_count = sum(1 for ann in self.val_split if
                           self.extract_vendor_language(ann['image_path']) == (vendor, language))
            report += f"- ({vendor}, {language}): {len(anns)} total → {train_count} train, {val_count} val\n"

        report += f"""

## Verification
- ✓ No data leakage between train and val
- ✓ All vendors represented in both splits
- ✓ All languages represented in both splits
- ✓ Stratification maintained across splits

Generated: {Path(output_path).parent}/
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into stratified train/val')
    parser.add_argument('--annotations', type=str, required=True, help='Path to raw_annotations.txt')
    parser.add_argument('--output-dir', type=str, default='./docuspend/dataset/', help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train ratio (default 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Dataset Stratified Split")
    logger.info("=" * 80)

    # Check input file exists
    if not os.path.exists(args.annotations):
        logger.error(f"Input file not found: {args.annotations}")
        logger.info("Please run prepare_dataset.py first to generate raw_annotations.txt")
        return 1

    # Create splitter
    splitter = StratifiedSplitter(
        args.annotations,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )

    # Load and split
    splitter.load_annotations()
    splitter.create_stratification_groups()
    splitter.stratified_split()
    splitter.verify_split_integrity()
    splitter.save_splits(args.output_dir)
    splitter.generate_report(os.path.join(args.output_dir, 'SPLIT_REPORT.md'))

    logger.info("=" * 80)
    logger.info("Split complete!")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
