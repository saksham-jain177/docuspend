"""
Dataset Utilities for DocuSpend
DocuSpend Project - PaddleOCR-VL Fine-tuning

This module provides dataset loaders and utilities for working with
receipt/invoice images and their annotations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ReceiptDataset:
    """Dataset loader for receipt/invoice images with annotations."""

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        split: str = "train",
        image_format: str = "png",
    ):
        """
        Initialize dataset.

        Args:
            image_dir: Directory containing preprocessed images
            annotation_dir: Directory containing annotation JSON files
            split: Data split (train, val, test)
            image_format: Image file format (default: png)
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.split = split
        self.image_format = image_format

        self.image_files = []
        self.annotations = {}

        self._load_dataset()

    def _load_dataset(self):
        """Load image paths and annotations."""
        # Find all images
        pattern = f"*.{self.image_format}"
        self.image_files = sorted(self.image_dir.glob(pattern))

        if not self.image_files:
            logger.warning(f"No images found in {self.image_dir}")

        # Load annotations
        for img_file in self.image_files:
            annotation_file = (
                self.annotation_dir / f"{img_file.stem}.json"
            )
            if annotation_file.exists():
                try:
                    with open(annotation_file, "r") as f:
                        self.annotations[str(img_file)] = json.load(f)
                except Exception as e:
                    logger.warning(
                        f"Failed to load annotation {annotation_file}: {str(e)}"
                    )
            else:
                logger.warning(f"Missing annotation for {img_file}")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get single sample.

        Args:
            idx: Index of sample

        Returns:
            Tuple of (image_array, annotation_dict)
        """
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range")

        img_file = self.image_files[idx]

        # Load image
        image = Image.open(img_file).convert("L")  # Convert to grayscale
        image_array = np.array(image)

        # Load annotation
        annotation = self.annotations.get(str(img_file), {})

        return image_array, annotation

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        image_sizes = []
        categories = {}
        total_items = 0
        total_amount = 0.0

        for annotation in self.annotations.values():
            # Track categories
            category = annotation.get("category", "other")
            categories[category] = categories.get(category, 0) + 1

            # Track items and amounts
            items = annotation.get("items", [])
            total_items += len(items)
            total_amount += annotation.get("total", 0.0)

        # Calculate image dimensions
        for img_file in self.image_files:
            try:
                img = Image.open(img_file)
                image_sizes.append(img.size)
            except Exception:
                pass

        return {
            "split": self.split,
            "total_samples": len(self.image_files),
            "total_annotations": len(self.annotations),
            "image_dimensions": {
                "min": min(image_sizes) if image_sizes else None,
                "max": max(image_sizes) if image_sizes else None,
                "avg": (
                    (
                        sum(s[0] for s in image_sizes) // len(image_sizes),
                        sum(s[1] for s in image_sizes) // len(image_sizes),
                    )
                    if image_sizes
                    else None
                ),
            },
            "categories": categories,
            "total_items": total_items,
            "avg_items_per_receipt": (
                total_items / len(self.annotations)
                if self.annotations
                else 0
            ),
            "total_amount": round(total_amount, 2),
            "avg_receipt_amount": (
                round(total_amount / len(self.annotations), 2)
                if self.annotations
                else 0
            ),
        }


class DatasetSplitter:
    """Utility for splitting dataset into train/val/test."""

    @staticmethod
    def split_dataset(
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """
        Split dataset into train/val/test.

        Args:
            source_dir: Directory with all images
            output_dir: Output directory for splits
            train_ratio: Proportion for training (default: 0.8)
            val_ratio: Proportion for validation (default: 0.1)
            test_ratio: Proportion for testing (default: 0.1)
            seed: Random seed for reproducibility

        Returns:
            Dictionary with split assignments
        """
        import random

        random.seed(seed)
        np.random.seed(seed)

        source_path = Path(source_dir)
        output_path = Path(output_dir)

        # Find all images
        image_files = sorted(source_path.glob("*.jpg")) + sorted(
            source_path.glob("*.png")
        )

        if not image_files:
            logger.error(f"No images found in {source_dir}")
            return {}

        # Shuffle and split
        random.shuffle(image_files)
        total = len(image_files)

        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_files = image_files[:train_size]
        val_files = image_files[train_size : train_size + val_size]
        test_files = image_files[train_size + val_size :]

        splits = {
            "train": [f.name for f in train_files],
            "val": [f.name for f in val_files],
            "test": [f.name for f in test_files],
        }

        # Create split directories
        for split_name, files in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Copy or link files
            for file_path in files:
                src_file = source_path / file_path
                dst_file = split_dir / file_path

                try:
                    # Create symbolic link (if available)
                    import shutil
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    logger.error(
                        f"Failed to copy {src_file} to {dst_file}: {str(e)}"
                    )

        logger.info(
            f"Split dataset: train={len(train_files)}, "
            f"val={len(val_files)}, test={len(test_files)}"
        )

        return splits

    @staticmethod
    def stratified_split(
        source_dir: str,
        annotation_file: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """
        Split dataset with stratification by category.

        Args:
            source_dir: Directory with all images
            annotation_file: JSON file with annotations (for categories)
            output_dir: Output directory for splits
            train_ratio: Proportion for training (default: 0.8)
            val_ratio: Proportion for validation (default: 0.1)
            test_ratio: Proportion for testing (default: 0.1)
            seed: Random seed

        Returns:
            Dictionary with split assignments
        """
        import random
        from sklearn.model_selection import train_test_split

        random.seed(seed)
        np.random.seed(seed)

        # Load annotations
        try:
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load annotations: {str(e)}")
            return {}

        # Group files by category
        source_path = Path(source_dir)
        image_files = sorted(source_path.glob("*.jpg")) + sorted(
            source_path.glob("*.png")
        )

        categories = {}
        for img_file in image_files:
            filename = img_file.name
            if filename in annotations:
                category = annotations[filename].get("category", "other")
                if category not in categories:
                    categories[category] = []
                categories[category].append(img_file)

        splits = {"train": [], "val": [], "test": []}

        # Stratified split by category
        for category, files in categories.items():
            train, temp = train_test_split(
                files, test_size=(val_ratio + test_ratio), random_state=seed
            )
            val, test = train_test_split(
                temp,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=seed,
            )

            splits["train"].extend([f.name for f in train])
            splits["val"].extend([f.name for f in val])
            splits["test"].extend([f.name for f in test])

        # Create split directories
        output_path = Path(output_dir)
        for split_name, files in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for filename in files:
                src_file = source_path / filename
                dst_file = split_dir / filename

                try:
                    import shutil
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    logger.error(
                        f"Failed to copy {src_file}: {str(e)}"
                    )

        logger.info(
            f"Stratified split: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        return splits


class AnnotationValidator:
    """Validate annotation files for completeness and correctness."""

    REQUIRED_FIELDS = [
        "filename",
        "vendor",
        "date",
        "total",
        "tax",
        "items",
        "category",
    ]

    VALID_CATEGORIES = [
        "groceries",
        "dining",
        "transport",
        "utilities",
        "shopping",
        "healthcare",
        "entertainment",
        "other",
    ]

    @classmethod
    def validate_annotation(cls, annotation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate single annotation.

        Args:
            annotation: Annotation dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in annotation:
                errors.append(f"Missing required field: {field}")

        # Validate field types and values
        if "total" in annotation:
            if not isinstance(annotation["total"], (int, float)):
                errors.append(f"Invalid total type: {type(annotation['total'])}")
            elif annotation["total"] < 0:
                errors.append(f"Negative total: {annotation['total']}")

        if "tax" in annotation:
            if not isinstance(annotation["tax"], (int, float)):
                errors.append(f"Invalid tax type: {type(annotation['tax'])}")
            elif annotation["tax"] < 0:
                errors.append(f"Negative tax: {annotation['tax']}")

        if "category" in annotation:
            if annotation["category"] not in cls.VALID_CATEGORIES:
                errors.append(
                    f"Invalid category: {annotation['category']}. "
                    f"Must be one of {cls.VALID_CATEGORIES}"
                )

        if "items" in annotation:
            if not isinstance(annotation["items"], list):
                errors.append(f"Items must be a list, got {type(annotation['items'])}")

        return len(errors) == 0, errors

    @classmethod
    def validate_dataset(cls, annotation_dir: str) -> Dict[str, Any]:
        """
        Validate all annotations in directory.

        Args:
            annotation_dir: Directory with annotation JSON files

        Returns:
            Dictionary with validation statistics
        """
        annotation_path = Path(annotation_dir)
        stats = {
            "total_files": 0,
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }

        for annotation_file in annotation_path.glob("*.json"):
            stats["total_files"] += 1

            try:
                with open(annotation_file, "r") as f:
                    annotation = json.load(f)

                is_valid, errors = cls.validate_annotation(annotation)

                if is_valid:
                    stats["valid"] += 1
                else:
                    stats["invalid"] += 1
                    stats["errors"].append({
                        "file": annotation_file.name,
                        "errors": errors
                    })

            except Exception as e:
                stats["invalid"] += 1
                stats["errors"].append({
                    "file": annotation_file.name,
                    "errors": [f"Failed to parse JSON: {str(e)}"]
                })

        return stats
