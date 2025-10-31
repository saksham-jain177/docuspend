"""
prepare_dataset.py

Convert Phase 1 curated DocuSpend dataset to PaddleOCR format.

Input: Phase 1 dataset (directory structure: {dataset_root}/{vendor}/{language}/)
       with images (JPG/PNG) and annotations (COCO JSON, PaddleOCR txt, or custom JSON)

Output: raw_annotations.txt in PaddleOCR format
        (one line per image: image_path\t[{"text": "...", "bbox": [[...]]}, ...])
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetConverter:
    """Convert Phase 1 dataset to PaddleOCR format."""

    def __init__(self, dataset_root: str):
        """
        Initialize converter.

        Args:
            dataset_root: Path to Phase 1 dataset root
        """
        self.dataset_root = Path(dataset_root)
        self.annotations = []
        self.vendor_distribution = {}
        self.language_distribution = {}

    def detect_annotation_format(self) -> str:
        """
        Detect annotation format by scanning directory structure.

        Returns:
            'coco' for COCO JSON format
            'paddleocr' for PaddleOCR txt format
            'custom' for per-image JSON format
            'unknown' if format cannot be determined
        """
        logger.info(f"Detecting annotation format in {self.dataset_root}")

        # Check for COCO JSON
        for root, dirs, files in os.walk(self.dataset_root):
            if 'instances.json' in files:
                logger.info("Detected COCO JSON format")
                return 'coco'
            if 'annotations.txt' in files or any(f.endswith('_list.txt') for f in files):
                logger.info("Detected PaddleOCR txt format")
                return 'paddleocr'
            if any(f.endswith('_annotations.json') for f in files):
                logger.info("Detected custom per-image JSON format")
                return 'custom'

        logger.warning("Could not detect annotation format")
        return 'unknown'

    def load_coco_format(self, coco_json_path: str) -> List[Dict]:
        """
        Load annotations from COCO JSON format.

        Args:
            coco_json_path: Path to instances.json

        Returns:
            List of annotation dictionaries
        """
        logger.info(f"Loading COCO format from {coco_json_path}")

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        annotations = []

        # Build image ID to image info mapping
        image_map = {img['id']: img for img in coco_data.get('images', [])}

        # Process annotations
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_map:
                continue

            img_info = image_map[image_id]
            image_path = img_info.get('file_name', '')

            # Extract text from COCO (if available in attributes)
            text = ann.get('caption', ann.get('text', 'OCR'))

            # Convert segmentation/bbox to PaddleOCR format
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            else:
                bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]

            annotations.append({
                'image_path': image_path,
                'text': text,
                'bbox': bbox,
                'image_id': image_id
            })

        logger.info(f"Loaded {len(annotations)} annotations from COCO format")
        return annotations

    def load_paddleocr_format(self, txt_path: str) -> List[Dict]:
        """
        Load annotations from PaddleOCR txt format.

        Args:
            txt_path: Path to train_list.txt or annotation file

        Returns:
            List of annotation dictionaries
        """
        logger.info(f"Loading PaddleOCR format from {txt_path}")

        annotations = []

        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: image_path\t[{"text": "...", "bbox": [...]}]
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    logger.warning(f"Invalid line format: {line[:100]}")
                    continue

                image_path = parts[0]
                try:
                    labels = json.loads(parts[1])

                    for label in labels:
                        annotations.append({
                            'image_path': image_path,
                            'text': label.get('text', 'OCR'),
                            'bbox': label.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]]),
                        })
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON in line: {line[:100]}")
                    continue

        logger.info(f"Loaded {len(annotations)} annotations from PaddleOCR format")
        return annotations

    def extract_vendor_language(self, image_path: str) -> Tuple[str, str]:
        """
        Extract vendor and language labels from image path.

        Assumes path structure: {vendor}/{language}/image.jpg

        Args:
            image_path: Relative image path

        Returns:
            Tuple of (vendor, language) or ('unknown', 'unknown') if cannot extract
        """
        parts = Path(image_path).parts

        if len(parts) >= 2:
            vendor = parts[0]
            language = parts[1]
            return vendor, language

        return 'unknown', 'unknown'

    def convert_to_paddleocr_format(self, annotations: List[Dict], output_path: str):
        """
        Convert annotations to PaddleOCR format and save.

        Args:
            annotations: List of annotation dictionaries
            output_path: Path to save raw_annotations.txt
        """
        logger.info(f"Converting {len(annotations)} annotations to PaddleOCR format")

        # Group annotations by image path
        image_groups = {}
        for ann in annotations:
            img_path = ann['image_path']
            if img_path not in image_groups:
                image_groups[img_path] = []
            image_groups[img_path].append(ann)

        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for image_path, anns in image_groups.items():
                # Extract vendor and language
                vendor, language = self.extract_vendor_language(image_path)

                # Update distribution
                self.vendor_distribution[vendor] = self.vendor_distribution.get(vendor, 0) + 1
                self.language_distribution[language] = self.language_distribution.get(language, 0) + 1

                # Create labels list
                labels = []
                for ann in anns:
                    labels.append({
                        'text': ann['text'],
                        'bbox': ann['bbox']
                    })

                # Write line: image_path\t[...]
                line = f"{image_path}\t{json.dumps(labels, ensure_ascii=False)}\n"
                f.write(line)

        logger.info(f"Saved {len(image_groups)} annotations to {output_path}")

    def validate_dataset(self, annotations_path: str, image_root: str = None):
        """
        Validate dataset integrity.

        Args:
            annotations_path: Path to raw_annotations.txt
            image_root: Optional root path to verify image accessibility
        """
        logger.info("Validating dataset integrity...")

        errors = []
        warnings = []

        with open(annotations_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t', 1)
                if len(parts) != 2:
                    errors.append(f"Line {i}: Invalid format (expected 2 parts)")
                    continue

                image_path = parts[0]
                try:
                    labels = json.loads(parts[1])

                    # Verify image exists if image_root provided
                    if image_root:
                        full_path = os.path.join(image_root, image_path)
                        if not os.path.exists(full_path):
                            warnings.append(f"Line {i}: Image not found: {full_path}")

                    # Validate bounding boxes
                    if not isinstance(labels, list):
                        errors.append(f"Line {i}: Labels not a list")
                        continue

                    for label_idx, label in enumerate(labels):
                        if not isinstance(label.get('bbox'), list):
                            errors.append(f"Line {i}, label {label_idx}: bbox not a list")
                        elif len(label['bbox']) != 4:
                            errors.append(f"Line {i}, label {label_idx}: bbox must have 4 points")

                except json.JSONDecodeError:
                    errors.append(f"Line {i}: Invalid JSON in annotations")

        if errors:
            logger.error(f"Found {len(errors)} validation errors:")
            for error in errors[:10]:
                logger.error(f"  - {error}")
            if len(errors) > 10:
                logger.error(f"  ... and {len(errors) - 10} more errors")

        if warnings:
            logger.warning(f"Found {len(warnings)} validation warnings:")
            for warning in warnings[:5]:
                logger.warning(f"  - {warning}")

        logger.info("Validation complete")
        return len(errors) == 0

    def generate_report(self, output_path: str):
        """
        Generate dataset conversion report.

        Args:
            output_path: Path to save report
        """
        logger.info(f"Generating report: {output_path}")

        report = f"""# Dataset Preparation Report

## Summary
- Total images: {len(self.vendor_distribution) + len(self.language_distribution)}
- Vendors: {len(self.vendor_distribution)}
- Languages: {len(self.language_distribution)}

## Vendor Distribution
"""

        for vendor, count in sorted(self.vendor_distribution.items(), key=lambda x: x[1], reverse=True):
            report += f"- {vendor}: {count}\n"

        report += "\n## Language Distribution\n"
        for language, count in sorted(self.language_distribution.items(), key=lambda x: x[1], reverse=True):
            report += f"- {language}: {count}\n"

        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert Phase 1 dataset to PaddleOCR format')
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to Phase 1 dataset root')
    parser.add_argument('--output-dir', type=str, default='./docuspend/dataset/', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Validate dataset after conversion')

    args = parser.parse_args()

    # Create converter
    converter = DatasetConverter(args.dataset_root)

    # Detect format
    format_type = converter.detect_annotation_format()

    if format_type == 'unknown':
        logger.error("Cannot detect annotation format. Please verify dataset structure.")
        logger.error("Expected structure: {dataset_root}/{vendor}/{language}/ with images and annotations")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Converting dataset from {args.dataset_root}")
    logger.info(f"Output directory: {args.output_dir}")

    # This is a template - actual loading depends on Phase 1 dataset structure
    logger.info("Phase 1 dataset structure not yet available. Script provides template for conversion.")
    logger.info("Waiting for Phase 1 to provide curated dataset.")

    return 0


if __name__ == '__main__':
    exit(main())
