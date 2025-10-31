"""
Preprocessing Pipeline for Receipt/Invoice Images
DocuSpend Project - PaddleOCR-VL Fine-tuning

This module provides image preprocessing functions for receipt and invoice images
to improve OCR performance. Operations include grayscale conversion, deskewing,
noise removal, adaptive thresholding, and resolution normalization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ReceiptPreprocessor:
    """Preprocessor for receipt/invoice images."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary from preprocessing_config.yaml
        """
        self.config = config.get("preprocessing", {})
        self.target_dpi = self.config.get("target_dpi", 300)
        self.output_format = self.config.get("output_format", "png")
        self.min_angle_threshold = self.config.get("deskew", {}).get(
            "min_angle_threshold", 0.5
        )
        self.max_angle_threshold = self.config.get("deskew", {}).get(
            "max_angle_threshold", 45
        )
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0,
            "average_time": 0.0,
        }

    def preprocess_receipt(self, image_path: str, output_path: str) -> bool:
        """
        Preprocess receipt/invoice image for OCR.

        Args:
            image_path: Path to input image
            output_path: Path to save preprocessed image

        Returns:
            bool: True if successful, False otherwise

        Steps executed:
            1. Load image (BGR)
            2. Convert to grayscale
            3. Apply Gaussian blur (noise removal)
            4. Deskew (rotation correction)
            5. Adaptive thresholding (binarization)
            6. Morphological closing (optional)
            7. Normalize to target DPI
            8. Add border padding (optional)
            9. Save as PNG
        """
        start_time = datetime.now()

        try:
            # Step 1: Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                self.stats["failed"] += 1
                return False

            # Step 2: Convert to grayscale
            if self.config.get("grayscale", {}).get("enabled", True):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Step 3: Noise removal (Gaussian blur)
            if self.config.get("noise_removal", {}).get("enabled", True):
                blur_config = self.config.get("noise_removal", {}).get(
                    "gaussian_blur", {}
                )
                kernel_size = tuple(blur_config.get("kernel_size", [5, 5]))
                sigma = blur_config.get("sigma", 0)
                gray = cv2.GaussianBlur(gray, kernel_size, sigma)

            # Step 4: Deskew (rotation correction)
            if self.config.get("deskew", {}).get("enabled", True):
                gray = self._deskew(gray)

            # Step 5: Adaptive thresholding (binarization)
            if self.config.get("adaptive_threshold", {}).get("enabled", True):
                threshold_config = self.config.get("adaptive_threshold", {})
                block_size = threshold_config.get("block_size", 11)
                c_constant = threshold_config.get("c_constant", 2)

                # Ensure block_size is odd
                if block_size % 2 == 0:
                    block_size += 1

                gray = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    c_constant,
                )

            # Step 6: Morphological operations (optional)
            if self.config.get("morphology", {}).get("enabled", True):
                morph_config = self.config.get("morphology", {})
                kernel_size = tuple(morph_config.get("kernel_size", [2, 2]))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, kernel_size
                )
                operation = morph_config.get("operation", "close")
                iterations = morph_config.get("iterations", 1)

                if operation == "close":
                    gray = cv2.morphologyEx(
                        gray, cv2.MORPH_CLOSE, kernel, iterations=iterations
                    )
                elif operation == "open":
                    gray = cv2.morphologyEx(
                        gray, cv2.MORPH_OPEN, kernel, iterations=iterations
                    )

            # Step 7: Resolution normalization to target DPI
            if self.config.get("preprocessing", {}).get("enabled", True):
                gray = self._normalize_resolution(gray)

            # Step 8: Border padding (optional)
            if self.config.get("border_padding", {}).get("enabled", True):
                padding_config = self.config.get("border_padding", {})
                pad_size = padding_config.get("size", 20)
                pad_color = padding_config.get("color", 255)
                gray = cv2.copyMakeBorder(
                    gray,
                    pad_size,
                    pad_size,
                    pad_size,
                    pad_size,
                    cv2.BORDER_CONSTANT,
                    value=pad_color,
                )

            # Step 9: Save preprocessed image
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Convert back to BGR for consistency (if needed)
            output_img = gray

            success = cv2.imwrite(output_path, output_img)
            if success:
                self.stats["successful"] += 1
            else:
                logger.error(f"Failed to write image: {output_path}")
                self.stats["failed"] += 1
                return False

            elapsed = (datetime.now() - start_time).total_seconds()
            self.stats["total_time"] += elapsed

            return True

        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {str(e)}")
            self.stats["failed"] += 1
            return False
        finally:
            self.stats["total_processed"] += 1

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Deskew image by detecting and correcting rotation angle.

        Args:
            gray: Grayscale image

        Returns:
            Deskewed image
        """
        try:
            # Apply Otsu threshold for angle detection
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find all non-zero pixel coordinates
            coords = np.column_stack(np.where(binary > 0))

            if len(coords) < 10:
                # Not enough pixels, skip deskewing
                return gray

            # Calculate minimum area rectangle
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]

            # Normalize angle to [-45, 45] range
            if angle < -45:
                angle += 90

            # Skip rotation if angle is very small
            if abs(angle) < self.min_angle_threshold:
                return gray

            # Reject if angle exceeds threshold
            if abs(angle) > self.max_angle_threshold:
                logger.warning(
                    f"Skipping deskew: angle {angle} exceeds threshold "
                    f"{self.max_angle_threshold}"
                )
                return gray

            # Apply rotation
            (h, w) = gray.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

            # Calculate new bounding dimensions
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])

            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust rotation matrix translation
            rotation_matrix[0, 2] += (new_w / 2) - (w / 2)
            rotation_matrix[1, 2] += (new_h / 2) - (h / 2)

            # Apply rotation with white background
            background_color = self.config.get("deskew", {}).get(
                "background_color", 255
            )
            rotated = cv2.warpAffine(
                gray,
                rotation_matrix,
                (new_w, new_h),
                borderValue=background_color,
                flags=cv2.INTER_CUBIC,
            )

            return rotated

        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}, continuing without deskew")
            return gray

    def _normalize_resolution(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image resolution to target DPI.

        Args:
            img: Input image

        Returns:
            Resolution-normalized image
        """
        try:
            current_h, current_w = img.shape[:2]

            # Default: assume 150 DPI input, normalize to target DPI
            current_dpi = 150
            target_dpi = self.target_dpi

            # Calculate scale factor
            scale_factor = target_dpi / current_dpi

            # Calculate new dimensions
            new_w = int(current_w * scale_factor)
            new_h = int(current_h * scale_factor)

            # Resize image
            if scale_factor > 1:
                # Upscaling - use INTER_CUBIC
                normalized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                # Downscaling - use INTER_AREA
                normalized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return normalized

        except Exception as e:
            logger.warning(f"Resolution normalization failed: {str(e)}")
            return img

    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        if self.stats["total_processed"] > 0:
            self.stats["average_time"] = (
                self.stats["total_time"] / self.stats["total_processed"]
            )

        return self.stats


def preprocess_receipt(
    image_path: str, output_path: str, config: Dict[str, Any]
) -> bool:
    """
    Preprocess receipt/invoice image for OCR (functional API).

    Args:
        image_path: Path to input image
        output_path: Path to save preprocessed image
        config: Preprocessing configuration dict (from YAML)

    Returns:
        bool: True if successful, False otherwise

    Steps executed:
        1. Load image (BGR)
        2. Convert to grayscale
        3. Apply Gaussian blur (noise removal)
        4. Deskew (rotation correction)
        5. Adaptive thresholding (binarization)
        6. Morphological closing (optional)
        7. Normalize to 300 DPI
        8. Add border padding (optional)
        9. Save as PNG
    """
    preprocessor = ReceiptPreprocessor(config)
    return preprocessor.preprocess_receipt(image_path, output_path)


def batch_preprocess(
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    file_pattern: str = "*.jpg",
) -> Dict[str, Any]:
    """
    Batch preprocess images from input directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save preprocessed images
        config: Preprocessing configuration
        file_pattern: File pattern to match (default: *.jpg)

    Returns:
        Dictionary with preprocessing statistics
    """
    preprocessor = ReceiptPreprocessor(config)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Process all matching files
    image_files = sorted(input_path.glob(file_pattern))

    for img_file in image_files:
        output_file = output_path / img_file.name.replace(".jpg", ".png").replace(
            ".jpeg", ".png"
        )
        preprocessor.preprocess_receipt(str(img_file), str(output_file))

    return preprocessor.get_statistics()
