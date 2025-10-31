"""
Inference Utilities for PaddleOCR-VL
DocuSpend Project - Receipt/Invoice Parsing

This module provides inference utilities for running PaddleOCR-VL on
receipt and invoice images, extracting structured data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ReceiptOCRInference:
    """Inference engine for receipt OCR using PaddleOCR-VL."""

    def __init__(self, model_name: str = "PaddleOCR-VL", lang: str = "en"):
        """
        Initialize inference engine.

        Args:
            model_name: Model name (default: PaddleOCR-VL)
            lang: Language code (default: en)
        """
        self.model_name = model_name
        self.lang = lang
        self.ocr = None
        self.model_loaded = False

        self._load_model()

    def _load_model(self):
        """Load PaddleOCR-VL model."""
        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(lang=self.lang, use_gpu=True)
            self.model_loaded = True
            logger.info(f"Loaded {self.model_name} model for language: {self.lang}")

        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load OCR model: {str(e)}")
            self.model_loaded = False

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with detected text and bounding boxes
        """
        if not self.model_loaded:
            logger.error("Model not loaded")
            return {}

        try:
            result = self.ocr.ocr(image_path, cls=True)

            # Process results
            processed = self._process_ocr_result(result)
            processed["image_path"] = str(image_path)
            processed["timestamp"] = datetime.now().isoformat()

            return processed

        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {str(e)}")
            return {}

    def _process_ocr_result(self, ocr_result: List) -> Dict[str, Any]:
        """
        Process raw OCR result to structured format.

        Args:
            ocr_result: Raw output from PaddleOCR

        Returns:
            Structured dictionary with text and bounding boxes
        """
        processed = {
            "detections": [],
            "text": "",
            "confidence": 0.0,
        }

        if not ocr_result or not ocr_result[0]:
            return processed

        confidences = []

        for line in ocr_result[0]:
            if len(line) >= 2:
                bbox, (text, confidence) = line[0], line[1]

                detection = {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": {
                        "x_min": float(min(p[0] for p in bbox)),
                        "y_min": float(min(p[1] for p in bbox)),
                        "x_max": float(max(p[0] for p in bbox)),
                        "y_max": float(max(p[1] for p in bbox)),
                    },
                }

                processed["detections"].append(detection)
                confidences.append(float(confidence))

        # Combine detected text
        processed["text"] = " ".join([d["text"] for d in processed["detections"]])

        # Average confidence
        if confidences:
            processed["confidence"] = sum(confidences) / len(confidences)

        return processed

    def extract_receipt_fields(
        self, image_path: str, annotation_template: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract structured receipt fields from image.

        Args:
            image_path: Path to receipt image
            annotation_template: Template annotation to fill (optional)

        Returns:
            Dictionary with extracted fields following DocuSpend schema
        """
        prediction = self.predict(image_path)

        if not prediction.get("detections"):
            return {
                "filename": Path(image_path).name,
                "vendor": None,
                "date": None,
                "total": None,
                "tax": None,
                "items": [],
                "category": "other",
                "confidence": 0.0,
            }

        # Extract fields (simplified heuristic approach)
        result = {
            "filename": Path(image_path).name,
            "vendor": self._extract_vendor(prediction),
            "date": self._extract_date(prediction),
            "total": self._extract_total(prediction),
            "tax": self._extract_tax(prediction),
            "items": self._extract_items(prediction),
            "category": "other",
            "confidence": prediction.get("confidence", 0.0),
            "raw_detections": prediction.get("detections", []),
        }

        return result

    def _extract_vendor(self, prediction: Dict[str, Any]) -> Optional[str]:
        """Extract vendor name (usually first line)."""
        if prediction.get("detections"):
            # Typically vendor is in the first few lines
            first_line = prediction["detections"][0]
            return first_line.get("text", "Unknown").strip()
        return None

    def _extract_date(self, prediction: Dict[str, Any]) -> Optional[str]:
        """Extract date from detected text."""
        import re

        text = prediction.get("text", "")

        # Look for date patterns
        patterns = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"\d{4}-\d{1,2}-\d{1,2}",
            r"\d{1,2}-\d{1,2}-\d{2,4}",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return None

    def _extract_total(self, prediction: Dict[str, Any]) -> Optional[float]:
        """Extract total amount from detected text."""
        import re

        text = prediction.get("text", "")

        # Look for patterns like "Total: $123.45" or similar
        patterns = [
            r"[Tt]otal[:\s]+\$?([\d,]+\.?\d*)",
            r"\$?([\d,]+\.\d{2})\s*$",  # Amount at end
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(",", "")
                try:
                    return float(amount_str)
                except ValueError:
                    pass

        return None

    def _extract_tax(self, prediction: Dict[str, Any]) -> Optional[float]:
        """Extract tax amount from detected text."""
        import re

        text = prediction.get("text", "")

        # Look for patterns like "Tax: $10.20"
        patterns = [
            r"[Tt]ax[:\s]+\$?([\d,]+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(",", "")
                try:
                    return float(amount_str)
                except ValueError:
                    pass

        return None

    def _extract_items(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract line items from detected text."""
        # This is a simplified heuristic. In production, would need more
        # sophisticated parsing or trained entity extractor.
        items = []

        for detection in prediction.get("detections", []):
            text = detection.get("text", "")

            # Look for patterns that might be line items
            # e.g., "Item name  2 @ $5.99 = $11.98"
            if text and len(text) > 3:
                # Very basic heuristic: if contains number + price
                import re
                price_pattern = r"\$?([\d]+\.?\d{2})"
                matches = re.findall(price_pattern, text)

                if matches and len(matches) >= 1:
                    # Assume first number is quantity/price
                    item = {
                        "description": text,
                        "quantity": 1,
                        "unit_price": 0.0,
                        "total": float(matches[0]) if matches else 0.0,
                    }
                    items.append(item)

        return items


class BatchInference:
    """Batch inference on multiple images."""

    def __init__(self, model_name: str = "PaddleOCR-VL", lang: str = "en"):
        """
        Initialize batch inference engine.

        Args:
            model_name: Model name
            lang: Language code
        """
        self.inference = ReceiptOCRInference(model_name, lang)
        self.results = []
        self.statistics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "average_confidence": 0.0,
        }

    def predict_batch(self, image_dir: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        Run inference on all images in directory.

        Args:
            image_dir: Directory containing images
            output_file: Optional file to save results as JSON

        Returns:
            List of prediction results
        """
        image_path = Path(image_dir)
        image_files = sorted(
            list(image_path.glob("*.jpg"))
            + list(image_path.glob("*.png"))
            + list(image_path.glob("*.jpeg"))
        )

        if not image_files:
            logger.warning(f"No images found in {image_dir}")
            return []

        results = []
        confidences = []

        for image_file in image_files:
            try:
                prediction = self.inference.extract_receipt_fields(str(image_file))
                results.append(prediction)
                self.statistics["successful"] += 1

                if prediction.get("confidence"):
                    confidences.append(prediction["confidence"])

                logger.info(f"Processed {image_file.name}")

            except Exception as e:
                logger.error(f"Failed to process {image_file}: {str(e)}")
                self.statistics["failed"] += 1

            finally:
                self.statistics["total_processed"] += 1

        # Calculate statistics
        if confidences:
            self.statistics["average_confidence"] = sum(confidences) / len(confidences)

        self.results = results

        # Save results if output file specified
        if output_file:
            self._save_results(output_file)

        return results

    def _save_results(self, output_file: str):
        """Save inference results to JSON file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output = {
                "timestamp": datetime.now().isoformat(),
                "model": self.inference.model_name,
                "total_results": len(self.results),
                "statistics": self.statistics,
                "results": self.results,
            }

            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)

            logger.info(f"Saved results to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get batch inference statistics."""
        return self.statistics


def run_inference_on_test_set(
    test_image_dir: str,
    output_dir: str,
    model_name: str = "PaddleOCR-VL",
) -> Dict[str, Any]:
    """
    Run inference on test set and save predictions.

    Args:
        test_image_dir: Directory with test images
        output_dir: Output directory for predictions
        model_name: Model name

    Returns:
        Dictionary with results and statistics
    """
    batch_inference = BatchInference(model_name)

    # Run inference
    results = batch_inference.predict_batch(
        test_image_dir,
        output_file=str(Path(output_dir) / "test_results.json"),
    )

    # Save individual predictions
    output_path = Path(output_dir) / "individual"
    output_path.mkdir(parents=True, exist_ok=True)

    for result in results:
        filename = Path(result["filename"]).stem
        output_file = output_path / f"{filename}.json"

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    return {
        "total_results": len(results),
        "statistics": batch_inference.get_statistics(),
        "output_dir": str(output_dir),
    }
