"""
Annotation Converter: CORU → DocuSpend Schema
DocuSpend Project - PaddleOCR-VL Fine-tuning

This module converts annotations from the CORU dataset format to the DocuSpend
annotation schema, handling field mapping, missing data, and validation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class AnnotationConverter:
    """Converter from CORU to DocuSpend annotation schema."""

    # Vendor keyword mappings for category inference
    CATEGORY_KEYWORDS = {
        "groceries": [
            "walmart",
            "costco",
            "kroger",
            "safeway",
            "whole foods",
            "trader joe",
            "sprouts",
            "target",
            "supermarket",
            "grocery",
            "market",
        ],
        "dining": [
            "starbucks",
            "mcdonald",
            "mcd",
            "pizza",
            "restaurant",
            "cafe",
            "coffee",
            "burger",
            "taco",
            "chipotle",
            "subway",
            "panera",
        ],
        "transport": [
            "shell",
            "chevron",
            "exxon",
            "mobil",
            "bp",
            "chevron",
            "gas",
            "fuel",
            "parking",
            "uber",
            "lyft",
            "transit",
        ],
        "utilities": [
            "electric",
            "water",
            "gas",
            "internet",
            "phone",
            "verizon",
            "att",
            "comcast",
            "utility",
        ],
        "shopping": [
            "amazon",
            "ebay",
            "bestbuy",
            "mall",
            "retail",
            "clothing",
            "apparel",
            "store",
        ],
        "healthcare": [
            "pharmacy",
            "cvs",
            "walgreens",
            "medical",
            "clinic",
            "doctor",
            "hospital",
            "health",
        ],
        "entertainment": [
            "movie",
            "theater",
            "cinema",
            "netflix",
            "spotify",
            "apple music",
            "events",
            "concert",
        ],
    }

    DOCUSPEND_CATEGORIES = [
        "groceries",
        "dining",
        "transport",
        "utilities",
        "shopping",
        "healthcare",
        "entertainment",
        "other",
    ]

    def __init__(self):
        """Initialize converter."""
        self.stats = {
            "total_converted": 0,
            "successful": 0,
            "failed": 0,
            "missing_fields": {},
            "inferred_categories": {},
        }

    def convert_coru_to_docuspend(
        self, coru_annotation: Dict[str, Any], filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Convert CORU annotation to DocuSpend schema.

        Args:
            coru_annotation: Original CORU annotation dict
            filename: Image filename

        Returns:
            dict: Annotation in DocuSpend schema, or None if critical error

        DocuSpend schema:
        {
            "filename": "receipt_001.jpg",
            "vendor": "Walmart",
            "date": "2024-03-15",
            "total": 127.48,
            "tax": 10.20,
            "items": [
                {
                    "description": "Organic Bananas",
                    "quantity": 2,
                    "unit_price": 1.99,
                    "total": 3.98
                }
            ],
            "category": "groceries"
        }
        """
        self.stats["total_converted"] += 1

        try:
            docuspend = {"filename": filename}

            # Convert vendor
            vendor = self._extract_vendor(coru_annotation)
            if not vendor:
                logger.warning(f"Missing vendor for {filename}")
                self.stats["missing_fields"]["vendor"] = (
                    self.stats["missing_fields"].get("vendor", 0) + 1
                )
                vendor = "Unknown"
            docuspend["vendor"] = vendor

            # Convert date
            date = self._extract_date(coru_annotation)
            if not date:
                logger.warning(f"Missing/invalid date for {filename}")
                self.stats["missing_fields"]["date"] = (
                    self.stats["missing_fields"].get("date", 0) + 1
                )
                date = None
            docuspend["date"] = date

            # Convert total
            total = self._extract_float(coru_annotation, "total")
            if total is None:
                logger.warning(f"Missing total for {filename}")
                self.stats["missing_fields"]["total"] = (
                    self.stats["missing_fields"].get("total", 0) + 1
                )
                total = 0.0
            elif total < 0:
                logger.error(f"Negative total for {filename}: {total}")
                self.stats["failed"] += 1
                return None
            docuspend["total"] = total

            # Convert tax
            tax = self._extract_float(coru_annotation, "tax")
            if tax is None:
                tax = 0.0
            else:
                # Validate tax is not greater than total
                if tax > total:
                    logger.warning(
                        f"Tax {tax} exceeds total {total} for {filename}, setting to 0"
                    )
                    tax = 0.0
            docuspend["tax"] = tax

            # Convert items
            items = self._extract_items(coru_annotation)
            docuspend["items"] = items

            # Infer category
            category = self._infer_category(vendor, coru_annotation)
            docuspend["category"] = category
            self.stats["inferred_categories"][category] = (
                self.stats["inferred_categories"].get(category, 0) + 1
            )

            self.stats["successful"] += 1
            return docuspend

        except Exception as e:
            logger.error(f"Failed to convert annotation for {filename}: {str(e)}")
            self.stats["failed"] += 1
            return None

    def _extract_vendor(self, annotation: Dict[str, Any]) -> Optional[str]:
        """Extract and clean vendor name from annotation."""
        # Try multiple possible keys
        for key in ["vendor", "vendor_name", "company", "shop", "store"]:
            if key in annotation and annotation[key]:
                vendor = str(annotation[key]).strip()
                # Clean common artifacts
                vendor = vendor.title()
                # Remove email addresses
                if "@" in vendor:
                    vendor = vendor.split("@")[0]
                return vendor

        return None

    def _extract_date(self, annotation: Dict[str, Any]) -> Optional[str]:
        """Extract and normalize date to YYYY-MM-DD format."""
        # Try multiple possible keys
        for key in ["date", "purchase_date", "transaction_date", "date_time"]:
            if key in annotation and annotation[key]:
                date_str = str(annotation[key]).strip()

                # Try to parse various date formats
                parsed = self._parse_date(date_str)
                if parsed:
                    return parsed

        return None

    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse date string to YYYY-MM-DD format.

        Handles common formats:
        - YYYY-MM-DD
        - MM/DD/YYYY
        - DD/MM/YYYY
        - YYYY/MM/DD
        - ISO format with time
        """
        import re

        # Remove time portion if present
        if "T" in date_str or " " in date_str:
            date_str = date_str.split("T")[0].split(" ")[0]

        # Already in YYYY-MM-DD format
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str

        # Try MM/DD/YYYY format
        match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
        if match:
            month, day, year = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # Try DD/MM/YYYY format (less common in US, but try)
        match = re.match(r"(\d{2})/(\d{2})/(\d{4})", date_str)
        if match:
            first, second, year = match.groups()
            # Assume MM/DD if first <= 12
            if int(first) <= 12:
                return f"{year}-{int(first):02d}-{int(second):02d}"
            else:
                # Must be DD/MM
                return f"{year}-{int(second):02d}-{int(first):02d}"

        # Try YYYY/MM/DD format
        match = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

        return None

    def _extract_float(
        self, annotation: Dict[str, Any], field: str
    ) -> Optional[float]:
        """Extract and validate float value from annotation."""
        if field not in annotation or annotation[field] is None:
            return None

        try:
            value = annotation[field]
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Remove currency symbols and clean
                value = re.sub(r"[^\d.-]", "", value)
                if value:
                    return float(value)
        except (ValueError, TypeError):
            pass

        return None

    def _extract_items(self, annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract line items from annotation."""
        items = []

        # Try multiple possible keys
        for key in ["items", "line_items", "products", "goods"]:
            if key in annotation and annotation[key]:
                items_raw = annotation[key]
                if isinstance(items_raw, list):
                    for item_raw in items_raw:
                        item = self._convert_item(item_raw)
                        if item:
                            items.append(item)
                    break

        return items

    def _convert_item(self, item_raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert single line item to DocuSpend format."""
        try:
            item = {}

            # Description
            for key in ["description", "name", "product", "product_name"]:
                if key in item_raw and item_raw[key]:
                    item["description"] = str(item_raw[key]).strip()
                    break

            if "description" not in item:
                item["description"] = "Item"

            # Quantity
            quantity = self._extract_int(item_raw, ["quantity", "qty", "count"])
            item["quantity"] = quantity if quantity is not None else 1

            # Unit price
            unit_price = self._extract_float(item_raw, "unit_price")
            if unit_price is None:
                unit_price = self._extract_float(item_raw, "price")
            item["unit_price"] = unit_price if unit_price is not None else 0.0

            # Total
            total = self._extract_float(item_raw, "total")
            if total is None:
                # Calculate from quantity and unit_price
                total = item["quantity"] * item["unit_price"]
            item["total"] = total

            return item

        except Exception as e:
            logger.warning(f"Failed to convert item: {str(e)}")
            return None

    def _extract_int(
        self, data: Dict[str, Any], keys: List[str]
    ) -> Optional[int]:
        """Extract integer value from data with multiple key options."""
        for key in keys:
            if key in data and data[key] is not None:
                try:
                    if isinstance(data[key], int):
                        return data[key]
                    if isinstance(data[key], str):
                        return int(re.sub(r"[^\d]", "", data[key]))
                    return int(data[key])
                except (ValueError, TypeError):
                    continue
        return None

    def _infer_category(
        self, vendor: str, annotation: Dict[str, Any]
    ) -> str:
        """Infer expense category from vendor name or metadata."""
        if not vendor or vendor == "Unknown":
            # Check if annotation has category metadata
            if "category" in annotation and annotation["category"]:
                return str(annotation["category"]).lower()
            return "other"

        vendor_lower = vendor.lower()

        # Check against keyword mappings
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in vendor_lower:
                    return category

        # Check if annotation provides category
        if "category" in annotation and annotation["category"]:
            category = str(annotation["category"]).lower()
            if category in self.DOCUSPEND_CATEGORIES:
                return category

        return "other"

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        return self.stats

    def convert_batch(
        self, coru_annotations: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert batch of CORU annotations.

        Args:
            coru_annotations: Dict of {filename: annotation_data}

        Returns:
            Dict of {filename: docuspend_annotation}
        """
        converted = {}

        for filename, annotation in coru_annotations.items():
            docuspend = self.convert_coru_to_docuspend(annotation, filename)
            if docuspend:
                converted[filename] = docuspend

        return converted


def convert_coru_to_docuspend(
    coru_annotation: Dict[str, Any], filename: str
) -> Optional[Dict[str, Any]]:
    """
    Convert single CORU annotation to DocuSpend schema (functional API).

    Args:
        coru_annotation: Original CORU annotation dict
        filename: Image filename

    Returns:
        dict: Annotation in DocuSpend schema, or None if critical error
    """
    converter = AnnotationConverter()
    return converter.convert_coru_to_docuspend(coru_annotation, filename)


def load_and_convert_annotations(
    coru_annotations_file: str, output_dir: str
) -> Dict[str, Any]:
    """
    Load CORU annotations from JSON file and convert to DocuSpend schema.

    Args:
        coru_annotations_file: Path to CORU annotations JSON
        output_dir: Directory to save converted annotations

    Returns:
        Dictionary with conversion statistics
    """
    converter = AnnotationConverter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(coru_annotations_file, "r") as f:
            coru_annotations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load CORU annotations: {str(e)}")
        return converter.get_statistics()

    # Convert each annotation
    for filename, annotation in coru_annotations.items():
        docuspend = converter.convert_coru_to_docuspend(annotation, filename)

        if docuspend:
            # Save converted annotation
            output_file = (
                output_path
                / f"{Path(filename).stem}.json"
            )
            try:
                with open(output_file, "w") as f:
                    json.dump(docuspend, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save annotation {output_file}: {str(e)}")

    return converter.get_statistics()
