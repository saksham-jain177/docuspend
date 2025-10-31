#!/usr/bin/env python3
"""
DocuSpend Setup Verification Script

This script performs comprehensive verification that the DocuSpend project
is properly set up and ready for fine-tuning. Run this after all notebooks.

Usage:
    python verify_setup.py
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime


def verify_directory_structure(base_dir: Path) -> tuple[bool, list[str]]:
    """Verify complete directory structure exists."""
    required_dirs = [
        "data/raw/train", "data/raw/val", "data/raw/test",
        "data/processed/train", "data/processed/val", "data/processed/test",
        "data/annotations/train", "data/annotations/val", "data/annotations/test",
        "data/splits",
        "models/pretrained", "models/checkpoints", "models/final",
        "configs",
        "outputs/predictions/individual", "outputs/visualizations", "outputs/logs/tensorboard",
        "notebooks", "src"
    ]

    missing = []
    for directory in required_dirs:
        dir_path = base_dir / directory
        if not dir_path.exists():
            missing.append(directory)

    return len(missing) == 0, missing


def verify_configuration_files(base_dir: Path) -> tuple[bool, list[str]]:
    """Verify all configuration YAML files exist."""
    required_configs = [
        "configs/paddleocr_vl_config.yaml",
        "configs/training_config.yaml",
        "configs/preprocessing_config.yaml",
    ]

    missing = []
    for config_file in required_configs:
        file_path = base_dir / config_file
        if not file_path.exists():
            missing.append(config_file)

    return len(missing) == 0, missing


def verify_source_files(base_dir: Path) -> tuple[bool, list[str]]:
    """Verify all Python source files exist."""
    required_src = [
        "src/preprocessing.py",
        "src/dataset.py",
        "src/annotation_converter.py",
        "src/train.py",
        "src/inference.py",
    ]

    missing = []
    for src_file in required_src:
        file_path = base_dir / src_file
        if not file_path.exists():
            missing.append(src_file)

    return len(missing) == 0, missing


def verify_notebooks(base_dir: Path) -> tuple[bool, list[str]]:
    """Verify all Jupyter notebooks exist."""
    required_notebooks = [
        "notebooks/01_environment_setup.ipynb",
        "notebooks/02_data_download.ipynb",
        "notebooks/03_data_preparation.ipynb",
        "notebooks/04_preprocessing.ipynb",
        "notebooks/05_fine_tuning.ipynb",
        "notebooks/06_inference_test.ipynb",
    ]

    missing = []
    for notebook in required_notebooks:
        file_path = base_dir / notebook
        if not file_path.exists():
            missing.append(notebook)

    return len(missing) == 0, missing


def verify_data_files(base_dir: Path) -> dict:
    """Verify data files are present and valid."""
    stats = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "train_annotations": 0,
        "val_annotations": 0,
        "test_annotations": 0,
        "preprocessed_images": 0,
    }

    # Count raw images
    for split in ["train", "val", "test"]:
        raw_dir = base_dir / f"data/raw/{split}"
        if raw_dir.exists():
            count = len(list(raw_dir.glob("*.jpg")))
            stats[f"{split}_images"] = count
            stats["total_images"] += count

    # Count annotations
    for split in ["train", "val", "test"]:
        ann_dir = base_dir / f"data/annotations/{split}"
        if ann_dir.exists():
            count = len(list(ann_dir.glob("*.json")))
            stats[f"{split}_annotations"] = count

    # Count preprocessed images
    processed_total = 0
    for split in ["train", "val", "test"]:
        processed_dir = base_dir / f"data/processed/{split}"
        if processed_dir.exists():
            count = len(list(processed_dir.glob("*.png")))
            processed_total += count
    stats["preprocessed_images"] = processed_total

    return stats


def verify_dependencies() -> dict:
    """Verify Python dependencies are installed."""
    dependencies = {
        "paddleocr": False,
        "cv2": False,
        "numpy": False,
        "torch": False,
        "datasets": False,
        "yaml": False,
    }

    for package_name, import_name in [
        ("paddleocr", "paddleocr"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("yaml", "yaml"),
    ]:
        try:
            __import__(import_name)
            dependencies[package_name] = True
        except ImportError:
            pass

    return dependencies


def verify_setup():
    """Execute comprehensive verification."""
    base_dir = Path(".")
    if not (base_dir / "docuspend").exists():
        base_dir = base_dir / "docuspend"
    if not base_dir.name == "docuspend":
        # Assume we're in docuspend directory
        pass

    print("\n" + "="*70)
    print("DOCUSPEND PROJECT SETUP VERIFICATION")
    print("="*70 + "\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "base_directory": str(base_dir),
        "checks": {},
        "data": {},
        "dependencies": {},
        "overall_status": "READY",
    }

    # 1. Directory structure
    print("1. Verifying directory structure...")
    dirs_ok, missing_dirs = verify_directory_structure(base_dir)
    results["checks"]["directories"] = {
        "status": "✅ PASS" if dirs_ok else "❌ FAIL",
        "missing": missing_dirs
    }
    print(f"   {'✅' if dirs_ok else '❌'} Directory structure: {len(missing_dirs)} missing")
    if missing_dirs:
        for d in missing_dirs[:5]:
            print(f"      - {d}")
        results["overall_status"] = "INCOMPLETE"

    # 2. Configuration files
    print("\n2. Verifying configuration files...")
    configs_ok, missing_configs = verify_configuration_files(base_dir)
    results["checks"]["configurations"] = {
        "status": "✅ PASS" if configs_ok else "❌ FAIL",
        "missing": missing_configs
    }
    print(f"   {'✅' if configs_ok else '❌'} Config files: {len(missing_configs)} missing")
    if missing_configs:
        for c in missing_configs:
            print(f"      - {c}")
        results["overall_status"] = "INCOMPLETE"

    # 3. Source code
    print("\n3. Verifying source code...")
    src_ok, missing_src = verify_source_files(base_dir)
    results["checks"]["source_code"] = {
        "status": "✅ PASS" if src_ok else "❌ FAIL",
        "missing": missing_src
    }
    print(f"   {'✅' if src_ok else '❌'} Source files: {len(missing_src)} missing")
    if missing_src:
        for s in missing_src:
            print(f"      - {s}")
        results["overall_status"] = "INCOMPLETE"

    # 4. Notebooks
    print("\n4. Verifying Jupyter notebooks...")
    nb_ok, missing_nb = verify_notebooks(base_dir)
    results["checks"]["notebooks"] = {
        "status": "✅ PASS" if nb_ok else "❌ FAIL",
        "missing": missing_nb
    }
    print(f"   {'✅' if nb_ok else '❌'} Notebooks: {len(missing_nb)} missing")
    if missing_nb:
        for n in missing_nb:
            print(f"      - {n}")
        results["overall_status"] = "INCOMPLETE"

    # 5. Data files
    print("\n5. Verifying data files...")
    data_stats = verify_data_files(base_dir)
    results["data"] = data_stats
    print(f"   Total images: {data_stats['total_images']}")
    print(f"      Train: {data_stats['train_images']} images, {data_stats['train_annotations']} annotations")
    print(f"      Val: {data_stats['val_images']} images, {data_stats['val_annotations']} annotations")
    print(f"      Test: {data_stats['test_images']} images, {data_stats['test_annotations']} annotations")
    print(f"   Preprocessed images: {data_stats['preprocessed_images']}")

    if data_stats['total_images'] == 0:
        print("   ⚠️  No data found yet - Run notebooks 02-04 to download and preprocess data")
        results["overall_status"] = "INCOMPLETE"
    elif (data_stats['train_images'] + data_stats['val_images'] + data_stats['test_images']) < 350:
        print("   ⚠️  Insufficient data - Some notebooks may not have completed")
        results["overall_status"] = "INCOMPLETE"

    # 6. Dependencies
    print("\n6. Verifying Python dependencies...")
    deps = verify_dependencies()
    results["dependencies"] = deps

    installed = sum(1 for v in deps.values() if v)
    for pkg, is_installed in deps.items():
        print(f"   {'✅' if is_installed else '❌'} {pkg}")

    if installed < len(deps):
        print(f"   ⚠️  Missing {len(deps) - installed} dependencies - Run: pip install -r requirements.txt")
        results["overall_status"] = "INCOMPLETE"

    # Final status
    print("\n" + "="*70)
    if results["overall_status"] == "READY":
        print("✅ SETUP VERIFICATION PASSED - READY FOR FINE-TUNING")
        print("="*70)
        print("\nNext steps:")
        print("  1. If data not present, run: python -m jupyter notebook notebooks/02_data_download.ipynb")
        print("  2. If data exists, proceed to fine-tuning phase")
        print("  3. Or run: jupyter notebook notebooks/05_fine_tuning.ipynb")
    else:
        print("❌ SETUP INCOMPLETE - MISSING COMPONENTS")
        print("="*70)
        print("\nMissing components:")
        if missing_dirs:
            print(f"  - Directories: {len(missing_dirs)} missing")
        if missing_configs:
            print(f"  - Config files: {len(missing_configs)} missing")
        if missing_src:
            print(f"  - Source files: {len(missing_src)} missing")
        if missing_nb:
            print(f"  - Notebooks: {len(missing_nb)} missing")
        if installed < len(deps):
            print(f"  - Dependencies: {len(deps) - installed} missing")

    print("\n" + "="*70 + "\n")

    # Save verification report
    report_file = base_dir / "outputs/logs/setup_verification.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Verification report saved to: {report_file}\n")

    return results["overall_status"] == "READY"


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
