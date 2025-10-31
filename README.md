# DocuSpend - PaddleOCR-VL Receipt & Invoice Parsing

**DocuSpend** uses the **PaddleOCR-VL v3.3.0** model to extract key details from invoices, bills, and receipts, then organizes them into structured data for automated expense tracking and analytics. This project demonstrates multimodal OCR, expense classification, and real-time analytics deployment.

## 🎯 Project Overview

**Objective:** Initialize the DocuSpend project with PaddleOCR-VL for receipt and invoice parsing on consumer hardware (Google Colab Free with T4 GPU).

**Key Features:**
- Receipt and invoice OCR using PaddleOCR-VL (0.9B parameters, ultra-compact)
- Automatic field extraction (vendor, date, total, tax, line items)
- Expense categorization (groceries, dining, transport, utilities, etc.)
- Compatible with Google Colab Free (16GB VRAM T4 GPU)
- Preprocessing pipeline (deskew, grayscale, adaptive binarization, normalization)
- Fine-tuning ready with mixed precision training (FP16)

## 📋 Specification

This project implements the Phase 1 setup of the DocuSpend initiative:

- **Model:** PaddleOCR-VL v3.3.0 (Released October 16, 2024)
- **Dataset:** CORU (Comprehensive Post-OCR Parsing and Receipt Understanding) - 450 images, CC-BY-4.0
- **Hardware:** Google Colab Free (T4 GPU, 16GB VRAM)
- **Total Setup Size:** ~5GB (model + data + outputs)
- **Status:** ✅ Environment, data, and preprocessing ready

## 📁 Project Structure

```
docuspend/
├── data/                                   # Dataset and annotations
│   ├── raw/                               # Original images
│   │   ├── train/                         # Training split (320-400 images)
│   │   ├── val/                           # Validation split (40-50 images)
│   │   └── test/                          # Test split (40-50 images)
│   ├── processed/                         # Preprocessed images
│   │   ├── train/                         # Grayscale, deskewed, normalized
│   │   ├── val/
│   │   └── test/
│   ├── annotations/                       # DocuSpend schema JSON files
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── splits/                            # Train/val/test metadata
│   └── dataset_info.json                  # Dataset summary
│
├── models/                                 # Model weights and checkpoints
│   ├── pretrained/                        # PaddleOCR-VL weights (~3-4GB)
│   ├── checkpoints/                       # Fine-tuning checkpoints
│   └── final/                             # Final trained models
│
├── configs/                                # Configuration files
│   ├── paddleocr_vl_config.yaml          # Model config
│   ├── training_config.yaml              # Training hyperparameters
│   └── preprocessing_config.yaml         # Preprocessing pipeline config
│
├── src/                                   # Python source code
│   ├── preprocessing.py                  # Image preprocessing functions
│   ├── dataset.py                        # Dataset loaders and utilities
│   ├── annotation_converter.py           # CORU → DocuSpend converter
│   ├── train.py                          # Training utilities
│   └── inference.py                      # Inference and prediction
│
├── notebooks/                             # Jupyter notebooks (main workflow)
│   ├── 01_environment_setup.ipynb        # Install dependencies, verify GPU
│   ├── 02_data_download.ipynb            # Download CORU dataset
│   ├── 03_data_preparation.ipynb         # Split data, convert annotations
│   ├── 04_preprocessing.ipynb            # Preprocess images
│   ├── 05_fine_tuning.ipynb              # Setup fine-tuning pipeline
│   └── 06_inference_test.ipynb           # Run inference on test set
│
├── outputs/                               # Predictions and logs
│   ├── predictions/                      # Model predictions
│   ├── visualizations/                   # Debug visualizations
│   └── logs/                             # Training and preprocessing logs
│
├── requirements.txt                       # Python dependencies
├── verify_setup.py                       # Setup verification script
├── .gitignore                            # Git configuration
└── README.md                             # This file
```

## 🚀 Quick Start

### Prerequisites
- Google Colab (Free tier with GPU)
- OR: Local machine with CUDA 11.8+ and 16GB+ VRAM

### Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd docuspend
   ```

2. **Run the environment setup notebook:**
   - Open Jupyter: `jupyter notebook`
   - Go to `notebooks/01_environment_setup.ipynb`
   - Execute all cells to install dependencies

3. **Follow the notebook sequence:**
   - `01_environment_setup.ipynb` → Verify GPU and install dependencies
   - `02_data_download.ipynb` → Download CORU dataset (450 images)
   - `03_data_preparation.ipynb` → Split data (80/10/10) and convert annotations
   - `04_preprocessing.ipynb` → Preprocess images (grayscale, deskew, normalize)
   - `05_fine_tuning.ipynb` → Setup training pipeline
   - `06_inference_test.ipynb` → Run inference on test set

### Quick Verification

```bash
# Run setup verification
python verify_setup.py

# Expected output: ✅ SETUP VERIFICATION PASSED - READY FOR FINE-TUNING
```

## 📊 Annotation Schema

Each receipt/invoice has a corresponding JSON annotation in the **DocuSpend schema**:

```json
{
  "filename": "receipt_000001.jpg",
  "vendor": "Walmart Supercenter",
  "date": "2024-03-15",
  "total": 127.48,
  "tax": 10.20,
  "items": [
    {
      "description": "Organic Bananas",
      "quantity": 2,
      "unit_price": 1.99,
      "total": 3.98
    },
    {
      "description": "Whole Milk 1gal",
      "quantity": 1,
      "unit_price": 4.29,
      "total": 4.29
    }
  ],
  "category": "groceries"
}
```

**Valid categories:** groceries, dining, transport, utilities, shopping, healthcare, entertainment, other

## 🔧 Configuration Files

### `configs/paddleocr_vl_config.yaml`
Model configuration for PaddleOCR-VL inference:
- Device: GPU (T4/P100)
- Mixed precision: FP16 for efficiency
- Language: English (109 languages supported)

### `configs/training_config.yaml`
Fine-tuning hyperparameters:
- Batch size: 4 (with 4x gradient accumulation = effective 16)
- Learning rate: 1e-4 with linear decay
- Epochs: 10
- Mixed precision training: Enabled
- Optimizer: AdamW

### `configs/preprocessing_config.yaml`
Image preprocessing pipeline:
- Target DPI: 300
- Operations: Grayscale, Gaussian blur, deskew, adaptive threshold, morphology, padding
- Output format: PNG (lossless)

## 🖼️ Preprocessing Pipeline

The preprocessing module (`src/preprocessing.py`) applies the following steps:

1. **Load image** (BGR color)
2. **Grayscale conversion** (single channel for OCR)
3. **Noise removal** (Gaussian blur, kernel 5×5)
4. **Deskewing** (rotation correction with min angle threshold 0.5°)
5. **Adaptive thresholding** (binarization with local threshold)
6. **Morphological operations** (optional: closing with 2×2 kernel)
7. **Resolution normalization** (scale to 300 DPI)
8. **Border padding** (20px white border)
9. **Save as PNG** (lossless compression)

**Average processing time:** ~5-7 seconds per image

## 📚 Python Modules

### `src/preprocessing.py`
- `ReceiptPreprocessor` class for image preprocessing
- `preprocess_receipt()` for single images
- `batch_preprocess()` for directory of images

### `src/dataset.py`
- `ReceiptDataset` class for loading images and annotations
- `DatasetSplitter` for train/val/test splits
- `AnnotationValidator` for data quality checks

### `src/annotation_converter.py`
- `AnnotationConverter` for CORU → DocuSpend conversion
- Category inference from vendor names
- Handling of missing/invalid data

### `src/train.py`
- `TrainingConfig` for configuration management
- `TrainingSetup` for pipeline setup
- `TrainingLogger` for metrics logging

### `src/inference.py`
- `ReceiptOCRInference` for single-image prediction
- `BatchInference` for batch processing
- Field extraction and structured output

## 📈 Expected Output After Phase 1

After running all notebooks, you should have:

- ✅ **450 images** downloaded from CORU dataset (~500MB)
- ✅ **Split into** train (360), val (45), test (45)
- ✅ **450 preprocessed images** (grayscale PNG, 300 DPI)
- ✅ **450 JSON annotations** in DocuSpend schema
- ✅ **PaddleOCR-VL model** downloaded (~3-4GB)
- ✅ **Configuration files** for training
- ✅ **Preprocessing quality check** with before/after visualizations
- ✅ **Inference predictions** on test set

**Total disk usage:** ~5GB

## 🎓 Model Information

### PaddleOCR-VL v3.3.0
- **Released:** October 16, 2024
- **Parameters:** 0.9B (ultra-compact)
- **Architecture:** NaViT-style dynamic resolution encoder + ERNIE-4.5-0.3B language model
- **Languages:** 109 languages supported
- **Capabilities:** Text, tables, formulas, charts recognition
- **License:** Apache 2.0
- **Model size:** 3-4GB weights
- **Download:** Automatic via PaddleOCR package or manual from Hugging Face

### Alternative Models
- PP-OCRv4 (lighter, faster but less accurate)
- SOTA baselines available in PaddleOCR repo

## 💾 Hardware Compatibility

### Google Colab Free (Recommended for this project)
- **GPU:** Tesla T4 (16GB VRAM) or P100
- **RAM:** 12-13GB available
- **Storage:** 78GB temporary per session
- **Session duration:** 12 hours max

### Local Machine Requirements
- **GPU:** NVIDIA GPU with ≥16GB VRAM
- **CUDA:** 11.8 or higher
- **Python:** 3.9+
- **Storage:** 8GB minimum

### Hardware Constraints
- **Model loading:** 3-4GB VRAM
- **Batch size:** 4 images with FP16 mixed precision
- **Max preprocessed data in memory:** 500MB
- **Gradient accumulation:** 4 steps for effective batch size 16

## 🔄 Data Pipeline

```
CORU Dataset (Hugging Face)
    ↓
[02_data_download.ipynb]
    ↓
Raw Images (450 × ~1.1MB = 500MB)
    ↓
[03_data_preparation.ipynb]
    ↓
Train (360) / Val (45) / Test (45)
+ Annotations converted to DocuSpend schema
    ↓
[04_preprocessing.ipynb]
    ↓
Preprocessed Images (PNG, 300 DPI, normalized)
+ Before/after visualizations
    ↓
Ready for Fine-tuning
```

## 📝 Notebook Descriptions

| Notebook | Purpose | Duration | Output |
|----------|---------|----------|--------|
| 01_environment_setup | GPU verification, dependency installation, project structure | 10-15 min | ✅ Verification report |
| 02_data_download | Download CORU, subset to 450 images, validate data | 20-30 min | 450 raw images |
| 03_data_preparation | Split into train/val/test, convert annotations | 5-10 min | Splits + annotations |
| 04_preprocessing | Preprocess all images, generate visualizations | 30-45 min | Preprocessed PNG images |
| 05_fine_tuning | Setup training pipeline (no actual training yet) | 5-10 min | ✅ Training ready |
| 06_inference_test | Run inference on test set, generate predictions | 10-20 min | Test predictions |
| **Total** | **Complete Phase 1 setup** | **~90 minutes** | **5GB output** |

## 🔍 Verification

Run the verification script to ensure complete setup:

```bash
python verify_setup.py
```

This checks:
- ✅ Directory structure
- ✅ Configuration files
- ✅ Source code modules
- ✅ Jupyter notebooks
- ✅ Data files (if downloaded)
- ✅ Python dependencies

## 🚨 Troubleshooting

### GPU not detected in Colab
```python
# In notebook, run:
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### PaddleOCR installation fails
```bash
pip install --upgrade pip
pip install paddleocr paddlepaddle-gpu
```

### Out of memory during preprocessing
- Reduce batch size in `configs/preprocessing_config.yaml`
- Process splits separately instead of all at once

### Data download slow
- Use VPN/proxy if in restricted region
- Download manually from Hugging Face: https://huggingface.co/datasets/abdoelsayed/CORU

## 📖 Documentation

- **PaddleOCR Docs:** https://www.paddleocr.ai/
- **CORU Dataset:** https://huggingface.co/datasets/abdoelsayed/CORU
- **Google Colab:** https://colab.research.google.com/

## ✅ Success Criteria

This phase is **complete** when:

1. ✅ All 6 notebooks execute without errors
2. ✅ 450 images downloaded from CORU
3. ✅ Images split into train/val/test (360/45/45)
4. ✅ All images preprocessed to PNG (300 DPI)
5. ✅ 450 annotation JSON files in DocuSpend schema
6. ✅ PaddleOCR-VL model downloaded
7. ✅ Configuration files created
8. ✅ `verify_setup.py` returns "READY"

## 🎯 Next Phase

After Phase 1 completion, Phase 2 will implement:

- Actual fine-tuning loop with loss functions
- Custom metrics (CER, WER, field accuracy)
- Hyperparameter tuning experiments
- Model evaluation and comparison
- Production inference optimization

## 📄 License

- **Project:** Proprietary (DocuSpend)
- **Model (PaddleOCR-VL):** Apache 2.0
- **Dataset (CORU):** CC-BY-4.0

## 🤝 Contributing

Contributions welcome! Please follow:
1. Create feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am "Add feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Submit pull request

## 📧 Support

For issues or questions:
1. Check troubleshooting section above
2. Review notebook outputs for error messages
3. Run `verify_setup.py` to diagnose setup issues
4. Check PaddleOCR documentation

---

**Last updated:** October 31, 2025
**Status:** ✅ Phase 1 Complete - Ready for Fine-tuning
