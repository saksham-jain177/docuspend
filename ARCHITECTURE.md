# DocuSpend Architecture & Design Document

**System Architecture for Receipt & Invoice OCR with PaddleOCR-VL**

---

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DocuSpend Project (Phase 1)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              INPUT: Raw Receipt/Invoice Images           │   │
│  │  (CORU Dataset: 450 images × ~1.1MB = ~500MB)           │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         DATA PIPELINE: Download, Split, Prepare         │   │
│  │  ├─ 02_data_download.ipynb (Download CORU dataset)      │   │
│  │  ├─ 03_data_preparation.ipynb (80/10/10 split)         │   │
│  │  └─ Annotation: CORU → DocuSpend schema conversion      │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │       PREPROCESSING: Image Enhancement Pipeline          │   │
│  │  ├─ Grayscale conversion                                │   │
│  │  ├─ Gaussian blur (noise removal)                       │   │
│  │  ├─ Deskewing (rotation correction)                     │   │
│  │  ├─ Adaptive thresholding (binarization)               │   │
│  │  ├─ Morphological operations (optional)                │   │
│  │  ├─ Resolution normalization (300 DPI)                 │   │
│  │  ├─ Border padding (20px white)                        │   │
│  │  └─ Output: PNG lossless format                         │   │
│  │                                                          │   │
│  │  Implemented: src/preprocessing.py                      │   │
│  │  Configured: configs/preprocessing_config.yaml          │   │
│  │  Orchestrated: 04_preprocessing.ipynb                   │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │      OUTPUT: Preprocessed Dataset Ready for Training    │   │
│  │  ├─ Train: 360 images (data/processed/train/)           │   │
│  │  ├─ Val: 45 images (data/processed/val/)                │   │
│  │  ├─ Test: 45 images (data/processed/test/)              │   │
│  │  └─ Annotations: DocuSpend JSON schema                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

                    ▼ PHASE 2 (Future)
        Fine-tuning with PaddleOCR-VL Model
        └─ Training loop, evaluation, optimization
```

---

## 🏗️ Component Architecture

### 1. Data Layer (`src/dataset.py`)
```
┌─────────────────────────────────────┐
│         ReceiptDataset              │
│ ┌───────────────────────────────┐   │
│ │ __init__(image_dir, annot_dir)│   │ Load images & annotations
│ │ __len__()                     │   │ Get dataset size
│ │ __getitem__(idx)              │   │ Get single sample
│ │ get_statistics()              │   │ Compute stats
│ └───────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│       DatasetSplitter               │
│ ┌───────────────────────────────┐   │
│ │ split_dataset()               │   │ Random split
│ │ stratified_split()            │   │ Stratified split
│ └───────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      AnnotationValidator            │
│ ┌───────────────────────────────┐   │
│ │ validate_annotation()         │   │ Validate single
│ │ validate_dataset()            │   │ Validate batch
│ └───────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 2. Preprocessing Layer (`src/preprocessing.py`)
```
┌──────────────────────────────────────────────────┐
│         ReceiptPreprocessor Class                │
├──────────────────────────────────────────────────┤
│                                                  │
│  Input: Raw Receipt Image (JPG, any DPI)        │
│          │                                      │
│          ▼                                      │
│  [Step 1] Load Image (BGR)                      │
│          │                                      │
│          ▼                                      │
│  [Step 2] Grayscale Conversion                  │
│          │                                      │
│          ▼                                      │
│  [Step 3] Gaussian Blur (5×5 kernel)            │
│          │                                      │
│          ▼                                      │
│  [Step 4] Deskew (rotation correction)          │
│          │                                      │
│          ▼                                      │
│  [Step 5] Adaptive Thresholding (block=11)      │
│          │                                      │
│          ▼                                      │
│  [Step 6] Morphological Close (2×2 kernel)      │
│          │                                      │
│          ▼                                      │
│  [Step 7] DPI Normalization (to 300 DPI)        │
│          │                                      │
│          ▼                                      │
│  [Step 8] Border Padding (20px white)           │
│          │                                      │
│          ▼                                      │
│  [Step 9] Save as PNG (lossless)                │
│          │                                      │
│          ▼                                      │
│  Output: Preprocessed Receipt (PNG, 300 DPI)   │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 3. Annotation Conversion Layer (`src/annotation_converter.py`)
```
┌─────────────────────────────────────────┐
│    AnnotationConverter Class             │
├─────────────────────────────────────────┤
│                                         │
│  Input: CORU Annotation JSON            │
│         {vendor, date, total, ...}      │
│                                         │
│         ▼                               │
│  [Extract Fields]                       │
│  ├─ Vendor: Clean & title case         │
│  ├─ Date: Parse (MM/DD/YYYY, etc.)     │
│  ├─ Total: Parse float                 │
│  ├─ Tax: Parse float, default 0        │
│  ├─ Items: Extract line items          │
│  └─ Category: Infer from vendor        │
│         │                               │
│         ▼                               │
│  [Validation]                           │
│  ├─ Check required fields               │
│  ├─ Validate data types                 │
│  ├─ Check constraints (tax < total)     │
│  └─ Log errors/warnings                │
│         │                               │
│         ▼                               │
│  Output: DocuSpend Annotation JSON      │
│  {filename, vendor, date, total, ...}   │
│                                         │
└─────────────────────────────────────────┘
```

### 4. Training Configuration Layer (`src/train.py`)
```
┌────────────────────────────────────────┐
│      TrainingConfig Class               │
├────────────────────────────────────────┤
│ • Load YAML configuration               │
│ • Validate settings                     │
│ • Export as dictionary                  │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│      TrainingSetup Class                │
├────────────────────────────────────────┤
│ • Create directories                    │
│ • Setup logging                         │
│ • Get device info (GPU)                 │
│ • Verify complete setup                 │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│      TrainingLogger Class               │
├────────────────────────────────────────┤
│ • Log epoch metrics                     │
│ • Log batch metrics                     │
│ • Generate summary                      │
│ • Save to JSON                          │
└────────────────────────────────────────┘
```

### 5. Inference Layer (`src/inference.py`)
```
┌──────────────────────────────────────────┐
│    ReceiptOCRInference Class             │
├──────────────────────────────────────────┤
│                                          │
│  Input: Receipt Image (JPG/PNG)          │
│         │                                │
│         ▼                                │
│  [Load Model]                            │
│  └─ PaddleOCR-VL v3.3.0 (0.9B params)   │
│         │                                │
│         ▼                                │
│  [Run Inference]                         │
│  └─ OCR detection + recognition          │
│         │                                │
│         ▼                                │
│  [Extract Fields]                        │
│  ├─ Vendor: First line                   │
│  ├─ Date: Regex pattern                  │
│  ├─ Total: Currency pattern              │
│  ├─ Tax: "Tax:" pattern                  │
│  └─ Items: Heuristic parsing             │
│         │                                │
│         ▼                                │
│  Output: Structured JSON                 │
│  {vendor, date, total, tax, items, ...}  │
│                                          │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│     BatchInference Class                 │
├──────────────────────────────────────────┤
│ • Process directory of images            │
│ • Collect statistics                     │
│ • Save results to JSON                   │
└──────────────────────────────────────────┘
```

---

## 💾 Data Flow Diagram

```
┌─────────────────────┐
│   CORU Dataset      │
│  (Hugging Face)     │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │ 02_Download  │  450 images
    │   Notebook   │  ~500MB
    └──────┬───────┘
           │
           ▼
    ┌─────────────────────────┐
    │  data/raw/all/          │
    │  ├─ receipt_000001.jpg  │
    │  ├─ receipt_000002.jpg  │
    │  └─ ...x448 more        │
    └──────┬───────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │ 03_Prepare Notebook  │
    │  (Split 80/10/10)    │
    └──────┬───────────────┘
           │
    ┌──────┴──────┬──────────┐
    │             │          │
    ▼             ▼          ▼
 Train        Val        Test
 360img       45img      45img
    │             │          │
    └─────┬───────┴──────┬───┘
          │              │
          ▼              ▼
    data/raw/         data/annotations/
    ├─train/          ├─train/ (JSON)
    ├─val/            ├─val/   (JSON)
    └─test/           └─test/  (JSON)
    │
    ▼
┌──────────────────────┐
│ 04_Preprocess        │
│    Notebook          │
└──────┬───────────────┘
       │
    ┌──┴──┬──────┐
    │     │      │
    ▼     ▼      ▼
 Train  Val    Test
 (PNG) (PNG)  (PNG)
    │
    └─► data/processed/
        ├─train/
        ├─val/
        └─test/
    │
    ▼
┌──────────────────────┐
│ Ready for Fine-tune  │
│  (Phase 2)           │
└──────────────────────┘
```

---

## 🔧 Configuration Architecture

```
configs/
│
├─ paddleocr_vl_config.yaml
│  ├─ Model information
│  │  ├─ Name: PaddleOCR-VL
│  │  ├─ Version: 3.3.0
│  │  └─ Parameters: 0.9B
│  ├─ Inference settings
│  │  ├─ Device: GPU
│  │  ├─ Mixed precision: FP16
│  │  └─ Batch size: 1
│  └─ Download & cache
│     └─ Location: ~/.paddleocr/
│
├─ training_config.yaml
│  ├─ Training parameters
│  │  ├─ Epochs: 10
│  │  ├─ Batch size: 4
│  │  └─ Gradient accumulation: 4
│  ├─ Optimization
│  │  ├─ Optimizer: AdamW
│  │  ├─ Learning rate: 1e-4
│  │  └─ Scheduler: Linear decay
│  ├─ Hardware
│  │  ├─ Device: CUDA
│  │  ├─ Max VRAM: 16GB
│  │  └─ FP16: Enabled
│  └─ Checkpointing
│     ├─ Save every: 50 steps
│     └─ Keep best: 3 models
│
└─ preprocessing_config.yaml
   ├─ Pipeline parameters
   │  ├─ Target DPI: 300
   │  ├─ Gaussian blur: 5×5
   │  ├─ Threshold block: 11
   │  ├─ Morph kernel: 2×2
   │  └─ Padding: 20px
   └─ Processing settings
      ├─ Format: PNG
      ├─ Workers: 2
      └─ Batch: 10
```

---

## 📊 Data Schema

### Input: CORU Format (Example)
```json
{
  "vendor_name": "WALMART SUPERCENTER",
  "date": "2024-03-15",
  "total_amount": 127.48,
  "tax_amount": 10.20,
  "line_items": [
    {
      "product_name": "Organic Bananas",
      "qty": 2,
      "price": 1.99
    }
  ]
}
```

### Output: DocuSpend Format (After Conversion)
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
    }
  ],
  "category": "groceries"
}
```

---

## 🎯 Design Principles

### 1. Modularity
- **Separation of Concerns:** Each module handles one responsibility
  - `preprocessing.py` → Image processing
  - `dataset.py` → Data loading
  - `annotation_converter.py` → Schema conversion
  - `train.py` → Training setup
  - `inference.py` → Model predictions

### 2. Configuration-Driven
- **All parameters in YAML files** (not hardcoded)
- Easy to adjust without code changes
- Reproducibility across runs

### 3. Error Handling
- **Comprehensive error checking** at each step
- **Logging for debugging** all operations
- **Graceful degradation** (skip bad samples, continue)

### 4. Extensibility
- **Class-based design** for easy inheritance
- **Functional APIs** for quick scripts
- **Configuration system** for custom pipelines

### 5. Validation
- **Data quality checks** throughout
- **Annotation validation** with detailed reports
- **Setup verification** script

---

## 🔄 Execution Workflow

```
Phase 1: Environment & Preparation (CURRENT)
│
├─ Step 1: Setup Environment
│  └─ 01_environment_setup.ipynb
│     ├─ Check GPU
│     ├─ Install deps
│     └─ Create folders
│
├─ Step 2: Download Data
│  └─ 02_data_download.ipynb
│     ├─ Load CORU
│     ├─ Subset 450 imgs
│     └─ Validate
│
├─ Step 3: Prepare Data
│  └─ 03_data_preparation.ipynb
│     ├─ Split 80/10/10
│     ├─ Convert schema
│     └─ Validate annot
│
├─ Step 4: Preprocess
│  └─ 04_preprocessing.ipynb
│     ├─ 9-step pipeline
│     ├─ Generate viz
│     └─ Collect stats
│
└─ Step 5: Verify & Setup
   ├─ 05_fine_tuning.ipynb (setup only)
   └─ 06_inference_test.ipynb (baseline)

             ▼
        PHASE 1 COMPLETE

             ▼

Phase 2: Fine-Tuning (Future)
│
├─ Training loop
├─ Evaluation
├─ Hyperparameter tuning
└─ Model export

             ▼

Phase 3: Deployment (Future)
│
├─ API development
├─ Inference optimization
└─ Production monitoring
```

---

## 💡 Key Design Decisions

### 1. PaddleOCR-VL v3.3.0
**Why:**
- Ultra-compact (0.9B parameters)
- Fits in 16GB VRAM with FP16
- 109 language support
- SOTA performance on document parsing

### 2. CORU Dataset
**Why:**
- Comprehensive annotations
- CC-BY-4.0 license (free to use)
- Recent (June 2024)
- Well-maintained on Hugging Face

### 3. 9-Step Preprocessing Pipeline
**Why:**
- Handles various receipt conditions
- Deskewing: Fixes tilted scans
- Adaptive thresholding: Works with different lighting
- DPI normalization: Consistent input
- PNG output: Lossless compression

### 4. DocuSpend Schema
**Why:**
- Comprehensive (vendor, date, total, tax, items, category)
- Extensible for future fields
- JSON format (universal compatibility)
- Supports missing data gracefully

### 5. Configuration-Driven Approach
**Why:**
- No code changes for tuning
- Reproducibility
- Easy experimentation
- Non-technical users can adjust

### 6. Notebook-Based Workflow
**Why:**
- Interactive development in Colab
- Step-by-step progress visible
- Easy debugging
- Familiar for data scientists

---

## 📈 Performance Considerations

### Memory Usage (Google Colab T4)
```
GPU Memory Budget: 16GB
├─ System reserved: ~2GB
├─ Available for compute: ~14GB
│  ├─ Model (FP16): ~2GB
│  ├─ Batch (4 imgs): ~1-2GB
│  ├─ Optimizer state: ~2GB
│  └─ Workspace: ~4-6GB
└─ Effective batch size: 16 (with grad accum)
```

### Processing Speed (Average)
```
Preprocessing: 5-7 sec/image
Inference (pretrained): 3-5 sec/image
Training (FP16): ~1-2 sec/batch
```

### Storage Requirements
```
Data: ~1GB (raw + processed)
Model: 3-4GB (PaddleOCR-VL)
Checkpoints: 1-2GB (Phase 2)
Total: ~5-8GB
```

---

## 🔐 Security & Best Practices

### Data Privacy
- ✅ Local processing (no cloud uploads)
- ✅ Optional: Google Drive for backup
- ✅ Sensitive info never logged

### Code Quality
- ✅ PEP 8 compliance
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Error handling & logging

### Reproducibility
- ✅ Fixed random seeds (42)
- ✅ Configuration versioning
- ✅ Dependency pinning (requirements.txt)
- ✅ Checkpoint saving

---

## 🔮 Future Extensions

### Phase 2: Fine-Tuning
- Custom loss functions for structured output
- Multi-task learning (OCR + classification)
- Hyperparameter optimization
- Model evaluation framework

### Phase 3: Deployment
- REST API (FastAPI)
- Batch processing service
- Web interface
- Monitoring & analytics

### Phase 4: Enhancement
- Multi-language support
- Handwritten receipt handling
- Real-time processing
- Mobile deployment

---

**Architecture Document**
**Created:** October 31, 2025
**Version:** 1.0
**Status:** ✅ Complete
