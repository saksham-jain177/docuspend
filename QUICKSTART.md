# DocuSpend Quick Start Guide

**Get started with DocuSpend in 5 minutes!**

---

## 🚀 Quick Start (Google Colab)

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Open Notebook"
3. Select "GitHub" tab
4. Paste: `https://github.com/docuspend/docuspend.git`
5. Or manually upload this project

### Step 2: Run Setup Notebook
```bash
# In Colab, upload or navigate to:
notebooks/01_environment_setup.ipynb

# Execute all cells (10-15 minutes)
# Verifies: GPU, installs dependencies, creates folders
```

### Step 3: Download Data
```bash
# Run next notebook:
notebooks/02_data_download.ipynb

# Downloads 450 receipt images from CORU dataset
# Time: 20-30 minutes
```

### Step 4: Prepare Data
```bash
# Run:
notebooks/03_data_preparation.ipynb

# Splits data (80/10/10), converts annotations
# Time: 5-10 minutes
```

### Step 5: Preprocess Images
```bash
# Run:
notebooks/04_preprocessing.ipynb

# Preprocesses all 450 images to 300 DPI PNG
# Generates before/after visualizations
# Time: 30-45 minutes
```

### Step 6: Verify & Test
```bash
# Run:
notebooks/05_fine_tuning.ipynb  # Setup training pipeline
notebooks/06_inference_test.ipynb  # Run test inference
```

**Total Time: ~90 minutes**

---

## 💻 Quick Start (Local Machine)

### Prerequisites
```bash
# GPU with ≥16GB VRAM
# CUDA 11.8+ (for nvidia-gpu)
# Python 3.9+
# ~8GB free storage
```

### Installation
```bash
# Clone repository
git clone https://github.com/docuspend/docuspend.git
cd docuspend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

### Run Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open and run notebooks in order:
# 1. notebooks/01_environment_setup.ipynb
# 2. notebooks/02_data_download.ipynb
# ... (continue as above)
```

---

## 🔍 Quick Verification

### Check Setup
```bash
python verify_setup.py
```

**Expected output:**
```
✅ SETUP VERIFICATION PASSED - READY FOR FINE-TUNING
```

### Check GPU (Colab)
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024} GB")
```

### Check Dependencies
```python
from paddleocr import PaddleOCR
import cv2
print(f"PaddleOCR: ✓")
print(f"OpenCV: {cv2.__version__}")
```

---

## 📊 Understanding the Data

### Annotation Format
```json
{
  "filename": "receipt_000001.jpg",
  "vendor": "Walmart",
  "date": "2024-03-15",
  "total": 127.48,
  "tax": 10.20,
  "items": [
    {"description": "Bananas", "quantity": 2, "unit_price": 1.99, "total": 3.98}
  ],
  "category": "groceries"
}
```

### Data Structure
```
data/
├── raw/              # Original images
│   ├── train/ (360)
│   ├── val/ (45)
│   └── test/ (45)
├── processed/        # Preprocessed PNG images
├── annotations/      # DocuSpend JSON files
└── dataset_info.json # Summary
```

---

## 🛠️ Configuration Files

### Quick Configuration Changes

#### Preprocessing (if images are too large/small)
**File:** `configs/preprocessing_config.yaml`
```yaml
preprocessing:
  target_dpi: 300  # Change to 150 or 600
  gaussian_blur_kernel: [5, 5]  # Adjust blur strength
```

#### Training (if running out of memory)
**File:** `configs/training_config.yaml`
```yaml
training:
  batch_size: 4  # Reduce to 2 if out of memory
  gradient_accumulation_steps: 4  # Reduce to 2
```

#### Model (if running slower than expected)
**File:** `configs/paddleocr_vl_config.yaml`
```yaml
inference:
  device: "gpu"  # Or "cpu" for testing
  use_mix_precision: true  # Disable if issues
```

---

## 🚨 Troubleshooting

### Problem: GPU not detected
**Solution:**
```python
# In Colab, enable GPU:
# Runtime → Change runtime type → GPU
# Then restart
import torch
assert torch.cuda.is_available()
```

### Problem: PaddleOCR installation fails
**Solution:**
```bash
pip install --upgrade pip
pip install paddlepaddle-gpu paddleocr --no-cache-dir
```

### Problem: Out of memory during preprocessing
**Solution:**
1. Reduce batch_size in preprocessing_config.yaml
2. Process train/val/test separately
3. Or reduce target_dpi to 150

### Problem: Slow data download
**Solution:**
```python
# In 02_data_download.ipynb, use manual download:
# 1. Download from: https://huggingface.co/datasets/abdoelsayed/CORU
# 2. Extract to data/raw/all/
# 3. Continue with notebook
```

### Problem: Jupyter kernel crashes
**Solution:**
```bash
# Restart kernel and run cells one by one
# Or reduce batch sizes and clear GPU cache:
import torch
torch.cuda.empty_cache()
```

---

## 📚 Key Classes & Functions

### Image Preprocessing
```python
from src.preprocessing import ReceiptPreprocessor, preprocess_receipt
import yaml

# Load config
with open('configs/preprocessing_config.yaml') as f:
    config = yaml.safe_load(f)

# Single image
preprocess_receipt('input.jpg', 'output.png', config)

# Batch
preprocessor = ReceiptPreprocessor(config)
preprocessor.preprocess_receipt('input.jpg', 'output.png')
```

### Dataset Loading
```python
from src.dataset import ReceiptDataset, AnnotationValidator

# Load dataset
dataset = ReceiptDataset(
    'data/processed/train',
    'data/annotations/train'
)

# Get sample
image, annotation = dataset[0]

# Validate
stats = AnnotationValidator.validate_dataset('data/annotations/train')
print(f"Valid: {stats['valid']}/{stats['total_files']}")
```

### Annotation Conversion
```python
from src.annotation_converter import AnnotationConverter

converter = AnnotationConverter()
docuspend_ann = converter.convert_coru_to_docuspend(
    coru_annotation,
    'receipt_001.jpg'
)
```

### Inference
```python
from src.inference import ReceiptOCRInference, BatchInference

# Single image
ocr = ReceiptOCRInference()
prediction = ocr.predict('receipt.jpg')

# Batch
batch = BatchInference()
results = batch.predict_batch('data/processed/test')
```

---

## 📈 Expected Results

### After 01 Environment Setup
- ✅ GPU detected
- ✅ Dependencies installed
- ✅ Project folders created

### After 02 Data Download
- ✅ 450 images downloaded (~500MB)
- ✅ All images validated
- ✅ Dataset info created

### After 03 Data Preparation
- ✅ Data split (360/45/45)
- ✅ Annotations converted
- ✅ DocuSpend schema verified

### After 04 Preprocessing
- ✅ 450 PNG images (300 DPI)
- ✅ Before/after visualizations
- ✅ Preprocessing statistics

### After 05 Fine-Tuning Setup
- ✅ Training config verified
- ✅ Device info collected
- ✅ Ready for training

### After 06 Inference Test
- ✅ Test predictions generated
- ✅ Evaluation metrics calculated
- ✅ Results saved to JSON

---

## 📞 Getting Help

1. **Check README.md** - Comprehensive documentation
2. **Run verify_setup.py** - Diagnose issues
3. **Check notebook outputs** - Error messages
4. **Review configs** - Configuration issues
5. **Read docstrings** - Function documentation

---

## 🎯 Next Steps

### Phase 1 Complete? Then:
1. ✅ Verify all data is processed: `python verify_setup.py`
2. ✅ Check outputs folder for visualizations
3. ✅ Review test predictions

### Ready for Phase 2? Then:
1. Proceed to fine-tuning with actual training loop
2. Run hyperparameter experiments
3. Evaluate on test set
4. Export final model

### Want to Deploy? Then:
1. Export model as ONNX/TorchScript
2. Create REST API
3. Deploy to production
4. Monitor performance

---

## 💡 Pro Tips

1. **Save to Google Drive (Colab):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Copy results after each notebook
   ```

2. **Monitor GPU Memory:**
   ```python
   import torch
   torch.cuda.reset_peak_memory_stats()
   # ... run code ...
   print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
   ```

3. **Restart Between Notebooks:**
   - Clear GPU cache: `torch.cuda.empty_cache()`
   - Or restart Jupyter kernel

4. **Check Progress:**
   - Look at `outputs/logs/preprocessing_stats.json`
   - Check `data/splits/split_summary.json`
   - Review visualizations in `outputs/visualizations/`

---

## 📋 Checklist

- [ ] GPU available with 16GB+ VRAM
- [ ] Python 3.9+ installed
- [ ] 8GB+ free storage space
- [ ] Project cloned/downloaded
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Verification passed (`python verify_setup.py`)
- [ ] Notebook 01 executed successfully
- [ ] Notebook 02 completed (data downloaded)
- [ ] Notebook 03 completed (data split)
- [ ] Notebook 04 completed (images preprocessed)
- [ ] Notebook 05 completed (training ready)
- [ ] Notebook 06 completed (inference tested)
- [ ] Outputs verified
- [ ] Ready for next phase

---

**Ready to get started? Run Notebook 01!** 🚀

For detailed documentation, see [README.md](README.md)
For complete deliverables, see [DELIVERABLES.md](DELIVERABLES.md)
