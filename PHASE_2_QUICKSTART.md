# Phase 2: Quick Start Guide

## Complete Training Pipeline in 5 Steps

### Step 1: Setup Environment (Colab)

```python
# In Colab notebook
!pip install paddlepaddle paddleocr albumentations matplotlib pyyaml
!git clone https://github.com/PaddlePaddle/PaddleOCR.git

from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### Step 2: Prepare Dataset

```bash
# Convert Phase 1 dataset to PaddleOCR format
python docuspend/scripts/prepare_dataset.py \
  --dataset-root /path/to/phase1/dataset \
  --output-dir docuspend/dataset/

# Split into stratified train/val (80/20)
python docuspend/scripts/split_dataset.py \
  --annotations docuspend/dataset/raw_annotations.txt \
  --output-dir docuspend/dataset/
```

**Output:**
- ✅ `docuspend/dataset/train_list.txt` (80%)
- ✅ `docuspend/dataset/val_list.txt` (20%)
- ✅ `docuspend/dataset/SPLIT_REPORT.md`

### Step 3: Download Pretrained Model

```python
# Download PaddleOCR-VL from HuggingFace
from transformers import AutoModel
model = AutoModel.from_pretrained("PaddlePaddle/PaddleOCR-VL")
# Save to: ./paddleocr_vl_checkpoint/
```

### Step 4: Launch Training

```bash
python docuspend/scripts/train.py \
  --config docuspend/configs/paddleocr_vl_finetune.yaml \
  --pretrained-model ./paddleocr_vl_checkpoint/ \
  --output-dir ./output/ \
  --sync-drive
```

**Expected duration:** ~2.5 hours (30 epochs × 5 min/epoch)

**What happens:**
1. ✅ Loads dataset and validates
2. ✅ Trains for up to 30 epochs with data augmentation
3. ✅ Saves checkpoints every 5 epochs
4. ✅ Tracks best model by combined metric (0.5 F1 + 0.5 accuracy)
5. ✅ Generates metric visualizations
6. ✅ Syncs to Google Drive
7. ✅ Exports best checkpoint to inference format
8. ✅ Creates training summary report

### Step 5: Verify Training Outputs

```bash
# Check generated files
ls -lh docuspend/logs/        # Training logs and plots
ls -lh ./output/              # Checkpoints
ls -lh ./inference/           # Inference format

# View metrics
cat docuspend/logs/metrics.csv
cat docuspend/TRAINING_SUMMARY.md
```

---

## Key Outputs

| File | Purpose |
|------|---------|
| `./output/best_accuracy/` | ⭐ Best model checkpoint |
| `./inference/det_model/` | ⭐ Ready for Phase 3 inference |
| `docuspend/logs/metrics.csv` | Training metrics per epoch |
| `docuspend/logs/*.png` | 5 visualization plots |
| `docuspend/TRAINING_SUMMARY.md` | Complete summary report |

---

## Configuration (Optional)

To adjust training parameters, edit `docuspend/configs/paddleocr_vl_finetune.yaml`:

```yaml
Global:
  epoch_num: 30              # Total epochs (max 30)
  use_amp: True              # Mixed precision (keep True for Colab)

Optimizer:
  lr:
    learning_rate: 0.001     # Initial learning rate
    warmup_epoch: 2          # Warmup epochs

Train:
  loader:
    batch_size_per_card: 8   # Batch size (max 8 for Colab Free)
```

---

## Expected Results

| Metric | Expected Value |
|--------|-----------------|
| **Detection F1** | 0.70–0.75 |
| **Recognition Accuracy** | 0.68–0.75 |
| **Combined Score** | 0.68–0.75 |
| **Training Time** | 120–150 minutes |
| **GPU Memory** | ~14–15 GB |

---

## Troubleshooting

### GPU Out-of-Memory
**Fix:** Reduce batch size in YAML to 4 or 2

### Training Loss Not Decreasing
**Fix:** Adjust learning rate (try 0.0005 or 0.002)

### Drive Sync Fails
**Fix:** Run `drive.mount('/content/drive')` in Colab first

### Export Fails
**Fix:** Ensure PaddleOCR repo cloned: `!git clone https://github.com/PaddlePaddle/PaddleOCR.git`

---

## Complete Command Reference

```bash
# 1. Prepare dataset
python docuspend/scripts/prepare_dataset.py --dataset-root <PATH> --output-dir docuspend/dataset/

# 2. Split dataset
python docuspend/scripts/split_dataset.py --annotations docuspend/dataset/raw_annotations.txt --output-dir docuspend/dataset/

# 3. Train
python docuspend/scripts/train.py --config docuspend/configs/paddleocr_vl_finetune.yaml --pretrained-model <CHECKPOINT> --output-dir ./output/ --sync-drive

# 4. Visualize metrics (if needed)
python docuspend/scripts/visualize_metrics.py --metrics-csv docuspend/logs/metrics.csv --output-dir docuspend/logs/

# 5. Export checkpoint (if needed)
python docuspend/scripts/export_checkpoint.py --checkpoint ./output/best_accuracy/ --output-dir ./inference/

# 6. Sync to Google Drive (if needed)
python docuspend/scripts/sync_to_drive.py --checkpoint-dir ./output/ --logs-dir docuspend/logs/
```

---

## Next Phase (Phase 3)

Ready to proceed when:
- ✅ Best model at `./output/best_accuracy/`
- ✅ Inference model at `./inference/det_model/`
- ✅ Metrics in `docuspend/TRAINING_SUMMARY.md`

Use best model for production inference in Phase 3.

---

**Quick Start Version:** 1.0
