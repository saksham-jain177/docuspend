# Phase 2: Fine-Tuning & Validation Pipeline Setup

## Overview

This phase sets up a complete fine-tuning infrastructure for **PaddleOCR-VL** (0.9B parameter vision-language model) on the **DocuSpend dataset** for multilingual document parsing.

**Target:** Colab Free GPU (T4/P100, 16GB VRAM)
**Expected Duration:** 20-30 minutes per 10 epochs (up to ~150 minutes for 30 epochs)
**Outputs:** Fine-tuned checkpoint, configs, logs, metric visualizations

---

## Prerequisites

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- PaddleOCR v2.9+
- Required packages: `paddlepaddle`, `paddleocr`, `albumentations`, `matplotlib`, `pyyaml`

### In Colab
```python
# Install dependencies
!pip install paddlepaddle paddleocr albumentations matplotlib pyyaml

# Clone PaddleOCR repo (for tools/train.py)
!git clone https://github.com/PaddlePaddle/PaddleOCR.git

# Mount Google Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')

# Set environment for GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### Dataset Requirements
- Phase 1 curated DocuSpend dataset (images + annotations)
- Expected structure: `{dataset_root}/{vendor}/{language}/` with JPG/PNG images
- Annotation format: COCO JSON, PaddleOCR txt, or custom JSON
- Minimum: 500 detection samples, balanced across vendors and languages

---

## Step 1: Prepare Dataset

### 1.1 Convert Phase 1 Dataset to PaddleOCR Format

The Phase 1 dataset needs to be converted to PaddleOCR's annotation format.

```bash
python docuspend/scripts/prepare_dataset.py \
  --dataset-root /path/to/phase1/dataset \
  --output-dir docuspend/dataset/ \
  --validate
```

**What it does:**
- Reads images and annotations from Phase 1 dataset
- Converts to PaddleOCR format: `image_path\t[{"text": "...", "bbox": [[...]]}, ...]`
- Extracts vendor and language labels from directory structure
- Validates all image paths and bounding boxes
- Outputs: `docuspend/dataset/raw_annotations.txt`

**Output example:**
```
images/invoice/en/001.jpg	[{"text": "Invoice", "bbox": [[10,10], [50,10], [50,30], [10,30]]}]
images/receipt/zh/002.jpg	[{"text": "金额", "bbox": [[100,100], [150,100], [150,130], [100,130]]}]
```

### 1.2 Split Dataset into Stratified Train/Val

Ensure vendor and language balance across train/val splits.

```bash
python docuspend/scripts/split_dataset.py \
  --annotations docuspend/dataset/raw_annotations.txt \
  --output-dir docuspend/dataset/ \
  --train-ratio 0.8 \
  --seed 42
```

**What it does:**
- Loads raw annotations
- Groups by (vendor, language) pair
- Stratified split: 80% train, 20% validation
- Ensures every vendor-language combination in both splits
- Generates reports with distribution statistics

**Output files:**
- `docuspend/dataset/train_list.txt` (80% of data)
- `docuspend/dataset/val_list.txt` (20% of data)
- `docuspend/dataset/SPLIT_REPORT.md` (distribution report)

**Example SPLIT_REPORT.md:**
```
## Overview
- Total samples: 1,000
- Train samples: 800 (80%)
- Val samples: 200 (20%)

## Vendor Distribution
| Vendor | Total | Train | Val |
|--------|-------|-------|-----|
| invoice | 400 | 320 | 80 |
| receipt | 600 | 480 | 120 |
```

---

## Step 2: Download Pretrained Model

Download the PaddleOCR-VL pretrained checkpoint:

```bash
# Option 1: From HuggingFace
from transformers import AutoModel
model = AutoModel.from_pretrained("PaddlePaddle/PaddleOCR-VL")

# Option 2: From PaddleOCR repo
# Automatic when training starts (if configured in YAML)
```

**Save to:** `./paddleocr_vl_checkpoint/` (specify in train.py)

---

## Step 3: Configure Training

### 3.1 Review Configuration

The main training configuration is in `docuspend/configs/paddleocr_vl_finetune.yaml`.

**Key parameters to verify:**
- `Global.epoch_num: 30` (max epochs)
- `Global.use_amp: True` (mixed precision for memory efficiency)
- `Train.loader.batch_size_per_card: 8` (Colab limit)
- `Optimizer.lr.learning_rate: 0.001` (initial learning rate)
- `Optimizer.lr.warmup_epoch: 2` (warmup for stability)

### 3.2 Data Augmentations

Augmentations are applied during training to improve robustness:

| Augmentation | Limit | Probability | Purpose |
|--------------|-------|-------------|---------|
| Rotation | ±15° | 30% | Document skew |
| Gaussian Blur | σ=0.5–3.0 | 20% | Focus/motion blur |
| Contrast | ±30% | 30% | Variable lighting |
| Noise | var≤100 | 20% | Scanning artifacts |
| ShiftScaleRotate | ±10%/±20% | 30% | Perspective variations |

---

## Step 4: Launch Training

### 4.1 Basic Training (Local or Colab)

```bash
python docuspend/scripts/train.py \
  --config docuspend/configs/paddleocr_vl_finetune.yaml \
  --pretrained-model ./paddleocr_vl_checkpoint/ \
  --output-dir ./output/ \
  --sync-drive
```

**Arguments:**
- `--config`: Path to training YAML config
- `--pretrained-model`: Path to pretrained checkpoint
- `--output-dir`: Where to save checkpoints
- `--sync-drive`: Enable Google Drive sync (Colab only)
- `--no-export`: Skip exporting to inference format

### 4.2 What Happens During Training

```
[Epoch 1/30]
  Batch 10/100: loss=3.42, f1_det=0.45, acc_rec=0.32, lr=0.0008
  Batch 20/100: loss=3.28, f1_det=0.48, acc_rec=0.35, lr=0.0008
  ...
  [Validation] loss=3.12, f1_det=0.48, acc_rec=0.36, combined=0.42
  ✓ Saved checkpoint: output/epoch_0001.pth

[Epoch 2/30] ...
```

**Monitoring:**
- Loss curves should decrease smoothly (sign of good training)
- F1 and accuracy metrics should improve or plateau
- If GPU OOM occurs, batch size is reduced automatically
- All metrics logged to `docuspend/logs/training_log.txt`

---

## Step 5: Training Outputs

### 5.1 Checkpoints

Saved in `./output/` directory:

```
output/
├── epoch_0001.pth      # Epoch 1 checkpoint
├── epoch_0005.pth      # Epoch 5 checkpoint
├── epoch_0010.pth      # Epoch 10 checkpoint
├── best_accuracy/      # Best checkpoint (by combined score)
│   ├── model_state.pth
│   └── optimizer_state.pth
└── final.pth           # Final epoch checkpoint
```

**Best checkpoint selection:**
```
combined_score = 0.5 * F1_detection + 0.5 * accuracy_recognition
best_checkpoint = argmax(combined_score over all epochs)
```

### 5.2 Metrics & Logs

Located in `docuspend/logs/`:

```
logs/
├── training_log.txt         # Full training log (text)
├── metrics.csv              # Per-epoch summary
│   Columns: epoch, avg_train_loss, avg_train_f1_det, avg_train_acc_rec,
│           val_loss, val_f1_det, val_acc_rec, combined_score, best_epoch
├── raw_metrics.csv          # Per-batch metrics (optional)
├── loss_curve.png           # Loss visualization
├── f1_curve.png             # F1 score curve
├── accuracy_curve.png       # Recognition accuracy curve
├── combined_score_curve.png # Combined metric visualization
├── lr_schedule.png          # Learning rate schedule
└── sync_report.txt          # Google Drive sync report
```

### 5.3 Metrics Explanation

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **F1 Detection** | 0–1 | Text region detection quality |
| **Accuracy Recognition** | 0–1 | Character-level OCR accuracy |
| **Combined Score** | 0–1 | Balanced metric (0.5 F1 + 0.5 acc) |
| **Loss** | 0–∞ | Should decrease over epochs |
| **Training Loss** | — | Loss on training set |
| **Validation Loss** | — | Loss on validation set |

**Expected values (after fine-tuning):**
- F1 Detection: ~0.65–0.75
- Recognition Accuracy: ~0.65–0.75
- Combined Score: ~0.65–0.75
- Training Loss: ~2.0–3.0 → decreases over epochs
- Validation Loss: ~2.0–3.5 (should not increase too much)

---

## Step 6: Post-Training Steps

### 6.1 Metric Visualization (Automatic)

If using `train.py`, metrics are visualized automatically after training:

```bash
python docuspend/scripts/visualize_metrics.py \
  --metrics-csv docuspend/logs/metrics.csv \
  --output-dir docuspend/logs/
```

**Generates:**
- Loss curve showing training vs validation loss
- F1 score improvement over epochs
- Recognition accuracy curves
- Combined score with best epoch highlighted
- Learning rate schedule

### 6.2 Export to Inference Format (Automatic)

Best checkpoint is exported for Phase 3 inference:

```bash
python docuspend/scripts/export_checkpoint.py \
  --checkpoint ./output/best_accuracy/ \
  --output-dir ./inference/
```

**Output:**
```
inference/
├── det_model/
│   ├── inference.pdmodel        # Model structure
│   ├── inference.pdiparams      # Model weights
│   └── inference.pdiparams.info # Parameter info
├── export_metadata.json
└── INFERENCE_GUIDE.md           # Usage guide for Phase 3
```

### 6.3 Google Drive Sync (Automatic if --sync-drive)

Checkpoints and logs are synced to Google Drive:

```bash
python docuspend/scripts/sync_to_drive.py \
  --checkpoint-dir ./output/ \
  --logs-dir docuspend/logs/
```

**Synced to:** `/My Drive/docuspend_checkpoints/`

```
docuspend_checkpoints/
├── epoch_*.pth          # Epoch checkpoints
├── best_model/          # Best checkpoint
├── final.pth            # Final checkpoint
├── metrics.csv          # Metrics file
├── raw_metrics.csv      # Raw metrics
├── training_log.txt     # Training log
├── *.png                # All plots
├── TRAINING_SUMMARY.md  # Summary report
└── sync_report.txt      # Sync report
```

---

## Step 7: Training Summary

After training, a summary report is generated:

**File:** `docuspend/TRAINING_SUMMARY.md`

Contains:
- Training configuration (model, parameters, GPU info)
- Dataset statistics (total samples, vendor/language distribution)
- Best model performance metrics
- Training duration
- List of generated artifacts
- Command to reproduce
- Recommendations for Phase 3

**Example:**
```markdown
# PaddleOCR-VL Fine-Tuning Summary

## Best Model Performance
- **Epoch:** 15 (out of 30)
- **Detection F1 score:** 0.72
- **Recognition accuracy:** 0.68
- **Combined score:** 0.70
- **Validation loss:** 2.35

## Training Duration
- **Total time:** 2 hours 15 minutes
- **Time per epoch:** 4.5 minutes
```

---

## Troubleshooting

### Issue: GPU Out-of-Memory (OOM)

**Symptoms:** `CUDA out of memory` error on epoch X

**Solutions:**
1. **Reduce batch size** (in YAML): `batch_size_per_card: 4`
2. **Reduce gradient accumulation steps** or remove
3. **Use gradient checkpointing** (if supported)
4. **Reduce image resolution** (if applicable)

### Issue: Training Loss Not Decreasing

**Symptoms:** Loss plateaus or increases

**Possible causes:**
1. Learning rate too high → reduce to 0.0005 or 0.0001
2. Learning rate too low → increase to 0.002
3. Dataset quality issues → review raw data
4. Insufficient training → run more epochs

**Fix:** Adjust `Optimizer.lr.learning_rate` in YAML and restart

### Issue: Validation Metrics Don't Improve

**Symptoms:** F1 and accuracy stay constant

**Possible causes:**
1. Model is overfitting to training data
2. Dataset is too small for fine-tuning
3. Augmentations are too aggressive
4. Pretrained model initialization issue

**Fix:**
1. Add more data augmentation or reduce probability
2. Increase dropout or use regularization
3. Train for more epochs with early stopping

### Issue: Google Drive Sync Fails

**Symptoms:** `Google Drive not mounted` error

**Fix:** Run in Colab cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Then retry training with `--sync-drive` flag.

### Issue: Export Fails

**Symptoms:** `tools/export_model.py not found`

**Fix:** Ensure PaddleOCR repo is available:
```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

---

## Next Steps (Phase 3)

### Ready for Phase 3 When:
1. ✅ Best checkpoint saved: `/models/checkpoints/best_accuracy/`
2. ✅ Inference format exported: `inference/det_model/`
3. ✅ Training summary generated: `docuspend/TRAINING_SUMMARY.md`
4. ✅ F1 and accuracy validated on test set

### Phase 3 Will Use:
- Best fine-tuned checkpoint for document parsing
- Inference format for deployment
- Metrics as baseline for performance tracking
- Configs for reproducibility

---

## File Structure

```
docuspend/
├── README.md                              # Project overview
├── PHASE_2_README.md                      # This file
├── TRAINING_SUMMARY.md                    # Generated after training
├── configs/
│   ├── paddleocr_vl_finetune.yaml         # Training config
│   └── finetune_metadata.json             # Metadata template
├── dataset/
│   ├── train_list.txt                     # Train split (80%)
│   ├── val_list.txt                       # Val split (20%)
│   ├── raw_annotations.txt                # Raw converted annotations
│   └── SPLIT_REPORT.md                    # Split report
├── scripts/
│   ├── prepare_dataset.py                 # Convert Phase 1 → PaddleOCR format
│   ├── split_dataset.py                   # Stratified train/val split
│   ├── train.py                           # Main training entrypoint
│   ├── visualize_metrics.py               # Generate metric plots
│   ├── sync_to_drive.py                   # Google Drive sync
│   └── export_checkpoint.py               # Export to inference format
├── logs/
│   ├── training_log.txt                   # Training log
│   ├── metrics.csv                        # Per-epoch metrics
│   ├── raw_metrics.csv                    # Per-batch metrics
│   ├── loss_curve.png                     # Visualizations
│   ├── f1_curve.png
│   ├── accuracy_curve.png
│   ├── combined_score_curve.png
│   ├── lr_schedule.png
│   └── sync_report.txt                    # Drive sync report
└── models/
    └── checkpoints/
        ├── epoch_0001.pth                 # Epoch checkpoints
        ├── epoch_0005.pth
        ├── best_accuracy/                 # Best checkpoint
        └── final.pth                      # Final checkpoint

inference/                                  # Exported inference model
├── det_model/
│   ├── inference.pdmodel
│   ├── inference.pdiparams
│   └── inference.pdiparams.info
├── export_metadata.json
└── INFERENCE_GUIDE.md
```

---

## Performance Benchmarks

### Expected Training Performance (Colab Free GPU)

| Metric | Value |
|--------|-------|
| **Time per epoch** | 4–5 minutes |
| **Total training time (30 epochs)** | 120–150 minutes (~2–2.5 hours) |
| **GPU memory usage** | ~14–15 GB (T4 GPU) |
| **Final F1 score** | ~0.70–0.75 |
| **Final accuracy** | ~0.68–0.75 |
| **Inference latency** | ~500–1000ms per image |

### Factors Affecting Performance

- **Dataset size:** Larger datasets may need longer training
- **Image resolution:** Higher resolution = slower training
- **Augmentation intensity:** More augmentations increase training time
- **GPU type:** P100 > T4 (slight performance difference)

---

## References

- [PaddleOCR Official Docs](https://www.paddleocr.ai/)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR-VL Paper](https://arxiv.org/abs/2510.14528)
- [PaddlePaddle Docs](https://www.paddlepaddle.org.cn/)

---

## Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Review `docuspend/logs/training_log.txt` for error details
3. Verify dataset format matches requirements
4. Check GPU memory with `nvidia-smi` (in Colab)

---

**Phase 2 Status:** ✅ Ready for fine-tuning
**Last Updated:** 2024-10-31
**Document Version:** 1.0
