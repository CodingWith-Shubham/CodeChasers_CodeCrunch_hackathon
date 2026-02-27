# Offroad Semantic Scene Segmentation
### Duality AI Hackathon — Segmentation Track | Team CodeChasers

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Model Architecture](#-model-architecture)
3. [Results](#-results)
4. [Project Structure](#-project-structure)
5. [Environment Setup](#-environment-setup)
6. [Dataset Setup](#-dataset-setup)
7. [Training](#-training)
8. [Testing & Inference](#-testing--inference)
9. [Visualizing Predictions](#-visualizing-predictions)
10. [Configuration Reference](#-configuration-reference)
11. [Troubleshooting](#-troubleshooting)
12. [Class Legend](#-class-legend)

---

## Project Overview

This project implements a **semantic segmentation pipeline** for off-road desert environments, developed for the Duality AI Offroad Autonomy Segmentation Hackathon.

The goal is to classify every pixel in a desert scene into one of **10 semantic classes** — enabling Unmanned Ground Vehicles (UGVs) to understand their surroundings for safe path planning and obstacle avoidance.

### Key Highlights

- **DINOv2 ViT-B/14** as a frozen feature extractor — pre-trained on 142M images, providing powerful domain-invariant features
- **ASPP (Atrous Spatial Pyramid Pooling)** decoder for multi-scale context understanding
- **Combined Focal + Dice Loss** to handle severe class imbalance (rare classes like Logs and Rocks)
- **Data Augmentation** — flips, rotation, color jitter, random crops for generalization to unseen environments
- **Per-class loss weighting** — rare classes receive up to 5× higher gradient emphasis
- Achieves **Val IoU ~0.47+** with ViT-S, expected **~0.55+ with ViT-B**

---

## Model Architecture

```
Input Image (3 × H × W)
        ↓
DINOv2 ViT-B/14 Backbone (FROZEN)
        ↓
Patch Tokens (B × N × 768)
        ↓
Reshape to spatial grid (B × 768 × tokenH × tokenW)
        ↓
Stem Conv (768 → 256, GroupNorm, GELU)
        ↓
ASPP Module (Multi-scale context)
  ├── 1×1 Conv (rate=1)
  ├── 3×3 Dilated Conv (rate=6)
  ├── 3×3 Dilated Conv (rate=12)
  ├── 3×3 Dilated Conv (rate=18)
  └── Global Average Pool
        ↓
Decoder (256 → 128 → 128, GroupNorm, GELU)
        ↓
Classifier Conv (128 → 10 classes)
        ↓
Bilinear Upsample to original resolution
        ↓
Segmentation Map (H × W)
```

### Why DINOv2?
DINOv2 is a self-supervised Vision Transformer trained on 142M diverse images. Its features generalize exceptionally well to novel environments — critical for this hackathon where the test images come from a **different desert location** than the training data. By freezing the backbone, we avoid overfitting and only train the lightweight ~3M parameter head.

### Why ASPP?
Desert scenes contain objects at vastly different scales — a tiny rock cluster vs. the entire sky. ASPP uses dilated convolutions at multiple rates (6, 12, 18) to capture context from small, medium, and large receptive fields simultaneously, improving segmentation across all class sizes.

---

## Results

### Training Performance (ViT-S, 25 epochs)

| Metric | Value |
|--------|-------|
| Best Val IoU | 0.4695 |
| Best Val Accuracy | 0.8143 |
| Final Val Loss | 0.3486 |

### Training Curves
Training curves are saved to `train_stats/training_curves.png` after each run, showing Loss, IoU, and Pixel Accuracy for both train and validation sets across all epochs.

### Per-Class Performance
After running `test.py --has_gt`, per-class IoU is saved to `predictions/per_class_iou.png`.

Expected strong classes: Sky, Background, Landscape (large, uniform regions)
Expected challenging classes: Logs, Ground Clutter (rare, visually similar to surroundings)

---

## Project Structure

```
CodeChasers_CodeCrunch/
│
├── scripts/
│   ├── train.py                        # Main training script
│   ├── test.py                         # Inference and evaluation script
│   ├── visualize_segmentation.py       # Colorize raw mask predictions
│   ├── segmentation_head_best.pth      # Best model weights (created after training)
│   └── segmentation_head.pth          # Final epoch weights (created after training)
│
├── ENV_SETUP/
│   ├── setup_env.bat                   # One-click Windows environment setup
│   ├── create_env.bat                  # Create conda environment only
│   └── install_packages.bat           # Install Python packages only
│
├── train_stats/                        # Created automatically during training
│   ├── training_curves.png            # Loss, IoU, Accuracy plots
│   └── training_metrics.txt          # Numerical summary of training results
│
├── predictions/                        # Created automatically during testing
│   ├── masks/                         # Raw prediction masks (class IDs 0–9)
│   ├── masks_color/                   # Colorized RGB prediction masks
│   ├── comparisons/                   # Side-by-side input/GT/prediction images
│   ├── per_class_iou.png             # Bar chart of per-class IoU
│   └── evaluation_metrics.txt        # Numerical evaluation results
│
├── Offroad_Segmentation_Training_Dataset/   # Training data (download separately)
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
│
├── Offroad_Segmentation_testImages/         # Test data (download separately)
│   └── Color_Images/
│
└── README.md
```

---

## ⚙️ Environment Setup

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed
- NVIDIA GPU recommended (CPU training is very slow)
- Internet access on first run (DINOv2 downloads ~300MB for ViT-B)

---

### Windows Setup (Recommended — One Click)

```bash
# 1. Open Anaconda Prompt (NOT regular PowerShell)
# 2. Navigate to the ENV_SETUP folder
cd path\to\CodeChasers_CodeCrunch\ENV_SETUP

# 3. Run the setup script
setup_env.bat
```

This creates a conda environment called `EDU` with all required packages.

---

### Manual Setup (Windows / Mac / Linux)

```bash
# Step 1: Create environment
conda create -n EDU python=3.10 -y
conda activate EDU

# Step 2: Install PyTorch with CUDA (GPU support)
# For CUDA 11.8 (most common):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slow, not recommended):
pip install torch torchvision torchaudio

# Step 3: Install other dependencies
pip install numpy pillow opencv-python matplotlib tqdm
```

---

### Verify GPU is Working

```bash
conda activate EDU

# Windows — set this first to avoid OpenMP conflict:
set KMP_DUPLICATE_LIB_OK=TRUE

# Then verify:
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX XXXX
```

---

### Permanent Fix for Windows OpenMP Warning

```bash
conda activate EDU
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda activate EDU
```

---

## Dataset Setup

Download the dataset from the Duality AI Falcon platform (link provided in hackathon instructions) and place it as follows:

```
CodeChasers_CodeCrunch/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/       ← RGB training images (.png)
│   │   └── Segmentation/      ← Corresponding mask files (.png, same filenames)
│   └── val/
│       ├── Color_Images/       ← RGB validation images (.png)
│       └── Segmentation/      ← Corresponding mask files (.png, same filenames)
│
└── Offroad_Segmentation_testImages/
    └── Color_Images/           ← Unseen test images (.png, no masks)
```

> ⚠️ **Important:** Image files and their corresponding mask files must have **identical filenames** within each folder pair.

---

## Training

### Basic Training Run

```bash
# Activate environment
conda activate EDU

# Navigate to project root
cd path\to\CodeChasers_CodeCrunch

# Start training
python scripts/train.py
```

### What Happens During Training

1. DINOv2 backbone loads (downloads on first run — ~300MB for ViT-B)
2. Dataset is loaded with augmentation for training, clean for validation
3. Each epoch: forward pass → loss → backward pass → optimizer step
4. After each epoch: full evaluation on both train and val sets
5. Best model (by Val IoU) is automatically saved to `scripts/segmentation_head_best.pth`
6. Final model saved to `scripts/segmentation_head.pth`
7. Training curves and metrics saved to `train_stats/`

### Expected Training Output

```
Using device: cuda
Train: 450 | Val: 50
Loading DINOv2...
Backbone loaded.
Embed dim: 768, Token grid: 19x34
Starting training...
Epoch  1 | Loss: 0.4800 / 0.4200 | IoU: 0.4100 / 0.3900 | Acc: 0.7600 / 0.7700
  ✓ New best model saved (Val IoU: 0.3900)
Epoch  2 | Loss: 0.4200 / 0.3900 | IoU: 0.4400 / 0.4200 | Acc: 0.7800 / 0.7900
  ✓ New best model saved (Val IoU: 0.4200)
...
```

### Estimated Training Time

| GPU | Batch Size | Time per Epoch | 30 Epochs Total |
|-----|-----------|----------------|-----------------|
| RTX 4090 | 4 | ~3 min | ~1.5 hours |
| RTX 3080 | 2 | ~6 min | ~3 hours |
| RTX 3050 6GB | 1 | ~10 min | ~5 hours |
| Google Colab T4 | 2 | ~7 min | ~3.5 hours |

---

### Customizing Training

Key parameters at the top of `main()` in `train.py`:

```python
batch_size = 1       # Reduce if CUDA out of memory
lr = 1e-4            # Learning rate
n_epochs = 30        # Number of training epochs
weight_decay = 1e-4  # L2 regularization
```

Backbone options (change one line in train.py):
```python
# Smaller, faster (384-dim features):
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# Larger, better accuracy (768-dim features) — RECOMMENDED:
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
```

---

## Testing & Inference

### Run on Test Images (No Ground Truth)

```bash
python scripts/test.py --data_dir Offroad_Segmentation_testImages
```

This processes all images and saves colorized predictions to `predictions/`.

---

### Evaluate on Validation Set (Get IoU Score)

```bash
python scripts/test.py --data_dir Offroad_Segmentation_Training_Dataset/val --has_gt
```

This gives you:
- Mean IoU across all classes
- Per-class IoU breakdown
- Pixel accuracy
- Bar chart of per-class IoU saved to `predictions/per_class_iou.png`

---

### Use a Specific Model Checkpoint

```bash
python scripts/test.py \
  --model_path scripts/segmentation_head_best.pth \
  --data_dir Offroad_Segmentation_testImages
```

---

### All Test Script Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `scripts/segmentation_head_best.pth` | Path to trained model weights |
| `--data_dir` | `Offroad_Segmentation_testImages` | Path to image directory |
| `--output_dir` | `./predictions` | Where to save outputs |
| `--batch_size` | `2` | Inference batch size |
| `--num_samples` | `5` | Number of comparison images to save |
| `--has_gt` | `False` | Set flag if folder has ground truth masks |

---

### Test Output Files

```
predictions/
├── masks/              ← Raw class ID masks (0–9), suitable for submission
├── masks_color/        ← Human-readable colorized masks
├── comparisons/        ← Side-by-side: Input | Ground Truth | Prediction
├── per_class_iou.png  ← Bar chart (only with --has_gt)
└── evaluation_metrics.txt  ← Numerical results (only with --has_gt)
```

---

## Visualizing Predictions

To colorize any folder of raw prediction masks:

```bash
python scripts/visualize_segmentation.py --input_folder predictions/masks
```

Colorized images are saved to `predictions/masks/colorized/`.

---

## Tips to Improve IoU

**1. Train more epochs** — if Val IoU is still improving, keep going:
```python
n_epochs = 40  # in train.py
```

**2. Test-Time Augmentation (TTA)** — free boost, no retraining needed. Add to test.py inference loop to average predictions over flipped versions of each image.

**3. Use ViT-B backbone** — bigger model, better features:
```python
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
```

**4. Mixed precision training** — 30–40% faster on modern GPUs:
```python
scaler = torch.cuda.amp.GradScaler()
# wrap forward pass with torch.cuda.amp.autocast()
```

---

## Configuration Reference

### Class Weights (in train.py)

```python
CLASS_WEIGHTS = torch.tensor([
    0.4,   # 0: Background  — very common, downweighted
    1.0,   # 1: Trees
    1.2,   # 2: Lush Bushes
    1.0,   # 3: Dry Grass
    1.2,   # 4: Dry Bushes
    3.0,   # 5: Ground Clutter — rare
    5.0,   # 6: Logs          — very rare, critical for obstacle avoidance
    4.0,   # 7: Rocks         — rare
    0.8,   # 8: Landscape
    0.6,   # 9: Sky           — common, easy
])
```

### Image Size

Images are resized to the nearest multiple of 14 (DINOv2 patch size):
- Input: 960×540 → Resized to: **476×266** (w×h)
- Token grid: **34×19** patch tokens fed to the segmentation head

---

## Troubleshooting

### Training hangs at "Starting training..."
**Cause:** `num_workers > 0` on Windows causes DataLoader deadlock.
**Fix:**
```python
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
```

---

### CUDA out of memory
**Fix:** Reduce batch size:
```python
batch_size = 1  # in train.py main()
```

---

### `torch.cuda.is_available()` returns False
**Cause:** PyTorch installed without CUDA support.
**Fix:** Reinstall with CUDA:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### OpenMP error on Windows (`libiomp5md.dll`)
**Fix:**
```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda activate EDU
```

---

### `ValueError: Expected more than 1 value per channel` (BatchNorm error)
**Cause:** BatchNorm2d doesn't work with 1×1 spatial dimensions or batch_size=1.
**Fix:** Replace all `nn.BatchNorm2d(c)` with `nn.GroupNorm(1, c)` in both ASPP and ImprovedSegmentationHead classes.

---

### `RuntimeError: Error(s) in loading state_dict` (Missing keys)
**Cause:** Model architecture in `test.py` doesn't match the one used in `train.py` (e.g., BatchNorm vs GroupNorm mismatch).
**Fix:** Make sure the `ASPP` and `ImprovedSegmentationHead` class definitions are **identical** in both `train.py` and `test.py`.

---

### `xFormers is not available` warnings
These are harmless — xFormers is an optional speed optimization for DINOv2. Training works fine without it.

---

## Class Legend

| Class ID | Name | Color | Notes |
|----------|------|-------|-------|
| 0 | Background | ⬛ Black | Catch-all, very common |
| 1 | Trees | 🟩 Forest Green | Joshua trees, desert shrubs |
| 2 | Lush Bushes | 🟢 Lime Green | Dense green vegetation |
| 3 | Dry Grass | 🟫 Tan | Dried grass patches |
| 4 | Dry Bushes | 🟤 Brown | Dried shrub clusters |
| 5 | Ground Clutter | 🫒 Olive | Mixed debris on ground |
| 6 | Logs | 🪵 Saddle Brown | Fallen logs — obstacle |
| 7 | Rocks | ⬜ Gray | Rock formations — obstacle |
| 8 | Landscape | 🟠 Sienna | General ground terrain |
| 9 | Sky | 🔵 Sky Blue | Open sky |

---

## Requirements

```
python >= 3.10
torch >= 2.0
torchvision >= 0.15
numpy
pillow
opencv-python
matplotlib
tqdm
```

---

## Submission Checklist

- [ ] `scripts/train.py` — training script
- [ ] `scripts/test.py` — inference script  
- [ ] `scripts/segmentation_head_best.pth` — trained model weights
- [ ] `train_stats/training_curves.png` — training plots
- [ ] `train_stats/training_metrics.txt` — metrics summary
- [ ] `predictions/` — test image predictions
- [ ] `ENV_SETUP/setup_env.bat` — environment setup
- [ ] `README.md` — this file
- [ ] GitHub repo set to **private**
- [ ] Collaborators added: `Maazsyedm`, `rebekah-bogdanoff`, `egold010`

---

*Built with ❤️ by Team CodeChasers for the Duality AI Hackathon 2026*
