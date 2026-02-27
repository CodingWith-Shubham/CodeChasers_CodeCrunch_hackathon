\# Offroad Semantic Scene Segmentation

\### Duality AI Hackathon — Segmentation Track



---



\## Overview



This project trains a \*\*semantic segmentation model\*\* for off-road desert environments using a \*\*DINOv2 backbone\*\* (pre-trained Vision Transformer) with a custom \*\*ASPP-based ConvNeXt segmentation head\*\*. The model segments 10 classes: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky.



---



\## Project Structure



```

project\_root/

├── scripts/

│   ├── train.py                   # Training script (improved model + augmentation)

│   ├── test.py                    # Inference/evaluation script

│   ├── visualize\_segmentation.py  # Colorize raw mask predictions

│   ├── segmentation\_head\_best.pth # Best model weights (after training)

│   └── segmentation\_head.pth     # Final epoch weights (after training)

├── ENV\_SETUP/

│   ├── setup\_env.bat              # One-click Windows environment setup

│   ├── create\_env.bat             # Create conda env only

│   └── install\_packages.bat      # Install packages only

├── train\_stats/

│   ├── training\_curves.png        # Training/val curves

│   └── training\_metrics.txt      # Final metrics summary

└── README.md

```



---



\## Environment Setup



\### Windows (Anaconda/Miniconda required)



```bash

\# Navigate to ENV\_SETUP folder

cd ENV\_SETUP



\# Run one-click setup (creates 'EDU' environment)

setup\_env.bat

```



\### Mac/Linux



```bash

conda create -n EDU python=3.10 -y

conda activate EDU



pip install torch torchvision torchaudio

pip install numpy pillow opencv-python matplotlib tqdm

```



---



\## Dataset Structure



Place your dataset as follows:



```

project\_root/../

├── Offroad\_Segmentation\_Training\_Dataset/

│   ├── train/

│   │   ├── Color\_Images/      # RGB .png images

│   │   └── Segmentation/      # Mask .png files (same names)

│   └── val/

│       ├── Color\_Images/

│       └── Segmentation/

└── Offroad\_Segmentation\_testImages/

&nbsp;   └── Color\_Images/          # Unseen test images

```



---



\## Training



```bash

conda activate EDU

cd scripts

python train.py

```



\*\*What happens:\*\*

\- DINOv2 ViT-S/14 is loaded as a frozen feature extractor

\- An ASPP-based segmentation head is trained on top

\- Data augmentation: flips, rotation, color jitter, random crops

\- Focal + Dice combined loss with per-class weighting

\- AdamW optimizer + cosine annealing LR scheduler

\- Best model saved automatically at `scripts/segmentation\_head\_best.pth`



---



\## Testing / Inference



```bash

\# On test images (no ground truth):

python test.py --data\_dir ../Offroad\_Segmentation\_testImages



\# On val images with ground truth (to get IoU):

python test.py --data\_dir ../Offroad\_Segmentation\_Training\_Dataset/val --has\_gt



\# With custom model path:

python test.py --model\_path segmentation\_head\_best.pth --data\_dir ../testImages

```



\*\*Outputs saved to `./predictions/`:\*\*

\- `masks/` — Raw class ID masks (0–9)

\- `masks\_color/` — Colorized RGB predictions

\- `comparisons/` — Side-by-side input/GT/prediction images

\- `per\_class\_iou.png` — Bar chart of per-class IoU

\- `evaluation\_metrics.txt` — Numerical results



---



\## Visualize Masks



```bash

python visualize\_segmentation.py --input\_folder path/to/raw/masks

```



---



\## Class Legend



| Class ID | Name           | Color         |

|----------|----------------|---------------|

| 0        | Background     | Black         |

| 1        | Trees          | Forest Green  |

| 2        | Lush Bushes    | Lime          |

| 3        | Dry Grass      | Tan           |

| 4        | Dry Bushes     | Brown         |

| 5        | Ground Clutter | Olive         |

| 6        | Logs           | Saddle Brown  |

| 7        | Rocks          | Gray          |

| 8        | Landscape      | Sienna        |

| 9        | Sky            | Sky Blue      |



---



\## Key Design Decisions



\- \*\*Frozen DINOv2 backbone\*\*: Leverages powerful pre-trained features without overfitting

\- \*\*ASPP module\*\*: Multi-scale context for better boundary detection

\- \*\*Combined Focal+Dice loss\*\*: Handles class imbalance (Logs, Rocks are rare)

\- \*\*Class-weighted loss\*\*: Rare classes get 2–3× higher penalty

\- \*\*Data augmentation\*\*: Flips, rotation, color jitter, random crops → better generalization



---



\## Requirements



\- Python 3.10+

\- PyTorch 2.0+

\- torchvision, numpy, pillow, opencv-python, matplotlib, tqdm

\- Internet access required on first run (DINOv2 downloads ~85MB)



---



\## Expected Results



| Metric         | Validation |

|----------------|-----------|

| Mean IoU       | ~0.55–0.70 |

| Pixel Accuracy | ~0.85–0.92 |



\*Actual values depend on hardware and number of training epochs.\*

