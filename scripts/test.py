"""
Offroad Semantic Segmentation - Inference/Test Script
Runs on unseen test images and saves colorized predictions + metrics
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

plt.switch_backend('Agg')

# ============================================================================
# Config
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 200, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color_mask[mask == c] = color_palette[c]
    return color_mask


# ============================================================================
# Dataset (test images - no ground truth masks required)
# ============================================================================

class TestImageDataset(Dataset):
    """For test images that have no segmentation masks."""
    def __init__(self, image_dir, img_size):
        self.image_dir = image_dir
        self.img_size = img_size  # (h, w)
        exts = ['.png', '.jpg', '.jpeg']
        self.data_ids = [f for f in os.listdir(image_dir)
                         if os.path.splitext(f)[1].lower() in exts]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        h, w = self.img_size
        img = img.resize((w, h), Image.BILINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img), data_id


class EvalDataset(Dataset):
    """For validation images that have ground truth masks."""
    def __init__(self, data_dir, img_size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.img_size = img_size
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        h, w = self.img_size

        img = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        img = img.resize((w, h), Image.BILINEAR)

        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)
        mask = mask.resize((w, h), Image.NEAREST)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img), torch.from_numpy(np.array(mask)).long(), data_id


# ============================================================================
# Model (must match train.py)
# ============================================================================

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1), nn.GroupNorm(1, out_channels), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.GroupNorm(1, out_channels), nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.GroupNorm(1, out_channels), nn.GELU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
            nn.GroupNorm(1, out_channels), nn.GELU())
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1), nn.GroupNorm(1, out_channels), nn.GELU())
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.GroupNorm(1, out_channels), nn.GELU(), nn.Dropout(0.3))

    def forward(self, x):
        h, w = x.shape[2:]
        pool = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.project(torch.cat([
            self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x), pool], dim=1))


class ImprovedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU())
        self.aspp = ASPP(hidden, hidden)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou_per_class(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        pred_c, tgt_c = pred == c, target == c
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        ious.append(float('nan') if union == 0 else (inter / union).item())
    return ious


def save_comparison(img_tensor, gt_mask, pred_mask, output_path, name):
    img = img_tensor.cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1) * std + mean
    img = np.clip(img, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title('Input Image'); axes[0].axis('off')
    axes[1].imshow(gt_color); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred_color); axes[2].set_title('Prediction'); axes[2].axis('off')
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=os.path.join(script_dir, 'segmentation_head_best.pth'))
    parser.add_argument('--data_dir', default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'))
    parser.add_argument('--output_dir', default='./predictions')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--has_gt', action='store_true',
                        help='Set if test data has ground truth masks')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    h = int(((540 / 2) // 14) * 14)
    w = int(((960 / 2) // 14) * 14)
    img_size = (h, w)

    # Load dataset
    if args.has_gt:
        dataset = EvalDataset(args.data_dir, img_size)
    else:
        img_dir = os.path.join(args.data_dir, 'Color_Images')
        if os.path.exists(img_dir):
            dataset = TestImageDataset(img_dir, img_size)
        else:
            dataset = TestImageDataset(args.data_dir, img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} images")

    # Backbone
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    # Get dims
    with torch.no_grad():
        dummy = torch.zeros(1, 3, h, w).to(device)
        feats = backbone.forward_features(dummy)["x_norm_patchtokens"]
    n_emb = feats.shape[2]
    tokenH, tokenW = h // 14, w // 14

    # Load model
    model = ImprovedSegmentationHead(n_emb, n_classes, tokenW, tokenH)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval().to(device)
    print(f"Model loaded from {args.model_path}")

    # Subdirs
    masks_dir = os.path.join(args.output_dir, 'masks')
    colors_dir = os.path.join(args.output_dir, 'masks_color')
    comps_dir = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, colors_dir, comps_dir]:
        os.makedirs(d, exist_ok=True)

    all_class_ious = []
    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if args.has_gt:
                imgs, labels, data_ids = batch
                imgs, labels = imgs.to(device), labels.to(device)
            else:
                imgs, data_ids = batch
                imgs = imgs.to(device)

            feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(outputs, dim=1)

            if args.has_gt:
                class_ious = compute_iou_per_class(outputs, labels)
                all_class_ious.append(class_ious)

            for i in range(imgs.shape[0]):
                name = os.path.splitext(data_ids[i])[0]
                pred = preds[i].cpu().numpy().astype(np.uint8)

                Image.fromarray(pred).save(os.path.join(masks_dir, f"{name}_pred.png"))
                color = mask_to_color(pred)
                cv2.imwrite(os.path.join(colors_dir, f"{name}_color.png"),
                            cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

                if args.has_gt and sample_count < args.num_samples:
                    save_comparison(imgs[i], labels[i], preds[i],
                                    os.path.join(comps_dir, f"sample_{sample_count}.png"),
                                    data_ids[i])
                sample_count += 1

    if all_class_ious:
        avg_class_iou = np.nanmean(all_class_ious, axis=0)
        mean_iou = np.nanmean(avg_class_iou)
        print(f"\n{'='*50}\nEVALUATION RESULTS\n{'='*50}")
        print(f"Mean IoU: {mean_iou:.4f}\n")
        print(f"{'Class':<20} {'IoU':>8}")
        print("-" * 30)
        for name, iou in zip(class_names, avg_class_iou):
            print(f"{name:<20} {iou:>8.4f}" if not np.isnan(iou) else f"{name:<20} {'N/A':>8}")

        # Per-class bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        valid = [v if not np.isnan(v) else 0 for v in avg_class_iou]
        bars = ax.bar(range(n_classes), valid,
                      color=[color_palette[i] / 255.0 for i in range(n_classes)],
                      edgecolor='black')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.axhline(y=mean_iou, color='red', linestyle='--', label=f'Mean IoU = {mean_iou:.4f}')
        ax.set_ylim(0, 1)
        ax.set_title('Per-Class IoU', fontsize=14, fontweight='bold')
        ax.set_ylabel('IoU')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'per_class_iou.png'), dpi=150)
        plt.close()

        # Save metrics
        with open(os.path.join(args.output_dir, 'evaluation_metrics.txt'), 'w') as f:
            f.write(f"Mean IoU: {mean_iou:.4f}\n\nPer-Class IoU:\n")
            for name, iou in zip(class_names, avg_class_iou):
                f.write(f"  {name:<20}: {iou:.4f}\n" if not np.isnan(iou) else f"  {name:<20}: N/A\n")

    print(f"\nDone! Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()