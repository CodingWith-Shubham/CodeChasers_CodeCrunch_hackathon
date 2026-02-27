"""
Offroad Semantic Segmentation - Training Script
DINOv2 + Multi-Scale ConvNeXt Segmentation Head
Optimized for Duality AI Hackathon
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm

plt.switch_backend('Agg')


# Class Configuration


value_map = {
    0: 0,       # Background
    100: 1,     # Trees
    200: 2,     # Lush Bushes
    300: 3,     # Dry Grass
    500: 4,     # Dry Bushes
    550: 5,     # Ground Clutter
    700: 6,     # Logs
    800: 7,     # Rocks
    7100: 8,    # Landscape
    10000: 9    # Sky
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Class weights to handle imbalance (rare classes like Logs, Rocks get higher weight)
CLASS_WEIGHTS = torch.tensor([
    0.4,   # Background
    1.0,   # Trees
    1.2,   # Lush Bushes
    1.0,   # Dry Grass
    1.2,   # Dry Bushes
    3.0,   # Ground Clutter
    5.0,   # Logs
    4.0,   # Rocks
    0.8,   # Landscape
    0.6,   # Sky
], dtype=torch.float32)

n_classes = len(value_map)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# Dataset with Augmentation


class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.augment = augment
        self.data_ids = os.listdir(self.image_dir)
        self.img_size = (int(((540 / 2) // 14) * 14), int(((960 / 2) // 14) * 14))  # (h, w)

    def __len__(self):
        return len(self.data_ids)

    def _augment(self, image, mask):
        """Apply joint augmentations to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip (less common for outdoor scenes)
        if random.random() > 0.8:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation (+/- 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Color jitter on image only
        if random.random() > 0.4:
            jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )
            image = jitter(image)

        # Random crop and resize
        if random.random() > 0.5:
            h, w = self.img_size
            crop_scale = random.uniform(0.75, 1.0)
            crop_h = int(h * crop_scale)
            crop_w = int(w * crop_scale)
            i = random.randint(0, h - crop_h)
            j = random.randint(0, w - crop_w)
            image = TF.crop(image, i, j, crop_h, crop_w)
            mask = TF.crop(mask, i, j, crop_h, crop_w)
            image = TF.resize(image, [h, w])
            mask = TF.resize(mask, [h, w], interpolation=Image.NEAREST)

        return image, mask

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        h, w = self.img_size
        image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = normalize(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask



# Improved Segmentation Head - Multi-Scale with Attention


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1), nn.GroupNorm(1, out_channels), nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.GroupNorm(1, out_channels), nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.GroupNorm(1, out_channels), nn.GELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
            nn.GroupNorm(1, out_channels), nn.GELU()
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1), nn.GroupNorm(1, out_channels), nn.GELU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.GroupNorm(1, out_channels), nn.GELU(), nn.Dropout(0.3)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        pool = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.project(torch.cat([
            self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x), pool
        ], dim=1))


class ImprovedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.GELU()
        )

        self.aspp = ASPP(hidden, hidden)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return self.classifier(x)

# Loss: Focal + Dice combined


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + self.smooth
        )
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight=None, focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(weight=weight)
        self.dice = DiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, inputs, targets):
        return self.fw * self.focal(inputs, targets) + self.dw * self.dice(inputs, targets)



# Metrics


def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        tgt_c = target == c
        intersection = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        ious.append(float('nan') if union == 0 else (intersection / union).item())
    return np.nanmean(ious)


def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().item()


def evaluate(model, backbone, loader, device, num_classes=10):
    model.eval()
    ious, accs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            ious.append(compute_iou(outputs, labels, num_classes))
            accs.append(compute_pixel_accuracy(outputs, labels))
    model.train()
    return np.nanmean(ious), np.mean(accs)


# Main


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Config
    batch_size = 2
    lr = 3e-4
    n_epochs = 25
    weight_decay = 1e-4

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Data
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir=data_dir, augment=True)
    valset = MaskDataset(data_dir=val_dir, augment=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {len(trainset)} | Val: {len(valset)}")

    # Backbone
    print("Loading DINOv2...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()
    backbone.to(device)
    for name, p in backbone.named_parameters():
        if "blocks.10" in name or "blocks.11" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    print("Backbone loaded.")

    # Get feature dim
    h = int(((540 / 2) // 14) * 14)
    w = int(((960 / 2) // 14) * 14)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, h, w).to(device)
        feats = backbone.forward_features(dummy)["x_norm_patchtokens"]
    n_emb = feats.shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Embed dim: {n_emb}, Token grid: {tokenH}x{tokenW}")

    # Model
    model = ImprovedSegmentationHead(n_emb, n_classes, tokenW, tokenH).to(device)

    # Loss and optimizer
    weights = CLASS_WEIGHTS.to(device)
    criterion = CombinedLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    history = {k: [] for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou',
                                'train_acc', 'val_acc']}
    best_val_iou = 0.0

    print("\nStarting training...")
    for epoch in range(n_epochs):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Val loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                val_losses.append(criterion(outputs, labels).item())

        # train_iou, train_acc = evaluate(model, backbone, train_loader, device)
        train_iou, train_acc = 0, 0  # Skip train metrics for speed, can be enabled if needed
        val_iou, val_acc = evaluate(model, backbone, val_loader, device)

        history['train_loss'].append(np.mean(losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:2d} | Loss: {np.mean(losses):.4f} / {np.mean(val_losses):.4f} "
              f"| IoU: {train_iou:.4f} / {val_iou:.4f} | Acc: {train_acc:.4f} / {val_acc:.4f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model_path = os.path.join(script_dir, "segmentation_head_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ New best model saved (Val IoU: {val_iou:.4f})")

    # Save final model
    model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(model.state_dict(), model_path)

    # Save plots
    _save_plots(history, output_dir)
    _save_metrics(history, output_dir)

    print(f"\nTraining complete! Best Val IoU: {best_val_iou:.4f}")
    print(f"Models saved in {script_dir}")


def _save_plots(history, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (train_key, val_key, title) in zip(axes, [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_iou', 'val_iou', 'Mean IoU'),
        ('train_acc', 'val_acc', 'Pixel Accuracy'),
    ]):
        ax.plot(history[train_key], label='Train', color='royalblue')
        ax.plot(history[val_key], label='Validation', color='tomato')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training curves.")


def _save_metrics(history, output_dir):
    with open(os.path.join(output_dir, 'training_metrics.txt'), 'w') as f:
        f.write("TRAINING RESULTS\n" + "=" * 50 + "\n")
        f.write(f"Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou'])+1})\n")
        f.write(f"Best Val Accuracy: {max(history['val_acc']):.4f}\n")
        f.write(f"Final Val IoU:     {history['val_iou'][-1]:.4f}\n")
        f.write(f"Final Val Loss:    {history['val_loss'][-1]:.4f}\n")
    print("Saved training metrics.")


if __name__ == "__main__":
    main()