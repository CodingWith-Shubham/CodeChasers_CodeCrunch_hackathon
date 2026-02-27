"""
Visualize segmentation masks with high-contrast colors.
Usage: python visualize_segmentation.py --input_folder path/to/masks
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image

# Class color palette
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

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

def colorize_mask(mask_path, output_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(color_palette)):
        color_img[mask == class_id] = color_palette[class_id]
    cv2.imwrite(str(output_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = input_folder / "colorized"
    output_folder.mkdir(exist_ok=True)

    exts = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    files = [f for f in input_folder.iterdir()
             if f.is_file() and f.suffix.lower() in exts]
    print(f"Found {len(files)} mask files")

    for f in sorted(files):
        out = output_folder / f"{f.stem}_color.png"
        if colorize_mask(f, out):
            print(f"  Colorized: {f.name}")

    print(f"\nDone! Colorized masks saved to: {output_folder}")

if __name__ == "__main__":
    main()