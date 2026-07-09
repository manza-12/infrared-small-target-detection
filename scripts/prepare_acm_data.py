#!/usr/bin/env python3
"""
Convert YOLO-format detection data to SIRST segmentation format for ACM training.

YOLO format expected:
  data/
    images/
    labels/  (YOLO .txt: cls cx cy w h, normalized 0-1)
    data.yaml

Output (SIRST segmentation format):
  data/ACM_<name>/
    images/             # *.png copies
    masks/              # *_pixels0.png (binary masks from bboxes)
    idx_427/
      trainval.txt      # training image names
      test.txt          # validation image names

Usage:
    python scripts/prepare_acm_data.py --data-dir data --split train val
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def yolo_bbox_to_mask(img_shape, yolo_boxes, img_path=None):
    """
    Convert YOLO-format boxes to a binary segmentation mask.

    Each small target bbox is rendered as a filled white ellipse/rectangle
    on a black background.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in yolo_boxes:
        cls_id, cx, cy, bw, bh = box
        # Denormalize
        x_c, y_c = int(cx * w), int(cy * h)
        bw_px, bh_px = int(bw * w), int(bh * h)

        x1 = max(0, x_c - bw_px // 2)
        y1 = max(0, y_c - bh_px // 2)
        x2 = min(w, x1 + bw_px)
        y2 = min(h, y1 + bh_px)

        # Draw filled rectangle (for small targets, this is sufficient)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def convert_dataset(data_dir, train_split='train', val_split='val',
                    output_name='ACM'):
    """
    Convert YOLO detection dataset to SIRST segmentation format.
    """
    data_dir = Path(data_dir)
    out_dir = data_dir / f'ACM_{output_name}'

    # Read data.yaml
    yaml_path = data_dir / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Create output structure
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)
    (out_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (out_dir / 'idx_427').mkdir(parents=True, exist_ok=True)

    for split_name, txt_name in [(train_split, 'trainval.txt'),
                                  (val_split, 'test.txt')]:
        split_dir = data_dir / 'images' / split_name
        label_dir = data_dir / 'labels' / split_name

        if not split_dir.exists():
            print(f'Warning: {split_dir} not found, skipping')
            continue

        names = []
        for img_path in sorted(split_dir.glob('*')):
            stem = img_path.stem
            names.append(stem)

            # Copy image
            img_dest = out_dir / 'images' / f'{stem}.png'
            if not img_dest.exists():
                shutil.copy2(img_path, img_dest)

            # Create mask from YOLO label
            label_path = label_dir / f'{stem}.txt'
            mask_path = out_dir / 'masks' / f'{stem}_pixels0.png'
            if label_path.exists() and not mask_path.exists():
                img = cv2.imread(str(img_path))
                boxes = []
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            boxes.append([float(x) for x in parts[:5]])
                if boxes:
                    mask = yolo_bbox_to_mask(img.shape, boxes)
                else:
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.imwrite(str(mask_path), mask)

        # Write index file
        with open(out_dir / 'idx_427' / txt_name, 'w') as f:
            f.write('\n'.join(names))

        print(f'{txt_name}: {len(names)} samples')

    print(f'Conversion complete! Output: {out_dir}')
    print(f'Run ACM training with: --data-dir {out_dir}')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Prepare ACM data from YOLO format')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root data directory with images/ and labels/ subdirs')
    parser.add_argument('--train-split', type=str, default='train',
                        help='Training split subdirectory under images/ and labels/')
    parser.add_argument('--val-split', type=str, default='val',
                        help='Validation split subdirectory')
    parser.add_argument('--output-name', type=str, default='SIRST',
                        help='Dataset name for output directory')
    args = parser.parse_args()

    convert_dataset(args.data_dir, args.train_split, args.val_split, args.output_name)


if __name__ == '__main__':
    main()
