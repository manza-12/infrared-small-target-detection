"""
Convert YOLO-format bounding box labels to binary masks
and restructure data for ACM model training.

Before:
  data/images/train/image-train/*.png    (images)
  data/images/val/images—val/*.png       (images)
  data/labels/train/labels_train/*.txt   (YOLO bboxes)
  data/labels/val/labels_val/*.txt       (YOLO bboxes)

After:
  data/sirst/images/*.png
  data/sirst/masks/*_pixels0.png         (binary masks from bboxes)
  data/sirst/idx_427/trainval.txt
  data/sirst/idx_427/test.txt
"""

import os
import sys
from PIL import Image
import shutil

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

# Source
TRAIN_IMG_SRC = os.path.join(DATA, 'images', 'train', 'image-train')
VAL_IMG_SRC   = os.path.join(DATA, 'images', 'val', 'images—val')
TRAIN_LBL_SRC = os.path.join(DATA, 'labels', 'train', 'labels_train')
VAL_LBL_SRC   = os.path.join(DATA, 'labels', 'val', 'labels_val')

# Destination
SIRST = os.path.join(DATA, 'sirst')
DST_IMG   = os.path.join(SIRST, 'images')
DST_MASK  = os.path.join(SIRST, 'masks')
DST_IDX   = os.path.join(SIRST, 'idx_427')

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_MASK, exist_ok=True)
os.makedirs(DST_IDX, exist_ok=True)


def yolo_to_mask(label_path, img_size):
    """
    Convert a YOLO label file to a binary mask (PIL Image).
    YOLO format per line:  class_id cx cy w h   (all normalised 0-1)
    Multiple targets → all filled as 255.
    """
    w_img, h_img = img_size
    mask = Image.new('L', (w_img, h_img), 0)

    if not os.path.exists(label_path):
        return mask  # empty mask if no label file

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])

            # Convert normalised → pixel coordinates
            x1 = int((cx - bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img)
            x2 = int((cx + bw / 2) * w_img)
            y2 = int((cy + bh / 2) * h_img)

            # Clamp
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y2 = max(0, min(y2, h_img - 1))

            # Make at least 1 pixel
            if x2 <= x1:
                x2 = min(x1 + 1, w_img - 1)
            if y2 <= y1:
                y2 = min(y1 + 1, h_img - 1)

            # Fill rectangle as white
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    mask.putpixel((x, y), 255)

    return mask


def process_split(img_src, lbl_src, name_list_file, split_name):
    """
    Copy images and convert labels for one split (train / val).
    """
    if not os.path.isdir(img_src):
        print(f'  SKIP — image dir not found: {img_src}')
        return []

    names = []
    for fname in sorted(os.listdir(img_src)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        name = os.path.splitext(fname)[0]  # e.g. "irstd1k_XDU0"

        src_img  = os.path.join(img_src, fname)
        dst_img  = os.path.join(DST_IMG, fname)
        lbl_file = os.path.join(lbl_src, name + '.txt')
        dst_mask = os.path.join(DST_MASK, name + '_pixels0.png')

        # Get image size
        with Image.open(src_img) as im:
            w, h = im.size

        # Convert YOLO → mask
        mask = yolo_to_mask(lbl_file, (w, h))

        # Copy image
        shutil.copy2(src_img, dst_img)
        # Save mask
        mask.save(dst_mask)

        names.append(name)

    # Write name list
    with open(name_list_file, 'w') as f:
        f.write('\n'.join(names))

    print(f'  {split_name}: {len(names)} images/masks')
    return names


if __name__ == '__main__':
    print('=== Converting YOLO labels → binary masks ===')
    print()

    # --- Train ---
    print('[Train split]')
    train_names = process_split(
        img_src=TRAIN_IMG_SRC,
        lbl_src=TRAIN_LBL_SRC,
        name_list_file=os.path.join(DST_IDX, 'trainval.txt'),
        split_name='train',
    )

    # --- Val ---
    print('[Val split]')
    val_names = process_split(
        img_src=VAL_IMG_SRC,
        lbl_src=VAL_LBL_SRC,
        name_list_file=os.path.join(DST_IDX, 'test.txt'),
        split_name='val',
    )

    print()
    print(f'=== Done ===')
    print(f'Images : {DST_IMG}')
    print(f'Masks  : {DST_MASK}')
    print(f'Idx    : {DST_IDX}')
    print(f'Train  : {len(train_names)} samples')
    print(f'Val    : {len(val_names)} samples')
