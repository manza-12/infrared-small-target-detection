#!/usr/bin/env python3
"""Create ACM prediction comparison images and bad-case candidates."""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACM_DIR = PROJECT_ROOT / "acm"
sys.path.insert(0, str(ACM_DIR))

from model.segmentation import ASKCResNetFPN, ASKCResUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ACM predictions")
    parser.add_argument("--backbone-mode", required=True, choices=["FPN", "UNet"])
    parser.add_argument("--fuse-mode", required=True, choices=["BiLocal", "AsymBi", "BiGlobal"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "sirst"))
    parser.add_argument("--split-file", default="idx_427/test.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bad-case-dir", default="")
    parser.add_argument("--bad-case-table", default="")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--blocks-per-layer", type=int, default=4)
    return parser.parse_args()


def build_model(args):
    layer_blocks = [args.blocks_per_layer] * 3
    channels = [8, 16, 32, 64]
    if args.backbone_mode == "FPN":
        return ASKCResNetFPN(layer_blocks, channels, args.fuse_mode)
    return ASKCResUNet(layer_blocks, channels, args.fuse_mode)


def load_state_dict(checkpoint):
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location="cpu")


def read_names(data_root, split_file):
    split_path = Path(data_root) / split_file
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pair(data_root, name, base_size):
    data_root = Path(data_root)
    img = Image.open(data_root / "images" / f"{name}.png").convert("RGB")
    mask = Image.open(data_root / "masks" / f"{name}_pixels0.png").convert("L")
    return img.resize((base_size, base_size), Image.BILINEAR), mask.resize((base_size, base_size), Image.NEAREST)


def infer_mask(model, img, threshold, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).detach().cpu()
    pred = torch.sigmoid(logits).numpy()[0]
    return np.squeeze(pred) > threshold


def classify_case(gt_mask, pred_mask, iou):
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())
    inter = int(np.logical_and(gt_mask, pred_mask).sum())
    if gt_area > 0 and pred_area == 0:
        return "missed_detection"
    if gt_area == 0 and pred_area > 0:
        return "false_alarm"
    if gt_area > 0 and pred_area > 0 and inter == 0:
        return "poor_localization"
    return "low_iou"


def save_triplet(path, img, gt_mask, pred_mask, title):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bad_dir = Path(args.bad_case_dir) if args.bad_case_dir else None
    if bad_dir:
        bad_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    model.load_state_dict(load_state_dict(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    names = read_names(args.data_root, args.split_file)
    scored = []
    positive_names = []

    for name in names:
        img, mask = load_pair(args.data_root, name, args.base_size)
        gt_mask = np.array(mask) > 0
        pred_mask = infer_mask(model, img, args.threshold, device)
        inter = int(np.logical_and(gt_mask, pred_mask).sum())
        union = int(np.logical_or(gt_mask, pred_mask).sum())
        iou = inter / union if union else 0.0
        row = {
            "model": f"{args.backbone_mode}+{args.fuse_mode}",
            "image_name": name,
            "case_type": classify_case(gt_mask, pred_mask, iou),
            "image_iou": iou,
            "gt_area": int(gt_mask.sum()),
            "pred_area": int(pred_mask.sum()),
            "notes": "candidate selected by image-level IoU and mask overlap heuristics",
        }
        scored.append((iou, name, img, gt_mask, pred_mask, row))
        if gt_mask.any():
            positive_names.append((name, img, gt_mask, pred_mask))

    saved_vis = 0
    for name, img, gt_mask, pred_mask in positive_names[: args.max_images]:
        save_triplet(
            out_dir / f"{args.backbone_mode}_{args.fuse_mode}_{name}_triplet.png",
            img,
            gt_mask,
            pred_mask,
            f"{args.backbone_mode}+{args.fuse_mode} - {name}",
        )
        saved_vis += 1

    bad_rows = []
    if bad_dir:
        scored_sorted = sorted(scored, key=lambda item: item[0])
        seen = set()
        for _, name, img, gt_mask, pred_mask, row in scored_sorted:
            if name in seen:
                continue
            seen.add(name)
            bad_rows.append(row)
            save_triplet(
                bad_dir / f"{args.backbone_mode}_{args.fuse_mode}_{row['case_type']}_{name}.png",
                img,
                gt_mask,
                pred_mask,
                f"{args.backbone_mode}+{args.fuse_mode} - {row['case_type']} - IoU {row['image_iou']:.4f}",
            )
            if len(bad_rows) >= 10:
                break

    if args.bad_case_table:
        table_path = Path(args.bad_case_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        exists = table_path.exists()
        with table_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "image_name", "case_type", "image_iou", "gt_area", "pred_area", "notes"],
            )
            if not exists:
                writer.writeheader()
            writer.writerows(bad_rows)

    print(f"visualizations_saved={saved_vis}")
    print(f"bad_cases_saved={len(bad_rows)}")


if __name__ == "__main__":
    main()
