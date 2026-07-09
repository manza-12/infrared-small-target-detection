#!/usr/bin/env python3
"""Compute image-level Pd and pixel false alarm statistics for an ACM checkpoint."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACM_DIR = PROJECT_ROOT / "acm"
sys.path.insert(0, str(ACM_DIR))

from model.segmentation import ASKCResNetFPN, ASKCResUNet
from utils.data import SirstDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Pd/Fa for an ACM checkpoint")
    parser.add_argument("--backbone-mode", required=True, choices=["FPN", "UNet"])
    parser.add_argument("--fuse-mode", required=True, choices=["BiLocal", "AsymBi", "BiGlobal"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "sirst"))
    parser.add_argument("--split", default="test", choices=["test", "val", "trainval", "train"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=480)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--blocks-per-layer", type=int, default=4)
    return parser.parse_args()


def dataset_mode_from_split(split):
    return "val" if split in {"test", "val"} else "train"


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


def save_outputs(out_dir, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_pd_fa.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "metrics_pd_fa.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    os.environ["SIRST_DATA_ROOT"] = str(Path(args.data_root).resolve())

    dataset_args = argparse.Namespace(
        crop_size=args.crop_size,
        base_size=args.base_size,
        batch_size=args.batch_size,
        blocks_per_layer=args.blocks_per_layer,
    )
    dataset = SirstDataset(dataset_args, mode=dataset_mode_from_split(args.split))
    loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args)
    model.load_state_dict(load_state_dict(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    target_images = 0
    detected_target_images = 0
    false_alarm_pixels = 0
    total_pixels = 0
    total_images = 0

    with torch.no_grad():
        for data, labels in tqdm(loader, desc=f"Pd/Fa {args.backbone_mode}+{args.fuse_mode}"):
            logits = model(data.to(device)).detach().cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = probs > args.threshold
            gts = labels.numpy() > 0

            for pred, gt in zip(preds, gts):
                pred_mask = np.squeeze(pred).astype(bool)
                gt_mask = np.squeeze(gt).astype(bool)
                has_gt = bool(gt_mask.any())
                has_hit = bool(np.logical_and(pred_mask, gt_mask).any())
                if has_gt:
                    target_images += 1
                    if has_hit:
                        detected_target_images += 1
                false_alarm_pixels += int(np.logical_and(pred_mask, ~gt_mask).sum())
                total_pixels += int(gt_mask.size)
                total_images += 1

    pd_value = detected_target_images / target_images if target_images else 0.0
    fa_pixels_per_image = false_alarm_pixels / total_images if total_images else 0.0
    fa_pixel_rate = false_alarm_pixels / total_pixels if total_pixels else 0.0
    metrics = {
        "backbone": args.backbone_mode,
        "fusion": args.fuse_mode,
        "split": args.split,
        "num_images": total_images,
        "target_images": target_images,
        "detected_target_images": detected_target_images,
        "Pd": float(pd_value),
        "Fa": float(fa_pixels_per_image),
        "Fa_definition": "false_alarm_pixels / total_images",
        "false_alarm_pixels": false_alarm_pixels,
        "false_alarm_pixel_rate": float(fa_pixel_rate),
        "threshold": args.threshold,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    save_outputs(out_dir, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics_pd_fa_json={out_dir / 'metrics_pd_fa.json'}")
    print(f"metrics_pd_fa_csv={out_dir / 'metrics_pd_fa.csv'}")


if __name__ == "__main__":
    main()
