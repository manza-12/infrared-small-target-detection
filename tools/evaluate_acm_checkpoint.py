#!/usr/bin/env python3
"""Evaluate an ACM checkpoint on a SIRST split without training."""

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

from model.loss import SoftLoULoss
from model.metrics import SamplewiseSigmoidMetric, SigmoidMetric
from model.segmentation import ASKCResNetFPN, ASKCResUNet
from utils.data import SirstDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ACM checkpoint on SIRST split")
    parser.add_argument("--backbone-mode", required=True, choices=["FPN", "UNet"])
    parser.add_argument("--fuse-mode", required=True, choices=["BiLocal", "AsymBi", "BiGlobal"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "sirst"))
    parser.add_argument("--split", default="test", choices=["test", "val", "trainval", "train"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=480)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--blocks-per-layer", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the full split")
    return parser.parse_args()


def build_model(args):
    layer_blocks = [args.blocks_per_layer] * 3
    channels = [8, 16, 32, 64]
    if args.backbone_mode == "FPN":
        return ASKCResNetFPN(layer_blocks, channels, args.fuse_mode)
    return ASKCResUNet(layer_blocks, channels, args.fuse_mode)


def dataset_mode_from_split(split):
    if split in {"test", "val"}:
        return "val"
    if split in {"trainval", "train"}:
        return "train"
    raise ValueError(f"Unsupported split: {split}")


def load_state_dict(checkpoint):
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location="cpu")


def write_config(args, out_dir, num_samples):
    split_file = "test.txt" if args.split in {"test", "val"} else "trainval.txt"
    lines = [
        f"backbone_mode={args.backbone_mode}",
        f"fuse_mode={args.fuse_mode}",
        f"checkpoint={Path(args.checkpoint).resolve()}",
        f"data_root={Path(args.data_root).resolve()}",
        f"split={args.split}",
        f"split_file={Path(args.data_root).resolve() / 'idx_427' / split_file}",
        f"batch_size={args.batch_size}",
        f"num_samples={num_samples}",
        f"max_batches={args.max_batches}",
    ]
    (out_dir / "config_used.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metrics(out_dir, metrics):
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SIRST_DATA_ROOT"] = str(Path(args.data_root).resolve())

    dataset_args = argparse.Namespace(
        crop_size=args.crop_size,
        base_size=args.base_size,
        batch_size=args.batch_size,
        blocks_per_layer=args.blocks_per_layer,
    )
    dataset = SirstDataset(dataset_args, mode=dataset_mode_from_split(args.split))
    loader = Data.DataLoader(dataset, batch_size=args.batch_size)

    model = build_model(args)
    state_dict = load_state_dict(args.checkpoint)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    criterion = SoftLoULoss()
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    losses = []
    batches = 0

    with torch.no_grad():
        for data, labels in tqdm(loader, desc=f"Eval {args.backbone_mode}+{args.fuse_mode}"):
            output = model(data.cuda()).cpu()
            loss = criterion(output, labels)
            losses.append(loss.item())
            iou_metric.update(output, labels)
            niou_metric.update(output, labels)
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break

    _, iou = iou_metric.get()
    _, niou = niou_metric.get()
    metrics = {
        "backbone": args.backbone_mode,
        "fusion": args.fuse_mode,
        "split": args.split,
        "num_samples": len(dataset),
        "evaluated_batches": batches,
        "batch_size": args.batch_size,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "iou": float(iou),
        "niou": float(niou),
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    save_metrics(out_dir, metrics)
    write_config(args, out_dir, len(dataset))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics_json={out_dir / 'metrics.json'}")
    print(f"metrics_csv={out_dir / 'metrics.csv'}")
    print(f"config_used={out_dir / 'config_used.txt'}")
    print("prediction_masks_saved=False")
    print("prediction_mask_note=This first-stage evaluator reports metrics only.")


if __name__ == "__main__":
    main()
