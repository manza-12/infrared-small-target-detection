#!/usr/bin/env python3
"""
Train YOLOv8n + PConv on infrared small target detection dataset.

Backbone's first two Conv layers are replaced with Pinwheel-shaped Convolution (PConv).

Usage:
    python scripts/train_pconv.py --data data.yaml [--epochs 200] [--batch 16]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from models.model_factory import create_pconv_model
from scripts.train_utils import setup_yolo_train_args


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8n + PConv')
    parser.add_argument('--data', type=str, default='data/data.yaml')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='yolov8n_pconv')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--pretrained', action='store_true',
                        help='Start from baseline YOLOv8n weights (transfer learning)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print('=' * 60)
    print('YOLOv8n + PConv — Infrared Small Target Detection')
    print('=' * 60)

    # Build PConv model
    model = create_pconv_model()

    # Optionally load baseline pretrained weights (partial transfer)
    if args.pretrained:
        print('Loading baseline YOLOv8n weights (partial transfer)...')
        base = YOLO('yolov8n.pt')
        pretrained_dict = base.model.state_dict()
        model_dict = model.model.state_dict()

        # Filter out mismatched keys (first two Conv layers → PConv)
        compatible = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        skipped = len(model_dict) - len(compatible)
        model.model.load_state_dict(compatible, strict=False)
        print(f'  Loaded {len(compatible)}/{len(model_dict)} weight groups ({skipped} skipped due to PConv)')

    # Build training args
    train_args = setup_yolo_train_args(
        data_yaml=args.data,
        project_dir=args.project,
        version_name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        pretrained=False,  # We handle pretrained manually above
        resume=args.resume,
        seed=args.seed,
    )

    # Train
    results = model.train(**train_args)

    # Validate
    print('\nValidating best model...')
    val_results = model.val()
    print(f'mAP50: {val_results.box.map50:.4f}')
    print(f'mAP50-95: {val_results.box.map:.4f}')

    # Export
    print('\nExporting to ONNX...')
    export_path = model.export(format='onnx')
    print(f'Exported to: {export_path}')

    print(f'\n✅ PConv training complete!')
    print(f'   Best weights: {args.project}/{args.name}/weights/best.pt')


if __name__ == '__main__':
    main()
