#!/usr/bin/env python3
"""
Train YOLOv8n baseline on infrared small target detection dataset.

Usage:
    python scripts/train_baseline.py --data data.yaml [--epochs 200] [--batch 16]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from scripts.train_utils import setup_yolo_train_args


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8n baseline')
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='',
                        help='e.g. "0" for GPU 0, "" for auto, "cpu" for CPU')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='yolov8n_baseline')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--pretrained', action='store_true',
                        help='Start from COCO pretrained weights')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print('=' * 60)
    print('YOLOv8n Baseline — Infrared Small Target Detection')
    print('=' * 60)

    # Load model
    model = YOLO('yolov8n.yaml')
    if args.pretrained:
        print('Loading COCO pretrained weights...')
        model = YOLO('yolov8n.pt')

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
        pretrained=args.pretrained,
        resume=args.resume,
        seed=args.seed,
    )

    # Train
    results = model.train(**train_args)

    # Validate on best model
    print('\nValidating best model...')
    val_results = model.val()
    print(f'mAP50: {val_results.box.map50:.4f}')
    print(f'mAP50-95: {val_results.box.map:.4f}')

    # Export to ONNX
    print('\nExporting to ONNX...')
    export_path = model.export(format='onnx')
    print(f'Exported to: {export_path}')

    print(f'\n✅ Baseline training complete!')
    print(f'   Best weights: {args.project}/{args.name}/weights/best.pt')
    print(f'   Training log: {args.project}/{args.name}/')


if __name__ == '__main__':
    main()
