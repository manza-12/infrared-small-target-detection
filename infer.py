#!/usr/bin/env python3
"""
Unified inference script for infrared small target detection.

Supports all three model variants:
  - YOLOv8n baseline
  - YOLOv8n + PConv
  - YOLOv8n + PConv + SD Loss

Usage:
    # Single image
    python infer.py --weights runs/train/yolov8n_baseline/weights/best.pt --source image.jpg

    # Image folder
    python infer.py --weights runs/train/yolov8n_pconv/weights/best.pt --source ./images/

    # Video
    python infer.py --weights runs/train/yolov8n_pconv_sd/weights/best.pt --source video.mp4

    # Compare all three
    python infer.py --weights baseline.pt pconv.pt pconv_sd.pt --source image.jpg --compare

    # Use PConv model architecture (needed for PConv weights)
    python infer.py --weights pconv.pt --source image.jpg --pconv
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from models.model_factory import create_pconv_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLOv8 Infrared Small Target Detection — Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--weights', type=str, nargs='+', required=True,
                        help='Model weight paths (one or more for comparison)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, folder, or video')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--device', type=str, default='',
                        help='Device: "0" GPU, "cpu" CPU, "" auto')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save output images')
    parser.add_argument('--show', action='store_true',
                        help='Display results in a window')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='',
                        help='Output subdirectory name')
    parser.add_argument('--compare', action='store_true',
                        help='Arrange multiple model outputs side-by-side')
    parser.add_argument('--pconv', action='store_true',
                        help='Use PConv model architecture (needed for PConv weights)')
    parser.add_argument('--pconv-sd', action='store_true',
                        help='Use PConv + SD Loss model architecture')
    parser.add_argument('--max-det', type=int, default=300,
                        help='Maximum detections per image')
    parser.add_argument('--line-width', type=int, default=2,
                        help='Bounding box line width')
    return parser.parse_args()


def load_model(weight_path, use_pconv=False, use_pconv_sd=False):
    """
    Load model from weights.

    For PConv variants, we need to create the custom architecture first,
    then load the state dict.
    """
    if use_pconv or use_pconv_sd:
        print(f'Loading PConv architecture and weights from {weight_path}...')
        model = create_pconv_model()
        # Load weights
        ckpt = torch.load(weight_path, map_location='cpu', weights_only=True)
        if 'model' in ckpt:
            model.model.load_state_dict(ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'state_dict')
                                        else ckpt['model'])
        else:
            model.model.load_state_dict(ckpt)
        return model
    else:
        print(f'Loading standard YOLO model from {weight_path}...')
        return YOLO(weight_path)


def draw_detections(img, detections, names, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image."""
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        # Scale to original image size if needed
        label = f'{names[int(cls_id)]} {conf:.2f}'
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(img, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img


def run_inference(args):
    if args.no_save:
        args.save = False

    # Models
    models = []
    model_names = []
    for w in args.weights:
        use_pconv = args.pconv or 'pconv' in Path(w).name.lower()
        model = load_model(w, use_pconv=use_pconv)
        models.append(model)
        model_names.append(Path(w).stem)

    # Source: single image or folder
    source_path = Path(args.source)
    if source_path.is_dir():
        image_paths = sorted(source_path.glob('*.[jp][pn]g')) + sorted(source_path.glob('*.bmp'))
    else:
        image_paths = [source_path]

    # Run
    for img_path in image_paths:
        results_list = []
        for i, model in enumerate(models):
            results = model(
                str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                max_det=args.max_det,
                save=args.save,
                project=args.project,
                name=args.name or model_names[i],
                exist_ok=True,
                line_width=args.line_width,
            )
            results_list.append(results)

            n_det = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f'  [{model_names[i]}] {img_path.name}: {n_det} detections')

        # Side-by-side comparison
        if args.compare and len(results_list) > 1:
            comparison = make_comparison_grid(results_list, model_names, img_path)
            if comparison is not None:
                save_dir = Path(args.project) / 'comparison'
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = save_dir / f'compare_{img_path.stem}.jpg'
                cv2.imwrite(str(out_path), comparison)
                print(f'  Comparison saved: {out_path}')

    print('Inference complete!')


def make_comparison_grid(results_list, model_names, img_path):
    """Stack model outputs side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (results, name) in enumerate(zip(results_list, model_names)):
        plotted = results[0].plot()
        axes[i].imshow(plotted[..., ::-1])  # BGR → RGB
        axes[i].set_title(name, fontsize=14)
        axes[i].axis('off')

    plt.tight_layout()
    save_dir = Path('runs/detect/comparison')
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'compare_{img_path.stem}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return cv2.imread(str(out_path))


def main():
    args = parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
