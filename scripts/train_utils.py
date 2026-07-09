"""
Shared utilities for YOLOv8 infrared small target detection training.

Handles: logging, checkpointing, loss curve plotting.
"""

import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """Logs training metrics and plots loss curves."""

    def __init__(self, save_dir, version_name):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.version_name = version_name
        self.metrics = {
            'train': {'loss': [], 'box_loss': [], 'cls_loss': [], 'dfl_loss': []},
            'val':   {'mAP50': [], 'mAP50-95': [], 'precision': [], 'recall': []},
            'epochs': [],
        }
        self.best_map = 0.0

    def log_train(self, epoch, loss_dict):
        self.metrics['epochs'].append(epoch)
        for k, v in loss_dict.items():
            if k in self.metrics['train']:
                self.metrics['train'][k].append(v)

    def log_val(self, metrics_dict):
        for k, v in metrics_dict.items():
            if k in self.metrics['val']:
                self.metrics['val'][k].append(v)
        current_map = metrics_dict.get('mAP50', 0)
        if current_map > self.best_map:
            self.best_map = current_map
            return True  # new best
        return False

    def save_metrics(self):
        path = self.save_dir / f'{self.version_name}_metrics.json'
        # Convert numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                return super().default(obj)
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2, cls=NpEncoder)

    def plot_loss_curve(self):
        """Plot training loss curves with epochs on x-axis."""
        train = self.metrics['train']
        epochs = self.metrics['epochs']
        if not epochs:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Total loss
        axes[0].plot(epochs, train['loss'], 'b-', linewidth=1.5)
        axes[0].set_title(f'{self.version_name} — Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

        # Sub-losses
        axes[1].plot(epochs, train['box_loss'], 'r-', label='Box', linewidth=1.5)
        axes[1].plot(epochs, train['cls_loss'], 'g-', label='Cls', linewidth=1.5)
        axes[1].plot(epochs, train['dfl_loss'], 'orange', label='DFL', linewidth=1.5)
        axes[1].set_title(f'{self.version_name} — Component Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Validation metrics
        val = self.metrics['val']
        if val['mAP50']:
            val_epochs = epochs[:len(val['mAP50'])] if len(epochs) >= len(val['mAP50']) else list(range(1, len(val['mAP50'])+1))
            axes[2].plot(val_epochs, val['mAP50'], 'b-', label='mAP50', linewidth=1.5)
            axes[2].plot(val_epochs, val['mAP50-95'], 'g-', label='mAP50-95', linewidth=1.5)
            axes[2].plot(val_epochs, val['precision'], 'orange', label='Precision', linewidth=1.5)
            axes[2].plot(val_epochs, val['recall'], 'r-', label='Recall', linewidth=1.5)
            axes[2].set_title(f'{self.version_name} — Validation Metrics')
            axes[2].set_xlabel('Epoch')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.save_dir / f'{self.version_name}_loss_curve.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path


def setup_yolo_train_args(
    data_yaml,
    project_dir='runs/train',
    version_name='exp',
    epochs=200,
    batch=16,
    imgsz=640,
    device='',
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
    optimizer='SGD',
    patience=50,
    resume=False,
    pretrained=False,
    seed=42,
):
    """
    Build a unified kwargs dict for YOLO.train().

    All hyperparameters tuned for infrared small target detection.
    """
    return dict(
        data=data_yaml,
        project=project_dir,
        name=version_name,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        optimizer=optimizer,
        patience=patience,
        resume=resume,
        pretrained=pretrained,
        seed=seed,
        # Tiny objects need lower NMS thresholds
        iou=0.5,
        conf=0.001,
        cos_lr=True,
        # Save & log
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )
