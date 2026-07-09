#!/usr/bin/env python3
"""
Train ACM (Asymmetric Context Modulation) on infrared small target dataset.

ACM is a segmentation-based method using ASKCResNet backbone with
asymmetric fusion. This adapter makes it work with SIRST-format data
(images + segmentation masks).

Expected data format (SIRST segmentation style):
  data/
    ACM_*/                          # One of: ACM_SIRST, ACM_NUAA, ACM_IRSTD
      images/                       # *.png images
      masks/                        # *_pixels0.png segmentation masks
      idx_427/
        trainval.txt                # one image name per line (no extension)
        test.txt                    # one image name per line (no extension)

Usage:
    python scripts/train_acm.py --data-dir data/ACM_SIRST --epochs 300
"""

import argparse
import sys
import os
import os.path as ops
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ACM imports
from acm.model.segmentation import ASKCResNetFPN
from acm.model.loss import SoftLoULoss
from acm.model.metrics import SigmoidMetric, SamplewiseSigmoidMetric
from acm.utils.lr_scheduler import adjust_learning_rate


# ── Adapted dataset ────────────────────────────────────────────────────

class SirstDataset(Data.Dataset):
    """Reads SIRST-format segmentation data for ACM."""

    def __init__(self, data_dir, mode='train', crop_size=480, base_size=512):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.crop_size = crop_size
        self.base_size = base_size

        txt_name = 'trainval.txt' if mode == 'train' else 'test.txt'
        self.list_path = self.data_dir / 'idx_427' / txt_name
        self.imgs_dir = self.data_dir / 'images'
        self.label_dir = self.data_dir / 'masks'

        with open(self.list_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(self.imgs_dir / f'{name}.png').convert('RGB')
        mask = Image.open(self.label_dir / f'{name}_pixels0.png')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        img = self.transform(img)
        mask = transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        if torch.rand(1) < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if torch.rand(1) < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        return img, mask


# ── Training ───────────────────────────────────────────────────────────

class ACMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset
        trainset = SirstDataset(args.data_dir, mode='train',
                                crop_size=args.crop_size, base_size=args.base_size)
        valset = SirstDataset(args.data_dir, mode='val',
                              crop_size=args.crop_size, base_size=args.base_size)
        self.train_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers)
        self.val_loader = Data.DataLoader(valset, batch_size=args.batch_size,
                                          num_workers=args.num_workers)

        # Model
        layer_blocks = [args.blocks_per_layer] * 3
        channels = [8, 16, 32, 64]
        self.net = ASKCResNetFPN(layer_blocks, channels, args.fuse_mode)
        self._weight_init(self.net)
        self.net = self.net.to(self.device)

        # Loss & optimizer
        self.criterion = SoftLoULoss()
        self.optimizer = torch.optim.Adagrad(self.net.parameters(),
                                             lr=args.lr, weight_decay=1e-4)

        # Metrics
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0

        # Logging
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.save_dir = Path(args.save_dir) / f'ACM_{args.fuse_mode}_{timestamp}'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)

        # Loss history for plotting
        self.train_losses = []
        self.eval_ious = []
        self.eval_nious = []
        self.eval_losses = []

        print(f'Device: {self.device}')
        print(f'Model: ASKCResNetFPN + {args.fuse_mode}')
        print(f'Training samples: {len(trainset)}')
        print(f'Validation samples: {len(valset)}')
        print(f'Save dir: {self.save_dir}')

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            self._train_epoch(epoch)
            self._val_epoch(epoch)

        # Plot final curves
        self._plot_curves()
        print(f'\nTraining complete! Best IoU: {self.best_iou:.4f}, Best nIoU: {self.best_nIoU:.4f}')

    def _train_epoch(self, epoch):
        losses = []
        self.net.train()
        tbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        for data, labels in tbar:
            data, labels = data.to(self.device), labels.to(self.device)

            output = self.net(data)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            tbar.set_postfix(loss=np.mean(losses))

        avg_loss = np.mean(losses)
        self.train_losses.append(avg_loss)
        adjust_learning_rate(self.optimizer, epoch, self.args.epochs,
                             self.args.lr, self.args.warm_up_epochs, 1e-6)

        cur_lr = self.optimizer.param_groups[0]['lr']
        tbar.write(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {cur_lr:.6f}')

    def _val_epoch(self, epoch):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        losses = []
        self.net.eval()

        tbar = tqdm(self.val_loader, desc=f'Val   Epoch {epoch}')
        for data, labels in tbar:
            with torch.no_grad():
                output = self.net(data.to(self.device)).cpu()

            loss = self.criterion(output, labels)
            losses.append(loss.item())

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            tbar.set_postfix(IoU=IoU, nIoU=nIoU)

        avg_loss = np.mean(losses)
        _, IoU = self.iou_metric.get()
        _, nIoU = self.nIoU_metric.get()
        self.eval_losses.append(avg_loss)
        self.eval_ious.append(IoU)
        self.eval_nious.append(nIoU)

        tbar.write(f'Epoch {epoch:3d} | Val Loss: {avg_loss:.4f} | IoU: {IoU:.4f} | nIoU: {nIoU:.4f}')

        # Save checkpoint
        is_best_iou = IoU > self.best_iou
        is_best_nIoU = nIoU > self.best_nIoU
        if is_best_iou or is_best_nIoU:
            ckpt_path = self.save_dir / 'checkpoints' / f'Epoch-{epoch:03d}_IoU-{IoU:.4f}_nIoU-{nIoU:.4f}.pth'
            torch.save(self.net.state_dict(), ckpt_path)
            if is_best_iou:
                self.best_iou = IoU
            if is_best_nIoU:
                self.best_nIoU = nIoU

    def _weight_init(self, m):
        for module in m.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.normal_(module.bias, 0)

    def _plot_curves(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.train_losses) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs, self.eval_losses, 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'ACM+{self.args.fuse_mode} Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.eval_ious, 'g-', label='IoU')
        axes[1].plot(epochs, self.eval_nious, 'orange', label='nIoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title(f'ACM+{self.args.fuse_mode} Validation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Curves saved to {self.save_dir / "training_curves.png"}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train ACM on infrared small target dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to SIRST-format data directory')
    parser.add_argument('--save-dir', type=str, default='runs/train',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--crop-size', type=int, default=480)
    parser.add_argument('--base-size', type=int, default=512)
    parser.add_argument('--warm-up-epochs', type=int, default=0)
    parser.add_argument('--blocks-per-layer', type=int, default=4)
    parser.add_argument('--fuse-mode', type=str, default='AsymBi',
                        choices=['BiLocal', 'AsymBi', 'BiGlobal'])
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = ACMTrainer(args)
    trainer.train()
