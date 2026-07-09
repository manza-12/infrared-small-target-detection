"""
Model factory: create model variants programmatically (YAML-free for custom modules).
"""

from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.ops import make_divisible
from ultralytics.nn.tasks import BaseModel, parse_model, DetectionModel
from ultralytics.nn.modules import Conv as UltralyticsConv

from models.pconv import PConv


def create_baseline_model(scale='n', nc=1):
    """Create standard YOLOv8n baseline model."""
    return YOLO(f'yolov8{scale}.yaml').model  # returns the underlying nn.Module (DetectionModel)


def _get_scaled_channels(c2, width=0.25, max_channels=1024):
    """Apply YOLOv8n scaling to channel count."""
    return make_divisible(min(c2, max_channels) * width, 8)


def create_pconv_model(scale='n', nc=1):
    """
    Create YOLOv8n + PConv model by loading baseline and surgically
    replacing the first two Conv layers with PConv.
    """
    # Load standard YOLOv8n
    base = YOLO(f'yolov8{scale}.yaml')

    # Access the DetectionModel's model (nn.Sequential)
    seq = base.model.model  # nn.Sequential of layers

    # ── Determine scaled channels ──
    width_map = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
    width = width_map.get(scale, 0.25)

    # Layer 0: Conv(3, 64, 3, 2) → PConv
    c1_0 = 3  # input channels
    c2_0_raw = 64
    c2_0 = _get_scaled_channels(c2_0_raw, width)

    # Layer 1: Conv(c2_0, 128, 3, 2) → PConv
    c1_1 = c2_0
    c2_1_raw = 128
    c2_1 = _get_scaled_channels(c2_1_raw, width)

    # ── Build PConv replacements ──
    pconv0 = PConv(c1_0, c2_0, k=3, s=2)
    pconv1 = PConv(c1_1, c2_1, k=3, s=2)

    # ── Attach metadata required by BaseModel._predict_once ──
    # Each layer needs .i (index), .f (from), .type (string)
    old0 = seq[0]
    old1 = seq[1]

    pconv0.i, pconv0.f, pconv0.type = old0.i, old0.f, 'models.pconv.PConv'
    pconv1.i, pconv1.f, pconv1.type = old1.i, old1.f, 'models.pconv.PConv'
    pconv0.np = sum(x.numel() for x in pconv0.parameters())
    pconv1.np = sum(x.numel() for x in pconv1.parameters())

    # ── Replace in-place ──
    seq[0] = pconv0
    seq[1] = pconv1

    return base


class SDLossPatcher:
    """
    Patch YOLOv8's loss function to use SDIoU for small-target-aware training.

    Usage:
        model = create_pconv_model()
        SDLossPatcher.patch(model, delta=0.5)

        # Then train normally
        model.train(data='data.yaml', epochs=100, ...)
    """

    @staticmethod
    def patch(model, delta=0.5):
        """
        Monkey-patch the model's BboxLoss to use SDIoU.

        Works with ultralytics v8DetectionLoss — patches the bbox_loss
        before training begins by hooking into the loss creation.
        """
        # Store delta for later use
        model._sdiou_delta = delta

        # v8DetectionLoss creates BboxLoss internally.
        # We patch after model.train() creates the loss,
        # by wrapping the _initialize_loss method.
        if hasattr(model, 'model') and hasattr(model.model, 'loss'):
            # Loss already exists
            SDLossPatcher._apply_patch(model.model.loss, delta)

    @staticmethod
    def _apply_patch(loss_fn, delta):
        """Apply SDIoU patch to an existing v8DetectionLoss."""
        if not hasattr(loss_fn, 'bbox_loss'):
            return

        bbox_loss = loss_fn.bbox_loss
        original_forward = bbox_loss.forward

        from models.sd_loss import bbox_sdiou

        def sdiou_forward(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                          target_scores, target_scores_sum, fg_mask):
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_sdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                             xywh=False, delta=delta)
            loss = ((1.0 - iou) * weight).sum() / target_scores_sum

            if bbox_loss.use_dfl:
                from ultralytics.utils.ops import bbox2dist
                target_ltrb = bbox2dist(anchor_points, target_bboxes, bbox_loss.reg_max)
                loss_dfl = bbox_loss._df_loss(
                    pred_dist[fg_mask].view(-1, bbox_loss.reg_max + 1),
                    target_ltrb[fg_mask]
                ) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.0, device=pred_dist.device)

            return loss, loss_dfl

        bbox_loss.forward = sdiou_forward
