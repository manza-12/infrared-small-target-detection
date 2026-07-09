"""
SD Loss (Scale-based Dynamic Loss) for infrared small target detection.

Reference: JN-Yang/PConv-SDloss-Data
Includes: SDIoU (Scale-based Dynamic IoU), SD Loss head integration
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_sdiou(box1, box2, xywh=True, eps=1e-7, delta=0.5):
    """
    Scale-based Dynamic IoU (SDIoU).

    Adjusts the IoU loss dynamically based on target scale — small targets
    get higher gradient weight to improve detection of dim/small objects.

    Args:
        box1: predicted boxes (1, 4) or (N, 4)
        box2: target boxes (N, 4)
        xywh: if True, input is (x, y, w, h); else (x1, y1, x2, y2)
        delta: scale modulation factor
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
        w1, h1 = w1.clamp(min=eps), h1.clamp(min=eps)
        w2, h2 = w2.clamp(min=eps), h2.clamp(min=eps)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
        w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)

    # Intersection
    inter = (
        (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
        * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    )
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Convex envelope
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    # Aspect ratio consistency
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    # Scale modulation: small targets → larger beta → more gradient weight
    beta = (w2 * h2 * delta) / 81
    beta = torch.where(beta > delta, torch.full_like(beta, delta), beta)

    sdiou = delta - beta + (1 - delta + beta) * (iou - v * alpha) - (1 + delta - beta) * (rho2 / c2)
    return sdiou


class SDIoULoss(nn.Module):
    """
    SDIoU-based bounding box loss for small target detection.

    Designed to replace standard IoU loss in YOLOv8's BboxLoss.
    """

    def __init__(self, delta=0.5):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, xywh=True):
        return 1.0 - bbox_sdiou(pred, target, xywh=xywh, delta=self.delta)


class SDLoss(nn.Module):
    """
    SD Loss wrapper that integrates with ultralytics v8DetectionLoss.

    This hooks into the existing YOLOv8 loss pipeline and replaces the
    standard IoU branch with SDIoU for small-target-aware training.

    Usage:
        loss_fn = v8DetectionLoss(model)
        # Replace the bbox loss with SDIoU
        loss_fn.bbox_loss = SDLoss.from_bbox_loss(loss_fn.bbox_loss, delta=0.5)
    """

    @staticmethod
    def from_bbox_loss(bbox_loss, delta=0.5):
        """
        Convert a standard BboxLoss to use SDIoU.

        Args:
            bbox_loss: ultralytics.utils.loss.BboxLoss instance
            delta: SDIoU scale modulation factor

        Returns:
            Modified BboxLoss with SDIoU
        """
        original_forward = bbox_loss.forward

        def sdiou_forward(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                          target_scores, target_scores_sum, fg_mask):
            # Weight by target scores
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            # Use SDIoU instead of standard IoU
            iou = bbox_sdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                             xywh=False, delta=delta)
            loss = ((1.0 - iou) * weight).sum() / target_scores_sum

            # DFL loss (unchanged)
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
        return bbox_loss
