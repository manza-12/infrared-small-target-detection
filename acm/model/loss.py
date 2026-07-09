import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    """
    Combined SoftIoU + BCE loss for infrared small target detection.

    Pure SoftIoU has extremely small gradients for small targets
    (e.g., target ~10 pixels in a 480x480 image -> |grad| ~ 1e-5).
    Adding BCE with per-batch pos_weight provides per-pixel gradient
    signal to drive learning even when the predicted mask has zero overlap
    with the ground truth.
    """
    def __init__(self, bce_weight=1.0):
        super(SoftLoULoss, self).__init__()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        # --- SoftIoU loss (region-level supervision) ---
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1.0

        intersection = pred_sigmoid * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred_sigmoid, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        soft_iou = (intersection_sum + smooth) / \
                   (pred_sum + target_sum - intersection_sum + smooth)
        soft_iou_loss = 1.0 - torch.mean(soft_iou)

        # --- Weighted BCE loss (pixel-level supervision) ---
        # Dynamically compute pos_weight to compensate for extreme
        # foreground/background imbalance in small-target images.
        num_pos = target_sum                        # shape: (B,)
        num_pixels = target[0].numel()              # H * W per image
        num_neg = num_pixels - num_pos

        # pos_weight = #neg / #pos, clamped to avoid overflow on empty masks
        pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=1000.0)

        # Per-sample BCE (pos_weight must be per-sample for proper weighting)
        bce_losses = []
        for b in range(pred.shape[0]):
            bce = F.binary_cross_entropy_with_logits(
                pred[b:b + 1], target[b:b + 1],
                pos_weight=pos_weight[b:b + 1]
            )
            bce_losses.append(bce)
        bce_loss = torch.stack(bce_losses).mean()

        return soft_iou_loss + self.bce_weight * bce_loss