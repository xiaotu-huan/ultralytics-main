# Ultralytics 🚀 Focal-EIoU Loss
# Reference: Focal-EIoU Loss: Improving Bounding Box Regression in Object Detection (Zhang et al. 2022)
# https://arxiv.org/abs/2207.08464

import torch
import torch.nn as nn

class FocalEIoULoss(nn.Module):
    """
    Focal-EIoU loss implementation for bounding box regression.

    Combines Efficient IoU (EIoU) and Focal modulation to better focus on hard examples.
    """

    def __init__(self, gamma: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted boxes [N, 4] in xyxy format
            target: ground-truth boxes [N, 4] in xyxy format
        Returns:
            Focal-EIoU loss tensor
        """
        # Ensure valid shapes
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        # Split coordinates
        px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        gx1, gy1, gx2, gy2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Width, height, area
        pw, ph = (px2 - px1).clamp(min=1e-6), (py2 - py1).clamp(min=1e-6)
        gw, gh = (gx2 - gx1).clamp(min=1e-6), (gy2 - gy1).clamp(min=1e-6)
        pa, ga = pw * ph, gw * gh

        # Intersection
        ix1, iy1 = torch.max(px1, gx1), torch.max(py1, gy1)
        ix2, iy2 = torch.min(px2, gx2), torch.min(py2, gy2)
        iw, ih = (ix2 - ix1).clamp(min=0), (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        union = pa + ga - inter + 1e-7
        iou = inter / union

        # Centers
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2

        # Distance between centers
        center_dist = (pcx - gcx).pow(2) + (pcy - gcy).pow(2)

        # Enclosing box
        cx1, cy1 = torch.min(px1, gx1), torch.min(py1, gy1)
        cx2, cy2 = torch.max(px2, gx2), torch.max(py2, gy2)
        cw, ch = cx2 - cx1, cy2 - cy1
        c2 = cw.pow(2) + ch.pow(2) + 1e-7

        # EIoU components
        iou_term = 1 - iou
        cw_term = ((pw - gw).pow(2)) / (cw.pow(2) + 1e-7)
        ch_term = ((ph - gh).pow(2)) / (ch.pow(2) + 1e-7)
        cd_term = center_dist / c2

        eiou = iou_term + cd_term + cw_term + ch_term

        # Focal weighting: focus on low IoU samples
        weight = (1 - iou).pow(self.gamma)
        loss = eiou * weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
