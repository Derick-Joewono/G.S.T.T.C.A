import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# DICE LOSS
# ==========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1):

        super().__init__()

        self.smooth = smooth

    def forward(self, pred, target):

        pred = torch.softmax(pred, dim=1)

        target = F.one_hot(
            target,
            num_classes=pred.shape[1]
        ).permute(0, 3, 1, 2).float()

        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        intersection = (pred * target).sum(-1)

        dice = (2 * intersection + self.smooth) / (
            pred.sum(-1) + target.sum(-1) + self.smooth
        )

        return 1 - dice.mean()


# ==========================================================
# BOUNDARY LOSS
# ==========================================================

class BoundaryLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def compute_boundary(self, mask):

        laplacian_kernel = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]],
            dtype=torch.float32,
            device=mask.device
        ).view(1, 1, 3, 3)

        mask = mask.unsqueeze(1).float()

        boundary = F.conv2d(
            mask,
            laplacian_kernel,
            padding=1
        )

        boundary = boundary.abs()

        return boundary


    def forward(self, pred, target):

        pred = torch.softmax(pred, dim=1)

        pred_boundary = self.compute_boundary(
            pred.max(dim=1)[0])

        gt_boundary = self.compute_boundary(target)

        loss = F.l1_loss(pred_boundary, gt_boundary)

        return loss


# ==========================================================
# TOTAL LOSS
# ==========================================================

class TotalLoss(nn.Module):

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=1.0
    ):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):

        ce = self.ce(pred, target)

        dice = self.dice(pred, target)

        boundary = self.boundary(pred, target)

        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.boundary_weight * boundary
        )

        return total
