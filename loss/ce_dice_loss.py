import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):

        # pred shape
        # (B,C,H,W)

        num_classes = pred.shape[1]

        # softmax probability
        pred = F.softmax(pred, dim=1)

        # one-hot encode target
        target_onehot = F.one_hot(target.long(), num_classes)
        target_onehot = target_onehot.permute(0,3,1,2).float()

        # flatten
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target_onehot = target_onehot.contiguous().view(target_onehot.shape[0], target_onehot.shape[1], -1)

        intersection = (pred * target_onehot).sum(dim=2)

        dice = (2 * intersection + self.smooth) / (
            pred.sum(dim=2) + target_onehot.sum(dim=2) + self.smooth
        )

        dice_loss = 1 - dice.mean()

        return dice_loss


class CrossEntropyDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):

        ce_loss = self.ce(pred, target.long())

        dice_loss = self.dice(pred, target)

        return ce_loss + dice_loss
