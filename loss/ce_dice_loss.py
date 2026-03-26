import torch
import torch.nn as nn
import torch.nn.functional as F
# ATAU gunakan full path jika dijalankan dari root
try:
    from loss.boundary_loss import FocalLoss, DiceLoss
except ModuleNotFoundError:
    from boundary_loss import FocalLoss, DiceLoss

class FocalDiceCELoss(nn.Module):
    def __init__(self, raw_weights, ce_weight=0.5, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        
        # Smoothing Bobot
        smoothed_weights = torch.sqrt(raw_weights)
        self.register_buffer('weights', smoothed_weights)

        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.focal = FocalLoss(weights=self.weights, gamma=2, alpha=0.25)
        self.dice = DiceLoss(weights=self.weights)

        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        l_ce = self.ce(pred, target)
        l_focal = self.focal(pred, target)
        l_dice = self.dice(pred, target)
        
        total = (self.ce_weight * l_ce + 
                 self.focal_weight * l_focal + 
                 self.dice_weight * l_dice)
        
        return total, {"ce": l_ce.item(), "focal": l_focal.item(), "dice": l_dice.item()}