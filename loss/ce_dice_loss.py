import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from loss.boundary_loss import FocalLoss, DiceLoss
except ModuleNotFoundError:
    # Fallback jika dijalankan lokal di folder yang sama
    from boundary_loss import FocalLoss, DiceLoss

class FocalDiceCELoss(nn.Module):
    def __init__(self, ce_weight=0.5, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        
        # SARAN WEIGHT BARU (70:30 Ratio)
        # 0: No Change, 1: Deforest, 2: Forest
        raw_weights = torch.tensor([0.4381, 2.9007, 2.6848])
        
        # Gunakan langsung atau smoothing ringan
        # Untuk GSWIN-TAC, disarankan gunakan langsung agar model lebih sensitif
        self.register_buffer('weights', raw_weights)

        # Inisialisasi komponen dengan weights baru
        self.ce = nn.CrossEntropyLoss(weight=self.weights, ignore_index=-1)
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
        
        # Return detail untuk logging
        return total, {
            "ce": l_ce.item(), 
            "focal": l_focal.item(), 
            "dice": l_dice.item()
        }