import torch
import torch.nn as nn
import torch.nn.functional as F

# Mencoba import sub-komponen Focal & Dice
try:
    from loss.boundary_loss import FocalLoss, DiceLoss
except ModuleNotFoundError:
    from boundary_loss import FocalLoss, DiceLoss

class FocalDiceCELoss(nn.Module):
    def __init__(self, raw_weights, ce_weight=0.5, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        
        # 1. Smoothing Bobot (Square Root)
        # Digunakan agar gap antar kelas tidak terlalu ekstrim saat training
        smoothed_weights = torch.sqrt(raw_weights)
        self.register_buffer('weights', smoothed_weights)

        # 2. Inisialisasi Komponen Loss
        self.ce = nn.CrossEntropyLoss(weight=self.weights, ignore_index=-1)
        self.focal = FocalLoss(weights=self.weights, gamma=2, alpha=0.25)
        self.dice = DiceLoss(weights=self.weights)

        # 3. Koefisien Bobot Loss (Alpha/Beta/Gamma)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        # Hitung masing-masing komponen
        l_ce = self.ce(pred, target)
        l_focal = self.focal(pred, target)
        l_dice = self.dice(pred, target)
        
        # Total Hybrid Loss
        total = (self.ce_weight * l_ce + 
                 self.focal_weight * l_focal + 
                 self.dice_weight * l_dice)
        
        # Mengembalikan total loss (untuk backward) dan dict detail (untuk logging)
        return total, {
            "ce": l_ce.item(), 
            "focal": l_focal.item(), 
            "dice": l_dice.item()
        }