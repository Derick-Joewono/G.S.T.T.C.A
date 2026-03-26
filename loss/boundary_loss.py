import torch
import torch.nn as nn
import torch.nn.functional as F

class DeforestationTotalLoss(nn.Module):
    def __init__(self, raw_weights, ce_weight=0.5, focal_weight=1.0, dice_weight=1.0, boundary_weight=0.5):
        super().__init__()
        
        # 1. SQUARE ROOT SMOOTHING
        # Meredam [1.0, 17.5, 16.5] -> [1.0, 4.18, 4.06] agar tidak over-bias
        smoothed_weights = torch.sqrt(raw_weights)
        self.register_buffer('weights', smoothed_weights)

        # 2. KOMPONEN LOSS
        self.ce = nn.CrossEntropyLoss(ignore_index=-1) # Anchor stabilitas global
        self.focal = FocalLoss(weights=self.weights, gamma=2, alpha=0.25) # Alpha untuk balancing
        self.dice = DiceLoss(weights=self.weights)
        self.boundary = BoundaryLoss()

        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        l_ce = self.ce(pred, target)
        l_focal = self.focal(pred, target)
        l_dice = self.dice(pred, target)
        l_boundary = self.boundary(pred, target)
        
        total = (self.ce_weight * l_ce + 
                 self.focal_weight * l_focal + 
                 self.dice_weight * l_dice + 
                 self.boundary_weight * l_boundary)
        
        # Return total DAN detail untuk monitoring di Tensorboard/WandB
        return total, {
            "ce": l_ce.item(),
            "focal": l_focal.item(),
            "dice": l_dice.item(),
            "boundary": l_boundary.item()
        }

# --- Sub-komponen Pendukung ---

class FocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.weights = weights
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights, ignore_index=-1)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

class DiceLoss(nn.Module):
    def __init__(self, weights=None, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.weights = weights

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        mask = (target != -1).float().view(target.shape[0], 1, -1)
        
        target_clean = target.clone()
        target_clean[target == -1] = 0
        target_onehot = F.one_hot(target_clean.long(), num_classes).permute(0, 3, 1, 2).float()

        pred_softmax = F.softmax(pred, dim=1).view(pred.shape[0], num_classes, -1)
        target_onehot = target_onehot.view(target_onehot.shape[0], num_classes, -1)

        pred_softmax = pred_softmax * mask
        target_onehot = target_onehot * mask

        intersection = (pred_softmax * target_onehot).sum(dim=2)
        cardinality = pred_softmax.sum(dim=2) + target_onehot.sum(dim=2)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        if self.weights is not None:
            return 1 - (dice_score.mean(dim=0) * self.weights).sum() / self.weights.sum()
        return 1 - dice_score.mean()

class BoundaryLoss(nn.Module):
    def compute_boundary(self, x):
        laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                        dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        C = x.shape[1]
        laplacian_kernel = laplacian_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, laplacian_kernel, padding=1, groups=C).abs()

    def forward(self, pred, target):
        mask = (target != -1).float().unsqueeze(1)
        target_clean = target.clone()
        target_clean[target == -1] = 0
        
        target_onehot = F.one_hot(target_clean.long(), pred.shape[1]).permute(0, 3, 1, 2).float()
        pred_prob = torch.softmax(pred, dim=1)
        
        b_pred = self.compute_boundary(pred_prob)
        b_gt = self.compute_boundary(target_onehot)
        
        return F.l1_loss(b_pred * mask, b_gt * mask)