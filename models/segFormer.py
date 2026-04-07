import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerConfig
from transformers import SegformerForSemanticSegmentation


class SegFormerEarlyFusion(nn.Module):
    def __init__(self, in_channels=20, num_classes=3):
        super().__init__()
        config = SegformerConfig(num_channels=in_channels, num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation(config)

    # CUKUP TERIMA 'x' SAJA
    def forward(self, x): 

        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        
        logits = F.interpolate(
            logits, size=(256, 256), mode="bilinear", align_corners=False
        )
        return logits


# initialization
model3 = SegFormerEarlyFusion(in_channels=20, num_classes=3)
