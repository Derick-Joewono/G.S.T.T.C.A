import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class STANet(nn.Module):

    def __init__(self, in_channels=10, num_classes=3):
        super().__init__()

        # =====================================================
        # 1. Backbone Encoder (ResNet18)
        # =====================================================

        backbone = models.resnet18(weights="DEFAULT")

        # Modify first convolution for multi-channel input
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Remove classifier layers
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])


        # =====================================================
        # 2. Change Feature Processing
        # =====================================================

        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)


        # =====================================================
        # 3. Spatial Attention
        # =====================================================

        self.attention = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )


        # =====================================================
        # 4. Segmentation Head
        # =====================================================

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)


    def forward(self, t1, t2):

        # =====================================================
        # 1. Siamese Feature Extraction
        # =====================================================

        f1 = self.encoder(t1)  # (B,512,8,8)
        f2 = self.encoder(t2)  # (B,512,8,8)


        # =====================================================
        # 2. Change Representation
        # =====================================================

        diff = torch.abs(f1 - f2)


        # =====================================================
        # 3. Feature Processing
        # =====================================================

        x = self.conv(diff)


        # =====================================================
        # 4. Attention Refinement
        # =====================================================

        att = self.attention(x)
        x = x * att


        # =====================================================
        # 5. Segmentation Prediction
        # =====================================================

        out = self.classifier(x)


        # =====================================================
        # 6. Upsample to original resolution
        # =====================================================

        out = F.interpolate(
            out,
            size=(256, 256),
            mode="bilinear",
            align_corners=True
        )

        return out


# Example initialization
model = STANet(in_channels=10, num_classes=3)
