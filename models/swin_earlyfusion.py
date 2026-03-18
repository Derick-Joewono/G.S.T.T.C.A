import torch 
import torch.nn as nn 
import timm 
import torch.nn.functional as F

class SwinEarlyFusionFPN(nn.Module):

    def __init__(self, in_channels=20, num_classes=3):#struktur model dibangun

        super().__init__() #constructior

        # Swin encoder
        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            in_chans=in_channels,
            features_only=True
        )

        encoder_channels = self.encoder.feature_info.channels()

        c1, c2, c3, c4 = encoder_channels

        # FPN lateral layers
        self.lateral4 = nn.Conv2d(c4, 256, 1)
        self.lateral3 = nn.Conv2d(c3, 256, 1)
        self.lateral2 = nn.Conv2d(c2, 256, 1)
        self.lateral1 = nn.Conv2d(c1, 256, 1)

        # smoothing conv
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, 3, padding=1)

        # final segmentation head
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)


    def forward(self, t1, t2): #pake struktur model utk forward propagation

        # Early fusion
        x = torch.cat([t1, t2], dim=1)

        # Encoder features
        features = self.encoder(x)

        f1, f2, f3, f4 = features

        # FPN top-down
        p4 = self.lateral4(f4)

        p3 = self.lateral3(f3) + F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(f2) + F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        p1 = self.lateral1(f1) + F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)

        out = self.head(p1)

        out = F.interpolate(out, size=(256,256), mode="bilinear", align_corners=False)

        return out