import torch 
import torch.nn as nn 
import timm 
import torch.nn.functional as F

class SwinEarlyFusionFPN(nn.Module):
    def __init__(self, in_channels=20, num_classes=3):
        super().__init__()

        # Swin encoder
        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
            img_size=256,
            window_size=8
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

    def forward(self, x):
        # 1. Encoder features
        # TIMM Swin dengan features_only=True biasanya mengeluarkan list tensor
        # Namun dimensinya sering dalam bentuk [B, H, W, C] bukan [B, C, H, W]
        features = self.encoder(x)
        
        # 🔥 PERBAIKAN: Permute fitur agar sesuai dengan Conv2d (B, C, H, W)
        # Kita cek jika dimensi terakhir adalah channel, kita tukar.
        formatted_features = []
        for f in features:
            if f.dim() == 4 and f.shape[-1] > f.shape[1]: # Jika formatnya B, H, W, C
                f = f.permute(0, 3, 1, 2).contiguous()
            formatted_features.append(f)
            
        f1, f2, f3, f4 = formatted_features

        # 2. FPN top-down
        p4 = self.lateral4(f4)

        p3 = self.lateral3(f3) + F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(f2) + F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        p1 = self.lateral1(f1) + F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)

        # 3. Head
        out = self.head(p1)
        out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)

        return out