import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ) #block isinya conv, bn, relu 2 kali

    def forward(self, x):

        return self.block(x) #


class UNetEarlyFusion(nn.Module):

    def __init__(self, in_channels=20, num_classes=3):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.c4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)

        # Segmentation head
        self.out = nn.Conv2d(64, num_classes, 1) # act as decoder


    def forward(self, x):

        # Encoder
        d1 = self.d1( x) 
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        # Bottleneck
        b = self.bottleneck(self.pool(d4))

        # Decoder
        u4 = self.up4(b)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.c4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.c3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.c2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.c1(u1)

        out = self.out(u1)

        out = F.interpolate(
            out,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        return out



