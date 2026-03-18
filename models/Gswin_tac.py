import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ==========================================================
# GCN AUXILIARY ENCODER
# ==========================================================

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):

        x = torch.matmul(adj, x)
        x = self.fc(x)

        return x


class GCNEncoder(nn.Module):

    def __init__(self, in_channels=512):

        super().__init__()

        self.gcn1 = GCNLayer(in_channels, in_channels)
        self.gcn2 = GCNLayer(in_channels, in_channels)

        self.downsample = nn.Conv2d(
            in_channels,
            1024,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):

        B, C, H, W = x.shape

        nodes = x.view(B, C, -1).permute(0, 2, 1)

        N = nodes.shape[1]

        adj = torch.eye(N).to(x.device)

        nodes = self.gcn1(nodes, adj)
        nodes = self.gcn2(nodes, adj)

        nodes = nodes.permute(0, 2, 1).view(B, C, H, W)

        out = self.downsample(nodes)

        return out


# ==========================================================
# FEATURE AGGREGATION MODULE
# ==========================================================

class GatedFAM(nn.Module):

    def __init__(self, channels=1024):

        super().__init__()

        self.gate_conv = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, swin_feat, gcn_feat):

        concat = torch.cat([swin_feat, gcn_feat], dim=1)

        gate = self.sigmoid(self.gate_conv(concat))

        out = gate * swin_feat + (1 - gate) * gcn_feat

        return out


# ==========================================================
# TEMPORAL CROSS ATTENTION
# ==========================================================

class TemporalCrossAttention(nn.Module):

    def __init__(self, dim=1024, heads=8):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.fusion_conv = nn.Conv2d(
            dim * 2,
            dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, f1, f2):

        B, C, H, W = f1.shape

        q = f1.flatten(2).transpose(1, 2)
        k = f2.flatten(2).transpose(1, 2)
        v = f2.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(q, k, v)

        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

        diff = torch.abs(f1 - f2)

        fusion = torch.cat([attn_out, diff], dim=1)

        out = self.fusion_conv(fusion)

        return out


# ==========================================================
# UNET DECODER
# ==========================================================

class DecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x, skip):

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x


# ==========================================================
# MAIN MODEL
# ==========================================================

class GSWIN_TAC(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        self.gcn = GCNEncoder()

        self.fam = GatedFAM()

        self.temporal_attn = TemporalCrossAttention()

        self.decoder1 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)

        self.final_up = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.head = nn.Sequential(

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):

        t1 = x[:, :10]
        t2 = x[:, 10:]

        f1 = self.encoder(t1)
        f2 = self.encoder(t2)

        s1_1, s1_2, s1_3, s1_4 = f1
        s2_1, s2_2, s2_3, s2_4 = f2

        gcn_feat = self.gcn(s1_3)

        fam_feat = self.fam(s1_4, gcn_feat)

        change_feat = self.temporal_attn(fam_feat, s2_4)

        d1 = self.decoder1(change_feat, s1_3)
        d2 = self.decoder2(d1, s1_2)
        d3 = self.decoder3(d2, s1_1)

        out = self.final_up(d3)

        out = self.head(out)

        out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)

        return out
