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
        # Linear layer memproses dimensi terakhir (Channel)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x shape: [B, N, C]
        # adj shape: [B, N, N]
        x = torch.matmul(adj, x) # Agregasi tetangga
        x = self.fc(x)           # Transformasi fitur channel
        return x

class GCNEncoder(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        # In_channels disesuaikan dengan output s1_3 (512 untuk swin_base)
        self.gcn1 = GCNLayer(in_channels, in_channels)
        self.gcn2 = GCNLayer(in_channels, in_channels)

        self.downsample = nn.Conv2d(
            in_channels,
            1024, # Output 1024 agar sinkron dengan s1_4 di FAM
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial HxW menjadi N nodes: [B, C, H, W] -> [B, N, C]
        nodes = x.view(B, C, -1).permute(0, 2, 1)
        
        N = nodes.shape[1]
        # Identity matrix sebagai adjacency dasar
        adj = torch.eye(N).to(x.device).unsqueeze(0).repeat(B, 1, 1)

        nodes = self.gcn1(nodes, adj)
        nodes = self.gcn2(nodes, adj)

        # Kembalikan ke shape [B, C, H, W]
        out = nodes.permute(0, 2, 1).view(B, C, H, W)
        out = self.downsample(out)
        return out

# ==========================================================
# FEATURE AGGREGATION MODULE (FAM)
# ==========================================================

class GatedFAM(nn.Module):
    def __init__(self, channels=1024):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, swin_feat, gcn_feat):
        # Pastikan ukuran spasial sama sebelum concat
        if swin_feat.shape[-2:] != gcn_feat.shape[-2:]:
            gcn_feat = F.interpolate(gcn_feat, size=swin_feat.shape[-2:], mode='bilinear', align_corners=False)
            
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
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        # Global Spatial Attention
        q = f1.flatten(2).transpose(1, 2) # [B, N, C]
        k = f2.flatten(2).transpose(1, 2)
        v = f2.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

        diff = torch.abs(f1 - f2)
        fusion = torch.cat([attn_out, diff], dim=1)
        out = self.fusion_conv(fusion)
        return out

# ==========================================================
# DECODER & MAIN MODEL
# ==========================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class GSWIN_TAC(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Encoder menggunakan Swin Base
        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            features_only=True,
            img_size=256,       
            in_chans=10,
        )

        # Inisialisasi modul dengan dimensi channel Swin Base
        # s1_1:128, s1_2:256, s1_3:512, s1_4:1024
        self.gcn = GCNEncoder(in_channels=512) 
        self.fam = GatedFAM(channels=1024)
        self.temporal_attn = TemporalCrossAttention(dim=1024)

        self.decoder1 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)

        self.final_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        t1 = x[:, :10]
        t2 = x[:, 10:]

        # Mendapatkan fitur dari Swin
        f1 = self.encoder(t1) 
        f2 = self.encoder(t2)

        # 🔍 PASTIKAN FORMAT: Swin terkadang mengembalikan (B, H, W, C)
        # Kita paksa semua stage menjadi (B, C, H, W) agar konsisten dengan GCN & Decoder
        def fix_shape(feats):
            fixed = []
            for feat in feats:
                # Jika formatnya (B, L, C) atau (B, H, W, C)
                if feat.dim() == 3: # (B, L, C)
                    B, L, C = feat.shape
                    H = W = int(L**0.5)
                    feat = feat.permute(0, 2, 1).view(B, C, H, W)
                elif feat.dim() == 4 and feat.shape[1] != 128 and feat.shape[1] != 256: 
                    # Ini deteksi jika C berada di dimensi terakhir (Channels-Last)
                    # Misal shape (B, 16, 16, 512)
                    feat = feat.permute(0, 3, 1, 2)
                fixed.append(feat)
            return fixed

        s1_1, s1_2, s1_3, s1_4 = fix_shape(f1)
        s2_1, s2_2, s2_3, s2_4 = fix_shape(f2)

        # 1. GCN sekarang menerima s1_3 yang SUDAH PASTI [B, 512, 16, 16]
        gcn_feat = self.gcn(s1_3) 
        
        # ... sisa kode FAM dan Decoder

        # 2. Gabungkan Swin s1_4 dengan GCN feat menggunakan Gated FAM
        fam_feat = self.fam(s1_4, gcn_feat)

        # 3. Temporal Attention antara T1 (fam) dan T2 (s2_4)
        change_feat = self.temporal_attn(fam_feat, s2_4)

        # 4. Decoding dengan skip connections
        d1 = self.decoder1(change_feat, s1_3)
        d2 = self.decoder2(d1, s1_2)
        d3 = self.decoder3(d2, s1_1)

        # 5. Final Upsampling ke resolusi asli
        out = self.final_up(d3)
        out = self.head(out)
        
        # Pastikan output pas 256x256
        if out.shape[-2:] != (256, 256):
            out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)

        return out