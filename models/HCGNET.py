import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 1. CORE MODULES (Shift Operator & Graph Conv)
# ==========================================================

class ShiftOperator(nn.Module):
    """
    Inovasi utama HCGNet: Menggeser fitur 1 piksel ke atas, bawah, kiri, kanan
    lalu menjumlahkannya dengan piksel tengah untuk menangkap fitur lokal (irregular edges).
    """
    def forward(self, x):
        # x shape: [B, C, H, W]
        x_pad = F.pad(x, (1, 1, 1, 1)) # Pad 1 piksel di semua sisi
        x_up = x_pad[:, :, :-2, 1:-1]
        x_down = x_pad[:, :, 2:, 1:-1]
        x_left = x_pad[:, :, 1:-1, :-2]
        x_right = x_pad[:, :, 1:-1, 2:]
        return x + x_up + x_down + x_left + x_right

class MaxRelativeGraphConv(nn.Module):
    """
    Max-Relative Graph Convolution untuk mengekstrak fitur Long-term (Global).
    Membangun graf KNN secara dinamis berdasarkan Euclidean distance fitur.
    """
    def __init__(self, in_c, out_c, k=9):
        super().__init__()
        self.k = k
        self.w1 = nn.Conv2d(in_c, out_c, 1)
        self.w2 = nn.Conv2d(out_c * 2, out_c, 1)
        self.w3 = nn.Conv2d(out_c, out_c, 1)

    def forward(self, x_s):
        B, C, H, W = x_s.shape
        x1 = self.w1(x_s) # [B, C', H, W]
        N = H * W
        
        # Flatten untuk kalkulasi jarak KNN: [B, N, C']
        x1_flat = x1.view(B, -1, N).transpose(1, 2) 

        # Euclidean distance antar node (piksel)
        dist = torch.cdist(x1_flat, x1_flat) # [B, N, N]
        _, knn_idx = torch.topk(dist, k=self.k, dim=-1, largest=False) # [B, N, k]

        # Ambil fitur dari k tetangga terdekat
        B_idx = torch.arange(B, device=x_s.device).view(-1, 1, 1).expand(-1, N, self.k)
        knn_feats = x1_flat[B_idx, knn_idx, :] # [B, N, k, C']

        # Kalkulasi selisih max antar node dan tetangganya
        x1_exp = x1_flat.unsqueeze(2) # [B, N, 1, C']
        x2, _ = torch.max(knn_feats - x1_exp, dim=2) # [B, N, C']

        # Concatenate dan proses akhir
        x3 = torch.cat([x1_flat, x2], dim=-1) # [B, N, 2C']
        x3 = x3.transpose(1, 2).view(B, -1, H, W) # Kembalikan ke spasial [B, 2C', H, W]

        x4 = self.w2(x3)
        x5 = self.w3(x4)
        return x5

class ShiftViGBlock(nn.Module):
    """
    Blok utama di Deep Layers (Stage 3, 4, 5).
    Gabungan Shift Operator (Lokal) + Graph Conv (Global) + MLP.
    """
    def __init__(self, channels, k=9):
        super().__init__()
        self.shift = ShiftOperator()
        self.gcn = MaxRelativeGraphConv(channels, channels, k)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1)
        )

    def forward(self, x):
        x_s = self.shift(x)
        x_G = self.gcn(x_s) + x # Residual connection 1
        x_V = self.mlp(x_G) + x_G # Residual connection 2
        return x_V

# ==========================================================
# 2. BASIC CNN BLOCKS & ATTENTION
# ==========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CAM(nn.Module):
    """Channel Attention Module (Squeeze-and-Excitation)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        red_c = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, red_c, 1),
            nn.ReLU(),
            nn.Conv2d(red_c, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)

# ==========================================================
# 3. UPSAMPLING & MERGING (UM) BLOCKS (Untuk Decoder)
# ==========================================================

class UMBlock_Diff(nn.Module):
    """UM Block untuk Differential Branch (Peta Kasar)"""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.cam = CAM(in_c + skip_c)
        self.proj = nn.Conv2d(in_c + skip_c, out_c, 1)

    def forward(self, x, e1, e2):
        x_up = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        diff = torch.abs(e1 - e2) # Mengambil selisih absolut antara T1 dan T2
        out = torch.cat([x_up, diff], dim=1)
        out = self.cam(out)
        return self.proj(out)

class UMBlock_Comp(nn.Module):
    """UM Block untuk Comprehensive Branch (Peta Halus/Final)"""
    def __init__(self, in_c, skip_c, diff_c, out_c):
        super().__init__()
        # Concat: upsample_X + E1 + E2 + diff_feature
        self.cam = CAM(in_c + skip_c * 2 + diff_c)
        self.proj = nn.Conv2d(in_c + skip_c * 2 + diff_c, out_c, 1)

    def forward(self, x, e1, e2, d_diff):
        x_up = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x_up, e1, e2, d_diff], dim=1)
        out = self.cam(out)
        return self.proj(out)

# ==========================================================
# 4. MAIN HCGNET ARCHITECTURE
# ==========================================================

class HCGNet(nn.Module):
    def __init__(self, in_channels=10, num_classes=3):
        super().__init__()
        # Dimensi Channel (C1 sampai C6 mengikuti kaidah ukuran piramida)
        c1, c2, c3, c4, c5 = 32, 64, 128, 256, 512
        
        # --- ENCODER (Siamese, Share Weights) ---
        self.enc1 = ConvBlock(in_channels, c1)
        self.ds1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1) # 256 -> 128
        
        self.enc2 = ConvBlock(c2, c2)
        self.ds2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1) # 128 -> 64
        
        # Stage 3, 4, 5 menggunakan Shift ViG Block
        self.enc3 = nn.Sequential(ShiftViGBlock(c3), ShiftViGBlock(c3))
        self.ds3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1) # 64 -> 32
        
        self.enc4 = nn.Sequential(ShiftViGBlock(c4), ShiftViGBlock(c4))
        self.ds4 = nn.Conv2d(c4, c5, 3, stride=2, padding=1) # 32 -> 16
        
        self.enc5 = nn.Sequential(*[ShiftViGBlock(c5) for _ in range(6)])
        self.ds5 = nn.Conv2d(c5, c5, 3, stride=2, padding=1) # 16 -> 8

        # --- FUSION MODULE ---
        self.fusion_proj = nn.Conv2d(c5 * 2, c5, 1)
        self.fusion_vig = nn.Sequential(ShiftViGBlock(c5), ShiftViGBlock(c5))

        # --- DECODER (Dual-Branch) ---
        # Branch 1: Differential (Diff)
        self.diff_um5 = UMBlock_Diff(c5, c5, c4)
        self.diff_vig5 = ShiftViGBlock(c4)
        
        self.diff_um4 = UMBlock_Diff(c4, c4, c3)
        self.diff_vig4 = ShiftViGBlock(c3)
        
        self.diff_um3 = UMBlock_Diff(c3, c3, c2)
        self.diff_vig3 = ShiftViGBlock(c2)
        
        self.diff_um2 = UMBlock_Diff(c2, c2, c1)
        self.diff_conv2 = ConvBlock(c1, c1)
        
        self.diff_um1 = UMBlock_Diff(c1, c1, 16)
        self.diff_conv1 = ConvBlock(16, 16)

        # Branch 2: Comprehensive (Comp)
        self.comp_um5 = UMBlock_Comp(c5, c5, c4, c4)
        self.comp_vig5 = ShiftViGBlock(c4)
        
        self.comp_um4 = UMBlock_Comp(c4, c4, c3, c3)
        self.comp_vig4 = ShiftViGBlock(c3)
        
        self.comp_um3 = UMBlock_Comp(c3, c3, c2, c2)
        self.comp_vig3 = ShiftViGBlock(c2)
        
        self.comp_um2 = UMBlock_Comp(c2, c2, c1, c1)
        self.comp_conv2 = ConvBlock(c1, c1)
        
        self.comp_um1 = UMBlock_Comp(c1, c1, 16, 16)
        self.comp_conv1 = ConvBlock(16, 16)

        # Final Projections
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.diff_head = nn.Conv2d(16, num_classes, 1)
        self.comp_head = nn.Conv2d(16, num_classes, 1)

    def forward_encoder(self, x):
        e1 = self.enc1(x)                # [B, 32, 256, 256]
        e2 = self.enc2(self.ds1(e1))     # [B, 64, 128, 128]
        e3 = self.enc3(self.ds2(e2))     # [B, 128, 64, 64]
        e4 = self.enc4(self.ds3(e3))     # [B, 256, 32, 32]
        e5 = self.enc5(self.ds4(e4))     # [B, 512, 16, 16]
        out = self.ds5(e5)               # [B, 512, 8, 8]
        return [e1, e2, e3, e4, e5], out

    # Ubah parameternya menjadi t1 dan t2 langsung
    def forward(self, t1, t2):
        
        # BARIS PEMISAHAN 'x' DIHAPUS SAJA
        
        # 1. Langsung masuk ke Encoder Branching (Siamese)
        skips1, out1 = self.forward_encoder(t1)
        skips2, out2 = self.forward_encoder(t2)

        # 2. Fusion Module
        fused = torch.cat([out1, out2], dim=1)
        fused = self.fusion_proj(fused)
        fused = self.fusion_vig(fused)

        # 3. Decoder - Differential Branch
        d_d5 = self.diff_vig5(self.diff_um5(fused, skips1[4], skips2[4]))
        d_d4 = self.diff_vig4(self.diff_um4(d_d5, skips1[3], skips2[3]))
        d_d3 = self.diff_vig3(self.diff_um3(d_d4, skips1[2], skips2[2]))
        d_d2 = self.diff_conv2(self.diff_um2(d_d3, skips1[1], skips2[1]))
        d_d1 = self.diff_conv1(self.diff_um1(d_d2, skips1[0], skips2[0]))

        # 4. Decoder - Comprehensive Branch
        d_c5 = self.comp_vig5(self.comp_um5(fused, skips1[4], skips2[4], d_d5))
        d_c4 = self.comp_vig4(self.comp_um4(d_c5, skips1[3], skips2[3], d_d4))
        d_c3 = self.comp_vig3(self.comp_um3(d_c4, skips1[2], skips2[2], d_d3))
        d_c2 = self.comp_conv2(self.comp_um2(d_c3, skips1[1], skips2[1], d_d2))
        d_c1 = self.comp_conv1(self.comp_um1(d_c2, skips1[0], skips2[0], d_d1))

        # 5. Final Upsampling ke resolusi asli
        out_comp = self.final_up(d_c1)
        final_pred = self.comp_head(out_comp)

        # Output dipastikan 256x256
        if final_pred.shape[-2:] != (256, 256):
            final_pred = F.interpolate(final_pred, size=(256, 256), mode="bilinear", align_corners=False)

        return final_pred