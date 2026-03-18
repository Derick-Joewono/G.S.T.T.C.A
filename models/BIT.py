import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# =========================================================
# TOKENIZATION MODULE
# =========================================================

class Tokenizer(nn.Module):

    def __init__(self, in_channels, num_tokens):

        super().__init__()

        self.conv = nn.Conv2d(in_channels, num_tokens, 1)

    def forward(self, x):

        B, C, H, W = x.shape

        attn = self.conv(x)

        attn = attn.view(B, -1, H * W)

        attn = torch.softmax(attn, dim=-1)

        x = x.view(B, C, H * W)

        tokens = torch.bmm(attn, x.transpose(1, 2))

        return tokens


# =========================================================
# TRANSFORMER ENCODER
# =========================================================

class TransformerEncoder(nn.Module):

    def __init__(self, dim, heads=8, layers=4):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

    def forward(self, x):

        return self.transformer(x)


# =========================================================
# DECODER
# =========================================================

class Decoder(nn.Module):

    def __init__(self, in_channels, num_classes):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)

        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.conv1(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.conv2(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.conv3(x))

        out = self.head(x)

        return out


# =========================================================
# BIT MODEL
# =========================================================

class BIT(nn.Module):

    def __init__(
        self,
        in_channels=10,
        num_classes=3,
        num_tokens=8,
        token_dim=512
    ):

        super().__init__()

        # Shared encoder
        self.encoder = timm.create_model(
            "resnet50",
            pretrained=True,
            features_only=True,
            in_chans=in_channels
        )

        encoder_channels = self.encoder.feature_info.channels()

        self.feature_dim = encoder_channels[-1]

        # Tokenization
        self.tokenizer = Tokenizer(
            self.feature_dim,
            num_tokens
        )

        # Token projection
        self.token_proj = nn.Linear(
            self.feature_dim,
            token_dim
        )

        # Transformer
        self.transformer = TransformerEncoder(
            token_dim
        )

        # Decoder
        self.decoder = Decoder(
            self.feature_dim,
            num_classes
        )

    def forward(self, t1, t2):

        f1 = self.encoder(t1)[-1]
        f2 = self.encoder(t2)[-1]

        B, C, H, W = f1.shape

        # Tokenization
        tokens1 = self.tokenizer(f1)
        tokens2 = self.tokenizer(f2)

        tokens = torch.cat([tokens1, tokens2], dim=1)

        tokens = self.token_proj(tokens)

        tokens = self.transformer(tokens)

        # Change representation
        diff = torch.abs(f1 - f2)

        out = self.decoder(diff)

        out = F.interpolate(
            out,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        return out
