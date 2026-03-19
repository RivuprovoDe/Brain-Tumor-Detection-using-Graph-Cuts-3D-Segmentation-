# =============================================================================
#  models/unet3d.py
#  Full 3D U-Net as described in the project report.
#  Reference: Çiçek et al. (2016) "3D U-Net: Learning Dense Volumetric
#             Segmentation from Sparse Annotation"
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive Conv3d → BatchNorm → ReLU layers."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout3d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock followed by MaxPool3d downsampling."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Trilinear upsample → concat skip → ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if shapes mismatch (can happen with odd patch sizes)
        if x.shape != skip.shape:
            pd = skip.size(2) - x.size(2)
            ph = skip.size(3) - x.size(3)
            pw = skip.size(4) - x.size(4)
            x = F.pad(x, [0, pw, 0, ph, 0, pd])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── attention gate ─────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate from Oktay et al. (2018).
    g  : gating signal  (from decoder)
    x  : skip features  (from encoder)
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.Wg = nn.Conv3d(g_ch,    inter_ch, kernel_size=1, bias=False)
        self.Wx = nn.Conv3d(x_ch,    inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        # Upsample g to match x spatial size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        alpha = self.psi(self.relu(g1 + x1))
        return x * alpha


# ── full 3D U-Net ─────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    3D U-Net with optional attention gates.

    Parameters
    ----------
    in_channels   : number of MRI modalities (4 for BraTS)
    out_channels  : 1 for binary segmentation
    base_filters  : feature maps at the first encoder level
    depth         : number of encoder / decoder levels
    use_attention : whether to attach attention gates on skip connections
    dropout       : dropout probability inside ConvBlocks
    """

    def __init__(self,
                 in_channels:  int = 4,
                 out_channels: int = 1,
                 base_filters: int = 32,
                 depth:        int = 4,
                 use_attention:bool = True,
                 dropout:      float = 0.2):
        super().__init__()
        self.depth = depth
        f = base_filters

        # ── encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = f * (2 ** i)
            self.encoders.append(EncoderBlock(in_ch, out_ch,
                                              dropout=dropout if i >= 2 else 0.0))
            in_ch = out_ch

        # ── bottleneck ────────────────────────────────────────────────────────
        bottleneck_ch = f * (2 ** depth)
        self.bottleneck = ConvBlock(in_ch, bottleneck_ch, dropout=dropout)

        # ── decoder ──────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        self.attn_gates = nn.ModuleList() if use_attention else None
        in_ch = bottleneck_ch
        for i in reversed(range(depth)):
            skip_ch = f * (2 ** i)
            out_ch  = skip_ch
            if use_attention:
                self.attn_gates.append(AttentionGate(in_ch, skip_ch, skip_ch // 2))
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch,
                                              dropout=dropout if i >= 2 else 0.0))
            in_ch = out_ch

        # ── output ────────────────────────────────────────────────────────────
        self.head = nn.Conv3d(in_ch, out_channels, kernel_size=1)
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x):
        skips = []

        # Encoder
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (reverse skip order)
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            if self.attn_gates is not None:
                skip = self.attn_gates[i](x, skip)
            x = dec(x, skip)

        # Output — raw logits; apply sigmoid externally when needed
        return self.head(x)

    # ─────────────────────────────────────────────────────────────────────────

    def predict_proba(self, x):
        """Forward pass → sigmoid probability map."""
        return torch.sigmoid(self.forward(x))

    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── loss functions ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice Loss.  Handles class imbalance naturally."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num   = 2.0 * (probs * targets).sum()
        den   = probs.sum() + targets.sum() + self.smooth
        return 1.0 - num / den


class DiceBCELoss(nn.Module):
    """Combination of Dice Loss + Binary Cross-Entropy (often trains better)."""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice      = DiceLoss()
        self.bce       = nn.BCEWithLogitsLoss()
        self.dw        = dice_weight
        self.bw        = bce_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + \
               self.bw * self.bce(logits, targets)


# ── metrics ───────────────────────────────────────────────────────────────────

def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute volumetric Dice coefficient.
    pred_mask : sigmoid probabilities  (B, 1, D, H, W)
    true_mask : binary labels          (B, 1, D, H, W)
    """
    pred = (pred_mask > threshold).float()
    inter = (pred * true_mask).sum()
    return float((2.0 * inter + smooth) / (pred.sum() + true_mask.sum() + smooth))


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance (in voxels)."""
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    pred_surf = pred ^ (pred & np.roll(pred, 1, axis=0))   # rough surface
    gt_surf   = gt   ^ (gt   & np.roll(gt,   1, axis=0))
    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return float('nan')
    dt_pred = distance_transform_edt(~pred.astype(bool))
    dt_gt   = distance_transform_edt(~gt.astype(bool))
    d1 = dt_gt[pred_surf.astype(bool)]
    d2 = dt_pred[gt_surf.astype(bool)]
    return float(np.percentile(np.concatenate([d1, d2]), 95))
