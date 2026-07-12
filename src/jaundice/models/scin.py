"""SCIN — Self-Calibrating Illuminant Normalization (novel module #1).

Calibration-card-free white balance for clinical color diagnosis. Prior smartphone-bilirubin work
(BiliCam, Picterus, neoSCB) requires a physical color-reference card in frame. SCIN instead estimates
the scene illuminant directly from in-frame near-neutral references (bright, low-saturation surfaces
such as NICU bedding), then applies a von-Kries diagonal transform to a canonical illuminant so that
downstream skin color reflects reflectance (bilirubin), not the light source / camera white balance.

Fully differentiable. The notion of "neutral pixel" is parameterized by learnable thresholds so the
model adapts what counts as a reference surface during training.

I/O: RGB in [0,1], shape [B,3,H,W]. Returns (normalized_image [B,3,H,W] in [0,1], illuminant [B,3]).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SCIN(nn.Module):
    def __init__(self, eps: float = 1e-4, neutral_frac_scale: float = 0.02):
        super().__init__()
        # learnable soft-selection of neutral (bright, low-saturation) reference pixels
        self.bright_thresh = nn.Parameter(torch.tensor(0.60))
        self.sat_thresh = nn.Parameter(torch.tensor(0.20))
        self.bright_sharp = nn.Parameter(torch.tensor(10.0))
        self.sat_sharp = nn.Parameter(torch.tensor(10.0))
        self.eps = eps
        # neutral COVERAGE FRACTION (of image area) needed before trusting the neutral estimate
        # over gray-world. Resolution-independent: an absolute mass threshold would saturate at high
        # resolution (any neutral surface -> mass >> const -> gray-world fallback becomes dead code)
        # and behave differently at smoke resolution.
        self.neutral_frac_scale = neutral_frac_scale

    def estimate_illuminant(self, x: torch.Tensor) -> torch.Tensor:
        npix = x.shape[-1] * x.shape[-2]              # H*W
        maxc = x.amax(1)                              # [B,H,W] brightness (max channel)
        minc = x.amin(1)
        sat = (maxc - minc) / (maxc + self.eps)       # HSV-like saturation
        w_bright = torch.sigmoid(self.bright_sharp * (maxc - self.bright_thresh))
        w_lowsat = torch.sigmoid(self.sat_sharp * (self.sat_thresh - sat))
        w = (w_bright * w_lowsat).unsqueeze(1)        # [B,1,H,W] soft neutral mask
        num = (x * w).sum(dim=(2, 3))                 # [B,3]
        mass = w.sum(dim=(2, 3))                       # [B,1]
        neutral_illum = num / (mass + self.eps)       # mean color of neutral refs
        gray_world = x.mean(dim=(2, 3))               # fallback if no neutral refs found
        # trust the neutral estimate proportional to the FRACTION of image area selected as neutral
        frac = mass / npix                             # [B,1] in [0,1], resolution-independent
        alpha = torch.clamp(frac / self.neutral_frac_scale, 0.0, 1.0)   # [B,1]
        illum = alpha * neutral_illum + (1.0 - alpha) * gray_world
        illum = illum / (illum.mean(dim=1, keepdim=True) + self.eps)    # unit-mean -> preserve brightness
        return illum.clamp_min(self.eps)

    def forward(self, x: torch.Tensor):
        illum = self.estimate_illuminant(x)           # [B,3]
        out = (x / illum.view(-1, 3, 1, 1)).clamp(0.0, 1.0)   # von-Kries diagonal WB
        return out, illum
