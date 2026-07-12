"""BiliAxis — physics-grounded bilirubin projection (novel module #2).

Bilirubin absorbs light around ~460 nm (blue), so jaundiced skin reflects less blue and appears more
yellow. On the SCIN white-balanced image we compute a per-pixel *bilirubin index* as a projection of
RGB onto a yellow-opponent direction, initialized to the physically-motivated (+R, +G, -2B) axis and
then refined during training. The index map is pooled over the skin-attention weights (so bilirubin is
measured where skin is) into a compact, interpretable feature. The index map itself is an explanation.

Returns (feat [B,2] = attn-weighted mean & std of the index, index_map [B,H,W]).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BiliAxis(nn.Module):
    OUT_DIM = 2

    def __init__(self):
        super().__init__()
        # yellowness ~ high R,G and low B; normalized so it starts near a b*-like opponent axis
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5, -1.0]).view(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        # physical prior direction (frozen) the learnable axis is anchored to, so gradient descent
        # can refine the axis but not silently rotate "bilirubin" into whatever separates the classes
        # (e.g. residual lighting) — which would make the physics/interpretability claim decorative.
        self.register_buffer("weight_init", self.weight.detach().flatten().clone())

    def forward(self, x_wb: torch.Tensor, weight: torch.Tensor):
        # x_wb [B,3,H,W] white-balanced [0,1]; weight [B,H,W] skin-attention (>=0)
        idx = (x_wb * self.weight).sum(1) * self.scale + self.bias      # [B,H,W]
        w = weight + 1e-6
        denom = w.sum(dim=(1, 2))
        mean = (idx * w).sum(dim=(1, 2)) / denom
        var = (((idx - mean.view(-1, 1, 1)) ** 2) * w).sum(dim=(1, 2)) / denom
        std = var.clamp_min(1e-8).sqrt()
        feat = torch.stack([mean, std], dim=1)                          # [B,2]
        anchor = 1.0 - torch.cosine_similarity(self.weight.flatten(), self.weight_init, dim=0)
        return feat, idx, anchor
