"""Causal deconfounding (novel module #3) via domain-adversarial invariance (DANN-style).

We want the representation to predict jaundice WITHOUT relying on nuisance factors:
  - illuminant residual (what SCIN could not remove),
  - melanin / skin-tone (fairness + a shortcut in this data).

A gradient-reversal layer feeds pooled features to adversaries that try to regress the nuisance; the
reversed gradient pushes the features to be invariant to it. Strength = lambda (in the GRL).

CAVEAT (documented, important research subtlety): bilirubin AND melanin both yellow the skin, so a
naive melanin-invariance can erase bilirubin signal. Melanin adversary is therefore gated by config
and uses ITA (lightness-vs-yellow angle) which we intend to further disentangle from bilirubin by its
different spatial signature in later work. Illuminant adversary is the clean, primary one.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class AdversaryHead(nn.Module):
    def __init__(self, dim: int, out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def rgb_to_lab(rgb: torch.Tensor):
    """sRGB [B,3,H,W] in [0,1] -> (L,a,b) each [B,H,W], D65."""
    def inv_gamma(c):
        return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = inv_gamma(rgb[:, 0]), inv_gamma(rgb[:, 1]), inv_gamma(rgb[:, 2])
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    x, y, z = X / 0.95047, Y / 1.0, Z / 1.08883

    def f(t):
        d = 6.0 / 29.0
        return torch.where(t > d ** 3, t.clamp_min(1e-6) ** (1.0 / 3.0), t / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return L, a, bb


def compute_ita(x_wb: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Individual Typology Angle over skin-weighted pixels -> [B,1]. Skin-tone proxy (nuisance target)."""
    L, _, b = rgb_to_lab(x_wb)
    w = weight + 1e-6
    denom = w.sum(dim=(1, 2))
    Lm = (L * w).sum(dim=(1, 2)) / denom
    bm = (b * w).sum(dim=(1, 2)) / denom
    ita = torch.atan2(Lm - 50.0, bm + 1e-6) * 180.0 / math.pi
    return ita.unsqueeze(1)
