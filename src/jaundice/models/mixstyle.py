"""MixStyle illuminant augmentation (domain generalization).

Image-level MixStyle: mixes per-channel (mean, std) colour statistics across a batch to synthesize the
same scene under a different / blended light source. Applied BEFORE SCIN during training only, so it
stress-tests the whole illuminant-invariance pipeline (SCIN removal + causal illuminant adversary see
randomized illuminants). Ref: Zhou et al., MixStyle, ICLR 2021 (adapted to the colour/illuminant axis).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MixStyleIlluminant(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = x.std(dim=[2, 3], keepdim=True) + self.eps
        x_norm = (x - mu) / sig
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return (x_norm * sig_mix + mu_mix).clamp(0.0, 1.0)
