"""Melanin<->bilirubin disentanglement (core-novelty deepening).

Problem: melanin and bilirubin both reduce blue reflectance / yellow the skin, so a naive
skin-tone-invariance erases the bilirubin signal. Fix: separate them by their DIFFERENT optical
signatures in CIELAB:
  - bilirubin  ~ +b* (yellow; ~460 nm absorption), lightness roughly preserved
  - melanin    ~ lower L* (broadband darkening) + brownish a*
We project white-balanced skin color onto two learnable axes over (L*, a*, b*), initialized to these
physical directions and kept near-orthogonal. The bilirubin factor feeds the classifier; the melanin
factor becomes the nuisance target for the causal adversary (invariance to melanin ONLY). A
decorrelation loss (applied in losses.py) further pushes the two factors apart across the batch.

Returns:
  b_feat   [B,2]  attn-weighted mean & std of the bilirubin index -> classifier feature
  m_scalar [B,1]  attn-weighted melanin factor -> nuisance/adversary target
  b_map    [B,H,W] bilirubin index map -> interpretable explanation
  ortho    scalar |cos| between the two axes -> orthogonality regularizer
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .causal import rgb_to_lab


class MelaninBilirubinDisentangle(nn.Module):
    OUT_DIM = 2

    def __init__(self):
        super().__init__()
        # axes over (L*, a*, b*), physics-initialized
        self.bili_dir = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))    # yellow (b*)
        self.mela_dir = nn.Parameter(torch.tensor([-1.0, 0.3, 0.0]))   # darkening (-L*) + brown (a*)
        self.bili_b = nn.Parameter(torch.zeros(1))
        self.mela_b = nn.Parameter(torch.zeros(1))
        self.bili_s = nn.Parameter(torch.ones(1))
        self.mela_s = nn.Parameter(torch.ones(1))
        # frozen physical priors to anchor the learnable axes to (see BiliAxis rationale): keeps the
        # bilirubin axis near +b* and the melanin axis near (-L*,+a*) instead of drifting into the
        # class-separating (possibly lighting) direction.
        self.register_buffer("bili_init", self.bili_dir.detach().clone())
        self.register_buffer("mela_init", self.mela_dir.detach().clone())

    def forward(self, x_wb: torch.Tensor, weight: torch.Tensor):
        L, a, b = rgb_to_lab(x_wb)
        lab = torch.stack([L, a, b], dim=1) / 100.0                    # rough scale to O(1)
        b_map = (lab * self.bili_dir.view(1, 3, 1, 1)).sum(1) * self.bili_s + self.bili_b
        m_map = (lab * self.mela_dir.view(1, 3, 1, 1)).sum(1) * self.mela_s + self.mela_b

        w = weight + 1e-6
        denom = w.sum(dim=(1, 2))
        b_mean = (b_map * w).sum(dim=(1, 2)) / denom
        b_var = (((b_map - b_mean.view(-1, 1, 1)) ** 2) * w).sum(dim=(1, 2)) / denom
        b_feat = torch.stack([b_mean, b_var.clamp_min(1e-8).sqrt()], dim=1)     # [B,2]
        m_scalar = ((m_map * w).sum(dim=(1, 2)) / denom).unsqueeze(1)           # [B,1]

        ortho = torch.cosine_similarity(self.bili_dir, self.mela_dir, dim=0).abs()
        anchor = ((1.0 - torch.cosine_similarity(self.bili_dir, self.bili_init, dim=0))
                  + (1.0 - torch.cosine_similarity(self.mela_dir, self.mela_init, dim=0)))
        return b_feat, m_scalar, b_map, ortho, anchor
