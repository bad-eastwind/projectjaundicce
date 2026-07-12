"""JaundiceNet: SCIN -> backbone -> gated-attention MIL pool -> [+ BiliAxis feature] -> classifier,
with optional causal adversaries (illuminant + melanin/ITA) fed via gradient reversal from the pooled
representation. Adversaries only run when return_aux=True (training); eval is a clean forward.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .scin import SCIN
from .bili_axis import BiliAxis
from .disentangle import MelaninBilirubinDisentangle
from .bili_grad import CephalocaudalBiliField
from .mixstyle import MixStyleIlluminant
from .causal import AdversaryHead, grad_reverse, compute_ita

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GatedAttentionPool(nn.Module):
    """Ilse et al. gated-attention MIL pooling over spatial tokens (weakly-supervised localization)."""
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, tokens: torch.Tensor):
        a = torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens))
        attn = torch.softmax(self.w(a).squeeze(-1), dim=1)     # [B,N]
        pooled = torch.einsum("bn,bnd->bd", attn, tokens)
        return pooled, attn


class JaundiceNet(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", scin: bool = True,
                 num_classes: int = 2, pretrained: bool = False,
                 bili_axis: bool = True, causal: dict | None = None,
                 lora: dict | None = None, disentangle: dict | None = None,
                 mixstyle: dict | None = None, bili_grad: dict | None = None):
        super().__init__()
        self.mixstyle = (MixStyleIlluminant(p=mixstyle.get("p", 0.5), alpha=mixstyle.get("alpha", 0.1))
                         if (mixstyle and mixstyle.get("enable")) else None)
        self.scin = SCIN() if scin else None
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.backbone = Backbone(backbone, pretrained=pretrained, lora=lora)
        dim = self.backbone.num_features
        self.pool = GatedAttentionPool(dim)

        # physics color feature. Precedence: BiliGrad (differential, novel) > disentangle > BiliAxis.
        self.grad = (CephalocaudalBiliField(regress_melanin=bili_grad.get("regress_melanin", True))
                     if (bili_grad and bili_grad.get("enable")) else None)
        self.dis = (MelaninBilirubinDisentangle()
                    if (self.grad is None and disentangle and disentangle.get("enable")) else None)
        self.bili = BiliAxis() if (bili_axis and self.grad is None and self.dis is None) else None
        phys_dim = (CephalocaudalBiliField.OUT_DIM if self.grad is not None
                    else MelaninBilirubinDisentangle.OUT_DIM if self.dis is not None
                    else BiliAxis.OUT_DIM if self.bili is not None else 0)
        self.head = nn.Linear(dim + phys_dim, num_classes)

        causal = causal or {}
        self.lambda_adv = float(causal.get("lambda_adv", 0.0))
        self.use_illum_adv = bool(causal.get("illuminant_adv", False))
        self.use_melanin_adv = bool(causal.get("melanin_adv", False))
        self.adv_illum = AdversaryHead(dim, 3) if self.use_illum_adv else None
        self.adv_mela = AdversaryHead(dim, 1) if self.use_melanin_adv else None

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        H, W = x.shape[-2:]
        if self.mixstyle is not None:
            x = self.mixstyle(x)                       # illuminant/style randomization (train only)
        illum = None
        if self.scin is not None:
            x_wb, illum = self.scin(x)
        else:
            x_wb = x
        tokens, (h, w) = self.backbone((x_wb - self.mean) / self.std)   # [B,N,D]
        B = tokens.shape[0]
        pooled, attn = self.pool(tokens)
        attn_map = attn.view(B, h, w)

        attn_up = F.interpolate(attn_map.unsqueeze(1), size=(H, W),
                                mode="bilinear", align_corners=False).squeeze(1)
        bili_map = m_scalar = ortho = phys_bili = axis_anchor = s_map = None
        if self.grad is not None:
            b_feat, bili_map, s_map, axis_anchor = self.grad(x_wb, attn_up)
            emb = torch.cat([pooled, b_feat], dim=1)
            phys_bili = b_feat[:, :1]                  # beta = cephalocaudal bilirubin slope
        elif self.dis is not None:
            b_feat, m_scalar, bili_map, ortho, axis_anchor = self.dis(x_wb, attn_up)
            emb = torch.cat([pooled, b_feat], dim=1)
            phys_bili = b_feat[:, :1]
        elif self.bili is not None:
            b_feat, bili_map, axis_anchor = self.bili(x_wb, attn_up)
            emb = torch.cat([pooled, b_feat], dim=1)
            phys_bili = b_feat[:, :1]
        else:
            emb = pooled
        logits = self.head(emb)                       # emb = head-input embedding (for retrieval)

        if not return_aux:
            return logits

        aux = {"illum": illum, "attn": attn_map, "bili_map": bili_map, "x_wb": x_wb,
               "ortho": ortho, "axis_anchor": axis_anchor, "phys_bili": phys_bili,
               "melanin": m_scalar, "s_map": s_map, "embedding": emb}
        if self.adv_illum is not None and illum is not None:
            aux["adv_illum_pred"] = self.adv_illum(grad_reverse(pooled, self.lambda_adv))
            # target = log-chromaticity of the estimated illuminant (illum is unit-mean, so log has
            # mean~0 and captures the colour cast in a better-conditioned space than raw ratios).
            aux["illum_target"] = torch.log(illum.clamp_min(1e-4)).detach()
        if self.adv_mela is not None:
            aux["adv_mela_pred"] = self.adv_mela(grad_reverse(pooled, self.lambda_adv))
            # nuisance target = disentangled melanin factor if available, else ITA proxy
            aux["mela_target"] = (m_scalar.detach() if m_scalar is not None
                                  else compute_ita(x_wb, attn_up).detach())
        return logits, aux
