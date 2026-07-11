"""Backbone abstraction so the physics/causal modules stay backbone-agnostic.

Returns spatial tokens [B, N, D] + grid (h, w) for either:
  - a CNN (timm features_only, e.g. convnext_tiny)  -> BASELINE
  - a ViT / DINOv2 (timm forward_features, patch tokens) -> foundation prior

PEFT: for the ViT path we can freeze the whole backbone and inject LoRA adapters into the attention /
MLP Linear layers, so only a tiny number of parameters train — the right regime for a 760-image set.
SCIN and BiliAxis operate on the image (before/around the backbone), so they work unchanged for both.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import timm


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen nn.Linear:  y = W0 x + (B A) x * (alpha/r)."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B stays zero -> adapter starts as no-op
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scaling


def _set_submodule(root: nn.Module, name: str, new: nn.Module):
    parts = name.split("."); obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new)


def inject_lora(model: nn.Module, targets=("qkv", "proj", "fc1", "fc2"),
                r: int = 8, alpha: int = 16, dropout: float = 0.0) -> int:
    hits = [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    for n, m in hits:
        _set_submodule(model, n, LoRALinear(m, r, alpha, dropout))
    return len(hits)


class Backbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = False, lora: dict | None = None):
        super().__init__()
        self.kind = "vit" if ("vit" in name or "dinov2" in name) else "cnn"
        lora = lora or {}
        if self.kind == "cnn":
            self.model = timm.create_model(name, pretrained=pretrained,
                                           features_only=True, out_indices=(-1,))
            self.num_features = self.model.feature_info[-1]["num_chs"]
            self.num_prefix = 0
        else:
            self.model = timm.create_model(name, pretrained=pretrained,
                                           num_classes=0, dynamic_img_size=True)
            self.num_features = self.model.num_features
            self.num_prefix = getattr(self.model, "num_prefix_tokens", 1)
            if lora.get("enable"):
                for p in self.model.parameters():
                    p.requires_grad_(False)          # freeze base -> PEFT
                n = inject_lora(self.model, tuple(lora.get("targets", ("qkv", "proj", "fc1", "fc2"))),
                                r=lora.get("r", 8), alpha=lora.get("alpha", 16),
                                dropout=lora.get("dropout", 0.0))
                print(f"[LoRA] injected into {n} Linear layers (r={lora.get('r', 8)}); base frozen")

    def forward(self, x: torch.Tensor):
        if self.kind == "cnn":
            f = self.model(x)[-1]                      # [B,D,h,w]
            B, D, h, w = f.shape
            return f.flatten(2).transpose(1, 2), (h, w)
        t = self.model.forward_features(x)             # [B, num_prefix+N, D]
        t = t[:, self.num_prefix:, :]                  # drop CLS/register tokens -> patch tokens
        B, N, D = t.shape
        h = w = int(round(math.sqrt(N)))
        return t, (h, w)
