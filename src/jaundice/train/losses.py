"""Auxiliary (non-classification) losses from our modules:
 + causal adversary regressions (illuminant, melanin factor) via GRL
 + melanin<->bilirubin disentanglement regularizers (axis orthogonality, batch decorrelation).

The classification term is owned by the DG objective (train/dg.py: erm/irm/groupdro), so these compose
with any DG method. Returns (extra_loss, logs); extra_loss is 0.0 when no aux modules are active.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def aux_losses(aux: dict, cfg: dict):
    total = 0.0
    logs = {}

    mcfg = cfg["model"]
    causal = mcfg.get("causal", {}) or {}
    dis = mcfg.get("disentangle", {}) or {}

    if causal.get("illuminant_adv") and aux.get("adv_illum_pred") is not None:
        li = F.mse_loss(aux["adv_illum_pred"], aux["illum_target"])
        total = total + li
        logs["adv_illum"] = float(li.item())

    if causal.get("melanin_adv") and aux.get("adv_mela_pred") is not None:
        # disentangled melanin factor is O(1); ITA proxy is O(tens of degrees) -> scale accordingly
        scale = 1.0 if dis.get("enable") else 1000.0
        lm = F.mse_loss(aux["adv_mela_pred"], aux["mela_target"]) / scale
        total = total + lm
        logs["adv_mela"] = float(lm.item())

    # anchor the learnable physics axis (BiliAxis or disentangle) near its physical prior direction
    anchor = aux.get("axis_anchor")
    if anchor is not None:
        w_anchor = float(mcfg.get("physics_anchor", 0.0))
        if w_anchor > 0:
            total = total + w_anchor * anchor
            logs["axis_anchor"] = float(anchor.item())

    if dis.get("enable"):
        if aux.get("ortho") is not None:
            total = total + dis.get("w_ortho", 0.1) * aux["ortho"]
            logs["ortho"] = float(aux["ortho"].item())
        pb, mel = aux.get("phys_bili"), aux.get("melanin")
        if pb is not None and mel is not None and pb.shape[0] > 1:
            x = pb.squeeze(1); m = mel.squeeze(1)
            xc = x - x.mean(); mc = m - m.mean()
            corr = (xc * mc).sum() / (xc.norm() * mc.norm() + 1e-6)
            total = total + dis.get("w_decorr", 0.1) * corr.pow(2)
            logs["decorr"] = float(corr.item())

    return total, logs
