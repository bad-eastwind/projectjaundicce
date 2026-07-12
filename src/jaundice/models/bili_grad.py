"""BiliGrad — Cephalocaudal Differential Bilirubin Field (novel core contribution).

Motivation / identifiability
----------------------------
In a single passive RGB photo, bilirubin, melanin and the illuminant are *collinear* in appearance:
all three modulate the yellow / blue-reflectance axis. So the diagnostic factor (bilirubin) is NOT
identifiable from the *marginal* skin colour — which is exactly why the color-stats baseline hits
0.98 AUC via a global-lighting shortcut, and why adversarial melanin-invariance erases the signal.

Bilirubin *is* identifiable from its distinct SPATIAL signature. Clinically (Kramer's rule) bilirubin
deposits cephalocaudally: face first, then trunk, then extremities as serum level rises. Melanin
(constitutive skin tone) and the scene illuminant do NOT follow this head->toe progression — to first
order they enter the yellow-axis (log-reflectance) field as a spatially-smooth OFFSET / field, not a
structured gradient along the body axis.

Method
------
Let b(p) be the yellow-axis (b*) value at skin pixel p, s(p) an anatomical coordinate along the body's
long axis (0..1), and m(p) an estimated melanin field. We fit, per image, a weighted least squares

    b(p) ~= c0 + beta * s(p) + gamma * m(p)          (weights = skin attention)

and read out the **cephalocaudal progression** of yellowness. Because melanin (gamma*m) and the
illuminant/constant (c0) are regressed OUT, the s-gradient is invariant to skin tone and to any
spatially-constant lighting/camera cast — a confounder-cancelling, calibration-free,
fairness-preserving bilirubin estimate. The fitted field and residual map are the interpretable
explanation (they reproduce the Kramer-zone reasoning clinicians already trust).

Anatomical coordinate (pose-free, orientation-invariant): each skin pixel is projected onto the
PRINCIPAL AXIS of the skin-attention mass (2x2 weighted covariance, closed-form eigenvector), so s runs
along the infant's long axis regardless of image orientation. The eigenvector SIGN is arbitrary per
image, so the readout is made **polarity-invariant**: the classifier features are the gradient
MAGNITUDE |beta|, the partial R^2 of s (how much along-axis yellowness variance s explains beyond
melanin+constant = gradient COHERENCE), the residual std, and the endpoint yellowness range. A linear
head cannot undo per-image sign flips, so signed features would be inconsistent across the dataset.

Optional `head_anchor` [B,2] (row,col in [0,1]) from a validated pose/keypoint or annotation source
resolves polarity (orients s so head=0) and upgrades the endpoint term to a SIGNED head-vs-distal
contrast (true cephalocaudal = yellower at head). Off by default: no external COCO-person pose net is
wired in — it is unreliable on swaddled neonates / body-part crops and would only add coordinate noise.

Generality: "estimate a confounded factor from its spatial signature via a nuisance-cancelling
relational readout, not from absolute appearance" transfers to any task with a spatial deposition prior
(cyanosis/perfusion gradients, pallor, rashes, wound healing).

Returns:
  feat   [B,4]   [|beta|, partial_R2(s), residual std, endpoint yellowness range/contrast]
  b_map  [B,H,W] yellow-axis field (interpretable explanation)
  s_map  [B,H,W] anatomical coordinate (for overlaying the fitted progression)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .causal import rgb_to_lab


class CephalocaudalBiliField(nn.Module):
    OUT_DIM = 4

    def __init__(self, ridge: float = 1e-3, regress_melanin: bool = True):
        super().__init__()
        self.ridge = ridge
        self.regress_melanin = regress_melanin
        # physics-initialized melanin direction over (L*,a*,b*); small learnable refinement only
        self.mela_dir = nn.Parameter(torch.tensor([-1.0, 0.3, 0.0]))
        self.register_buffer("mela_init", self.mela_dir.detach().clone())

    def _principal_axis(self, w: torch.Tensor):
        """Major-axis unit direction (sin, cos) + centroid of the skin-attention mass, per image."""
        B, H, W = w.shape
        ys = torch.linspace(0, 1, H, device=w.device).view(1, H, 1).expand(B, H, W)
        xs = torch.linspace(0, 1, W, device=w.device).view(1, 1, W).expand(B, H, W)
        wsum = w.sum(dim=(1, 2)) + 1e-6
        my = (w * ys).sum(dim=(1, 2)) / wsum
        mx = (w * xs).sum(dim=(1, 2)) / wsum
        dy, dx = ys - my.view(-1, 1, 1), xs - mx.view(-1, 1, 1)
        cyy = (w * dy * dy).sum(dim=(1, 2)) / wsum
        cxx = (w * dx * dx).sum(dim=(1, 2)) / wsum
        cxy = (w * dy * dx).sum(dim=(1, 2)) / wsum
        theta = 0.5 * torch.atan2(2 * cxy, (cyy - cxx) + 1e-6)      # major-axis angle
        return torch.sin(theta), torch.cos(theta), my, mx, dy, dx

    @staticmethod
    def _wls_ss(phi, y, w, ridge):
        """Weighted least squares. Returns weighted residual sum-of-squares and coefficients."""
        wphi = phi * w.unsqueeze(2)
        A = torch.einsum("bnk,bnj->bkj", wphi, phi) + ridge * torch.eye(
            phi.shape[2], device=phi.device).unsqueeze(0)
        rhs = torch.einsum("bnk,bn->bk", wphi, y)
        coeff = torch.linalg.solve(A, rhs.unsqueeze(2)).squeeze(2)
        resid = y - torch.einsum("bnk,bk->bn", phi, coeff)
        ss = ((resid ** 2) * w).sum(1)
        return ss, coeff

    def forward(self, x_wb: torch.Tensor, weight: torch.Tensor, head_anchor: torch.Tensor | None = None):
        B, _, H, W = x_wb.shape
        L, a, b = rgb_to_lab(x_wb)
        lab = torch.stack([L, a, b], dim=1) / 100.0
        b_map = lab[:, 2]                                            # b* (yellow) field  [B,H,W]
        m_map = (lab * self.mela_dir.view(1, 3, 1, 1)).sum(1)        # melanin field      [B,H,W]

        sin_t, cos_t, my, mx, dy, dx = self._principal_axis(weight)
        s = dy * sin_t.view(-1, 1, 1) + dx * cos_t.view(-1, 1, 1)    # along-axis coordinate
        wflat = (weight + 1e-6).view(B, -1)
        wm = weight.view(B, -1) > 1e-4
        s_flat = s.view(B, -1)
        smin = torch.where(wm, s_flat, torch.full_like(s_flat, float("inf"))).amin(1).view(-1, 1, 1)
        smax = torch.where(wm, s_flat, torch.full_like(s_flat, float("-inf"))).amax(1).view(-1, 1, 1)
        s_map = ((s - smin) / (smax - smin + 1e-6)).clamp(0, 1)      # [B,H,W] in [0,1]

        # optional pose: orient s so the head end is s=0 (resolves the arbitrary eigenvector sign)
        oriented = False
        if head_anchor is not None:
            hy = (head_anchor[:, 0].view(-1, 1, 1) - my.view(-1, 1, 1))
            hx = (head_anchor[:, 1].view(-1, 1, 1) - mx.view(-1, 1, 1))
            head_s = (hy * sin_t.view(-1, 1, 1) + hx * cos_t.view(-1, 1, 1)).view(B, 1, 1)
            head_s = ((head_s - smin) / (smax - smin + 1e-6))
            flip = (head_s.view(B) > 0.5)                            # head sits at the high-s end
            s_map = torch.where(flip.view(-1, 1, 1), 1.0 - s_map, s_map)
            oriented = True

        y = b_map.view(B, -1)
        ones = torch.ones_like(s_flat)
        s_col = s_map.view(B, -1)
        reduced_cols = [ones, m_map.view(B, -1)] if self.regress_melanin else [ones]
        full_cols = reduced_cols[:1] + [s_col] + reduced_cols[1:]    # [1, s, (m)]
        phi_full = torch.stack(full_cols, dim=2)
        phi_red = torch.stack(reduced_cols, dim=2)                   # s dropped

        ss_full, coeff = self._wls_ss(phi_full, y, wflat, self.ridge)
        ss_red, _ = self._wls_ss(phi_red, y, wflat, self.ridge)
        ybar = (y * wflat).sum(1) / wflat.sum(1)
        ss_tot = (((y - ybar.view(-1, 1)) ** 2) * wflat).sum(1)

        beta = coeff[:, 1]
        grad_mag = beta.abs()                                       # polarity-invariant gradient size
        partial_r2 = ((ss_red - ss_full) / (ss_tot + 1e-6)).clamp(0, 1)   # coherence of the s-gradient
        rstd = (ss_full / (wflat.sum(1) + 1e-6)).clamp_min(1e-8).sqrt()

        # endpoint yellowness term. With a head anchor -> SIGNED head-vs-distal contrast (cephalocaudal
        # direction is discriminative); without -> polarity-invariant |range|.
        hi = (s_map < 0.34).float() * weight
        lo = (s_map > 0.66).float() * weight
        b_hi = (b_map * hi).sum(dim=(1, 2)) / (hi.sum(dim=(1, 2)) + 1e-6)
        b_lo = (b_map * lo).sum(dim=(1, 2)) / (lo.sum(dim=(1, 2)) + 1e-6)
        endpoint = (b_hi - b_lo) if oriented else (b_hi - b_lo).abs()

        feat = torch.stack([grad_mag, partial_r2, rstd, endpoint], dim=1)   # [B,4]
        anchor = 1.0 - torch.cosine_similarity(self.mela_dir, self.mela_init, dim=0)
        return feat, b_map, s_map, anchor
