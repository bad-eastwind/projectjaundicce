"""Model-independent skin-tone reference used for fairness stratification.

The reference is deliberately outside the learned model: fixed resize, fixed
gray-world white balance, fixed classical skin mask, then ITA over CIELAB skin
pixels. This makes the protected-attribute proxy stable across model variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# ITA (Individual Typology Angle) skin-tone strata, dermatology convention
# ordered light -> dark.
ITA_EDGES = [55.0, 41.0, 28.0, 10.0, -30.0]
ITA_NAMES = ["very_light", "light", "intermediate", "tan", "brown", "dark"]


@dataclass(frozen=True)
class ReferenceSkinTone:
    ita: float
    stratum: str
    skin_pixels: int
    skin_fraction: float


def ita_stratum(ita: float) -> str:
    for name, edge in zip(ITA_NAMES, ITA_EDGES):
        if ita > edge:
            return name
    return ITA_NAMES[-1]


def rgb_to_lab(im: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB [0,1] -> CIELAB L*, a*, b* arrays."""
    def inv(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r, g, b = inv(im[..., 0]), inv(im[..., 1]), inv(im[..., 2])
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    x, y, z = X / 0.95047, Y, Z / 1.08883

    def f(t):
        d = 6 / 29
        return np.where(t > d ** 3, np.cbrt(np.clip(t, 1e-6, None)), t / (3 * d * d) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def gray_world_wb(im: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Fixed global gray-world diagonal white balance for RGB [0,1]."""
    illum = im.reshape(-1, 3).mean(0) + eps
    illum = illum / max(float(illum.mean()), eps)
    return np.clip(im / illum.reshape(1, 1, 3), 0.0, 1.0)


def fixed_skin_mask(im: np.ndarray) -> np.ndarray:
    """Dependency-free RGB+YCbCr skin rule over RGB [0,1]."""
    r, g, b = im[..., 0] * 255, im[..., 1] * 255, im[..., 2] * 255
    mx, mn = im.max(-1) * 255, im.min(-1) * 255
    rgb_rule = (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & ((mx - mn) > 15)
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    ycbcr_rule = (cr > 133) & (cr < 180) & (cb > 77) & (cb < 128) & (y > 60)
    return rgb_rule | ycbcr_rule


def reference_skin_tone(path: str | Path, size: int = 128, min_skin_pixels: int = 20) -> ReferenceSkinTone:
    """Compute frozen ITA reference for one image.

    If the fixed skin rule finds too few pixels, fall back to the full image and
    record that larger support through `skin_fraction=1.0`.
    """
    im = np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=float) / 255.0
    wb = gray_world_wb(im)
    skin = fixed_skin_mask(wb)
    if int(skin.sum()) < min_skin_pixels:
        skin = np.ones(wb.shape[:2], bool)

    L, _, b = rgb_to_lab(wb)
    Lm, bm = float(L[skin].mean()), float(b[skin].mean())
    ita = float(np.degrees(np.arctan2(Lm - 50.0, bm + 1e-6)))
    return ReferenceSkinTone(
        ita=ita,
        stratum=ita_stratum(ita),
        skin_pixels=int(skin.sum()),
        skin_fraction=float(skin.mean()),
    )
