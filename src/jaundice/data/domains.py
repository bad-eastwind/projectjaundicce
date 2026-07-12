"""Discover pseudo-domains inside dataset3 for leave-one-domain-out (LOCO) generalization.

Until the external hospital set arrives, we simulate cross-domain shift by clustering the core set on
classical *nuisance* descriptors (scene illuminant chroma, brightness, warmth, skin-tone ITA) with
KMeans. The resulting domain id is written back into manifest.csv. Leave-one-domain-out then trains on
k-1 clusters and tests on the held-out one — a legitimate domain-generalization protocol. No model
training here (pure classical CV).

Run:  PYTHONPATH=src python -m jaundice.data.domains --config configs/base.yaml --k 3
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jaundice.utils import load_config


def _rgb_to_lab(im):
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
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)   # L, a, b


def _skin_mask(im):
    r, g, b = im[..., 0] * 255, im[..., 1] * 255, im[..., 2] * 255
    mx, mn = im.max(-1) * 255, im.min(-1) * 255
    rule = (r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return rule | ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 128) & (y > 60))


def descriptor(path: str, size: int = 128) -> list[float]:
    im = np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=float) / 255.0
    mx, mn = im.max(-1), im.min(-1)
    sat = (mx - mn) / (mx + 1e-4)
    neutral = (mx > 0.6) & (sat < 0.2)
    if neutral.sum() < 20:
        neutral = np.ones_like(mx, bool)
    illum = im[neutral].mean(0) + 1e-4
    illum = illum / illum.sum()                       # illuminant chroma
    bright = float(mx.mean())
    warmth = float((im[..., 0] - im[..., 2]).mean())
    skin = _skin_mask(im)
    if skin.sum() < 20:
        skin = np.ones_like(mx, bool)
    L, _, b = _rgb_to_lab(im)
    Lm, bm = float(L[skin].mean()), float(b[skin].mean())
    ita = float(np.degrees(np.arctan2(Lm - 50.0, bm + 1e-6)))   # skin-tone proxy
    return [float(illum[0]), float(illum[2]), bright, warmth, ita]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    k = args.k or cfg["data"]["domain"]["k"]
    manifest = Path(cfg["data"]["manifest_out"])

    with open(manifest) as f:
        reader = csv.DictReader(f); rows = list(reader); fields = list(reader.fieldnames)
    if "domain" not in fields:
        fields.append("domain")

    core = [r for r in rows if r["dataset_id"] == "neo_skin_core"]
    print(f"computing descriptors for {len(core)} core images ...")
    feats = np.array([descriptor(r["path"]) for r in core])

    # FIT the scaler + KMeans on the TRAIN split only, then PREDICT for every core image. Fitting on
    # all rows lets cluster boundaries (the pseudo-domain definition) see val/test statistics -> a
    # subtle leak. `split` is the deterministic stratified split written by manifest.py.
    is_train = np.array([r.get("split") == "train" for r in core])
    if is_train.sum() < k:
        print(f"[warn] <{k} train rows; fitting domains on all core rows")
        is_train = np.ones(len(core), bool)
    scaler = StandardScaler().fit(feats[is_train])
    Z = scaler.transform(feats)
    km = KMeans(n_clusters=k, random_state=cfg["seed"], n_init=10).fit(Z[is_train])
    labels = km.predict(Z)

    for r in rows:
        r.setdefault("domain", "")
    for r, lab in zip(core, labels):
        r["domain"] = int(lab)

    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    print(f"wrote domain column (k={k}, fit on {int(is_train.sum())} train rows) to {manifest}\n")
    print(f"{'domain':<8}{'n':>5}{'jaundice':>10}{'normal':>8}{'pos_rate':>10}")
    pos_rates = []
    for c in range(k):
        idx = labels == c
        n = int(idx.sum())
        j = sum(int(core[i]["label"]) for i in np.where(idx)[0])
        pr = j / max(n, 1)
        pos_rates.append(pr)
        print(f"{c:<8}{n:>5}{j:>10}{n-j:>8}{pr:>10.2f}")
    # label-shift diagnostic: if positive-rate varies a lot across domains, leave-one-domain-out
    # partly measures LABEL shift, not domain shift. Reviewers will probe this — report it honestly.
    if pos_rates:
        spread = max(pos_rates) - min(pos_rates)
        flag = "  <-- HIGH: LOCO here conflates label shift with domain shift" if spread > 0.2 else ""
        print(f"\n[label-shift] positive-rate spread across domains = {spread:.2f}{flag}")


if __name__ == "__main__":
    main()
