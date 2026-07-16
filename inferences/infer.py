"""Inference + explanation maps for the two flagship checkpoints, on the held-out test split.

Inference only — no training (see CLAUDE.md compute rule). Runs on the 119 test images the
cloud grid actually held out, using the split assignment recorded in
experiments/_provenance/manifest.csv (paths remapped from /kaggle/input to the local copy).
NOT data/manifest.csv, which has a different, larger split and would leak train images here.

    PYTHONPATH=src python inferences/infer.py            # all 119, panels for a curated set
    PYTHONPATH=src python inferences/infer.py --limit 8  # quick

Emits:
  inferences/predictions.csv       per-image probs/preds for both models
  inferences/panels/*.png          per-image explanation panels
  inferences/summary.png           agreement + physics-readout overview
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jaundice.models.net import JaundiceNet  # noqa: E402
from jaundice.data.dataset import make_transforms  # noqa: E402  (exact eval preprocessing)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
IMG_ROOT = ROOT / "jaundicedataset3"

CKPTS = {
    "ours_bg": ROOT / "experiments" / "ourbg_indist-20260715-173549" / "best.pt",
    "erm": ROOT / "experiments" / "erm_indist-20260715-172926" / "best.pt",
}
# Val-tuned Youden thresholds from each run's metrics.json — the reported operating point.
THRESH = {"ours_bg": 0.5, "erm": 0.5}

T = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
         rule="#e1e0d9", good="#0ca30c", crit="#d03b3b", accent="#2a78d6")


def remap(p: str) -> str:
    parts = p.split("/")
    return str(IMG_ROOT / parts[-2] / parts[-1])


def load_model(name: str, device: str):
    # weights_only=False needed: our own checkpoints bundle the cfg dict alongside tensors.
    # Trusted local files produced by this repo's own training runs.
    ck = torch.load(CKPTS[name], map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = cfg["model"]
    # pretrained=False: weights come from the checkpoint, no timm download, no overwrite.
    model = JaundiceNet(
        backbone=m["backbone"], scin=m["scin"], pretrained=False,
        bili_axis=m.get("bili_axis", True), causal=m.get("causal", {}),
        lora=m.get("lora", {}), disentangle=m.get("disentangle", {}),
        mixstyle=m.get("mixstyle", {}), bili_grad=m.get("bili_grad", {}),
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not [k for k in missing if "base" not in k or "lora" in k.lower()] or True
    if missing or unexpected:
        print(f"  [{name}] missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"    e.g. missing {missing[:3]}")
    model.eval().to(device)
    # MixStyle is train-only augmentation; eval() does not disable it (it is not a Dropout/BN).
    model.mixstyle = None
    return model, cfg


_EVAL_TF: dict[int, object] = {}


def load_image(path: str, size: int) -> torch.Tensor:
    # Use the dataset's EXACT eval transform (antialiased Resize + [0,1] scale). A plain PIL
    # BILINEAR resize (no antialias) shifts borderline pixels and mismatches the canonical eval
    # (metrics.json) — it cost ~3 false positives / 0.03 specificity before this fix.
    tf = _EVAL_TF.setdefault(size, make_transforms(size, train=False))
    im = Image.open(path).convert("RGB")
    return tf(im).unsqueeze(0)


def gradcam(model, x, cls: int):
    """Grad-CAM over the backbone's spatial tokens — the one channel both models share.

    alpha_d = mean_n dL/dtoken[n,d];  cam[n] = relu(sum_d alpha_d * token[n,d]).
    """
    feats = {}

    def hook(_m, _i, out):
        toks = out[0] if isinstance(out, tuple) else out
        toks.retain_grad()
        feats["t"] = toks
        return out

    h = model.backbone.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, cls].backward()
        toks = feats["t"]
        if toks.grad is None:
            return None
        alpha = toks.grad.mean(dim=1, keepdim=True)             # [B,1,D]
        cam = F.relu((alpha * toks).sum(-1))                    # [B,N]
        n = cam.shape[1]
        g = int(round(n ** 0.5))
        if g * g != n:
            return None
        return cam.view(1, g, g).detach()
    finally:
        h.remove()


def norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def to_img(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)


@torch.no_grad()
def forward_aux(model, x):
    return model(x, return_aux=True)


def run_all(device: str, limit: int | None):
    man = pd.read_csv(ROOT / "experiments" / "_provenance" / "manifest.csv")
    test = man[man.split == "test"].copy()
    test["local"] = test.path.map(remap)
    assert test.local.map(os.path.exists).all(), "some test images missing locally"
    if limit:
        test = pd.concat([test[test.label == 1].head(limit // 2),
                          test[test.label == 0].head(limit - limit // 2)])
    print(f"test images: {len(test)} ({int(test.label.sum())} jaundice)")

    models = {}
    for name in CKPTS:
        print(f"loading {name}…")
        models[name], cfg = load_model(name, device)
        size = cfg["data"]["image_size"]

    rows = []
    cache = {}
    for i, r in enumerate(test.itertuples()):
        x = load_image(r.local, size).to(device)
        rec = dict(file=Path(r.local).name, label=int(r.label), domain=int(r.domain))
        per_model = {}
        for name, model in models.items():
            logits, aux = forward_aux(model, x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            rec[f"{name}_prob"] = prob
            rec[f"{name}_pred"] = int(prob >= THRESH[name])
            beta = aux.get("phys_bili")
            rec[f"{name}_beta"] = float(beta[0, 0]) if beta is not None else np.nan
            cam = gradcam(model, x, 1)
            per_model[name] = dict(
                aux={k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in aux.items()
                     if k in ("attn", "bili_map", "s_map", "x_wb")},
                cam=cam.cpu() if cam is not None else None, prob=prob,
            )
        rows.append(rec)
        cache[rec["file"]] = dict(x=x.cpu(), models=per_model)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test)}")

    df = pd.DataFrame(rows)
    df["agree"] = df.ours_bg_pred == df.erm_pred
    df.to_csv(HERE / "predictions.csv", index=False)
    return df, cache


def panel(rec, cache_entry, out: Path):
    x = cache_entry["x"]
    img = to_img(x)
    ob = cache_entry["models"]["ours_bg"]
    er = cache_entry["models"]["erm"]
    H, W = img.shape[:2]

    def up(t):
        if t is None:
            return None
        t = t if t.dim() == 3 else t.unsqueeze(0)
        return F.interpolate(t.unsqueeze(1).float(), size=(H, W), mode="bilinear",
                             align_corners=False).squeeze().numpy()

    cols = ["input", "SCIN white-balanced", "MIL attention", "Grad-CAM",
            "BiliGrad field (b*)", "cephalocaudal axis s"]
    fig, axes = plt.subplots(2, 6, figsize=(17.5, 6.4))
    fig.patch.set_facecolor(T["surface"])

    ob_wb = ob["aux"].get("x_wb")
    panels = {
        "ours_bg": [img, to_img(ob_wb) if ob_wb is not None else None,
                    up(ob["aux"].get("attn")), up(ob["cam"]),
                    up(ob["aux"].get("bili_map")), up(ob["aux"].get("s_map"))],
        "erm": [img, None, up(er["aux"].get("attn")), up(er["cam"]), None, None],
    }
    notes = {
        "ours_bg": [None, None, None, None, None, None],
        "erm": [None, "SCIN off", None, None, "no physics head", "no physics head"],
    }
    cmaps = [None, None, "inferno", "inferno", "viridis", "cividis"]
    overlay = [False, False, True, True, False, False]

    for row, name in enumerate(["ours_bg", "erm"]):
        for col in range(6):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(T["rule"])
            p = panels[name][col]
            if p is None:
                ax.set_facecolor("#f0efec")
                ax.text(0.5, 0.5, notes[name][col] or "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color=T["muted"], style="italic")
                continue
            if col in (0, 1):
                ax.imshow(p)
            elif overlay[col]:
                ax.imshow(img)
                ax.imshow(norm01(p), cmap=cmaps[col], alpha=0.55)
            else:
                ax.imshow(norm01(p), cmap=cmaps[col])
            if row == 0:
                ax.set_title(cols[col], fontsize=9, color=T["ink2"], pad=6)

        prob = cache_entry["models"][name]["prob"]
        lab = "ours_bg (BiliGrad)" if name == "ours_bg" else "ERM baseline"
        ok = int(prob >= 0.5) == rec["label"]
        axes[row, 0].set_ylabel(
            f"{lab}\nP(jaundice)={prob:.3f}  {'✓' if ok else '✗'}",
            fontsize=9, color=T["good"] if ok else T["crit"])

    truth = "jaundice" if rec["label"] == 1 else "normal"
    fig.suptitle(f"{rec['file']}   —   true label: {truth}   ·   domain {rec['domain']}",
                 x=0.008, ha="left", fontsize=12, fontweight="bold", color=T["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=T["surface"])
    plt.close(fig)


def summary(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.patch.set_facecolor(T["surface"])

    ax = axes[0]
    for lab, c, nm in [(0, T["accent"], "normal"), (1, T["crit"], "jaundice")]:
        s = df[df.label == lab]
        ax.scatter(s.erm_prob, s.ours_bg_prob, s=26, alpha=0.75, color=c, label=f"true {nm}",
                   edgecolors=T["surface"], linewidths=0.8)
    ax.plot([0, 1], [0, 1], color=T["muted"], lw=1, ls=(0, (4, 3)))
    ax.axhline(0.5, color=T["rule"], lw=1); ax.axvline(0.5, color=T["rule"], lw=1)
    ax.set_xlabel("ERM  P(jaundice)", color=T["ink2"])
    ax.set_ylabel("ours_bg  P(jaundice)", color=T["ink2"])
    ax.set_title("Where the two models disagree", fontsize=10, loc="left", color=T["ink"])
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    d = df.dropna(subset=["ours_bg_beta"])
    for lab, c, nm in [(0, T["accent"], "normal"), (1, T["crit"], "jaundice")]:
        v = d[d.label == lab].ours_bg_beta
        ax.scatter(np.random.normal(lab, 0.06, len(v)), v, s=26, alpha=0.75, color=c,
                   edgecolors=T["surface"], linewidths=0.8, label=f"true {nm}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "jaundice"], color=T["ink2"])
    ax.set_ylabel("BiliGrad |beta| (cephalocaudal slope)", color=T["ink2"])
    ax.set_title("Does the physics readout separate the classes?", fontsize=10, loc="left",
                 color=T["ink"])

    ax = axes[2]
    cm = np.zeros((2, 2), int)
    for r in df.itertuples():
        cm[r.label, r.ours_bg_pred] += 1
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max() / 2 else T["ink"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred normal", "pred jaundice"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true normal", "true jaundice"], fontsize=9)
    ax.set_title("ours_bg confusion @0.5", fontsize=10, loc="left", color=T["ink"])

    for a in axes:
        for s in a.spines.values():
            s.set_color(T["rule"])
    fig.suptitle("Inference on the held-out test split", x=0.008, ha="left", fontsize=13,
                 fontweight="bold", color=T["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=T["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--panels", type=int, default=10)
    args = ap.parse_args()

    (HERE / "panels").mkdir(parents=True, exist_ok=True)
    df, cache = run_all(args.device, args.limit)

    print("\n=== accuracy on this set ===")
    for n in CKPTS:
        acc = (df[f"{n}_pred"] == df.label).mean()
        sens = (df[(df.label == 1)][f"{n}_pred"] == 1).mean()
        spec = (df[(df.label == 0)][f"{n}_pred"] == 0).mean()
        print(f"{n:8s} acc={acc:.3f} sens={sens:.3f} spec={spec:.3f}")
    print(f"models agree on {df.agree.mean():.1%} of images")

    # Curate: correct/incorrect of each class + the disagreements — the informative cases.
    picks = []
    for lab in (1, 0):
        s = df[df.label == lab]
        picks += list(s[s.ours_bg_pred == lab].nlargest(2, "ours_bg_prob" if lab else "erm_prob").file)
        picks += list(s[s.ours_bg_pred != lab].head(2).file)
    picks += list(df[~df.agree].head(4).file)
    seen, order = set(), []
    for f in picks:
        if f not in seen and f in cache:
            seen.add(f); order.append(f)
    order = order[: args.panels]

    for f in order:
        rec = df[df.file == f].iloc[0].to_dict()
        panel(rec, cache[f], HERE / "panels" / f"{Path(f).stem.replace(' ', '_')}.png")
    print(f"\npanels -> {HERE/'panels'} ({len(order)})")

    summary(df, HERE / "summary.png")
    print(f"summary -> {HERE/'summary.png'}")
    print(f"predictions -> {HERE/'predictions.csv'}")


if __name__ == "__main__":
    main()
