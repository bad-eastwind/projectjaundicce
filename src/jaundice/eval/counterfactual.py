"""Counterfactual-recoloring evaluation (causal-validity of the bilirubin readout).

We apply physically-plausible interventions that change the NUISANCES while leaving the true bilirubin
signal fixed, then ask whether the model's prediction and its physics readout stay invariant:

  - illuminant / white-balance shift : per-channel von-Kries gain (warm/cool/tint). SCIN is meant to
    undo this, so a causally-valid pipeline barely moves; a lighting-shortcut model swings.
  - melanin / skin-tone shift        : move skin pixels along the melanin axis in CIELAB (shift L*, a*)
    while HOLDING b* (the yellow / bilirubin axis) FIXED — same bilirubin, different skin tone. A
    fair, causal model is invariant; a skin-tone-confounded model swings.

For each image we compare, across the counterfactual set:
  - P(jaundice) swing + class flip-rate      (prediction stability; lower = more causal)
  - physics readout (beta / bilirubin index) coefficient-of-variation (readout stability)
  - a RAW reference: skin-region mean b* of the *input* — this SHOULD move under illuminant CF,
    proving the intervention is real and that SCIN, not luck, is what stabilizes the readout.

Pass --baseline-config/--baseline-ckpt to print a naive model side-by-side (expected: larger swing).

Run (smoke, random weights, just checks plumbing):
  PYTHONPATH=src python -m jaundice.eval.counterfactual --config configs/smoke_bili_grad.yaml --num 8
Run (cloud):
  PYTHONPATH=src python -m jaundice.eval.counterfactual --config configs/bili_grad.yaml \
      --ckpt experiments/<run>/best.pt --num 64 \
      --baseline-config configs/erm.yaml --baseline-ckpt experiments/<erm>/best.pt
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from jaundice.utils import load_config, seed_everything, pick_device
from jaundice.data.dataset import build_loaders
from jaundice.models.causal import rgb_to_lab
from jaundice.explain.run import build_model

# --- counterfactual interventions -------------------------------------------------------------

# illuminant white-balance gains (per-channel), normalized to unit mean so brightness is preserved
CF_ILLUM = [("warm+", (1.20, 1.00, 0.80)), ("warm++", (1.35, 1.00, 0.70)),
            ("cool+", (0.80, 1.00, 1.20)), ("cool++", (0.70, 1.00, 1.35)),
            ("green", (0.95, 1.10, 0.95)), ("magenta", (1.08, 0.90, 1.08))]
# melanin/skin-tone shifts in CIELAB: (dL*, da*); db* == 0 keeps the bilirubin axis fixed
CF_MELA = [("darker+", (-10.0, 3.0)), ("darker++", (-20.0, 6.0)), ("darker+++", (-30.0, 9.0)),
           ("lighter+", (10.0, -3.0)), ("lighter++", (20.0, -5.0))]


def lab_to_rgb(L, a, b):
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    d = 6.0 / 29.0
    inv = lambda t: torch.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))
    X, Y, Z = inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bb = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    gam = lambda c: torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
    return torch.stack([gam(r.clamp_min(0)), gam(g.clamp_min(0)), gam(bb.clamp_min(0))], 1).clamp(0, 1)


def skin_mask(x: torch.Tensor) -> torch.Tensor:
    """Classical RGB+YCbCr skin rule -> [B,1,H,W] float mask (dependency-free)."""
    r, g, b = x[:, 0] * 255, x[:, 1] * 255, x[:, 2] * 255
    mx, mn = x.amax(1) * 255, x.amin(1) * 255
    rule = (r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) & ((r - g).abs() > 15) & (r > g) & (r > b)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    ycc = (cr > 133) & (cr < 180) & (cb > 77) & (cb < 128) & (y > 60)
    return (rule | ycc).unsqueeze(1).float()


def illuminant_shift(x, gain):
    g = torch.tensor(gain, device=x.device, dtype=x.dtype)
    g = g / g.mean()
    return (x * g.view(1, 3, 1, 1)).clamp(0, 1)


def melanin_shift(x, dL, da, mask):
    L, a, b = rgb_to_lab(x)
    shifted = lab_to_rgb(L + dL, a + da, b)          # b* (bilirubin axis) held fixed
    return mask * shifted + (1 - mask) * x           # apply on skin only; leave background exact


@torch.no_grad()
def readout(model, x):
    """Returns P(jaundice) [B], physics beta [B], raw skin-b* of the INPUT [B] (reference)."""
    logits, aux = model(x, return_aux=True)
    P = torch.softmax(logits, 1)[:, 1].float().cpu()
    pb = aux.get("phys_bili")
    beta = (pb[:, 0].float().cpu() if pb is not None else torch.full((x.shape[0],), float("nan")))
    _, _, b = rgb_to_lab(x)
    m = skin_mask(x)[:, 0]
    raw_b = ((b * m).sum((1, 2)) / (m.sum((1, 2)) + 1e-6)).float().cpu()
    return P.numpy(), beta.numpy(), raw_b.numpy()


def _gather(loader, dev, num):
    xs = []
    for b in loader:
        xs.append(b["image"])
        if sum(t.shape[0] for t in xs) >= num:
            break
    if not xs:
        raise RuntimeError("no images in loader")
    return torch.cat(xs)[:num].to(dev)


def evaluate_model(model, X):
    base_P, base_beta, base_raw = readout(model, X)
    out = {}
    for cname, cfset, fn in [("illuminant", CF_ILLUM, "illum"), ("melanin", CF_MELA, "mela")]:
        P_cols, beta_cols, raw_cols = [base_P], [base_beta], [base_raw]
        mask = skin_mask(X) if fn == "mela" else None
        for _, params in cfset:
            Xc = (illuminant_shift(X, params) if fn == "illum"
                  else melanin_shift(X, params[0], params[1], mask))
            P, beta, raw = readout(model, Xc)
            P_cols.append(P); beta_cols.append(beta); raw_cols.append(raw)
        P = np.stack(P_cols, 1); beta = np.stack(beta_cols, 1); raw = np.stack(raw_cols, 1)
        base = P[:, :1]
        pred_swing = float(np.mean(np.abs(P[:, 1:] - base)))                 # mean |ΔP| from original
        flip_rate = float(np.mean((P[:, 1:] >= 0.5) != (base >= 0.5)))
        beta_cov = float(np.nanmean(np.nanstd(beta, 1) / (np.nanmean(np.abs(beta), 1) + 1e-6)))
        raw_swing = float(np.mean(np.abs(raw[:, 1:] - raw[:, :1])))          # reference (should move)
        out[cname] = {"pred_swing": pred_swing, "flip_rate": flip_rate,
                      "beta_cov": beta_cov, "raw_bstar_swing": raw_swing}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bili_grad.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--split", default="test", choices=["test", "val", "train_eval"])
    ap.add_argument("--baseline-config", default=None)
    ap.add_argument("--baseline-ckpt", default=None)
    ap.add_argument("--tag", default="cf")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    dev = pick_device(cfg["train"]["device"])
    is_smoke = "smoke" in cfg
    loaders = build_loaders(cfg, smoke=is_smoke)
    X = _gather(loaders[args.split], dev, args.num)
    print(f"counterfactual eval on {X.shape[0]} {args.split} images")

    model = build_model(cfg, dev, ckpt=args.ckpt, is_smoke=is_smoke)
    results = {"ours": evaluate_model(model, X)}
    if args.baseline_config:
        bcfg = load_config(args.baseline_config)
        bmodel = build_model(bcfg, dev, ckpt=args.baseline_ckpt, is_smoke=("smoke" in bcfg))
        results["baseline"] = evaluate_model(bmodel, X)

    outdir = Path("experiments") / f"{args.tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump({"n": int(X.shape[0]), "ckpt": args.ckpt, "results": results},
              open(outdir / "counterfactual.json", "w"), indent=2)
    _write_md(results, X.shape[0], outdir / "counterfactual.md", args)
    for who, r in results.items():
        print(f"\n[{who}]")
        for cf, m in r.items():
            print(f"  {cf:11s} pred_swing={m['pred_swing']:.3f} flip={m['flip_rate']:.3f} "
                  f"beta_cov={m['beta_cov']:.3f} (raw_b* swing={m['raw_bstar_swing']:.3f})")
    print(f"\nsaved: {outdir}")


def _write_md(results, n, path, args):
    lines = [f"# Counterfactual-recoloring invariance ({n} {args.split} images)", "",
             "Interventions change the nuisance, hold true bilirubin fixed. Lower pred_swing / flip / "
             "beta_cov = more causally robust. `raw_b*_swing` is the input-side reference that SHOULD "
             "move (proves the intervention is real).", ""]
    for who, r in results.items():
        lines += [f"## {who}", "",
                  "| intervention | pred_swing | flip_rate | beta_cov | raw_b*_swing (ref) |",
                  "|---|---|---|---|---|"]
        for cf, m in r.items():
            lines.append(f"| {cf} | {m['pred_swing']:.3f} | {m['flip_rate']:.3f} | "
                         f"{m['beta_cov']:.3f} | {m['raw_bstar_swing']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
