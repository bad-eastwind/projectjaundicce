"""Canonical trainer. Same code path for local smoke and Cloud/HPC — smoke just caps data + steps
via the `smoke:` block in configs/smoke.yaml, so smoke-testing exercises the REAL trainer.

Local smoke:  PYTHONPATH=src python -m jaundice.train.train --config configs/smoke.yaml
Cloud/HPC:    PYTHONPATH=src python -m jaundice.train.train --config configs/base.yaml --tag v1_all
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from jaundice.utils import load_config, seed_everything, pick_device
from jaundice.data.dataset import build_loaders
from jaundice.models.net import JaundiceNet
from jaundice.models.causal import compute_ita
from jaundice.train.metrics import (classification_metrics, pick_threshold, fairness_report)
from jaundice.train.losses import aux_losses
from jaundice.train.dg import DGObjective


@torch.no_grad()
def collect(model, loader, dev, want_ita: bool = False):
    """Returns (y, prob_pos[, ita]) over a loader. ITA (skin-tone angle) is measured on the SCIN
    white-balanced image over the model's own skin-attention, for the fairness stratification."""
    model.eval()
    ys, ps, itas = [], [], []
    for b in loader:
        x = b["image"].to(dev)
        if want_ita:
            logits, aux = model(x, return_aux=True)
            attn_up = F.interpolate(aux["attn"].unsqueeze(1), size=x.shape[-2:],
                                    mode="bilinear", align_corners=False).squeeze(1)
            itas.append(compute_ita(aux["x_wb"], attn_up).squeeze(1).float().cpu().numpy())
        else:
            logits = model(x)
        ps.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
        ys.append(b["label"].numpy())
    if not ys:
        return (np.array([]), np.array([]), np.array([])) if want_ita else (np.array([]), np.array([]))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return (y, p, np.concatenate(itas)) if want_ita else (y, p)


def evaluate(model, loader, dev, thresh: float = 0.5) -> dict:
    y, p = collect(model, loader, dev)
    if len(y) == 0:
        return {}
    return classification_metrics(y, p, thresh)


def run(cfg: dict, tag: str = "run") -> dict:
    """Train + evaluate for one config. Returns {test, best_val_score, outdir}. Used by the CLI and
    by the sweep runner (train/sweep.py)."""
    seed_everything(cfg["seed"])
    dev = pick_device(cfg["train"]["device"])
    is_smoke = "smoke" in cfg
    max_steps = cfg.get("smoke", {}).get("max_steps", None)
    print(f"device: {dev} | smoke: {is_smoke}")

    loaders = build_loaders(cfg, smoke=is_smoke)
    m, t = cfg["model"], cfg["train"]
    pretrained = (not is_smoke) and m.get("pretrained", True)
    model = JaundiceNet(backbone=m["backbone"], scin=m["scin"], pretrained=pretrained,
                        bili_axis=m.get("bili_axis", True), causal=m.get("causal", {}),
                        lora=m.get("lora", {}), disentangle=m.get("disentangle", {}),
                        mixstyle=m.get("mixstyle", {}), bili_grad=m.get("bili_grad", {})).to(dev)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    physics = ("bili_grad" if m.get("bili_grad", {}).get("enable")
               else "disentangle" if m.get("disentangle", {}).get("enable")
               else "bili_axis" if m.get("bili_axis", True) else "none")
    print(f"model: {m['backbone']} scin={m['scin']} physics={physics} "
          f"lora={m.get('lora', {}).get('enable', False)} "
          f"causal={m.get('causal', {})} pretrained={pretrained}")
    print(f"params: total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M "
          f"({100*trainable/total:.1f}%)")

    dgc = t.get("dg", {}) or {}
    dg_method = dgc.get("method", "erm")
    num_groups = int(cfg["data"].get("domain", {}).get("k", 1) or 1)
    dg = DGObjective(dg_method, num_groups=num_groups, lam=dgc.get("lambda", 1.0),
                     eta=dgc.get("eta", 0.01), anneal_steps=dgc.get("anneal_steps", 0))
    print(f"dg: {dg_method} (groups={num_groups})")

    opt = optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(t["epochs"], 1))
    # IRM needs a second-order grad; skip AMP for it to avoid GradScaler interactions
    use_amp = bool(t.get("amp", False)) and dev == "cuda" and dg_method != "irm"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    outdir = Path("experiments") / f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    best, hist = -1.0, []
    for epoch in range(t["epochs"]):
        model.train()
        run, nb = 0.0, 0
        for step, b in enumerate(loaders["train"]):
            if max_steps is not None and step >= max_steps:
                break
            x, y = b["image"].to(dev), b["label"].to(dev)
            dom = b["domain"].to(dev)
            opt.zero_grad()
            if use_amp:
                with torch.autocast("cuda"):
                    logits, aux = model(x, return_aux=True)
                    cls_loss, _ = dg(logits, y, dom)
                    extra, logs = aux_losses(aux, cfg)
                    loss = cls_loss + extra
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logits, aux = model(x, return_aux=True)
                cls_loss, _ = dg(logits, y, dom)
                extra, logs = aux_losses(aux, cfg)
                loss = cls_loss + extra
                loss.backward(); opt.step()
            run += loss.item(); nb += 1
        sched.step()

        vm = evaluate(model, loaders["val"], dev)
        row = {"epoch": epoch, "train_loss": run / max(nb, 1),
               **{f"val_{k}": v for k, v in vm.items()}}
        hist.append(row)
        score = vm.get("auc")
        if score is None or np.isnan(score):
            score = vm.get("balanced_acc", 0.0)
        print(f"epoch {epoch}: loss={run/max(nb,1):.4f} "
              f"val_bacc={vm.get('balanced_acc',float('nan')):.3f} "
              f"val_auc={vm.get('auc',float('nan')):.3f}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "cfg": cfg, "val": vm, "epoch": epoch},
                       outdir / "best.pt")

    # --- final test evaluation: threshold-free + val-tuned operating points + fairness ---
    thr_cfg = t.get("threshold", {}) or {}
    y_val, p_val = collect(model, loaders["val"], dev)
    t_youden = pick_threshold(y_val, p_val, "youden") if len(y_val) else 0.5
    t_screen = (pick_threshold(y_val, p_val, "target_sens",
                thr_cfg.get("target_sensitivity", 0.95)) if len(y_val) else 0.5)

    y_test, p_test, ita_test = collect(model, loaders["test"], dev, want_ita=True)
    tm = classification_metrics(y_test, p_test, 0.5) if len(y_test) else {}
    tm_youden = classification_metrics(y_test, p_test, t_youden) if len(y_test) else {}
    tm_screen = classification_metrics(y_test, p_test, t_screen) if len(y_test) else {}
    fairness = fairness_report(y_test, p_test, ita_test, t_youden) if len(y_test) else {}

    json.dump({"tag": tag, "history": hist, "best_val_score": best,
               "test": tm, "test_youden": tm_youden, "test_screening": tm_screen,
               "thresholds": {"youden": t_youden, "screening": t_screen},
               "test_fairness": fairness},
              open(outdir / "metrics.json", "w"), indent=2)
    print("TEST @0.5:", {k: round(v, 4) for k, v in tm.items()})
    print(f"TEST @youden({t_youden:.2f}): bacc={tm_youden.get('balanced_acc', float('nan')):.3f} "
          f"sens={tm_youden.get('sensitivity', float('nan')):.3f} "
          f"spec={tm_youden.get('specificity', float('nan')):.3f}")
    print(f"skin-tone fairness gap (bacc): {fairness.get('bacc_gap', float('nan')):.3f}")
    print(f"saved: {outdir}")
    return {"test": tm, "test_youden": tm_youden, "test_screening": tm_screen,
            "test_fairness": fairness, "best_val_score": best, "outdir": str(outdir)}


def apply_overrides(cfg: dict, holdout: int | None = None, dg: str | None = None) -> dict:
    if holdout is not None:
        cfg["data"]["split_mode"] = "leave_domain_out"
        cfg["data"].setdefault("domain", {})["holdout"] = holdout
    if dg is not None:
        cfg["train"].setdefault("dg", {})["method"] = dg
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--holdout", type=int, default=None,
                    help="LOCO: force leave_domain_out with this held-out domain id")
    ap.add_argument("--dg", default=None, choices=["erm", "irm", "groupdro"],
                    help="override DG classification objective")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), holdout=args.holdout, dg=args.dg)
    run(cfg, args.tag)


if __name__ == "__main__":
    main()
