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
from torch import nn, optim

from jaundice.utils import load_config, seed_everything, pick_device
from jaundice.data.dataset import build_loaders
from jaundice.models.net import JaundiceNet
from jaundice.train.metrics import classification_metrics
from jaundice.train.losses import aux_losses
from jaundice.train.dg import DGObjective


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    ys, ps = [], []
    for b in loader:
        logits = model(b["image"].to(dev))
        ps.append(torch.softmax(logits, 1)[:, 1].float().cpu().numpy())
        ys.append(b["label"].numpy())
    if not ys:
        return {}
    return classification_metrics(np.concatenate(ys), np.concatenate(ps))


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
                        mixstyle=m.get("mixstyle", {})).to(dev)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    physics = ("disentangle" if m.get("disentangle", {}).get("enable")
               else ("bili_axis" if m.get("bili_axis", True) else "none"))
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

    tm = evaluate(model, loaders["test"], dev)
    json.dump({"tag": tag, "history": hist, "test": tm, "best_val_score": best},
              open(outdir / "metrics.json", "w"), indent=2)
    print("TEST:", {k: round(v, 4) for k, v in tm.items()})
    print(f"saved: {outdir}")
    return {"test": tm, "best_val_score": best, "outdir": str(outdir)}


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
