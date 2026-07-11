"""Local smoke test — verify the full pipeline end-to-end on a tiny subset, CPU, a few steps.
NEVER a real training run. Real training happens on Cloud GPU / HPC.

Run:  PYTHONPATH=src python -m jaundice.train.smoke --config configs/smoke.yaml
"""
from __future__ import annotations
import argparse
import torch
from torch import nn, optim

from jaundice.utils import load_config, seed_everything, pick_device
from jaundice.data.dataset import build_loaders
from jaundice.models.net import JaundiceNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    dev = pick_device(cfg["train"]["device"])
    print(f"device: {dev} | image_size: {cfg['data']['image_size']}")

    loaders = build_loaders(cfg, smoke=True)
    model = JaundiceNet(backbone=cfg["model"]["backbone"],
                        scin=cfg["model"]["scin"], pretrained=False).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg['model']['backbone']} | scin={cfg['model']['scin']} | params={n_params/1e6:.1f}M")

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    lossf = nn.CrossEntropyLoss()

    model.train()
    max_steps = cfg.get("smoke", {}).get("max_steps", 2)
    it = iter(loaders["train"])
    for step in range(max_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loaders["train"]); batch = next(it)
        x, y = batch["image"].to(dev), batch["label"].to(dev)
        logits, aux = model(x, return_aux=True)
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        illum0 = [round(v, 3) for v in aux["illum"][0].tolist()] if aux["illum"] is not None else None
        print(f"  step {step}: loss={loss.item():.4f} | logits={tuple(logits.shape)} | "
              f"attn_map={tuple(aux['attn'].shape)} | illum[0]={illum0}")

    model.eval()
    with torch.no_grad():
        vb = next(iter(loaders["val"]))
        vlogits = model(vb["image"].to(dev))
    assert vlogits.shape[1] == 2
    print(f"  val forward: logits={tuple(vlogits.shape)} OK")
    print(f"SMOKE PASS ({max_steps} train steps + 1 val forward)")


if __name__ == "__main__":
    main()
