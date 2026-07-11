"""Build the retrieval bank from the train split and produce case-based explanations for test images.

Run (smoke):  PYTHONPATH=src python -m jaundice.explain.run --config configs/smoke_disentangle.yaml
Run (cloud):  PYTHONPATH=src python -m jaundice.explain.run --config configs/disentangle.yaml \
                  --ckpt experiments/<run>/best.pt --num-queries 20
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch

from jaundice.utils import load_config, seed_everything, pick_device
from jaundice.data.dataset import build_loaders
from jaundice.models.net import JaundiceNet
from jaundice.explain.retrieval import RetrievalBank


def build_model(cfg, dev, ckpt=None, is_smoke=False):
    m = cfg["model"]
    pretrained = (not is_smoke) and m.get("pretrained", True)
    model = JaundiceNet(backbone=m["backbone"], scin=m["scin"], pretrained=pretrained,
                        bili_axis=m.get("bili_axis", True), causal=m.get("causal", {}),
                        lora=m.get("lora", {}), disentangle=m.get("disentangle", {}),
                        mixstyle=m.get("mixstyle", {})).to(dev)
    if ckpt:
        state = torch.load(ckpt, map_location=dev, weights_only=True)["model"]
        model.load_state_dict(state)
        print(f"loaded checkpoint: {ckpt}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/disentangle.yaml")
    ap.add_argument("--ckpt", default=None, help="trained best.pt (omit for smoke/random)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--num-queries", type=int, default=8)
    ap.add_argument("--tag", default="explain")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    dev = pick_device(cfg["train"]["device"])
    is_smoke = "smoke" in cfg
    loaders = build_loaders(cfg, smoke=is_smoke)

    model = build_model(cfg, dev, ckpt=args.ckpt, is_smoke=is_smoke)
    bank = RetrievalBank.build(model, loaders["train"], dev)
    print(f"bank: {bank.emb.shape[0]} exemplars, dim={bank.emb.shape[1]}")

    outdir = Path("experiments") / f"{args.tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    bank.save(outdir / "retrieval_bank.pt")

    entries, correct, n = [], 0, 0
    model.eval()
    with torch.no_grad():
        for b in loaders["test"]:
            logits, aux = model(b["image"].to(dev), return_aux=True)
            pc = torch.softmax(logits, 1)[:, 1].float().cpu()
            pr, ent, idx, sims = bank.query(aux["embedding"], k=args.k, tau=args.tau)
            for i in range(len(b["label"])):
                neighbors = [{"path": os.path.basename(bank.paths[j]),
                              "label": int(bank.labels[j]),
                              "sim": round(float(sims[i, t]), 3)}
                             for t, j in enumerate(idx[i].tolist())]
                entries.append({
                    "query": os.path.basename(b["path"][i]),
                    "true_label": int(b["label"][i]),
                    "retrieval_prob": round(float(pr[i]), 3),
                    "classifier_prob": round(float(pc[i]), 3),
                    "uncertainty": round(float(ent[i]), 3),
                    "neighbors": neighbors,
                })
                correct += int((pr[i] >= 0.5) == bool(b["label"][i])); n += 1
                if len(entries) >= args.num_queries:
                    break
            if len(entries) >= args.num_queries:
                break

    json.dump(entries, open(outdir / "explanations.json", "w"), indent=2)
    _write_md(entries, correct / max(n, 1), outdir / "explanations.md", args)
    print(f"retrieval agreement with labels on {n} queries: {correct/max(n,1):.3f}")
    print(f"saved: {outdir}")


def _write_md(entries, acc, path, args):
    lbl = {0: "normal", 1: "jaundice"}
    lines = [f"# Retrieval explanations (k={args.k}, tau={args.tau})", "",
             f"Non-parametric retrieval agreement with labels: **{acc:.3f}** on {len(entries)} queries.", ""]
    for e in entries:
        lines += [f"### {e['query']}  (true: {lbl[e['true_label']]})",
                  f"- retrieval P(jaundice) = {e['retrieval_prob']} | classifier = {e['classifier_prob']} | "
                  f"uncertainty = {e['uncertainty']}",
                  "- nearest exemplars:"]
        for nb in e["neighbors"]:
            lines.append(f"  - `{nb['path']}` ({lbl[nb['label']]}, sim {nb['sim']})")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
