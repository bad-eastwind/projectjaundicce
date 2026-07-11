"""Ablation / LOCO sweep runner.

Runs a matrix of (method x split) and/or component ablations in-process via train.run, collects each
run's TEST metrics, and writes results.csv + results.md (flat table + AUC pivot) into
experiments/sweep-<ts>/. Real sweeps run on Cloud/HPC; --smoke runs a tiny CPU version to verify the
runner; --dry-run prints the matrix without training.

Examples:
  PYTHONPATH=src python -m jaundice.train.sweep --matrix method   --holdouts indist,0,1,2
  PYTHONPATH=src python -m jaundice.train.sweep --matrix ablation  --holdouts 1
  PYTHONPATH=src python -m jaundice.train.sweep --matrix all --dry-run
  PYTHONPATH=src python -m jaundice.train.sweep --matrix method --smoke --holdouts 1
  # Kaggle/Cloud/HPC: override the dataset root for every run in the matrix (config files unchanged)
  PYTHONPATH=src python -m jaundice.train.sweep --matrix all --data-root /kaggle/input/jaundicedataset3/jaundicedataset3
"""
from __future__ import annotations
import argparse, csv, traceback
from pathlib import Path
import time

from jaundice.utils import load_config
from jaundice.train.train import run, apply_overrides

# method-comparison matrix: name -> (config, dg override)
METHODS = {
    "erm":      ("configs/erm.yaml", "erm"),
    "irm":      ("configs/irm.yaml", "irm"),
    "groupdro": ("configs/groupdro.yaml", "groupdro"),
    "ours":     ("configs/disentangle.yaml", None),
}

# ablations: base = ours (disentangle.yaml), toggle ONE component off
ABLATIONS = {
    "ours_full":        {},
    "ours_no_scin":     {"model.scin": False},
    "ours_no_disent":   {"model.disentangle.enable": False},          # -> falls back to BiliAxis
    "ours_no_causal":   {"model.causal.illuminant_adv": False, "model.causal.melanin_adv": False},
    "ours_no_mixstyle": {"model.mixstyle.enable": False},
    "ours_no_lora":     {"model.backbone": "convnext_tiny", "model.lora.enable": False},
}

SMOKE_OVERRIDES = {
    "data.image_size": 64, "train.device": "cpu", "train.epochs": 1,
    "train.num_workers": 0, "train.amp": False,
    "model.backbone": "convnext_tiny", "model.lora.enable": False, "model.pretrained": False,
}


def set_dotted(cfg: dict, key: str, val):
    o = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        o = o.setdefault(p, {})
    o[parts[-1]] = val


def make_cfg(config, overrides, holdout, dg, smoke, data_root=None):
    cfg = load_config(config)
    if data_root:
        set_dotted(cfg, "data.neo_skin_core", data_root)
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    apply_overrides(cfg, holdout=(None if holdout == "indist" else int(holdout)), dg=dg)
    if smoke:
        for k, v in SMOKE_OVERRIDES.items():
            set_dotted(cfg, k, v)
        cfg.setdefault("smoke", {}).update({"max_per_class": 6, "max_steps": 2})
    return cfg


def build_matrix(which, holdouts):
    runs = []
    if which in ("method", "all"):
        for name, (cfgp, dg) in METHODS.items():
            for h in holdouts:
                runs.append(dict(group="method", name=name, split=h, config=cfgp, overrides={}, dg=dg, holdout=h))
    if which in ("ablation", "all"):
        for name, ov in ABLATIONS.items():
            for h in holdouts:
                runs.append(dict(group="ablation", name=name, split=h,
                                 config="configs/disentangle.yaml", overrides=ov, dg=None, holdout=h))
    return runs


def fmt(r, k):
    if r is None:
        return "-"
    if "error" in r:
        return "ERR"
    v = r.get(k)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def write_results(rows, outdir):
    keys = ["group", "name", "split", "balanced_acc", "auc", "sensitivity",
            "specificity", "f1", "acc", "error", "outdir"]
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    lines = ["# Sweep results", "",
             "| group | run | split | bal_acc | auc | sens | spec | f1 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['group']} | {r['name']} | {r['split']} | "
                     f"{fmt(r,'balanced_acc')} | {fmt(r,'auc')} | {fmt(r,'sensitivity')} | "
                     f"{fmt(r,'specificity')} | {fmt(r,'f1')} |")

    names, splits = [], []
    for r in rows:
        if r["name"] not in names: names.append(r["name"])
        if r["split"] not in splits: splits.append(r["split"])
    cell = {(r["name"], r["split"]): r for r in rows}
    lines += ["", "## AUC by run x split", "",
              "| run | " + " | ".join(str(s) for s in splits) + " |",
              "|---|" + "|".join("---" for _ in splits) + "|"]
    for n in names:
        lines.append("| " + n + " | " + " | ".join(fmt(cell.get((n, s)), "auc") for s in splits) + " |")
    (outdir / "results.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="method", choices=["method", "ablation", "all"])
    ap.add_argument("--holdouts", default="indist,0,1,2",
                    help="comma list of splits: 'indist' or domain ids")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help="override data.neo_skin_core for every run (e.g. Kaggle-mounted path)")
    args = ap.parse_args()

    holdouts = [h.strip() for h in args.holdouts.split(",") if h.strip()]
    runs = build_matrix(args.matrix, holdouts)
    print(f"matrix={args.matrix} | {len(runs)} runs | holdouts={holdouts}")

    if args.dry_run:
        for r in runs:
            print(f"  [{r['group']}] {r['name']:16s} split={r['split']:6} "
                  f"config={r['config']} dg={r['dg']} overrides={r['overrides']}")
        return

    outdir = Path("experiments") / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(runs):
        tag = f"{r['group']}-{r['name']}-{r['split']}"
        print(f"\n===== [{i+1}/{len(runs)}] {tag} =====")
        try:
            cfg = make_cfg(r["config"], r["overrides"], r["holdout"], r["dg"], args.smoke,
                          data_root=args.data_root)
            res = run(cfg, tag=tag)
            rows.append({"group": r["group"], "name": r["name"], "split": r["split"],
                         **res["test"], "outdir": res["outdir"]})
        except Exception as e:  # keep sweeping even if one run fails
            traceback.print_exc()
            rows.append({"group": r["group"], "name": r["name"], "split": r["split"], "error": str(e)})
        write_results(rows, outdir)   # incremental -> partial results survive a crash

    print(f"\nsweep done -> {outdir}/results.md")


if __name__ == "__main__":
    main()
