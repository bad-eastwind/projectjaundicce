"""Aggregate every experiments/*/metrics.json into flat analysis tables.

Keys (method, split, seed) come from the sweep results.csv files, which are the
authoritative record of what each run was; metrics.json supplies the depth
(operating points, per-stratum fairness, training history).

    PYTHONPATH=src python observations/aggregate.py

Writes observations/tables/{runs,strata,history,summary_auc,summary_bacc,
deltas_vs_erm,ablation}.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = Path(__file__).resolve().parent / "tables"

# Flagship "insurance" runs trained outside the sweep matrix.
FLAGSHIP = {
    "erm_indist": ("flagship", "erm", "indist", 42),
    "ourbg_indist": ("flagship", "ours_bg", "indist", 42),
}

STRATA_ORDER = ["very_light", "light", "intermediate", "tan", "brown", "dark"]


def run_keys() -> pd.DataFrame:
    """(group, name, split, seed) -> run dir, from the sweep manifests."""
    rows = []
    for csv in sorted(EXP.glob("sweep-*/results.csv")):
        df = pd.read_csv(csv)
        if "seed" not in df.columns:
            continue  # pre-seed smoke sweeps from 2026-07-11
        for r in df.itertuples():
            outdir = Path(str(r.outdir))
            if not (EXP / outdir.name / "metrics.json").exists():
                continue
            rows.append(
                dict(
                    run=outdir.name,
                    group=r.group,
                    method=r.name,
                    split=str(r.split),
                    seed=int(r.seed),
                    sweep=csv.parent.name,
                )
            )
    for d in sorted(EXP.glob("*/metrics.json")):
        name = d.parent.name
        for prefix, (group, method, split, seed) in FLAGSHIP.items():
            if name.startswith(prefix):
                rows.append(
                    dict(run=name, group=group, method=method, split=split, seed=seed, sweep="")
                )
    df = pd.DataFrame(rows).drop_duplicates(subset="run")
    return df


def flatten(run: str) -> dict:
    m = json.loads((EXP / run / "metrics.json").read_text())
    out: dict = {"run": run, "best_val_auc": m.get("best_val_score")}
    for block in ("test", "test_youden", "test_screening"):
        for k, v in m.get(block, {}).items():
            out[f"{block}.{k}"] = v
    for k, v in m.get("thresholds", {}).items():
        out[f"thr.{k}"] = v
    fair = m.get("test_fairness", {})
    out["fair.bacc_gap"] = fair.get("bacc_gap")
    out["fair.sensitivity_gap"] = fair.get("sensitivity_gap")

    ps = fair.get("per_stratum", {})
    out["fair.n_strata"] = len(ps)
    ns = [v["n"] for v in ps.values()]
    out["fair.min_stratum_n"] = min(ns) if ns else np.nan
    out["fair.max_stratum_n"] = max(ns) if ns else np.nan
    # Which stratum actually sets the reported gap, and how many images back it.
    baccs = {k: v["balanced_acc"] for k, v in ps.items() if not math.isnan(v.get("balanced_acc", np.nan))}
    if baccs:
        worst = min(baccs, key=baccs.get)
        out["fair.worst_stratum"] = worst
        out["fair.worst_stratum_n"] = ps[worst]["n"]
        out["fair.worst_stratum_bacc"] = baccs[worst]
    # Gap recomputed over strata with enough images to mean anything.
    for floor in (10, 20):
        big = {k: v for k, v in ps.items() if v["n"] >= floor and not math.isnan(v.get("balanced_acc", np.nan))}
        vals = [v["balanced_acc"] for v in big.values()]
        out[f"fair.bacc_gap_n{floor}"] = (max(vals) - min(vals)) if len(vals) >= 2 else np.nan
        out[f"fair.n_strata_n{floor}"] = len(big)

    hist = m.get("history", [])
    out["epochs"] = len(hist)
    if hist:
        aucs = [h.get("val_auc") for h in hist if h.get("val_auc") is not None]
        out["best_epoch"] = int(np.argmax(aucs)) if aucs else np.nan
    return out


def strata_long(keys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in keys.itertuples():
        m = json.loads((EXP / r.run / "metrics.json").read_text())
        for name, v in m.get("test_fairness", {}).get("per_stratum", {}).items():
            rows.append(
                dict(
                    run=r.run, group=r.group, method=r.method, split=r.split, seed=r.seed,
                    stratum=name, n=v["n"], pos=v["pos"],
                    balanced_acc=v["balanced_acc"], sensitivity=v["sensitivity"],
                    specificity=v["specificity"], auc=v["auc"],
                )
            )
    return pd.DataFrame(rows)


def history_long(keys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in keys.itertuples():
        m = json.loads((EXP / r.run / "metrics.json").read_text())
        for h in m.get("history", []):
            rows.append(dict(run=r.run, method=r.method, split=r.split, seed=r.seed, **h))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = run_keys()
    runs = keys.merge(pd.DataFrame([flatten(r) for r in keys.run]), on="run")
    runs = runs.sort_values(["group", "method", "split", "seed"])
    runs.to_csv(OUT / "runs.csv", index=False)

    strata_long(keys).to_csv(OUT / "strata.csv", index=False)
    history_long(keys).to_csv(OUT / "history.csv", index=False)

    grid = runs[runs.group == "method"]

    # Method x split summaries over the 3 seeds.
    for metric, fname in [
        ("test.auc", "summary_auc.csv"),
        ("test_youden.balanced_acc", "summary_bacc.csv"),
        ("test_screening.sensitivity", "summary_screen_sens.csv"),
    ]:
        s = (
            grid.groupby(["method", "split"])[metric]
            .agg(mean="mean", std="std", min="min", max="max", n="count")
            .reset_index()
        )
        s["ci95_halfwidth"] = 1.96 * s["std"] / np.sqrt(s["n"])  # indicative only at n=3
        s.to_csv(OUT / fname, index=False)

    # Paired per-seed deltas vs ERM: same seed, same split -> removes seed variance.
    base = grid[grid.method == "erm"].set_index(["split", "seed"])
    rows = []
    for method in ["irm", "groupdro", "ours", "ours_bg"]:
        sub = grid[grid.method == method].set_index(["split", "seed"])
        for metric in ["test.auc", "test_youden.balanced_acc"]:
            d = (sub[metric] - base[metric]).dropna()
            for split in d.index.get_level_values("split").unique():
                v = d.xs(split, level="split").values
                rows.append(
                    dict(
                        method=method, split=split, metric=metric, n_seeds=len(v),
                        mean_delta=v.mean(), std_delta=v.std(ddof=1),
                        min_delta=v.min(), max_delta=v.max(),
                        beats_erm_all_seeds=bool((v > 0).all()),
                    )
                )
    pd.DataFrame(rows).to_csv(OUT / "deltas_vs_erm.csv", index=False)

    abl = runs[runs.group == "ablation"][
        ["method", "seed", "test.auc", "test_youden.balanced_acc",
         "test_youden.sensitivity", "test_youden.specificity",
         "fair.bacc_gap", "fair.bacc_gap_n20", "fair.worst_stratum",
         "fair.worst_stratum_n", "fair.min_stratum_n", "epochs"]
    ]
    abl.to_csv(OUT / "ablation.csv", index=False)

    print(f"runs={len(runs)}  method-grid={len(grid)}  ablations={len(abl)}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
