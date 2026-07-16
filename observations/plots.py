"""Render the observation figures from observations/tables/*.csv.

    PYTHONPATH=src python observations/plots.py

Writes observations/figures/{light,dark}/*.png. Palette is the validated
categorical set (adjacent-pair CVD dE 9.1 light / 8.4 dark); the three
sub-3:1 slots are always direct-labelled, which is the documented relief.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TAB = HERE / "tables"

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series=["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"],
        good="#0ca30c", warning="#fab219", critical="#d03b3b", serious="#ec835a",
        band="#f0efec",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series=["#3987e5", "#008300", "#d55181", "#c98500", "#199e70"],
        good="#0ca30c", warning="#fab219", critical="#d03b3b", serious="#ec835a",
        band="#383835",
    ),
}

METHODS = ["erm", "irm", "groupdro", "ours", "ours_bg"]
LABEL = {
    "erm": "ERM", "irm": "IRM", "groupdro": "GroupDRO",
    "ours": "Ours (disentangle)", "ours_bg": "Ours (BiliGrad)",
}
SPLITS = ["indist", "0", "1", "2"]
SPLIT_LABEL = {
    "indist": "In-distribution", "0": "LOCO domain 0  (n=454)",
    "1": "LOCO domain 1  (n=235)", "2": "LOCO domain 2  (n=71)",
}


def style(T):
    mpl.rcParams.update({
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"],
        "savefig.facecolor": T["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": T["ink"], "axes.labelcolor": T["ink2"],
        "xtick.color": T["muted"], "ytick.color": T["muted"],
        "axes.edgecolor": T["axis"], "axes.linewidth": 0.8,
        "grid.color": T["grid"], "grid.linewidth": 0.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def title(fig, t, sub=None, T=None):
    """Place title/subtitle in inches from the top, so they hold at any figure height.

    Returns the tight_layout rect top the caller should reserve.
    """
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.22 / h, t, ha="left", va="top", fontsize=13,
             fontweight="bold", color=T["ink"])
    if sub:
        fig.text(0.012, 1 - 0.50 / h, sub, ha="left", va="top", fontsize=9.5, color=T["ink2"])
    return 1 - (0.78 if sub else 0.45) / h


# ---------------------------------------------------------------- figures

def fig_generalization(runs, T, out):
    """Per-seed dots + mean. At n=3 a bar+errorbar would hide the spread."""
    g = runs[runs.group == "method"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.1), sharex=True)
    for ax, sp in zip(axes, SPLITS):
        sub = g[g.split == sp]
        for i, m in enumerate(METHODS):
            v = sub[sub.method == m]["test.auc"].values
            y = len(METHODS) - 1 - i
            c = T["series"][i]
            ax.plot([v.min(), v.max()], [y, y], color=c, lw=2, alpha=0.35,
                    solid_capstyle="round", zorder=1)
            ax.scatter(v, [y] * len(v), s=26, color=c, zorder=3,
                       edgecolors=T["surface"], linewidths=1.2)
            ax.scatter([v.mean()], [y], s=110, marker="|", color=c, zorder=4, linewidths=2.2)
            ax.text(1.005, y, f"{v.mean():.3f}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.5, color=T["ink2"],
                    fontfamily="monospace")
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([LABEL[m] for m in METHODS[::-1]], fontsize=8.5, color=T["ink2"])
        ax.set_xlim(0.79, 1.02)
        ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax.xaxis.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
        despine(ax)
        bad = sp == "2"
        ax.set_title(SPLIT_LABEL[sp], color=T["critical"] if bad else T["ink"],
                     fontsize=9.5, pad=8, loc="left")
        if bad:
            ax.text(0.5, -0.16, "confounded — not valid evidence",
                    transform=ax.transAxes, ha="center", fontsize=8.5,
                    color=T["critical"], style="italic")
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_xlabel("Test ROC-AUC", color=T["ink2"])
    top = title(fig, "Generalization: every method, every split, all 3 seeds shown",
          "Dot = one seed · bar = mean of 3 · methods separated by well under the seed spread", T=T)
    fig.tight_layout(rect=[0, 0.02, 1, top])
    fig.savefig(out / "fig1_generalization.png")
    plt.close(fig)


def fig_paired_delta(runs, T, out):
    """The load-bearing figure: paired vs ERM, 95% CI, does it clear zero?"""
    g = runs[runs.group == "method"]
    base = g[g.method == "erm"].set_index(["split", "seed"])["test.auc"]
    rows = []
    for m in ["irm", "groupdro", "ours", "ours_bg"]:
        sub = g[g.method == m].set_index(["split", "seed"])["test.auc"]
        for sp in ["indist", "0", "1"]:
            d = (sub - base).xs(sp, level="split").dropna().values
            ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(method=m, split=sp, mean=d.mean(), ci=ci, vals=d))
        d = (sub - base).loc[["0", "1"]].dropna().values
        ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rows.append(dict(method=m, split="pooled", mean=d.mean(), ci=ci, vals=d))
    df = pd.DataFrame(rows)

    order = [(m, s) for m in ["irm", "groupdro", "ours", "ours_bg"]
             for s in ["indist", "0", "1", "pooled"]]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.axvline(0, color=T["axis"], lw=1.2, zorder=1)
    ax.axvspan(-0.005, 0.005, color=T["band"], alpha=0.55, zorder=0, lw=0)
    for i, (m, sp) in enumerate(order):
        r = df[(df.method == m) & (df.split == sp)].iloc[0]
        y = len(order) - 1 - i
        c = T["series"][METHODS.index(m)]
        hero = sp == "pooled"
        ax.plot([r["mean"] - r["ci"], r["mean"] + r["ci"]], [y, y], color=c,
                lw=2.4 if hero else 1.6, alpha=0.9, solid_capstyle="round", zorder=2)
        ax.scatter(r["vals"], [y] * len(r["vals"]), s=14, color=c, alpha=0.4, zorder=3)
        ax.scatter([r["mean"]], [y], s=64 if hero else 40, color=c, zorder=4,
                   edgecolors=T["surface"], linewidths=1.2)
        ax.text(0.076, y, f"{r['mean']:+.3f}", va="center", ha="right",
                fontsize=8.5, color=T["ink2"], fontfamily="monospace")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [f"{LABEL[m]}  ·  {'d0+d1 pooled' if s == 'pooled' else SPLIT_LABEL[s].split('  ')[0]}"
         for m, s in order][::-1], fontsize=8.5, color=T["ink2"])
    for i, (m, sp) in enumerate(order):
        if sp == "pooled":
            ax.get_yticklabels()[len(order) - 1 - i].set_color(T["ink"])
            ax.get_yticklabels()[len(order) - 1 - i].set_fontweight("600")
    ax.set_xlim(-0.062, 0.08)
    ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06])
    ax.set_xlabel("Paired AUC delta vs ERM  (same seed, same split)  ·  95% CI", color=T["ink2"])
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    ax.text(0.0, len(order) - 0.35, " every interval crosses zero", fontsize=8.5,
            color=T["critical"], style="italic", ha="left")
    top = title(fig, "No method beats ERM once seeds are paired",
          "Faint dots = individual seed deltas · pooled row = domains 0+1 (n=6), "
          "excludes confounded domain 2", T=T)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out / "fig2_paired_delta.png")
    plt.close(fig)


def fig_counterfactual(cf, T, out):
    ours = cf["results"]["ours"]
    base = cf["results"]["baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for ax, metric, lab in zip(
        axes, ["pred_swing", "flip_rate"],
        ["Prediction swing under intervention", "Class flip rate under intervention"],
    ):
        x = np.arange(2)
        w = 0.34
        o = [ours["illuminant"][metric], ours["melanin"][metric]]
        b = [base["illuminant"][metric], base["melanin"][metric]]
        rb = ax.bar(x - w / 2 - 0.011, b, w, label="ERM baseline", color=T["muted"],
                    zorder=2, linewidth=0)
        ro = ax.bar(x + w / 2 + 0.011, o, w, label="Ours (BiliGrad)", color=T["series"][4],
                    zorder=2, linewidth=0)
        for rects in (rb, ro):
            for r in rects:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.002,
                        f"{r.get_height():.3f}", ha="center", va="bottom",
                        fontsize=8.5, color=T["ink2"], fontfamily="monospace")
        ax.set_xticks(x)
        ax.set_xticklabels(["Illuminant\n(raw b* moves 12.5)", "Melanin\n(raw b* moves 1.7)"],
                           fontsize=9, color=T["ink2"])
        ax.set_title(lab, fontsize=9.5, loc="left", pad=8, color=T["ink"])
        ax.yaxis.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(o + b) * 1.28)
        despine(ax)
    # Grouped bars sit at x -/+ (w/2 + gap); annotate the mark, not the group centre.
    axes[0].annotate("5.5x lower than ERM\n= the result that holds", xy=(0.19, 0.017),
                     xytext=(0.52, 0.052), fontsize=8.5, color=T["good"], ha="left",
                     arrowprops=dict(arrowstyle="->", color=T["good"], lw=1.2,
                                     connectionstyle="arc3,rad=-0.2"))
    axes[1].annotate("no win here", xy=(1.19, 0.077), xytext=(0.72, 0.094),
                     fontsize=8.5, color=T["serious"], ha="left",
                     arrowprops=dict(arrowstyle="->", color=T["serious"], lw=1.2,
                                     connectionstyle="arc3,rad=0.2"))
    axes[0].legend(frameon=False, fontsize=9, loc="upper left",
                   labelcolor=T["ink2"])
    top = title(fig, "Counterfactual invariance: illuminant is won, melanin is not",
          f"n={cf['n']} test images · lower is better · physically-plausible interventions "
          "that hold true bilirubin fixed", T=T)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out / "fig3_counterfactual.png")
    plt.close(fig)


def fig_ablation(abl, T, out):
    a = abl.set_index("method").loc[
        ["ours_full", "ours_no_scin", "ours_no_disent", "ours_no_mixstyle",
         "ours_no_causal", "ours_no_lora"]]
    names = ["Full method", "− SCIN", "− disentangle", "− MixStyle", "− causal adv.", "− LoRA"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    y = np.arange(len(a))[::-1]
    vals = a["test.auc"].values
    cols = [T["critical"] if v < 0.6 else T["series"][0] for v in vals]
    ax.barh(y, vals, height=0.62, color=cols, zorder=2, linewidth=0)
    ax.axvline(vals[0], color=T["axis"], lw=1, ls=(0, (4, 3)), zorder=3)
    for yy, v in zip(y, vals):
        if v < 0.6:  # collapsed run: label inside the bar, note takes the free space
            ax.text(v - 0.008, yy, f"{v:.3f}", va="center", ha="right", fontsize=8.5,
                    color=T["surface"], fontfamily="monospace")
        else:
            ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=8.5,
                    color=T["ink2"], fontfamily="monospace")
    ax.text(vals[0], len(a) - 0.35, " full method", fontsize=8, color=T["muted"], ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=T["ink2"])
    ax.set_xlim(0.45, 1.06)
    ax.set_xlabel("Test ROC-AUC  ·  single seed (42)", color=T["ink2"])
    ax.set_title("Removing components barely moves AUC", fontsize=9.5, loc="left", pad=8,
                 color=T["ink"])
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    # The LoRA row's free space is the only gap in this panel — no arrow needed there.
    ax.text(0.515, 0, "collapsed to a constant prediction —\nLR mismatch, not an architecture result",
            fontsize=7.5, color=T["critical"], va="center", ha="left")

    ax = axes[1]
    rep = a["fair.bacc_gap"].values
    n_worst = a["fair.worst_stratum_n"].values
    ax.barh(y, rep, height=0.62, color=T["serious"], zorder=2, linewidth=0, alpha=0.9)
    for yy, v, n in zip(y, rep, n_worst):
        ax.text(v + 0.012, yy, f"{v:.3f}   set by a stratum of n={int(n)}",
                va="center", fontsize=8, color=T["ink2"], fontfamily="monospace")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("Reported fairness gap (bacc)", color=T["ink2"])
    ax.set_title("…and the fairness column is an artifact", fontsize=9.5, loc="left", pad=8,
                 color=T["critical"])
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    top = title(fig, "Ablations: directional only — single seed, and one metric is broken",
          "Do not cite any delta here. The fairness gaps are set by strata of 2–10 images.", T=T)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out / "fig4_ablation.png")
    plt.close(fig)


def fig_fairness_instability(strata, T, out):
    """Same 119 images, re-stratified by every run. The instrument moves."""
    s = strata[strata.split == "indist"]
    order = ["very_light", "light", "intermediate", "tan", "brown", "dark"]
    p = s.pivot_table(index=["method", "seed"], columns="stratum", values="n", fill_value=0)
    p = p.reindex(columns=order, fill_value=0)
    p = p.loc[[(m, sd) for m in METHODS for sd in [42, 1, 2] if (m, sd) in p.index]]
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    y = np.arange(len(p))[::-1]
    left = np.zeros(len(p))
    for st, c in zip(order, ramp):
        v = p[st].values
        ax.barh(y, v, left=left, height=0.62, color=c, label=st.replace("_", " "),
                zorder=2, linewidth=1.1, edgecolor=T["surface"])
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL[m]}  seed {sd}" for m, sd in p.index], fontsize=8.5,
                       color=T["ink2"])
    ax.set_xlabel("Test images assigned to each ITA skin-tone stratum  (always the same 119 images)",
                  color=T["ink2"])
    ax.set_xlim(0, 119)
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), labelcolor=T["ink2"])
    for m in ["ours"]:
        ys = [len(p) - 1 - i for i, (mm, _) in enumerate(p.index) if mm == m]
        ax.plot([121.5, 121.5], [min(ys) - 0.35, max(ys) + 0.35], color=T["critical"],
                lw=2, clip_on=False, solid_capstyle="round", zorder=5)
        ax.text(124.5, np.mean(ys), "same model,\nsame 119 images,\n3 seeds —\nstrata move anyway",
                fontsize=8, color=T["critical"], va="center", clip_on=False)
    top = title(fig, "The fairness metric measures the model, not the babies",
          "ITA is read off each run's own white-balanced image under its own attention — "
          "so the strata move per seed", T=T)
    fig.tight_layout(rect=[0, 0, 0.93, top])
    fig.savefig(out / "fig5_fairness_instability.png")
    plt.close(fig)


def fig_domain_confound(manifest, runs, T, out):
    m = manifest
    dom = m.groupby("domain").agg(n=("label", "size"), pos=("label", "sum")).reset_index()
    dom["rate"] = dom.pos / dom.n
    g = runs[runs.group == "method"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    x = np.arange(len(dom))
    bars = ax.bar(x, dom.rate, width=0.5, zorder=2, linewidth=0,
                  color=[T["critical"] if d == 2 else T["series"][0] for d in dom.domain])
    for r, n, p in zip(bars, dom.n, dom.pos):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                f"{r.get_height():.2f}\n{int(p)}/{int(n)}", ha="center", va="bottom",
                fontsize=8.5, color=T["ink2"], fontfamily="monospace")
    ax.set_xticks(x)
    ax.set_xticklabels([f"domain {int(d)}" for d in dom.domain], fontsize=9, color=T["ink2"])
    ax.set_ylabel("Jaundice positive rate", color=T["ink2"])
    ax.set_ylim(0, 0.52)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Pseudo-domains carry label shift", fontsize=9.5, loc="left", pad=8,
                 color=T["ink"])
    ax.annotate("spread 0.28\n= the domain boundary\npartly IS the label",
                xy=(2, 0.40), xytext=(0.75, 0.44), fontsize=8, color=T["critical"],
                arrowprops=dict(arrowstyle="->", color=T["critical"], lw=1.1))

    ax = axes[1]
    for i, m_ in enumerate(METHODS):
        v = [g[(g.method == m_) & (g.split == sp)]["test.auc"].mean() for sp in ["0", "1", "2"]]
        ax.plot([0, 1, 2], v, marker="o", ms=5, lw=2, color=T["series"][i], label=LABEL[m_],
                zorder=2)
    ax.axhspan(0.97, 1.005, color=T["critical"], alpha=0.09, zorder=0, lw=0)
    ax.text(0.04, 0.988, "ceiling — nobody can fail this test", fontsize=8,
            color=T["critical"], ha="left", va="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["domain 0", "domain 1", "domain 2"], fontsize=9, color=T["ink2"])
    ax.set_ylabel("Held-out AUC (mean of 3 seeds)", color=T["ink2"])
    ax.set_ylim(0.79, 1.01)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("…so the 'hardest' domain scores highest, for everyone", fontsize=9.5,
                 loc="left", pad=8, color=T["ink"])
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=T["ink2"])
    top = title(fig, "Why domain-2 LOCO numbers cannot be cited",
          "Plain ERM hits .998 on the held-out domain — a generalization test nobody can fail "
          "is not measuring generalization", T=T)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out / "fig6_domain_confound.png")
    plt.close(fig)


def fig_power(runs, T, out):
    g = runs[runs.group == "method"]
    base = g[g.method == "erm"].set_index(["split", "seed"])["test.auc"]

    def power_paired(dz, n, alpha=0.05):
        df = n - 1
        ncp = dz * np.sqrt(n)
        crit = stats.t.ppf(1 - alpha / 2, df)
        return 1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)

    def seeds_for(dz, power=0.8):
        for n in range(3, 400):
            if power_paired(abs(dz), n) >= power:
                return n
        return 400

    rows = []
    for m in ["ours", "ours_bg"]:
        sub = g[g.method == m].set_index(["split", "seed"])["test.auc"]
        for sp in ["indist", "0", "1"]:
            d = (sub - base).xs(sp, level="split").dropna().values
            dz = d.mean() / d.std(ddof=1)
            rows.append(dict(method=m, split=sp, dz=dz, need=seeds_for(dz),
                             power3=power_paired(abs(dz), 3), delta=d.mean()))
    df = pd.DataFrame(rows).sort_values("need")

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    y = np.arange(len(df))[::-1]
    cols = [T["good"] if n <= 12 else (T["warning"] if n <= 30 else T["critical"])
            for n in df.need]
    ax.barh(y, df.need, height=0.6, color=cols, zorder=2, linewidth=0)
    ax.axvline(3, color=T["ink"], lw=1.4, zorder=3)
    ax.text(3.4, len(df) - 0.4, "what we ran (3 seeds)", fontsize=8.5, color=T["ink"])
    for yy, r in zip(y, df.itertuples()):
        ax.text(r.need + 1.5, yy, f"{r.need} seeds   (delta {r.delta:+.4f}, power now {r.power3:.0%})",
                va="center", fontsize=8.5, color=T["ink2"], fontfamily="monospace")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL[r.method]}  ·  {SPLIT_LABEL[r.split].split('  ')[0]}"
                        for r in df.itertuples()], fontsize=8.5, color=T["ink2"])
    ax.set_xlim(0, 128)
    ax.set_xlabel("Seeds needed for 80% power at the effect size we actually observed",
                  color=T["ink2"])
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    top = title(fig, "What it would take to prove the physics claim",
          "Green = affordable. The cheapest real win is one cell, not another whole grid.", T=T)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out / "fig7_power.png")
    plt.close(fig)


def fig_operating(runs, T, out):
    g = runs[(runs.group == "method") & (runs.split == "indist")]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    marks = {"test": "o", "test_youden": "s", "test_screening": "D"}
    names = {"test": "threshold 0.5", "test_youden": "Youden J", "test_screening": "screening (recall-first)"}
    for i, m in enumerate(METHODS):
        sub = g[g.method == m]
        for blk, mk in marks.items():
            ax.scatter(sub[f"{blk}.specificity"], sub[f"{blk}.sensitivity"], marker=mk,
                       s=34, color=T["series"][i], alpha=0.85, zorder=3,
                       edgecolors=T["surface"], linewidths=0.9)
    ax.axhline(0.90, color=T["good"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.415, 0.905, "screening target sensitivity 0.90", fontsize=8, color=T["good"])
    ax.set_xlabel("Specificity", color=T["ink2"])
    ax.set_ylabel("Sensitivity", color=T["ink2"])
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.03)
    ax.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    h1 = [plt.Line2D([], [], marker="o", ls="", color=T["series"][i], label=LABEL[m])
          for i, m in enumerate(METHODS)]
    h2 = [plt.Line2D([], [], marker=mk, ls="", color=T["muted"], label=names[b])
          for b, mk in marks.items()]
    leg1 = ax.legend(handles=h1, frameon=False, fontsize=8, loc="lower left",
                     labelcolor=T["ink2"])
    ax.add_artist(leg1)
    ax.legend(handles=h2, frameon=False, fontsize=8, loc="lower center",
              labelcolor=T["ink2"], title="operating point",
              title_fontsize=8)
    top = title(fig, "Operating points: the screening threshold buys recall with specificity",
          "In-distribution test, every method x seed. Colour = method, shape = threshold rule.", T=T)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out / "fig8_operating_points.png")
    plt.close(fig)


def fig_curves(hist, T, out):
    h = hist[hist.split == "indist"]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for i, m in enumerate(METHODS):
        sub = h[h.method == m]
        for _, run in sub.groupby("run"):
            run = run.sort_values("epoch")
            ax.plot(run.epoch, run.val_auc, color=T["series"][i], lw=1.4, alpha=0.75, zorder=2)
    ax.set_xlabel("Epoch", color=T["ink2"])
    ax.set_ylabel("Validation AUC", color=T["ink2"])
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(handles=[plt.Line2D([], [], color=T["series"][i], lw=2, label=LABEL[m])
                       for i, m in enumerate(METHODS)],
              frameon=False, fontsize=8, loc="lower right", labelcolor=T["ink2"], ncol=2)
    top = title(fig, "Training: all methods converge fast, then early-stop",
          "In-distribution runs, 3 seeds each. Runs end at the patience-15 early stop.", T=T)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out / "fig9_training_curves.png")
    plt.close(fig)


def main() -> None:
    runs = pd.read_csv(TAB / "runs.csv", dtype={"split": str})
    strata = pd.read_csv(TAB / "strata.csv", dtype={"split": str})
    hist = pd.read_csv(TAB / "history.csv", dtype={"split": str})
    abl = pd.read_csv(TAB / "ablation.csv")
    manifest = pd.read_csv(HERE.parent / "experiments" / "_provenance" / "manifest.csv")
    import json
    cf = json.loads((HERE.parent / "experiments" / "cf-20260715-174811" /
                     "counterfactual.json").read_text())

    for theme, T in THEMES.items():
        out = HERE / "figures" / theme
        out.mkdir(parents=True, exist_ok=True)
        style(T)
        fig_generalization(runs, T, out)
        fig_paired_delta(runs, T, out)
        fig_counterfactual(cf, T, out)
        fig_ablation(abl, T, out)
        fig_fairness_instability(strata, T, out)
        fig_domain_confound(manifest, runs, T, out)
        fig_power(runs, T, out)
        fig_operating(runs, T, out)
        fig_curves(hist, T, out)
        print(f"{theme}: 9 figures -> {out}")


if __name__ == "__main__":
    main()
