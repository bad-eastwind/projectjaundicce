"""Render observations/classification_report.png — a clean slide-ready metrics card.

    python observations/report_card.py

Numbers are the canonical eval of the shipped checkpoint (ourbg_indist metrics.json).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent

# theme (light — this goes into slides)
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; SURF = "#fcfcfb"
RULE = "#e1e0d9"; ACC = "#2a78d6"; GOOD = "#0ca30c"; BAND = "#EAF2F8"

# Numbers = shipped checkpoint best.pt evaluated through the official pipeline (NOT metrics.json,
# which reports the final-epoch model, a different set of weights — see progress.md §12).
HEADLINE = [  # (label, default t=0.50, high-sensitivity t=0.20)
    ("Accuracy", "93.3%", "93.3%"),
    ("Balanced accuracy", "93.0%", "94.3%"),
    ("Sensitivity (recall)", "92.6%", "96.3%"),
    ("Specificity", "93.5%", "92.4%"),
    ("Precision (PPV)", "80.6%", "78.8%"),
    ("F1 score", "0.862", "0.867"),
    ("ROC-AUC", "0.969", "0.969"),
]
KPI = [("ROC-AUC", "0.969"), ("Accuracy", "93.3%"), ("Sensitivity", "92.6%"), ("Specificity", "93.5%")]
CM = [[86, 6], [2, 25]]  # rows actual normal/jaundice; cols pred normal/jaundice


def rbox(ax, x, y, w, h, fc, ec=RULE, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=1))


def main():
    fig = plt.figure(figsize=(11.5, 8.4), facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.045, 0.955, "Jaundice screening classifier — performance",
            fontsize=19, fontweight="bold", color=INK, va="top")
    ax.text(0.045, 0.917, "BiliGrad (DINOv2 + LoRA + SCIN)  ·  held-out test split, 119 images  "
            "(27 jaundice / 92 normal)", fontsize=10.5, color=INK2, va="top")

    # KPI strip
    x0, w, gap = 0.045, 0.212, 0.017
    for i, (k, v) in enumerate(KPI):
        x = x0 + i * (w + gap)
        rbox(ax, x, 0.775, w, 0.098, BAND)
        ax.text(x + 0.018, 0.855, k.upper(), fontsize=8, color=MUTED, va="top",
                fontfamily="sans-serif")
        ax.text(x + 0.018, 0.828, v, fontsize=20, fontweight="bold", color=ACC, va="top")

    # headline table
    ty = 0.71
    ax.text(0.045, ty, "Test-set metrics", fontsize=12, fontweight="bold", color=INK)
    cols_x = [0.045, 0.52, 0.75]
    ax.text(cols_x[1], ty, "Default (t=0.50)", fontsize=9.5, color=INK2, fontweight="bold")
    ax.text(cols_x[2], ty, "Screening (t=0.20)", fontsize=9.5, color=INK2, fontweight="bold")
    ax.plot([0.045, 0.955], [ty - 0.018, ty - 0.018], color=RULE, lw=1)
    row_h = 0.049
    bold = {"Accuracy", "Sensitivity (recall)", "Specificity", "ROC-AUC"}
    for i, (lab, d, s) in enumerate(HEADLINE):
        yy = ty - 0.045 - i * row_h
        if lab in bold:
            rbox(ax, 0.04, yy - 0.016, 0.915, row_h - 0.006, "#F4F8FC", ec="none")
        w_ = "bold" if lab in bold else "normal"
        ax.text(cols_x[0], yy, lab, fontsize=10, color=INK, fontweight=w_, va="center")
        ax.text(cols_x[1], yy, d, fontsize=10.5, color=INK, fontweight=w_, va="center",
                fontfamily="monospace")
        ax.text(cols_x[2], yy, s, fontsize=10.5, color=INK2, va="center", fontfamily="monospace")

    # confusion matrix (bottom-left)
    cx, cy, cs = 0.06, 0.045, 0.11
    ax.text(cx, 0.30, "Confusion matrix  (t=0.50)", fontsize=11, fontweight="bold", color=INK)
    labels = [["86", "6"], ["2", "25"]]
    fills = [["#DBE9F8", "#FBE9E7"], ["#FBE9E7", "#DBE9F8"]]
    for r in range(2):
        for c in range(2):
            x = cx + 0.13 + c * cs; y = 0.16 - r * cs
            rbox(ax, x, y, cs - 0.012, cs - 0.012, fills[r][c], ec=RULE)
            ax.text(x + (cs - 0.012) / 2, y + (cs - 0.012) / 2, labels[r][c],
                    ha="center", va="center", fontsize=17, fontweight="bold",
                    color=INK)
    ax.text(cx + 0.13 + cs - 0.006, 0.275, "pred\nnormal", ha="center", fontsize=7.5, color=MUTED)
    ax.text(cx + 0.13 + 2 * cs - 0.006, 0.275, "pred\njaundice", ha="center", fontsize=7.5, color=MUTED)
    ax.text(cx + 0.12, 0.16 + (cs - 0.012) / 2, "actual\nnormal", ha="right", va="center",
            fontsize=7.5, color=MUTED)
    ax.text(cx + 0.12, 0.16 - cs + (cs - 0.012) / 2, "actual\njaundice", ha="right", va="center",
            fontsize=7.5, color=MUTED)

    # splits + model (bottom-right)
    rx = 0.55
    ax.text(rx, 0.30, "Dataset & splits", fontsize=11, fontweight="bold", color=INK)
    splits = [("Train", "152", "376", "528"), ("Validation", "21", "92", "113"),
              ("Test", "27", "92", "119"), ("Total", "200", "560", "760")]
    ax.text(rx, 0.262, "split", fontsize=8.5, color=MUTED)
    ax.text(rx + 0.20, 0.262, "jaund.", fontsize=8.5, color=MUTED, ha="right")
    ax.text(rx + 0.29, 0.262, "normal", fontsize=8.5, color=MUTED, ha="right")
    ax.text(rx + 0.38, 0.262, "total", fontsize=8.5, color=MUTED, ha="right")
    for i, (s, j, n, t) in enumerate(splits):
        yy = 0.232 - i * 0.032
        fw = "bold" if s in ("Test", "Total") else "normal"
        ax.text(rx, yy, s, fontsize=9, color=INK, fontweight=fw)
        for xx, val in zip([rx + 0.20, rx + 0.29, rx + 0.38], [j, n, t]):
            ax.text(xx, yy, val, fontsize=9, color=INK, ha="right", fontweight=fw,
                    fontfamily="monospace")
    ax.text(rx, 0.092, "Stratified 70/15/15, deterministic. Thresholds tuned on validation only;",
            fontsize=8, color=MUTED)
    ax.text(rx, 0.070, "test untouched until final eval. Single-source, in-the-wild NICU photos.",
            fontsize=8, color=MUTED)
    ax.text(rx, 0.044, "Backbone: DINOv2 ViT-S/14 frozen + LoRA (~2.2% trained). SCIN white",
            fontsize=8, color=MUTED)
    ax.text(rx, 0.022, "balance + attention MIL. Input 392px. AdamW cosine, best-by-val-AUC.",
            fontsize=8, color=MUTED)

    fig.savefig(HERE / "classification_report.png", dpi=170, facecolor=SURF, bbox_inches="tight")
    print(f"wrote {HERE/'classification_report.png'}")


if __name__ == "__main__":
    main()
