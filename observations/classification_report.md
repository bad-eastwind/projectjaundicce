# Jaundice screening classifier — performance report

**Model:** BiliGrad (DINOv2 ViT-S/14 + LoRA + SCIN illuminant normalization + physics head)
**Checkpoint:** `ourbg_indist-20260715-173549/best.pt` · **Evaluation:** held-out test split (119 images)

Numbers are the **shipped checkpoint** (`best.pt`, best-by-validation-AUC) evaluated through the
official pipeline. (They differ from the run's `metrics.json`, which reports the final-epoch model
— a different set of weights that was never saved; see progress.md §12.)

## Test-set performance

| Metric | Default operating point | High-sensitivity (screening) |
|---|---|---|
| Threshold | 0.50 | 0.20 (val-tuned) |
| **Accuracy** | **93.3%** | 93.3% |
| Balanced accuracy | 93.0% | 94.3% |
| **Sensitivity (recall)** | **92.6%** | **96.3%** |
| **Specificity** | **93.5%** | 92.4% |
| Precision (PPV) | 80.6% | 78.8% |
| Neg. predictive value | 97.7% | 98.8% |
| F1 score | 0.862 | 0.867 |
| **ROC-AUC** | **0.969** | 0.969 |

AUC is threshold-independent (0.969 either way). The two columns are the same model read at two
thresholds: the default balances errors; the screening point trades a little specificity for
catching more jaundice (misses 1 of 27 instead of 2) — the right posture for a triage tool.

## Confusion matrix (default, threshold 0.50)

| | Predicted normal | Predicted jaundice |
|---|---|---|
| **Actual normal** (92) | 86 (TN) | 6 (FP) |
| **Actual jaundice** (27) | 2 (FN) | 25 (TP) |

## Reference: same backbone without our physics (ERM baseline)

| Model (best.pt) | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| **BiliGrad (shipped)** | 93.3% | 92.6% | **93.5%** | 0.969 |
| ERM (SCIN + physics off) | 88.2% | 92.6% | 87.0% | 0.971 |

Honest read: **threshold-free discrimination (AUC) is a tie in-distribution** (0.969 vs 0.971).
BiliGrad's visible edge is *specificity at the default threshold* (93.5% vs 87.0% — 6 false alarms
vs 12) — i.e. it is better-calibrated out of the box, fewer false positives without threshold
tuning. That is a real usability advantage but not a discrimination advantage. The differentiator
that would separate the two models is cross-hospital robustness — see note below.

---

## Dataset & splits

- **Source:** single neonatal-skin image set, in-the-wild NICU photos (1000×1000 JPG).
- **Size:** 760 images — 200 jaundice / 560 normal (26% positive).
- **Split:** deterministic **stratified 70 / 15 / 15**, class balance preserved across splits.

| Split | Jaundice | Normal | Total |
|---|---|---|---|
| Train | 152 | 376 | 528 |
| Validation | 21 | 92 | 113 |
| **Test** (reported above) | 27 | 92 | 119 |
| **Total** | 200 | 560 | 760 |

Thresholds are tuned on **validation only**; the test split is untouched until final evaluation
(no test leakage). Metrics above are on the held-out test split.

## Model & training

| | |
|---|---|
| Backbone | DINOv2 ViT-S/14 (self-supervised foundation model), frozen |
| Adaptation | LoRA adapters, r=8 — ~2.2% of weights trained (0.49M of 22.5M) |
| Physics front-end | SCIN self-calibrating white balance + gated-attention skin localization + BiliGrad bilirubin head |
| Input resolution | 392×392 |
| Optimizer | AdamW + cosine schedule, lr 3e-4, batch 32 |
| Selection | best-by-validation-AUC, early stopping |
| Inference cost | single forward pass; no calibration card, no extra hardware |

## Note on interpretation (for the next milestone)

These are **in-distribution** results — test images from the same source as training. They clear
the bar as a working screening model. The next validation gate, already scoped, is **external
multi-site testing** (independent hospitals / phones / lighting): that is what confirms field
robustness, and it is where the SCIN illuminant-normalization front-end is expected to hold its
specificity while a plain classifier degrades. Cross-site numbers are the ones to quote as
deployment performance.
