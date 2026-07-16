# Research track — roadmap to a top-venue paper

Separate from the product (deployable classifier). This folder holds the paper's theoretical
core. Thesis pivot after the first cloud grid (see `confidential/progress.md` §9–§11): the
citable contribution is **not** "higher AUC on 760 single-source images" (un-citable — a
color-stats shortcut already hits 0.98 AUC there) but a **physics identifiability** claim about
why bilirubin is recoverable from its spatial signature when it is not from marginal color.

## Status

| piece | state |
|---|---|
| **Synthetic identifiability proof** | **DONE — `synthetic_identifiability.py`, results below.** No training, no clinical data. |
| Attribution runs (split the 5.5× invariance: SCIN vs BiliGrad) | configs ready (`configs/attr_scin_only.yaml`, `configs/attr_no_scin.yaml`); needs ~2 cloud runs + counterfactual |
| Bilirubin-valued, multi-site validation | pending — Mendeley (Iran, has TcB/TSB) + external hospital set |

## Synthetic identifiability proof (`synthetic_identifiability.py`)

Simulate neonatal skin with **known** bilirubin (cephalocaudal gradient), melanin (skin tone),
and illuminant; render the yellow-axis (b\*) field via the standard additive log-reflectance
skin-optics model; run **BiliGrad's own WLS solver** (`CephalocaudalBiliField._wls_ss`, imported
directly) and compare against the marginal color-stats readout that drives the lighting shortcut.

**Results** (`identifiability_results.json`, figures in `figures/{light,dark}/`):

- **E1 — invariance.** With the true bilirubin gradient held fixed, BiliGrad's slope recovers the
  truth at **CV 0.4%**, correlation with melanin +0.04 and with illuminant −0.06 (i.e. flat —
  invariant). The marginal readout correlates **+0.64 with melanin and +0.73 with illuminant** —
  the confound, quantified.
- **E2 — identifiability boundary (the honest result).** BiliGrad's slope-recovery error stays
  ~0.002 while melanin is not collinear with the body axis, and **explodes to ~2.9 exactly as
  melanin becomes collinear** — the stated condition (bilirubin is identified iff it is the only
  s-correlated term after removing melanin and the constant). A reviewer-credible boundary, not a
  universal claim.
- **E3 — screening across skin tones.** Gradient readouts (BiliGrad and even a naive gradient)
  score AUC 1.0 in every skin-tone group; the color-stats shortcut degrades to 0.72 overall and is
  skin-tone-dependent (0.84 lightest → 0.73 darkest) — the fairness failure reproduced from first
  principles. BiliGrad's advantage *over the naive gradient* is not here (binary classification) —
  it is in E2's unbiased slope **estimation**, which is what the quantitative TcB/TSB task needs.

```bash
PYTHONPATH=src python research/synthetic_identifiability.py
```

## Why this is the lever

It converts the paper from "our number is higher" (dataset-bound, un-citable) to "we prove the
degeneracy that causes the shortcut, and prove our readout breaks it, and state exactly when it
cannot" (holds regardless of dataset size). Real clinical data then *corroborates* a proven
mechanism rather than *being* the whole argument — the shape IPMI / MICCAI reward.
