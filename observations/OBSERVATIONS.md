# Observations — first full cloud training grid

Analysis of 68 cloud runs (Kaggle T4, commit `da46344`, 2026-07-15/16).
Everything here is derived from `experiments/*/metrics.json` by `aggregate.py` + `plots.py`;
tables in `tables/`, figures in `figures/{light,dark}/`. Nothing is hand-entered.

---

## Headline

**One result holds, and it is the physics one.** Illuminant counterfactual invariance is a
clean 5.5x win over ERM and is publication-grade as-is.

**The classification grid proves nothing yet.** Not one method × split cell beats ERM at
p<0.05 — every 95% CI crosses zero. The "physics edges out DG baselines by 1–2 AUC pts"
reading in progress.md §9.1 is real in direction but sits inside seed noise. At 3 seeds the
grid has **6–23% statistical power**; it was never capable of resolving an effect this size.

**Two of the five open items in §9.5 are chasing measurement bugs, not model behaviour.**
The fairness metric and the domain-2 split are both broken instruments. Fix the instruments
before drawing conclusions from what they read.

---

## 1. What holds — illuminant invariance (fig3)

| intervention | ours pred_swing | ERM pred_swing | ours flip_rate | ERM flip_rate | raw b* moves |
|---|---|---|---|---|---|
| illuminant | **0.013** | 0.073 | **0.013** | 0.076 | 12.49 |
| melanin | 0.087 | 0.082 | 0.075 | 0.069 | 1.71 |

The intervention genuinely moves the input (raw skin b* swings 12.5 units) and the model
barely moves — 5.5x less prediction swing, 5.8x fewer class flips than ERM. That is SCIN
doing real work, not luck. **Draft this section now.**

The melanin arm is a genuine non-result (slight regression vs ERM). It is not a metric
artifact — the counterfactual is a per-image intervention and doesn't touch the broken
stratification below. So the causal/fairness half of the thesis is still unlanded, but the
*evidence* for that is this figure alone, not the ablation column that appears to agree
with it (see §3).

## 2. What does not clear noise — the AUC grid (fig1, fig2, fig7)

Paired per-seed deltas vs ERM (same seed, same split — removes seed variance):

| method | in-dist | domain 0 | domain 1 | d0+d1 pooled (n=6) |
|---|---|---|---|---|
| IRM | −0.002 | +0.006 | −0.011 | −0.002 (p=0.70) |
| GroupDRO | −0.007 | +0.002 | +0.001 | +0.002 (p=0.78) |
| ours (disentangle) | +0.004 | +0.015 | +0.008 | **+0.012 (p=0.13, wins 5/6)** |
| ours_bg (BiliGrad) | +0.005 | +0.006 | +0.009 | +0.008 (p=0.20, wins 4/6) |

`ours` on domain 0 is the only cell that beats ERM on all 3 seeds (+0.015, p=0.175).
Pooling the two valid domains is the most favourable honest framing: +0.012, still p=0.13.

Seeds needed for 80% power **at the effect size actually observed**:

| cell | observed delta | power at 3 seeds | seeds needed |
|---|---|---|---|
| ours · domain 0 | +0.0152 | 23% | **~8** |
| ours_bg · domain 1 | +0.0094 | 13% | ~15 |
| ours · domain 1 | +0.0082 | 7% | ~52 |
| ours_bg · domain 0 | +0.0060 | 7% | ~59 |
| ours · in-dist | +0.0036 | 7% | ~72 |
| ours_bg · in-dist | +0.0048 | 6% | ~95 |

This reframes "rerun with seeds": **another 3-seed grid changes nothing.** The affordable
move is depth on one cell (`ours` vs `erm`, domain 0, ~8–10 seeds), not breadth.

## 3. What is a measurement bug — the fairness metric (fig4, fig5)

Two independent defects, both fatal to any fairness claim from this grid:

**(a) The gap is set by strata of 1–10 images.** `ours_full`'s headline 0.500 gap is one
stratum of n=2 (brown, 2 positives, 1 caught → bacc 0.500). `no_causal`'s "better" 0.100 is
likewise set by strata of n=1–2. 14 of 68 runs have their reported gap set by a stratum of
≤5 images. Recomputed over strata with n≥20, `ours_full`'s gap is **0.013**, not 0.500.

**(b) The strata are model-dependent.** ITA is measured on each run's *own* SCIN-white-balanced
image under its *own* attention. So the same 119 test images get re-stratified by every run —
`ours` seed 42 calls 73 of them "very_light"; `ours` seed 1 calls 18. Same architecture, same
images, different seed. The protected attribute is a function of the model being audited.

Consequence: **§9.3's "removing the causal adversary improves fairness" is not a finding.**
It is one image in an n=2 bucket, measured on a stratification that run invented. The second
"independent signal" for melanin-adversary underperformance evaporates; only the counterfactual
(§1) survives.

Fix: compute ITA once per test image from a fixed, model-independent reference (fixed
gray-world WB + fixed skin mask), freeze it into `manifest.csv`, and report only strata with
n≥20 (or use a continuous ITA regression instead of bins at this sample size).

## 4. What is confounded — domain-2 LOCO (fig6)

Pseudo-domain positive rates: d0 = 0.32 (145/454), d1 = 0.11 (27/235), **d2 = 0.39 (28/71)**.
Spread 0.28 — the cluster boundary partly *is* the label. Every method including plain ERM
scores .979–.998 held out on d2. A generalization test nobody can fail is not measuring
generalization. Confirms §9.5 item 4: do not cite d2. The valid LOCO evidence is d0 and d1 only.

Note: progress.md §2 still lists the pre-fix cluster sizes (d0=449/d1=246/d2=65). The manifest
actually shipped d0=454/d1=235/d2=71 — §2 predates the train-only KMeans fit. §9.1 is correct.

## 5. What is uninterpretable — `no_lora`

Collapsed to constant-positive (AUC 0.500, threshold pinned to 0.0). A 22M-param full ViT
fine-tune at the LoRA-tuned LR on 528 images. Not evidence LoRA is architecturally required.
Its 0.500 "fairness gap" is the same n=2 artifact as above, compounding the meaninglessness.

## 6. Secondary — retrieval, operating points, training (fig8, fig9)

- Retrieval explanations: 0.900 label agreement on 20 queries — but the classifier scores
  0.900 on the same 20. The retrieval head is not adding independent signal at this n; keep it
  as an interpretability affordance, don't claim it as corroboration.
- Screening threshold behaves as designed: `ours_bg` reaches 0.914 sensitivity at 0.946
  specificity in-dist. IRM buys 0.988 sensitivity at 0.757 specificity — worth noting as the
  recall-first extreme.
- Physics runs early-stop sooner (28–29 epochs vs 35–36 for the DG baselines). Minor, but
  consistent with the physics feature giving the head an easier target.

---

## Revised next steps

Ordered by what unblocks the paper fastest. Diverges from the original plan where the
analysis changed the premise — noted inline.

1. **Draft the illuminant-invariance section now.** Unchanged from your #5. It is the one
   publication-grade result in the grid.

2. **Fix the fairness instrument before anything else touches fairness.** *(New — this
   displaces part of your #1.)* Freeze a model-independent ITA into the manifest, add an
   n≥20 floor, then re-report. Cheap (no retraining — recompute from stored predictions if
   they're kept, otherwise one eval pass). Until this lands, every fairness number in the
   repo is uninterpretable, including the ones that currently look favourable.

3. **Depth, not breadth, on the seeds.** *(Revises your #2.)* Run `ours` vs `erm` on domain 0
   at 8–10 seeds. That single cell is powered at the observed effect size; the full grid at 3
   seeds is not, and rerunning it at 3 seeds again yields another inconclusive table for the
   same GPU-hours. If it holds at n=8, you have a citable generalization claim on one domain;
   if it doesn't, you've learned that cheaply.

4. **Investigate the melanin adversary on the counterfactual only.** *(Narrows your #1.)*
   The ablation-side evidence was an artifact, so `lambda_adv` / ITA-proxy tuning should be
   scored against `cf` melanin pred_swing, not against `fair_gap_bacc`. The
   disentangle+BiliGrad combination question stands and is worth one run — `net.py`'s
   precedence makes them mutually exclusive today, which is an untested assumption, not a
   design decision.

5. **Rerun `no_lora` at a lower LR.** Unchanged from your #3, but low priority — it answers
   a reviewer question, not a thesis question.

6. **Domain-2: agreed, do not cite.** Unchanged from your #4. Worth stating explicitly in the
   paper's limitations that pseudo-domains carry label shift, rather than silently dropping
   d2 — a reviewer who recomputes will notice the omission.

**On the external-hospital dataset:** items 3 and 6 both bottom out in the same place — 760
images from a single source cannot support a generalization claim no matter how many seeds
are burned. The seed work is worth doing because it's cheap and decides whether the effect is
real; the external set is what makes it publishable.

---

## Reproduce

```bash
source .venv/bin/activate
python observations/aggregate.py    # experiments/*/metrics.json -> tables/*.csv
python observations/plots.py        # tables/*.csv -> figures/{light,dark}/*.png
open observations/dashboard.html    # self-contained, figures embedded
```

| File | Contents |
|---|---|
| `tables/runs.csv` | all 68 runs × every metric, operating point, fairness summary |
| `tables/strata.csv` | per-ITA-stratum sens/spec/AUC/n, long form |
| `tables/history.csv` | per-epoch training curves |
| `tables/summary_{auc,bacc,screen_sens}.csv` | method × split, mean/std/min/max over seeds |
| `tables/deltas_vs_erm.csv` | paired per-seed deltas vs ERM |
| `tables/ablation.csv` | ablations + the recomputed n≥20 fairness gap |
