# Inference + explanation maps

Inference only — no training. Runs the two flagship checkpoints on the **119 held-out test
images** the cloud grid actually used (split taken from `experiments/_provenance/manifest.csv`,
paths remapped to the local `jaundicedataset3/` copy; the local `data/manifest.csv` has a
different, larger split that would leak train images and is deliberately not used).

```bash
PYTHONPATH=src python inferences/infer.py            # all 119
PYTHONPATH=src python inferences/infer.py --limit 8  # quick
```

## Models compared

| | checkpoint | SCIN | physics head | explanation channels |
|---|---|---|---|---|
| **ours_bg** | `ourbg_indist-20260715-173549` | on | BiliGrad | attention · Grad-CAM · b* field · axis-s |
| **ERM** | `erm_indist-20260715-172926` | off | none | attention · Grad-CAM only |

ERM here is our own ablation — the identical DINOv2+LoRA backbone with SCIN and the physics
head switched off. It is a controlled reference point, not an external published method.

## Results on the held-out test set (119 images, 27 jaundice)

| model | acc | sensitivity | specificity |
|---|---|---|---|
| **ours_bg** | 0.933 | 0.926 | **0.935** |
| ERM | 0.882 | 0.926 | 0.870 |

Both catch the same jaundice cases (equal sensitivity). ours_bg's edge is **specificity** — it
false-alarms less on normal babies (6 vs 12 false positives).

These are the **shipped `best.pt`** checkpoints evaluated here. They are ~0.03 lower than each
run's `metrics.json`, because `metrics.json` reports the *final-epoch* model rather than the saved
best-validation checkpoint (a train.py reporting bug — see progress.md §12). The numbers here are
the honest deployable ones.

## What the panels show

`panels/*.png` — six columns per model: input, SCIN white-balanced, MIL attention, Grad-CAM,
BiliGrad b* field, cephalocaudal axis s. `summary.png` — model agreement, physics-readout
separation, confusion matrix. `predictions.csv` — per-image probs/preds/β for both models.

**Three things a stakeholder should take away, two of them cautionary:**

1. **`normal_(1055)` — the lighting shortcut, caught live.** A warm-lit normal baby. ERM →
   0.998 jaundice (**wrong**), its Grad-CAM firing on the yellow-cast torso. ours_bg → 0.015
   normal (**correct**): SCIN white-balances the warm cast out (panel 2) and the readout no
   longer fires. This is the single clearest qualitative case for the physics half of the thesis.
   7 of the 10 model disagreements are this pattern — ERM false-positive, ours_bg correct.

2. **`summary.png` middle panel — the physics readout does NOT separate the classes on its own.**
   BiliGrad |β| overlaps heavily between normal (0–0.4) and jaundice (0–0.55). The backbone is
   doing the classification; the physics feature is a small concatenated add-on. This is honest,
   load-bearing evidence that BiliGrad's discriminative value is **unproven** on this data — it
   agrees with the analysis in `observations/OBSERVATIONS.md`.

3. **The `cephalocaudal axis s` column is a smooth gradient, not a head→toe orientation.**
   `head_anchor` is off (no pose net), so the axis is the principal axis of skin mass with an
   arbitrary per-image sign — which is why the classifier uses |β| and partial-R², not signed β.
   The map is shown faithfully rather than dressed up. Resolving polarity (pose/anchor) is the
   experiment that would make the "cephalocaudal" claim real.

## Caveats

- In-distribution test images (random split), not cross-hospital — same limitation as the whole
  grid. These panels illustrate behaviour; they are not a generalization claim.
- Explanation maps are the model's own decision path (attention/BiliGrad field), except Grad-CAM,
  which is the standard post-hoc approximation over backbone tokens — included because it is the
  one channel both models share.
