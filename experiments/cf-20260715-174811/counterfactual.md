# Counterfactual-recoloring invariance (64 test images)

Interventions change the nuisance, hold true bilirubin fixed. Lower pred_swing / flip / beta_cov = more causally robust. `raw_b*_swing` is the input-side reference that SHOULD move (proves the intervention is real).

## ours

| intervention | pred_swing | flip_rate | beta_cov | raw_b*_swing (ref) |
|---|---|---|---|---|
| illuminant | 0.013 | 0.013 | 0.198 | 12.490 |
| melanin | 0.087 | 0.075 | 0.362 | 1.709 |

## baseline

| intervention | pred_swing | flip_rate | beta_cov | raw_b*_swing (ref) |
|---|---|---|---|---|
| illuminant | 0.073 | 0.076 | nan | 12.490 |
| melanin | 0.082 | 0.069 | nan | 1.709 |
