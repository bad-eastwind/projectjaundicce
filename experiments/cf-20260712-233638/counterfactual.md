# Counterfactual-recoloring invariance (8 test images)

Interventions change the nuisance, hold true bilirubin fixed. Lower pred_swing / flip / beta_cov = more causally robust. `raw_b*_swing` is the input-side reference that SHOULD move (proves the intervention is real).

## ours

| intervention | pred_swing | flip_rate | beta_cov | raw_b*_swing (ref) |
|---|---|---|---|---|
| illuminant | 0.020 | 0.062 | 0.142 | 11.278 |
| melanin | 0.044 | 0.100 | 0.239 | 1.437 |

## baseline

| intervention | pred_swing | flip_rate | beta_cov | raw_b*_swing (ref) |
|---|---|---|---|---|
| illuminant | 0.030 | 0.000 | 0.771 | 11.278 |
| melanin | 0.123 | 0.225 | 0.352 | 1.437 |
