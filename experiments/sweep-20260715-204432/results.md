# Sweep results

| group | run | split | bal_acc | auc | sens | spec | f1 |
|---|---|---|---|---|---|---|---|
| ablation | ours_full | indist | 0.930 | 0.967 | 0.926 | 0.935 | 0.862 |
| ablation | ours_no_scin | indist | 0.954 | 0.977 | 0.963 | 0.946 | 0.897 |
| ablation | ours_no_disent | indist | 0.941 | 0.982 | 0.926 | 0.957 | 0.893 |
| ablation | ours_no_causal | indist | 0.919 | 0.968 | 0.926 | 0.913 | 0.833 |
| ablation | ours_no_mixstyle | indist | 0.936 | 0.971 | 0.926 | 0.946 | 0.877 |
| ablation | ours_no_lora | indist | 0.500 | 0.500 | 1.000 | 0.000 | 0.370 |

## AUC by run x split

| run | indist |
|---|---|
| ours_full | 0.967 |
| ours_no_scin | 0.977 |
| ours_no_disent | 0.982 |
| ours_no_causal | 0.968 |
| ours_no_mixstyle | 0.971 |
| ours_no_lora | 0.500 |
