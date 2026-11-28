# Stage 1 survey-weighted sensitivity evaluation

Six frozen models evaluated on 7,004 retained temporal participants using day-one dietary weights WTDRD1PP. Design: 24 masked strata, 49 masked PSUs, 25 design degrees of freedom. Stratified delete-PSU jackknife, t-based 95% intervals. No model retraining.

| Model | Outcome | Unweighted AUROC | Weighted AUROC | 95% interval |
|---|---|---:|---:|---|
| logistic_regression | TARGET_DIABETES | 0.7965 | 0.8227 | [0.8057, 0.8398] |
| logistic_regression | TARGET_CVD | 0.8291 | 0.8477 | [0.8354, 0.8599] |
| logistic_regression | TARGET_OBESITY | 0.7445 | 0.7348 | [0.7137, 0.7558] |
| hist_gradient_boosting | TARGET_DIABETES | 0.8124 | 0.8399 | [0.8239, 0.8559] |
| hist_gradient_boosting | TARGET_CVD | 0.8252 | 0.8470 | [0.8279, 0.8661] |
| hist_gradient_boosting | TARGET_OBESITY | 0.7557 | 0.7394 | [0.7162, 0.7625] |
| catboost | TARGET_DIABETES | 0.8138 | 0.8407 | [0.8236, 0.8578] |
| catboost | TARGET_CVD | 0.8288 | 0.8512 | [0.8329, 0.8696] |
| catboost | TARGET_OBESITY | 0.7578 | 0.7422 | [0.7179, 0.7666] |
| mlp | TARGET_DIABETES | 0.7996 | 0.8254 | [0.8093, 0.8415] |
| mlp | TARGET_CVD | 0.8275 | 0.8453 | [0.8293, 0.8613] |
| mlp | TARGET_OBESITY | 0.7519 | 0.7440 | [0.7212, 0.7668] |
| bionutrigpt | TARGET_DIABETES | 0.8107 | 0.8402 | [0.8203, 0.8601] |
| bionutrigpt | TARGET_CVD | 0.8288 | 0.8468 | [0.8281, 0.8656] |
| bionutrigpt | TARGET_OBESITY | 0.7467 | 0.7330 | [0.7077, 0.7583] |
| ft_transformer | TARGET_DIABETES | 0.8109 | 0.8367 | [0.8194, 0.8539] |
| ft_transformer | TARGET_CVD | 0.8350 | 0.8528 | [0.8385, 0.8672] |
| ft_transformer | TARGET_OBESITY | 0.7539 | 0.7384 | [0.7144, 0.7624] |

Weights do not correct additional complete-outcome, recall, laboratory and plausibility selection. These results are a sensitivity analysis of the retained cohort, not proof of national representativeness, Indian validity or clinical utility. Paired comparisons are exploratory without multiplicity adjustment. Subgroup survey-design evaluation remains pending.

Verification independently checked weighted AUROC by tied-score pair counting, Brier arithmetic and saved-replicate standard errors/interval arithmetic. Replicate generation has not been cross-validated against an external survey package. Full independent methodological validation remains pending.

Source: [CDC first-day dietary guidance](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DR1TOT.htm). Exact protocol, hashes, 54 metric rows, 45 paired contrasts and replicate estimates are stored in Saved Model/stage1_survey_v1/.
