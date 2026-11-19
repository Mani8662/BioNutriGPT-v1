# Stage 1 repaired comparison — current results

Updated 2026-11-19. Four of six planned model families are completed and verified. Same temporal N=7,004; no paired uncertainty or superiority inference yet.

| Outcome | Logistic AUROC | Histogram AUROC | CatBoost AUROC | MLP AUROC |
|---|---:|---:|---:|---:|
| Diabetes | 0.7965 | 0.8124 | 0.8138 | 0.7996 |
| Obesity | 0.7445 | 0.7557 | 0.7578 | 0.7519 |
| Cvd | 0.8291 | 0.8252 | 0.8288 | 0.8275 |

## Probability accuracy

Lower Brier is better. Both prespecified raw and calibrated variants are retained.

| Model | Outcome | Raw Brier | Calibrated Brier |
|---|---|---:|---:|
| logistic_regression | Diabetes | 0.124165 | 0.123871 |
| logistic_regression | Obesity | 0.206221 | 0.205941 |
| logistic_regression | Cvd | 0.088055 | 0.087806 |
| hist_gradient_boosting | Diabetes | 0.121064 | 0.120650 |
| hist_gradient_boosting | Obesity | 0.201805 | 0.201099 |
| hist_gradient_boosting | Cvd | 0.089176 | 0.088704 |
| catboost | Diabetes | 0.120351 | 0.120133 |
| catboost | Obesity | 0.201964 | 0.200574 |
| catboost | Cvd | 0.087834 | 0.087991 |
| mlp | Diabetes | 0.124715 | 0.123189 |
| mlp | Obesity | 0.205542 | 0.203210 |
| mlp | Cvd | 0.088440 | 0.087865 |

Pending: BioNutriGPT and reference FT-Transformer. Production remains legacy. No clinical cutoff or statistically supported winner is established. Details, calibration limitations and timing exclusions are in the individual model reports.
