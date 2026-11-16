# Stage 1 results: reconciled comparison

## Primary comparison

All models below are scored on the same NHANES temporal holdout and with the same `WTMECPRP` survey weights. Lower Brier score is better. PR-AUC is average precision.

| Outcome | Model | Weighted AUROC | Weighted PR-AUC | Weighted Brier |
|---|---|---:|---:|---:|
| Diabetes | BioNutriGPT Stage1 | 0.8151 | 0.4464 | 0.1660 |
| Diabetes | Logistic Regression | 0.8080 | 0.4244 | 0.1539 |
| Diabetes | Random Forest | 0.8024 | 0.4352 | 0.1074 |
| Diabetes | Gradient Boosted Trees | 0.8279 | 0.4603 | 0.1034 |
| Obesity | BioNutriGPT Stage1 | 0.7501 | 0.6821 | 0.2017 |
| Obesity | Logistic Regression | 0.7471 | 0.6750 | 0.2021 |
| Obesity | Random Forest | 0.7328 | 0.6571 | 0.2117 |
| Obesity | Gradient Boosted Trees | 0.7503 | 0.6805 | 0.2025 |
| Cvd | BioNutriGPT Stage1 | 0.8427 | 0.3685 | 0.1509 |
| Cvd | Logistic Regression | 0.8436 | 0.3576 | 0.1471 |
| Cvd | Random Forest | 0.8202 | 0.3383 | 0.0760 |
| Cvd | Gradient Boosted Trees | 0.8407 | 0.3449 | 0.0748 |

## Why the reports differed

- `stage1_test_evaluation_report.json` reports **unweighted** results. Its diabetes, obesity, and CVD AUROC, PR-AUC, and Brier values match the BioNutriGPT unweighted results in the benchmark report to a maximum absolute difference of `0.000049` (rounding precision).
- The benchmark report also calculates **survey-weighted** metrics using `WTMECPRP`; those are the primary comparison above. The weighted diabetes AUROC for BioNutriGPT is 0.8151, while its unweighted value is 0.7999. These are two different summaries of the same test predictions, not conflicting model runs.
- The older `stage1_test_evaluation_report_pre_hba1c.json` and `stage1_test_evaluation_report_88feat_legacy.json` contain the legacy diabetes AUROC of 0.9558. They are not the current 87-feature, HbA1c-excluded result and should not be used as the current headline result.

## Data and method checks

- Training set: 23,694 rows. Test set: 7,046 rows; 7,046 unique `SEQN` identifiers.
- Current train/test files contain 87 model features; `LBXGH` is absent from the test predictors.
- The raw-data merge supplied a non-missing `WTMECPRP` value for all 7,046 test rows.
- The baseline script evaluates BioNutriGPT, logistic regression, random forest, and histogram gradient boosting on the same test rows and feature list. The current reports were created after the current checkpoint was saved; this reconciliation validates and consolidates those saved evaluations rather than refitting the models.

## Interpretation limits

- BioNutriGPT does not lead on weighted AUROC: gradient boosting leads for diabetes by 0.0128 AUROC; obesity is effectively tied (0.7503 vs 0.7501); logistic regression is marginally highest for CVD (0.8436 vs 0.8427). These are descriptive point-estimate differences, not evidence of statistically significant differences.
- BioNutriGPT has a higher (worse) weighted Brier score than the best reported baseline for diabetes and CVD. For obesity, its Brier score is slightly lower than the other models.
- The existing 1,000-row bootstrap intervals are not design-based NHANES intervals: the code resamples individual rows and does not account for strata (`SDMVSTRA`) or primary sampling units (`SDMVPSU`). They are omitted from this reconciled table; do not use them to claim significance. Calibration and DCA fields in the benchmark report are also unweighted.
- This is a temporal holdout within NHANES, not independent external validation. Baseline training is not survey-weighted, and the logistic-regression baseline uses class balancing; interpret as a same-test-set benchmark, not a fully survey-weighted training comparison.

Full precision/source reports: `stage1_test_evaluation_report.json` and `tripod_baseline_evaluation_report.json`.
