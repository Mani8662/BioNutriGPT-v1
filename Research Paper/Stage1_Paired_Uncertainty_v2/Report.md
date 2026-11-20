# Stage 1 paired uncertainty analysis — version 2

Completed and verified 2026-11-19. All six frozen models were compared on the same 7,004 temporal participants. No models or calibrators were refitted.

## Method

20,000 ordinary participant bootstrap resamples, seed 20261119, with identical row multiplicities across models and outcomes. Twelve CPU workers were available; numerical libraries used one thread per worker. The analysis reports percentile 95% pointwise intervals. The 20 primary contrasts compare BioNutriGPT with each of five alternatives on diabetes/CVD raw AUROC and calibrated Brier. Bonferroni-adjusted intervals use the 0.125th and 99.875th bootstrap percentiles (99.75% individual intervals, approximate 95% family-wise coverage). Obesity and raw Brier contrasts are descriptive/exploratory. This analysis plan was fixed after inspecting model point estimates; it is not preregistered confirmatory evidence.

Differences below are **BioNutriGPT minus comparator**. Positive AUROC differences favor BioNutriGPT; negative Brier differences favor BioNutriGPT. An interval containing zero does not establish equivalence.

## Primary paired contrasts

| Outcome | Metric | Comparator | Difference | Pointwise 95% interval | Family-adjusted interval |
|---|---|---|---:|---|---|
| DIABETES | raw_auroc | logistic_regression | 0.014221 | [0.007830, 0.020680] | [0.004490, 0.024047] |
| DIABETES | raw_auroc | hist_gradient_boosting | -0.001664 | [-0.006755, 0.003519] | [-0.009547, 0.006200] |
| DIABETES | raw_auroc | catboost | -0.003069 | [-0.007183, 0.000982] | [-0.009483, 0.003003] |
| DIABETES | raw_auroc | mlp | 0.011077 | [0.005103, 0.017176] | [0.001577, 0.020406] |
| DIABETES | raw_auroc | ft_transformer | -0.000204 | [-0.005202, 0.004849] | [-0.008115, 0.007670] |
| DIABETES | calibrated_brier | logistic_regression | -0.002957 | [-0.004538, -0.001383] | [-0.005394, -0.000586] |
| DIABETES | calibrated_brier | hist_gradient_boosting | 0.000264 | [-0.001168, 0.001687] | [-0.001984, 0.002523] |
| DIABETES | calibrated_brier | catboost | 0.000781 | [-0.000300, 0.001868] | [-0.000939, 0.002504] |
| DIABETES | calibrated_brier | mlp | -0.002275 | [-0.003841, -0.000772] | [-0.004685, 0.000046] |
| DIABETES | calibrated_brier | ft_transformer | 0.000185 | [-0.001134, 0.001492] | [-0.001912, 0.002233] |
| CVD | raw_auroc | logistic_regression | -0.000253 | [-0.005826, 0.005271] | [-0.008953, 0.008323] |
| CVD | raw_auroc | hist_gradient_boosting | 0.003623 | [-0.002243, 0.009448] | [-0.005214, 0.013013] |
| CVD | raw_auroc | catboost | -0.000018 | [-0.004754, 0.004752] | [-0.007281, 0.007271] |
| CVD | raw_auroc | mlp | 0.001289 | [-0.004440, 0.007120] | [-0.007879, 0.010269] |
| CVD | raw_auroc | ft_transformer | -0.006172 | [-0.010489, -0.001902] | [-0.012936, 0.000531] |
| CVD | calibrated_brier | logistic_regression | -0.000452 | [-0.001685, 0.000786] | [-0.002370, 0.001453] |
| CVD | calibrated_brier | hist_gradient_boosting | -0.001351 | [-0.002546, -0.000152] | [-0.003253, 0.000482] |
| CVD | calibrated_brier | catboost | -0.000638 | [-0.001659, 0.000366] | [-0.002238, 0.000971] |
| CVD | calibrated_brier | mlp | -0.000511 | [-0.001616, 0.000583] | [-0.002181, 0.001259] |
| CVD | calibrated_brier | ft_transformer | 0.000576 | [-0.000402, 0.001542] | [-0.000888, 0.002054] |

## Conclusion

After adjusting the 20 primary comparisons, BioNutriGPT has higher diabetes AUROC than logistic regression and MLP, and lower calibrated diabetes Brier error than logistic regression. No other primary interval excludes zero. In particular, there is no demonstrated diabetes advantage over histogram boosting, CatBoost or reference FT-Transformer, and no demonstrated CVD advantage over any comparator.

The unadjusted CVD AUROC comparison favors FT-Transformer over BioNutriGPT (difference -0.006172, 95% interval [-0.010489, -0.001902]), but its family-adjusted interval includes zero. BioNutriGPT versus MLP diabetes Brier and versus histogram boosting CVD Brier also exclude zero only before adjustment. Neither isolated unadjusted findings nor overlapping point estimates should be used to claim overall superiority or equivalence.

These results support a limited diabetes improvement over simpler baselines, not an overall scientific advantage for the custom Transformer or its MoE/modality components. Component-level benefits still require ablations.

## Scope and verification

These intervals describe participant sampling uncertainty conditional on the existing fitted models, selected hyperparameters and calibrators. They exclude training-seed variability and selection uncertainty. The resampling is unweighted and does not account for NHANES strata/PSUs; it cannot support nationally representative inference. The cohort is selected and the historical temporal period has already been inspected. No Indian external validation, clinical threshold, equivalence margin or clinical benefit is established. Extreme adjusted endpoints have Monte Carlo uncertainty (approximately 25 resamples per tail).

Verification matched input/output hashes, reproduced all 45 contrast intervals, checked all original-sample metrics against sklearn, and independently reproduced 162 bootstrap metrics from explicitly repeated participant rows. All 20,000 x 54 saved metrics are finite. No participant-level rows are copied into this report; bootstrap metrics and provenance are retained alongside it.

## Next evidence task

Plan repeated-seed runs and targeted architecture ablations to measure training variability and whether MoE/modality pooling add value. Each independent training run remains subject to user approval. Clinical utility, robustness and survey-design sensitivity remain separate tasks. Production weights and the manuscript/submission package remain unchanged pending consolidated reconciliation.
