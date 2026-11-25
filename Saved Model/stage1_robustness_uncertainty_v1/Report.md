# Stage 1 robustness uncertainty: independently verified results

Updated 2026-11-22. Post-hoc evaluation of frozen predictions for 7,004 temporal participants, six models and eight stress scenarios; raw and calibrated probabilities, three outcomes and two metrics.

AUROC: 32 contrasts; 2 intervals exclude zero (2 favour BioNutriGPT; 0 favour the comparator). BRIER: 32 contrasts; 8 intervals exclude zero (3 favour BioNutriGPT; 5 favour the comparator). These are exploratory, unadjusted pointwise intervals. Scenario-specific findings do not establish an overall robustness advantage or equivalence.

## Independent verification

All 2,688 interval pairs (5,376 endpoints) were independently reproduced using the exact 10,000 participant resamples. This comprises three intervals in each of 576 within-model rows and two intervals in each of 480 between-model rows. A count-weighted tied-score pair-counting AUROC implementation was used instead of the original expanded-sample trapezoidal calculation. All within-model baseline/stress point estimates were also checked with sklearn. No model was retrained and no prediction was regenerated. Original reports/receipts are retained under stage1_robustness_endpoint_audit_v2/before_correction/.

Resample SHA-256: `320d4a0a49a531d210a4e3ad29f720b626a179fb079b4783ee89109576dc3067`. Source bindings passed. Full endpoint receipt: Saved Model/stage1_robustness_endpoint_audit_v2/verification.json. The older verifier only checked a bootstrap mean and did not reproduce interval endpoints; it is superseded for that purpose.

## BioNutriGPT calibrated deterioration

Positive deterioration means worse: baseline minus stress for AUROC, stress minus baseline for Brier.

| Outcome | Scenario | Metric | Baseline | Stressed | Deterioration [95% CI] |
|---|---|---|---:|---:|---|
| TARGET_DIABETES | missing_diet | auroc | 0.810722 | 0.800802 | +0.009920 [+0.006107, +0.013668] |
| TARGET_DIABETES | missing_diet | brier | 0.120914 | 0.124530 | +0.003615 [+0.002502, +0.004689] |
| TARGET_DIABETES | missing_lipids | auroc | 0.810722 | 0.760739 | +0.049983 [+0.041936, +0.057806] |
| TARGET_DIABETES | missing_lipids | brier | 0.120914 | 0.132341 | +0.011427 [+0.009411, +0.013337] |
| TARGET_DIABETES | missing_both | auroc | 0.810722 | 0.744079 | +0.066643 [+0.057670, +0.075378] |
| TARGET_DIABETES | missing_both | brier | 0.120914 | 0.134954 | +0.014039 [+0.011961, +0.016041] |
| TARGET_DIABETES | missing_diet_20pct_seed17 | auroc | 0.810722 | 0.809049 | +0.001673 [-0.000252, +0.003605] |
| TARGET_DIABETES | missing_diet_20pct_seed17 | brier | 0.120914 | 0.121555 | +0.000641 [+0.000033, +0.001244] |
| TARGET_DIABETES | missing_diet_20pct_seed29 | auroc | 0.810722 | 0.809717 | +0.001005 [-0.000936, +0.002911] |
| TARGET_DIABETES | missing_diet_20pct_seed29 | brier | 0.120914 | 0.121492 | +0.000578 [-0.000036, +0.001172] |
| TARGET_DIABETES | diet_noise_10pct_seed17 | auroc | 0.810722 | 0.810518 | +0.000204 [-0.000245, +0.000656] |
| TARGET_DIABETES | diet_noise_10pct_seed17 | brier | 0.120914 | 0.120949 | +0.000035 [-0.000078, +0.000149] |
| TARGET_DIABETES | lipid_noise_5pct_seed17 | auroc | 0.810722 | 0.810089 | +0.000633 [-0.000301, +0.001569] |
| TARGET_DIABETES | lipid_noise_5pct_seed17 | brier | 0.120914 | 0.121095 | +0.000181 [-0.000060, +0.000420] |
| TARGET_DIABETES | joint_noise_seed17 | auroc | 0.810722 | 0.809445 | +0.001277 [+0.000224, +0.002333] |
| TARGET_DIABETES | joint_noise_seed17 | brier | 0.120914 | 0.121111 | +0.000196 [-0.000068, +0.000461] |
| TARGET_CVD | missing_diet | auroc | 0.828809 | 0.827284 | +0.001525 [-0.001698, +0.004889] |
| TARGET_CVD | missing_diet | brier | 0.087353 | 0.091444 | +0.004090 [+0.002883, +0.005278] |
| TARGET_CVD | missing_lipids | auroc | 0.828809 | 0.810920 | +0.017890 [+0.011135, +0.024830] |
| TARGET_CVD | missing_lipids | brier | 0.087353 | 0.091906 | +0.004552 [+0.002820, +0.006339] |
| TARGET_CVD | missing_both | auroc | 0.828809 | 0.813467 | +0.015343 [+0.007998, +0.022939] |
| TARGET_CVD | missing_both | brier | 0.087353 | 0.091005 | +0.003651 [+0.002206, +0.005126] |
| TARGET_CVD | missing_diet_20pct_seed17 | auroc | 0.828809 | 0.828510 | +0.000299 [-0.001141, +0.001755] |
| TARGET_CVD | missing_diet_20pct_seed17 | brier | 0.087353 | 0.087893 | +0.000540 [+0.000158, +0.000920] |
| TARGET_CVD | missing_diet_20pct_seed29 | auroc | 0.828809 | 0.828392 | +0.000418 [-0.001050, +0.001870] |
| TARGET_CVD | missing_diet_20pct_seed29 | brier | 0.087353 | 0.087912 | +0.000559 [+0.000183, +0.000928] |
| TARGET_CVD | diet_noise_10pct_seed17 | auroc | 0.828809 | 0.828441 | +0.000369 [-0.000024, +0.000759] |
| TARGET_CVD | diet_noise_10pct_seed17 | brier | 0.087353 | 0.087379 | +0.000026 [-0.000046, +0.000100] |
| TARGET_CVD | lipid_noise_5pct_seed17 | auroc | 0.828809 | 0.828337 | +0.000472 [-0.000415, +0.001376] |
| TARGET_CVD | lipid_noise_5pct_seed17 | brier | 0.087353 | 0.087515 | +0.000162 [-0.000036, +0.000364] |
| TARGET_CVD | joint_noise_seed17 | auroc | 0.828809 | 0.827882 | +0.000927 [-0.000056, +0.001944] |
| TARGET_CVD | joint_noise_seed17 | brier | 0.087353 | 0.087655 | +0.000302 [+0.000076, +0.000530] |

## Direct contrasts whose pointwise intervals exclude zero

Difference = BioNutriGPT deterioration minus comparator deterioration; negative means BioNutriGPT deteriorated less. Absolute stressed performance remains a separate consideration.

| Comparator | Outcome | Scenario | Metric | Difference [95% CI] |
|---|---|---|---|---|
| catboost | TARGET_DIABETES | missing_lipids | brier | -0.001576 [-0.002467, -0.000679] |
| catboost | TARGET_DIABETES | missing_both | brier | -0.003224 [-0.004812, -0.001640] |
| catboost | TARGET_DIABETES | missing_diet_20pct_seed29 | auroc | -0.002573 [-0.004562, -0.000641] |
| catboost | TARGET_CVD | missing_diet | brier | +0.003785 [+0.002589, +0.004979] |
| ft_transformer | TARGET_CVD | missing_diet | brier | +0.002166 [+0.001379, +0.002962] |
| ft_transformer | TARGET_CVD | missing_lipids | brier | +0.001129 [+0.000350, +0.001925] |
| catboost | TARGET_CVD | missing_both | auroc | -0.004996 [-0.009636, -0.000289] |
| catboost | TARGET_CVD | missing_both | brier | -0.001551 [-0.003108, -0.000002] |
| catboost | TARGET_CVD | missing_diet_20pct_seed17 | brier | +0.000448 [+0.000065, +0.000828] |
| catboost | TARGET_CVD | missing_diet_20pct_seed29 | brier | +0.000579 [+0.000201, +0.000953] |

## Limits

Intervals are conditional on fitted models and the specific perturbation realizations. They do not include training variability, complex survey design or uncertainty across new perturbation seeds. Stress magnitudes are engineering scenarios, not established clinical measurement-error distributions. Obesity and raw-probability results remain exploratory. Smaller deterioration does not automatically mean higher absolute performance. No clinical utility, Indian external validity or journal-tier claim follows.
