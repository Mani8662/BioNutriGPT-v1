# Stage 1 repaired comparison — current results

Updated 2026-11-19. All six planned model families are completed and verified. Same temporal N=7,004 and 87 predictors; paired uncertainty is reported in Stage1_Paired_Uncertainty_v2/Report.md.

| Model | Diabetes AUROC | Obesity AUROC | CVD AUROC |
|---|---:|---:|---:|
| logistic_regression | 0.7965 | 0.7445 | 0.8291 |
| hist_gradient_boosting | 0.8124 | 0.7557 | 0.8252 |
| catboost | 0.8138 | 0.7578 | 0.8288 |
| mlp | 0.7996 | 0.7519 | 0.8275 |
| bionutrigpt | 0.8107 | 0.7467 | 0.8288 |
| ft_transformer | 0.8109 | 0.7539 | 0.8350 |

## Probability accuracy

Lower Brier is better. Both prespecified raw and calibrated variants are retained.

| Model | Outcome | Raw Brier | Calibrated Brier |
|---|---|---:|---:|
| logistic_regression | DIABETES | 0.124165 | 0.123871 |
| logistic_regression | OBESITY | 0.206221 | 0.205941 |
| logistic_regression | CVD | 0.088055 | 0.087806 |
| hist_gradient_boosting | DIABETES | 0.121064 | 0.120650 |
| hist_gradient_boosting | OBESITY | 0.201805 | 0.201099 |
| hist_gradient_boosting | CVD | 0.089176 | 0.088704 |
| catboost | DIABETES | 0.120351 | 0.120133 |
| catboost | OBESITY | 0.201964 | 0.200574 |
| catboost | CVD | 0.087834 | 0.087991 |
| mlp | DIABETES | 0.124715 | 0.123189 |
| mlp | OBESITY | 0.205542 | 0.203210 |
| mlp | CVD | 0.088440 | 0.087865 |
| bionutrigpt | DIABETES | 0.121065 | 0.120914 |
| bionutrigpt | OBESITY | 0.204283 | 0.204624 |
| bionutrigpt | CVD | 0.087336 | 0.087353 |
| ft_transformer | DIABETES | 0.121541 | 0.120729 |
| ft_transformer | OBESITY | 0.206514 | 0.201285 |
| ft_transformer | CVD | 0.086987 | 0.086778 |

## Compute

Training intervals include validation/checkpointing but exclude startup, calibration, exports, verification and interrupted attempts. Warm raw prediction medians exclude loading, preprocessing and recalibration. Device differences prevent treating this as a device-matched speed ranking.

| Model | Training seconds (4 configurations) | Training device | Prediction device | One participant (ms) | Batch 128 (ms) |
|---|---:|---|---|---:|---:|
| logistic_regression | 3.94 | cpu | cpu | 0.298 | 0.324 |
| hist_gradient_boosting | 19.70 | cpu | cpu | 10.586 | 12.235 |
| catboost | 111.06 | cuda | cpu | 3.269 | 4.196 |
| mlp | 59.52 | cuda | cuda | 0.673 | 0.913 |
| bionutrigpt | 650.62 | cuda | cuda | 17.032 | 20.796 |
| ft_transformer | 488.32 | cuda | cuda | 3.065 | 20.839 |

## Interpretation and remaining evidence

CatBoost has the highest diabetes/obesity AUROC point estimates. Reference FT-Transformer has the highest CVD AUROC and lowest CVD Brier error in both variants. BioNutriGPT does not show an overall advantage; the earlier five-model CVD Brier lead no longer holds after adding FT-Transformer. These point-estimate rankings do not establish superiority; see the paired analysis below.

All runs use seed 42 and four configurations, an equal-count rather than equal-compute search. Results are unweighted on selected participants from an already-inspected historical temporal period. Paired participant intervals are complete. Seed variability, robustness, architectural ablations, survey-design analysis and clinical utility remain separate evidence tasks. No validated clinical cutoff, Indian external validation or clinical benefit is established. Production and manuscript/submission performance remain legacy pending reconciliation.

## Paired uncertainty update — 2026-11-19

All six model families and paired uncertainty analysis are complete and verified. On 7,004 temporal participants, 20,000 paired bootstrap resamples show that BioNutriGPT improves diabetes AUROC over logistic regression and MLP, and calibrated diabetes Brier over logistic regression, after adjustment across 20 primary contrasts. No other primary contrast excludes zero; no overall advantage over competitive boosted-tree/reference Transformer models is established. These are unweighted conditional-on-fitted-model intervals, not survey-design or training-seed uncertainty. See Research Paper/Stage1_Paired_Uncertainty_v2/Report.md. Repeated-seed and component-ablation work remains pending; no new training launched. Production/manuscript performance remains legacy pending reconciliation.
