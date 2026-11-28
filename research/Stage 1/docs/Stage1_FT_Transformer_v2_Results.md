# Repaired Stage 1: reference FT-Transformer GPU results

Reported 2026-11-19 (Asia/Calcutta). The separately authorized run completed all four configurations; independent artifact/prediction verification passed. All six planned model families are now complete.

The reference implementation uses rtdl_revisiting_models with all 87 predictors supplied as numeric features after the common training-only preprocessing. Training/validation/calibration/temporal sample sizes are 16,894/3,621/3,621/7,004. Validation-only mean diabetes/CVD AUROC selected trial 02: three blocks, learning rate 0.0001, weight decay 0.0001, 925,635 parameters. The selected checkpoint was epoch 12. The four configurations stopped at epochs 20, 15, 19 and 11. The common seed 42, four-configuration budget, 40-epoch cap and seven-epoch patience were retained.

## Temporal results

| Outcome | AUROC | Average precision | Raw Brier | Calibrated Brier | Calibrated intercept | Calibrated slope |
|---|---:|---:|---:|---:|---:|---:|
| DIABETES | 0.8109 | 0.4886 | 0.121541 | 0.120729 | 0.0485 | 0.9800 |
| OBESITY | 0.7539 | 0.6885 | 0.206514 | 0.201285 | 0.2492 | 1.0538 |
| CVD | 0.8350 | 0.4029 | 0.086987 | 0.086778 | -0.0482 | 0.9477 |

Reserved-set recalibration reduced Brier error for all three outcomes. It did not improve every calibration statistic: the CVD slope moved slightly farther from one. Obesity retains a positive calibration intercept. AUROC and AP are unchanged by the monotonic recalibrators.

FT-Transformer has the highest CVD AUROC and lowest raw/calibrated CVD Brier error among all six families. CatBoost retains the highest diabetes and obesity AUROC. FT-Transformer exceeds BioNutriGPT's AUROC point estimates on all three outcomes, with a very small diabetes difference. No paired confidence intervals have yet been computed, so statistical superiority is not established. These findings do not support claiming an overall advantage for the custom BioNutriGPT architecture.

## Compute and verification

Recorded training/validation/checkpointing time across four configurations: 488.32 seconds on the RTX 3050. Warm synchronized GPU prediction medians (20 repeats): 3.065 ms for one participant, 20.839 ms for a batch of 128. Training time excludes startup, calibration, exports and verification; inference excludes loading, preprocessing and recalibration. These are run-specific measurements rather than a standardized device-matched benchmark against CPU models.

Verification passed all seven categories: final/trial hashes, source/data/environment binding, frozen split/schema, validation selection, exported IDs/labels, reloaded model/calibrator prediction agreement and metric reproduction. Evidence: Stage1_ft_transformer_v2_Verification.json and Saved Model/stage1_comparison_v2/ft_transformer/completed.json.

## Remaining work

The six-family comparison is complete; paired uncertainty analysis is the next proposed independent task and has not been launched. Single-seed variability, ablations, robustness, survey inference and clinical utility remain unresolved. The temporal period was already inspected historically; this is not a new untouched external cohort. Production and manuscript/submission results remain unchanged until reconciliation.
