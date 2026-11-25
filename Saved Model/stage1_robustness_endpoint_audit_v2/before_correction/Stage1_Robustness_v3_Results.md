# Stage 1 frozen-model robustness study

Completed 2026-11-20. Six repaired model families, nine fixed scenarios (baseline plus eight stresses), 7,004 identical temporal participants, three outcomes and raw/frozen-calibrated probabilities: 324 metric rows. No retraining, calibration refit, temporal selection or production-model replacement. Neural inference used CUDA; classical inference used CPU parallelism. Baseline reloads reproduced saved predictions within absolute tolerance 1e-6. Original weights/preprocessing/calibrators remained hash-identical.

## Prespecified engineering scenarios

Whole dietary modality missing (39 raw nutrients), whole lipid modality missing (HDL, total cholesterol and stored non-HDL), both missing, independent 20% dietary-cell loss at seeds 17 and 29, uniform multiplicative dietary errors within +/-10%, lipid errors within +/-5%, and joint errors (seed 17). All participants remain in every scenario. Labels are never perturbed.

Missing values use the existing frozen training medians and missingness flags, not zeros. Engineered ratios are regenerated through the repaired training preprocessor. For lipid perturbations, non-HDL is recomputed as max(total-HDL,0). For full missing lipids, each missing raw field uses its existing trained imputation policy, which may not enforce exact biochemical relationships between separately imputed medians. Food constituent sums are not constrained under independent nutrient perturbation: these are coordinate-wise engineering stresses, not validated physiological or assay error models. Their magnitudes are chosen sensitivity scenarios, not literature-derived measurement error distributions. Cohort selection originally favored complete records; simulated missingness does not represent real nonrandom missingness.

## Calibrated diabetes discrimination

| Model | Baseline | Diet missing | Lipids missing | Both missing | Joint noise |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.7965 | 0.7909 | 0.7519 | 0.7420 | 0.7950 |
| hist_gradient_boosting | 0.8124 | 0.7999 | 0.7625 | 0.7423 | 0.8100 |
| catboost | 0.8138 | 0.8022 | 0.7646 | 0.7449 | 0.8128 |
| mlp | 0.7996 | 0.7934 | 0.7564 | 0.7397 | 0.7991 |
| bionutrigpt | 0.8107 | 0.8008 | 0.7607 | 0.7441 | 0.8094 |
| ft_transformer | 0.8109 | 0.8043 | 0.7628 | 0.7478 | 0.8098 |

## BioNutriGPT calibrated results

| Scenario | Diabetes AUROC | Diabetes Brier | CVD AUROC | CVD Brier | Obesity AUROC |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8107 | 0.120914 | 0.8288 | 0.087353 | 0.7467 |
| missing_diet | 0.8008 | 0.124530 | 0.8273 | 0.091444 | 0.7331 |
| missing_lipids | 0.7607 | 0.132341 | 0.8109 | 0.091906 | 0.6669 |
| missing_both | 0.7441 | 0.134954 | 0.8135 | 0.091005 | 0.6397 |
| missing_diet_20pct_seed17 | 0.8090 | 0.121555 | 0.8285 | 0.087893 | 0.7460 |
| missing_diet_20pct_seed29 | 0.8097 | 0.121492 | 0.8284 | 0.087912 | 0.7445 |
| diet_noise_10pct_seed17 | 0.8105 | 0.120949 | 0.8284 | 0.087379 | 0.7471 |
| lipid_noise_5pct_seed17 | 0.8101 | 0.121095 | 0.8283 | 0.087515 | 0.7463 |
| joint_noise_seed17 | 0.8094 | 0.121111 | 0.8279 | 0.087655 | 0.7459 |

## Interpretation

Missing lipids materially reduces discrimination, particularly diabetes/obesity, across the models. Smaller simulated perturbations produce much smaller aggregate changes in this study. The custom Transformer does not establish an overall robustness advantage. Preserving AUROC does not guarantee stable individual decisions: for example, BioNutriGPT's calibrated obesity decisions at 0.5 change for 12.78% of participants when all diet inputs are missing. This fixed threshold is descriptive, not clinically validated. Probability shifts, AP, calibration slope/intercept and decisions are retained for every model/outcome in metrics.csv.

All results are descriptive, unweighted, conditional on selected fitted models and scenario seeds. Calibration slope/intercept are diagnostic fits on evaluation predictions only and are not used to alter predictions. These stresses do not establish clinical safety, natural missing-data performance, Indian generalization, or production performance.

## Paired Uncertainty Evaluation Addendum (2026-11-21)

Completed post-hoc paired uncertainty evaluation across all six models, eight stress scenarios, three outcomes, two variants, and two metrics using 10,000 participant-level bootstrap resamples (seed `20261121`). Full report, tables, and verification receipts are archived in [Report.md](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/Report.md).
- **Within-Model Deterioration:** For BioNutriGPT calibrated diabetes, missing lipids causes +0.0500 [+0.0419, +0.0578] AUROC loss and +0.0114 [+0.0094, +0.0133] Brier increase; missing diet causes +0.0099 [+0.0061, +0.0137] AUROC loss; modest noise causes <= +0.0013 AUROC loss. For CVD, missing diet AUROC loss is +0.0015 [-0.0017, +0.0049] (spans zero).
- **Direct Paired Contrasts:** Under complete dietary omission, BioNutriGPT calibrated diabetes AUROC deterioration difference versus CatBoost is -0.0016 [-0.0051, +0.0018] and versus FT-Transformer is +0.0033 [-0.0006, +0.0073]; both intervals span zero. Across all primary calibrated AUROC contrast scenarios with CatBoost and FT-Transformer, intervals span zero, confirming that BioNutriGPT exhibits no statistically detectable discrimination robustness advantage over competitive tabular baselines.
- **Verification:** Independent verification script `verify_stage1_robustness_uncertainty_v1.py` passed all 7 checks (protocol disclosure, 55 SHA-256 bindings, index reproduction, sklearn recomputations, contrast arithmetic, spot bootstrap distribution reproduction, and invariance/calibration contracts).

## Verification and artifacts

The independent verifier recomputed AUROC, AP, Brier and decision flips from all 54 persisted prediction archives and checked labels, IDs, finite probabilities and frozen artifact hashes. completed.json and verification.json record completion. protocol.json, binding.json, scenario_checks.json, timings.json and per-scenario files retain reproducibility evidence. NPZ files contain local participant IDs/labels and must not be distributed indiscriminately. V1/v2 exports failed with Windows OSError 22 and are marked incomplete; v3 avoids repeated rewriting of scenario result files. No experimental conditions changed between those attempts.

Reproduce: python run_stage1_robustness_v3.py (refuses to overwrite existing output). Verify existing results: python verify_stage1_robustness_v3.py. Paired uncertainty analysis: python run_stage1_robustness_uncertainty_v1.py and python verify_stage1_robustness_uncertainty_v1.py.
