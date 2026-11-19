# Repaired Stage 1: histogram gradient boosting

Reported 2026-11-19 (Asia/Calcutta). Status: completed and verified; two of six planned model families now complete. This run was explicitly approved separately from logistic regression.

Four configurations each fitted three binary outcome heads using 16,894 training participants. Validation-only mean diabetes/CVD AUROC selected learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0 and max_iter=300. No internal random validation split was used. Calibration then used the separate 3,621-person calibration partition. Temporal evaluation N=7,004. All models used the same repaired 87-predictor matrix and source partitions as logistic regression.

## Results

| Outcome | AUROC | Average precision | Raw Brier | Calibrated Brier | Raw calibration slope | Calibrated slope |
|---|---:|---:|---:|---:|---:|---:|
| Diabetes | 0.8124 | 0.4899 | 0.121064 | 0.120650 | 0.8990 | 0.9752 |
| Obesity | 0.7557 | 0.6879 | 0.201805 | 0.201099 | 1.0799 | 1.0599 |
| Cvd | 0.8252 | 0.3667 | 0.089176 | 0.088704 | 0.8474 | 0.9513 |

Recalibration lowered Brier error for all outcomes and moved calibration slopes closer to one. Calibrated intercepts are 0.0461 diabetes, 0.2631 obesity and -0.0455 CVD; calibration is not perfect. The positive recalibration slopes preserve rankings, so AUROC/AP remain unchanged.

## Comparison so far (same temporal participants)

| Outcome | Logistic AUROC | Histogram AUROC | Histogram minus logistic | Logistic calibrated Brier | Histogram calibrated Brier |
|---|---:|---:|---:|---:|---:|
| Diabetes | 0.7965 | 0.8124 | +0.0159 | 0.123871 | 0.120650 |
| Obesity | 0.7445 | 0.7557 | +0.0112 | 0.205941 | 0.201099 |
| Cvd | 0.8291 | 0.8252 | -0.0039 | 0.087806 | 0.088704 |

Histogram boosting has better diabetes/obesity point estimates for both AUROC and calibrated Brier; logistic regression has better CVD point estimates. No statistically supported superiority claim is made: paired uncertainty and seed variability remain pending, and four model families have not run. Do not compare these results to legacy models trained/evaluated under different endpoints and cohorts.

At the fixed descriptive threshold 0.5, calibrated sensitivity is 23.8% diabetes, 49.7% obesity and 11.8% CVD. These are not clinically justified screening thresholds. A useful screening action and development-only threshold-selection policy remain to be established.

## Selection, timing and verification

Validation primary mean AUROCs in predefined configuration order: 0.823584, 0.818253, 0.821093, 0.814212. The first configuration was selected before calibration and temporal prediction. All four training/validation/checkpointing intervals total 19.70 seconds; this excludes calibration, exports, verification and startup. CPU numerical libraries were configured for all 12 logical cores. Warm raw-model median inference: 10.586 ms for one participant, 12.235 ms per batch of 128. These timings exclude loading, preprocessing and recalibration; the corresponding logistic-regression medians were 0.298 and 0.324 ms.

Independent verification passed: all completion/trial hashes, current source/data/environment binding, frozen schemas and splits, validation selection, every exported participant ID/outcome, reloaded checkpoint/calibrator prediction reproduction and exported AUROC/AP/Brier/log-loss.

Artifacts: Saved Model/stage1_comparison_v2/hist_gradient_boosting. Verification: Research Paper/Stage1_hist_gradient_boosting_v2_Verification.json. Local predictions retain participant identifiers and have not been added to the submission archive.

## Remaining work

Next independent family: CatBoost on the RTX 3050 GPU, pending approval. MLP, BioNutriGPT and reference FT-Transformer follow individually. Unweighted single-seed point estimates with modest tuning remain limited; paired uncertainty, survey analysis, robustness, ablations and deployment parity are separate tasks. Production and manuscript/submission ZIP remain unchanged pending full reconciliation.
