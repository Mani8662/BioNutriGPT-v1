# Repaired Stage 1: logistic-regression baseline

Reported 2026-11-19 (Asia/Calcutta). Status: completed and verified; first of six planned model families. This report describes the explicitly approved logistic-regression run only.

Four configurations (C=0.01, 0.1, 1 and 10) each fitted three outcome heads on 16,894 training participants. Validation on 3,621 participants selected C=10 using mean diabetes/CVD AUROC. Separate logit recalibrators were then fitted on 3,621 calibration participants. Temporal evaluation used 7,004 participants. CPU numerical libraries were configured for all 12 logical cores; this does not imply every operation continuously used every core.

## Temporal results

| Outcome | AUROC | Average precision | Raw Brier | Calibrated Brier | Raw calibration slope | Calibrated slope |
|---|---:|---:|---:|---:|---:|---:|
| Diabetes | 0.7965 | 0.4680 | 0.124165 | 0.123871 | 0.8898 | 1.0044 |
| Obesity | 0.7445 | 0.6710 | 0.206221 | 0.205941 | 1.0200 | 1.0553 |
| Cvd | 0.8291 | 0.3856 | 0.088055 | 0.087806 | 0.8882 | 0.9538 |

Lower Brier is better. Recalibration slightly reduces Brier error for all three outcomes; ranking metrics remain unchanged because the fitted recalibrations are monotonic increasing. The raw/calibrated diabetes slopes move from 0.8898 to 1.0044 and CVD from 0.8882 to 0.9538. Obesity's slope moves from 1.0200 to 1.0553, illustrating that a small Brier improvement does not mean every calibration statistic improves. Calibrated intercepts are 0.0942 diabetes, 0.2814 obesity and -0.0070 CVD, so calibration is not perfect, particularly for obesity.

At the descriptive probability threshold 0.5, calibrated sensitivity is 19.1% for diabetes, 45.3% for obesity and 14.2% for CVD. These are not established screening cutoffs; a clinically appropriate decision threshold needs its own justified development-only selection. AUROC is not percentage accuracy.

## Validation selection and compute

Validation mean diabetes/CVD AUROC by configuration: 0.819053, 0.821722, 0.822753 and 0.822887. The selected value is at the upper edge of the four-point C grid; this is the best tested setting, not a claim of global optimization. No search was expanded after temporal evaluation.

The sum of recorded training/validation/checkpointing times across four configurations is 3.94 seconds. This is not the total command duration and excludes the later calibration/export/verification stages. Warm raw inference median is 0.298 ms for one participant and 0.324 ms for a batch of 128 on CPU. Timing excludes loading, preprocessing and recalibration.

## Verification and artifacts

`verify_stage1_comparison_results_v2.py --model logistic_regression` passed. It verified completion and trial hashes, current source/data/environment bindings, frozen split membership, all exported IDs and labels, validation-based selection, prediction reproduction from reloaded model/calibrators, and AUROC/AP/Brier/log-loss recomputation from exported temporal predictions.

Authoritative artifacts: `Saved Model/stage1_comparison_v2/logistic_regression/completed.json`, four trial checkpoints, `calibrators.joblib`, `temporal_metrics.csv`, `calibration_curve.csv` and local participant-level prediction CSVs. Verification: `Research Paper/Stage1_logistic_regression_v2_Verification.json`.

## Interpretation and next step

This establishes a verified simple baseline under the repaired definition and data splits. It does not establish superiority over any other model. Five planned families remain: histogram gradient boosting, CatBoost, MLP, BioNutriGPT and reference FT-Transformer. Next independent run is histogram gradient boosting, subject to the user's requested approval.

Single seed, modest tuning, unweighted selected-sample evaluation and previously inspected temporal data remain limitations. Paired confidence intervals, survey analysis, multi-seed stability, robustness and architectural ablations are not completed by this run. Legacy and repaired results use different outcome/cohort definitions and cannot be treated as a controlled before/after model comparison. Production was not updated, and the manuscript/submission ZIP still contains legacy Stage 1 results pending full reconciliation.
