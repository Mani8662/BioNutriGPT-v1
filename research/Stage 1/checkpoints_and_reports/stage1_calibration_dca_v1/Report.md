# Stage 1 weighted calibration and exploratory decision curves

Completed 2026-11-20. Six frozen models, 7,004 temporal participants. WTDRD1PP, 24 strata/49 masked PSUs, R survey JKn mse=TRUE, t-based intervals with 25 design degrees of freedom. Calibration diagnostics fitted on the holdout are for evaluation only: no prediction recalibration, threshold selection, training or production changes.

## Calibration

Ideal calibration-in-the-large (offset intercept) is 0; ideal joint slope is 1. Joint intercept and offset intercept are different quantities. Raw and frozen calibrated predictions were both evaluated. The table below shows BioNutriGPT calibrated predictions.

| Outcome | Offset intercept [95% CI] | Joint slope [95% CI] |
|---|---|---|
| TARGET_DIABETES | 0.0512 [-0.0230, 0.1255] | 1.0873 [0.9740, 1.2005] |
| TARGET_CVD | 0.0273 [-0.1454, 0.2000] | 0.9602 [0.8813, 1.0391] |
| TARGET_OBESITY | 0.2447 [0.1265, 0.3628] | 0.9465 [0.8256, 1.0675] |

Diabetes and CVD offset intervals include zero and slope intervals include one; this does not prove perfect calibration or equivalence. Obesity has a positive offset with an interval excluding zero, indicating average underprediction on the weighted retained cohort. Probability-bin plots are descriptive, not smooth estimates or evidence of calibration everywhere. Sparse bins (N<100 or positives/negatives<20) are marked; no uncertainty bands are claimed for bin curves.

## Decision curves

Net benefit = weighted TP proportion minus weighted FP proportion times threshold/(1-threshold). Thresholds 0.05–0.50 in steps of 0.01 were specified before this evaluation. Diabetes and CVD only. Assess-all and assess-none reference policies and paired BioNutriGPT-minus-comparator intervals use the same replicate design. No measurement-cost term or additional test harm is included.

The cohort includes already diagnosed disease and prevalent self-reported CVD. Thus these are hypothetical further-assessment decision curves, not demonstrated benefit for identifying undiagnosed disease, incident CVD prediction, treatment selection or meal planning. Thresholds are exploratory and not clinical recommendations. Pointwise intervals are unadjusted for the many models and thresholds; no overall architectural superiority claim is justified.

At threshold 0.20, BioNutriGPT net benefit is 0.05269 for diabetes and 0.03133 for CVD. Its paired differences from CatBoost and FT-Transformer include zero for both outcomes at this threshold. This is an illustration from the fixed grid, not an optimal threshold selection. At CVD threshold 0.50, the model-versus-none interval includes zero.

## Verification and artifacts

Independent Python checks passed for all 108 calibration coefficient point estimates, 360 bin rows, 736 decision-curve estimates/uncertainty intervals and 644 paired contrasts. Calibration coefficients were checked by independently solving weighted score equations. Decision-curve uncertainty was independently reconstructed with stratified delete-PSU weights. Calibration coefficient uncertainty is from R survey and was not independently refitted in Python. Simple all/none/perfect-classifier arithmetic checks passed. R emitted locale startup warnings; numerical verification passed.

Saved CSVs, protocol, R session details and verification hashes are in Saved Model/stage1_calibration_dca_v1/. Figures: calibration.png/.pdf and decision_curves.png/.pdf. No journal-readiness or fairness guarantee follows. Remaining limits include analytic selection, conditional-on-model uncertainty and lack of Indian prediction validation.

Sources: [R survey model documentation](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyglm.html), [decision curve methodology and examples](https://www.danieldsjoberg.com/dcurves/articles/dca.html).
