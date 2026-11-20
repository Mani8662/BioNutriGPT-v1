# Stage 2 production-artifact stability pilot

2026-11-20. Six runs on the existing backend sample with 50 Adam steps each, using CUDA and production model assets. No production files or model selection changed. This is a single-fixture computational pilot, not cohort evidence, clinical efficacy or food-plan feasibility.

## Artifact finding

Production Stage2 uses bionutrigpt_stage1_best.pt and Data/preprocessing_artifacts.joblib. Repaired comparison uses trial_03/model.pt and Data/stage1_repaired_v2/preprocessing_artifacts.joblib. Input files and code are SHA-256 bound in binding.json. Stage2 applies raw sigmoid probabilities and fixed cutoffs 0.617/0.445/0.603; it does not apply the repaired comparison calibrators. Therefore these results cannot be attributed to the repaired model or its reported calibration.

## Corrected pilot results

Every case recomputes dietary ratios from its nutrient vector. Baseline/repeat and +/-1% weight and dietary perturbations all passed existing Stage2 filters, had no envelope violations at tolerance 1e-4, and preserved all non-diet/non-derived-ratio features exactly in the reconstructed candidate. Baseline repeat was identical. Weight changes adjusted the prescribed energy/bounds but did not change the optimized nutrient vector for this fixture. This does not imply weight is ignored generally.

| Case | Maximum normalized target change |
|---|---:|
| baseline | 0.0000% |
| repeat | 0.0000% |
| weight_minus_1pct | 0.0000% |
| weight_plus_1pct | 0.0000% |
| diet_minus_1pct | 1.4576% |
| diet_plus_1pct | 1.0311% |

Metric: maximum absolute target difference divided by max(abs(baseline target),1), with the numerical floor in each nutrient's own unit. This is not a universal clinical distance or an ordinary percentage for baseline targets below one. All six runs being certified means only that the implemented checks passed; independent clinical validation is absent.

## Preserved initial pilot and input issue

stage2_stability_v1 retained supplied sample ratios for baseline/weight cases but recomputed them for dietary perturbations. It therefore confounded dietary perturbation with ratio correction. Its large apparent copper change must not be presented as isolated dietary instability. V1 is retained, not overwritten. V2 explicitly recomputes ratios consistently before baseline and all perturbations. This exposes an input-consistency question for production: supplied derived ratios need validation/reconciliation before making sensitivity claims. No production policy was changed in this step.

## Still required

Verify/repair derived-feature consistency at input, explicitly choose and bind the Stage1 model/calibrator/threshold contract, then extend to multiple eligible profiles, measurements and guideline-only targets. Test Stage3 realization separately. The earlier hard-threshold discontinuities remain; local perturbations in one fixture do not test or resolve them. Any lower model score after feature changes remains model behavior, not predicted causal health benefit.
