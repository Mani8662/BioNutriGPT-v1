# CARRS-to-BioNutriGPT Model Compatibility Audit

**Date:** 15 November 2026  
**Scope:** Compare the current BioNutriGPT Stage 1 schema and targets with published CARRS cohort measures. This is a document-level compatibility assessment; the restricted CARRS data dictionary and files have not been reviewed.

## Executive conclusion

**CARRS is a strong scientific fit for a redesigned South Asian longitudinal diabetes-risk study, but it is not plug-compatible with the current trained model.**

The current Stage 1 model has 87 NHANES-specific inputs and predicts three same-period prevalent outcomes. CARRS has repeated clinical measurements, diabetes follow-up, and a broad overlap in health domains, but its variable definitions, diet instruments, participant population, and outcome task differ. The present checkpoint, preprocessing artifact, and 87-column feature vector must not be applied to CARRS.

Recommended path: rebuild the feature schema and preprocessing from CARRS; make incident diabetes the primary outcome; compare a core non-blood model with a diet-augmented model only in the diet-measured subset; use a separate lab-enhanced model as a benchmark. Treat the Transformer as one candidate, not as a fixed requirement.

## 1. What the current model expects

According to Current_Model_Explanation.txt and train_bionutrigpt_stage1.py, current Stage 1:

- Uses 87 fixed, ordered predictors:
  - 5 lipid features.
  - 46 NHANES Day 1 diet nutrient/derived features.
  - 19 NHANES lifestyle/sleep/activity/smoking features.
  - 17 NHANES demographic/context features.
- Includes NHANES variable names and encodings such as DR1T nutrient fields, NHANES physical-activity fields, race indicators, marital status, and income-to-poverty ratio.
- Predicts cross-sectional prevalent diabetes, obesity, and CVD labels.
- Excludes HbA1c from predictors, but the diabetes label uses HbA1c, fasting glucose, or self-reported prior diagnosis.
- Was trained and preprocessed on NHANES. Its learned weights, imputers, clipping values, scalers, feature order, and risk calibration are tied to that data.

## 2. Published CARRS measure coverage

The CARRS study measures page lists age, sex, income, education, occupation, caste, marital status, family history, disease history, medication, dietary habits measured by modified FFQ, physical activity by IPAQ, sleep questionnaires, anthropometry, blood pressure, glucose/HbA1c/insulin, lipids, cardiovascular events, and mortality. CARRS-1 and CARRS-2 are separate recruitment waves in Chennai and Delhi; the cohort site reports repeated follow-up, with more follow-up rounds in CARRS-1 than CARRS-2. The cohort profile reports 24-hour diet recall in a subsample. Exact variable availability by wave and the diet-subset size must be confirmed in the approved data dictionary.

## 3. Feature compatibility matrix

| Current BioNutriGPT input group | CARRS overlap | Compatibility | What must change |
|---|---|---|---|
| Lipids: HDL, total cholesterol, non-HDL, and ratios | CARRS measures total cholesterol, HDL, LDL, and triglycerides | Partial | Confirm units, fasting/visit timing, and missingness. Derive non-HDL and ratios only when source values are available and definitions match. Retrain; do not reuse NHANES scaling. |
| Diet: 39 Day 1 nutrient fields plus 7 ratios | CARRS lists a modified FFQ and a 24-hour recall in a subsample | Low/conditional | These are not automatically equivalent to NHANES Day 1 nutrient totals. Confirm the data package contains food/nutrient-level fields, the instrument, visit, portion handling, and sample size. Build a separate diet model on eligible participants if adequately powered. |
| Lifestyle: sleep, sedentary time, activity, smoking indicators | CARRS lists sleep duration/quality, IPAQ physical activity, alcohol, and tobacco | Partial | Harmonize constructs, units, recall windows, and category coding. NHANES-specific fields such as PAD680 and PAQ-derived minutes cannot be copied directly. |
| Demographics/context | CARRS lists age, sex, income, education, occupation, caste, marital status, household size, and place of birth | Partial | Replace NHANES race/ethnicity indicators and income-to-poverty ratio with justified CARRS fields. Avoid treating different social categories as interchangeable. |
| Blood pressure and body measurements | CARRS directly measures blood pressure, height, weight, waist, and hip circumference | New opportunity | The current Stage 1 feature schema does not include these as core predictors. Consider them in the new model if available at the intended prediction time. |
| Diabetes, obesity, CVD targets | CARRS measures glycaemia, disease history, anthropometry, cardiovascular events, and repeated follow-up | Strong domain overlap, different task | The current target is prevalent cross-sectional classification. Define a diabetes-free baseline cohort and incident diabetes during follow-up. Obesity/CVD may be secondary outcomes only if their event definitions and event counts support them. |

## 4. Target compatibility

The current model answers: “Does this participant meet a same-period label for diabetes, obesity, or CVD?”

The proposed CARRS study should answer: “Among participants without diabetes at baseline, who develops diabetes at a later documented follow-up?”

That change requires:

- A prespecified baseline exclusion for prevalent diabetes using CARRS-compatible laboratory, diagnosis, and medication information.
- A prespecified incident diabetes definition and follow-up horizon.
- Correct handling of repeated visits, time to event, loss to follow-up, and any competing risks.
- Participant-level splits. Visits from one participant must never appear in both training and test data.
- Temporal validation within CARRS-1 and independent validation on CARRS-2 only if compatible follow-up outcomes and covariates are released.
- Validation of calibration at the chosen horizon, not just AUROC.

Do not claim future prediction if the files contain only baseline data or if follow-up outcome timing cannot be reconstructed.

## 5. Model transfer decision

### Reusable as a research candidate

- The general tabular Transformer/MoE code can be adapted after defining a CARRS feature schema.
- The idea of comparing a Transformer with regression and tree models is reusable.
- Some data-validation utilities may be reusable after removing NHANES-specific assumptions.

### Not reusable as-is

- Trained checkpoint and output thresholds.
- NHANES feature order, variable names, preprocessing artifact, imputation values, clipping bounds, scaler, and calibration.
- The three-head cross-sectional outcome structure.
- NHANES survey weighting and temporal split logic.
- Attention-based explanations as clinical explanations.

Retrain from scratch. Begin with survival/pooled-logistic regression and tree-based baselines. Train the Transformer only if the number of incident events is sufficient and it shows a stable, calibrated advantage.

## 6. Recommended CARRS modeling design

1. **Core model:** Baseline non-blood variables, such as demographics, anthropometry, blood pressure, family history, and lifestyle fields verified in the codebook. Do not include baseline glucose, HbA1c, diabetes diagnosis, or diabetes medication as predictors for an incident-diabetes screening model.
2. **Diet-augmented model:** Add only validated dietary variables available before the prediction point. Compare against the core model on exactly the same diet-measured participants. Treat this as a subset analysis; report its sample and event counts.
3. **Laboratory-enhanced benchmark:** Add baseline blood markers in a clearly separate comparator to measure incremental prediction value. Do not call this non-blood screening.
4. **Validation:** Develop and internally validate with participant-level, time-aware splits. Use CARRS-2 as independent validation only after confirming compatible follow-up and sufficient incident cases.
5. **Reporting:** Include cohort flow, event count, missingness, calibration, uncertainty, subgroup performance, and loss-to-follow-up handling.

## 7. Pipeline compatibility

- **Stage 1 risk model:** Needs a complete rebuild for CARRS.
- **Stage 2 counterfactual nutrient search:** Do not connect its current score-minimization logic to the CARRS risk model. Changing a model score is not evidence that a diet intervention changes disease risk. Replace it with guideline-based or clinician-provided meal targets.
- **Stage 3 meal optimizer:** The optimization engine can be evaluated separately, but CARRS participant diet data cannot be joined to INDB recipes without a validated food-item mapping. Use INDB as the recipe-composition source and disclose the separate data provenance.
- **Stage 4 explanation:** The constrained explanation pattern can remain, but generated content must be based on a validated meal basket and approved source text. It cannot diagnose or claim health benefit.
- **End-to-end claim:** Until human or clinical validation exists, describe this as a research prototype with a longitudinal screening model and a separately evaluated meal-planning module, not as a clinically proven integrated treatment.

## 8. Access and go/no-go checks

CARRS requests for publications, presentations, and ancillary analyses are reviewed by the CARRS Publications, Presentations, and Ancillary Studies Committee. Approved users must sign a CARRS Data Use Agreement. The published process asks requestors to submit two weeks before the monthly committee meeting; a decision is stated to follow within one week of the meeting. Review approved requests for overlap before applying.

Before model coding, obtain or verify:

1. Exact data release, variables, visit dates, participant linkage, and data use conditions.
2. Baseline and follow-up diabetes definitions and lab availability.
3. Incident case counts and attrition by wave/site.
4. Availability and sample size of modified FFQ and 24-hour recall variables at relevant visits.
5. Units, laboratory methods, covariate missingness, weights, strata, and cluster variables.
6. Whether CARRS-2 follow-up is released and supports independent validation.
7. Whether the proposed analysis overlaps an already approved or published CARRS project.

If these conditions are not met, keep the claim narrower or choose a different accessible dataset. Do not assume CARRS data access is immediate or guaranteed.

## 9. Overall assessment

- **Scientific fit for a South Asian longitudinal diabetes study:** High.
- **Direct compatibility with the current NHANES checkpoint:** Low.
- **Potential to support a stronger paper after redevelopment:** High, conditional on data access, outcome events, comparable diet measurements, and independent validation.
- **Diet-planner validation from CARRS alone:** Limited; CARRS diet records and INDB recipes are not necessarily item-level compatible, and neither establishes user acceptance or clinical benefit.

## Sources

- Current model feature schema and targets: repository file Current_Model_Explanation.txt and train_bionutrigpt_stage1.py.
- CARRS study measures: https://www.carrsprogram.org/study-measures
- CARRS cohort at a glance: https://www.carrsprogram.org/cohort-at-a-glance
- CARRS requests and proposals: https://www.carrsprogram.org/requests
- CARRS cohort profile, including diet subsample and BioLINCC availability: https://doi.org/10.1093/ije/dyac014
