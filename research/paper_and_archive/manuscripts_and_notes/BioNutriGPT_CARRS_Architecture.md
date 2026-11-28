# BioNutriGPT: CARRS-Based Architecture and Workflow

**Status:** Proposed target architecture. This is not the architecture currently validated in the repository.  
**Primary cohort candidate:** CARRS-1 and, if access and harmonized outcomes permit, independent CARRS-2.  
**Meal composition source:** INDB, used separately from participant-level CARRS records.

## 1. Research objective

Build and evaluate a longitudinal screening model for incident type 2 diabetes in South Asian adults, then demonstrate a separately validated Indian recipe-planning module.

The primary scientific question is:

> Among South Asian adults without diabetes at baseline, can a model using non-blood predictors estimate the risk of incident type 2 diabetes over a follow-up interval supported by CARRS, and does measured dietary information add predictive value beyond demographic, anthropometric, lifestyle, and other non-blood inputs?

The diet-augmented comparison is conditional on the released CARRS data containing enough participants with compatible dietary measurements. Do not assume every CARRS participant has detailed diet data.

## 2. Cohort and target

CARRS recruited adults aged 20 and above in Chennai and Delhi in India and Karachi in Pakistan. It has repeated follow-up and biospecimens. The cohort profile reports dietary habits and 24-hour dietary recall in a subsample, and says CARRS-1 baseline and its first three follow-up visits are available through NHLBI BioLINCC. Confirm the exact released variables, diet subsample size, outcome timing, survey/retention weights, and access conditions from the data package before freezing the protocol.

### Target outcome

- Include adults who are free of diabetes at baseline using a prespecified CARRS-compatible definition.
- Predict incident diabetes at a follow-up visit using documented laboratory criteria and/or physician diagnosis/medication according to a frozen definition.
- Choose the prediction horizon to match the actual observation intervals and number of outcome events. Do not promise a 3-year or 5-year estimate until the data dictionary confirms that it can be measured.
- Exclude baseline glucose, HbA1c, diabetes diagnosis, and diabetes medication from the non-blood predictor model. Baseline laboratory measures may define eligibility and outcomes, not serve as hidden predictors.
- If a laboratory-enhanced model is useful, report it as a distinct comparator, not as the non-blood screening model.
- Generalizability is initially to the CARRS recruitment settings, not all India or all South Asia.

## 3. Research data architecture

CARRS data request and governance  
-> data dictionary, forms, linkage and access review  
-> participant-level visit table with one row per person-visit  
-> baseline eligibility and diabetes-free cohort  
-> outcome adjudication across follow-up visits  
-> predictor table restricted to information available at baseline  
-> split by participant and cohort/wave  
-> model development and locked evaluation  
-> reproducible performance, calibration, subgroup and missingness reports

### Cohort splitting

Preferred design, subject to data availability:

1. Develop the model using eligible CARRS-1 baseline participants and subsequent CARRS-1 follow-up outcomes.
2. Use a later, untouched follow-up or a separate CARRS-2 cohort for validation only if its data access, outcome definitions, predictor fields, and follow-up are compatible.
3. Never split visit rows randomly. All visits from a participant must stay together.
4. If CARRS-2 follow-up cannot be accessed, use a prespecified temporal validation within CARRS-1 and label it honestly. It is not an independent external validation.
5. Account for loss to follow-up, competing death where relevant, missingness, and the cohort's sampling design as supported by its documentation.

## 4. Prediction models

### Model A: non-blood baseline model

Candidate predictors, only after codebook confirmation:

- Age and sex.
- Height, weight, BMI, and waist circumference.
- Blood pressure.
- Family history.
- Education or other socioeconomic fields if justified and consistently available.
- Smoking, alcohol, physical activity, and other baseline lifestyle measures.
- No blood biomarkers or direct indicators of already diagnosed diabetes.

### Model B: diet-augmented model

On the subset with suitable baseline diet measures, add the available dietary variables or nutrient estimates. First inspect whether CARRS provides food-level intake, only an FFQ, or a 24-hour recall subsample for the relevant wave. Use only data that are actually released and documented. Do not infer exact recipe consumption from broad food-frequency fields.

Compare Model B against Model A on the same diet-measured participants. This paired comparison estimates whether diet information adds predictive value within that subsample; it does not establish that a diet causes or prevents diabetes.

### Baselines and candidate models

- Cox or pooled logistic regression, selected to match event timing and censoring.
- A validated established diabetes risk score, if its inputs and target population are suitable.
- One or two tree-based models.
- Transformer only if the effective sample size and incident event count support it and it demonstrates repeatable value over simpler models.

Tune and select models inside the development data only. Use identical input sets, participants, weighting, and test splits for every comparison. The model choice is based on calibration and clinical usefulness as well as discrimination, not architecture name.

### Evaluation

Report, as appropriate to the endpoint:

- Time-dependent AUROC and precision-recall performance.
- Calibration at the selected prediction horizon, including calibration plot, slope/intercept, and Brier score.
- Sensitivity and specificity at prespecified thresholds that correspond to a plausible referral for clinical testing.
- Confidence intervals using participant-level and, where appropriate, cluster-aware resampling.
- Decision curves only where the clinical action and threshold are clearly defined.
- Performance and calibration by sex, age, city/site, and socioeconomic subgroups when sample sizes permit.
- Missing-data and loss-to-follow-up analyses.
- Complete flow counts, incident event counts, feature definitions, code, and model version.

The risk score is a prediction of future observed outcome over a stated time horizon. It is not a diagnosis, causal estimate, or guaranteed benefit from changing any model feature.

## 5. Complete application workflow: user input to output

### Step 1: Welcome and consent

The app explains its purpose, privacy practices, and limits. It says the risk score is not a diagnosis and that an elevated score should prompt professional assessment. The user can decline without entering health data.

### Step 2: Eligibility and routing

Ask age, whether the user has already been diagnosed with diabetes or uses glucose-lowering medication, and whether they have a condition requiring specialized nutrition care.

- Outside the validated CARRS age or population range: do not return a CARRS risk score.
- Known diabetes or glucose-lowering medication: skip incident-diabetes screening and direct the user to the clinician-supported planning path.
- Pregnancy, kidney disease, or another condition needing specialist diet advice: do not provide an individualized clinical meal prescription.
- Eligible and within model scope: continue.

The final routing rules must be reviewed by a clinical advisor and tied to the study population.

### Step 3: Screening inputs

Collect only fields included in the frozen model, with value, unit, date, and measurement source. Likely fields are age, sex, height, weight, waist circumference, blood pressure, family history, and lifestyle measures. Do not request blood tests in the non-blood model path.

The app explains how to measure height, weight, and waist if home measurements are accepted. If the model was developed on staff-measured values and has not been checked with home measurements, the app must not treat those sources as equivalent.

### Step 4: Validate and score

The app validates ranges and units, calculates derived measurements deterministically, and flags missing or contradictory fields. It returns no score if required inputs are absent or the profile is outside the validated range.

The model service returns a calibrated absolute risk only for a horizon supported by the data, plus a validated uncertainty/support indicator. The threshold for a high-risk referral is prespecified and evaluated; it is not selected after looking at final test performance.

### Step 5: Show the screening result

The user receives:

- Estimated risk of incident diabetes over the stated horizon, only if validated.
- A plain-language statement that this is not a diagnosis.
- A recommended next step, such as discussing clinical testing with a qualified health professional.
- Contributing model variables described as associations, not causes.
- The population and limits: the CARRS settings and the validation population, not all people everywhere.

### Step 6: Ask whether the user wants a meal plan

Collect meal-planning information separately from risk inputs:

- Vegetarian, egg, or non-vegetarian preference.
- Foods avoided, allergies, and intolerances.
- Regional or cuisine preference only where the recipe database supports it.
- Household serving count, cooking time, and available equipment if those constraints are implemented.
- Clinician-provided targets and their units/date, if available.

Do not use a risk score to invent calorie, carbohydrate, sodium, or medical targets. Without clinician-provided targets, generate a clearly labeled general example aligned to documented population guidance. Do not optimize affordability unless current local prices are available.

### Step 7: Build the meal plan

The meal service uses INDB recipe records and deterministic nutrient calculations. It filters only verified exclusions, then uses a discrete optimizer to select whole servings under the declared objective and constraints. An independent validator recalculates nutrients and checks all hard rules.

If no feasible plan exists, the app says so and requests a change or professional review. It never silently relaxes an allergy or clinical limit. INDB data are not linked to an individual CARRS participant's consumed meals.

### Step 8: Explain and return the plan

The LLM receives only the validated plan structure, recipe IDs, portions, calculated nutrients, and approved guidance. It may explain and format the plan. It cannot add a food, alter a portion, create nutrient values, or make a treatment claim. A deterministic template is used if the LLM output fails validation.

Final output contains separate panels:

- **Risk panel:** calibrated risk, horizon, uncertainty, referral guidance, and study-population note.
- **Meal panel:** meal slots, recipe names, household portions, recalculated nutrient summary, unmet targets, and source/version.
- **Limits panel:** states that the plan is not proven to prevent diabetes or lower the user's future risk and is not a clinical prescription unless reviewed by a clinician.

## 6. Meal-planning research module

The meal planner is a secondary computational contribution, not the primary diabetes-prediction result.

Use the same INDB recipes, nutrition constraints, objective, and synthetic test profiles for:

1. Discrete MILP with whole servings.
2. Continuous optimizer.
3. Real LLM baseline whose output is independently scored from the recipe database.

Report nutrient deviations, constraint violations, infeasible cases, fractional portions, diversity, runtime, cost, and repair/fallback rates. Do not call computational scores human acceptability, clinical safety, or health benefit. A separate human review study would be needed for those claims.

CARRS diet measures may inform realistic test profiles only if the released fields and sample support that use. If diet intake is not available at the needed detail, keep the profiles explicitly synthetic and state that diet is not linked to model participants.

## 7. Main architecture diagram

CARRS-1/2 cohort files  
-> governance, linkage, codebook and assay review  
-> diabetes-free baseline cohort  
-> baseline non-blood features + optional diet-measured subset  
-> longitudinal incident-diabetes labels  
-> leakage-safe training and cohort/time validation  
-> calibrated risk service  
-> user risk estimate and referral guidance

Separately:

User food preferences + clinician targets (if supplied)  
-> INDB recipe database  
-> discrete optimizer  
-> independent nutrient and exclusion validator  
-> constrained LLM explanation  
-> example meal plan

The risk score does not set the meal optimizer's clinical targets. No causal diet effect is inferred from the risk model.

## 8. What makes this a stronger paper

The main novelty candidate is not “we used a Transformer.” It is a prospective, South Asian risk-prediction study that:

- predicts incident rather than prevalent diabetes;
- compares non-blood predictors with a diet-augmented model on the same participants where diet data exist;
- evaluates on later or independent CARRS data when accessible;
- reports calibration, uncertainty, subgroup performance, and loss-to-follow-up handling;
- links the research output to a separately audited Indian recipe optimizer without claiming clinical benefit.

Existing CARRS studies have already examined dietary patterns and cardiometabolic outcomes, and incident diabetes analyses have been published. Therefore, the novelty must be the specific predictive design, validation, incremental-value analysis, and reproducible matched meal-optimizer evaluation—not the broad claim that diet is related to diabetes in South Asians.

No architecture guarantees Q1 acceptance. A Q1-strength claim depends on obtaining compatible CARRS follow-up data, enough incident events, diet coverage for the paired analysis, robust external validation, actual results, and a literature review showing the exact contribution is not already done.

## 9. Go/no-go checks before coding the final model

1. Obtain CARRS data access through the appropriate investigator/supervisor and verify the exact release.
2. Confirm baseline and follow-up diabetes definitions, lab fields, dates, visit linkage, and incident case counts.
3. Confirm which diet variables exist, at which waves, and for how many people.
4. Confirm whether CARRS-2 has a compatible released outcome follow-up. Do not promise external validation until confirmed.
5. Check missingness, sampling weights, site, clustering, and follow-up loss.
6. Freeze the analysis plan and model comparisons before final evaluation.
7. Proceed with the diet-augmented aim only if the relevant diet subset has enough cases for stable evaluation; otherwise report it as exploratory or omit it.
