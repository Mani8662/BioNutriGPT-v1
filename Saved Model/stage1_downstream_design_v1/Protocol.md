# Downstream added-value study: implementation protocol v1

Status: design and input-availability audit complete; no comparative optimizer study run. This is a forward protocol after extensive inspection of temporal model performance, not prospective preregistration.

## Question and scope
Does adding a frozen Stage 1 prediction change meal-planning feasibility, nutrient compliance and burden beyond observed patient characteristics and reported diagnoses? This engineering study cannot establish health benefit. Lowering the same model's risk score is not independent evidence of improved health.

## Sequential experiments
1. Isolate prediction-driven rule branching using saved repaired calibrated probabilities; no gradient optimization. This avoids confusing the value of predictions with the value of counterfactual search.
2. Only after a repaired-model adapter reproduces baseline predictions and gradients, compare adding counterfactual nutrient optimization. Keep this separate from experiment 1.

## Paired arms
A: age, sex, measured weight/height/BMI, reported diabetes and CVD history, and fixed reviewed nutrition rules. No Stage 1 probabilities. Do not substitute composite TARGET_DIABETES for reported history: that would introduce HbA1c outcome information into the baseline.
B: same baseline plus repaired BioNutriGPT calibrated predictions for additional rule branches.
C: same baseline plus repaired CatBoost calibrated predictions as the strong simpler comparator.
D: same baseline plus repaired FT-Transformer calibrated predictions as the reference neural comparator.
Measured BMI determines obesity-related energy rules in every arm; predicted obesity must not override a directly measured value. Known diagnoses must not be negated by a below-threshold prediction. Missing clinical history remains unknown, not negative.

## Required implementation before running
- Refactor rule selection to explicit clinical flags and separately attributed predicted flags; never represent diagnoses with invented probabilities such as 0.85/0.10.
- Define the additional action triggered by each predicted flag. Use development-only validation/calibration data for a prespecified threshold policy or explicitly exploratory fixed-grid sensitivity; do not reuse legacy raw-probability cutoffs for repaired calibrated models or select cutoffs from temporal outcomes.
- Preserve clinical exclusions for renal conditions, pregnancy/lactation and age; report unknown eligibility. No implication of clinical safety from solver acceptance.
- Recover measured anthropometry and reported histories by exact SEQN join from the temporal raw source. Log exclusions before predictions or optimizer results are examined. Activity must use a documented mapping or one fixed engineering scenario identical across arms; do not describe assumed activity as measured.
- Keep catalog version/hash, dietary modes, solver time limit/gap/seed, objective, servings, baseline dietary inputs and nutrient data coverage identical. All condition-dependent differences must be attributed in the audit trail.
- Maintain separate research outputs; no production checkpoint or rule changes during this experiment.

## Outcomes and safeguards
Primary paired engineering endpoints: certified feasibility and hard-rule violations under independently recomputed food totals. Evaluate all arms against both their assigned envelope and a common measured-profile reference envelope; compliance with different self-assigned rules is not a fair quality comparison on its own.
Secondary: nutrient deviation, changed dishes/servings, food mass, objective components, cost only if verified price data exist, runtime, solve gap, and failure reasons. Do not claim palatability without independent ratings.
Save every basket, bounds, inputs, branch reason, complete solver status, timeouts and failures. No best-case replacement or exclusion of infeasible arms. Reduced model score is descriptive only.
Use paired participant-level inference preserving repeated dietary modes; survey-design sensitivity for NHANES. Freeze the primary contrast and multiplicity policy before comparative execution. Use a deterministic small implementation pilot with separate reporting before the full study; never count pilot-selected cases as an untouched evaluation.

## Dataset boundary
Use the NHANES temporal cohort for the repaired prediction comparison because complete input features and saved predictions are available. LASI provides planning profiles but is not an external prediction-validation cohort without matching predictors and outcome ascertainment. An Indian food catalog on US profiles is a computational stress test, not Indian clinical validation.

## Acceptance gates
Exact ID/label and input hashes; source metadata coverage; calibrated prediction reproduction; rules-only arm calls no model; known-condition precedence; missing history preserved; identical solver/cost/catalog settings; independent nutrient replay; all failures retained. Then run the paired pilot and inspect implementation correctness, not tune for favourable outcome.
