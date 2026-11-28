# BioNutriGPT: End-to-End Indian Application Design

**Purpose:** Concrete product and research workflow from a person's first screen to the final app output.  
**Status:** Proposed target design, not a description of the currently implemented NHANES prototype.  
**Target data:** LASI Wave 1 for screening-model development/evaluation; Indian Nutrient Databank (INDB) for Indian recipe composition.  
**Important:** This design aims for rigorous, reviewable research. No design can guarantee Q1 acceptance or approval by another AI reviewer.

## 1. What the application does

BioNutriGPT has two user-facing functions:

1. **Screening support:** Estimate whether an eligible user may have elevated current HbA1c, using only permitted non-blood inputs. This is a screening estimate, not a diagnosis or prediction of future disease.
2. **Meal-planning support:** Build an example Indian meal plan from a verified recipe database and explicit nutrition constraints. The user may enter preferences and clinician-provided targets. The screening score does not itself set a treatment or nutrient target.

The two functions appear in one application, but the evidence behind them is separate. LASI does not contain detailed individual food diaries, and INDB does not identify which meals a LASI participant consumed. The paper must not present the meal plan as learned from a person's observed diet or as proven to reduce their glucose.

## 2. Full user journey

### Screen 1: Welcome, consent, and scope

The app explains in plain language:

- It estimates a screening result and cannot diagnose diabetes.
- A high screening result should be discussed with a qualified health professional and confirmed using an accepted clinical test.
- Meal plans are examples unless a clinician has provided and reviewed the relevant targets.
- The user can continue without creating an account. Collect no name, phone number, or precise address unless necessary for a separately approved study.
- Show the privacy notice, research-use notice, data retention period, and a clear consent choice.

If consent is not given, do not collect health information.

### Screen 2: Choose a task

Offer two buttons:

- **Check my screening result**
- **Create an Indian meal plan**

The user may do either or both. Do not imply that choosing a higher screening result automatically changes the meal prescription.

### Screen 3A: Screening eligibility and safety routing

Ask only the fields necessary to determine whether the model applies:

- Age in years.
- Whether the user has already been diagnosed with diabetes or takes glucose-lowering medication.
- Whether the user is pregnant, if applicable to the intended app population.
- Whether they have a condition requiring specialized nutrition care, such as kidney disease, or have been told to follow a clinician-specific diet.

Routing:

- **Under 45:** The LASI-trained model is outside its target population. Do not return a LASI-based score.
- **Known diabetes or glucose-lowering medication:** Do not run the undiagnosed-screening model. Offer the meal-planning path only with clinician-provided targets or refer to a clinician/dietitian.
- **Pregnancy or a condition needing specialized nutrition care:** Do not generate an individualized medical diet. Provide a clear referral for professional advice.
- **Eligible:** Continue to screening inputs.

These rules must match the final research protocol and the app's validated target population. They are not substitutes for clinical triage.

### Screen 3B: Screening inputs

The final input list must exactly match variables available, valid, and measured consistently in LASI. Candidate fields to verify against the codebook include:

- Age and sex.
- Height, weight, and waist circumference.
- Blood pressure.
- Family history and selected health history that do not reveal the target label.
- Smoking, alcohol, physical activity, or other lifestyle fields only if the codebook supports them and they are available at the same assessment.

For each measurement, ask for the value, unit, date, and source: measured at home, measured by a clinician, or recalled. The model must be validated for the source/type of measurements used in the app. If it was developed using staff-measured values and has not been evaluated with self-measurements, the app must not silently treat the two as interchangeable.

Do not ask for HbA1c, glucose, other blood results, diabetes diagnosis, or glucose-lowering medication as model predictors. The latter fields may be used for eligibility and safety routing, but they must not enter the undiagnosed-screening model.

### Screen 4: Input checks

A deterministic validation layer:

- Converts supported units to the model's required units.
- Checks dates, plausible ranges, required fields, and contradictory answers.
- Calculates derived values such as BMI only from validated height and weight.
- Shows the user any correction and asks them to confirm it.
- Marks missing values explicitly. It does not invent a clinical value.
- Returns “not enough information to estimate” if required inputs are absent or inconsistent.
- Returns “outside the validated population” if the user's profile is outside the training/evaluation domain.

Any imputation must be part of the frozen training pipeline, reported in the paper, and evaluated for its effect. The application must not improvise a different imputation method.

### Screen 5: Screening inference

The model service receives a validated, whitelisted feature vector. It returns:

- A calibrated estimate for the study's defined current HbA1c outcome, if the model passes validation.
- An uncertainty or data-support indicator.
- A threshold-based referral category only if the threshold was prespecified and supported by the evaluation.
- No causal claims and no future-disease prediction.

If the model is not calibrated or fails the defined quality gate, the app should show no probability. It should instead state that the screening estimate is unavailable and recommend routine clinical assessment.

### Screen 6: Screening result

Example response structure, with no invented numeric result:

- **Result:** “Your inputs are associated with a [model-calculated category] chance of meeting the study's elevated-HbA1c definition.”
- **What this means:** “This is a screening estimate based on people in the LASI validation population. It is not a diagnosis.”
- **Next step:** “Discuss confirmatory testing with a health professional.”
- **Factors considered:** list the relevant model inputs with neutral wording, such as “age and waist measurement contributed to this estimate.” Do not say these factors caused disease or that changing them will lower the score.
- **Uncertainty:** display the validated uncertainty/support message.
- **Data note:** state that the model was studied in Indian adults aged 45+ and give the study's limits.

Do not show a personalized probability until a LASI model has actually been trained, evaluated, calibrated, and approved for the intended research use. The current NHANES model cannot be presented as a validated Indian screening model.

### Screen 7A: Meal-plan inputs

The user chooses whether they want a general example or a clinician-directed plan.

**For a general example**, collect:

- Vegetarian, egg, or non-vegetarian preference.
- Foods the user avoids for preference or religious/cultural reasons.
- Allergies and intolerances, using a validated food-to-ingredient map.
- Preferred region/cuisine only where INDB coverage supports it.
- Number of people and meals, cooking time, and available equipment if these constraints are implemented.

**For a clinician-directed plan**, additionally collect the exact energy, carbohydrate, sodium, or other targets supplied by the clinician, with units and date. The user must confirm the values.

Do not infer an individualized calorie prescription or disease-specific limits from the risk score. Do not use an unverified budget filter: the recipe database does not by itself provide current, local prices. If a required allergy or clinical exclusion cannot be verified against all recipe ingredients, stop and request professional review rather than returning a “safe” plan.

### Screen 7B: Meal target selection

- If the user has clinician-provided targets, use those targets and display their source and date.
- If no targets are supplied, generate only a general guideline-aligned example under the explicitly chosen, non-clinical constraints. Label it as a general example, not therapy.
- Use versioned ICMR-NIN guidance for any population-level targets. Document which recommendation maps to which optimizer constraint.
- Keep hard rules (allergen exclusions and explicit contraindications) separate from soft preferences (variety, cuisine, cooking time).
- If constraints conflict or no plan exists, say so. Never weaken a hard safety rule silently.

### Screen 8: Recipe selection and optimization

The optimizer uses only recipe records that passed data checks in the selected INDB release:

1. Filter recipes using verified allergy and exclusion tags.
2. Convert serving amounts and nutrient units using deterministic code.
3. Solve for recipe servings using the fixed objective and constraints.
4. Prefer whole household servings if the study tests that method.
5. If the solver is infeasible, return an infeasibility message or a clearly labeled fallback. Do not call a fallback an optimal MILP result.
6. Independently recalculate nutrient totals from the selected recipe IDs before display.

The screening probability does not enter the objective function as a target to “drive down.” The optimizer cannot claim a food choice reverses the model's prediction.

### Screen 9: Language explanation

The LLM receives a structured, already validated result containing recipe IDs, portions, calculated nutrient totals, constraint status, and approved source text. It may:

- Organize the verified recipes into a readable meal schedule.
- Explain why each item appears, using the stated preference or target.
- Summarize limitations and suggest a clinician review where required.

It may not:

- Add a new food, portion, nutrient value, medical claim, or contraindication.
- Change an optimizer quantity.
- say the plan prevents, treats, or reverses diabetes.
- Override the validator.

A deterministic template must remain available if the LLM is unavailable or fails output validation.

### Screen 10: Final user output

Show one report with distinct sections:

**Screening section**
- The model result or the reason no result was returned.
- The target outcome and population in plain language.
- The next step, including confirmatory clinical testing when indicated.
- Uncertainty and a short explanation of contributing inputs.

**Meal-plan section**
- Meal slots, recipe names, portions, and household measures.
- Nutrient summary recalculated from the database.
- The target constraints that were met or missed.
- Substitutions only if they were independently recalculated and validated.
- Source/version and a note that the plan is an example, not a medical prescription unless reviewed by a clinician.

**Audit section**
- Date and version of the model, recipe database, constraints, and optimizer.
- A downloadable report with no unnecessary personal identifiers.
- A way for the user to report an error; feedback is not treated as clinical validation unless collected under an approved protocol.

## 3. System architecture

User interface  
-> consent and eligibility router  
-> input validation and unit normalization  
-> screening service (LASI-trained, calibrated model; no blood biomarkers in predictors)  
-> referral result and explanation

In parallel, if the user requests food planning:  
meal preferences and clinician targets  
-> recipe database with provenance and verified ingredient tags  
-> deterministic MILP optimizer  
-> independent nutrient/safety validator  
-> constrained language explanation  
-> final plan

Both branches  
-> versioned audit log and privacy-controlled storage

The risk model and meal optimizer communicate only through an approved, documented interface. A screening risk score cannot automatically create clinical nutrition targets.

## 4. Research data workflow

### LASI screening study

1. Request LASI Wave 1 main and DBS datasets and obtain the official codebook, survey weights, biomarker documentation, and assay validation.
2. Define the eligible population as LASI respondents aged 45+ and report the actual linked DBS sample, not the full LASI survey count.
3. Predefine a primary HbA1c outcome compatible with the DBS assay. If a clinical cutoff is not validated for the assay, call the endpoint elevated HbA1c rather than diabetes.
4. Exclude HbA1c, glucose, diabetes diagnosis, and glucose-lowering treatment from prediction inputs where they reveal the outcome.
5. Compare survey-weighted logistic regression, strong tree-based models, and the Transformer on identical participants/features.
6. Validate by held-out states or state groups using a prewritten split. This is geographic internal-external validation within LASI, not validation on a separate cohort.
7. Report AUROC, AUPRC, calibration, Brier score, threshold-specific referral performance, design-aware confidence intervals, and sufficiently powered subgroup results.
8. Evaluate measurement-source shift before claiming that a model developed on staff-measured anthropometry is ready for home-entered values.

### INDB meal-planning study

1. Inspect and version the INDB workbook, serving definitions, nutrient units, source metadata, and use terms.
2. Verify ingredient and allergy mappings. Mark unsupported safety fields as unknown rather than safe.
3. Compare discrete MILP, continuous optimization, and an actual LLM using the same recipes, constraints, objective, and profiles.
4. Use synthetic profiles only as standardized test cases and label them clearly.
5. Report feasibility, constraint violations, nutrient deviation, fractional servings, recipe diversity, runtime, cost, and failure/fallback rates.
6. Do not infer human acceptability, health benefit, safety, or affordability from computational metrics.

## 5. Model use and product readiness gates

The application may show a risk estimate only after all of these are true:

- LASI access and target variables are verified.
- The DBS outcome definition and assay are justified.
- Input leakage is excluded.
- Survey design and DBS sample selection are handled.
- The Transformer and simpler baselines are compared fairly.
- Calibration, subgroup performance, uncertainty, and geographic validation meet prespecified criteria.
- The app measurement method is close enough to the study measurement method or has been separately validated.
- The model card states population, intended use, exclusions, performance, limitations, and update policy.

The meal-planning output may be shown only if:

- Recipe nutrient calculations and portions pass data checks.
- Allergies and hard exclusions are checked against ingredient-level data.
- Infeasibility and fallback behavior is tested.
- Every displayed nutrient value is calculated independently of the LLM.

Until clinical external validation and appropriate human review have occurred, label the product as a research prototype. Do not claim clinical benefit or autonomous nutrition care.

## 6. Reviewer challenge checklist

An independent reviewer should be able to answer “yes” to these questions from the manuscript and released artifacts:

- Is the exact novelty claim narrower than “AI personalization,” and is it compared with Indian and international prior work?
- Are LASI eligibility, data access, linkage, DBS sample size, outcome definition, weights, and exclusions transparent?
- Are HbA1c and other label-revealing variables kept out of the screening predictors?
- Are all models evaluated on the same records, inputs, weighting, splits, and metrics?
- Are geographic splits described honestly as internal validation?
- Are calibration, uncertainty, clinically meaningful referral thresholds, and subgroup limitations reported?
- Is the meal optimizer compared with matched baselines on the same Indian recipes and constraints?
- Are synthetic cases, missing food fields, and unsupported claims clearly labeled?
- Does the output avoid implying that a risk score is a diagnosis or that a generated meal plan changes clinical outcomes?
- Can another researcher reproduce the data processing, model evaluation, and optimizer scoring from versioned materials?

These answers can strengthen a submission, but neither AI reviewers nor journal editors can be guaranteed to approve it. Q1 depends on actual results, prior-art novelty, journal scope, and independent peer review.

## 7. Current project versus target design

The repository currently contains NHANES-based Stage 1 data and a prototype meal-planning pipeline. It does not contain LASI participant data. The existing Stage 2 score-perturbation workflow must not be described as a causal diet intervention. The Indian LASI workflow above is a proposed redesign: it becomes an empirical result only after data access, analysis, and reproducible evaluation.

## 8. Sources

- LASI Wave 1 data and request information, IIPS: https://www.iipsindia.ac.in/node/6702
- LASI Wave 1 instruments and measures, IIPS: https://www.iipsindia.ac.in/content/lasi-wave-i
- Indian Nutrient Databank, Anuvaad: https://www.anuvaad.org.in/indian-nutrient-databank/
- ICMR-NIN, Dietary Guidelines for Indians 2024: https://nin.res.in/dietaryguidelines/pdfjs/locale/DGI_2024.pdf
- ICMR-INDIAB dietary profiles and metabolic-risk analysis, Nature Medicine (2025): https://www.nature.com/articles/s41591-025-03949-4
- TRIPOD+AI: https://www.bmj.com/content/385/bmj-2023-078378
- PROBAST+AI: https://www.bmj.com/content/388/bmj-2024-082505
