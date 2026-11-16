# BioNutriGPT: Indian secondary-data research design

**Status:** Student-feasible proposed study. This is a research plan, not a claim that the analyses have already been completed.  
**Design date:** 15 November 2026  
**Data strategy:** Existing Indian survey data plus an Indian recipe composition database. No new participant recruitment or repeated measurements are required for this first paper.

> **Superseded for the selected cohort:** This earlier version uses LASI. The selected architecture is now CARRS; use [BioNutriGPT_CARRS_Architecture.md](BioNutriGPT_CARRS_Architecture.md) as the active design. The LASI-specific proposal is retained only as a historical alternative.

## 1. Decision: use two Indian data sources for two different jobs

Use the **Longitudinal Ageing Study in India (LASI), Wave 1** for participant-level health and screening-model evaluation, and the **Indian Nutrient Databank (INDB)** for meal planning.

These sources complement each other, but they are not linked:

| Source | What it contributes | What it does not provide for this study |
|---|---|---|
| LASI Wave 1 | Indian adults aged 45+, demographics, health history and behaviours, food-security measures, measured height/weight/waist, blood pressure, and a separate dried-blood-spot (DBS) dataset that includes HbA1c | It is not a detailed individual food diary or nutrient-intake survey. The DBS participants are a subset; exact linked sample size, assay documentation, variables, and weights must be checked. |
| INDB | Nutrient composition for 1,014 common Indian cooked recipes, with values per 100 g and per serving | It does not say which recipes a LASI participant actually ate. It does not supply each participant's preferences, acceptance, clinical response, or current local food prices. |

**Do not join LASI people to INDB recipes as if their meals were observed.** The model and meal optimizer can be evaluated in the same paper as two application modules, but current data do not support a claim that a generated meal plan is personalized to a participant's measured diet or proven to improve that participant's glucose.

IIPS says LASI Wave 1 represents 73,396 adults aged 45 and above across Indian states and union territories. Access is for legitimate academic research, requires valid identity proof, and is requested through IIPS; its page says requests are normally reviewed within 24-48 working hours. That is a review estimate, not a guarantee of immediate access.

## 2. Focused, answerable research question

> Among LASI Wave 1 participants aged 45 and above, how accurately and consistently can models that use no blood-test inputs identify a prespecified HbA1c-defined elevated-glycaemia outcome, and how does an auditable whole-serving meal optimizer perform on Indian recipes from INDB under the same nutrition constraints as continuous optimization and a real LLM?

The first clause is the **primary scientific study**. The meal-optimizer comparison is a **secondary system-methods study**. This keeps the main result clear and makes the evidence limits visible.

### What the paper can and cannot say

- It can study **cross-sectional identification of measured glycaemia** at the LASI assessment.
- It cannot call that **prediction of future diabetes**, prevention, or treatment benefit. LASI Wave 1 is not repeated-measure follow-up for this question.
- “No blood-test inputs” describes the model's predictors. HbA1c is used as the reference label during development and evaluation. In a future screening workflow, a high model score would prompt confirmatory clinical testing; the model would not diagnose diabetes.
- Do not call the full study “non-invasive” without qualification. Say **blood-test-excluded screening model** or **model using no blood biomarker inputs**.
- Food-security/access variables are not a substitute for an individual's food intake. The model must not be described as learning nutrient effects from LASI.

## 3. Working title and defensible contribution

### Working title

**Blood-Test-Excluded Screening for Elevated Glycaemia and Auditable Indian Meal Optimization: A Secondary-Data Study Using LASI and INDB**

### Candidate contribution, pending a documented literature search

> We evaluate blood-test-excluded identification of a prespecified HbA1c-based outcome in a nationally representative Indian ageing survey, using survey-aware geographic validation, calibration, and subgroup analysis; we separately benchmark discrete whole-serving optimization over a documented Indian recipe database against matched continuous-optimization and LLM baselines.

This is more specific than “AI for personalized nutrition.” The contribution is not simply using a Transformer, LLM, or optimizer. The claims to investigate are the Indian population setting, separation of screening inputs from the laboratory reference outcome, transparent survey-aware evaluation, and a matched comparison of discrete meal planning using Indian recipes.

A 2025 ICMR-INDIAB analysis has already examined dietary profiles and metabolic risk in India. Therefore, broad claims such as “first Indian AI study of diet and diabetes” or “first study linking Indian diet to metabolic risk” are not defensible. A focused search must also compare existing Indian diabetes-screening models and LLM/optimization meal-planning studies before the manuscript claims novelty.

## 4. Data acquisition and cohort definition

### LASI

1. Request both the LASI Wave 1 main/state-UT dataset and separate Wave 1 DBS dataset through the IIPS data page. Attach the required identity proof and state the academic research purpose. This is a data request; the files should not be represented as already downloaded or approved.
2. Obtain questionnaires, codebook, survey-weight documentation, biomarker manual, and DBS assay/validation documentation before defining final variables.
3. Restrict the primary analysis to LASI respondents aged 45 and above. State this population restriction because LASI also interviews some spouses regardless of age.
4. Draw a participant-flow diagram: full eligible sample, records with a valid linked DBS/HbA1c measure, exclusions, missing predictors, and final analysis set. Do not quote the 73,396 survey count as the HbA1c sample size.
5. Confirm the correct biomarker or subsample weight, strata, and primary sampling unit variables. Use them where supported. Describe limitations if the DBS documentation does not support a particular weighted analysis.

### Outcome and predictors

- **Primary outcome:** a prespecified elevated-glycaemia/diabetes label based on the LASI DBS HbA1c measurement and documented assay method. Confirm that a clinical threshold is valid for this DBS method before fixing the cutoff. If not validated, use the study's documented definition and call it elevated HbA1c rather than clinical diabetes.
- **Secondary outcome:** a published, codebook-compatible definition combining laboratory measurement with self-reported diagnosis or glucose-lowering medication, if variables and timing support it.
- **Predictors:** only inputs collectable without a blood test at screening, such as age, sex, anthropometry, blood pressure, relevant family/health history, and documented lifestyle factors.
- Exclude HbA1c, other blood biomarkers, glucose results, diabetes diagnosis fields, and glucose-lowering medication fields from predictors when they are part of or reveal the outcome. This prevents target leakage.
- Treat food security as an optional contextual predictor, not diet intake. Test whether including it changes calibration and subgroup performance.

Before modeling, verify that LASI has each proposed variable with the needed coding and missingness. Freeze the cohort, outcome, predictors, exclusions, and analysis plan before opening final test results.

## 5. Model and evaluation plan

### Models

Compare all models on the same participants, same predictor set, same target definition, and same survey-weighting/evaluation procedure:

1. Survey-weighted logistic regression as the interpretable baseline.
2. A published risk score only if all required inputs and an appropriate Indian implementation are available.
3. One or two strong tree-based tabular models, such as XGBoost or CatBoost.
4. The project's Transformer only if it can be trained on the available sample and adds stable value over simpler methods.

Do not presume that the Transformer is the contribution. If a simpler model performs as well or calibrates better, report that and use it for the screening prototype.

### Validation and reporting

- Use geographic internal-external validation: hold out states or state groups, train on the remaining areas, and repeat where sample counts allow. Decide the grouping after reviewing DBS case counts; do not choose an easy split after seeing results.
- This is **geographic validation within LASI**, not external validation in a separate cohort.
- Tune models only within training data. Keep held-out geography untouched until final evaluation.
- Report AUROC and AUPRC, sensitivity and specificity at prespecified referral thresholds, calibration plot/slope/intercept, Brier score, and confidence intervals. A threshold should correspond to a plausible action such as referral for confirmatory testing.
- Use a survey-design-aware or cluster-aware uncertainty method. Ordinary row-level bootstrap confidence intervals can be too optimistic for a clustered survey.
- Report results by sex, age group, region, urban/rural status, and socioeconomic group where sample sizes allow. Include uncertainty; do not call a model fair just because average AUROC is similar.
- Use decision-curve analysis only if its referral action and threshold range are explicitly defined. It cannot establish that the model improves health outcomes.
- Explain predictions as model associations, not causes. Do not say that changing an explanatory feature will reduce an individual's diabetes risk.

Follow TRIPOD+AI for reporting prediction-model development and evaluation, and use PROBAST+AI concepts to assess risk of bias. These guidelines do not compensate for missing external validation.

## 6. Indian meal-planning module

### Data and purpose

Use INDB as the recipe-nutrient source after reviewing the spreadsheet columns, units, serving definitions, recipe coverage, and reuse terms. It lists 1,014 recipes with per-100-g and per-serving nutrient values and reports that recipe ingredients draw on IFCT 2017, Indian food tables, CoFID, and USDA FoodData Central.

The meal module is a **computational planning demonstration**, not a clinical intervention. It may produce example meal patterns under clearly stated nutrition constraints. It must not imply that the optimizer learned which foods prevent diabetes from LASI or that its plans improve HbA1c.

### Matched comparison

Compare:

- the project's discrete MILP using whole servings;
- a continuous optimizer using the same recipes, target, objective, and constraints; and
- a real LLM meal-planning baseline using those same recipes and constraints.

Use a fixed, published set of test profiles and constraints. If profiles are synthetic, label them synthetic and do not describe them as LASI patients. Do not give the LLM numerical authority over nutrient totals; independently calculate every method's output from the same recipe table.

Report hard safety/constraint violations and infeasible cases, nutrient-target deviation, whole-serving realism, recipe diversity, runtime, and API cost where available. Report failure, repair, and fallback rates. These are computational measures. Without dietitian or patient review, do not claim palatability, cultural acceptability, clinical safety, or practical acceptability. Without measured local prices, do not claim affordability. Use ICMR-NIN guidance for stated population-level constraints, not as proof that one plan fits every person.

## 7. Architecture aligned to the data

1. **LASI analysis layer:** versioned cohort construction, outcome definition, survey design, and data checks.
2. **Screening model:** blood-test-excluded predictors to an estimated probability of the prespecified current HbA1c outcome, with calibration and uncertainty.
3. **Referral explanation:** plain-language factors and a clear recommendation for confirmatory clinical assessment when appropriate. The output is not a diagnosis.
4. **Recipe layer:** versioned INDB recipe records with source, units, serving size, and estimated or missing values marked.
5. **Optimizer:** deterministic nutrient calculations and constraints; whole servings when feasible; explicit infeasibility and fallback messages.
6. **Language layer:** explains the verified plan and sources but cannot invent foods, serving quantities, nutrient totals, or medical advice.
7. **Audit trail:** save data version, model version, input fields, recipe IDs, constraints, solver result, and validator result.

The LASI screening model and meal optimizer are connected at the software-workflow level, but the available data do not establish a participant-level relationship between measured risk and actual dietary intake or response. The manuscript must state this.

## 8. Practical work plan

### First: start the data request

Prepare the IIPS request for LASI Wave 1 main and DBS files, attach valid identity proof, and state that the project is academic secondary analysis of non-blood predictors of a documented HbA1c outcome. Confirm current access conditions on the IIPS page. Do not wait to write code against guessed column names.

### In parallel: prepare analysis scaffolding

- Download the official questionnaires and codebook.
- Make a variable dictionary with source, label, unit, missing codes, time point, and whether each field is allowed as an input or outcome.
- Draft the participant-flow logic and a one-page statistical analysis plan.
- Inspect existing project reports; label NHANES-only results as exploratory US-data results and do not combine them with LASI as one cohort.
- Review the INDB workbook and decide whether its actual fields support the current optimizer. Replace or augment the local IFCT-only recipe table only after checking units, serving definitions, and reuse conditions.

### After files arrive

- Reconcile links, design variables, outcome assay, and missingness before modeling.
- Run logistic and tree baselines first; then benchmark the Transformer on the same folds.
- Run held-out-geography evaluation with calibration and subgroup reporting.
- Run the matched meal-planning benchmark on INDB recipes.
- Update the manuscript only from saved, reproducible results.

If LASI access is delayed or the necessary HbA1c/weight variables are unavailable, the project cannot honestly report a LASI model result. Do not substitute unverified numbers. Continue the INDB optimizer study as a separate computational paper or wait for proper data access.

## 9. Novelty and journal-level reality check

This plan is much more feasible than enrolling a cohort or running a six-month trial, and it gives the project a defensible Indian setting. It does **not guarantee a Q1 publication**. One cross-sectional survey, no linked individual diet records, no independent external cohort, and no patient/dietitian evaluation limit the strength of the integrated clinical claim. A rigorous study could be a credible journal paper, but a strong Q1 target would normally need a clear methodological advance and stronger independent or human validation.

The honest first-paper claim is narrower: careful Indian survey-based evaluation of screening without blood-marker inputs, paired with a separately and fairly benchmarked Indian recipe optimizer. If the literature review shows that exact combination is not new, narrow or change the question rather than claiming priority. The strongest improvement available without recruiting participants is an independent existing Indian cohort with comparable variables and validated labels; its access and compatibility must be verified before promising external validation.

## 10. Boundaries for the manuscript

Do not claim:

- future diabetes prediction from LASI Wave 1 cross-sectional data;
- actual Indian dietary intake from LASI food-security answers;
- that LASI participants ate the INDB recipes generated by the planner;
- that planned meals lower HbA1c or prevent diabetes;
- clinical diagnosis from a blood-test-excluded score;
- Transformer superiority without matched held-out results;
- human acceptability, safety, or affordability from software checks alone;
- external validation from a geographic split within LASI.

## 11. Sources

- LASI Wave 1 data and request information, International Institute for Population Sciences: https://www.iipsindia.ac.in/node/6702
- LASI Wave 1 instruments and measures, International Institute for Population Sciences: https://www.iipsindia.ac.in/content/lasi-wave-i
- Indian Nutrient Databank overview and methodology: https://www.anuvaad.org.in/indian-nutrient-databank/
- Vijayakumar A, Dubasi HB, Awasthi A, Jaacks LM. Development of an Indian Food Composition Database. Current Developments in Nutrition. 2024. Available from the INDB publication page above.
- ICMR-NIN, Dietary Guidelines for Indians 2024: https://nin.res.in/dietaryguidelines/pdfjs/locale/DGI_2024.pdf
- Anjana RM et al. Dietary profiles and associated metabolic risk factors in India from the ICMR-INDIAB survey-21. Nature Medicine. 2025. https://www.nature.com/articles/s41591-025-03949-4
- TRIPOD+AI: https://www.bmj.com/content/385/bmj-2023-078378
- PROBAST+AI: https://www.bmj.com/content/388/bmj-2024-082505
