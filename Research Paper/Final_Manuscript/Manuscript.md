# BioNutriGPT: An Auditable Pipeline for Cardiometabolic Screening and Constraint-Preserving Conversational Meal Planning

## Abstract

**Background:** Conversational meal planning requires persistent preferences and reliable numerical verification as users request changes.

**Objective:** To develop and evaluate an auditable pipeline linking cardiometabolic screening, nutritional rules, discrete meal optimization and controlled conversational revision.

**Methods:** Six prediction models were evaluated with separate training, validation and calibration partitions and a 7,004-participant NHANES temporal cohort. Stage 2 evaluation included matched target-generation methods, 600 component-ablation tasks, frozen rescoring by three models and 800 input-sensitivity tasks. Separate meal-planning and conversational studies were analyzed separately. A selected LASI profile evaluation assessed planning rather than external prediction validity.

**Results:** BioNutriGPT AUROCs were 0.8107 for diabetes, 0.7467 for obesity and 0.8288 for CVD, without overall superiority over competitive baselines. In the Stage 2 ablation cohort, removing joint projection reduced numerical acceptance from 100/100 to 8/100; removing the model objective reduced overall acceptance from 62/100 to 49/100. Comparator models agreed in mean obesity/CVD score direction but disagreed on diabetes. All 800 sensitivity outputs met numerical constraints, although small input changes sometimes substantially altered nutrient targets. The controlled conversational evaluation passed 76/78 guarded turns; the matched 24-turn comparison yielded 24/24 guarded versus 2/24 unguarded Qwen protocol passes.

**Conclusions:** Constraint enforcement and explicit state management support auditable computational planning. Prediction accuracy, model-score improvement and numerical feasibility are distinct endpoints; none establishes clinical benefit. Sensitivity, model disagreement and selected-case evaluation limit deployment claims.

**Keywords:** nutrition decision support; conversational systems; mixed-integer optimization; provenance; tabular Transformer; Indian meals; reproducibility.

## 1. Introduction

Personalized meal planning combines nutritional accounting, food preferences, portion selection, and repeated requests for change. A user may ask to remove a dish, retain a preferred food, reduce the number of dishes, or inspect the nutrients in an updated plan. These operations require consistent numerical accounting and persistent state. A fluent answer alone does not establish that the updated basket satisfies its declared requirements.

BioNutriGPT addresses this engineering problem through four connected stages. A screening model describes prevalent cardiometabolic outcomes in the NHANES study setting. A nutrient-rule engine supplies an explicit planning envelope. A discrete optimizer selects food portions. A conversational service explains the resulting plan and routes revisions back through optimization and validation. Observed patient inputs remain separate from proposed and accepted meals.

The principal research question is: **Can a stateful, tool-backed nutrition assistant support repeated meal-plan revisions while preserving declared constraints and previously recorded restrictions?** Secondary questions concern the computational trade-offs of whole portions and the usefulness of the screening component relative to simpler models. Clinical benefit is outside the present evaluation.

### 1.1 Related work and scope of contribution

Conversational nutrition recommendation already has substantial precedent. ChatDiet combines personal and population information through an orchestration framework to produce personalized food recommendations [1]. Therefore, adding a chatbot to nutrition recommendations is not, by itself, a defensible novelty claim.

Integer and goal-programming approaches to meal planning also exist. A recent preprint explicitly studies serving granularity and compares mixed-integer goal programming with continuous and rounded alternatives [2]. Whole portions, HiGHS optimization, and nutrient-deviation objectives must therefore be treated as established methods rather than new algorithms.

Tabular Transformers likewise belong to an established methodological family. Comparative work on tabular deep learning demonstrates the importance of strong baselines and consistent evaluation protocols [3]. The use of a Transformer or mixture-of-experts component does not establish predictive advantage without direct experimental support.

The contribution investigated here is the **combined, testable revision workflow**: catalog-grounded answers; explicit separation of observed, proposed, and accepted state; persistent food restrictions; re-optimization after requests; independent acceptance validation; and auditable plan history. The present related-work comparison establishes overlap, not an exhaustive literature search or a claim that this combination is the first of its kind.

## 2. Methods

### 2.1 Study design and evidence boundaries

This is a retrospective prediction and computational systems study. We distinguish four evidence layers:

1. **Prediction evaluation:** saved Stage 1 results on the NHANES temporal holdout.
2. **Separate system experiments:** optimizer and conversational comparisons performed on distinct food catalogs.
3. **Implementation verification:** a descriptive evaluation of the 49-entry catalog and regression checks .
4. **LASI real-profile pilot:** adapted rule-based planning on 100 Indian profiles and a separately audited conversational subset.

These layers are not pooled. In particular, software regression checks are not counted as patient cases, and results from different food catalogs are not combined into a pooled success rate. TRIPOD+AI informs the prediction-reporting requirements; completed checklist adherence is not asserted [4].

### 2.2 Cohorts, outcomes, and predictors

The analytic cohort includes adults aged at least 20, consistent with the CVD questionnaire scope. Development data from 2007–2016 contain 24,136 participants; the temporal evaluation cohort from 2017–March 2020 contains 7,004. Preprocessing was fitted only on training participants. Validation supports model selection; calibration participants are separate. This temporal period had already been inspected and is not untouched prospective validation. Source lineage starts at available merged raw files, not a reconstruction of all eligible NHANES participants.

| Partition | Participants | Diabetes-positive | Obesity-positive | CVD-positive |
|---|---:|---:|---:|---:|
| Training | 16,894 | 2,616 | 6,427 | 1,773 |
| Validation | 3,621 | 562 | 1,377 | 380 |
| Calibration | 3,621 | 561 | 1,377 | 379 |
| Temporal evaluation | 7,004 | 1,301 | 3,057 | 845 |

Primary diabetes is reported diabetes or HbA1c at least 6.5%; a negative requires observed HbA1c below threshold and an explicitly eligible questionnaire response. Glucose is excluded from the primary phenotype and retained in sensitivity counts because ascertainment differs across fasting subsamples. CVD is positive if any of five disease responses is positive, and negative only when all five are explicitly negative. Missing components cannot establish a negative. Obesity uses BMI at least 30 kg/m² and remains exploratory. These operational labels are not diagnoses made by the application.

The model has 87 predictors: five lipid measurements or derived lipid measures; 46 dietary features comprising 39 nutrient fields and seven derived ratios; 19 lifestyle features; and 17 demographic or missingness features. HbA1c and fasting glucose are not predictors. Because lipid inputs require laboratory measurements, the system is described as **non-glycemic laboratory-assisted screening**, not non-invasive screening. The complete predictor list is provided in Appendix A.

### 2.3 Architecture and screening model

```mermaid
flowchart TD
    A[Structured user values or TXT input] --> B[Stage 1: Cardiometabolic screening]
    B --> C[Stage 2: Nutrient rules]
    C --> D[Stage 3: Whole-portion meal optimization]
    F[Food catalog and nutrient provenance] --> D
    D --> V[Independent basket validation]
    V --> E[Stage 4: Plan explanation and controlled chat]
    E --> R[Revision request and saved preferences]
    R --> D
    E --> P[Accepted plan versions and history]
```

Stage 1 tokenizes the 87 scalar predictors into 128-dimensional representations with modality embeddings. A three-layer Transformer encoder uses four attention heads per layer. Learned modality-query pooling produces biochemical, dietary, lifestyle, and demographic vectors. A four-expert sparse layer uses top-two routing, followed by outcome-specific prediction heads.

Five architecture variants across five seeds were evaluated. The adjusted primary contrasts did not establish an overall benefit of the added components; modality pooling and expert routing are not claimed as independently validated innovations.

The matched comparison evaluates logistic regression, histogram gradient boosting, CatBoost, an MLP, BioNutriGPT and FT-Transformer under matched partitions, four configurations per family, and reserved-set calibration. Equal configuration counts do not imply equal computation.

### 2.4 Stage 2: Nutritional planning rules

Stage 2 combines patient metadata, energy calculations, nutrient bounds, and model-based hypothetical dietary search. A counterfactual candidate changes proposed dietary inputs for computational analysis; it is not a measured intervention or a causal estimate of disease reduction. The chatbot does not overwrite observed intake or reinterpret a proposed food swap as an observed clinical improvement.

Of 39 nutrient/exposure fields, 30 are active planning fields, seven are report-only, and two—caffeine and alcohol—are tracked separately as exposures. Total sugar is not treated as free sugar; total folate, synthetic folic acid, food folate, and dietary folate equivalents are distinguished. Numerical search caps are not described as tolerable upper intake levels.

Several rules remain project assumptions, including parts of the protein, carbohydrate, sodium, potassium, and cholesterol policy. References in the rule metadata identify context; they do not imply that a clinical authority endorsed every project threshold. The implementation uses a mixture of reference sources and project choices and is not represented as complete ICMR-NIN guideline compliance [6].

The intended computational scope is adults. Pregnancy, lactation, pediatric prescribing, renal disease management, and medication adjustment are outside the chatbot's supported scope.

### 2.5 Food composition and missingness

It contains 49 records: 20 INDB recipe variants and 29 USDA single-food proxies. INDB is a published recipe-composition resource, while USDA records provide additional food-composition data [7,8]. Selected ingredient-level gap filling also uses documented food matches from CoFID [9]. These sources do not make every proxy identical to an Indian household preparation.

| Catalog characteristic | Current snapshot |
|---|---:|
| Total records | 49 |
| INDB recipe variants | 20 |
| USDA single-food proxies | 29 |
| Vegetarian records | 43 |
| Non-vegetarian records | 6 |
| Records without declared active-field gaps, before other filters | 43 |
| Records with at least one known allergen tag | 20 |
| Records retaining possible unknown allergens | 49 |

All 49 records must not be described as laboratory-validated cooked Indian dishes. Standardized gram portions are modeling units, and agreement with a household bowl, recipe yield, or serving convention has not been established for every entry. Calculated ingredient-level nutrients and matched proxies are distinguished from directly reported recipe values through provenance metadata.

The current catalog still has declared active-field gaps in synthetic folic acid, folate equivalents, alpha-tocopherol, retinol, or vitamin B12 for a subset of recipes. Foods missing a required active nutrient are excluded from a solve rather than assigned zero. Passing computational checks establishes consistency with the selected catalog values, not the accuracy of those values under actual preparation.

### 2.6 Stage 3: Discrete planning and acceptance

For food i, let q_i be its integer number of standardized portions, z_i indicate selection, and a_ki denote nutrient k per portion. Selected foods satisfy q_i in {0,1,2,3} with linking constraints z_i <= q_i <= 3z_i. A plan contains four to eight active foods unless a supported tighter dish limit is requested. Dietary-mode-specific food-category and serving constraints further restrict the basket.

For active nutrient k, the accepted basket must satisfy:

`L_k <= sum_i(a_ki * q_i) <= U_k`.

The implementation retains deviation variables from a goal-programming formulation, but fixes active nutrient-bound deviations to zero during the current production solve. Report-only fields and separately tracked exposures are not converted into active adequacy requirements.

Within the feasible set, the objective penalizes food-selection cost and absolute energy-target deviation. For revisions, an additional term penalizes `sum_i |q_i - q_i_previous|`, encouraging smaller changes. The selection-cost parameter is an engineering preference and is not assumed to be a verified retail price. Preferred foods influence optimization without authorizing violations of protected rules.

Every proposed or accepted basket passes a separate validator that recalculates nutrients and checks relevant structural requirements. A rejected candidate is not substituted as a successful plan. Failure responses retain the previously accepted plan and distinguish data insufficiency, unavailable locked foods, solver limits, and other reported conflicts.

### 2.7 Stage 4: Grounded conversation and persistent state

The controlled chat service uses explicit backend tools for plan inspection, nutrient totals, revision proposals, and conflict explanations. Intent detection is primarily deterministic. It is not described as a fully autonomous LLM agent. The separate report-compilation path can use a local Qwen model under schema and quantity checks; its behavior is evaluated separately from the controlled chatbot.

Patient records, proposals, and accepted versions are separate objects. Revision requests retain existing exclusions, preferred foods, daily food-quantity locks, ingredient exclusions, and dish limits. Persistent-session acceptance requires an exact match to a server-stored pending proposal for the current version, followed by independent validation. Accepting a revision creates a new version; rollback restores a prior basket and its plan preferences without deleting later patient-level allergy declarations.

Allergy questions do not, by themselves, update the patient record. Explicit declarations create or update persistent allergy state and invalidate older pending proposals. Foods with documented matching allergens are excluded from subsequent revisions. Because allergen metadata are incomplete, warnings remain visible after numerical acceptance and rollback. The system does not certify a plan as allergen-free.

Optional unconstrained LLM rewriting of chatbot answers is disabled. A previous number-preservation check could not prevent semantic changes, including removal of caveats. Verified deterministic answers therefore remain authoritative. This restriction does not remove the separately implemented report compiler.

The frontend supports proposal review and an optional automatic application setting. When enabled, that setting applies only server-verified proposals; studies must record whether it was enabled. Daily food-quantity locks do not establish that an entire meal slot remains unchanged after rescheduling.

### 2.8 Evaluation and analysis

For Stage 1, we report the saved unweighted temporal-holdout AUROC, average precision, and Brier score. A subsequent survey-weighted sensitivity analysis uses WTDRD1PP and stratified delete-PSU jackknife uncertainty, with an independent R survey reproduction; results are reported separately from unweighted primary estimates. Participant-level bootstrap intervals do not, by themselves, account for NHANES stratification and clustering.

LLM-related reporting is mapped separately using TRIPOD-LLM; the working reporting map does not assert checklist completion [10].

The matched comparison fits recalibration on the separate 3,621-person calibration partition and reports raw and calibrated results. Each family uses four configurations and seed 42: an equal-count, not equal-compute, tuning budget. Paired participant bootstrap uncertainty uses 20,000 resamples and adjustment across 20 primary contrasts. These intervals are conditional on fitted models, not survey-design uncertainty. The application uses a frozen checkpoint, matching preprocessing and reserved-set calibration. Its 0.5 operating threshold is descriptive research routing, not a clinically validated treatment policy. Downstream studies retain their declared model, rule and catalog configurations; results from distinct experiments are not pooled.

Separate preliminary optimizer and conversational comparisons use 29-entry and 31-entry catalogs respectively. They are reported separately from the 49-entry planning evaluation and are not pooled.

### 2.9 LASI real-profile computational evaluation

The new evaluation uses LASI Wave 1 anthropometry, demographics and reported diagnoses [11]. The extraction summary reports 66,859 merged biomarker/individual records and 65,368 eligible records after age, anthropometric, BMI and diagnosis-completeness filters. These are archive-specific analytic counts, not the official total LASI recruitment count. A deterministic sample of 100 unique profiles contains 25 diabetes-only, 25 obesity-only, 25 CVD-only and 25 multimorbid profiles, with 52 male and 48 female profiles across 29 states/UTs. Eligibility includes ages 18–100, measured height 120–210 cm, weight 30–180 kg and BMI 14–55. Disease responses must have valid yes/no codes. Diabetes is self-reported; obesity strata use measured BMI at least 30; CVD combines the selected heart-disease and stroke items. National disease-distribution estimates are not inferred: the benchmark is deliberately disease- and sex-balanced and unweighted. The sampling frame of LASI principally concerns older adults and spouses, rather than all Indian adult age groups.

The benchmark adapter computes energy from observed age, sex, height, weight and a coarse activity-category mapping. It invokes the production nutrient-bound function directly, assigns disease-routing values of 0.85 for positive status and 0.10 for negative status, and sets sodium/potassium baselines to 2800/2600 mg. Other unavailable nutrient baselines are rule-derived. These values are assumptions, not measured dietary intake or Stage 1 probabilities. This evaluates adapted Stage 2 bounds with Stage 3 and Stage 4; it does not execute the full Stage 2 counterfactual optimization or externally validate Stage 1.

Stage 3 applies three dietary modes to each of the same 100 profiles, yielding 300 solves with two-second limits. The final conversational study, , uses 36 distinct participants: nine per disease stratum and 12 per dietary mode, with three participants in every mode-by-stratum cell. Each guarded variant is evaluated on 78 turns. A 12-participant subset contains one participant per cell and 24 turns. Participants are selected from profiles with an accepted starting plan; this conditions the conversational evaluation on initial feasibility. The same frozen participants, starting baskets and request sequences are retained across comparisons. The earlier diabetes-only and adaptive-prompt LASI experiments are retained as historical development runs, not pooled into the final results.

The  Qwen prompt supplies the same complete-data food eligibility, all 30 active nutrient coefficients at round-trip precision, bounds, objective weights, cumulative restrictions and canonical post-checks. Six catalog entries with missing active values are excluded; prompt catalogs contain 37 foods for vegetarian mode and 43 for the other modes. Food selection costs are engineering parameters, not verified retail prices. The post-checks use 5% energy tolerance, protein 10–35%, carbohydrate 45–65% and fat 20–35% of energy, the recorded baseline sodium/(potassium+1) ratio, and both fiber checks. These are the implementation's acceptance rules, not a claim that every threshold is clinically validated.

The assistant solves revisions under a two-second limit; a separate three-second oracle distinguishes feasible requests, reported proven infeasibility and unresolved search. Timeouts count as protocol failures. Protocol success includes valid proposals, justified refusals, grounded answers and exact restoration as applicable; it is reported separately from proposal production. Nutrient-answer checks use number proximity within 5% relative or 0.3 absolute tolerance and are not exhaustive semantic hallucination detection. The comparison tests a deterministic tool-backed assistant against a specified solver-free LLM configuration, not two otherwise identical language agents.

The change-penalty study reports both 29 trajectory-level successful pairs and a restricted analysis of 25 successful pairs with identical preceding basket and recorded input restrictions. Restricting to successful pairs is a conditional analysis, not a randomized causal estimate. Exploratory 95% intervals use 1,000 participant/scenario-level bootstrap resamples with seed 42; paired comparisons resample the same scenario IDs. Whole-scenario Wilson bounds are also provided under a working independent-binomial model because all-success empirical bootstrap intervals can degenerate. These intervals do not support survey-weighted national inference.

Actual GPU execution used an RTX 3050 with 4-bit NF4 Qwen 2.5 3B, model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy decoding and a 512-token output cap. Full prompts, raw outputs, validation certificates, input states, per-turn journals and completed-stage checkpoints are retained locally. Fourteen source/input hashes were unchanged across execution. Warm-model response timing excludes model loading and Qwen prompt construction; GPU work is synchronized. The final checkpointed run took 1,548.32 seconds including benchmark work. A prior attempt failed at report export; its incomplete summaries are archived separately and excluded from the measured final dataset.

### 2.9 Stage 2 component and sensitivity studies

The evaluated Stage 2 search uses 50 Adam steps (learning rate 0.05), a weighted model-score objective (2 CVD + 2 diabetes + obesity), a 50-weight nonincrease penalty, an unscaled squared dietary-change penalty weighted by 0.001, and a bound-violation penalty. Box clipping is followed by a joint constraint projection and sequential checks. The joint projection uses fixed reference scales, an LP feasibility witness and SLSQP, with a checked LP fallback when required. Final numerical acceptance and a score nonincrease tolerance of 0.0015 per outcome are evaluated separately. These are engineering acceptance criteria, not evidence of health improvement.

A separate 100-participant cohort supports six component variants, 600 total optimizations. Filters 1–3 are removed jointly, not individually. The five overall-acceptance contrasts use Holm adjustment; continuous paired bootstrap intervals are exploratory. Saved baseline and candidate inputs were subsequently scored by frozen BioNutriGPT, CatBoost and FT-Transformer models, without retraining or regenerating targets. This comprises 700 evaluated input rows per model, not necessarily distinct feature vectors. Models share observational development data and therefore do not provide independent-cohort validation.

The same 100 participants support eight stability conditions: baseline, same-seed repeat, all raw dietary nutrients scaled by +/-1% and +/-5%, and both measured lipids scaled by +/-5%. Derived features are recomputed; anthropometry is unchanged. A separate three-person pilot is not pooled. Paired target displacements use all 39 nutrient fields and fixed daily reference scales. Probability routing uses 0.5 for each outcome. Confidence intervals use 10,000 participant bootstrap resamples (seed 20261124), are pointwise and exploratory, and do not account for complex survey design. Cohort reuse creates related analyses, not independent replications.

## 3. Results

### 3.1 Stage 1 temporal-holdout results

All six model families were evaluated on the same repaired 7,004-person temporal cohort. Values below are unweighted AUROC estimates, not evidence of population representativeness.

| Model | Diabetes AUROC | Obesity AUROC | CVD AUROC |
|---|---:|---:|---:|
| logistic_regression | 0.7965 | 0.7445 | 0.8291 |
| hist_gradient_boosting | 0.8124 | 0.7557 | 0.8252 |
| catboost | 0.8138 | 0.7578 | 0.8288 |
| mlp | 0.7996 | 0.7519 | 0.8275 |
| bionutrigpt | 0.8107 | 0.7467 | 0.8288 |
| ft_transformer | 0.8109 | 0.7539 | 0.8350 |

BioNutriGPT calibrated Brier scores were 0.120914 (diabetes), 0.204624 (obesity), and 0.087353 (CVD).  The paired analysis supports selected diabetes improvements over logistic regression/MLP, but not an overall advantage over boosted trees or the reference FT-Transformer.

The separate five-variant, five-seed ablation study reports no primary architectural advantage after adjustment across 16 contrasts. Removing MoE reduces parameters from 1,295,623 to 700,675 (45.9%) without an established primary performance disadvantage. This does not prove equivalence or justify claiming the extra components are necessary.  The descriptive robustness study and unweighted demographic subgroup evaluation are complete; robustness uncertainty is descriptive and the downstream added-value evaluation has completed an  review (Section 3.2.2). A subsequent survey-weighted sensitivity analysis used WTDRD1PP and stratified delete-PSU jackknife variance (24 strata, 49 masked PSUs, 25 degrees of freedom). R survey independently reproduced overall estimates and uncertainty and extended analysis to demographic domains. For BioNutriGPT, weighted AUROC was 0.8402 for diabetes, 0.8468 for CVD and 0.7330 for obesity. These are conditional-on-model sensitivity results in the retained cohort; weighting does not remove additional analytic selection.  Survey-weighted calibration diagnostics for frozen calibrated BioNutriGPT probabilities yielded offset intercepts of 0.0512 (95% CI -0.0230 to 0.1255) for diabetes, 0.0273 (-0.1454 to 0.2000) for CVD and 0.2447 (0.1265 to 0.3628) for obesity, indicating average obesity underprediction. Exploratory decision curves across thresholds 0.05–0.50 used paired survey-design uncertainty. Because the cohort includes known diagnoses and prevalent CVD, these curves represent hypothetical further assessment and do not establish undiagnosed-disease screening or meal-planning utility. No overall custom-model advantage or optimal clinical threshold is claimed.  In the subgroup evaluation of 7,004 temporal participants, calibrated diabetes AUROC across six models and age strata ranged from 0.706 to 0.778, and CVD AUROC from 0.710 to 0.764. Among 120 diabetes/CVD AUROC and Brier contrasts against CatBoost, histogram boosting and FT-Transformer, 12 pointwise 95% intervals excluded zero with mixed directions. These exploratory contrasts were not adjusted for multiplicity and do not establish overall superiority, equivalence or clinical fairness. Intervals used 2,000 shared participant bootstrap resamples, conditional on fitted models and without complex survey-design adjustment.

### 3.1.1 Frozen-model robustness stresses and paired uncertainty evaluation

All six evaluated models were evaluated under nine fixed engineering scenarios on the same 7,004 temporal participants, without refitting preprocessing, weights or calibration. Missing dietary/lipid inputs used the frozen training imputation policy and missingness indicators. Partial dietary loss used two seeds; multiplicative dietary (±10%) and lipid (±5%) errors were chosen engineering sensitivities, not validated measurement-error distributions. Derived ratios were regenerated.

Uncertainty was evaluated post-hoc using 10,000 paired participant-level bootstrap resamples (seed 20261121, float64 precision, shared index matrix across models, scenarios, and outcomes). Within-model deterioration is defined such that positive values indicate performance loss:
- **Missing Lipids:** Dominant failure mode across all architectures. For BioNutriGPT calibrated diabetes, AUROC dropped by +0.0500 (95% CI: +0.0419 to +0.0578; stressed AUROC: 0.7607) and Brier score increased by +0.0114 (+0.0094 to +0.0133). Under omission of both diet and lipids, AUROC dropped by +0.0666 (+0.0577 to +0.0754).
- **Missing Dietary Inputs:** Complete dietary omission degraded calibrated diabetes AUROC by +0.0099 (+0.0061 to +0.0137; stressed AUROC: 0.8008) and Brier score by +0.0036 (+0.0025 to +0.0047). For CVD, discrimination loss was not statistically detectable (+0.0015, 95% CI: -0.0017 to +0.0049; interval spans zero), although Brier score increased by +0.0041 (+0.0029 to +0.0053). Modest 20% random feature omission produced minor drops (≤ +0.0017 AUROC) with intervals touching or crossing zero.
- **Input Perturbations (Noise):** 10% dietary noise (+0.0002, -0.0002 to +0.0007), 5% lipid noise (+0.0006, -0.0003 to +0.0016), and joint noise (+0.0013, +0.0002 to +0.0023) produced negligible discrimination loss across primary endpoints.
- **Direct paired robustness contrasts:** AUROC: 32 contrasts; 2 intervals exclude zero (2 favour BioNutriGPT; 0 favour the comparator). BRIER: 32 contrasts; 8 intervals exclude zero (3 favour BioNutriGPT; 5 favour the comparator). These are exploratory, unadjusted pointwise intervals. Scenario-specific findings do not establish an overall robustness advantage or equivalence. Full independent reproduction of all 2,688 bootstrap intervals is documented in Saved Model/stage1_robustness_endpoint_audit_v2/verification.json.

Full protocol, bindings, 576 within-model deterioration rows, 480 between-model contrast rows, and full independent verification of 2,688 interval pairs are archived in Saved Model/stage1_robustness_uncertainty_v1/Report.md.

### 3.2 Current catalog and implementation verification

The evidence snapshot confirms the current 49-entry catalog and 30 active planning fields. Forty-three records have no declared active-field gaps before dietary-mode, allergy, dislike, or other preference filters. This is catalog completeness under the declared schema, not a 43-food feasibility guarantee for every patient.

### 3.2.2 Stage 1 downstream added-value controlled study ()

An in-silico controlled study evaluated whether Stage 1 predictions add measurable engineering value to Stage 2-Stage 3 meal planning beyond observed demographics and reported clinical history. The study evaluated 100 randomly sampled temporal NHANES adult participants (seed `20261120`, strictly disjoint from the 5 pilot participants) across three dietary modes (`pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`), four arms (`rules_only`, `bionutrigpt`, `catboost`, `ft_transformer`), and five exploratory threshold levels (0.05, 0.10, 0.20, 0.30, 0.50), totaling 4,800 planned solves. All solves used HiGHS MILP with a 3.0-second limit, zero active nutrient slack, and reference-clipped baseline targets. Results were calculated from saved participant-level outputs.

Key findings:
1. **Feasibility:** Rules-only baseline achieved 98.67% feasibility (296/300 participant-mode pairs). Across 1,500 solves each, feasibility was 95.80% for BioNutriGPT (paired difference vs. rules-only: -2.87 percentage points [pp], 95% CI [-4.33, -1.53]), 96.07% for CatBoost (-2.60 pp [-4.00, -1.33]), and 96.00% for FT-Transformer (-2.67 pp [-4.07, -1.33]). Predictive rule branches exclusively tighten constraints (carbohydrate, saturated fat, sodium ceilings; fiber, potassium floors) and therefore cannot enlarge the feasible region. In model-vs-rules-only contrasts, exactly 12 of 45 cell-specific intervals exclude zero (all in `pure_vegetarian` mode at lower thresholds 0.05–0.30, reflecting strictly lower feasibility from constraint tightening). In zero cases did prediction-assisted planning make an infeasible profile feasible.
2. **Direct Model Contrasts:** Direct participant-level paired contrasts across all 1,500 solve pairs (2,000 bootstrap resamples with seed `20261120`) reveal minimal differences: BioNutriGPT vs. CatBoost = -0.27 pp [-0.60, 0.00] (-4 net solves); BioNutriGPT vs. FT-Transformer = -0.20 pp [-0.53, +0.13] (-3 net solves); CatBoost vs. FT-Transformer = +0.07 pp [-0.33, +0.47] (+1 net solve). Across all 45 cell-specific direct model-vs-model comparisons (15 per model pair), exactly 0 of 45 bootstrap intervals exclude zero (all contain zero). An interval containing zero indicates insufficient evidence of difference, not proof of equivalence. BioNutriGPT demonstrates no detected engineering advantage over standard tree-based or tabular transformer models.
3. **Failure Classification & Timing:** Of 186 unaccepted solves, 182 were solver-certified mathematical infeasibilities (Status 8, solver code 2, all in `pure_vegetarian`); 4 were computational timeouts (Status 13, solver code 1, all in `veg_nonveg_combo`). In `pure_non_vegetarian`, zero failures occurred (1,600/1,600 accepted). Recomputed timing distributions show that infeasible solve times ranged from 13.68 ms to 965.96 ms (median 24.03 ms, mean 128.57 ms), with 33 cases exceeding 40 ms solver execution time. Furthermore, 4 rules-only cases also failed with infeasibility in `pure_vegetarian` mode (cases 864, 2640, 3408, 3984).
4. **Plan Quality:** Among mutually accepted plans, 100% (4,318/4,318) satisfied all common-reference rules-only bounds (common-reference pass rate among all attempts was 95.96%). Normalized nutrient deviation using standard Daily Values as non-zero descriptive scaling references was 0.8% to 12.6% across models.

This computational study evaluated algorithmic constraint satisfaction and does not establish clinical efficacy, disease risk reduction, or Indian clinical validity. Full artifacts and audit records are preserved in `Saved Model/stage1_downstream_study_v1_audit_v3/` and `Research Paper/Stage1_Downstream_Added_Value_Results_v1.md`.

### 3.3 Stage 2 comparative added-value study

A post-result corrected evaluation reran the same 100 reserved complete-input temporal NHANES profiles across three target-construction methods. All methods used common model-informed routing and envelopes. The amendment removed duplicate search from timing, aligned floating-point checks and paired analysis statistics; it did not change model assets or the optimization objective.

| Method | Completed | Numerical pass | All-check accepted | Median runtime (s) | Median normalized change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules-only construction | 100/100 | 100/100 | 48/100 | 0.031 | 4.8190 |
| Minimum-change projection | 100/100 | 100/100 | 70/100 | 0.181 | 1.8494 |
| Model-guided search | 100/100 | 100/100 | 77/100 | 2.879 | 2.1992 |

Model-guided search accepted 77/100 profiles versus 70/100 with minimum-change projection and 48/100 with rules-only construction. The primary paired difference was +7.00 percentage points (10,000-resample percentile bootstrap 95% CI +2.00 to +13.00); the exact two-sided discordance test gave p=0.0390625 (8 wins, 1 losses). This is a model-score acceptance result, not demonstrated health improvement.

The paired mean additional normalized displacement for model-guided search was 0.2717 (bootstrap 95% CI 0.2325 to 0.3153). The scales are fixed engineering weights; this does not measure meal acceptability.

All methods passed numerical checks in the corrected evaluation. The  minimum-change exception was floating-point rounding, not meaningful protein inadequacy. Fresh model replay and separate statistical recomputation verified 300 records. The primary exact test is unadjusted; secondary acceptance tests use Bonferroni adjustment and other endpoints remain exploratory. Timings describe one fixed-order run, not a controlled hardware benchmark. These findings do not establish clinical benefit, Indian validity or discrete meal feasibility.  and Saved Model/stage2_comparison_study_v2/audit_complete.json.

### 3.4 Stage 2 to Stage 3 translation: comparative evaluation

Reanalysis of 900 existing records (100 participants, three arms and three dietary modes) changed no acceptance decisions. Model-guided versus minimum-change end-to-end acceptance differed by 7.33 percentage points (participant-cluster bootstrap 95% CI 2.00 to 13.33). The earlier pooled exact p-value was invalid because dietary modes were treated as independent people; it is withdrawn. No pooled exact p-value is reported.

| Method | Verified Stage 3 plans | End-to-end accepted | Conditional success after Stage 2 | Median recorded call time |
| --- | ---: | ---: | ---: | ---: |
| Rules-only | 298/300 | 144/300 | 144/144 | 3.031 s |
| Minimum-change | 297/300 | 209/300 | 209/210 | 3.031 s |
| Model-guided | 298/300 | 231/300 | 231/231 | 3.032 s |

Archived solver metadata show 881 time-limited solves with independently verified plans, 12 solver-status-0 results, three solver-reported infeasibilities and four timeouts without a verified plan. Status 0 is interpreted under the configured solver gap tolerance (1%), not as an independently proven exact optimum. The audit does not rerun or independently certify infeasibility. All 893 returned baskets passed fresh arithmetic and structural checks.

With active nutrient slack fixed to zero and the same energy prescription and bounds, arm-specific non-energy targets do not change the effective selection objective or mathematical feasible set. They can affect coefficients on fixed-zero slack variables, so the unreduced numerical model need not be byte-identical. Identical problems also need not produce identical incumbents within a time limit. Only 274 of 300 participant-mode groups had identical baskets across all three arms; 26 differed. The one-plan operational difference is not evidence that model guidance expands feasibility. End-to-end differences primarily inherit the upstream model-score gate, with a one-cell downstream timeout difference.

| Mode: model-guided vs minimum-change end-to-end | Difference (pp) | Exact p | Holm-adjusted p (nine comparisons) |
| --- | ---: | ---: | ---: |
| pure_vegetarian | +7.00 | 0.0390625 | 0.0781250 |
| veg_nonveg_combo | +8.00 | 0.0214844 | 0.0644531 |
| pure_non_vegetarian | +7.00 | 0.0390625 | 0.0781250 |

The protocol did not fix a primary mode or a multiplicity family. This correction is explicitly retrospective: pooled confidence intervals resample 100 participants with their modes together; per-mode exact tests are exploratory, with Holm adjustment across nine mode-by-arm comparisons per binary endpoint. The pointwise 95% intervals are not multiplicity-adjusted. Model-guided versus minimum-change is not significant at .05 in any individual mode after this adjustment. This does not prove equivalence.

The original all-field weighted distance is withdrawn from comparative interpretation: it was incorrectly labeled as using Stage 2 scales and active nutrients. A new explicitly post-hoc metric uses all 30 active nutrients, the exact fixed Stage 2 scales and only jointly successful participant-mode pairs. Unknown contributions are excluded rather than treated as known zero. Model-guided minus minimum-change paired mean distance difference: -0.0841, pointwise bootstrap 95% CI [-0.1648, -0.0043], over 100 participants and 297 paired mode cells. Each participant contributes the mean over their available matched modes. This exploratory geometric difference is not evidence of better meals or clinical benefit; it compares distances to different target vectors. Per-nutrient absolute errors and units are retained locally.

Fresh arithmetic and structural replay verified all 900 records; independent statistical replay checked 27 intervals and 18 exact tests. Historical protocol-freezing limitations remain, and these results do not establish clinical benefit or Indian external validity.  for the amendment, scope and verification receipts.

### 3.4.1 Target-aware experimental optimizer: comparative evaluation

The target-aware experimental optimizer reduced target error among jointly successful pairs, but produced fewer valid plans under the tested three-second budget. Acceptance counts were unchanged by fresh constraint verification. These are exploratory results on the reused 100-participant cohort, which also appeared in development tests; they are not independent validation.

| Target source | A valid plans | B valid plans | A mean target error | B mean target error | Paired error difference, 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| rules_only | 294/300 | 261/300 | 0.9072 | 0.4851 | -0.4221 [-0.4536, -0.3909] |
| minimum_change | 292/300 | 261/300 | 0.9434 | 0.5347 | -0.4086 [-0.4395, -0.3776] |
| model_guided | 294/300 | 260/300 | 0.9199 | 0.5133 | -0.4065 [-0.4378, -0.3761] |

Continuous means and 10,000-resample confidence intervals now both weight participants equally. Within each participant, only jointly successful matched dietary modes contribute. The three within-arm distance contrasts use 259, 257 and 257 paired cells respectively; available participant counts are recorded in analysis_summary.json. Missing plans have no target-error value. Binary endpoints retain all 100 participants and all modes; pooled intervals resample participant clusters. Per-mode exact tests are exploratory with retrospective Holm adjustment across 18 contrasts per endpoint, and intervals remain pointwise. The earlier p < 1e-15 statement is withdrawn.

Among 773 jointly successful pairs, 771 baskets differed and 2 were identical. The old 887/900 statistic conflated meal changes with failed empty outputs.

The archive contains status labels and gaps but omits raw solver codes, messages and incumbent flags. Therefore labels cannot independently prove optimality, lack of any incumbent, or mathematical feasibility for timed-out cases. The 299/300 mathematical-feasibility claim and the untested 5–10-second deployment recommendation are withdrawn. Variant A always ran before B; recorded runtimes are descriptive. The historical protocol was rewritable, so current hash agreement does not retrospectively establish preregistration. Engineering normalization scales are not uniformly official FDA/RDA values.

 for independent record replay and statistical verification. Production behavior was unchanged.

### 3.4.2 Controlled solver-budget experiment: paired analysis

| Budget | Current valid plans | Target-aware valid plans | Difference (pp), participant-cluster 95% CI |
| --- | ---: | ---: | --- |
| 3s | 895/900 | 891/900 | -0.44 [-1.22, +0.33] |
| 5s | 897/900 | 897/900 | +0.00 [+0.00, +0.00] |
| 10s | 897/900 | 896/900 | -0.11 [-0.33, +0.00] |

At five seconds, both objectives returned 897/900 valid plans with no discordant cells in this sample. This is identical observed acceptance, not demonstrated equivalence or noninferiority. The [0, 0] empirical bootstrap interval occurs because every sampled paired difference is zero; it does not establish zero population difference. Each budget contains nine repeated target-source/mode cells for each of 100 participants. Pooled McNemar p-values have been removed; participant-cluster intervals and the 27 within-cell exact tests with their Holm adjustments are retained.

At ten seconds, the target-aware solver recorded status 0 for 501/900 tasks under a configured 1% relative-gap tolerance. Describe this as solver termination within that tolerance, not mathematical proof of an exact integer optimum. The three common failures per budget are solver-reported infeasibilities; additional no-plan timeouts exist at three seconds and for one target-aware task at ten seconds. Their cause must not be generalized as proven extreme-calorie infeasibility without further evidence.

At ten seconds, recorded mean runtime was approximately 7.214 seconds for target-aware and 9.091 seconds for current-objective solves. This is an observed timing difference. No experiment established earlier branch-and-bound pruning as its cause. No causal explanation for the runtime difference is claimed.

Target-aware outputs had lower target error among jointly successful pairs at each measured budget. This conditional objective-aligned result does not establish clinical benefit.  and the  statistical verification receipt for corrected reporting.

### 3.4.3 Matched budget and target-source comparisons

Retrospective analysis uses 5,400 existing solve records from 100 participants, not 5,400 independent participants. No new optimization or training was performed. Five seconds is a candidate engineering setting on this sample, not an established clinical operating point or user-latency preference.

For the target-aware objective, observed valid plans were 891/900 at three seconds, 897/900 at five seconds and 896/900 at ten seconds. At five seconds both objectives accepted the same 897 cells; this does not establish population equivalence. At ten seconds, target-source success rates are not all identical. Across arms, the matched five-minus-three-second target-error differences were -0.0181 to -0.0145, with mean runtime differences +1.66 to +1.69 seconds. Ten-minus-five-second target-error differences were -0.0141 to -0.0115, at +2.79 to +2.92 seconds. These continuous comparisons condition on joint success. Common-success sensitivity retained the direction of lower target error; this does not establish clinical benefit. Normalization uses engineering reference scales, not a verified FDA nutrient standard. Recorded status 0 indicates termination under the configured 1% relative MIP-gap tolerance, not exact integer optimality.

At five seconds with the target-aware objective, model-guided versus minimum-change end-to-end acceptance differed by +7.67 percentage points (exploratory participant-bootstrap 95% CI [1.33, 14.33]). However, the per-mode exact tests have Holm-adjusted p=0.6943359375 across the declared 54-test end-to-end family. An advantage over minimum-change is therefore not established after multiplicity correction. The unadjusted bootstrap interval and adjusted tests answer different inferential questions. The prior claim that all 54 adjusted tests had p<0.001 was incorrect.

Saved Stage 2 flags distinguish numerical constraint checks from the model-score gate: rules-only 99/100 numerical, 49/100 score-gate and 48/100 overall; minimum-change 100/100, 69/100 and 69/100; model-guided 100/100, 77/100 and 77/100. These are counts of saved flags, not freshly recomputed Stage 2 predictions. The higher model-guided overall acceptance is principally associated with the model-score gate, not evidence of uniquely resolving clinical-rule conflicts. No external clinical benefit, target-source independence, practical equivalence or clinical superiority is demonstrated. Comparing error to different sources' own targets cannot rank their clinical quality.

Verification  replays statistical calculations from the existing reconstructed records; it is not a fresh food-catalog or solver audit. Six corruption fixtures now invoke the same validator used for clean data, and unexpected acceptance raises an uncaught failure. The original  fixture harness could catch its own failure sentinel and was not adequate evidence of rejection. Original results and historical artifacts are retained.  and Saved Model/stage3_budget_comparisons_v2/verification.json. Production behavior remains unchanged by this repair.

### 3.5 Preliminary optimizer comparison: 29-entry catalog

The saved comparison used 20 constructed cases, seven scored nutrient bounds, and shared food, constraint, and objective definitions. It predates the current strict 30-active-field acceptance policy and must not be interpreted as a current production acceptance study.

| Method | Plate rules passed | All seven bounds satisfied | Integer plans | Median runtime |
|---|---:|---:|---:|---:|
| Fully relaxed LP | 45% | 35% | 0% | 6.8 ms |
| Continuous portions with binary dish selection | 100% | 30% | 0% | 546.6 ms |
| Integer MILP | 100% | 15% | 100% | 1,921.5 ms |
| Prompt-only local Qwen 2.5 3B | 0% | 0% | 100% | 13,924.6 ms |

The integer model improved the whole-portion structural property but satisfied all seven scored bounds less often than the continuous-serving comparator. Integer output alone therefore did not establish a better overall diet. Reported monetary costs were based on unverified benchmark assumptions and are not interpreted as observed household expenditure. Practical acceptability was not measured.

### 3.6 Preliminary conversational comparison: 31-entry catalog

The saved 16-turn benchmark reported task completion in 15/16 turns for the guarded system and 1/16 for prompt-only Qwen. Recorded active-rule violations were 0/7 in each system; unsupported claims were 0/4 and 3/4, respectively. Safety/conflict handling used different denominators, 8/8 and 1/7, which prevents treating those percentages as a simple paired comparison.

Turns were clustered within eight scenarios and generated from two base profiles. These observations are preliminary: low violation counts do not establish success when task completion is low, and zero observed failures in a small selected sample does not establish a zero failure rate. The evaluation preceded the latest allergy and state-persistence repairs, so its accuracy and latency figures are not assigned to the repaired system.

### 3.7 LASI profile evaluation and audited interpretation

Case-level accepted flags and failure counts reconcile with the saved 300-solve summary. The same 100 unique people were evaluated repeatedly across modes.

| Dietary mode | Accepted / 100 | Reported infeasible | Timeout without valid plan | Median solve time |
|---|---:|---:|---:|---:|
| Vegetarian | 91 | 9 | 0 | 2003.1 ms |
| Combined vegetarian and non-vegetarian | 96 | 1 | 3 | 2005.6 ms |
| Non-vegetarian | 98 | 0 | 2 | 2006.1 ms |

Overall, 285/300 mode-specific solves were accepted. This is not a population-weighted national acceptance rate. The vegetarian low-energy subgroup had 21/27 accepted plans below 1500 kcal, compared with 27/27 in each other mode. Catalog or rule limitations at lower energy therefore warrant investigation. Accepted flags come from the production solver path; saved case rows do not contain full baskets and nutrient certificates for independent numerical replay.

The full guarded run passed 76/78 turns (97.4%). Of 42 revision requests, 29 produced validated proposals, 11 received justified refusals and two ended in timeout-related failure; one of these was an oracle-feasible request. Valid proposals were produced for 29/30 oracle-feasible revisions. No invalid proposals were recorded. Thirty-four of 36 scenarios passed every turn. The turn-weighted scenario-bootstrap 95% interval for protocol success was 93.67–100%; whole-scenario success was 34/36 with Wilson bounds of 81.86–98.46% under the stated working model.

| Matched  endpoint | Guarded assistant | Prompt-only Qwen 2.5 3B |
|---|---:|---:|
| Protocol passes | 24/24 | 2/24 |
| Valid proposals on oracle-feasible requests | 9/9 | 0/9 |
| Invalid revision proposals | 0 | 8 |
| Targeted nutrient-answer checks passed | 3/3 | 2/3 |
| Mean response time | 664.14 ms | 46,111.91 ms |

Qwen's two successful turns were targeted nutrient answers; they were not valid meal revisions. Its 22/24 parseable outputs and absence of unknown food IDs did not establish nutritional compliance. Conversely, the guarded 24/24 protocol rate includes three justified refusals, rather than 12 completed swaps. All 12 guarded scenarios passed every turn; the empirical bootstrap interval is degenerate at 100%, whereas the working whole-scenario Wilson lower bound is 75.75%. The comparison supports the tested system/configuration and tasks, not a general ranking of all LLMs.

For the 29 paired successful trajectories, mean serving change was 8.34 with the penalty and 16.76 without it, a 50.21% descriptive reduction. Four later pairs began from different recorded states after earlier outputs diverged. In the 25 identical-input-state pairs, means were 7.72 and 16.56: 53.38% fewer serving changes, with a mean paired difference of 8.84 servings and scenario-bootstrap 95% interval 6.72–11.12. This is evidence of reduced numerical disruption within the selected successful pairs; acceptability and clinical outcomes were not measured.

All 24 saved Qwen prompts were checked for eligibility parity and canonical rules; journals and four completed-stage checkpoints match the final report. Frozen request equality, participant uniqueness, disease balance and source hashes were verified. The final report and aggregate verification summary are the authoritative  evidence; historical summaries are not combined with them.

The earlier audit reran the 11 targeted chatbot regression checks: ten passed and one failed because the current parser records an explicit allergy statement ending in a question mark, whereas the earlier test expects no state change. This contract discrepancy must be resolved; the earlier 24-check pass is historical verification, not proof that the current source passes all checks.

### 3.8 Stage 2 input sensitivity

On the reused 100-participant ablation evaluation cohort, eight conditions (baseline, identical-input repeat, uniform dietary scaling by +/-1% and +/-5%, and both measured lipids scaled by +/-5%) produced 800 optimization records. All met the saved numerical acceptance checks; overall numerical-plus-score acceptance ranged from 62 to 70 participants by condition. Repeated inputs yielded identical targets in the tested environment. Displacements used all 39 nutrients normalized by fixed engineering reference scales; bootstrap intervals were exploratory and pointwise.

Lipid perturbations crossed the fixed 0.5 research routing cutoffs in six and five cases respectively; calorie-budget changes reached +500 kcal. Importantly, sizable individual nutrient changes also occurred without routing changes. A 1% dietary decrease changed a B2 target from 2.7169 to 1.2920 mg, and a 1% increase changed a B12 target from 5.3191 to 2.8101 micrograms. These differences were already present in post-Adam box-clipped outputs, before joint projection, with no reported solver fallback. Numerical feasibility and repeatability therefore do not establish local prescription stability or clinical appropriateness. Saved checkpoints do not isolate gradient-level mechanisms. This study does not evaluate anthropometric error, missing inputs or forced boundary profiles.  and Stage2_Stability_Retrospective_Audit_v1.md.

### 3.9 Stage 2 component contributions and cross-model consistency

In the six-variant study, the full reference achieved numerical acceptance in 100/100 cases and numerical-plus-score acceptance in 62/100. Removing the model-score objective reduced overall acceptance to 49/100 (difference -13 percentage points; 95% CI -20 to -7; Holm-adjusted p=0.000977). This endpoint partly evaluates the same model used by the objective and must not be read as clinical superiority.

| Variant | Numerical pass / 100 | Overall accepted / 100 |
|---|---:|---:|
| Full reference | 100 | 62 |
| No model-score objective | 100 | 49 |
| No change penalty | 100 | 64 |
| No barrier penalty | 100 | 65 |
| No joint projection | 8 | 7 |
| No sequential filters 1–3 | 100 | 62 |

Without joint projection, overall acceptance fell by 55 percentage points (95% CI -65 to -45; Holm-adjusted p=2.78e-16). Removing the change penalty increased normalized displacement by 0.9164 (95% CI 0.6847 to 1.1650). Removing the sequential filters changed no final outputs in these cases; this does not establish universal redundancy. The barrier did not demonstrate an acceptance advantage, although its removal changed mean displacement. These results distinguish numerical constraint enforcement from model-score optimization.

For full-reference proposals, mean probability changes under frozen comparator models were:

| Scorer | Diabetes change [95% CI] | Obesity change [95% CI] | CVD change [95% CI] |
|---|---|---|---|
| BioNutriGPT | -0.0098 [-0.0143, -0.0054] | -0.0414 [-0.0490, -0.0341] | -0.0068 [-0.0100, -0.0037] |
| CatBoost | +0.0132 [+0.0061, +0.0207] | -0.0317 [-0.0381, -0.0258] | -0.0051 [-0.0069, -0.0033] |
| FT-Transformer | -0.0024 [-0.0113, +0.0067] | -0.0345 [-0.0448, -0.0245] | -0.0083 [-0.0123, -0.0045] |

Mean obesity and CVD scores decrease under all three models, whereas diabetes differs in direction or uncertainty. An interval spanning zero does not establish equivalence. Neither shared-data agreement nor numerical feasibility demonstrates a causal response to dietary intervention.

## 4. Discussion

### 4.1 Interpretation and scientific contribution

The current contribution is an inspectable integration of established methods around a controlled revision process. Its distinctive evaluation target is whether restrictions and numerical requirements survive repeated conversational changes, acceptance, and rollback. The separation of screening evidence, food provenance, solver acceptance, and conversational state makes otherwise hidden failures observable.

The results also show why implementation should not be equated with superiority. The Transformer has no established overall advantage over simpler models. Integer portions improve one structural property but can reduce feasibility. An LLM can return integer quantities while still failing nutritional and plate checks. A guard can prevent an invalid plan from being accepted while failing to complete the user's request. A useful evaluation must report all these outcomes together.

### 4.2 Limitations

First, screening was developed in a U.S. survey population. Indian recipes do not transfer the prediction model's validity to Indian users. Laboratory-assisted screening is also a different use case from a questionnaire-only tool.

Second, some labels are partially observed, CVD is self-reported, and the obesity label is based on measured BMI. If weight and height are already available to the application, direct BMI assessment is an essential comparator for the obesity output. The usefulness of predicting a readily measurable outcome needs a specific justification.

Third, the Stage 2 engine combines reference intervals with project policies. Satisfaction of those policies is not evidence that the prescribed diet is appropriate for a person with multimorbidity, renal impairment, pregnancy, medication interactions, or unrecorded supplements.

Fourth, food composition varies by ingredient, cooking method, portion size, and provenance. Some records use proxies and calculated nutrient values. Missing fields and incomplete allergen metadata limit the scope of accepted plans.

Fifth, the historical benchmark catalogs differ from the current catalog and acceptance policy. The conversational experiment remains a small, selected, English-language scripted benchmark using one quantized LLM configuration and initially feasible profiles. The earlier two-profile conversational evaluation is too narrow to support claims across user populations or language styles. Daily food locks also do not prove meal-slot preservation.

Stage 2 also exhibits local sensitivity: sizable nutrient-target changes can occur without routing crossings and arise before joint projection. The exact gradient-level mechanism is unresolved, and no claim of stable clinical prescriptions is supported. The engineering results can be reported without treating this limitation as resolved.

Finally, the LASI study is Indian real-profile computational testing, not Indian prediction validation or a clinical intervention. No completed dietitian evaluation, prospective intervention or longitudinal outcome study is established by the reviewed artifacts. The existing dietitian protocol is future work; no collaboration, preregistration, blinding, or participant recruitment is claimed without supporting records.

## 5. Conclusion

BioNutriGPT implements a versioned, tool-backed meal-planning workflow that preserves a defined boundary between conversation and numerical acceptance. Current evidence supports implementation feasibility and identifies important evaluation limitations. The LASI pilot strengthens Indian profile relevance and provides preliminary evidence for less disruptive revisions. The matched comparison and conditional same-state ablation strengthen the engineering evidence. Independent dialogue testing, stronger conversational comparators and clinical evaluation are needed for broader claims. The completed evaluated Stage 1 comparisons do not establish overall Transformer superiority; Stage 2 sensitivity and cross-model disagreement further limit intervention claims.

## Declarations

**Data availability:** NHANES data are available through the National Center for Health Statistics [5]. LASI access is governed by the source data provider [11]. Food-composition sources are cited in references [7–9]. Participant-level records are not included with this manuscript.

**Ethics and participation:** This study analyzes secondary survey data and computational scenarios. No new participant recruitment, prospective intervention or completed dietitian assessment is reported.

**Patient and public involvement:** No patient or public involvement activity was conducted in the reported computational evaluation.

## References

1. Yang Z, et al. ChatDiet: Empowering personalized nutrition-oriented food recommender chatbots through an LLM-augmented framework. *Smart Health*. 2024;32:100465. [Article](https://doi.org/10.1016/j.smhl.2024.100465); [author manuscript](https://arxiv.org/abs/2403.00781).
2. Aguilera Moreno F. Mixed Integer Goal Programming for Personalized Meal Optimization with User-Defined Serving Granularity. 2026. Preprint; peer-review status is not assumed. [arXiv record](https://arxiv.org/abs/2605.13849).
3. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting Deep Learning Models for Tabular Data. NeurIPS. 2021. [Paper](https://arxiv.org/abs/2106.11959).
4. Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. [Statement](https://www.bmj.com/content/385/bmj-2023-078378).
5. CDC/NCHS. NHANES questionnaires, datasets, and related documentation: 2017–2020. [Survey documentation](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020).
6. ICMR–National Institute of Nutrition. Dietary Guidelines for Indians. 2024. [Guidelines](https://www.nin.res.in/dietaryguidelines/pdfjs/locale/DGI24thJune2024fin.pdf).
7. Development of an Indian Food Composition Database. *Current Developments in Nutrition*. 2024;103790. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11277795/).
8. USDA Agricultural Research Service. FoodData Central: data documentation. [Documentation](https://fdc.nal.usda.gov/data-documentation/).
9. Public Health England. McCance and Widdowson's Composition of Foods Integrated Dataset. 2021. [Dataset and user guide](https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid).
10. Gallifant J, et al. The TRIPOD-LLM reporting guideline for studies using large language models. *Nature Medicine*. 2025;31:60–69. [Guideline](https://www.nature.com/articles/s41591-024-03425-5).

11. Cohort Profile The Longitudinal Ageing Study in India LASI. [Cohort description](https://pmc.ncbi.nlm.nih.gov/articles/PMC9365624/); [official LASI data information](https://iipsindia.ac.in/content/LASI-data).

## Appendix A Predictor inventory

The model uses 87 predictors. HbA1c and fasting glucose are excluded as inputs. Feature names are reproduced below to make the input specification explicit.

{
  "BIO_COLS": [
    "LBDHDD",
    "LBXTC",
    "NON_HDL_CHOL",
    "CHOL_HDL_RATIO",
    "NON_HDL_TO_HDL_RATIO"
  ],
  "DIET_COLS": [
    "DR1TKCAL",
    "DR1TPROT",
    "DR1TCARB",
    "DR1TSUGR",
    "DR1TFIBE",
    "DR1TTFAT",
    "DR1TSFAT",
    "DR1TMFAT",
    "DR1TPFAT",
    "DR1TCHOL",
    "DR1TATOC",
    "DR1TRET",
    "DR1TVARA",
    "DR1TACAR",
    "DR1TBCAR",
    "DR1TLYCO",
    "DR1TLZ",
    "DR1TVB1",
    "DR1TVB2",
    "DR1TNIAC",
    "DR1TVB6",
    "DR1TFOLA",
    "DR1TFA",
    "DR1TFF",
    "DR1TFDFE",
    "DR1TVB12",
    "DR1TVC",
    "DR1TVK",
    "DR1TCALC",
    "DR1TPHOS",
    "DR1TMAGN",
    "DR1TIRON",
    "DR1TZINC",
    "DR1TCOPP",
    "DR1TSODI",
    "DR1TPOTA",
    "DR1TSELE",
    "DR1TCAFF",
    "DR1TALCO",
    "PROT_PCT_KCAL",
    "CARB_PCT_KCAL",
    "FAT_PCT_KCAL",
    "FIBER_PER_1000_KCAL",
    "SUGAR_TO_FIBER_RATIO",
    "SODIUM_POTASSIUM_RATIO",
    "SAT_TO_UNSAT_FAT_RATIO"
  ],
  "LIFE_COLS": [
    "SLEEP_HOURS",
    "SLEEP_DEV_FROM_7_5",
    "PAD680",
    "PA_VIG_WORK_MIN_WK",
    "PA_MOD_WORK_MIN_WK",
    "PA_WALK_BIKE_MIN_WK",
    "PA_VIG_REC_MIN_WK",
    "PA_MOD_REC_MIN_WK",
    "PA_TOTAL_MET_MIN_WK",
    "SLQ050",
    "PAQ605",
    "PAQ620",
    "PAQ635",
    "PAQ650",
    "PAQ665",
    "SMOKE_NEVER",
    "SMOKE_FORMER",
    "SMOKE_SOMEDAYS",
    "SMOKE_EVERYDAY"
  ],
  "DEMO_COLS": [
    "RIDAGEYR",
    "INDFMPIR",
    "DMDEDUC2",
    "SEX_FEMALE",
    "RACE_MEX_AMER",
    "RACE_OTHER_HISP",
    "RACE_NH_WHITE",
    "RACE_NH_BLACK",
    "RACE_OTHER_MULTI",
    "MARITAL_MARRIED",
    "MARITAL_PREV_MARRIED",
    "MARITAL_NEVER_MARRIED",
    "INDFMPIR_MISS",
    "SLEEP_HOURS_MISS",
    "PAD680_MISS",
    "PA_TOTAL_MET_MIN_WK_MISS",
    "OTHER_FEATURE_MISS"
  ]
}
