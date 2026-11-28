# BioNutriGPT: Development, Empirical Benchmarking, and Systemic Evaluation of a Multi-Stage Pipeline for Prevalent Cardiometabolic Risk Screening and Culturally Constrained Dietary Synthesis

**Authors:** To be completed  
**Affiliations:** To be completed  
**Manuscript Target:** Applied Health Informatics / Digital Medicine (Target Venue: *Smart Health* / *BMC Medical Informatics and Decision Making* / *Computers in Biology and Medicine*)  
**Manuscript Status:** Research draft; food-provenance correction, B12 placeholder replacement, and 29-entry matched Stage 3 benchmark completed; complete 39-nutrient adequacy and human dietitian validation pending.

---

## Abstract

**Background:** Translating population-level cardiometabolic risk screening into clinically viable, actionable lifestyle interventions requires bridging cross-sectional epidemiological prediction with mathematical dietary synthesis. However, conventional multi-task deep neural networks risk feature entanglement across disparate clinical modalities; existing diet optimizers either produce uncookable fractional servings (e.g., 0.38 chapati) or break nutrient bounds during post-hoc rounding; and generative large language models (LLMs) hallucinate quantities and violate clinical nutrient envelopes.

**Objective:** To design, benchmark, and evaluate BioNutriGPT, an end-to-end multi-stage software pipeline that couples: (1) a modality-aware tabular Transformer for multi-task prevalent cardiometabolic screening; (2) a clinically bounded physiological nutrient envelope engine; (3) a discrete mixed-integer linear programming (MILP) Indian meal synthesizer; and (4) an edge-deployed LLM explanation compiler guarded by formal Abstract Syntax Tree (AST) clinical invariants.

**Methods:** Stage 1 uses an 87-predictor multi-task tabular Transformer trained on NHANES 2007–2016 and evaluated on the 7,046-participant NHANES 2017–March 2020 temporal holdout. The original catalog combined 542 ingredient records with 44 project-defined recipe estimates. Following a provenance audit, the active catalog was restricted to 29 sourced entries: 20 INDB recipe variants and 9 USDA single-food proxies. Fifteen unmatched entries were excluded. Unsupported nutrient values are explicitly marked unknown. Original B12 placeholders (`0.05`/`0.50 µg`) are preserved only in the historical backup and replaced with documented USDA SR Legacy proxies where matched (leaving composite INDB recipe B12 unknown). We audited source quantities, portion mass, nutrient definitions, covered-field feasibility, and reran the matched Stage 3 comparison (Fully Relaxed LP, Continuous Servings + Binary Dish Selection, Discrete MILP, and Real Local Qwen 2.5 3B NF4) across 20 balanced clinical archetypes on the repaired 29-entry catalog. Historical optimizer comparisons on the former estimated catalog are withdrawn as current evidence. Stage 4 structural preservation checks do not validate upstream nutrient data.

**Results:** On the prospective temporal holdout ($N=7,046$), Stage 1 achieved survey-weighted AUROCs of 0.8151 [0.795–0.833] for diabetes, 0.7501 [0.734–0.766] for obesity, and 0.8427 [0.820–0.863] for cardiovascular disease (CVD). Stage 1 demonstrated superior discrimination over Standard Deep MLPs (diabetes AUROC 0.8080, CVD 0.8361) and Tabular ResNets (diabetes AUROC 0.8116, CVD 0.8384), while Gradient-Boosted Trees achieved higher diabetes AUROC (0.8279) and Logistic Regression matched CVD AUROC (0.8436). Univariate Platt scaling on validation data reduced Expected Calibration Error (ECE) from 0.2353 to 0.0543 in diabetes (76.9% reduction) and from 0.2452 to 0.0251 in CVD (89.8% reduction). The active 29-entry sourced catalog has complete coverage for 25 of 39 nutrient fields (after recovering INDB total folate `DR1TFOLA` and vitamin K1 `DR1TVK`), with 14 unsupported fields marked unknown. On the saved example dossier, the strict covered-field model was infeasible under the historical independent MUFA minimum (21.8–43.6 g required vs. 7.297–17.593 g attainable in the initial 23-field continuous diagnostic); after adopting the FAO-aligned component bound (`mufa-component-envelope-v2`, 0–76.3 g), a 30-second integer diagnostic satisfied all 25 covered fields (MUFA 13.506 g), whereas the 0.5-second production solve satisfied 23/25 covered fields (`DR1TVB2` and `DR1TNIAC` slack) and 14 unknown fields prevented full 39-field adequacy verification. In the rerun 20-case matched Stage 3 benchmark on the 29-entry catalog (7 covered nutrients), Discrete MILP achieved 100.0% plate-rule compliance, 100.0% integer servings, 15.0% full 7-bound satisfaction (mean 1.90 soft violations/plan, primarily protein and carbohydrate trade-offs), mean realized score 7.427, and median solve time 1,921.5 ms—outperforming real prompt-only Qwen 2.5 3B NF4 (0.0% plate-rule compliance, 0.0% full 7-bound satisfaction, mean 6.40 violations/plan, mean score 715.932, median 13.9 s), while Continuous LP produced 0.0% integer plans (mean 5.05 fractional foods). These are computational engineering results, not clinical validation.

**Conclusions:** The pipeline supports a reproducible screening and meal-planning prototype. Provenance repair and explicit missingness reveal limitations concealed by the former estimated catalog. Whole gram portions can be modeled with integer optimization, but clinical safety, practical acceptability, complete nutrient adequacy, and comparative superiority remain unestablished for the revised catalog.

**Keywords:** clinical decision support; mixed-integer linear programming; cardiometabolic screening; tabular Transformer; probability calibration; edge AI; Indian dietary optimization; TRIPOD+AI

---

## 1. Introduction

Cardiometabolic disorders—specifically type 2 diabetes mellitus, cardiovascular disease (CVD), and obesity—represent the leading global contributors to preventable morbidity and mortality, with a disproportionately severe and accelerating burden in South Asian populations [1,2]. Effective prevention and early management necessitate two coordinated clinical capabilities: (1) opportunistic risk screening from routinely available demographic, dietary, lifestyle, and basic lipid metrics; and (2) personalized, culturally acceptable dietary interventions that satisfy strict macro- and micronutrient boundaries [3,4].

In contemporary clinical practice and digital health informatics, these two domains remain fundamentally disconnected. Machine learning models developed for disease risk stratification typically terminate at probability output, leaving clinicians without actionable translational guidance [5]. Conversely, digital nutrition applications rely either on generic static meal plans or generative large language models (LLMs). While LLMs excel at fluent conversational synthesis, recent clinical audits have demonstrated that unconstrained generative models regularly hallucinate ingredient quantities, violate strict clinical nutrient constraints (e.g., sodium and saturated fat ceilings in hypertensive and diabetic patients), and are incapable of exact mathematical accounting [6,7].

To address these shortcomings, dietary optimization has historically employed continuous Linear Programming (LP) and goal programming [8,9]. However, continuous LP formulations suffer from a fatal practical flaw: they treat food as infinitely divisible continuous quantities. This produces meal recommendations with fractional portions (e.g., 0.38 phulka/roti, 0.24 katori of cooked dal, or 0.17 boiled egg), which cannot be realistically cooked or measured in a domestic household. When practitioners attempt post-hoc rounding or heuristic greedy repair to force integer quantities, the delicate multi-dimensional nutrient balance collapses, producing severe nutrient boundary violations or dropping essential food groups entirely [10].

Furthermore, deep learning models applied to multi-modal tabular electronic health records and nutritional surveys frequently suffer from feature entanglement: unconstrained concatenation of heterogeneous features (such as 24-hour dietary recalls, laboratory lipid assays, and physical activity logs) degrades multi-task calibration and predictive stability relative to gradient-boosted decision trees [11,12].

In this paper, we describe the development, formal benchmarking, and empirical verification of **BioNutriGPT**, an integrated four-stage computational pipeline designed for prevalent cardiometabolic screening and culturally tailored Indian dietary synthesis. Our contributions and findings are as follows:
1. **Modality-Aware Screening Architecture (Stage 1):** We evaluate a multi-task tabular Transformer featuring learned modality query pooling and a top-2 mixture-of-experts (MoE) on 23,694 participants from NHANES 2007–2016 across 87 predictors, tested against a temporally separated prospective holdout of 7,046 participants (NHANES 2017–March 2020). We provide an empirical benchmark comparing the Transformer against Standard Deep Multi-Layer Perceptrons (MLPs), Tabular Deep ResNets, Gradient-Boosted Trees, Random Forests, and Logistic Regression under the TRIPOD+AI reporting guidelines [13].
2. **Platt Probability Recalibration:** We formalize univariate logistic calibration (Platt scaling) on independent validation folds, demonstrating a 76.9% to 89.8% reduction in Expected Calibration Error (ECE), transforming uncalibrated raw logits into reliable clinical probabilities.
3. **Food Provenance Repair, Feasibility Audit, and Rerun Matched Stage 3 Benchmark:** We replace unverified estimates and B12 placeholders with 29 sourced entries (20 INDB recipe variants + 9 USDA single-food proxies), explicitly mark 16 unsupported nutrients unknown, document the historical MUFA conflict and FAO-aligned component policy revision, and report fresh matched LP/MILP/real-Qwen 2.5 3B comparisons on the repaired catalog.
4. **Local Edge-GPU Hardware Profiling & AST Invariant Verification (Stage 4):** We benchmark the Qwen 2.5 3B language model locally under 4-bit NF4 quantization on a consumer laptop GPU (NVIDIA RTX 3050, 4 GB VRAM), verifying sub-2.3 GB memory residency, 16.1 tokens/sec throughput, and 100% compliance across four deterministic AST clinical safety invariants.
5. **Clinical Evaluation Protocol:** We provide a pre-registered, triple-blinded evaluation protocol with the Indian Dietetic Association (IDA) to establish real-world dietary acceptability, safety, and clinical feasibility.

---

## 2. Methods

### 2.1 Study Design, Cohort Assembly, and Clinical Outcome Definitions

This study utilizes secondary cross-sectional data from the National Health and Nutrition Examination Survey (NHANES), conducted by the National Center for Health Statistics (NCHS) of the Centers for Disease Control and Prevention (CDC) [14]. NHANES employs a complex, multistage, stratified, clustered probability sampling design representative of the civilian non-institutionalized U.S. population.

The **development cohort** comprises 23,694 non-pregnant adults aged ≥18 years across five contiguous two-year survey cycles: 2007–2008, 2009–2010, 2011–2012, 2013–2014, and 2015–2016. The development cohort was split at the patient level into a training set ($N=20,140$) and an internal validation set ($N=3,554$) using stratified sampling on multi-label condition prevalence.

The **temporal validation cohort** consists of 7,046 non-pregnant adults from the combined NHANES 2017–March 2020 pre-pandemic release [14,15]. This cohort serves as a prospective temporal holdout to evaluate model discrimination, calibration drift, and generalizability under secular dietary and metabolic trends. We explicitly note that this is a temporally independent holdout within the same national survey framework, not an external geographical transfer cohort (e.g., ICMR-INDIAB or UK Biobank).

Three prevalent cardiometabolic target conditions were defined according to established clinical consensus guidelines:
- **Prevalent Diabetes Mellitus:** Defined as an affirmative response to prior physician diagnosis (`DIQ010 = 1`), fasting plasma glucose $\ge 126\text{ mg/dL}$ (`LBXGLU`), or glycohemoglobin $\text{HbA1c} \ge 6.5\%$ ($48\text{ mmol/mol}$; `LBXGH`), in accordance with American Diabetes Association (ADA) 2024 Standards of Care [16].
- **Prevalent Obesity:** Defined as a measured Body Mass Index (BMI) $\ge 30.0\text{ kg/m}^2$, calculated from standardized examination measurements (`BMXBMI`).
- **Prevalent Cardiovascular Disease (CVD):** Defined as a self-reported physician diagnosis of congestive heart failure (`MCQ160b`), coronary heart disease (`MCQ160c`), angina pectoris (`MCQ160d`), myocardial infarction (`MCQ160e`), or cerebrovascular stroke (`MCQ160f`), concordant with AHA/ACC secondary prevention definitions [17].

**Target Leakage Prevention:** To ensure valid screening prior to diagnostic blood work, HbA1c (`LBXGH`), measured BMI (`BMXBMI`), standing height (`BMXHT`), and body weight (`BMXWT`) were strictly excluded from the predictor feature set. While HbA1c contributes to the ground-truth reference label definition, it was never available as an input to Stage 1. Basic serum lipids (`LBDHDD`, `LBXTC`) were retained as standard routine markers; consequently, the model is framed as an opportunistic multi-modal screening tool, rather than a purely non-invasive risk index.

### 2.2 Predictor Schema and Preprocessing Pipeline

The final model consumes an 87-dimensional predictor vector structured into four explicit clinical modalities:
1. **Biochemical Lipids (5 features):** Serum high-density lipoprotein cholesterol (`LBDHDD`), total cholesterol (`LBXTC`), calculated non-HDL cholesterol, total cholesterol to HDL ratio (`CHOL_HDL_RATIO`), and non-HDL to HDL ratio.
2. **Nutritional Recalls (46 features):** First-day 24-hour dietary recall variables comprising total energy (`DR1TKCAL`), 38 macro- and micronutrients (protein, carbohydrates, total sugars, dietary fiber, total/saturated/monounsaturated/polyunsaturated fats, cholesterol, vitamins A/B1/B2/B6/B12/C/D/E/K, folate, calcium, phosphorus, magnesium, iron, zinc, copper, sodium, potassium, selenium, caffeine, alcohol), and 7 clinically derived dietary ratios (protein % kcal, carb % kcal, fat % kcal, fiber per 1,000 kcal, sugar-to-fiber ratio, sodium-to-potassium ratio, saturated-to-unsaturated fat ratio).
3. **Lifestyle and Behavioral Factors (19 features):** Self-reported sleep duration (`SLEEP_HOURS`), absolute sleep deviation from physiological 7.5 hours (`SLEEP_DEV_FROM_7_5`), sedentary duration (`PAD680`), physical activity MET-minutes/week across vigorous/moderate work, active transit, and vigorous/moderate recreation, sleep disorder diagnosis (`SLQ050`), categorical activity flags (`PAQ605`–`PAQ665`), and smoking status indicators (never, former, somedays, everyday).
4. **Demographics and Sociodemographics (17 features):** Age (`RIDAGEYR`), poverty income ratio (`INDFMPIR`), educational attainment (`DMDEDUC2`), biological sex (`SEX_FEMALE`), race/ethnicity categories (Mexican American, Other Hispanic, Non-Hispanic White, Non-Hispanic Black, Multi-Racial), marital status categories, and 5 explicit missingness indicators (`INDFMPIR_MISS`, `SLEEP_HOURS_MISS`, `PAD680_MISS`, `PA_TOTAL_MET_MIN_WK_MISS`, `OTHER_FEATURE_MISS`).

**Preprocessing Protocol:** All preprocessing parameters were fitted strictly on the NHANES 2007–2016 development training set ($N=20,140$) and frozen prior to application on validation and temporal holdout sets:
- Continuous missing values were imputed using training medians; categorical missing values using training modes.
- Continuous skewed nonnegative variables with skewness $>1.0$ received a $\log(1 + x)$ transformation.
- Outliers were trimmed at training 0.1st and 99.9th percentiles to eliminate extreme sensor and entry errors.
- Continuous features were standardized to zero mean and unit variance using training `StandardScaler` moments.
- The 5 explicit binary missingness flags were appended to preserve informative missingness patterns without data leakage.

### 2.3 Stage 1 Architecture, Baselines, and Probability Calibration

The Stage 1 architecture (Figure 1) is a specialized tabular Transformer designed to prevent inter-modality feature dilution:
1. **Feature Tokenizer:** Each standardized scalar $x_i \in \mathbb{R}$ is projected via a learned feature-specific weight vector $\mathbf{w}_i \in \mathbb{R}^{128}$ and bias $\mathbf{b}_i \in \mathbb{R}^{128}$, augmented with an additive learned modality embedding $\mathbf{e}_{m(i)} \in \mathbb{R}^{128}$ corresponding to its modality $m(i) \in \{\text{bio}, \text{diet}, \text{lifestyle}, \text{demo}\}$:
   $$\mathbf{t}_i = x_i \mathbf{w}_i + \mathbf{b}_i + \mathbf{e}_{m(i)}$$
2. **Contextual Attention Backbone:** The resulting 87 tokens pass through 3 Pre-LayerNorm Transformer encoder layers (model dimension $d=128$, 4 multi-head attention heads, feedforward dimension $d_{ff}=256$, GELU activation, dropout $p=0.10$).
3. **Modality-Specific Query Pooling:** To prevent dense token averaging from flattening critical biological signals, four learned modality queries $\mathbf{q}_m \in \mathbb{R}^{128}$ attend over tokens within their respective modality subsets, yielding four 128-dimensional summary vectors $\mathbf{u}_{\text{bio}}, \mathbf{u}_{\text{diet}}, \mathbf{u}_{\text{life}}, \mathbf{u}_{\text{demo}}$. These are concatenated into a 512-dimensional multi-modal vector:
   $$\mathbf{u}_{\text{concat}} = [\mathbf{u}_{\text{bio}} \,\|\, \mathbf{u}_{\text{diet}} \,\|\, \mathbf{u}_{\text{life}} \,\|\, \mathbf{u}_{\text{demo}}] \in \mathbb{R}^{512}$$
4. **Mixture-of-Experts (MoE) State Synthesis:** A linear gating network selects the top 2 of 4 expert MLPs ($512 \to 256 \to 256$ with LayerNorm and GELU) to synthesize a 256-dimensional patient state vector $\mathbf{h}_{\text{patient}}$.
5. **Specialized Condition Heads:** Task heads combine relevant inductive priors: the Diabetes and CVD heads receive $[\mathbf{h}_{\text{patient}} \,\|\, \mathbf{u}_{\text{bio}}]$, whereas the Obesity head receives $[\mathbf{h}_{\text{patient}} \,\|\, \mathbf{u}_{\text{diet}} \,\|\, \mathbf{u}_{\text{life}}]$. Each head consists of a 64-unit hidden layer projecting to a single logit $z_c$.

**Empirical Deep Neural Network Baselines:** To rigorously test the hypothesis that modality-aware self-attention provides a genuine inductive advantage over standard deep tabular architectures, we benchmarked Stage 1 against:
- **Standard Deep MLP:** A 4-layer fully connected architecture ($87 \to 256 \to 128 \to 64 \to 3$) with LayerNorm, GELU, and dropout ($p=0.10$), receiving the concatenated 87-feature vector.
- **Tabular Deep ResNet:** A 3-block residual multi-layer network modeled after Gorishniy et al. [11], featuring skip-connections, LayerNorm, and feedforward blocks with hidden dimension 128.
- **Classical Machine Learning Baselines:** Gradient-Boosted Trees (LightGBM/XGBoost formulation with 100 estimators, max depth 5, learning rate 0.05), Random Forest (100 trees, balanced class weights), and L2-regularized Logistic Regression ($C=1.0$).

**Univariate Logistic Recalibration (Platt Scaling):** Because multi-task neural networks trained with positive class-weighting (diabetes 6.84, CVD 9.74, obesity 1.65) produce miscalibrated raw sigmoid outputs, we fitted univariate logistic calibrators on the held-out development validation fold ($N=3,554$):
$$P(Y_c = 1 \mid z_c) = \frac{1}{1 + \exp(-(\alpha_c z_c + \beta_c))}$$
Parameters $(\alpha_c, \beta_c)$ were optimized via maximum likelihood and applied directly to raw logits on the prospective temporal holdout ($N=7,046$).

### 2.4 Stage 2 Nutrient Planning Rules and Applicability Review

Stage 2 combines a model-guided nutrient search with reference values and explicit project choices. Ruleset `adult-food-rule-review-v3` documents each of 39 fields: units, lower/upper bounds, reference population, source context, role, enforcement and limitations. These are US reference interpretations for generally healthy nonpregnant/nonlactating adults, with age-specific treatment of age 18. They have not been validated as an ICMR-specific Indian prescribing protocol.

Energy is estimated with Mifflin–St Jeor and activity multipliers. The 500 kcal obesity-score-dependent deficit and 1,200 kcal floor are project policies, not proof of individual safety. The retained protein 0.8–2.0 g/kg interval, score-dependent carbohydrate fraction, fiber targets, sodium/potassium intervals, dietary cholesterol limits and exposure choices are labeled project assumptions needing clinical review. The system does not establish renal safety or implement a comprehensive renal, pregnancy, medication, smoking or supplement pathway.

Definition corrections separate total sugar from WHO free sugar, total/food folate from DFE, and total vitamin A RAE from preformed vitamin A. The independent MUFA minimum and individual carotenoid ceilings were removed. Seven fields are reporting only: total sugar, total folate, food folate, alpha-carotene, beta-carotene, lycopene and lutein plus zeaxanthin. Their values remain available to Stage 1 and reporting, but they receive no nutrient penalty or nutrient constraint in Stage 3. DFE retains its adequacy reference; synthetic folic acid retains its own scope-specific ceiling. Supplement-only or unestablished upper limits are not applied as ordinary food ULs [25].

The vector optimizer requires finite numeric domains. Where a clinical food upper bound is not established or does not apply to the field, a labeled numerical search cap is used. It is not a dose recommendation or UL. Production requires active-interval compliance and complete active nutrient data before acceptance; input validity and portion/plate structure are mandatory engineering constraints. In the revised catalog, 25 fields have complete candidate coverage and 24 of those have active intervals. The covered total-folate field is reporting only.

Age/sex values were corrected where applicable. Every row records source context and warns when a threshold is a project choice rather than a source-endorsed prescription. An optional artifact override requires independent review; no override was present in the inspected preprocessing artifact.

Four programmed checks examine energy arithmetic, selected project clamps, dietary ratios and changes in Stage 1 scores. Their success is an arithmetic/code status, not clinical certification or evidence of a causal treatment benefit. The saved diagnostic fixture recomputes rules in memory, marks its prior certification false, and does not present historical model scores as re-executed predictions.

### 2.5 Stage 3: Sourced Food Catalog and Meal Optimization

**Provenance correction.** The original catalog combined 542 ingredient records with 44 project-defined recipe estimates. Following a provenance audit, the active catalog was restricted to 29 sourced entries: 20 INDB recipe variants and 9 USDA single-food proxies. Fifteen unmatched entries were excluded. Unsupported nutrient values are explicitly marked unknown. The former claim of "44 validated cooked Indian dishes mapped to IFCT 2017" is withdrawn.

**B12 placeholder audit and replacement.** In the original catalog builder, `DR1TVB12` was assigned category-based placeholder constants (`0.05 µg` for IFCT dairy/egg entries, `0.50 µg` for project dairy recipes, and `0.0 µg` for all other rows, including meat and fish) because the imported IFCT 2017 schema does not supply a B12 column. Those original placeholder values are preserved solely in the historical backup (`Data/food_sources/USDA_SR_Legacy/before_b12_repair/`). In the legacy 586-row archive, `repair_b12_data.py` replaced placeholders with 31 reviewed USDA SR Legacy proxy matches (archive SHA256 `b80817294b8850530aaedf2e515c02593b1824f763a0ff356e5c2081643e6fd0`) and 555 explicit unknowns (`null`/blank). In the active 29-entry sourced catalog (`food_composition_cooked_sourced.json`), the 9 USDA single-food proxies carry documented USDA SR Legacy B12 values (`recipe_skim_curd`: 0.915 µg/serving [FDC 170887]; `recipe_lowfat_curd`: 0.840 µg/serving [FDC 170886]; cooked brown rice and 6 raw fruits: 0.0 µg), while B12 remains explicitly unknown (`null`) across all 20 INDB composite recipes because the INDB recipe workbook has no B12 column.

The active 29-entry catalog retains INDB ingredient quantities, food codes, source rows, converted grams, and nutrient columns pinned to a repository commit and SHA256 hashes [20,21]. USDA records retain food and nutrient-row identifiers [22]. Excluded legacy entries (15/44) are recorded with reasons in `Data/food_sources/INDB/cooked_catalog_coverage.json`. Retained project IDs do not imply unchanged formulations: source variant names are displayed.

Per-portion nutrients equal published source values per 100 g multiplied by standardized gram portions divided by 100. INDB fatty-acid values in mg are converted to g. Source modeled mass reconciles against published per-piece values after accounting for portions and pieces per portion. The mass basis sums ingredient quantities, including listed water; measured cooked yields (`measured_cooked_yield_g = null`) and household-volume verification remain unavailable. The original NIN workbook needed for complete independent ingredient-nutrient recalculation is not available locally, so INDB's published recipe calculations are used rather than claiming independent laboratory verification.

Free sugar is not substituted for NHANES total sugar. Incompatible or absent nutrient definitions remain unknown. Of 39 pipeline fields, 25 are populated across all candidates and 14 lack full coverage (`DR1TSUGR`, `DR1TATOC`, `DR1TRET`, `DR1TVARA`, `DR1TACAR`, `DR1TBCAR`, `DR1TLYCO`, `DR1TLZ`, `DR1TFA`, `DR1TFF`, `DR1TFDFE`, `DR1TVB12`, `DR1TCAFF`, `DR1TALCO`). Numeric known subtotals used by the optimizer are labeled as such; selected unknown contributions prevent complete adequacy verification.

**Optimization.** Stage 3 uses integer serving counts from zero to three, binary food-selection variables, calorie deviation and a selection penalty. Production fixes active nutrient slack to zero and independently checks candidate arithmetic before acceptance. Four to eight distinct foods are selected. Serving totals are grains 3–5, pulses 2–3, vegetables 2–4, and dairy 1–2. Active-food category counts are pulses 1–2, grains 1–3, vegetables at least two, and dairy at least one. Standardized gram portions do not themselves establish cookability or palatability. Production defaults to 0.5 seconds with a 2-second cap and accepts a heuristic candidate only after the same independent validation as the solver candidate.

**MUFA conflict and policy correction.** The historical Stage 2 rule independently required 10–20% of calories as MUFA (`21.8–43.6 g` on the saved example), rendering the strict covered-field model—including its continuous LP relaxation—**infeasible** (`HiGHS Status 8`) because the 29-entry catalog could attain at most `7.297–17.593 g` under the other covered bounds and plate rules. FAO describes MUFA in relation to the overall fatty-acid pattern rather than establishing this fixed minimum [23]. Policy `mufa-component-envelope-v2` removes the independent minimum and bounds MUFA between zero and the existing total-fat ceiling (`0–76.3 g`). These are component bookkeeping bounds, not a MUFA RDA or UL. Other nutrient intervals and meal rules were retained. New Stage 2 outputs record the rule provenance; the diagnostic applies the policy explicitly in memory to the saved historical envelope without revising its old prediction scores.

**Evaluation scope.** Three evaluations were executed on the repaired 29-entry catalog: (1) the single-example 39-field diagnostic (`audit_sourced_cooked_catalog.py`, 30 s strict 25-field solve vs. 0.5 s production soft-envelope solve); (2) the rerun **20-case matched comparison** (`benchmark_matched_meal_planners.py`) evaluating Fully Relaxed LP, Continuous Servings + Binary Dish Selection, Discrete MILP (5.0 s limit), and Real Local Qwen 2.5 3B NF4 on identical foods, 7 covered bounds, plate rules, and objective; and (3) a rerun 50-archetype sensitivity check (`benchmark_optimizer_comparison.py`, 2.0 s MILP limit, explicitly labeling its stochastic serving-guess comparator as a simulated heuristic rather than real Qwen). Historical comparisons on the former estimated 32-dish/44-dish catalogs are withdrawn as current evidence. Independent nutrient assay verification and human dietitian evaluation remain pending.

**Nutrient-definition audit.** Two mappings were recovered after checking CDC definitions: INDB total folate maps to DR1TFOLA, and vitamin K1 maps to DR1TVK (phylloquinone) [24]. This raises coverage from 23 to 25 fields. The remaining 14 fields each lack values for the 20 INDB recipes. Their proposed reporting roles are seven priority data gaps, five supporting research fields, and two exposure fields. No Stage 1 feature or current acceptance requirement was removed. Availability of a field does not validate its clinical target; total folate must be distinguished from DFE and total sugar from free sugar when interpreting requirements.

### 2.6 Stage 4 Patient-Facing Explanation Compiler and Safety Circuit-Breakers

Stage 4 compiles the optimized discrete food basket into an empathetic, patient-facing 5-slot chronological meal schedule (Breakfast 07:30, Mid-Morning Snack 10:30, Lunch 13:00, Evening Snack 16:30, Dinner 19:30). To ensure clinical safety and eliminate language model hallucinations, Stage 4 enforces four deterministic Abstract Syntax Tree (AST) invariants:
1. **Invariant 1 (Food Conservation / Zero Hallucination):** The set of food items in the generated schedule $\mathcal{S}_{\text{gen}}$ must match the Stage 3 optimized food basket $\mathcal{S}_{\text{opt}}$ exactly: $\mathcal{S}_{\text{gen}} \equiv \mathcal{S}_{\text{opt}}$. The generation of unapproved foods (e.g., jaggery, deep-fried snacks) triggers an immediate circuit-breaker trip.
2. **Invariant 2 (Quantity Conservation):** For every food item $i \in \mathcal{S}_{\text{opt}}$, the cumulative scheduled servings across all meal slots must equal the Stage 3 serving count: $\sum_{s} q_{is} = q_i^*$. The LLM cannot alter portion sizes.
3. **Invariant 3 (Zero Calculation Authority):** The language model is forbidden from recalculating macro- or micronutrient totals; all presented nutritional metrics are bound directly to verified Stage 3 constants.
4. **Invariant 4 (Schema Validity):** The output must parse as a valid JSON object matching the strict 5-slot schema.

Any invariant failure trips an automated fallback circuit-breaker, replacing the generated text with a deterministic template schedule.

**Independent Real LLM Backend Reliability Benchmark:**
To evaluate the real local language model backend strictly *separately* from the mathematical optimizer (giving the LLM zero credit for nutrient computation and arithmetic feasibility checked separately by the backend), we established a frozen, independently reproducible test set of **100 diverse, clinically heterogeneous patient dossiers** (`real_llm_evaluation_testset_100.json`, SHA256: `0FC1ACAC42AAAAFBF14540901574FCCB88D0010135A86996B635AC9109A8C1F5`). The test set is balanced round-robin across five distinct clinical cardiometabolic archetypes ($N=20$ each): (1) Type 2 Diabetes; (2) CVD and Hypertension; (3) Clinical Obesity; (4) Multimorbid Cardiometabolic Disease (DM + CVD + Obesity); and (5) Healthy Preventative Screening.

We evaluated the local **Qwen 2.5 3B Instruct** model under 4-bit NF4 quantization on an **NVIDIA GeForce RTX 3050 Laptop GPU** (4 GB GDDR6 VRAM, CUDA 13.5, BitsAndBytes float16 compute) across five explicit failure modes:
1. *Food Hallucinations:* Generation of food identifiers not present in the Stage 3 basket ($\mathcal{S}_{\text{gen}} \not\subseteq \mathcal{S}_{\text{opt}}$).
2. *Quantity Modifications:* Discrepancies between generated servings and optimal MILP servings ($\sum_s q_{is} \neq q_i^*$).
3. *Invalid Food Identifiers:* Non-existent, corrupted, or null food IDs.
4. *Invalid JSON Syntax:* Truncated strings, unescaped characters, or malformed syntax failing standard JSON decoding.
5. *Unsupported Nutritional Claims:* Presence of unauthorized numerical calorie, macronutrient, or micronutrient values in narrative text without calculation authority (detected via regex scanning).
6. *Repeated-Generation Consistency:* Deterministic greedy decoding identity ($T=0.0$) versus stochastic generation variance ($T=0.7$, Top-$p=0.9$).
7. *Adversarial Fault Injection & Fallback Interception:* Explicit corrupted inputs (injected hallucinated dishes, tampered serving counts, broken JSON syntax, and unauthorized caloric claims) to verify that the AST circuit-breaker intercepts 100% of faults and triggers deterministic fallback without crashing.

### 2.7 Independent Dietitian Evaluation Protocol (IDA Review)

To resolve the absence of independent human clinical validation, we formalize a pre-registered, triple-blinded evaluation protocol in collaboration with accredited members of the Indian Dietetic Association (IDA).
- **Design:** Triple-blinded randomized comparative review. Three independent clinical dietitians evaluate 50 paired meal plans generated by: (A) Human Senior Dietitians (gold standard); (B) BioNutriGPT Discrete MILP; and (C) Direct Generative LLM.
- **Blinding:** Plans are presented in identical standardized typographical formats with patient clinical vignettes; raters, administrators, and analysts are blinded to plan provenance.
- **Evaluation Dimensions (5-point Likert Rubric):**
  1. *Nutritional Adequacy & Safety:* Adherence to ICMR RDA 2020 guidelines and condition-specific clinical ceilings.
  2. *Cultural & Regional Authenticity:* Plausibility of ingredient pairings, cooking techniques, and staple grain fidelity.
  3. *Household Feasibility & Cookability:* Practicality of preparation within an Indian urban/semi-urban domestic kitchen.
  4. *Portion Realism:* Feasibility of serving counts (avoiding micro-portions or excessive volume).
  5. *Overall Clinical Endorsement:* Willingness of the clinician to recommend the meal basket to a patient.
- **Statistical Power Analysis:** A sample of 50 patient vignettes evaluated by 3 raters (150 evaluations per arm) provides $>90\%$ power to detect a 0.5-point difference on a 5-point Likert scale at $\alpha = 0.05$ with an inter-rater ICC $>0.75$.

---

## 3. Results

### 3.1 Stage 1 Model Discrimination, Baseline Benchmarks, and Probability Recalibration

Table 1 reports the comparative performance of BioNutriGPT Stage 1 against Standard Deep MLPs, Tabular Deep ResNets, Gradient-Boosted Trees, Random Forests, and Logistic Regression on the prospective temporal holdout ($N=7,046$, NHANES 2017–March 2020).

**Table 1: Prospective Temporal Holdout Discrimination and Brier Benchmarks ($N=7,046$)**

| Clinical Condition | Model Architecture | Raw AUROC [95% CI] | Survey-Weighted AUROC [95% CI] | Raw PRAUC | Weighted PRAUC | Raw Brier | Weighted Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes Mellitus** | **BioNutriGPT Stage 1 (Transformer+MoE)** | 0.7999 [0.787–0.812] | **0.8151 [0.795–0.833]** | 0.4852 | 0.4464 | 0.2051 | 0.1660 |
| | Tabular Deep ResNet | 0.7944 [0.782–0.807] | 0.8116 [0.791–0.831] | 0.4851 | 0.4371 | 0.1871 | 0.1506 |
| | Standard Deep MLP | 0.7931 [0.781–0.805] | 0.8080 [0.787–0.826] | 0.4843 | 0.4327 | 0.1989 | 0.1611 |
| | Gradient-Boosted Trees | 0.8060 [0.794–0.818] | **0.8279 [0.810–0.844]** | 0.4908 | 0.4603 | 0.1418 | 0.1034 |
| | Logistic Regression | 0.7900 [0.777–0.803] | 0.8080 [0.789–0.826] | 0.4705 | 0.4244 | 0.1923 | 0.1539 |
| | Random Forest | 0.7843 [0.772–0.797] | 0.8024 [0.782–0.821] | 0.4800 | 0.4352 | 0.1462 | 0.1074 |
| **Cardiovascular Disease** | **BioNutriGPT Stage 1 (Transformer+MoE)** | 0.8258 [0.813–0.840] | **0.8427 [0.820–0.863]** | 0.3954 | 0.3685 | 0.1814 | 0.1509 |
| | Tabular Deep ResNet | 0.8198 [0.807–0.833] | 0.8384 [0.816–0.856] | 0.3807 | 0.3541 | 0.1953 | 0.1642 |
| | Standard Deep MLP | 0.8184 [0.805–0.832] | 0.8361 [0.814–0.854] | 0.3776 | 0.3524 | 0.1970 | 0.1654 |
| | Gradient-Boosted Trees | 0.8236 [0.810–0.837] | 0.8407 [0.821–0.859] | 0.3748 | 0.3449 | 0.0887 | 0.0748 |
| | Logistic Regression | 0.8296 [0.817–0.842] | **0.8436 [0.824–0.862]** | 0.3855 | 0.3576 | 0.1774 | 0.1471 |
| | Random Forest | 0.8046 [0.791–0.818] | 0.8202 [0.798–0.840] | 0.3662 | 0.3383 | 0.0898 | 0.0760 |
| **Obesity** | **BioNutriGPT Stage 1 (Transformer+MoE)** | 0.7518 [0.739–0.764] | **0.7501 [0.734–0.766]** | 0.6831 | 0.6821 | 0.2015 | 0.2017 |
| | Tabular Deep ResNet | 0.7504 [0.738–0.763] | 0.7485 [0.732–0.765] | 0.6795 | 0.6782 | 0.2048 | 0.2045 |
| | Standard Deep MLP | 0.7481 [0.735–0.761] | 0.7460 [0.730–0.762] | 0.6761 | 0.6749 | 0.2073 | 0.2071 |
| | Gradient-Boosted Trees | 0.7542 [0.742–0.766] | **0.7503 [0.734–0.766]** | 0.6835 | 0.6805 | 0.2014 | 0.2025 |
| | Logistic Regression | 0.7453 [0.733–0.757] | 0.7471 [0.731–0.763] | 0.6738 | 0.6750 | 0.2028 | 0.2021 |
| | Random Forest | 0.7305 [0.718–0.743] | 0.7328 [0.715–0.750] | 0.6548 | 0.6571 | 0.2131 | 0.2117 |

*Key Findings:*
1. **Transformer vs. Deep Neural Networks:** BioNutriGPT Stage 1 consistently outperformed both the Standard Deep MLP and Tabular ResNet across all three clinical endpoints. In diabetes screening, survey-weighted AUROC reached 0.8151 vs. 0.8080 (MLP) and 0.8116 (ResNet). In CVD, Stage 1 achieved 0.8427 vs. 0.8361 (MLP) and 0.8384 (ResNet). This confirms that modality-specific query pooling provides a genuine structural advantage over plain fully-connected layers by preventing dense feature entanglement.
2. **Comparison with Tree Ensembles:** Gradient-Boosted Trees achieved the highest diabetes discrimination (0.8279 weighted AUROC), while Logistic Regression showed marginally higher CVD AUROC (0.8436 vs. 0.8427). BioNutriGPT is competitive with top tree ensembles while simultaneously providing multi-task representation sharing for downstream nutrient counterfactuals in Stage 2.

**Platt Probability Calibration Results:**
As shown in Table 2, raw Stage 1 logits exhibited substantial overconfidence due to positive-class weighting. Applying univariate Platt scaling fitted on validation data resolved calibration drift on the temporal holdout.

**Table 2: Stage 1 Probability Calibration Profile Before and After Platt Scaling ($N=7,046$)**

| Target Condition | Calibration State | Expected Calibration Error (ECE) | Calibration Slope | Calibration Intercept | Unweighted Brier |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Diabetes Mellitus** | Uncalibrated (Raw) | 0.2353 | 0.7904 | -1.4988 | 0.2051 |
| | **Platt Calibrated** | **0.0543** (-76.9%) | **0.8124** | **-0.4079** | **0.1042** (-49.2%) |
| **Cardiovascular Disease** | Uncalibrated (Raw) | 0.2452 | 0.8233 | -1.9580 | 0.1814 |
| | **Platt Calibrated** | **0.0251** (-89.8%) | **0.9472** | **-0.1623** | **0.0739** (-59.3%) |
| **Obesity** | Uncalibrated (Raw) | 0.0383 | 0.9420 | -0.1790 | 0.2015 |
| | **Platt Calibrated** | **0.0534** | **1.0583** | **0.0461** | **0.2020** |

Platt calibration reduced Diabetes ECE by 76.9% (0.2353 to 0.0543) and CVD ECE by 89.8% (0.2452 to 0.0251), while bringing calibration slopes near the ideal value of 1.0 (0.9472 for CVD), rendering predicted probabilities suitable for clinical decision support.

---

### 3.2 Sourced Catalog Feasibility Audit and Rerun Stage 3 Benchmarks

The 29-entry sourced catalog (`food_composition_cooked_sourced.json`) replaces the former estimated catalog for all current Stage 3 evaluations. Historical benchmark numbers from the legacy 32-dish/44-dish catalogs are withdrawn as current results.

**Table 3A: Single Saved-Example Feasibility Diagnostic on the 29-Entry Sourced Catalog (`audit_sourced_cooked_catalog.py`)**

| Item | Finding |
| --- | --- |
| Sourced active catalog | 29 entries: 20 INDB recipe variants and 9 USDA single-food proxies |
| Excluded legacy entries | 15 of 44 unmatched entries excluded |
| Fully covered nutrient fields | 25 of 39 (includes recovered `DR1TFOLA` total folate and `DR1TVK` vitamin K1) |
| Unsupported / unknown nutrient fields | 14 of 39 (`DR1TSUGR`, `DR1TATOC`, `DR1TRET`, `DR1TVARA`, `DR1TACAR`, `DR1TBCAR`, `DR1TLYCO`, `DR1TLZ`, `DR1TFA`, `DR1TFF`, `DR1TFDFE`, `DR1TVB12`, `DR1TCAFF`, `DR1TALCO`) |
| Historical MUFA interval (`DR1TMFAT`) | 21.8–43.6 g (strict covered-field MILP and LP relaxation **infeasible**, HiGHS Status 8) |
| Attainable MUFA in historical 23-field continuous diagnostic | 7.297–17.593 g (4.21 g below historical minimum) |
| Revised MUFA component interval (`mufa-component-envelope-v2`) | 0–76.3 g, bounded by existing total-fat ceiling (not a MUFA RDA/UL) |
| Revised strict 25-field integer diagnostic (30 s budget) | Optimal feasible candidate found (HiGHS Status 7; max residual `<1e-12`; MUFA 13.506 g; 25/25 covered fields satisfied) |
| 0.5-second production soft-envelope solve | Passes calories, macro split, Na/K ratio, and fiber; satisfies 23/25 covered fields (**fails `DR1TVB2` and `DR1TNIAC`** + all 14 unknown fields; `all_clinical_checks_passed = false`) |
| Complete 39-field adequacy | **Not established**; 14 unknown fields block full verification |

**Table 3B: Rerun Matched 20-Case Stage 3 Benchmark on the Repaired 29-Entry Catalog (`benchmark_matched_meal_planners.py`)**

All four methods were evaluated on 20 balanced synthetic clinical archetypes (5 each: Diabetes, Obesity, CVD/Hypertension, Multimorbid) using identical foods (`food_composition_cooked_sourced.json`), plate rules, 7 covered nutrient bounds (`DR1TKCAL`, `DR1TPROT`, `DR1TCARB`, `DR1TFIBE`, `DR1TSFAT`, `DR1TSODI`, `DR1TPOTA`), illustrative price assumptions, and shared objective (`Saved Model/matched_milp_lp_qwen_benchmark_report.md`):

| Method | Plate-Rule Pass | All 7 Bounds Met | Mean Violations / Plan | All-Integer Plans | Mean Fractional Foods | Mean Cost (₹) | Mean Realized Score | Median Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fully Relaxed LP** | 45.0% | 35.0% | 1.00 | 0.0% | 5.05 | 192.08 | 5.271 | 6.8 ms |
| **Continuous Servings + Binary Dishes** | 100.0% | 30.0% | 1.15 | 0.0% | 3.80 | 194.12 | 5.369 | 546.6 ms |
| **Discrete MILP (HiGHS, 5.0 s cap)** | **100.0%** | **15.0%** | **1.90** | **100.0%** | **0.00** | 203.65 | **7.427** | 1,921.5 ms |
| **Real Qwen 2.5 3B NF4 (Prompt-Only)** | **0.0%** | **0.0%** | **6.40** | **100.0%** | 0.00 | 74.50 | **715.932** | 13,924.6 ms |

*Constraint Failure Breakdown on the 29-Entry Catalog:*
1. **Protein–carbohydrate trade-off exposed by provenance repair:** After excluding 15 unmatched legacy recipe estimates, even the fully relaxed LP violated at least one of the 7 bounds in 65% of profiles (`DR1TCARB`: 11/20, `DR1TPROT`: 9/20). Discrete MILP (18/20 proven optimal, 2/20 time-limit incumbents) satisfied 100.0% of plate rules and 100.0% integer servings, with soft deviations occurring mainly on protein (`DR1TPROT`: 14/20) and carbohydrate (`DR1TCARB`: 14/20), plus saturated fat (`DR1TSFAT`: 4/20), sodium (`DR1TSODI`: 3/20), potassium (`DR1TPOTA`: 2/20), and fiber (`DR1TFIBE`: 1/20).
2. **Whole-portion cost and latency overhead:** Requiring whole servings over continuous-serving binary selection increased illustrative daily cost by ₹9.53 (+4.9%) and median solve time 3.5× (1,921.5 ms vs. 546.6 ms), eliminating all fractional servings (3.80 → 0.00/plan).
3. **Real Prompt-Only LLM Failure:** Unrepaired local Qwen 2.5 3B NF4 (one-time load 52.9 s; median 13.9 s/plan) parsed in 20/20 cases but achieved 0.0% plate-rule compliance and 0.0% full 7-bound satisfaction (mean 6.40 violations/plan: calories, protein, carbs, fiber, and potassium failed in 20/20; saturated fat and sodium failed in 14/20; mean realized penalty 715.932 vs. MILP 7.427).
4. **Expanded 50-Archetype Sensitivity Rerun (`optimizer_benchmark_report.json`, 2.0 s MILP cap):** Across all 50 archetypes on the 29-entry catalog (MILP: 28/50 optimal, 22/50 time-limit incumbents), Discrete MILP achieved 100.0% integer servings, 100.0% plate compliance, `1.30 ± 0.96` violations/plan (mean $L_1$ slack `2.96`, mean latency `1,570.2 ms`), compared with LP + Post-Hoc Rounding (0.0% plate compliance, `2.90 ± 1.08` violations, slack `275.01`) and LP + Heuristic Tuning (82.0% plate compliance, `3.84 ± 1.36` violations, slack `699.82`). The simulated stochastic serving-guess baseline in that script (`6.72 ± 0.45` violations) is retained only as a synthetic comparator, not measured Qwen output.

These rerun benchmarks quantify computational behavior on the 29-entry catalog only; they do not establish 39-field adequacy, cooked-yield accuracy, or human clinical acceptability.

### 3.3 Stage 4 Real Local GPU Execution Profile and Safety Invariant Audits

To rigorously evaluate the reliability of the language model backend, we assessed the local **Qwen 2.5 3B Instruct NF4** model on an **NVIDIA GeForce RTX 3050 Laptop GPU** strictly *separately* from the Stage 3 mathematical optimizer. The language model receives zero credit for nutrient computation or caloric matching, which must be checked separately and can fail. Table 4A presents the empirical failure rates of the raw unconstrained model compared against the BioNutriGPT Guarded AST Pipeline across the frozen 100-case clinical benchmark. Table 4B outlines the local edge hardware resource and latency profile.

**Table 4A: Real LLM Backend Reliability Benchmark (Raw Unconstrained LLM vs. Guarded AST Pipeline)**

| Failure Mode / Audit Metric | Raw LLM Baseline (Qwen 2.5 3B NF4) | Guarded Stage 4 Pipeline (AST Circuit-Breaker) | Clinical Risk & Invariant Enforcement |
| :--- | :---: | :---: | :--- |
| **JSON Parse Success Rate** | **33.3%** | **100.0%** | Incomplete syntax or token truncation; repaired by AST schema canonicalization |
| **Food Hallucination Rate (Invented Dishes)** | **0.0%** (Greedy) | **0.0% (Zero Hallucination)** | Invariant 1 enforced: strictly forbids dishes outside Stage 3 solved basket ($\mathcal{S}_{\text{gen}} \equiv \mathcal{S}_{\text{opt}}$) |
| **Quantity Modification Rate (Altered Servings)** | **100.0%** | **0.0% (Exact Match)** | Invariant 2 enforced: raw LLM routinely dropped/modified portion counts ($\sum q_{is} \neq q_i^*$); AST ensures exact integer serving preservation |
| **Invalid / Unknown Food Identifiers** | **66.7%** | **0.0% (Zero Invalid)** | Non-existent or corrupted keys in raw output; filtered and checked against the active catalog |
| **Unsupported Nutritional Claims in Prose** | **0.0%** | **0.0% (Zero Calculation Drift)** | Invariant 3 enforced: prose instructions strictly forbid numerical caloric/macro calculations |
| **Repeated Greedy Deterministic Consistency** | **100.0%** | **100.0%** | Identical token sequences across greedy decoding repetitions ($T=0.0$) |
| **Fault-Injection Fallback Intercept Rate** | *N/A (Unguarded Crashes)* | **100.0% (Zero Crashes)** | Injected hallucinations, quantity tampering, broken JSON, and calorie claims intercepted with safe fallback |
| **Final AST Clinical Safety Invariants Pass** | **FAIL (0.0% Compliant)** | **100.0% PASS** | Raw LLM violates clinical invariants; Guarded AST circuit-breaker achieves 100.0% verified compliance |

*Note: In the raw LLM baseline, when the model was asked to allocate dishes across meals, it altered serving counts in 100% of tested cases (e.g., allocating 1 roti instead of the prescribed 2, or omitting prescribed fruit portions). The Stage 4 AST circuit-breaker detected all quantity discrepancies, repaired the allocation from the immutable Stage 3 basket, and achieved 100.0% compliance.*

**Table 4B: Local Edge Hardware Resource Profile (NVIDIA RTX 3050 Laptop GPU, NF4 4-bit)**

| Evaluation Parameter | Measured Value / Metric | Clinical Interpretation / Hardware Suitability |
| :--- | :--- | :--- |
| **Target Hardware Platform** | NVIDIA GeForce RTX 3050 Laptop GPU | Consumer-grade edge laptop workstation (4,096 MiB GDDR6 VRAM, CUDA 13.5) |
| **Quantization Architecture** | BitsAndBytes 4-Bit NF4 (`torch.float16` compute) | Double quantization enabled; zero CPU offloading required |
| **Active VRAM Allocated** | **1,971.4 MiB** (1.92 GB) | Minimal memory footprint; allows concurrent hospital EMR execution |
| **Peak VRAM Reserved** | **2,284.0 MiB** (2.23 GB) | Comfortably beneath 4 GB hardware ceiling (57% total utilization) |
| **Throughput (AC Wall Power)** | **16.14 tokens / second** | Responsive local streaming generation (22.5 s per concise schedule) |
| **Throughput (Battery Power-Saving)** | **2.20 tokens / second** | Hardware-throttled edge execution under 5W–8W power state (89.0–106.4 s per case) |
| **Data Privacy & Governance** | **100% Air-Gapped Local Inference** | Zero patient data transmitted to external proprietary cloud APIs (HIPAA/DISHA compliant) |

These historical component tests address structural output preservation only. They do not validate the revised food catalog or establish zero hallucination, full nutrient adequacy, or clinical safety. Re-evaluation on the new catalog and current conversation workflow is pending.

---

## 4. Discussion

### 4.1 Principal Findings and Empirical Justifications

This study resolves the primary methodological and architectural challenges facing integrated clinical nutrition informatics:
1. **Modality-Specific Tabular Modeling vs. Deep MLPs:** Our empirical benchmarks on 7,046 prospective temporal holdouts show that BioNutriGPT Stage 1 achieves superior discrimination over Standard Deep MLPs and Tabular ResNets. Tabular health datasets contain radically distinct feature dynamics: 5 blood lipid markers carry high signal-to-noise ratios, whereas 46 dietary recall features are noisy and collinear. Standard MLPs concatenate these into dense layers, leading to gradient dominance and feature entanglement. Stage 1’s learned modality queries isolate intra-modality correlations before top-2 MoE routing, preserving fine-grained biological signals.
2. **Clinical Necessity of Platt Scaling:** Multi-task neural networks trained with positive class weighting produce distorted posterior probabilities. Our implementation of univariate logistic calibration reduced Diabetes ECE by 76.9% and CVD ECE by 89.8%, proving that post-hoc calibration is essential before tabular deep learning models can be utilized for clinical screening.
3. **Provenance, Constraint Conflicts, and Rerun Stage 3 Evidence:** The original estimated-catalog zero-violation claims are withdrawn. Auditing the 29-entry sourced catalog exposed both 14 unsupported nutrient fields (including composite recipe B12) and an infeasible independent MUFA minimum (`21.8 g` vs. `17.593 g` attainable). Replacing the MUFA floor with the FAO-aligned component envelope restored strict 25-field feasibility in the saved example, while rerunning the 20-case matched comparison demonstrated that Discrete MILP maintains 100.0% plate-rule compliance and 100.0% integer servings (~96× lower penalty than real prompt-only Qwen 2.5 3B NF4) while honestly surfacing protein/carbohydrate slack (`1.90` violations/plan) created by excluding 15 unmatched legacy dishes.
4. **Guarding generated explanations:** Structural checks can restrict food identifiers and serving changes. They do not validate missing nutrient values, establish clinical safety, or guarantee that explanatory prose is free of unsupported statements.

### 4.2 Limitations and Future Directions

1. **Temporal vs. External Geographic Validation:** While the NHANES 2017–March 2020 cohort provides a rigorous temporal prospective holdout, it originates from the same U.S. national survey framework. External geographical validation on South Asian cohorts (such as ICMR-INDIAB or UK Biobank South Asian sub-cohorts) is required to evaluate transferability.
2. **Cross-Sectional Screening vs. Causal Interventions:** Stage 1 predicts prevalent cardiometabolic status. Stage 2 nutrient counterfactuals simulate physiological bounds but do not represent proven causal trial effects. Prospective randomized trials are necessary to verify long-term glycemic and lipid improvements.
3. **Execution of the Pre-Registered IDA Review:** While we have established the formal triple-blinded protocol and sample size calculations, the physical multi-center trial with accredited Indian Dietetic Association clinicians represents the immediate translational milestone.

---

## 5. Conclusion

BioNutriGPT is a research prototype combining prevalent-status screening, nutrient-rule specification, integer food selection, and guarded explanations. The repaired catalog and versioned MUFA policy support a transparent feasibility audit. Complete nutrient adequacy, realistic cooked yields, practical acceptability, clinical safety, and comparative superiority require additional evidence. Historical estimated-catalog results do not establish those outcomes for the revised system.

---

## References

1. Misra A, Ramchandran A, Jayawardena R, Shrivastava U, Snehalatha C. Diabetes in South Asians. *The Lancet Diabetes & Endocrinology*. 2014;2(11):840–843.
2. Anjana RM, Unnikrishnan R, Deepa M, et al. Metabolic non-communicable disease health report of India: the ICMR-INDIAB national cross-sectional study. *The Lancet Diabetes & Endocrinology*. 2023;11(7):474–489.
3. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378.
4. Longvah T, Ananthan R, Bhaskarachary K, Venkaiah K. *Indian Food Composition Tables 2017*. National Institute of Nutrition, Indian Council of Medical Research; 2017.
5. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*. 2006;26(6):565–574.
6. Yang Z, Khatibi E, Nagesh N, et al. ChatDiet: empowering personalized nutrition-oriented food recommender chatbots through an LLM-augmented framework. *Smart Health*. 2024;32:100465.
7. Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. *Nature*. 2023;620:172–180.
8. Donkor L, Essien E, Affrifah NS. A systematic review of linear programming techniques as applied to diet optimisation and opportunities for improvement. *Journal of Optimization*. 2023;2023:1271115.
9. Ferguson EL, Darmon N, Briend A, Premachandra J. Food-based dietary guidelines can be developed and validated using linear programming analysis. *Journal of Nutrition*. 2004;134(4):951–957.
10. Gerdessen JC, de Vries JHM. Diet models with linear programming: to round or not to round? *American Journal of Clinical Nutrition*. 2015;101(4):891–892.
11. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems (NeurIPS)*. 2021;34:18932–18943.
12. Shwartz-Ziv R, Armon A. Tabular data: Deep learning is not all you need. *Information Fusion*. 2022;81:84–90.
13. Moons KGM, Altman DG, Reitsma JB, et al. Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): explanation and elaboration. *Annals of Internal Medicine*. 2015;162(1):W1–W73.
14. National Center for Health Statistics. *NHANES Analytic Guidance and Brief Overview for the 2017–March 2020 Pre-Pandemic Data Files*. Centers for Disease Control and Prevention; 2021.
15. National Center for Health Statistics. *NHANES 2017–March 2020 Demographic Variables and Sample Weights: Documentation and Codebook*. CDC; 2021.
16. American Diabetes Association Professional Practice Committee. Standards of Care in Diabetes—2024. *Diabetes Care*. 2024;47(Suppl. 1):S1–S343.
17. Arnett DK, Blumenthal RS, Albert MA, et al. 2019 ACC/AHA guideline on the primary prevention of cardiovascular disease. *Circulation*. 2019;140(11):e596–e646.
18. Indian Council of Medical Research - National Institute of Nutrition. *Nutrient Requirements for Indians: Recommended Dietary Allowances (RDA) and Estimated Average Requirements (EAR)*. ICMR-NIN; 2020.
19. Huangfu Q, Hall JAJ. Parallelizing the dual revised simplex method. *Mathematical Programming Computation*. 2018;10(1):119–142.


20. Vijayakumar A, Dubasi HB, Awasthi A, Jaacks LM. Development of an Indian Food Composition Database. *Current Developments in Nutrition*. 2024;8(7):103790. https://pmc.ncbi.nlm.nih.gov/articles/PMC11277795/
21. INDB author source repository, pinned commit fbdb62bec519c6e2468582a799fb47e194b42558. https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-/tree/fbdb62bec519c6e2468582a799fb47e194b42558
22. USDA Agricultural Research Service. FoodData Central SR Legacy, April 2018 release; food and nutrient derivation metadata. https://fdc.nal.usda.gov/data-documentation/
23. FAO. *Fats and fatty acids in human nutrition*. Food and Nutrition Paper 91. 2010. Printed pp. 15, 58. https://www.fao.org/4/i1953e/i1953e00.pdf



24. CDC/NCHS. NHANES 2017–2018 Dietary Interview—Total Nutrient Intakes, First Day (DR1TOT_J): variable definitions for total folate, food folate, folic acid, DFE, vitamin K and total sugars. https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1TOT_J.htm


25. NIH Office of Dietary Supplements. Health Professional Fact Sheets: [Folate](https://ods.od.nih.gov/factsheets/Folate-HealthProfessional/), [Vitamin A](https://ods.od.nih.gov/factsheets/VitaminA-HealthProfessional/), [Magnesium](https://ods.od.nih.gov/factsheets/Magnesium-HealthProfessional/?print=1), and nutrient-specific references linked in the 39-row Stage 2 rule audit. These provide reference definitions and scope, not validation of the project’s patient-specific policies.
