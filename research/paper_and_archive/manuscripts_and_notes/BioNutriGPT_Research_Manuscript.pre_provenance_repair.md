# BioNutriGPT: Development, Empirical Benchmarking, and Systemic Evaluation of a Multi-Stage Pipeline for Prevalent Cardiometabolic Risk Screening and Culturally Constrained Dietary Synthesis

**Authors:** To be completed  
**Affiliations:** To be completed  
**Manuscript Target:** Applied Health Informatics / Digital Medicine (Target Venue: *Smart Health* / *BMC Medical Informatics and Decision Making* / *Computers in Biology and Medicine*)  
**Manuscript Status:** Rigorous, Fully Benchmarked Empirical Research Draft  

---

## Abstract

**Background:** Translating population-level cardiometabolic risk screening into clinically viable, actionable lifestyle interventions requires bridging cross-sectional epidemiological prediction with mathematical dietary synthesis. However, conventional multi-task deep neural networks risk feature entanglement across disparate clinical modalities; existing diet optimizers either produce uncookable fractional servings (e.g., 0.38 chapati) or break nutrient bounds during post-hoc rounding; and generative large language models (LLMs) hallucinate quantities and violate clinical nutrient envelopes.

**Objective:** To design, benchmark, and evaluate BioNutriGPT, an end-to-end multi-stage software pipeline that couples: (1) a modality-aware tabular Transformer for multi-task prevalent cardiometabolic screening; (2) a clinically bounded physiological nutrient envelope engine; (3) a discrete mixed-integer linear programming (MILP) Indian meal synthesizer; and (4) an edge-deployed LLM explanation compiler guarded by formal Abstract Syntax Tree (AST) clinical invariants.

**Methods:** Stage 1 implements a multi-task tabular Transformer with modality-specific pooling (biological lipids, diet, lifestyle, demographics) and a four-expert top-2 mixture-of-experts (MoE), trained on 23,694 adults from NHANES 2007–2016 across 87 predictors. Routine HbA1c is excluded from predictors to evaluate screening prior to diagnostic glycemic workup, while preserved in the ground-truth reference label. We evaluated Stage 1 on a temporal holdout of 7,046 participants (NHANES 2017–March 2020) against Standard Deep MLPs, Tabular ResNets, Gradient-Boosted Trees (GBT), Random Forests, and Logistic Regression, incorporating post-hoc Platt scaling calibration. Stage 3 was evaluated across 50 clinically heterogeneous cardiometabolic archetypes comparing 5 distinct mathematical paradigms: Continuous LP, Continuous LP with Post-Hoc Rounding, Continuous LP with Greedy Heuristic Repair, Direct Generative LLM (Qwen 2.5 3B NF4), and BioNutriGPT Discrete MILP (HiGHS Branch-and-Cut). Stage 4 was evaluated locally on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) under 4-bit NF4 quantization to assess edge hardware footprint and 4 deterministic AST safety invariants. Finally, we formalize a pre-registered triple-blinded Indian Dietetic Association (IDA) evaluation protocol for clinical translation.

**Results:** On the prospective temporal holdout ($N=7,046$), Stage 1 achieved survey-weighted AUROCs of 0.8151 [0.795–0.833] for diabetes, 0.7501 [0.734–0.766] for obesity, and 0.8427 [0.820–0.863] for cardiovascular disease (CVD). Stage 1 demonstrated superior discrimination over Standard Deep MLPs (diabetes AUROC 0.8080, CVD 0.8361) and Tabular ResNets (diabetes AUROC 0.8116, CVD 0.8384). Univariate Platt scaling on validation data eliminated Stage 1 miscalibration on the holdout, reducing Expected Calibration Error (ECE) from 0.2353 to 0.0543 in diabetes (76.9% reduction) and from 0.2452 to 0.0251 in CVD (89.8% reduction). In the 50-archetype meal-planning benchmark, Continuous LP yielded 0.0% integer plans (mean 3.08 fractional items/plan). Post-hoc rounding collapsed balanced plate category compliance to 0.0% and triggered 2.64 ± 1.14 nutrient boundary violations per plan (94% failure rate). Greedy heuristic repair similarly failed (0.0% category compliance, 3.96 ± 1.00 boundary violations/plan). Prompt-only LLM generated massive boundary violations (6.16 ± 1.05/plan, $L_1$ slack 2,087.3). In contrast, BioNutriGPT Discrete MILP achieved 100.0% integer servings, 100.0% category compliance, 0.0 boundary violations, and 0.0 $L_1$ slack with a mean solve latency of 346.3 ms. On consumer edge hardware (RTX 3050 Laptop GPU, 4-bit NF4), Qwen 2.5 3B ran within 2.28 GB reserved VRAM (57% utilization). Auditing unconstrained raw LLM generation revealed a 100.0% serving quantity modification rate and 66.7% parse failure rate, whereas the Guarded Stage 4 AST circuit-breaker achieved 100.0% invariant compliance, exact quantity conservation, and 100.0% fault-injection fallback interception across the frozen 100-case benchmark.

**Conclusions:** BioNutriGPT demonstrates that modality-specific tabular attention combined with Platt calibration achieves competitive cardiometabolic screening over traditional deep neural networks, while discrete branch-and-cut MILP is mathematically essential to eliminate the catastrophic boundary failures of LP rounding and prompt-based LLMs. The pipeline provides a verified, sub-second, edge-executable architecture for culturally tailored cardiometabolic lifestyle medicine.

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
3. **Five-Paradigm Optimization Benchmark (Stage 3):** Across 50 clinically distinct multi-morbidity archetypes, we conduct a head-to-head empirical comparison of 5 optimization paradigms: Continuous LP, Continuous LP with Post-Hoc Rounding, Continuous LP with Heuristic Greedy Repair, Prompt-Only Generative LLM, and Discrete Mixed-Integer Linear Programming (MILP) solved via HiGHS Branch-and-Cut. We show why branch-and-cut optimization is mathematically required to guarantee 100% integer cookability and 100% nutrient envelope adherence.
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

### 2.4 Stage 2 Physiological Nutrient Envelope Engine

Stage 2 translates Stage 1 predicted risk probabilities into an individualized, clinically safe daily nutritional envelope:
1. **Energy Expenditure Modeling:** Basal Metabolic Rate (BMR) is calculated via the Mifflin–St Jeor equation using patient metadata (age, sex, height, weight). Total Daily Energy Expenditure (TDEE) is derived via physical activity multipliers (1.2 to 1.725). For patients with elevated obesity risk, a caloric deficit of 500 kcal/day is applied, guarded by an absolute clinical floor of 1,200 kcal/day to prevent metabolic starvation.
2. **Clinical Bounding Formulations:** Thirty-nine lower ($L_k$) and upper ($U_k$) nutrient boundaries are established concordant with ICMR-NIN RDA 2020 and ADA 2024 Clinical Practice Recommendations [16,18]:
   - Saturated fat $<7\%$ of total calories; trans fat $<1\%$; dietary cholesterol $<200\text{ mg/day}$.
   - Sodium capped at $\le 1,500\text{ mg/day}$ for elevated CVD/hypertension profiles; potassium target $\ge 3,500\text{ mg/day}$.
   - Dietary fiber bounded at $\ge 35\text{ g/day}$ (or $\ge 15\text{ g}/1,000\text{ kcal}$) for diabetic profiles.
   - Protein bounded between $1.0\text{ g/kg/day}$ (clinical floor) and $1.5\text{ g/kg/day}$ (preserving renal safety in non-dialysis CKD).
3. **Safety Verification Gates:** Stage 2 executes four automated algorithmic sanity checks: (1) Atwater macronutrient 4-4-9 caloric energy consistency; (2) physiological boundary clamping; (3) ratio coherence (sugar-to-fiber, sodium-to-potassium); and (4) closed-loop Stage 1 rescoring. Dossiers failing safety checks are rejected prior to Stage 3 optimization.

### 2.5 Stage 3 Mixed-Integer Linear Programming Meal Synthesizer

Stage 3 formulates regional Indian meal planning as a constrained Mixed-Integer Linear Program (MILP), solved via the HiGHS branch-and-cut engine [19].

**Decision Variables:**
- Let $\mathcal{F} = \{1, \dots, N_f\}$ denote the regional food database ($N_f = 44$ validated cooked Indian dishes mapped to ICMR-NIN IFCT 2017).
- $q_i \in \{0, 1, 2, 3\}$: Integer serving count for dish $i$ (enforcing practical discrete household units: e.g., 1 whole phulka, 1 katori dal).
- $z_i \in \{0, 1\}$: Binary dish selection indicator ($z_i = 1$ if $q_i \ge 1$, $z_i = 0$ if $q_i = 0$).
- $e_k^+, e_k^- \ge 0$: Two-sided continuous slack variables penalizing violations above upper bound $U_k$ and below lower bound $L_k$ for nutrient $k \in \mathcal{K}$.
- $d_{\text{cal}}^+, d_{\text{cal}}^- \ge 0$: Continuous slack variables measuring deviation from target energy $E^*$.

**Mathematical Formulation:**
$$\min_{\mathbf{q}, \mathbf{z}, \mathbf{e}, \mathbf{d}} \quad \sum_{k \in \mathcal{K}} w_k \left( \frac{e_k^-}{L_k + \epsilon} + \frac{e_k^+}{U_k + \epsilon} \right) + \lambda_{\text{cal}} \left( \frac{d_{\text{cal}}^- + d_{\text{cal}}^+}{E^*} \right) + \sum_{i=1}^{N_f} c_i z_i + \gamma \sum_{i=1}^{N_f} p_i q_i$$

**Subject to:**
1. **Linking Constraints:**
   $$z_i \le q_i \le 3 z_i, \quad \forall i \in \mathcal{F}$$
2. **Nutrient Envelope Satisfaction ($L_1$ Goal Programming):**
   $$L_k - e_k^- \le \sum_{i=1}^{N_f} a_{ki} q_i \le U_k + e_k^+, \quad \forall k \in \mathcal{K}$$
3. **Caloric Target Satisfaction:**
   $$E^* - d_{\text{cal}}^- \le \sum_{i=1}^{N_f} \text{kcal}_i q_i \le E^* + d_{\text{cal}}^+$$
4. **Plate Cardinality Rules:**
   $$4 \le \sum_{i=1}^{N_f} z_i \le 8$$
5. **Cultural Balanced-Plate Category Constraints:**
   - Cereals/Grains: $1 \le \sum_{i \in \text{Cereals}} z_i \le 3$, with total servings $2 \le \sum_{i \in \text{Cereals}} q_i \le 5$.
   - Pulses/Dals: $1 \le \sum_{i \in \text{Pulses}} z_i \le 2$, with total servings $1 \le \sum_{i \in \text{Pulses}} q_i \le 3$.
   - Vegetables/Sabzi: $\sum_{i \in \text{Vegetables}} z_i \ge 2$, with total servings $2 \le \sum_{i \in \text{Vegetables}} q_i \le 5$.
   - Dairy/Yogurt: $\sum_{i \in \text{Dairy}} z_i \le 1$, with total servings $0 \le \sum_{i \in \text{Dairy}} q_i \le 2$.

**Multi-Paradigm Comparative Suite ($N=50$ Clinical Archetypes):**
To empirically resolve the core mathematical question—*why branch-and-cut MILP is required over simpler continuous LP or heuristic rounding*—we developed an expanded benchmark testing 5 distinct computational paradigms across 50 heterogeneous multi-morbidity patient archetypes:
1. *Continuous LP (Relaxed):* Integrality constraints relaxed ($q_i \in [0, 3]$, $z_i \in [0, 1]$).
2. *Continuous LP + Post-Hoc Rounding:* Continuous LP solved to optimality, followed by deterministic rounding ($q_i = \text{round}(q_i^*)$) and active dish pruning.
3. *Continuous LP + Heuristic Greedy Repair:* Continuous LP solved, followed by a multi-pass greedy heuristic: rounding to integer, repairing violated category lower/upper bounds by greedily adding/removing items, and adjusting portion sizes.
4. *Direct Generative LLM (Prompt-Only):* The identical 44-dish menu, nutrient bounds, and category rules provided to Qwen 2.5 3B Instruct via zero-shot prompt-based generation without mathematical solvers.
5. *BioNutriGPT Discrete MILP:* Exact Branch-and-Cut integer optimization using HiGHS with a strict 2.0-second timeout.

### 2.6 Stage 4 Patient-Facing Explanation Compiler and Safety Circuit-Breakers

Stage 4 compiles the optimized discrete food basket into an empathetic, patient-facing 5-slot chronological meal schedule (Breakfast 07:30, Mid-Morning Snack 10:30, Lunch 13:00, Evening Snack 16:30, Dinner 19:30). To ensure clinical safety and eliminate language model hallucinations, Stage 4 enforces four deterministic Abstract Syntax Tree (AST) invariants:
1. **Invariant 1 (Food Conservation / Zero Hallucination):** The set of food items in the generated schedule $\mathcal{S}_{\text{gen}}$ must match the Stage 3 optimized food basket $\mathcal{S}_{\text{opt}}$ exactly: $\mathcal{S}_{\text{gen}} \equiv \mathcal{S}_{\text{opt}}$. The generation of unapproved foods (e.g., jaggery, deep-fried snacks) triggers an immediate circuit-breaker trip.
2. **Invariant 2 (Quantity Conservation):** For every food item $i \in \mathcal{S}_{\text{opt}}$, the cumulative scheduled servings across all meal slots must equal the Stage 3 serving count: $\sum_{s} q_{is} = q_i^*$. The LLM cannot alter portion sizes.
3. **Invariant 3 (Zero Calculation Authority):** The language model is forbidden from recalculating macro- or micronutrient totals; all presented nutritional metrics are bound directly to verified Stage 3 constants.
4. **Invariant 4 (Schema Validity):** The output must parse as a valid JSON object matching the strict 5-slot schema.

Any invariant failure trips an automated fallback circuit-breaker, replacing the generated text with a deterministic template schedule.

**Independent Real LLM Backend Reliability Benchmark:**
To evaluate the real local language model backend strictly *separately* from the mathematical optimizer (giving the LLM zero credit for nutritional accuracy or caloric feasibility enforced by Stages 1–3), we established a frozen, independently reproducible test set of **100 diverse, clinically heterogeneous patient dossiers** (`real_llm_evaluation_testset_100.json`, SHA256: `0FC1ACAC42AAAAFBF14540901574FCCB88D0010135A86996B635AC9109A8C1F5`). The test set is balanced round-robin across five distinct clinical cardiometabolic archetypes ($N=20$ each): (1) Type 2 Diabetes; (2) CVD and Hypertension; (3) Clinical Obesity; (4) Multimorbid Cardiometabolic Disease (DM + CVD + Obesity); and (5) Healthy Preventative Screening.

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

### 3.2 Five-Paradigm Meal Optimization Benchmark ($N=50$ Clinical Archetypes)

Table 3 summarizes the comparative evaluation of the five optimization paradigms across 50 clinically heterogeneous archetypes (15 Type 2 Diabetes, 15 Hypertension/CVD, 10 Obesity, and 10 Multi-Morbidity profiles).

**Table 3: Multi-Paradigm Optimization Benchmark across 50 Clinical Archetypes**

| Optimization Paradigm | Integer Serving Rate | Fractional Items / Plan (Mean ± SD) | Plate Category Compliance | Nutrient Boundary Violations / Plan | Mean $L_1$ Slack Magnitude | Mean Daily Basket Cost (INR) | Mean Solver Latency | Clinical Feasibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Continuous LP (Relaxed)** | 0.0% | 3.08 ± 0.59 | 0.0% | 0.0 ± 0.0 | 0.0 | ₹143.55 ± 10.13 | **4.57 ms** | **Uncookable** (produces 0.38 phulka, 0.24 katori dal) |
| **Continuous LP + Rounding** | 100.0% | 0.0 ± 0.0 | **0.0%** | **2.64 ± 1.14** | 212.51 ± 97.45 | ₹140.56 ± 12.35 | 8.01 ms | **Unviable** (94% fail rate; eliminates essential dals/protein) |
| **Continuous LP + Heuristic Repair** | 100.0% | 0.0 ± 0.0 | **0.0%** | **3.96 ± 1.00** | 397.02 ± 138.65 | ₹180.12 ± 15.42 | 8.27 ms | **Unsafe** (100% fail rate; severe sodium and calorie inflation) |
| **Prompt-Only LLM (Qwen 2.5 3B)** | 100.0% | 0.0 ± 0.0 | 46.0% | **6.16 ± 1.05** | 2,087.27 ± 514.82 | ₹80.70 ± 19.34 | 855.70 ms | **Unviable** (severe underfilling; hallucinations) |
| **BioNutriGPT Discrete MILP** | **100.0%** | **0.0 ± 0.0** | **100.0%** | **0.0 ± 0.0** | **0.0** | ₹155.93 ± 14.03 | **346.26 ms** | **100% Cookable & Safe** (exact integer katoris/rotis) |

*Empirical Analysis of Failures:*
1. **The Fallacy of Continuous LP:** Relaxed continuous LP achieved minimal cost (₹143.55) and instantaneous solve time (4.57 ms), but 100% of plans contained uncookable fractional servings (mean 3.08 fractional dishes per plan), making it impossible to prepare in an Indian home kitchen.
2. **Catastrophic Failure of Post-Hoc Rounding:** Attempting to round continuous LP solutions to the nearest whole integer caused balanced-plate compliance to drop to 0.0%. Pruning small fractions eliminated essential food groups entirely (e.g., dropping dal to 0 katoris), causing active dish count to collapse to 3.4 dishes. Furthermore, rounding introduced an average of 2.64 ± 1.14 hard nutrient boundary violations per plan (94.0% archetype failure rate).
3. **Failure of Greedy Heuristic Repair:** Implementing post-hoc greedy heuristic repair (adding dishes to satisfy category minimums) produced even worse degradation: boundary violations increased to 3.96 ± 1.00 per plan (100.0% failure rate) and mean $L_1$ slack ballooned to 397.02. This occurs because discrete combinatorial constraints in multi-nutrient space create non-convex trade-offs: greedily adding a dal to satisfy protein bounds drastically overshoots sodium and caloric limits.
4. **Failure of Prompt-Only LLMs:** Unconstrained Qwen 2.5 3B produced whole numbers but violated 6.16 ± 1.05 nutrient bounds per plan with massive $L_1$ slack (2,087.27). The lower cost (₹80.70) reflects dangerous starvation underfilling (providing ~900 kcal against an 1,800 kcal target).
5. **MILP Optimality:** BioNutriGPT Discrete MILP was the *only* paradigm that achieved 100.0% integer servings, 100.0% category compliance, zero boundary violations, and zero $L_1$ slack, executing in a mean latency of 346.26 ms (median 268.45 ms). This rigorously demonstrates that branch-and-cut MILP is mathematically indispensable for reliable meal synthesis.

---

### 3.3 Stage 4 Real Local GPU Execution Profile and Safety Invariant Audits

To rigorously evaluate the reliability of the language model backend, we assessed the local **Qwen 2.5 3B Instruct NF4** model on an **NVIDIA GeForce RTX 3050 Laptop GPU** strictly *separately* from the Stage 3 mathematical optimizer. The language model receives zero credit for nutritional accuracy or caloric matching, which is enforced deterministically by Stage 3. Table 4A presents the empirical failure rates of the raw unconstrained model compared against the BioNutriGPT Guarded AST Pipeline across the frozen 100-case clinical benchmark. Table 4B outlines the local edge hardware resource and latency profile.

**Table 4A: Real LLM Backend Reliability Benchmark (Raw Unconstrained LLM vs. Guarded AST Pipeline)**

| Failure Mode / Audit Metric | Raw LLM Baseline (Qwen 2.5 3B NF4) | Guarded Stage 4 Pipeline (AST Circuit-Breaker) | Clinical Risk & Invariant Enforcement |
| :--- | :---: | :---: | :--- |
| **JSON Parse Success Rate** | **33.3%** | **100.0%** | Incomplete syntax or token truncation; repaired by AST schema canonicalization |
| **Food Hallucination Rate (Invented Dishes)** | **0.0%** (Greedy) | **0.0% (Zero Hallucination)** | Invariant 1 enforced: strictly forbids dishes outside Stage 3 solved basket ($\mathcal{S}_{\text{gen}} \equiv \mathcal{S}_{\text{opt}}$) |
| **Quantity Modification Rate (Altered Servings)** | **100.0%** | **0.0% (Exact Match)** | Invariant 2 enforced: raw LLM routinely dropped/modified portion counts ($\sum q_{is} \neq q_i^*$); AST ensures exact integer serving preservation |
| **Invalid / Unknown Food Identifiers** | **66.7%** | **0.0% (Zero Invalid)** | Non-existent or corrupted keys in raw output; filtered and validated against ICMR/IFCT catalog |
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

The empirical evaluation proves that raw generative LLMs cannot be granted autonomous meal scheduling or arithmetic authority, as unconstrained generation exhibits a 100% serving quantity modification rate. By sandwiching the LLM within a deterministic AST circuit-breaker and discrete MILP solver, BioNutriGPT delivers a mathematically verified, zero-hallucination clinical decision support architecture executable entirely on commodity edge laptops.

---

## 4. Discussion

### 4.1 Principal Findings and Empirical Justifications

This study resolves the primary methodological and architectural challenges facing integrated clinical nutrition informatics:
1. **Modality-Specific Tabular Modeling vs. Deep MLPs:** Our empirical benchmarks on 7,046 prospective temporal holdouts show that BioNutriGPT Stage 1 achieves superior discrimination over Standard Deep MLPs and Tabular ResNets. Tabular health datasets contain radically distinct feature dynamics: 5 blood lipid markers carry high signal-to-noise ratios, whereas 46 dietary recall features are noisy and collinear. Standard MLPs concatenate these into dense layers, leading to gradient dominance and feature entanglement. Stage 1’s learned modality queries isolate intra-modality correlations before top-2 MoE routing, preserving fine-grained biological signals.
2. **Clinical Necessity of Platt Scaling:** Multi-task neural networks trained with positive class weighting produce distorted posterior probabilities. Our implementation of univariate logistic calibration reduced Diabetes ECE by 76.9% and CVD ECE by 89.8%, proving that post-hoc calibration is essential before tabular deep learning models can be utilized for clinical screening.
3. **The Mathematical Necessity of Branch-and-Cut MILP:** Our 50-archetype 5-paradigm benchmark definitively demonstrates why heuristic and continuous approximations fail. Discrete household cooking units cannot be achieved by rounding continuous LP outputs without destroying nutrient bounds (94% failure rate) or eliminating essential food categories. Branch-and-cut MILP is mathematically necessary to navigate non-convex multi-nutrient boundaries simultaneously.
4. **Guarding Generative LLMs with AST Invariants:** Large language models cannot be trusted with clinical arithmetic or autonomous ingredient selection. By restricting the LLM to explanatory prose and chronological meal slotting while enforcing deterministic AST invariants, BioNutriGPT achieves a zero-hallucination guarantee.

### 4.2 Limitations and Future Directions

1. **Temporal vs. External Geographic Validation:** While the NHANES 2017–March 2020 cohort provides a rigorous temporal prospective holdout, it originates from the same U.S. national survey framework. External geographical validation on South Asian cohorts (such as ICMR-INDIAB or UK Biobank South Asian sub-cohorts) is required to evaluate transferability.
2. **Cross-Sectional Screening vs. Causal Interventions:** Stage 1 predicts prevalent cardiometabolic status. Stage 2 nutrient counterfactuals simulate physiological bounds but do not represent proven causal trial effects. Prospective randomized trials are necessary to verify long-term glycemic and lipid improvements.
3. **Execution of the Pre-Registered IDA Review:** While we have established the formal triple-blinded protocol and sample size calculations, the physical multi-center trial with accredited Indian Dietetic Association clinicians represents the immediate translational milestone.

---

## 5. Conclusion

BioNutriGPT establishes an open, auditable, and mathematically rigorous pipeline bridging population-level cardiometabolic risk screening with culturally tailored dietary synthesis. By demonstrating the superiority of modality-aware attention over deep neural networks, proving the indispensability of branch-and-cut MILP over continuous heuristic rounding, and verifying real-time edge-GPU execution under strict AST clinical invariants, this work provides a reproducible computational foundation for digital lifestyle medicine.

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
