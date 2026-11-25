# Stage 2 Counterfactual Optimization Engine — Component Map

## Overview
This document maps every computational step, code path, and mathematical operation in the BioNutriGPT Stage 2 Counterfactual Prescription Engine (`stage2_counterfactual_engine.py` under the `repaired_v1` deployment binding).

Each component is categorized into:
- **Category A (Candidate Constructor / Modifier):** Operations that create, alter, or transform candidate nutrient values.
- **Category B (Evaluator / Gate / Auditor):** Operations that evaluate, audit, accept, or reject candidate nutrient values without altering their coordinates.

---

## 1. Execution Flow & Component Mapping

| Step | Function / Code Location | Description | Category | Mathematical / Algorithmic Specification |
| :--- | :--- | :--- | :---: | :--- |
| **1.1** | `canonicalize_dietary_ratios` (lines 668–709) | Input Validation & Biomarker Derivation | **A** | Validates 88-feature input vector; enforces missingness indicator constraints (`_MISS == 0`), age $\ge 20$, categorical one-hot encoding, binary encoding $\{0,1\}$, and lipid consistency ($TC \ge HDL > 0$). Recomputes derived biomarkers: $NON\_HDL = TC - HDL$, $CHOL\_HDL\_RATIO = TC / HDL$, $NON\_HDL\_TO\_HDL = (TC - HDL) / HDL$, $SLEEP\_DEV = \|SLEEP - 7.5\|$. Derives all 7 dietary ratio features from raw intake. |
| **1.2** | `_validate_patient_metadata` (lines 192–243) | Metadata Integrity | **B** | Enforces positive weight, height, age $\ge 18$, known physical activity level ($PAL \in \{1.20, 1.375, 1.55, 1.725\}$), and exclusion of unmodeled special populations (renal disease, pregnancy, lactation). |
| **1.3** | `_load_stage1_model` & `deployment.wrap_model` (lines 435–457) | Frozen Neural Model & Platt Calibrators | **B** | Loads frozen `BioNutriGPTStage1` checkpoint (`model.pt`) and wraps with Platt scaling calibrators (`calibrators.joblib`) for calibrated probabilities $p_d, p_o, p_c$. Parameters frozen (`requires_grad=False`, `eval()`). |
| **1.4** | `_normalize_numpy` & `_normalize_torch` (lines 325–433) | Differentiable Feature Normalization | **A** | `log1p` on declared skewed columns, clip to `clip_bounds`, and apply fitted `StandardScaler` mean and scale. In repaired mode, straight-through 6-decimal rounding preserves floating-point alignment with offline evaluation. |
| **1.5** | `_predict_probabilities` (lines 489–501) | Baseline Risk Scoring | **B** | Predicts baseline risk probabilities $p_{d,0}, p_{o,0}, p_{c,0}$ for diabetes, obesity, and CVD. |
| **2.1** | Caloric Budget Calculation (lines 1321–1337) | Energy Target Prescription | **B** (sets bounds) | Calculates Mifflin-St Jeor BMR: $10.0 \times \text{weight} + 6.25 \times \text{height} - 5.0 \times \text{age} + (-161.0 \text{ if female else } +5.0)$. $TDEE = BMR \times PAL$. If $p_{o,0} \ge 0.5$, $t_{\text{kcal}} = \max(TDEE - 500, 1200)$; else $\max(TDEE, 1200)$. |
| **2.2** | `_build_nutrient_bounds` (lines 538–628) | Participant Nutrient Constraint Envelope | **B** (sets bounds) | Constructs component bounds $[lower, upper]$ for all 39 nutrients based on $t_{\text{kcal}}$, weight, sex, age, and baseline disease routing ($p_{d,0}, p_{o,0}, p_{c,0} \ge 0.5$). Incorporates RDA/AI lower bounds and UL upper bounds. |
| **3.1** | `_run_counterfactual_optimization` (lines 766–844) | Adam Search Initialization | **A** | Initializes decision variable $\delta N = \mathbf{0} \in \mathbb{R}^{39}$ (`requires_grad=True`), starting from observed patient intake $N_{\text{current}} = \text{raw}[6:45]$. Configures `torch.optim.Adam([delta_N], lr=0.05)`. |
| **3.2** | Adam Objective Evaluation (lines 790–820) | Loss Function Computation | **A** | Computes multi-term scalar loss:<br>$L = 2.0 p_c + 2.0 p_d + 1.0 p_o$<br>$+ 50.0 \sum_{k} \text{ReLU}(p_k - p_{k,0})$ (Pareto penalty)<br>$+ 0.001 \|\delta N\|_2^2$ (Dietary-change penalty)<br>$+ 10.0 \sum_i [\text{ReLU}(lower_i - N_i)^2 + \text{ReLU}(N_i - upper_i)^2]$ (Barrier penalty). |
| **3.3** | Gradient Descent Loop (lines 789–830) | 50 Adam Iterations | **A** | Computes autograd backpropagation `loss.backward()` and executes `optimizer.step()` for exactly 50 steps without early stopping. |
| **3.4** | Post-Adam Clamping (line 832) | Component Box Clamping | **A** | Clamps candidate to component box: $N_{\text{adam}} = \text{clamp}(N_{\text{current}} + \delta N, lower, upper)$. |
| **4.1** | `_build_tightened_joint_bounds` (lines 862–891) | Filter 2 Invariant Bounding Box | **B** (sets bounds) | Tightens bounding box with clinical safety invariants: $KCAL \ge 1200$, $PROT \in [0.8\text{wt}, 2.0\text{wt}]$, $SODI \le 1500/2300$, $POTA \ge 3500/2600$. |
| **4.2** | `_build_linear_constraint_matrices` (lines 893–929) | Linear Inequality Matrix ($A_{\text{ub}} n \le b_{\text{ub}}$) | **B** (sets constraints) | Formulates 5 linear inequalities: thermodynamic energy balance within $\pm 0.99\%$ envelope ($4P + 4C + 9F + 7A \approx KCAL$), and clinical ratio ceilings for sugar/fiber, sodium/potassium, and sat/unsat fat relative to baseline. |
| **4.3** | `_project_to_joint_feasible_polytope` (lines 931–1070) | Convex Quadratic Joint Projection | **A** | Solves $\min_n \frac{1}{2} \sum_{i=0}^{38} \left(\frac{n_i - n_{\text{adam},i}}{s_i}\right)^2$ subject to $lower_{\text{tight}} \le n \le upper_{\text{tight}}$ and $A_{\text{ub}} n \le b_{\text{ub}}$. Uses HiGHS LP to find a guaranteed feasible witness $x_0$, then runs SciPy SLSQP QP from $x_0$. If SLSQP fails to converge or exceeds residual tolerance $10^{-6}$, safely falls back to $x_0$ (`status: feasible_lp_fallback`). |
| **5.1** | `_apply_filter_1_thermodynamic_balance` (lines 1072–1097) | Thermodynamic Energy Audit | **B** (in repaired mode) | Checks $|4P + 4C + 9F + 7A - KCAL| / KCAL \le 0.01$. **Code-Path Finding:** In repaired deployment (`if adjusted and not deployment.REPAIRED:`), Filter 1 does NOT modify $KCAL$. It operates purely as an audit check. |
| **5.2** | `_apply_filter_2_iom_safety` (lines 1099–1142) | IOM Safety Clamp Audit | **B** (when preceded by projection) | Clamps $KCAL \ge 1200$, $SODI \le 1500/2300$, $POTA \ge 3500/2600$, $PROT \in [0.8\text{wt}, 2.0\text{wt}]$. **Code-Path Finding:** Because Step 4.1 already tightened the bounding box to these exact values, Filter 2 introduces 0 modifications when projection is present. |
| **5.3** | `_apply_filter_3_ratio_coherence` (lines 1144–1235) | Clinical Ratio Adjustment Audit | **B** (when preceded by projection) | Increases fiber, increases potassium, or reduces saturated fat if ratios exceed baseline. **Code-Path Finding:** Because Step 4.2 strictly enforces these linear inequalities during projection, Filter 3 introduces 0 modifications when projection is present. |
| **5.4** | `_audit_final_filter_status` (lines 1237–1265) | Sequential Invariant Re-audit | **B** | Re-evaluates Filter 1, 2, and 3 mathematical invariants on the final vector. |
| **6.1** | Closed-Loop Rescoring (lines 1404–1415) | Post-Intervention Risk Scoring | **B** | Constructs 88-feature candidate vector with target nutrients, normalizes, and passes through frozen Stage 1 model to predict $p_d^*, p_o^*, p_c^*$. |
| **6.2** | Model-Score Acceptance Gate (lines 1424–1432) | Risk Reduction Gate | **B** | Enforces $\Delta p_d \le 0.0015$, $\Delta p_o \le 0.0015$, $\Delta p_c \le 0.0015$. |
| **6.3** | Overall Certification (lines 1481–1483) | Stage 2 Certification Decision | **B** | Passes if all filters (1, 2, 3), projection audit, and model-score gate pass. |
| **7.1** | External Numerical Validator (`stage2_comparison_checks_v2.check`) | Independent Reconstructive Validator | **B** | Completely independent check of all 39 bounds, energy balance ($\pm 1.00\%$), ratio rules, and Filter 2 invariants under a documented $10^{-6}$ absolute tolerance. |

---

## 2. Key Code-Path Findings Regarding Named Components

1. **Dietary-Change Penalty ($0.001 \|\delta N\|_2^2$):**
   - **Status:** **Present in executable code.**
   - Location: Line 817 of `stage2_counterfactual_engine.py`.
   - Implementation: Added as an unscaled $L_2$ Euclidean regularization term on the decision variable vector $\delta N \in \mathbb{R}^{39}$ with a fixed weight of $0.001$.
   - Note: It does not use `DAILY_REFERENCE_SCALES` during Adam; normalization scales are applied separately during the joint projection step.

2. **Sequential Filters 1–3 Redundancy Under Joint Projection:**
   - **Status:** **Empirically and mathematically redundant when preceded by Step 4.5.**
   - In repaired mode (`deployment.REPAIRED`), Filter 1's legacy unilateral $KCAL$ overwrite is disabled (`if adjusted and not deployment.REPAIRED:`).
   - In Step 4, the joint projection solves a constrained QP whose feasible region is the intersection of the Filter 2 tightened bounds, the thermodynamic energy balance envelope, and the ratio constraint inequalities.
   - Consequently, when Step 4.5 runs, Filters 1, 2, and 3 find all invariants pre-satisfied. Across all 100 cases in the comparison study, Filters 2 and 3 made exactly 0 modifications.
   - Variant 5 in the ablation study explicitly tests this invariance by bypassing Filters 1–3 after joint projection.

3. **Joint Projection Necessity:**
   - **Status:** **Essential for numerical feasibility.**
   - Unprojected Adam outputs (`n_adam_clipped`) satisfy individual component bounds $[lower, upper]$ via box clamping, but violate the coupled linear constraints (energy balance and ratio envelopes).
   - Variant 4 explicitly disables joint projection to quantify this failure mode.
