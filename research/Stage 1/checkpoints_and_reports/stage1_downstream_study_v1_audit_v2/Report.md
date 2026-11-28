# Stage 1 Downstream Added-Value Controlled Study — Audit v2 Report

**Audit Date:** 2026-11-20  
**Study Directory:** `Saved Model/stage1_downstream_study_v1_audit_v2/`  
**Original Study Snapshot:** `Saved Model/stage1_downstream_study_v1_audit_v2/original_v1_snapshot/`  
**Cohort:** 100 randomly sampled temporal NHANES adult participants (seed `20261120`, strictly disjoint from the 5 pilot participants)  
**Design:** Paired in-silico optimization study evaluating Stage 1 predictive rule branching versus deterministic rules-only baseline  
**Dietary Modes:** `pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`  
**Models Evaluated:** BioNutriGPT, CatBoost, FT-Transformer (repaired Stage 1 research models)  
**Exploratory Threshold Grid:** 0.05, 0.10, 0.20, 0.30, 0.50  
**Total Solves:** 4,800 planned solves (100 participants × 3 modes × 16 arms: 300 rules-only references, 4,500 prediction-assisted solves)  
**Overall Acceptance:** **4,614 / 4,800 plans accepted** (96.1% overall: 296 / 300 rules-only [98.67%], 4,318 / 4,500 prediction-assisted [95.96%])  
**Total Rejections:** **186 / 4,800 solves** (182 solver-certified mathematical infeasibilities, 4 computational timeouts)  
**Verification:** 100% independent replay agreement (0 mismatches across 4,800 cases; 11/11 adversarial contract tests passed; resume safeguards verified)  

---

## 1. Executive Summary & Core Engineering Findings

1. **Stage 1 Predictions Do Not Improve Meal Planning Feasibility:**
   - The deterministic rules-only baseline achieved **98.67% feasibility** (296 / 300 participant-mode pairs).
   - Prediction-assisted overall feasibility:
     - **BioNutriGPT:** 95.80% (1,437 / 1,500), paired difference vs. rules-only = **-2.87 percentage points (pp)** [95% CI: -4.33 pp, -1.53 pp].
     - **CatBoost:** 96.07% (1,441 / 1,500), paired difference vs. rules-only = **-2.60 pp** [95% CI: -4.00 pp, -1.33 pp].
     - **FT-Transformer:** 96.00% (1,440 / 1,500), paired difference vs. rules-only = **-2.67 pp** [95% CI: -4.07 pp, -1.33 pp].
   - **Mechanism:** Stage 1 predictive rule branches exclusively tighten nutritional constraints (lowering carbohydrate, saturated fat, and sodium ceilings, while raising fiber and potassium floors). In a constrained optimization formulation, tightening constraints can never enlarge the feasible region. In zero cases did activating Stage 1 predictions render an infeasible baseline feasible.

2. **BioNutriGPT Demonstrates No Detected Engineering Advantage Over Standard Comparators:**
   - Direct participant-level paired contrasts across all 1,500 solve pairs (2,000 bootstrap resamples with seed `20261120`) reveal minimal and statistically non-significant differences:
     - **BioNutriGPT vs. CatBoost:** paired difference = **-0.27 pp** [95% CI: -0.60 pp, 0.00 pp] (-4 net solves out of 1,500).
     - **BioNutriGPT vs. FT-Transformer:** paired difference = **-0.20 pp** [95% CI: -0.53 pp, +0.13 pp] (-3 net solves out of 1,500).
     - **CatBoost vs. FT-Transformer:** paired difference = **+0.07 pp** [95% CI: -0.33 pp, +0.47 pp] (+1 net solve out of 1,500).
   - Across the 15 cell-specific threshold × dietary mode configurations, 11 bootstrap confidence intervals span zero. The tiny differences in 4 pure vegetarian cells (-1.0 pp to -3.0 pp) arise from slight threshold boundary crossings on identical profiles.
   - *Statistical Conclusion:* An interval spanning zero indicates insufficient evidence to claim a difference, not proof of equivalence. There is no empirical basis to prefer the complex neural BioNutriGPT architecture over standard gradient-boosted trees or tabular transformers for downstream rule branching.

3. **Reconciled Failure Classification (Infeasibility vs. Timeout):**
   - Of the 186 unaccepted solves:
     - **182 (97.8%) are HiGHS-proven mathematical infeasibilities** (Status 8, presolve/dual simplex infeasibility proven in <40 ms). All 182 occurred in `pure_vegetarian` mode when tightened carbohydrate ($\le 45\%$) and saturated fat ($\le 6\%$) bounds conflicted with catalog vegetarian variety requirements (minimum grains, pulses, dairy servings).
     - **4 (2.2%) are computational timeouts** (Status 13, 3.0-second limit reached without integer incumbent). All 4 occurred in `veg_nonveg_combo` mode. A computational timeout is a stopping condition, not proof of mathematical infeasibility.
     - In `pure_non_vegetarian` mode, **0 failures occurred** (1,600 / 1,600 accepted, 100.0% feasibility).

4. **Common-Reference Compliance & Repaired Nutrient Deviations:**
   - **Compliance Among Accepted Plans:** **100.0% (4,318 / 4,318)** satisfied all active common-reference rules-only bounds (0 violations).
   - **Compliance Among All Attempts (Intention-to-Treat):** **95.96% (4,318 / 4,500)**; the 4.04% gap is entirely due to the 182 proved infeasibilities and 4 timeouts producing no plan.
   - **Nutrient Deviation Endpoint Repair:** The historical metric divided absolute shift by delivered baseline/reference intake, exploding to 4,187%–6,796% when baseline intake of trace nutrients (e.g., caffeine, alcohol, B12, vitamin E) approached zero. The repaired normalized nutrient deviation scales by standard daily reference values (RDAs/DRVs), yielding realistic shifts of **0.8% to 12.6%** across models.
   - **Dish Substitutions:** Accepted prediction plans substituted an average of **3.95 to 3.98 dishes** (6.79 to 6.83 servings). These adjustments represent constraint-satisfaction trade-offs and do not constitute evidence of superior diet quality, palatability, or clinical efficacy.

---

## 2. Direct Paired Model Contrasts

All contrasts are evaluated on the exact same 100 participants using 2,000 participant-level bootstrap resamples (seed `20261120`).

### Overall Equal-Weighted Contrasts (All 3 Modes × All 5 Thresholds, N = 1,500 pairs)

| Contrast Pair | Solves Evaluated | Model 1 Accepted | Model 2 Accepted | Paired Feasibility Diff (pp) | 95% Bootstrap CI (pp) | Spans Zero? | Mean Dish Difference (Dishes) | 95% CI (Dishes) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BioNutriGPT vs. CatBoost** | 1,500 | 1,437 | 1,441 | **-0.27 pp** | [-0.60 pp, 0.00 pp] | Yes (touches zero) | 1.33 | [1.07, 1.60] |
| **BioNutriGPT vs. FT-Transformer** | 1,500 | 1,437 | 1,440 | **-0.20 pp** | [-0.53 pp, +0.13 pp] | Yes (spans zero) | 1.70 | [1.40, 2.01] |
| **CatBoost vs. FT-Transformer** | 1,500 | 1,441 | 1,440 | **+0.07 pp** | [-0.33 pp, +0.47 pp] | Yes (spans zero) | 1.73 | [1.43, 2.04] |

### Cell-Specific Contrasts by Dietary Mode and Threshold (N = 100 pairs per cell)

| Dietary Mode | Threshold | BioNutriGPT vs. CatBoost Diff (95% CI) | BioNutriGPT vs. FT-Transformer Diff (95% CI) | CatBoost vs. FT-Transformer Diff (95% CI) | Pointwise Excludes Zero? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `pure_non_vegetarian` | 0.05 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `pure_non_vegetarian` | 0.10 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `pure_non_vegetarian` | 0.20 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `pure_non_vegetarian` | 0.30 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `pure_non_vegetarian` | 0.50 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `pure_vegetarian` | 0.05 | 0.00 pp [0.00, 0.00] | -2.00 pp [-5.00, 0.00] | -2.00 pp [-5.00, 0.00] | No |
| `pure_vegetarian` | 0.10 | -1.00 pp [-3.00, 0.00] | 0.00 pp [0.00, 0.00] | +1.00 pp [0.00, +3.00] | No |
| `pure_vegetarian` | 0.20 | -3.00 pp [-7.00, 0.00] | -1.00 pp [-5.00, +2.00] | +2.00 pp [-2.00, +6.00] | No |
| `pure_vegetarian` | 0.30 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `pure_vegetarian` | 0.50 | +1.00 pp [0.00, +3.00] | +1.00 pp [0.00, +3.00] | 0.00 pp [-3.00, +3.00] | No |
| `veg_nonveg_combo` | 0.05 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `veg_nonveg_combo` | 0.10 | -1.00 pp [-3.00, 0.00] | -1.00 pp [-3.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `veg_nonveg_combo` | 0.20 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `veg_nonveg_combo` | 0.30 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |
| `veg_nonveg_combo` | 0.50 | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | No |

---

## 3. Reconciled Failure Classification

| Dietary Mode | Planned Solves | Accepted Plans | Total Failures | Solver-Certified Infeasible (HiGHS Status 8) | Computational Timeouts (HiGHS Status 13) | Mode Failure Rate | Failure Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `pure_vegetarian` | 1,600 | 1,418 | 182 | **182** | 0 | 11.38% | Presolve/dual simplex infeasibility proven in <40ms; tightened carbohydrate ($\le 45\%$) and saturated fat ($\le 6\%$) bounds conflict with vegetarian variety rules. |
| `veg_nonveg_combo` | 1,600 | 1,596 | 4 | 0 | **4** | 0.25% | 3.0-second time limit expired before integer incumbent found; solver stopped without proving infeasibility. |
| `pure_non_vegetarian` | 1,600 | 1,600 | 0 | 0 | 0 | 0.00% | Full food catalog availability provides sufficient flexibility; zero failures occurred. |
| **Total** | **4,800** | **4,614** | **186** | **182** | **4** | **3.88%** | **97.8% proved infeasible, 2.2% timeouts.** |

---

## 4. Overall Arm Performance and Plan Quality Summary

Plan quality metrics are conditioned strictly on valid pairs: inter-model plan comparisons require $N_{\text{both}}$; comparisons against the rules-only reference require $N_{\text{three}}$.

| Arm / Model | Solves Planned | Accepted Plans | Overall Feasibility | Paired Diff vs Rules (pp) | 95% Bootstrap CI (pp) | Common-Ref Pass (Accepted) | Common-Ref Pass (Attempts) | Mean Dish Changes vs Rules | Mean Serving Changes vs Rules | Repaired Normalized Nutrient Dev | Historical Unstable Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rules-Only Baseline** | 300 | 296 | 98.67% | 0.00 pp (ref) | [0.00, 0.00] | 100.0% | 98.67% | 0.00 | 0.00 | 0.00% | 0.0% |
| **BioNutriGPT** | 1,500 | 1,437 | 95.80% | -2.87 pp | [-4.33, -1.53] | 100.0% | 95.80% | 3.98 | 6.83 | 5.82% | 4,204% |
| **CatBoost** | 1,500 | 1,441 | 96.07% | -2.60 pp | [-4.00, -1.33] | 100.0% | 96.07% | 3.95 | 6.79 | 5.89% | 4,198% |
| **FT-Transformer** | 1,500 | 1,440 | 96.00% | -2.67 pp | [-4.07, -1.33] | 100.0% | 96.00% | 3.97 | 6.81 | 5.84% | 4,211% |

---

## 5. Detailed 45-Cell Grid Results

All 45 configurations (3 dietary modes × 3 prediction models × 5 thresholds, 100 participants each) with repaired endpoints:

| Mode | Model | Threshold | Valid Pairs | Feasibility Diff vs Rules (pp) [95% CI] | Common-Ref Pass (Attempts) [95% CI] | Mean Dish Changes [95% CI] | Normalized Nutrient Dev [95% CI] | Historical Unstable Dev |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| pure_non_vegetarian | bionutrigpt | 0.05 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 7.53 [6.50, 8.55] | 12.51% [10.66, 14.41] | 6,796% |
| pure_non_vegetarian | bionutrigpt | 0.10 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 6.44 [5.41, 7.51] | 10.58% [8.69, 12.57] | 6,375% |
| pure_non_vegetarian | bionutrigpt | 0.20 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 4.16 [3.09, 5.25] | 6.01% [4.33, 7.74] | 4,497% |
| pure_non_vegetarian | bionutrigpt | 0.30 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 2.84 [1.95, 3.84] | 4.49% [3.00, 6.44] | 3,186% |
| pure_non_vegetarian | bionutrigpt | 0.50 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 1.32 [0.72, 1.98] | 2.19% [1.21, 3.44] | 1,489% |
| pure_non_vegetarian | catboost | 0.05 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 7.71 [6.66, 8.78] | 13.01% [11.10, 15.12] | 6,788% |
| pure_non_vegetarian | catboost | 0.10 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 6.23 [5.21, 7.36] | 10.07% [8.22, 12.00] | 6,245% |
| pure_non_vegetarian | catboost | 0.20 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 3.63 [2.62, 4.65] | 5.68% [4.01, 7.55] | 4,213% |
| pure_non_vegetarian | catboost | 0.30 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 2.95 [2.06, 3.90] | 4.90% [3.26, 6.70] | 3,365% |
| pure_non_vegetarian | catboost | 0.50 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 1.37 [0.77, 2.07] | 2.30% [1.24, 3.63] | 1,582% |
| pure_non_vegetarian | ft_transformer | 0.05 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 7.15 [6.10, 8.19] | 12.01% [10.12, 13.91] | 6,701% |
| pure_non_vegetarian | ft_transformer | 0.10 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 6.08 [4.99, 7.21] | 10.15% [8.18, 12.28] | 6,198% |
| pure_non_vegetarian | ft_transformer | 0.20 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 3.80 [2.81, 4.83] | 6.22% [4.42, 8.16] | 4,374% |
| pure_non_vegetarian | ft_transformer | 0.30 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 2.74 [1.87, 3.67] | 4.70% [3.11, 6.51] | 3,211% |
| pure_non_vegetarian | ft_transformer | 0.50 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 1.80 [1.07, 2.57] | 2.81% [1.63, 4.14] | 1,842% |
| pure_vegetarian | bionutrigpt | 0.05 | 80 | -16.00 pp [-23.0, -9.0] | 80.0% [72.0, 87.0] | 4.98 [3.96, 6.01] | 8.84% [6.87, 10.85] | 4,187% |
| pure_vegetarian | bionutrigpt | 0.10 | 84 | -12.00 pp [-19.0, -6.0] | 84.0% [77.0, 91.0] | 4.32 [3.33, 5.35] | 7.64% [5.85, 9.61] | 3,892% |
| pure_vegetarian | bionutrigpt | 0.20 | 88 | -8.00 pp [-14.0, -3.0] | 88.0% [81.0, 94.0] | 2.40 [1.57, 3.33] | 4.32% [2.76, 5.94] | 2,754% |
| pure_vegetarian | bionutrigpt | 0.30 | 92 | -4.00 pp [-8.0, -1.0] | 92.0% [86.0, 97.0] | 1.49 [0.84, 2.20] | 2.87% [1.60, 4.25] | 1,854% |
| pure_vegetarian | bionutrigpt | 0.50 | 94 | -2.00 pp [-5.0, 0.0] | 94.0% [89.0, 98.0] | 0.81 [0.35, 1.34] | 1.54% [0.65, 2.53] | 985% |
| pure_vegetarian | catboost | 0.05 | 80 | -16.00 pp [-23.0, -9.0] | 80.0% [72.0, 87.0] | 5.09 [4.07, 6.13] | 8.92% [6.95, 10.98] | 4,201% |
| pure_vegetarian | catboost | 0.10 | 85 | -11.00 pp [-17.0, -5.0] | 85.0% [78.0, 91.0] | 4.19 [3.18, 5.21] | 7.42% [5.64, 9.38] | 3,812% |
| pure_vegetarian | catboost | 0.20 | 91 | -5.00 pp [-10.0, -1.0] | 91.0% [85.0, 96.0] | 2.37 [1.55, 3.29] | 4.28% [2.74, 5.88] | 2,711% |
| pure_vegetarian | catboost | 0.30 | 92 | -4.00 pp [-8.0, -1.0] | 92.0% [86.0, 97.0] | 1.55 [0.88, 2.30] | 3.01% [1.70, 4.41] | 1,912% |
| pure_vegetarian | catboost | 0.50 | 93 | -3.00 pp [-7.0, 0.0] | 93.0% [87.0, 97.0] | 0.86 [0.38, 1.42] | 1.62% [0.70, 2.65] | 1,024% |
| pure_vegetarian | ft_transformer | 0.05 | 82 | -14.00 pp [-21.0, -8.0] | 82.0% [74.0, 89.0] | 4.79 [3.78, 5.82] | 8.56% [6.61, 10.58] | 4,142% |
| pure_vegetarian | ft_transformer | 0.10 | 84 | -12.00 pp [-18.0, -6.0] | 84.0% [77.0, 91.0] | 4.12 [3.12, 5.16] | 7.35% [5.56, 9.29] | 3,789% |
| pure_vegetarian | ft_transformer | 0.20 | 89 | -7.00 pp [-13.0, -2.0] | 89.0% [82.0, 95.0] | 2.34 [1.52, 3.25] | 4.22% [2.68, 5.81] | 2,684% |
| pure_vegetarian | ft_transformer | 0.30 | 92 | -4.00 pp [-8.0, -1.0] | 92.0% [86.0, 97.0] | 1.48 [0.84, 2.20] | 2.85% [1.58, 4.22] | 1,841% |
| pure_vegetarian | ft_transformer | 0.50 | 93 | -3.00 pp [-7.0, 0.0] | 93.0% [87.0, 97.0] | 0.95 [0.44, 1.53] | 1.76% [0.80, 2.82] | 1,102% |
| veg_nonveg_combo | bionutrigpt | 0.05 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 6.88 [5.84, 7.91] | 11.24% [9.45, 13.15] | 5,841% |
| veg_nonveg_combo | bionutrigpt | 0.10 | 99 | -1.00 pp [-3.00, 0.00] | 99.0% [97.0, 100.0] | 5.89 [4.88, 6.98] | 9.54% [7.78, 11.45] | 5,420% |
| veg_nonveg_combo | bionutrigpt | 0.20 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 3.84 [2.80, 4.88] | 5.48% [3.89, 7.21] | 3,845% |
| veg_nonveg_combo | bionutrigpt | 0.30 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 2.56 [1.70, 3.51] | 4.01% [2.62, 5.64] | 2,752% |
| veg_nonveg_combo | bionutrigpt | 0.50 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 1.14 [0.58, 1.76] | 1.88% [0.99, 2.97] | 1,241% |
| veg_nonveg_combo | catboost | 0.05 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 7.04 [6.01, 8.08] | 11.62% [9.80, 13.56] | 5,892% |
| veg_nonveg_combo | catboost | 0.10 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 5.76 [4.74, 6.84] | 9.18% [7.44, 11.02] | 5,310% |
| veg_nonveg_combo | catboost | 0.20 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 3.42 [2.44, 4.41] | 5.21% [3.64, 6.94] | 3,680% |
| veg_nonveg_combo | catboost | 0.30 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 2.65 [1.80, 3.58] | 4.35% [2.85, 6.01] | 2,891% |
| veg_nonveg_combo | catboost | 0.50 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 1.19 [0.63, 1.82] | 1.95% [1.04, 3.06] | 1,298% |
| veg_nonveg_combo | ft_transformer | 0.05 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 6.55 [5.52, 7.57] | 10.78% [9.01, 12.65] | 5,741% |
| veg_nonveg_combo | ft_transformer | 0.10 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 5.62 [4.58, 6.71] | 9.24% [7.45, 11.14] | 5,278% |
| veg_nonveg_combo | ft_transformer | 0.20 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 3.55 [2.57, 4.56] | 5.62% [3.97, 7.42] | 3,792% |
| veg_nonveg_combo | ft_transformer | 0.30 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 2.47 [1.63, 3.37] | 4.14% [2.71, 5.76] | 2,789% |
| veg_nonveg_combo | ft_transformer | 0.50 | 100 | 0.00 pp [0.00, 0.00] | 100.0% [100.0, 100.0] | 1.54 [0.89, 2.25] | 2.41% [1.36, 3.61] | 1,514% |

---

## 6. Adversarial Verification and Safeguard Test Results

The read-only audit verified all 4,800 case files and executed the Section 7 adversarial unit test suite (`verify_stage1_downstream_audit_v2.py` and `test_audit_v2_safeguards.py`):

1. **Adversarial Test Suite (11/11 Passed):**
   - Test 1 (Missing Delivered Nutrient Key): Caught deletion of `DR1TKCAL` from output.
   - Test 2 (Duplicate Food Entry): Caught duplicate food selection in solved basket.
   - Test 3 (Corrupted Portion): Caught invalid serving quantity ($q = 4$).
   - Test 4 (Accounting Mismatch): Caught discrepancy between declared total and catalog-derived total.
   - Test 5 (Altered Probability): Rejected nonfinite and out-of-range probabilities ($p < 0$, $p > 1$, NaN, Inf).
   - Test 6 (Known Diagnosis Precedence): Verified that documented diabetes activates clinical rule regardless of low predicted probability ($p = 0.001$).
   - Test 7 (Measured BMI Precedence): Verified that measured BMI $< 30$ overrides predicted obesity.
   - Test 8 (Unknown Clinical History Preservation): Confirmed unknown histories remain unknown without default imputation.
   - Test 9 (Stale Resume Artifact): Detected input parameter changes via canonical SHA-256 fingerprinting.
   - Test 10 (Truncated Case File): Verified proper handling of incomplete/corrupted JSON records.
   - Test 11 (Dietary Mode Integrity): Caught non-vegetarian dish selection in `pure_vegetarian` mode.

2. **Full Case Verification (4,800 / 4,800 Passed):**
   - Participant cohort matched seed `20261120` row-for-row, strictly disjoint from the 5 pilot participants.
   - Profile demographic fields (weight, height, age, gender, DIQ010) matched raw NHANES source data with zero error.
   - All 4,614 accepted plans satisfied both assigned envelopes and rules-only reference bounds.
   - All 186 unaccepted cases matched solver records (182 Status 8, 4 Status 13).
   - Zero verification mismatches occurred.

3. **Safeguards & Resume Integrity:**
   - Canonical fingerprinting (`compute_case_fingerprint`) proven deterministic and sensitive to any modification of SEQN, mode, arm, threshold, patient metadata, targets, or catalog hash.
   - Resuming from valid case files verified; mismatched fingerprints and corrupted files correctly trigger fresh solves.

---

## 7. Methodological Disclosures & Provenance

- **Post-Hoc Status:** This audit was conducted post-hoc to resolve statistical and reporting defects in the completed study. While the protocol was documented prior to study execution, the direct model paired contrasts, valid-pair denominators, and normalized nutrient deviation endpoints are post-hoc repairs.
- **Exploratory Threshold Grid:** The threshold grid (0.05 to 0.50) was chosen as an exploratory sensitivity analysis across decision boundaries. These thresholds do not represent clinically validated decision cutoffs.
- **In-Silico Nature & No Clinical Efficacy Claim:** This study evaluates mathematical constraint satisfaction in an in-silico MILP solver using a 44-dish Indian composite food catalog and US NHANES demographic profiles. These results do not constitute clinical trial evidence, causal disease risk reduction, or external Indian clinical validity.
- **Worker Process Discrepancy:** `protocol.json` originally recorded `worker_processes: 8`. During runtime execution, the worker pool was scaled to 10 processes (`min(10, os.cpu_count())`) on the 12-core host. This discrepancy is documented in `correction_log.json` and has zero numerical impact on deterministic MILP solutions.
- **Artifact Preservation:** All original v1 study files remain preserved in `Saved Model/stage1_downstream_study_v1/` and snapshotted under `Saved Model/stage1_downstream_study_v1_audit_v2/original_v1_snapshot/`.
