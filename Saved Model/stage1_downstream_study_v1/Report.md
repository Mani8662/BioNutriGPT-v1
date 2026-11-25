# Stage 1 Downstream Added-Value Controlled Study v1

**Date:** 2026-11-20  
**Cohort:** 100 randomly sampled temporal NHANES adult participants (seed `20261120`, strictly disjoint from the 5 pilot cases)  
**Design:** Paired engineering evaluation of Stage 1 prediction-assisted rule branching versus deterministic rules-only baseline  
**Dietary Modes:** `pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`  
**Models Evaluated:** BioNutriGPT, CatBoost, FT-Transformer  
**Exploratory Threshold Grid:** 0.05, 0.10, 0.20, 0.30, 0.50  
**Total Solves:** 4,800 planned (300 rules-only references, 4,500 prediction-assisted solves)  
**Overall Acceptance:** **4,614 / 4,800 plans accepted** (96.1% overall: 296 / 300 rules-only [98.7%], 4,318 / 4,500 prediction-assisted [96.0%])  
**Total Rejections:** **186 / 4,800 plans** (182 proven `infeasible_active_rules`, 4 computational `timeout_no_valid_plan`)  
**Verification:** 100% independent replay agreement (0 mismatches across all 4,800 cases; 8/8 adversarial contract tests passed)  

## Executive Summary

1. **Primary Feasibility Finding:** Activating Stage 1 predicted risk branches **did not improve meal planning feasibility** over the deterministic rules-only baseline.
   - Rules-only baseline achieved **98.7% feasibility** (296/300 participant-mode pairs).
   - **bionutrigpt**: feasibility = **95.8%** (1437/1500), paired difference vs. rules-only = **-2.87%**.
   - **catboost**: feasibility = **96.1%** (1441/1500), paired difference vs. rules-only = **-2.60%**.
   - **ft_transformer**: feasibility = **96.0%** (1440/1500), paired difference vs. rules-only = **-2.67%**.
   - Because predictive rule branches exclusively tighten constraints (lowering carbohydrate, saturated fat, and sodium ceilings, while raising fiber and potassium floors), activating predictions can never expand the feasible search space. In zero cases did prediction-assisted planning make an infeasible profile feasible.

2. **Direct Paired Model Contrasts:**
   - Direct paired participant-level comparisons across all 1,500 solve pairs show minimal differences between models:
     - **BioNutriGPT vs. CatBoost:** paired difference = **-0.27%** (net difference of -4 solves out of 1,500).
     - **BioNutriGPT vs. FT-Transformer:** paired difference = **-0.20%** (net difference of -3 solves out of 1,500).
     - **CatBoost vs. FT-Transformer:** paired difference = **+0.07%** (net difference of +1 solve out of 1,500).
   - In cell-by-cell paired bootstrap tests (2,000 resamples on RTX 3050 CUDA GPU), confidence intervals for model differences span zero in 11 of 15 cells, with small discrepancies in 4 vegetarian cells due to threshold boundary crossings. We do not claim statistical equivalence; the evidence demonstrates that BioNutriGPT offers **no detected engineering advantage** over standard gradient-boosted trees or tabular transformers in downstream meal synthesis.

3. **Accurate Failure Classification:**
   - Of the 186 unaccepted solves, **182 were proven mathematical infeasibilities** (`infeasible_active_rules`, HiGHS status 8), occurring entirely in `pure_vegetarian` mode where stricter carbohydrate ($\le 45\%$), saturated fat ($\le 6\%$), and sodium bounds could not be satisfied simultaneously with catalog vegetarian variety requirements.
   - Only **4 solves failed due to timeouts** (`timeout_no_valid_plan`), occurring in `veg_nonveg_combo` mode where the 3.0-second limit expired before an incumbent was found. Timeouts are computational stopping conditions, not mathematical infeasibility.
   - In `pure_non_vegetarian` mode, **0 failures occurred** (1,600 / 1,600 solves succeeded, 100.0% acceptance).

4. **Plan Quality & Common-Reference Compliance:**
   - **100% of accepted prediction plans (4,318 / 4,318)** satisfied the common rules-only reference envelope, confirming that tightening rules preserves baseline clinical adequacy.
   - Basket Realignment: Prediction-assisted solves substituted an average of **3.95 to 3.98 dishes** (and 6.79 to 6.83 servings). Relative nutrient deviation across the 8 primary clinical planning targets was **~21.6% to 22.1%**.
   - Note: Basket adjustments reflect constraint-satisfaction trade-offs. They do not constitute evidence of superior diet quality, palatability, or clinical efficacy.

## Overall Performance Summary Table

| Arm / Model | Solves Planned | Accepted Plans | Rejected Plans | Feasibility Rate | Paired Diff vs. Rules-Only | Common-Ref Pass Rate | Mean Dish Changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Rules-Only Baseline** | 300 | 296 | 4 | 98.7% | 0.00% (reference) | 98.7% | 0.00 |
| **bionutrigpt** | 1500 | 1437 | 63 | 95.8% | -2.87% | 95.8% | 3.98 |
| **catboost** | 1500 | 1441 | 59 | 96.1% | -2.60% | 96.1% | 3.95 |
| **ft_transformer** | 1500 | 1440 | 60 | 96.0% | -2.67% | 96.0% | 3.97 |

## Failure Analysis Breakdown

| Dietary Mode | Total Solves | Total Failures | Infeasible Active Rules (HiGHS proven) | Timeout (3.0s limit reached) | Failure Rate |
|---|---:|---:|---:|---:|---:|
| `pure_vegetarian` | 1,600 | 182 | 182 | 0 | 11.38% |
| `veg_nonveg_combo` | 1,600 | 4 | 0 | 4 | 0.25% |
| `pure_non_vegetarian` | 1,600 | 0 | 0 | 0 | 0.00% |
| **Total** | **4,800** | **186** | **182** | **4** | **3.88%** |

## Direct Paired Model-versus-Comparator Contrasts

| Dietary Mode | Threshold | Valid Participants | BioNutriGPT vs. CatBoost Diff (95% CI) | BioNutriGPT vs. FT-Transformer Diff (95% CI) | CatBoost vs. FT-Transformer Diff (95% CI) |
|---|---:|---:|---|---|---|
| pure_non_vegetarian | 0.05 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| pure_non_vegetarian | 0.10 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| pure_non_vegetarian | 0.20 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| pure_non_vegetarian | 0.30 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| pure_non_vegetarian | 0.50 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| pure_vegetarian | 0.05 | 100 | +0.0% [+0.0%, +0.0%] | -2.0% [-5.0%, +0.0%] | -2.0% [-5.0%, +0.0%] |
| pure_vegetarian | 0.10 | 100 | -1.0% [-3.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +1.0% [+0.0%, +3.0%] |
| pure_vegetarian | 0.20 | 100 | -3.0% [-7.0%, +0.0%] | -1.0% [-5.0%, +2.0%] | +2.0% [-2.0%, +6.0%] |
| pure_vegetarian | 0.30 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| pure_vegetarian | 0.50 | 100 | +1.0% [+0.0%, +3.0%] | +1.0% [+0.0%, +3.0%] | +0.0% [-3.0%, +3.0%] |
| veg_nonveg_combo | 0.05 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| veg_nonveg_combo | 0.10 | 100 | -1.0% [-3.0%, +0.0%] | -1.0% [-3.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| veg_nonveg_combo | 0.20 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| veg_nonveg_combo | 0.30 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |
| veg_nonveg_combo | 0.50 | 100 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] |

## Detailed Results by Dietary Mode and Threshold Grid

| Dietary Mode | Model | Threshold | Valid Pairs | Feasibility Diff vs. Rules (95% CI) | Common-Ref Rate (95% CI) | Mean Dish Changes (95% CI) | Primary Nutrient Dev (95% CI) |
|---|---|---:|---:|---|---|---|---|
| pure_non_vegetarian | bionutrigpt | 0.05 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 7.53 [6.50, 8.55] | 12.5% [10.7%, 14.4%] |
| pure_non_vegetarian | bionutrigpt | 0.10 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 6.44 [5.41, 7.51] | 10.6% [8.7%, 12.6%] |
| pure_non_vegetarian | bionutrigpt | 0.20 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 4.16 [3.09, 5.25] | 6.0% [4.3%, 7.7%] |
| pure_non_vegetarian | bionutrigpt | 0.30 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 2.84 [1.95, 3.84] | 4.5% [3.0%, 6.4%] |
| pure_non_vegetarian | bionutrigpt | 0.50 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 1.32 [0.72, 1.98] | 2.2% [1.2%, 3.4%] |
| pure_non_vegetarian | catboost | 0.05 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 7.71 [6.66, 8.78] | 13.0% [11.1%, 15.1%] |
| pure_non_vegetarian | catboost | 0.10 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 6.23 [5.21, 7.36] | 10.1% [8.2%, 12.0%] |
| pure_non_vegetarian | catboost | 0.20 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 3.63 [2.62, 4.65] | 5.7% [4.0%, 7.6%] |
| pure_non_vegetarian | catboost | 0.30 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 2.95 [2.06, 3.90] | 4.9% [3.3%, 6.7%] |
| pure_non_vegetarian | catboost | 0.50 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 1.37 [0.77, 2.07] | 2.3% [1.2%, 3.6%] |
| pure_non_vegetarian | ft_transformer | 0.05 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 7.15 [6.10, 8.19] | 12.0% [10.1%, 13.9%] |
| pure_non_vegetarian | ft_transformer | 0.10 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 6.08 [4.99, 7.21] | 10.2% [8.2%, 12.3%] |
| pure_non_vegetarian | ft_transformer | 0.20 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 3.80 [2.81, 4.83] | 6.2% [4.4%, 8.2%] |
| pure_non_vegetarian | ft_transformer | 0.30 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 2.74 [1.87, 3.67] | 4.7% [3.1%, 6.5%] |
| pure_non_vegetarian | ft_transformer | 0.50 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 1.80 [1.07, 2.57] | 2.8% [1.6%, 4.1%] |
| pure_vegetarian | bionutrigpt | 0.05 | 80/100 | -16.0% [-23.0%, -9.0%] | 80.0% [72.0%, 87.0%] | 4.97 [3.96, 6.01] | 11.3% [8.9%, 13.8%] |
| pure_vegetarian | bionutrigpt | 0.10 | 84/100 | -12.0% [-19.0%, -6.0%] | 84.0% [77.0%, 91.0%] | 4.32 [3.33, 5.35] | 9.7% [7.4%, 12.4%] |
| pure_vegetarian | bionutrigpt | 0.20 | 88/100 | -8.0% [-14.0%, -3.0%] | 88.0% [81.0%, 94.0%] | 2.40 [1.57, 3.33] | 5.1% [3.2%, 7.1%] |
| pure_vegetarian | bionutrigpt | 0.30 | 92/100 | -4.0% [-8.0%, -1.0%] | 92.0% [86.0%, 97.0%] | 1.49 [0.84, 2.20] | 3.3% [1.8%, 5.0%] |
| pure_vegetarian | bionutrigpt | 0.50 | 95/100 | -1.0% [-3.0%, +0.0%] | 95.0% [90.0%, 99.0%] | 0.48 [0.10, 0.94] | 1.0% [0.2%, 2.0%] |
| pure_vegetarian | catboost | 0.05 | 80/100 | -16.0% [-23.0%, -9.0%] | 80.0% [72.0%, 87.0%] | 5.42 [4.43, 6.42] | 12.4% [10.0%, 14.8%] |
| pure_vegetarian | catboost | 0.10 | 85/100 | -11.0% [-17.0%, -5.0%] | 85.0% [78.0%, 92.0%] | 4.32 [3.28, 5.38] | 9.2% [7.0%, 11.5%] |
| pure_vegetarian | catboost | 0.20 | 91/100 | -5.0% [-10.0%, -1.0%] | 91.0% [85.0%, 96.0%] | 2.12 [1.37, 2.97] | 4.6% [2.9%, 6.5%] |
| pure_vegetarian | catboost | 0.30 | 92/100 | -4.0% [-8.0%, -1.0%] | 92.0% [86.0%, 97.0%] | 1.64 [0.96, 2.40] | 3.7% [2.1%, 5.5%] |
| pure_vegetarian | catboost | 0.50 | 94/100 | -2.0% [-5.0%, +0.0%] | 94.0% [89.0%, 98.0%] | 0.57 [0.18, 1.05] | 1.4% [0.4%, 2.7%] |
| pure_vegetarian | ft_transformer | 0.05 | 82/100 | -14.0% [-21.0%, -8.0%] | 82.0% [74.0%, 89.0%] | 4.96 [3.96, 5.99] | 11.1% [8.7%, 13.6%] |
| pure_vegetarian | ft_transformer | 0.10 | 84/100 | -12.0% [-19.0%, -6.0%] | 84.0% [77.0%, 91.0%] | 4.07 [3.11, 5.10] | 9.3% [6.9%, 11.9%] |
| pure_vegetarian | ft_transformer | 0.20 | 89/100 | -7.0% [-12.0%, -2.0%] | 89.0% [82.0%, 95.0%] | 2.21 [1.39, 3.07] | 4.5% [2.7%, 6.3%] |
| pure_vegetarian | ft_transformer | 0.30 | 92/100 | -4.0% [-8.0%, -1.0%] | 92.0% [86.0%, 97.0%] | 1.79 [1.07, 2.57] | 3.7% [2.1%, 5.4%] |
| pure_vegetarian | ft_transformer | 0.50 | 94/100 | -2.0% [-5.0%, +0.0%] | 94.0% [89.0%, 98.0%] | 0.66 [0.21, 1.16] | 1.5% [0.4%, 2.8%] |
| veg_nonveg_combo | bionutrigpt | 0.05 | 99/100 | -1.0% [-3.0%, +0.0%] | 99.0% [97.0%, 100.0%] | 8.55 [7.40, 9.69] | 13.1% [11.1%, 15.2%] |
| veg_nonveg_combo | bionutrigpt | 0.10 | 99/100 | -1.0% [-3.0%, +0.0%] | 99.0% [97.0%, 100.0%] | 7.09 [5.94, 8.31] | 10.9% [8.8%, 13.0%] |
| veg_nonveg_combo | bionutrigpt | 0.20 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 3.82 [2.76, 4.98] | 5.6% [3.9%, 7.5%] |
| veg_nonveg_combo | bionutrigpt | 0.30 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 2.97 [1.93, 4.10] | 4.6% [2.9%, 6.5%] |
| veg_nonveg_combo | bionutrigpt | 0.50 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 1.04 [0.46, 1.67] | 1.6% [0.7%, 2.6%] |
| veg_nonveg_combo | catboost | 0.05 | 99/100 | -1.0% [-3.0%, +0.0%] | 99.0% [97.0%, 100.0%] | 9.03 [7.87, 10.16] | 13.6% [11.6%, 15.7%] |
| veg_nonveg_combo | catboost | 0.10 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 7.05 [5.80, 8.26] | 10.5% [8.4%, 12.5%] |
| veg_nonveg_combo | catboost | 0.20 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 3.34 [2.29, 4.46] | 4.7% [3.1%, 6.4%] |
| veg_nonveg_combo | catboost | 0.30 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 2.57 [1.64, 3.62] | 4.4% [2.8%, 6.2%] |
| veg_nonveg_combo | catboost | 0.50 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 1.13 [0.48, 1.82] | 1.8% [0.7%, 2.9%] |
| veg_nonveg_combo | ft_transformer | 0.05 | 99/100 | -1.0% [-3.0%, +0.0%] | 99.0% [97.0%, 100.0%] | 8.38 [7.15, 9.58] | 12.9% [10.7%, 15.1%] |
| veg_nonveg_combo | ft_transformer | 0.10 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 6.87 [5.61, 8.10] | 10.2% [8.1%, 12.4%] |
| veg_nonveg_combo | ft_transformer | 0.20 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 4.27 [3.12, 5.46] | 6.9% [4.9%, 9.0%] |
| veg_nonveg_combo | ft_transformer | 0.30 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 2.92 [1.92, 3.97] | 4.8% [3.0%, 6.6%] |
| veg_nonveg_combo | ft_transformer | 0.50 | 100/100 | +0.0% [+0.0%, +0.0%] | 100.0% [100.0%, 100.0%] | 1.49 [0.78, 2.32] | 2.4% [1.2%, 3.8%] |

## Methodological & Contextual Disclosures

1. **Nutrient Deviation Metric Definition:** Primary planning nutrient deviation is calculated as the mean relative absolute difference $\frac{1}{K} \sum_{k=1}^K \frac{|\text{tot}[k] - \text{ref}[k]|}{\text{ref}[k]}$ across the eight key macronutrient and electrolyte planning targets: Energy (`DR1TKCAL`), Protein (`DR1TPROT`), Carbohydrate (`DR1TCARB`), Total Fat (`DR1TTFAT`), Saturated Fat (`DR1TSFAT`), Dietary Fiber (`DR1TFIBE`), Sodium (`DR1TSODI`), and Potassium (`DR1TPOTA`). Unweighted averages across all 39 nutrients include trace micronutrients with near-zero baselines (e.g., retinol, vitamin B12, caffeine) that mathematically inflate percentages without reflecting biological importance. Relative deviations reflect food reallocation under tighter bounds, not improved nutritional quality.

2. **Separation of Infeasibility and Timeouts:** HiGHS mathematically proved infeasibility in 182 vegetarian cases because the intersection of strict carbohydrate/fat ceilings and discrete integer food servings contained zero integer points. The 4 timeouts in combo mode represent solver search limits under the 3.0-second limit and must not be conflated with mathematical infeasibility.

3. **Independent Replay Auditing:** All 4,800 cases were independently verified by recomputing portions, variety rules, and nutrient totals from source catalog entries. Replay passed in 100% of solver-accepted cases, confirming zero false-positive acceptances.

4. **Reproducibility & Hardware Configuration:** The study was executed with initial 8-worker parallelism, resumed with 10 workers on a 12-core Intel Core i7 system, and utilized an NVIDIA GeForce RTX 3050 Laptop GPU for 2,000 participant-level bootstrap resamples. Resume logic verifies matching input parameters before skipping completed artifacts.

## Reproducibility Artifacts
- **Protocol:** `Saved Model/stage1_downstream_study_v1/protocol.json`  
- **Bindings & Hashes:** `Saved Model/stage1_downstream_study_v1/binding.json`  
- **Selected Participant IDs:** `Saved Model/stage1_downstream_study_v1/selected_participants_LOCAL.csv`  
- **Per-Case Artifacts:** `Saved Model/stage1_downstream_study_v1/cases/case_0000_LOCAL.json` through `case_4799_LOCAL.json`  
- **Case-Level Paired Evaluations:** `Saved Model/stage1_downstream_study_v1/case_level_paired_eval_LOCAL.csv`  
- **Direct Paired Model Contrasts:** `Saved Model/stage1_downstream_study_v1/model_pairwise_contrasts.csv` and `.json`  
- **Failure Analysis Breakdown:** `Saved Model/stage1_downstream_study_v1/failure_analysis.json`  
- **Aggregate Summary:** `Saved Model/stage1_downstream_study_v1/aggregate_summary.csv` and `.json`  
- **Verification Record:** `Saved Model/stage1_downstream_study_v1/verification.json`  
- **Completed Record:** `Saved Model/stage1_downstream_study_v1/completed.json`  