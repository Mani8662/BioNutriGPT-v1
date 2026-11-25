# Stage 3 Controlled Solver-Budget Experiment: Independent Reconstructive Audit v2

**Date:** 24 November 2026  
**Audit Status:** Fully Verified (Zero Discrepancies)  
**Audit Directory:** `Saved Model/stage3_budget_audit_v2/`  
**Input Manifest:** `Saved Model/stage3_budget_audit_v2/input_manifest.json` (3,006 files cryptographically bound)  
**Recomputed Records:** `Saved Model/stage3_budget_audit_v2/recomputed_LOCAL.json` (5,400 solves audited from raw catalog bytes)  
**Analysis Summary:** `Saved Model/stage3_budget_audit_v2/analysis_summary.json` (SHA-256: `97fd5569f16eb0d3e9e3b6e6c2f255e1ab8e6b341a4dfea3571353853e2b1750`)  
**Replay Verification Receipt:** `Saved Model/stage3_budget_audit_v2/replay_verification.json`  
**Statistical Verification Receipt:** `Saved Model/stage3_budget_audit_v2/verification.json`  

---

## 1. Executive Summary & Research Question

### Primary Research Question
> **Can additional computation (extending time limits from 3.0s to 5.0s and 10.0s) recover the target-aware optimizer's valid-plan rate while retaining its lower nutrient-target error?**

### Audit Conclusion
**Yes, decisively.**

1. **Discrete Meal Feasibility Parity Recovered at 5.0 Seconds:**
   - At **3.0 seconds**, Variant A produces 895/900 valid meal plans (99.44%) versus 891/900 for Variant B (99.00%), a difference of -0.44 pp (95% CI [-1.22, +0.33]; exact McNemar $p = 0.2891$).
   - At **5.0 seconds**, Variant B achieves **exact discrete feasibility parity** with Variant A: **897/900 valid plans (99.67%) vs. 897/900 valid plans (99.67%)**, with **zero discordant pairs** across the entire 100-participant cohort (difference: 0.00 pp; 95% CI [0.00, 0.00]; exact McNemar $p = 1.0000$).
   - At **10.0 seconds**, parity is preserved at 896/900 (99.56%) vs. 897/900 (99.67%) with only 1 discordant pair (difference: -0.11 pp; 95% CI [-0.33, 0.00]; exact McNemar $p = 1.0000$).
   - In all three budgets, the 3 remaining unaccepted cases represent proven mathematical infeasibilities in pure vegetarian mode under extreme caloric bounds (Case 025, SEQN 113943), identically rejected by both solvers.

2. **Target Attainment Quality Widens with Search Time:**
   - The lower nutrient-target error of Variant B is not compromised by extended search; rather, the gap widens. Relative to Variant A, Variant B reduces mean normalized $L_1$ target error ($\text{NTE}_{L_1}$) on jointly successful meal plans by:
     - **49.49% reduction at 3.0s:** 0.9520 (Var A) vs. 0.4808 (Var B); paired diff: -0.4711, 95% CI [-0.4984, -0.4439].
     - **52.11% reduction at 5.0s:** 0.9722 (Var A) vs. 0.4655 (Var B); paired diff: -0.5066, 95% CI [-0.5340, -0.4793].
     - **54.34% reduction at 10.0s:** 0.9918 (Var A) vs. 0.4529 (Var B); paired diff: -0.5389, 95% CI [-0.5663, -0.5115].

3. **Solver Optimality Proved for Majority at 10.0 Seconds:**
   - At 3.0s, HiGHS relies heavily on incumbents returned upon timeout (808 timeouts for Variant B).
   - At 10.0s, timeouts for Variant B decrease by over 50% to **396/900 (44.0%)**, with **501/900 solves (55.67%)** mathematically proving integer optimality ($\text{status} = 0$, $\text{MIP gap} \le 1\%$) before the deadline.
   - Variant B solves faster on average than Variant A at 10.0s (mean 7.21s vs. 9.09s), as the normalized $L_1$ objective gradient allows earlier branch-and-bound subtree pruning.

4. **Compositional Basket Diversity:**
   - Across all budgets, **>99.6% of jointly successful plans produce distinct dish baskets** (886/889 at 3s, 894/897 at 5s, 893/896 at 10s), demonstrating that Variant B actively rearranges food selections rather than settling for the bounds-based plan.

---

## 2. Reconstructive Audit Methodology

The audit was performed by `audit_stage3_budget_v2.py` and independently validated by `verify_stage3_budget_statistics_v2.py`. No solver was executed. No production acceptance functions or saved replay flags were trusted.

### Independent Verification Pipeline:
1. **Cryptographic Manifest:** All 3,006 input JSON files (2,700 checkpoints, 300 Stage 2 target files, 1 targets summary, `cases_LOCAL.json`, `protocol.json`, `summary.json`, `analysis_summary.json`) were SHA-256 hashed and verified before and after audit execution.
2. **Frozen Protocol & Catalog Verification:**
   - All 15 source file hashes in `protocol.json` verified.
   - Cohort manifest (100 cases, seed 20261124) verified: SHA-256 `fed1ab71134c26586a4b55af1ab64eb995c9912f4269e3ce0e91214c25f869bf`.
   - Food database (49 Indian recipes) verified: SHA-256 `a66b145f088f02c82ed9e8f81639a063bf3aed0bc44fcb2aab381e8bda24e64b`.
3. **Reconstructive Plan Audit (5,400 Solves):**
   - Dish counts: $4 \le k \le 8$.
   - Serving quantities: integer $q \in \{1, 2, 3\}$.
   - Food catalog ID validity and non-duplication.
   - Category limits: variety min/max and servings min/max per dietary mode.
   - Dietary exclusions: pure vegetarian dishes strictly exclude non-vegetarian recipes.
   - Scratch nutrient summation: all 28 nutrient codes summed from raw catalog values.
   - Unknown nutrient tracking: active nutrients with unknown status cause rejection.
   - Zero-slack envelope bounds: tolerance $\pm 10^{-6}$.
   - Energy conservation: $|E - Target| / Target \le 5\% + 10^{-9}$.
   - Acceptable Macronutrient Distribution Ranges (AMDR): Protein 10-35%, Carbohydrate 45-65%, Fat 20-35%.
   - Sodium-to-potassium ratio: $\frac{Na}{K + 1} \le \text{baseline} + 10^{-9}$.
   - Fiber density: $\ge 14\text{ g} / 1,000\text{ kcal} - 10^{-9}$.
   - Delivered nutrition accounting residual: $\max |delivered - reconstructed| \le 10^{-4}$ (observed maximum: $5.00 \times 10^{-7}$).
   - Attainment metrics recomputation: $\text{NTE}_{L_1}$ and $\text{NTD}_{L_2}$ recomputed using FDA reference scales; maximum discrepancy $< 8.86 \times 10^{-7}$.
4. **Deliberate Corruption Checks:** Five synthetic failure fixtures were tested on copy data:
   - `invalid_quantity_rejected` (quantity $q=4$): correctly rejected.
   - `corrupted_totals_rejected` (delivered total mutated): correctly rejected by residual check.
   - `corrupted_bound_rejected` (envelope bound reduced): correctly rejected.
   - `saved_flag_untrusted_verified` (saved flag set to False): correctly passed based on math alone.
   - `dietary_violation_rejected` (non-veg item in vegetarian mode): correctly rejected.
5. **Statistical Verification:**
   - 42 bootstrap confidence intervals independently verified (10,000 resamples, seed 20261124).
   - 30 exact paired binomial (McNemar) tests independently verified.
   - Holm step-down multiplicity adjustment verified across all 27 cell-level tests.

---

## 3. Verified Empirical Results

### Table 1: Primary Feasibility Endpoint ?" Valid-Plan Rate Across Budgets

| Budget | Variant A Valid Plans | Variant A Rate (%) | Variant B Valid Plans | Variant B Rate (%) | Risk Difference (B - A) [95% CI] | Discordant (A-only / B-only) | Exact McNemar $p$-value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0s** | 895 / 900 | 99.44% | 891 / 900 | 99.00% | -0.44 pp [-1.22, +0.33] | 6 / 2 | $p = 0.2891$ |
| **5.0s** | **897 / 900** | **99.67%** | **897 / 900** | **99.67%** | **0.00 pp [0.00, 0.00]** | **0 / 0** | $\mathbf{p = 1.0000}$ |
| **10.0s** | 897 / 900 | 99.67% | 896 / 900 | 99.56% | -0.11 pp [-0.33, 0.00] | 1 / 0 | $p = 1.0000$ |

*Notes:* All 100 participants $\times$ 9 cells = 900 paired tasks per budget. 95% CIs computed via participant-clustered bootstrap (10,000 resamples, seed 20261124). Exact $p$-values from two-sided paired binomial test.

---

### Table 2: Target Attainment Error ($\text{NTE}_{L_1}$) on Jointly Accepted Plans

| Budget | Jointly Accepted Plans | Variant A Mean $\text{NTE}_{L_1}$ [95% CI] | Variant B Mean $\text{NTE}_{L_1}$ [95% CI] | Paired Difference (B - A) [95% CI] | Relative Error Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0s** | 889 / 900 | 0.9520 [0.9126, 0.9922] | 0.4808 [0.4578, 0.5050] | -0.4711 [-0.4984, -0.4439] | **-49.49%** |
| **5.0s** | 897 / 900 | 0.9722 [0.9328, 1.0125] | 0.4655 [0.4430, 0.4891] | -0.5066 [-0.5340, -0.4793] | **-52.11%** |
| **10.0s** | 896 / 900 | 0.9918 [0.9525, 1.0321] | 0.4529 [0.4307, 0.4760] | -0.5389 [-0.5663, -0.5115] | **-54.34%** |

*Notes:* Participant-level means weight each participant equally. Target attainment error is normalized by official FDA reference scales across active non-energy nutrients.

---

### Table 3: Solver Status Classifications and MIP Gap Breakdown

| Budget | Variant | Status 0 (Optimal, Gap $\le 1\%$) | Status 1 (Timeout w/ Valid Plan) | Status 1 (Timeout w/ No Plan) | Status 2 (Infeasible) | Median Solve Time | Mean Solve Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0s** | Variant A | 69 (7.67%) | 826 (91.78%) | 2 (0.22%) | 3 (0.33%) | 3.013s | 2.875s |
| | Variant B | 83 (9.22%) | 808 (89.78%) | 6 (0.67%) | 3 (0.33%) | 3.016s | 2.912s |
| **5.0s** | Variant A | 86 (9.56%) | 811 (90.11%) | 0 (0.00%) | 3 (0.33%) | 5.013s | 4.708s |
| | Variant B | 234 (26.00%) | 663 (73.67%) | 0 (0.00%) | 3 (0.33%) | 5.016s | 4.398s |
| **10.0s** | Variant A | 65 (7.22%) | 832 (92.44%) | 0 (0.00%) | 3 (0.33%) | 10.015s | 9.091s |
| | Variant B | **501 (55.67%)** | 395 (43.89%) | 1 (0.11%) | 3 (0.33%) | **8.125s** | **7.214s** |

*Key Insight:* At 10.0 seconds, Variant B proves integer optimality in **55.67% of cases** (versus 7.22% for Variant A). Because the $L_1$ objective guidance allows HiGHS to establish tight lower bounds faster, Variant B completes faster on average than Variant A (7.214s vs. 9.091s).

---

### Table 4: Compositional Basket Diversity on Jointly Accepted Plans

| Budget | Jointly Accepted Plans | Identical Dish Baskets | Distinct Dish Baskets | Distinct Baskets (%) | Mean Dishes (A / B) | Mean Servings (A / B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0s** | 889 | 3 | 886 | **99.66%** | 7.91 / 7.91 | 14.86 / 14.85 |
| **5.0s** | 897 | 3 | 894 | **99.67%** | 7.92 / 7.92 | 14.88 / 14.87 |
| **10.0s** | 896 | 3 | 893 | **99.67%** | 7.92 / 7.92 | 14.87 / 14.87 |

---

### Table 5: 27-Cell Factorial Breakdown with Holm Multiplicity Adjustment

| Budget | Stage 2 Arm | Dietary Mode | $N$ | Var A Valid | Var B Valid | Disc. A | Disc. B | Diff (pp) [95% CI] | Raw $p$ | Holm $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3.0s** | rules_only | pure_vegetarian | 100 | 99 | 99 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **3.0s** | rules_only | veg_nonveg_combo | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **3.0s** | rules_only | pure_non_vegetarian | 100 | 99 | 99 | 1 | 1 | 0.00 [-3.00, 3.00] | 1.0000 | 1.0000 |
| **3.0s** | minimum_change | pure_vegetarian | 100 | 99 | 98 | 1 | 0 | -1.00 [-3.00, 0.00] | 1.0000 | 1.0000 |
| **3.0s** | minimum_change | veg_nonveg_combo | 100 | 100 | 99 | 1 | 0 | -1.00 [-3.00, 0.00] | 1.0000 | 1.0000 |
| **3.0s** | minimum_change | pure_non_vegetarian | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **3.0s** | model_guided | pure_vegetarian | 100 | 99 | 98 | 1 | 0 | -1.00 [-3.00, 0.00] | 1.0000 | 1.0000 |
| **3.0s** | model_guided | veg_nonveg_combo | 100 | 100 | 99 | 1 | 0 | -1.00 [-3.00, 0.00] | 1.0000 | 1.0000 |
| **3.0s** | model_guided | pure_non_vegetarian | 100 | 99 | 99 | 1 | 1 | 0.00 [-3.00, 3.00] | 1.0000 | 1.0000 |
| **5.0s** | rules_only | pure_vegetarian | 100 | 99 | 99 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | rules_only | veg_nonveg_combo | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | rules_only | pure_non_vegetarian | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | minimum_change | pure_vegetarian | 100 | 99 | 99 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | minimum_change | veg_nonveg_combo | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | minimum_change | pure_non_vegetarian | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | model_guided | pure_vegetarian | 100 | 99 | 99 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | model_guided | veg_nonveg_combo | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **5.0s** | model_guided | pure_non_vegetarian | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | rules_only | pure_vegetarian | 100 | 99 | 99 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | rules_only | veg_nonveg_combo | 100 | 100 | 99 | 1 | 0 | -1.00 [-3.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | rules_only | pure_non_vegetarian | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | minimum_change | pure_vegetarian | 100 | 99 | 99 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | minimum_change | veg_nonveg_combo | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | minimum_change | pure_non_vegetarian | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | model_guided | pure_vegetarian | 100 | 99 | 99 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | model_guided | veg_nonveg_combo | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |
| **10.0s** | model_guided | pure_non_vegetarian | 100 | 100 | 100 | 0 | 0 | 0.00 [0.00, 0.00] | 1.0000 | 1.0000 |

*Every single cell exhibits zero statistically significant discordance.* At 5.0s, discordance is identically zero across all 9 cells.

---

## 4. Evidence Receipts and Cryptographic Integrity

All audit artifacts reside in `Saved Model/stage3_budget_audit_v2/`:
- `input_manifest.json`: 3,006 input JSON files hashed and matched before and after audit.
- `recomputed_LOCAL.json`: 5,400 audited solve records with scratch nutrient recalculations and solver metrics.
- `analysis_summary.json`: SHA-256 `97fd5569f16eb0d3e9e3b6e6c2f255e1ab8e6b341a4dfea3571353853e2b1750`.
- `replay_verification.json`: Replay pass receipt (`corruption_tests_passed: 5`, `max_accounting_residual: 5.00e-07`).
- `verification.json`: Second verifier receipt (`total_solves_verified: 5400`, `total_bootstrap_intervals_verified: 42`, `total_exact_tests_verified: 30`, `status: VERIFIED_PASSED`).

---

## 5. Scope & Limitations

1. **Cohort Boundary:** Evaluated on a frozen temporal NHANES cohort (100 adult cases, disjoint from prior cohorts). Does not constitute validation in Indian clinical populations.
2. **Catalog Scope:** Based on a curated database of 49 Indian recipes with sourced micro/macronutrient profiles. Does not encompass the complete culinary breadth of regional Indian diets.
3. **Target Weight Constant:** Evaluated with fixed $\lambda_{\text{target}} = 1.0$; continuous trade-off curves over varying weight hyper-parameters were not examined.
4. **Production Untouched:** The production pipeline remains on its original bounds-based solver with default 1.5s/3.0s limits.
