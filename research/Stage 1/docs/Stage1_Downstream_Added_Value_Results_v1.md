# Stage 1 Downstream Added-Value Controlled Study v1 (Audit v3)

**Audit & Update Date:** 2026-11-21  
**Study Directory:** `Saved Model/stage1_downstream_study_v1_audit_v3/`  
**Prior Audit Snapshots:** `Saved Model/stage1_downstream_study_v1_audit_v3/audit_v2_snapshot/` (Audit v2) and `Saved Model/stage1_downstream_study_v1/` (Original v1)  
**Cohort:** 100 randomly sampled temporal NHANES adult participants (seed `20261120`, strictly disjoint from the 5 pilot cases)  
**Design:** Paired in-silico optimization study evaluating Stage 1 predictive rule branching versus deterministic rules-only baseline  
**Dietary Modes:** `pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`  
**Models Evaluated:** BioNutriGPT, CatBoost, FT-Transformer (repaired Stage 1 research models)  
**Exploratory Threshold Grid:** 0.05, 0.10, 0.20, 0.30, 0.50  
**Total Solves Planned:** 4,800 planned solves (100 participants × 3 modes × 16 arms: 300 rules-only references, 4,500 prediction-assisted solves)  
**Overall Acceptance:** **4,614 / 4,800 plans accepted** (96.12% overall: 296 / 300 rules-only [98.67%], 4,318 / 4,500 prediction-assisted [95.96%])  
**Total Unaccepted Solves:** **186 / 4,800 solves** (182 solver-certified mathematical infeasibilities, 4 computational timeouts)  
**Verification:** 100% independent replay agreement (0 mismatches across 4,800 cases; 11/11 adversarial contract tests passed; actual runner resume safeguards verified)  

---

## 1. Executive Summary & Core Engineering Findings

1. **Stage 1 Predictions Do Not Improve Meal Planning Feasibility:**
   - The deterministic rules-only baseline achieved **98.67% feasibility** (296 / 300 participant-mode pairs; 4 cases were infeasible in `pure_vegetarian` mode).
   - Prediction-assisted overall feasibility across 1,500 solves per model:
     - **bionutrigpt**: feasibility = **95.80%** (1437 / 1,500), paired difference vs. rules-only = **-2.87 pp** (95% CI: [-4.33, -1.60]).
     - **catboost**: feasibility = **96.07%** (1441 / 1,500), paired difference vs. rules-only = **-2.60 pp** (95% CI: [-4.00, -1.33]).
     - **ft_transformer**: feasibility = **96.00%** (1440 / 1,500), paired difference vs. rules-only = **-2.67 pp** (95% CI: [-4.20, -1.40]).
   - **Optimization Mechanism:** Predictive rule branches exclusively tighten nutritional constraints (lowering carbohydrate, saturated fat, and sodium ceilings, while raising fiber and potassium floors). In a constrained mathematical program, tightening constraint bounds can never expand the feasible search space. In zero cases did prediction-assisted planning render an infeasible baseline feasible.
   - Among the 45 cell-specific model-versus-rules-only feasibility contrasts, exactly **12 of 45 exclude zero** (all 4 lower thresholds in `pure_vegetarian` mode for all 3 models; intervals strictly negative, e.g. -16.00 pp [-23.00, -9.00]). In the remaining 33 cell configurations (including threshold 0.50 in `pure_vegetarian`, and all cells in `pure_non_vegetarian` and `veg_nonveg_combo`), the feasibility difference intervals contain zero.

2. **Direct Paired Model Contrasts Demonstrate No Detected Engineering Advantage:**
   - Direct participant-level paired contrasts across all 1,500 solve pairs (2,000 bootstrap resamples with seed `20261120`):
     - **bionutrigpt vs. catboost:** paired feasibility difference = **-0.27 pp** [95% CI: -0.60 pp, +0.00 pp] (-4 net solves out of 1,500).
     - **bionutrigpt vs. ft_transformer:** paired feasibility difference = **-0.20 pp** [95% CI: -0.53 pp, +0.13 pp] (-3 net solves out of 1,500).
     - **catboost vs. ft_transformer:** paired feasibility difference = **+0.07 pp** [95% CI: -0.33 pp, +0.47 pp] (+1 net solve out of 1,500).
   - In the cell-specific comparisons (3 model pairs × 3 modes × 5 thresholds = 45 cells), **exactly 0 of 45 direct feasibility intervals exclude zero** (none exclude zero; all 45 intervals contain zero).
   - *Methodological Distinction:* An interval containing zero does not prove equivalence between models; it indicates insufficient empirical evidence to establish a performance difference. There is no detected engineering basis to prefer the complex neural BioNutriGPT architecture over standard gradient-boosted trees or tabular transformers for downstream rule branching.

3. **Reconciled Failure Classification (Infeasibility vs. Timeout):**
   - Of the 186 unaccepted solves:
     - **182 (97.8%) are solver-certified mathematical infeasibilities** (HiGHS Status 8, solver status code 2). All 182 occurred in `pure_vegetarian` mode.
     - **Crucial Context on Rules-Only Infeasibility:** Four of these 182 infeasibilities occurred in the deterministic `rules_only` arm (cases 864, 2640, 3408, 3984 for SEQNs 111776, 118051, 121462, 122724). Therefore, vegetarian infeasibilities are not exclusively attributable to prediction tightening; baseline clinical rules themselves proved infeasible for certain vegetarian profiles.
     - **Diagnostic Boundary:** Solver infeasibility status establishes that the formulated problem was mathematically infeasible under the catalog and plate constraints. Because no irreducible inconsistent subsystem (IIS) diagnostics were computed, specific minimal conflicting constraints cannot be asserted as mathematical fact.
     - **Timing Reconciliation:** Infeasible solves had a median solver time of **24.03 ms** (mean 128.57 ms, range 13.68 ms to 965.96 ms; wall-clock median 33.02 ms, range 18.61 ms to 975.56 ms). Exactly **33 cases had solver times exceeding 40 ms** (18.1% of infeasibilities; maximum 965.96 ms; 44 cases had wall-clock time exceeding 40 ms). The prior claim that all infeasibilities occurred below 40 ms was inaccurate.
     - **4 (2.2%) are computational timeouts** (HiGHS Status 13, solver status code 1), occurring exclusively in `veg_nonveg_combo` mode. Wall-clock solve time reached 3013.27–3018.08 ms (~3.02 s) without finding an integer incumbent. A timeout is a computational stopping condition, not proof of mathematical infeasibility.
     - In `pure_non_vegetarian` mode, **zero failures occurred** (1,600 / 1,600 accepted, 100.0% feasibility).

4. **Plan Quality, Valid-Pair Conditioning & Normalized Nutrient Deviation:**
   - **Common-Reference Compliance:** Among accepted plans, **100.0% (4,318 / 4,318)** satisfied all active common-reference rules-only bounds (0 violations observed in sample). Note: Observing zero violations in this 4,318-plan sample does not guarantee 100% compliance for all future unseen patients. Among all attempted solves, the intention-to-evaluate pass rate was **95.96% (4,318 / 4,500)**, with the 4.04% deficit caused entirely by the 182 proved infeasibilities and 4 timeouts having no plan.
   - **Valid-Pair Subpopulation Analysis:** Conditioning on mutual acceptance ($N_{\text{both}}$ for inter-model pairs; $N_{\text{three}}$ for comparisons against rules-only) restricts the analysis to the mutually feasible subpopulation. This changes the population under description and does not eliminate selection bias. Within this mutually feasible subpopulation, calorie errors (0.7%–0.9%) and plate masses (~1,070g–1,150g) are comparable across models.
   - **Repaired Normalized Nutrient Deviation:** Dividing relative shifts by baseline delivered intake produced severe division singularities (up to 6,796%) when trace nutrients approached zero. The repaired metric scales absolute nutrient shifts against fixed standard reference scales (FDA Daily Values: 2,000 kcal, 50g protein, 275g carbohydrate, 28g fiber, 78g fat, 20g saturated fat, 300mg cholesterol, 2,300mg sodium, 4,700mg potassium). These constants serve as descriptive scaling factors to avoid near-zero denominators; they are not individualized RDAs or Indian clinical targets. Across all 45 cell configurations, normalized nutrient deviation ranges from **0.83% to 12.56%**.
   - **Dish Substitutions:** Prediction-assisted solves substituted an average of **3.95 to 3.98 dishes** (6.79 to 6.83 servings). These adjustments represent constraint-satisfaction trade-offs and do not constitute evidence of superior diet quality, palatability, or clinical efficacy.

5. **Worker-Count Provenance & Concurrency Disclosure:**
   - `protocol.json` originally recorded `worker_processes: 8`. In actual execution, the runner launched 10 worker processes (`min(10, os.cpu_count() or 1)`) on the 12-core host.
   - Execution logs confirm that all 4,800 tasks completed across the 10-process pool.
   - While deterministic MILP solves with fixed seeds yield identical solutions under identical time limits, whether CPU core contention under 10 workers versus 8 workers impacted marginal time-limited solves (specifically the 4 timeouts near 3.0 seconds) is unmeasured. Stating that this scaling had 'zero numerical impact' is unsupported without a controlled benchmark.

---

## 2. Overall Arm Performance and Plan Quality Summary Table

| Arm / Model | Solves Planned | Accepted Plans | Overall Feasibility | Paired Diff vs. Rules (pp) | 95% Bootstrap CI (pp) | Common-Ref Pass (Accepted) | Common-Ref Pass (Attempts) | Mean Dish Changes vs. Rules | Mean Serving Changes vs. Rules | Repaired Normalized Nutrient Dev | Historical Unstable Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rules-Only Baseline** | 300 | 296 | 98.67% | 0.00 pp (ref) | [0.00, 0.00] | 100.0% | 98.67% | 0.00 | 0.00 | 0.00% | 0.0% |
| **bionutrigpt** | 1,500 | 1,437 | 95.80% | -2.87 pp | [-4.33, -1.60] | 100.0% | 95.80% | 4.01 | 6.87 | 5.80% | 550% |
| **catboost** | 1,500 | 1,441 | 96.07% | -2.60 pp | [-4.00, -1.33] | 100.0% | 96.07% | 3.98 | 6.83 | 5.85% | 995% |
| **ft_transformer** | 1,500 | 1,440 | 96.00% | -2.67 pp | [-4.20, -1.40] | 100.0% | 96.00% | 4.00 | 6.88 | 5.91% | 620% |

---

## 3. Failure Analysis Breakdown

| Dietary Mode | Total Solves | Total Failures | Solver-Certified Infeasible (HiGHS Status 8) | Computational Timeouts (HiGHS Status 13) | Mode Failure Rate | Failure Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `pure_vegetarian` | 1,600 | 182 | 182 | 0 | 11.38% | All 182 failures are solver-certified infeasibilities (including 4 rules-only cases). Problem formulated was mathematically infeasible under plate rules; minimal conflicting constraints unmeasured. |
| `veg_nonveg_combo` | 1,600 | 4 | 0 | 4 | 0.25% | All 4 failures are computational timeouts (3.0s limit reached without integer incumbent). Computational stopping condition, not proof of infeasibility. |
| `pure_non_vegetarian` | 1,600 | 0 | 0 | 0 | 0.00% | Zero failures occurred; full catalog availability provided sufficient mathematical feasibility. |
| **Total** | **4,800** | **186** | **182** | **4** | **3.88%** | **97.8% solver-certified infeasible, 2.2% computational timeouts.** |

### Infeasible and Timeout Timing Distribution

- **Solver-Certified Infeasible Cases (N = 182):**
  - Solver Time (ms): Min = 13.68, 25% = 20.75, Median = 24.03, Mean = 128.57, 75% = 29.24, Max = 965.96.
  - Solver Time > 40 ms: **33 cases** (18.1%).
  - Wall-Clock Time (ms): Min = 18.61, 25% = 29.28, Median = 33.02, Mean = 137.27, 75% = 39.39, Max = 975.56.
  - Wall-Clock Time > 40 ms: **44 cases** (24.2%).
- **Computational Timeout Cases (N = 4):**
  - Wall-Clock Time (ms): Min = 3013.27, Median = 3017.65, Max = 3018.08 (~3.02 s limit reached).

---

## 4. Direct Paired Model Contrasts

All contrasts evaluate identical participants using 2,000 participant-level bootstrap resamples (seed `20261120`).

### Overall Equal-Weighted Contrasts (All 3 Modes × All 5 Thresholds, N = 1,500 solve pairs)

| Contrast Pair | Solves Evaluated | Model 1 Accepted | Model 2 Accepted | Paired Feasibility Diff (pp) | 95% Bootstrap CI (pp) | Excludes Zero? | Mean Dish Difference (Dishes) | 95% CI (Dishes) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **bionutrigpt vs. catboost** | 1,500 | 1,437 | 1,441 | **-0.27 pp** | [-0.60, 0.00] | No (touches zero) | 1.33 | [1.07, 1.60] |
| **bionutrigpt vs. ft_transformer** | 1,500 | 1,437 | 1,440 | **-0.20 pp** | [-0.53, 0.13] | No (spans zero) | 1.70 | [1.40, 2.01] |
| **catboost vs. ft_transformer** | 1,500 | 1,441 | 1,440 | **+0.07 pp** | [-0.33, 0.47] | No (spans zero) | 1.73 | [1.43, 2.04] |

### Cell-Specific Contrasts by Dietary Mode and Threshold (N = 100 participants per cell)

| Dietary Mode | Threshold | Contrast Pair | Model 1 Accepted | Model 2 Accepted | Valid Both ($N_{\text{both}}$) | Paired Feasibility Diff (pp) [95% CI] | Mean Dish Difference [95% CI] | Pointwise Excludes Zero? |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pure_vegetarian` | 0.05 | bionutrigpt_minus_catboost | 80 | 80 | 80 | +0.00 pp [0.00, 0.00] | 1.31 [0.69, 2.00] | No |
| `pure_vegetarian` | 0.10 | bionutrigpt_minus_catboost | 84 | 85 | 84 | -1.00 pp [-3.00, 0.00] | 0.87 [0.38, 1.45] | No |
| `pure_vegetarian` | 0.20 | bionutrigpt_minus_catboost | 88 | 91 | 88 | -3.00 pp [-7.00, 0.00] | 0.69 [0.24, 1.27] | No |
| `pure_vegetarian` | 0.30 | bionutrigpt_minus_catboost | 92 | 92 | 92 | +0.00 pp [0.00, 0.00] | 0.63 [0.21, 1.16] | No |
| `pure_vegetarian` | 0.50 | bionutrigpt_minus_catboost | 95 | 94 | 94 | +1.00 pp [0.00, 3.00] | 0.47 [0.09, 0.92] | No |
| `veg_nonveg_combo` | 0.05 | bionutrigpt_minus_catboost | 99 | 99 | 99 | +0.00 pp [0.00, 0.00] | 1.85 [1.04, 2.76] | No |
| `veg_nonveg_combo` | 0.10 | bionutrigpt_minus_catboost | 99 | 100 | 99 | -1.00 pp [-3.00, 0.00] | 1.89 [1.06, 2.80] | No |
| `veg_nonveg_combo` | 0.20 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.39 [0.71, 2.24] | No |
| `veg_nonveg_combo` | 0.30 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.06 [0.48, 1.82] | No |
| `veg_nonveg_combo` | 0.50 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.25 [0.60, 1.96] | No |
| `pure_non_vegetarian` | 0.05 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.98 [1.17, 2.83] | No |
| `pure_non_vegetarian` | 0.10 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.18 [1.36, 3.08] | No |
| `pure_non_vegetarian` | 0.20 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.64 [0.92, 2.48] | No |
| `pure_non_vegetarian` | 0.30 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.36 [0.70, 2.10] | No |
| `pure_non_vegetarian` | 0.50 | bionutrigpt_minus_catboost | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.50 [0.85, 2.23] | No |
| `pure_vegetarian` | 0.05 | bionutrigpt_minus_ft_transformer | 80 | 82 | 80 | -2.00 pp [-5.00, 0.00] | 1.36 [0.73, 2.07] | No |
| `pure_vegetarian` | 0.10 | bionutrigpt_minus_ft_transformer | 84 | 84 | 84 | +0.00 pp [0.00, 0.00] | 1.58 [0.92, 2.31] | No |
| `pure_vegetarian` | 0.20 | bionutrigpt_minus_ft_transformer | 88 | 89 | 87 | -1.00 pp [-5.00, 2.00] | 0.77 [0.28, 1.35] | No |
| `pure_vegetarian` | 0.30 | bionutrigpt_minus_ft_transformer | 92 | 92 | 92 | +0.00 pp [0.00, 0.00] | 1.01 [0.48, 1.65] | No |
| `pure_vegetarian` | 0.50 | bionutrigpt_minus_ft_transformer | 95 | 94 | 94 | +1.00 pp [0.00, 3.00] | 0.34 [0.06, 0.69] | No |
| `veg_nonveg_combo` | 0.05 | bionutrigpt_minus_ft_transformer | 99 | 99 | 99 | +0.00 pp [0.00, 0.00] | 2.30 [1.37, 3.31] | No |
| `veg_nonveg_combo` | 0.10 | bionutrigpt_minus_ft_transformer | 99 | 100 | 99 | -1.00 pp [-3.00, 0.00] | 2.32 [1.42, 3.30] | No |
| `veg_nonveg_combo` | 0.20 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.02 [1.23, 2.95] | No |
| `veg_nonveg_combo` | 0.30 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.91 [1.12, 2.83] | No |
| `veg_nonveg_combo` | 0.50 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.39 [0.70, 2.21] | No |
| `pure_non_vegetarian` | 0.05 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.42 [1.55, 3.34] | No |
| `pure_non_vegetarian` | 0.10 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.66 [1.76, 3.62] | No |
| `pure_non_vegetarian` | 0.20 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.23 [1.39, 3.11] | No |
| `pure_non_vegetarian` | 0.30 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.44 [0.78, 2.13] | No |
| `pure_non_vegetarian` | 0.50 | bionutrigpt_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.72 [0.97, 2.48] | No |
| `pure_vegetarian` | 0.05 | catboost_minus_ft_transformer | 80 | 82 | 80 | -2.00 pp [-5.00, 0.00] | 1.55 [0.85, 2.29] | No |
| `pure_vegetarian` | 0.10 | catboost_minus_ft_transformer | 85 | 84 | 84 | +1.00 pp [0.00, 3.00] | 1.52 [0.85, 2.27] | No |
| `pure_vegetarian` | 0.20 | catboost_minus_ft_transformer | 91 | 89 | 88 | +2.00 pp [-2.00, 6.00] | 0.53 [0.15, 1.01] | No |
| `pure_vegetarian` | 0.30 | catboost_minus_ft_transformer | 92 | 92 | 92 | +0.00 pp [0.00, 0.00] | 1.18 [0.58, 1.86] | No |
| `pure_vegetarian` | 0.50 | catboost_minus_ft_transformer | 94 | 94 | 93 | +0.00 pp [-3.00, 3.00] | 0.45 [0.10, 0.86] | No |
| `veg_nonveg_combo` | 0.05 | catboost_minus_ft_transformer | 99 | 99 | 99 | +0.00 pp [0.00, 0.00] | 2.15 [1.29, 3.13] | No |
| `veg_nonveg_combo` | 0.10 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.37 [1.43, 3.33] | No |
| `veg_nonveg_combo` | 0.20 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.26 [1.44, 3.23] | No |
| `veg_nonveg_combo` | 0.30 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.56 [0.82, 2.45] | No |
| `veg_nonveg_combo` | 0.50 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.46 [0.78, 2.24] | No |
| `pure_non_vegetarian` | 0.05 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.46 [1.60, 3.34] | No |
| `pure_non_vegetarian` | 0.10 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.49 [1.61, 3.38] | No |
| `pure_non_vegetarian` | 0.20 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.94 [1.14, 2.77] | No |
| `pure_non_vegetarian` | 0.30 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 2.00 [1.19, 2.88] | No |
| `pure_non_vegetarian` | 0.50 | catboost_minus_ft_transformer | 100 | 100 | 100 | +0.00 pp [0.00, 0.00] | 1.83 [1.10, 2.64] | No |

---

## 5. Detailed 45-Cell Grid Results (Model vs. Rules-Only)

| Dietary Mode | Model | Threshold | Model Accepted | Valid Pairs ($N_{\text{both}}$) | Feasibility Diff vs. Rules (pp) [95% CI] | Feasibility Excludes Zero? | Common-Ref Pass (Attempts) [95% CI] | Common-Ref Pass (Accepted) | Mean Dish Changes [95% CI] | Normalized Nutrient Dev [95% CI] | Historical Unstable Dev |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pure_vegetarian` | bionutrigpt | 0.05 | 80 | 80 | -16.00 pp [-23.00, -9.00] | Yes | 80.0% [72.0, 87.0] | 100.0% | 4.97 [3.96, 6.01] | 8.84% [6.87, 10.85] | 4187% |
| `pure_vegetarian` | bionutrigpt | 0.10 | 84 | 84 | -12.00 pp [-19.00, -6.00] | Yes | 84.0% [77.0, 91.0] | 100.0% | 4.32 [3.33, 5.35] | 7.58% [5.68, 9.54] | 848% |
| `pure_vegetarian` | bionutrigpt | 0.20 | 88 | 88 | -8.00 pp [-14.00, -3.00] | Yes | 88.0% [81.0, 94.0] | 100.0% | 2.40 [1.57, 3.33] | 4.04% [2.55, 5.71] | 367% |
| `pure_vegetarian` | bionutrigpt | 0.30 | 92 | 92 | -4.00 pp [-8.00, -1.00] | Yes | 92.0% [86.0, 97.0] | 100.0% | 1.49 [0.84, 2.20] | 2.55% [1.36, 3.81] | 103% |
| `pure_vegetarian` | bionutrigpt | 0.50 | 95 | 95 | -1.00 pp [-3.00, 0.00] | No | 95.0% [90.0, 99.0] | 100.0% | 0.48 [0.10, 0.94] | 0.83% [0.15, 1.66] | 44% |
| `pure_vegetarian` | catboost | 0.05 | 80 | 80 | -16.00 pp [-23.00, -9.00] | Yes | 80.0% [72.0, 87.0] | 100.0% | 5.42 [4.43, 6.42] | 9.92% [7.92, 11.97] | 6797% |
| `pure_vegetarian` | catboost | 0.10 | 85 | 85 | -11.00 pp [-17.00, -5.00] | Yes | 85.0% [78.0, 92.0] | 100.0% | 4.32 [3.28, 5.38] | 7.38% [5.58, 9.27] | 847% |
| `pure_vegetarian` | catboost | 0.20 | 91 | 91 | -5.00 pp [-10.00, -1.00] | Yes | 91.0% [85.0, 96.0] | 100.0% | 2.12 [1.37, 2.97] | 3.56% [2.23, 5.00] | 152% |
| `pure_vegetarian` | catboost | 0.30 | 92 | 92 | -4.00 pp [-8.00, -1.00] | Yes | 92.0% [86.0, 97.0] | 100.0% | 1.64 [0.96, 2.40] | 2.80% [1.59, 4.16] | 109% |
| `pure_vegetarian` | catboost | 0.50 | 94 | 94 | -2.00 pp [-5.00, 0.00] | No | 94.0% [89.0, 98.0] | 100.0% | 0.57 [0.18, 1.05] | 1.08% [0.29, 2.08] | 30% |
| `pure_vegetarian` | ft_transformer | 0.05 | 82 | 82 | -14.00 pp [-21.00, -8.00] | Yes | 82.0% [74.0, 89.0] | 100.0% | 4.96 [3.96, 5.99] | 8.83% [6.79, 10.80] | 2858% |
| `pure_vegetarian` | ft_transformer | 0.10 | 84 | 84 | -12.00 pp [-19.00, -6.00] | Yes | 84.0% [77.0, 91.0] | 100.0% | 4.07 [3.11, 5.10] | 7.15% [5.32, 9.07] | 2709% |
| `pure_vegetarian` | ft_transformer | 0.20 | 89 | 89 | -7.00 pp [-12.00, -2.00] | Yes | 89.0% [82.0, 95.0] | 100.0% | 2.21 [1.39, 3.07] | 3.63% [2.20, 5.14] | 159% |
| `pure_vegetarian` | ft_transformer | 0.30 | 92 | 92 | -4.00 pp [-8.00, -1.00] | Yes | 92.0% [86.0, 97.0] | 100.0% | 1.79 [1.07, 2.57] | 3.02% [1.73, 4.41] | 149% |
| `pure_vegetarian` | ft_transformer | 0.50 | 94 | 94 | -2.00 pp [-5.00, 0.00] | No | 94.0% [89.0, 98.0] | 100.0% | 0.66 [0.21, 1.16] | 1.24% [0.35, 2.24] | 51% |
| `veg_nonveg_combo` | bionutrigpt | 0.05 | 99 | 99 | -1.00 pp [-3.00, 0.00] | No | 99.0% [97.0, 100.0] | 100.0% | 8.55 [7.40, 9.69] | 12.08% [10.21, 14.07] | 533% |
| `veg_nonveg_combo` | bionutrigpt | 0.10 | 99 | 99 | -1.00 pp [-3.00, 0.00] | No | 99.0% [97.0, 100.0] | 100.0% | 7.09 [5.94, 8.31] | 9.85% [8.01, 11.84] | 48% |
| `veg_nonveg_combo` | bionutrigpt | 0.20 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 3.82 [2.76, 4.98] | 5.08% [3.47, 6.85] | 12% |
| `veg_nonveg_combo` | bionutrigpt | 0.30 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 2.97 [1.93, 4.10] | 4.04% [2.54, 5.80] | 14% |
| `veg_nonveg_combo` | bionutrigpt | 0.50 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 1.04 [0.46, 1.67] | 1.44% [0.63, 2.36] | 8% |
| `veg_nonveg_combo` | catboost | 0.05 | 99 | 99 | -1.00 pp [-3.00, 0.00] | No | 99.0% [97.0, 100.0] | 100.0% | 9.03 [7.87, 10.16] | 12.56% [10.73, 14.52] | 529% |
| `veg_nonveg_combo` | catboost | 0.10 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 7.05 [5.80, 8.26] | 9.79% [7.89, 11.78] | 23% |
| `veg_nonveg_combo` | catboost | 0.20 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 3.34 [2.29, 4.46] | 4.39% [2.92, 6.07] | 10% |
| `veg_nonveg_combo` | catboost | 0.30 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 2.57 [1.64, 3.62] | 3.93% [2.46, 5.69] | 8% |
| `veg_nonveg_combo` | catboost | 0.50 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 1.13 [0.48, 1.82] | 1.75% [0.70, 2.89] | 9% |
| `veg_nonveg_combo` | ft_transformer | 0.05 | 99 | 99 | -1.00 pp [-3.00, 0.00] | No | 99.0% [97.0, 100.0] | 100.0% | 8.38 [7.15, 9.58] | 12.15% [10.10, 14.24] | 127% |
| `veg_nonveg_combo` | ft_transformer | 0.10 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 6.87 [5.61, 8.10] | 9.35% [7.42, 11.33] | 32% |
| `veg_nonveg_combo` | ft_transformer | 0.20 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 4.27 [3.12, 5.46] | 6.19% [4.40, 8.04] | 23% |
| `veg_nonveg_combo` | ft_transformer | 0.30 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 2.92 [1.92, 3.97] | 4.49% [2.91, 6.32] | 1434% |
| `veg_nonveg_combo` | ft_transformer | 0.50 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 1.49 [0.78, 2.32] | 2.08% [1.06, 3.29] | 5% |
| `pure_non_vegetarian` | bionutrigpt | 0.05 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 7.53 [6.50, 8.55] | 10.94% [9.32, 12.57] | 429% |
| `pure_non_vegetarian` | bionutrigpt | 0.10 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 6.44 [5.41, 7.51] | 9.33% [7.68, 11.07] | 409% |
| `pure_non_vegetarian` | bionutrigpt | 0.20 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 4.16 [3.09, 5.25] | 5.26% [3.86, 6.75] | 2164% |
| `pure_non_vegetarian` | bionutrigpt | 0.30 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 2.84 [1.95, 3.84] | 3.90% [2.58, 5.36] | 44% |
| `pure_non_vegetarian` | bionutrigpt | 0.50 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 1.32 [0.72, 1.98] | 1.94% [1.03, 3.05] | 19% |
| `pure_non_vegetarian` | catboost | 0.05 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 7.71 [6.66, 8.78] | 11.50% [9.82, 13.28] | 3394% |
| `pure_non_vegetarian` | catboost | 0.10 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 6.23 [5.21, 7.36] | 9.03% [7.37, 10.84] | 2238% |
| `pure_non_vegetarian` | catboost | 0.20 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 3.63 [2.62, 4.65] | 4.79% [3.42, 6.29] | 28% |
| `pure_non_vegetarian` | catboost | 0.30 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 2.95 [2.06, 3.90] | 4.20% [2.85, 5.60] | 2178% |
| `pure_non_vegetarian` | catboost | 0.50 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 1.37 [0.77, 2.07] | 2.01% [1.07, 3.08] | 39% |
| `pure_non_vegetarian` | ft_transformer | 0.05 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 7.15 [6.10, 8.19] | 10.31% [8.69, 12.00] | 426% |
| `pure_non_vegetarian` | ft_transformer | 0.10 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 6.08 [4.99, 7.21] | 8.70% [7.00, 10.39] | 53% |
| `pure_non_vegetarian` | ft_transformer | 0.20 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 3.80 [2.81, 4.83] | 5.16% [3.72, 6.60] | 29% |
| `pure_non_vegetarian` | ft_transformer | 0.30 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 2.74 [1.87, 3.67] | 4.02% [2.67, 5.50] | 25% |
| `pure_non_vegetarian` | ft_transformer | 0.50 | 100 | 100 | +0.00 pp [0.00, 0.00] | No | 100.0% [100.0, 100.0] | 100.0% | 1.80 [1.07, 2.57] | 2.80% [1.60, 4.09] | 2154% |

---

## 6. Verification and Safeguard Test Results

- **Actual Runner Worker Safeguards Tested (`test_real_runner_safeguards.py`):**
  1. Matching valid artifact: solver is not called (resumed = True, call_count = 0).
  2. Changed input (e.g. participant age): stale artifact rejected, solver called.
  3. Changed bound or target: stale artifact rejected, solver called.
  4. Changed prediction or catalog binding: stale artifact rejected, solver called.
  5. Missing fingerprint: legacy artifact rejected according to strict policy, solver called.
  6. Truncated or corrupted JSON: rejected, solver called.
  7. Incomplete output despite matching fingerprint: rejected, solver called.
  8. Recorded solver failures: valid failure artifact resumes without silent retry (solver is not called).
  - All 9 actual runner tests passed.
- **Adversarial Contract Test Suite (11/11 Passed):** Exercised corrupted inputs, nonfinite probabilities, missing nutrient keys, duplicate basket items, invalid portions, accounting discrepancies, clinical precedence rules (reported diagnoses and measured BMI override low model predictions), and stale/corrupted resume files.
- **Independent Case Audit (4,800/4,800 Passed):** Read-only verification confirmed zero discrepancies in basket contents, delivered nutrition completeness, constraint satisfaction, and failure codes.
- **Safe Resume Verified:** SHA-256 fingerprinting confirmed deterministic caching and re-solve triggers.

---

## 7. Methodological Disclosures & Provenance

- **Post-Hoc Status:** This audit was conducted post-hoc to resolve statistical, metric, and reporting defects in the completed study. While the protocol was documented prior to study execution, the direct model paired contrasts, valid-pair denominators, and normalized nutrient deviation endpoints are post-hoc repairs.
- **Exploratory Threshold Grid:** The threshold grid (0.05 to 0.50) was chosen as an exploratory sensitivity analysis across decision boundaries. These thresholds do not represent clinically validated decision cutoffs.
- **In-Silico Nature & No Clinical Efficacy Claim:** This study evaluates mathematical constraint satisfaction in an in-silico MILP solver using a 49-entry food catalog (20 INDB recipe variants + 29 USDA SR Legacy single-food proxies; 43 eligible in combo/non-veg, 37 in pure vegetarian, 6 excluded for missing active data) and US NHANES demographic profiles. These results do not constitute clinical trial evidence, causal disease risk reduction, or external Indian clinical validity.
- **Worker Process Discrepancy:** `protocol.json` originally recorded `worker_processes: 8`. During runtime execution, the runner pool was scaled to 10 processes (`min(10, os.cpu_count())`) on the 12-core host. This discrepancy is documented in `correction_log.json`. CPU contention effects on marginal time-limited solves remain unmeasured without a controlled benchmark.
- **Artifact Preservation:** All original v1 study files remain preserved in `Saved Model/stage1_downstream_study_v1/`, snapshotted under `Saved Model/stage1_downstream_study_v1_audit_v2/original_v1_snapshot/`, and Audit v2 files snapshotted under `Saved Model/stage1_downstream_study_v1_audit_v3/audit_v2_snapshot/`.
