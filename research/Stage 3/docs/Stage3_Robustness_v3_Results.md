# Stage 3 Robustness Benchmark v3 Results — 24 November 2026

## Executive Summary & Study Status

The full controlled engineering robustness evaluation of the Stage 3 target-aware meal optimizer has been executed, completely verified independently, and recorded with zero discrepancies.

- **Status:** **COMPLETED AND VERIFIED**
- **Protocol:** v3 frozen specification (`Saved Model/stage3_robustness_v3/protocol.json`)
- **Cohort:** 100 participants from NHANES 2017–March 2020 pre-pandemic holdout (`Saved Model/stage3_budget_experiment_v1/cases_LOCAL.json`, seed `20261124`).
  - *Disclosure:* Reuses the budget-study cohort; this is an exploratory robustness extension, not an independent confirmatory study.
- **Stage 2 Target Source:** Model-guided counterfactual targets (`Saved Model/stage3_budget_experiment_v1/stage2_targets/case_*_model_guided.json`).
- **Dietary Modes (3):** `pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`.
- **Objectives (2):**
  - **Variant A (Bounds-Based):** Calorie deviation + food selection cost.
  - **Variant B (Target-Aware, $\lambda=1.0$):** Calorie deviation + food selection cost + normalized $L_1$ deviations from Stage 2 nutrient targets.
- **Budget:** 5.0 seconds per solve passed directly to HiGHS via `scipy.optimize.milp` without silent clamping.
- **Execution:** Sequential solver execution; total 4,200 checkpoint records (3,000 optimization solver calls + 1,200 arithmetic nutrient replays).
- **Verification:** Both the plan reconstruction audit (`r.verify()`) and the separate independent statistical verifier (`verify_stage3_robustness_v3.py`) passed byte-for-byte, verifying 19 bootstrap confidence intervals, 21 exact paired tests, Holm step-down adjustments, and active rejection of altered confidence intervals.

---

## 1. Verified Results Table

All point estimates average dietary modes within each participant before computing cohort means (equal participant weighting, $N=100$). Confidence intervals are 95% percentiles from 10,000 paired participant-cluster bootstrap resamples (seed `20261124`). Exact tests are two-sided paired binomial tests within each mode and scenario; Holm adjustments apply across the declared 21-test family.

| Scenario | Variant A Accepted | Variant B Accepted | Difference (pp) [95% CI] | Joint $L_1$ Target Error Diff [95% CI] ($N$) | Runtime Diff (s) [95% CI] | Mode Exact Tests (Wins / Losses / Raw $p$ / Holm $p$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nominal Baseline** | 299/300 (99.7%) | 299/300 (99.7%) | 0.00 [0.00, 0.00] | -0.4877 [-0.5122, -0.4629] (100) | -0.5217 [-0.6613, -0.3889] | **PV:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**Combo:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**NonVeg:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$) |
| **Food Availability 10%** | 296/300 (98.7%) | 296/300 (98.7%) | 0.00 [0.00, 0.00] | -0.4370 [-0.4607, -0.4130] (100) | -0.3448 [-0.4696, -0.2235] | **PV:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**Combo:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**NonVeg:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$) |
| **Food Availability 20%** | 224/300 (74.7%) | 224/300 (74.7%) | 0.00 [0.00, 0.00] | -0.4657 [-0.4972, -0.4331] (100) | -0.3366 [-0.4806, -0.2012] | **PV:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**Combo:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**NonVeg:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$) |
| **Dietary Exclusion: Dairy** | 200/300 (66.7%) | 200/300 (66.7%) | 0.00 [0.00, 0.00] | -0.5719 [-0.5998, -0.5421] (100) | -0.6107 [-0.7732, -0.4594] | **PV:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**Combo:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**NonVeg:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$) |
| **Dietary Exclusion: Legumes** | 100/300 (33.3%) | 100/300 (33.3%) | 0.00 [0.00, 0.00] | -0.5417 [-0.5780, -0.5050] (100) | -0.2548 [-0.3351, -0.1760] | **PV:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**Combo:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**NonVeg:** 0/0 ($p=1.00$, $p_{\text{adj}}=1.00$) |
| **Nutrient Sensitivity $\pm 5\%$** | 236/300 (78.7%) | 215/300 (71.7%) | -7.00 [-14.00, -0.00] | -0.5014 [-0.5339, -0.4684] (90) | *N/A (Replay)* | **PV:** 15/33 ($p=0.0133$, $p_{\text{adj}}=0.2790$)<br>**Combo:** 20/15 ($p=0.4996$, $p_{\text{adj}}=1.00$)<br>**NonVeg:** 12/20 ($p=0.2153$, $p_{\text{adj}}=1.00$) |
| **Nutrient Sensitivity $\pm 10\%$** | 192/300 (64.0%) | 159/300 (53.0%) | -11.00 [-19.67, -2.33] | -0.5185 [-0.5586, -0.4797] (75) | *N/A (Replay)* | **PV:** 19/38 ($p=0.0164$, $p_{\text{adj}}=0.3270$)<br>**Combo:** 21/21 ($p=1.00$, $p_{\text{adj}}=1.00$)<br>**NonVeg:** 15/29 ($p=0.0488$, $p_{\text{adj}}=0.9266$) |

---

## 2. Key Scientific & Engineering Findings

### A. Structural Robustness under Food Removals and Dietary Exclusions
- Across all five optimization solver scenarios (`nominal_baseline`, `food_avail_10pct`, `food_avail_20pct`, `dietary_excl_dairy`, `dietary_excl_legume`), Variant A and Variant B achieved **identical plan acceptance rates**:
  - Nominal Baseline: 299/300 (99.7%) in both objectives. Failure classifications are audited separately; the cause is not inferred from the calorie value.
  - Food Availability 10% removal: 296/300 (98.7%) in both objectives.
  - Food Availability 20% removal: 224/300 (74.7%) in both objectives.
  - Dietary Exclusion (Dairy-free, `allow_zero_dairy=True`): 200/300 (66.7%) in both objectives. See the all-rule audit for failure codes; no nutrient-feasibility cause is inferred solely from a rejected output.
  - Dietary Exclusion (Legumes/dal-pulse): 100/300 (33.3%) in both objectives (PV and combo require dal by category rule, returning clean `insufficient_nutrient_data` failures; all 100 non-veg cases solved by substituting animal protein).
- **Zero Plan Availability Loss:** The experimental target-aware optimizer suffered no loss in meal plan generation rate compared to the bounds-based baseline under supply disruptions or hard exclusions.
- **Substantially Lower Target Error:** Among jointly valid plans, Variant B consistently reduced normalized $L_1$ target attainment error by $-0.4370$ to $-0.5719$ (exploratory 95% confidence intervals reported; no corresponding target-error p-value was calculated), confirming that the target-aware objective actively aligns discrete food choices with Stage 2 guidance.
- **Observed Runtime:** Mean solver wall time for Variant B was slightly faster than Variant A (differences of $-0.25$s to $-0.61$s across scenarios). As prespecified, this is an empirical observation and does not prove faster branch-and-bound pruning causality.

### B. Uncertainty Trade-offs under Nutrient Perturbations
- When nominal plans were re-evaluated under perturbed nutrient compositions ($\pm 5\%$ and $\pm 10\%$):
  - Both optimizers experienced compliance erosion (at $\pm 5\%$: Variant A 78.7%, Variant B 71.7%; at $\pm 10\%$: Variant A 64.0%, Variant B 53.0%).
  - Point estimates reflect lower available compliant plan rates for Variant B ($-7.00$ pp at $\pm 5\%$; $-11.00$ pp at $\pm 10\%$).
  - **Multiplicity Adjustment:** Across the declared 21-test family, within-mode exact paired differences were **not statistically significant** after Holm-Bonferroni correction (all Holm-adjusted $p \ge 0.2789$).
  - **Mechanism not established:** Smaller distances to some constraint boundaries are a hypothesis requiring direct margin and all-rule failure analysis. The fixed-perturbation comparison alone does not establish why compliance differs.
  - Among participants whose nominal plans remained compliant under perturbation ($N=90$ at $\pm 5\%$, $N=75$ at $\pm 10\%$), Variant B maintained its target attainment advantage ($-0.5014$ and $-0.5185$).

---

## 3. Disclosures, Constraints & Scope Boundaries

1. **Cohort Reuse:** This study reused 100 participants from the previously inspected budget-study cohort; case 0 was used during pilot development. Findings represent an exploratory engineering robustness extension, not an independent prospective validation.
2. **Fixed Perturbation Draws:** One fixed proportional factor per food scales its known nutrient values, preserving the catalog's existing energy-to-nutrient relationships, unknowns and zeros. It does not enforce 4P + 4C + 9F = E. Inference is conditional on these fixed scenarios, not an empirical nutrient-error distribution.
3. **Allergen & Clinical Scope:** The dairy exclusion demonstrates that the solver satisfies mathematical constraints under `allow_zero_dairy = True`. However, because catalog annotations do not record spice or facility cross-contact, this does not certify commercial allergen-free safety.
4. **Engineering Evaluation:** These findings evaluate the algorithmic stability of discrete MILP solvers under constraint modifications. They do not demonstrate patient adherence, dietary acceptability, cardiometabolic health benefits, or external validity in Indian clinical populations.
5. **No Production Changes:** Production code (`stage3_milp_solver.py`, `backend/pipeline_service.py`), Stage 1 models, and Stage 2 targets remain 100% untouched.

---

## 4. Verification Artifacts & Checksums

- **Verification Receipt:** `Saved Model/stage3_robustness_v3/full/verification.json`
- **Independent Statistical Verification Receipt:** `Saved Model/stage3_robustness_v3/full/independent_statistics_verification.json`
- **Analysis Data:** `Saved Model/stage3_robustness_v3/full/analysis.json` (SHA-256: `f123eb3922d34cd2d7e887587f6e361cbf4deceff6fcac097c4140acbd393488`)
- **Independent Verifier Script:** `verify_stage3_robustness_v3.py` (SHA-256: `6bb5784ff6aa071e354b8c3bf568a99a13eccc8e3a446f6c3055bdf587c87fa2`)
- **Total Checkpoints Verified:** 4,200 JSON files byte-for-byte.

### Stage 3 retrospective failure/margin audit — 24 November 2026

Completed all-rule audit of 4,200 saved records without solver/model reruns: zero acceptance disagreements; checkpoint hashes unchanged; 94 paired margin intervals independently reproduced from rule rows. Under fixed 10% nutrient perturbation, PUFA upper-bound violations A/B=19/39, folate DFE lower-bound violations=1/16, potassium upper-bound violations=21/3. Target-aware nominal PUFA/folate margins were smaller, potassium margins larger: evidence of rule-specific vulnerability, not universal boundary proximity or clinical harm. Violation counts overlap. Nominal rejected plans were coded infeasible, not timeout; dairy-exclusion failures were pure-vegetarian, not non-vegetarian; legume exclusions conflict with mandatory pulse categories in vegetarian/mixed modes. Removed unsupported target-error p<0.001 and exact 4P+4C+9F=E claims. Full benchmark remains completed; this is additional retrospective analysis, not a new trial. See Research Paper/Stage3_Robustness_Failure_Margin_Audit_v1.md. No production changes.
