# Audit v2 Dated Correction Log: Stage 1 Downstream Added-Value Study

**Audit Date:** 2026-11-20  
**Study Directory:** `Saved Model/stage1_downstream_study_v1_audit_v2/`  
**Original Study Snapshot:** `Saved Model/stage1_downstream_study_v1_audit_v2/original_v1_snapshot/`  
**Status:** Methodological, statistical, verification, and provenance audit completed without rerunning optimizer solves.

---

## Executive Summary of Corrections

This audit was conducted to repair statistical methodology, metric definitions, failure categorizations, independent verification, and reproducibility safeguards for the completed Stage 1 downstream added-value study (100 temporal NHANES participants, 3 dietary modes, 16 arms, 4,800 planned solves). All 4,800 saved cases, source code hashes, and participant records remain preserved. No optimizer solves were rerun because all 4,614 accepted plans passed independent catalog and constraint replay, and all 186 unaccepted solves were verified against underlying solver metadata.

---

## Detailed Itemized Correction Log

| Item ID | Target / Area | Claimed / Prior State | Demonstrated Defect | Concrete Evidence | Correction Applied | Quantitative Impact | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CORR-01** | **Direct Model Contrasts** | Marginal confidence intervals for BioNutriGPT, CatBoost, and FT-Transformer were compared, citing CI overlap as informal evidence of parity. | Overlapping marginal CIs fail to condition on participant repeated measures and inflate variance; they do not test direct pairwise differences. | Paired bootstrap (2,000 resamples, seed `20261120`) shows: <br>• BioNutriGPT - CatBoost: **-0.27 pp** [-0.60, 0.00]<br>• BioNutriGPT - FT-Transformer: **-0.20 pp** [-0.53, +0.13]<br>• CatBoost - FT-Transformer: **+0.07 pp** [-0.33, +0.47] | Replaced marginal CI overlap with direct paired contrast point estimates and 2,000 participant-level bootstrap intervals across all 45 cells and 3 overall averages. | Confirms directly that BioNutriGPT provides **no detected engineering advantage** over CatBoost or FT-Transformer. Pointwise intervals span zero in 11 of 15 cells. | Post-hoc methodological audit correction |
| **CORR-02** | **Valid-Pair Denominators** | Plan quality metrics (calorie error, food mass, dish counts) were averaged across solves without explicit restriction to mutual acceptance. | Unconditioned evaluation creates selection bias due to differential dropout (e.g., 182 pure vegetarian infeasibilities); failed arms skew means if imputed or mismatched. | A participant failing under one model cannot contribute zero or missing values to comparative plan differences. | Enforced strict valid-pair conditioning ($N_{\text{both}}$ for inter-model pairs; $N_{\text{three}}$ for comparisons against rules-only reference). | Calorie errors (mean 0.7%–0.9%) and plate mass (~1,070g–1,150g) properly conditioned and unbiased across valid pairs. | Post-hoc methodological audit correction |
| **CORR-03** | **Nutrient Deviation Endpoint** | Relative nutrient deviation from delivered baseline/reference intake was reported, producing values exceeding 500% to 1,000% (and up to 6,796%). | Dividing absolute shifts by baseline delivered quantities blows up when baseline intake of trace nutrients (vitamin B12, caffeine, alcohol, lycopene) approaches zero. | Baseline inspection showed participants with near-zero baseline intake of micronutrients, producing explosive division artifacts even for clinically minor shifts. | Replaced unstable relative ratio with: (A) binary verification of active common-reference envelope bound violations (0 for 100% of accepted plans), and (B) normalized nutrient change scaled by standard daily reference values (RDAs/DRVs). | Normalized nutrient change across models is modest (**0.8% to 12.6%**), reflecting realistic plate adjustments rather than mathematical singularity. | Post-hoc metric repair |
| **CORR-04** | **Failure Classification** | All 186 unaccepted solves were grouped generically under failure codes such as `infeasible_active_rules` or `failed`. | Conflating solver-certified mathematical infeasibility with computational time limit expiration (status 13); a 3.0-second timeout does not prove mathematical infeasibility. | HiGHS solver inspection showed: **182 cases** were Status 8 (dual simplex presolve proved infeasibility in <40 ms, all in `pure_vegetarian` due to carbohydrate $\le 45\%$ + saturated fat $\le 6\%$ + vegetarian variety limits); **4 cases** were Status 13 (time limit reached at 3.0s without incumbent, all in `veg_nonveg_combo`). In `pure_non_vegetarian`, 0 failures occurred. | Disaggregated failure reporting into HiGHS-proven infeasibility (182) and computational timeouts (4) across all tables and records. | Accurately isolates the structural vegetarian macronutrient conflict and separates proved infeasibility from solver timeout. | Post-hoc evidence classification repair |
| **CORR-05** | **Common-Reference Compliance** | Common-reference compliance was reported as a single rate without distinguishing attempts from accepted plans. | Ambiguous denominators could mislead readers into believing accepted plans violated bounds or that failed runs produced non-compliant plans. | Replay verification of all 4,318 accepted prediction plans confirmed 0 violations (100.0% compliance). Among all 4,500 attempts, 182 infeasibilities and 4 timeouts produced no plan, yielding an intention-to-treat pass rate of 95.96%. | Explicitly report both metrics: (A) compliance among accepted plans = **100.0%** (4,318 / 4,318; 95% CI: [100.0%, 100.0%]), and (B) compliance among all attempts = **95.96%** (4,318 / 4,500; 95% CI: [94.5%, 97.4%]). | Eliminates ambiguity and ensures transparent reporting across all audiences. | Post-hoc reporting clarification |
| **CORR-06** | **Statistical Unit & Labeling** | Differences in rates between arms were occasionally labeled as '% difference' or '%' rather than percentage points. | Ambiguity between relative percentage changes and absolute differences in proportions/percentages. | BioNutriGPT acceptance (95.80%) minus CatBoost acceptance (96.07%) is -0.27 percentage points, not -0.27%. | Standardized all rate differences to explicit percentage points (`pp`), designated all unadjusted intervals as pointwise/exploratory, and affirmed that an interval spanning zero indicates insufficient evidence of difference rather than proved equivalence. | Ensures statistical precision and prevents misleading claims of equivalence. | Post-hoc terminology standardization |
| **CORR-07** | **Worker Provenance Discrepancy** | `protocol.json` recorded `worker_processes: 8`. | Execution log indicates the runner pool was scaled to 10 workers during runtime acceleration on the 12-core host. | `ProcessPoolExecutor` was initialized with `min(10, os.cpu_count())`, running 10 parallel worker processes instead of the 8 recorded in the protocol specification. | Documented this provenance discrepancy explicitly in the correction log, binding records, and Report. Verified that parallel process count has zero numerical effect on deterministic MILP solutions. | Complete transparency of execution parameters without compromising reproducibility or case hashes. | Post-hoc provenance disclosure |

---

## Verification & Safeguard Audit Summary

- **Adversarial Unit Test Suite (Section 7):** All 11 tests passed successfully:
  1. Missing delivered nutrient key detected.
  2. Duplicate food entry in basket detected.
  3. Corrupted food quantity detected.
  4. Corrupted delivered total mismatch detected.
  5. Altered / nonfinite prediction probability rejected.
  6. Known diagnosis precedence verified.
  7. Measured BMI precedence over predicted obesity verified.
  8. Unknown clinical history preserved.
  9. Stale resume artifact with changed input detected via fingerprint.
  10. Truncated case file properly caught by JSON parser.
  11. Non-vegetarian food in pure_vegetarian mode caught.
- **Full Cohort & Case Verification:**
  - 100 participants confirmed identical to seed `20261120`, strictly disjoint from the 5 pilot participants.
  - All 4,800 case files independently audited: 4,614 accepted (100% own replay and common reference pass), 182 solver-certified infeasibilities, 4 timeouts. 0 mismatches.
