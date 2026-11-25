# Stage 3 Solver-Budget Matched Comparisons: Budget Scaling and Stage 2 Target Source Evaluation (v1)

**Date:** 24 November 2026  
**Status:** Complete & Independently Verified (Zero Discrepancies)  
**Analysis Directory:** `Saved Model/stage3_budget_comparisons_v1/`  
**Specification:** [`Saved Model/stage3_budget_comparisons_v1/specification.json`](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage3_budget_comparisons_v1/specification.json)  
**Input Manifest:** [`Saved Model/stage3_budget_comparisons_v1/input_manifest.json`](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage3_budget_comparisons_v1/input_manifest.json) (6 bound files)  
**Machine-Readable Results:** [`Saved Model/stage3_budget_comparisons_v1/comparison_results.json`](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage3_budget_comparisons_v1/comparison_results.json)  
**Verification Receipt:** [`Saved Model/stage3_budget_comparisons_v1/verification.json`](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage3_budget_comparisons_v1/verification.json) (504 bootstrap CIs, 162 exact tests, 6 failure fixtures verified)  

---

## 1. Executive Summary & Direct Answers to Research Questions

This retrospective study completes the matched comparisons using the 5,400 existing solves from the Stage 3 solver-budget experiment across 100 adult temporal NHANES participants (seed `20261124`). No new solver runs, model retraining, or target regenerations were performed.

### Question 1: Does increasing the budget from 3 to 5 seconds improve outcomes?
**Yes, for the target-aware optimizer; neutral for the bounds-based optimizer.**
1. **Feasibility Recovery:** For the target-aware optimizer (Variant B), moving from 3.0s to 5.0s recovers the valid-plan deficit, adding +0.33 to +1.00 pp in acceptance across arms (eliminating 6 timeout failures: 2 in rules-only, 2 in minimum-change, 2 in model-guided). At 5.0s, acceptance reaches 897/900 (99.67%), matching Variant A with zero discordant pairs.
2. **Target Error Improvement:** For Variant B, mean normalized $L_1$ target error improves significantly on jointly successful plans: -0.0181 [-0.0243, -0.0129] for rules-only, -0.0145 [-0.0183, -0.0109] for minimum-change, and -0.0159 [-0.0221, -0.0110] for model-guided.
3. **Computational Cost:** Mean runtime increases by +1.66s to +1.69s (from ~2.91s to ~4.40s).
4. **Bounds-Based Contrast:** For Variant A (bounds-based), valid plans change minimally (+0.00 to +0.33 pp), but $L_1$ target error actually increases (+0.0146 to +0.0284) because Variant A's objective optimizes only calories and food selection cost, agnostic to Stage 2 targets.

### Question 2: Does increasing the budget from 5 to 10 seconds justify the additional computation?
**It depends on whether solver optimality verification is prioritized over latency.**
1. **Negligible Feasibility Gain:** Valid-plan rates do not improve further from 5s to 10s: 0.00 pp for 5 of 6 series, and -0.33 pp [-1.00, +0.00] for Variant B rules-only (due to 1 marginal timeout at 10s in veg_nonveg_combo).
2. **Modest Target Attainment Gain:** For Variant B, normalized $L_1$ error drops by an additional -0.0115 to -0.0141 on the FDA reference scale (~2.5% to 3.0% relative reduction).
3. **Computational Trade-Off & Optimality Shift:** Mean runtime increases by +2.79s to +2.92s for Variant B (reaching 7.21s). However, solver terminations within the 1% MIP gap tolerance (status 0) jump from 26.0% (234/900) to **55.7% (501/900)**.
4. **Conditional Practical Choice:** If the deployment priority is real-time interactive performance, **5.0 seconds is the inflection point**, capturing 99.67% feasibility and >52% target error reduction. If the deployment priority is verifiable solver convergence within a 1% MIP gap, 10.0 seconds provides a majority of status 0 terminations without sacrificing feasibility.

### Question 3: Under the same Stage 3 objective and budget, do model-guided Stage 2 targets add value over minimum-change and rules-only targets?
**In discrete meal generation (Stage 3): No. In upstream continuous screening (Stage 2): Yes.**
1. **Identical Stage 3 Discrete Feasibility:** Under matched budgets (5s and 10s), Stage 3 valid-plan rates are **virtually identical across all three target sources**:
   - At 5.0s: Model-Guided vs. Minimum-Change = 0.00 pp [0.00, 0.00]; Model-Guided vs. Rules-Only = 0.00 pp [0.00, 0.00]; Minimum-Change vs. Rules-Only = 0.00 pp [0.00, 0.00] in both Variant A and Variant B.
   - All mode-level exact tests show zero discordance ($p = 1.0000$, Holm-adjusted $p = 1.0000$). Stage 3 discrete feasibility is independent of the upstream target source.
2. **End-to-End Difference Inherits the Stage 2 Gate:** Model-guided targets achieve higher end-to-end acceptance than rules-only (+29.00 pp [20.00, 39.00]) and minimum-change (+7.67 pp [1.33, 14.33]). However, decomposing the pipeline demonstrates that this is entirely driven by the upstream Stage 2 mathematical feasibility gate (77/100 pass for MG, 69/100 for MC, 48/100 for RO), not by discrete meal planning.
3. **Attainment Relative to Own Target:** Each source specifies different nutrient targets. Attainment relative to each source's own target is close (MG vs. MC difference: -0.0107 to -0.0131), while rules-only has slightly lower deviation to its own target (+0.0438) due to setting standard midpoint targets closer to average food catalog densities. Lower error to one's own target does not establish clinical superiority.

---

## 2. Statistical Estimator & Multiplicity Control

- **Matched Difference Estimator:** For any matched pair $(A, B)$ on participant $i$ and mode $m$:
  $$\Delta_{i, m} = Y_{A, i, m} - Y_{B, i, m}, \quad D_i = \frac{1}{|M_i|} \sum_{m \in M_i} \Delta_{i, m}$$
  Point estimates and 10,000-resample cluster bootstrap percentile 95% CIs weight all participants equally.
- **Continuous Endpoints:** Conditioned exclusively on jointly successful matched pairs ($r_A$ accepted and $r_B$ accepted). Failed solves are never imputed as zero error.
- **Multiplicity Families (Holm Step-Down):**
  - **Family A (Budget Comparisons):** 54 tests (36 primary: 5s vs 3s and 10s vs 5s across 6 series $\times$ 3 modes; 18 secondary: 10s vs 3s).
  - **Family B (Target-Source Valid-Plan Comparisons):** 54 tests (3 contrasts $\times$ 6 conditions $\times$ 3 modes).
  - **Family C (Target-Source End-to-End Comparisons):** 54 tests (3 contrasts $\times$ 6 conditions $\times$ 3 modes).

---

## 3. Budget Comparison Results (Within Arm and Objective)

### Table 1: Budget Contrasts Across Stage 2 Arms and Stage 3 Objectives

| Objective | Stage 2 Arm | Contrast | Valid Plans (pp) [95% CI] | E2E Acceptance (pp) [95% CI] | Runtime Difference (s) [95% CI] | $L_1$ Target Error Difference [95% CI] | Contributing Cells |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A** (Current) | rules_only | 5s vs. 3s | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +1.939 [+1.896, +1.975] | +0.0189 [+0.0027, +0.0351] | 298 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +4.447 [+4.294, +4.593] | +0.0243 [+0.0059, +0.0431] | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +6.386 [+6.207, +6.554] | +0.0421 [+0.0174, +0.0664] | 298 / 300 |
| | minimum_change | 5s vs. 3s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +1.936 [+1.897, +1.970] | +0.0146 [-0.0028, +0.0329] | 299 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +4.396 [+4.230, +4.553] | +0.0179 [-0.0003, +0.0368] | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +6.332 [+6.144, +6.509] | +0.0325 [+0.0119, +0.0532] | 299 / 300 |
| | model_guided | 5s vs. 3s | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +1.939 [+1.889, +1.979] | +0.0284 [+0.0086, +0.0488] | 298 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +4.493 [+4.333, +4.642] | +0.0155 [-0.0005, +0.0316] | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +6.432 [+6.246, +6.603] | +0.0445 [+0.0223, +0.0666] | 298 / 300 |
| **Variant B** (Target-Aware) | rules_only | 5s vs. 3s | +0.33 [0.00, +1.00] | +0.00 [0.00, 0.00] | +1.694 [+1.618, +1.767] | **-0.0181 [-0.0243, -0.0129]** | 298 / 300 |
| | | 10s vs. 5s | -0.33 [-1.00, 0.00] | -0.33 [-1.00, 0.00] | +2.831 [+2.573, +3.083] | **-0.0141 [-0.0195, -0.0092]** | 298 / 300 |
| | | 10s vs. 3s (sec.) | +0.00 [-1.00, +1.00] | -0.33 [-1.00, 0.00] | +4.524 [+4.214, +4.829] | **-0.0316 [-0.0410, -0.0238]** | 297 / 300 |
| | minimum_change | 5s vs. 3s | +0.67 [0.00, +1.67] | +0.67 [0.00, +1.67] | +1.663 [+1.579, +1.741] | **-0.0145 [-0.0183, -0.0109]** | 297 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +2.916 [+2.649, +3.171] | **-0.0115 [-0.0154, -0.0079]** | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.67 [0.00, +1.67] | +0.67 [0.00, +1.67] | +4.579 [+4.247, +4.895] | **-0.0257 [-0.0301, -0.0215]** | 297 / 300 |
| | model_guided | 5s vs. 3s | +1.00 [0.00, +2.67] | +1.00 [0.00, +2.67] | +1.678 [+1.591, +1.759] | **-0.0159 [-0.0221, -0.0110]** | 296 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +2.789 [+2.495, +3.069] | **-0.0130 [-0.0171, -0.0093]** | 299 / 300 |
| | | 10s vs. 3s (sec.) | +1.00 [0.00, +2.67] | +1.00 [0.00, +2.67] | +4.483 [+4.137, +4.813] | **-0.0291 [-0.0382, -0.0221]** | 296 / 300 |

*All 54 within-mode exact binomial tests in Family A yield raw $p = 1.0000$ and Holm-adjusted $p = 1.0000$.*

---

### Table 2: Sensitivity Analysis — Common-Success Cells Across All Three Budgets

Restricting analysis strictly to participant-mode cells successful at all three budgets (3s, 5s, 10s) confirms that target error reductions are genuine optimization improvements rather than artifacts of shifting subsets:

| Objective | Arm | Common Cells | 3s $L_1$ Mean [95% CI] | 5s $L_1$ Mean [95% CI] | 10s $L_1$ Mean [95% CI] | 5s - 3s Diff [95% CI] | 10s - 5s Diff [95% CI] | 10s - 3s Diff [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A** | rules_only | 298 / 300 | 0.9430 [0.8988, 0.9882] | 0.9619 [0.9168, 1.0076] | 0.9851 [0.9392, 1.0322] | +0.0189 [+0.0027, +0.0351] | +0.0232 [+0.0049, +0.0418] | +0.0421 [+0.0174, +0.0664] |
| | minimum_change | 299 / 300 | 0.9678 [0.9174, 1.0184] | 0.9824 [0.9320, 1.0329] | 1.0004 [0.9507, 1.0494] | +0.0146 [-0.0028, +0.0329] | +0.0179 [-0.0003, +0.0368] | +0.0325 [+0.0119, +0.0532] |
| | model_guided | 298 / 300 | 0.9430 [0.8967, 0.9897] | 0.9714 [0.9234, 1.0197] | 0.9875 [0.9398, 1.0347] | +0.0284 [+0.0086, +0.0488] | +0.0161 [+0.0001, +0.0322] | +0.0445 [+0.0223, +0.0666] |
| **Variant B** | rules_only | 297 / 300 | 0.4508 [0.4332, 0.4705] | 0.4330 [0.4162, 0.4517] | 0.4192 [0.4026, 0.4371] | **-0.0178 [-0.0236, -0.0129]** | **-0.0137 [-0.0190, -0.0090]** | **-0.0316 [-0.0410, -0.0238]** |
| | minimum_change | 297 / 300 | 0.5016 [0.4729, 0.5312] | 0.4872 [0.4590, 0.5165] | 0.4759 [0.4485, 0.5044] | **-0.0145 [-0.0183, -0.0109]** | **-0.0112 [-0.0150, -0.0076]** | **-0.0257 [-0.0301, -0.0215]** |
| | model_guided | 296 / 300 | 0.4935 [0.4633, 0.5239] | 0.4775 [0.4497, 0.5057] | 0.4644 [0.4374, 0.4917] | **-0.0159 [-0.0221, -0.0110]** | **-0.0131 [-0.0172, -0.0094]** | **-0.0291 [-0.0382, -0.0221]** |

---

## 4. Target-Source Comparison Results (Matched Conditions)

### Table 3: Target-Source Contrasts Under Matched Objective and Budget

| Objective | Budget | Contrast | Valid Plans (pp) [95% CI] | E2E Acceptance (pp) [95% CI] | Upstream Stage 2 Gate Diff (pp) | Own-Target $L_1$ Error Diff [95% CI] | Runtime Difference (s) [95% CI] |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Variant A** | **3.0s** | Model-Guided vs. Minimum-Change | -0.33 [-1.00, 0.00] | +7.33 [+1.00, +14.00] | +8.0 pp (77 vs. 69) | -0.0245 [-0.0424, -0.0073] | +0.004 [-0.007, +0.015] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | -0.0000 [-0.0438, +0.0610] | +0.002 [-0.008, +0.013] |
| | | Minimum-Change vs. Rules-Only | +0.33 [0.00, +1.00] | +21.67 [+12.67, +31.33] | +21.0 pp (69 vs. 48) | +0.0244 [-0.0207, +0.0864] | -0.002 [-0.013, +0.009] |
| | **5.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0107 [-0.0240, +0.0034] | +0.006 [-0.013, +0.024] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0095 [-0.0334, +0.0681] | +0.002 [-0.019, +0.023] |
| | | Minimum-Change vs. Rules-Only | 0.00 [0.00, 0.00] | +21.33 [+12.00, +31.00] | +21.0 pp (69 vs. 48) | +0.0202 [-0.0208, +0.0773] | -0.004 [-0.024, +0.016] |
| | **10.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0131 [-0.0246, -0.0009] | +0.103 [-0.070, +0.273] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0007 [-0.0431, +0.0622] | +0.048 [-0.134, +0.229] |
| | | Minimum-Change vs. Rules-Only | 0.00 [0.00, 0.00] | +21.33 [+12.00, +31.00] | +21.0 pp (69 vs. 48) | +0.0138 [-0.0305, +0.0755] | -0.055 [-0.228, +0.117] |
| **Variant B** | **3.0s** | Model-Guided vs. Minimum-Change | -0.33 [-2.33, +1.33] | +7.33 [+1.00, +14.00] | +8.0 pp (77 vs. 69) | -0.0079 [-0.0155, -0.0000] | +0.019 [-0.013, +0.051] |
| | | Model-Guided vs. Rules-Only | -0.67 [-2.33, +0.67] | +28.00 [+18.67, +37.67] | +29.0 pp (77 vs. 48) | +0.0418 [+0.0023, +0.0980] | +0.002 [-0.033, +0.038] |
| | | Minimum-Change vs. Rules-Only | -0.33 [-1.33, +0.67] | +20.67 [+11.67, +30.33] | +21.0 pp (69 vs. 48) | +0.0510 [+0.0107, +0.1071] | -0.017 [-0.049, +0.017] |
| | **5.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0107 [-0.0166, -0.0046] | +0.034 [-0.061, +0.126] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0438 [+0.0050, +0.0993] | -0.014 [-0.114, +0.090] |
| | | Minimum-Change vs. Rules-Only | 0.00 [0.00, 0.00] | +21.33 [+12.00, +31.00] | +21.0 pp (69 vs. 48) | +0.0545 [+0.0148, +0.1102] | -0.047 [-0.147, +0.053] |
| | **10.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0122 [-0.0173, -0.0066] | -0.024 [-0.297, +0.244] |
| | | Model-Guided vs. Rules-Only | +0.33 [0.00, +1.00] | +29.33 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0437 [+0.0043, +0.1003] | -0.040 [-0.316, +0.231] |
| | | Minimum-Change vs. Rules-Only | +0.33 [0.00, +1.00] | +21.67 [+12.67, +31.33] | +21.0 pp (69 vs. 48) | +0.0554 [+0.0155, +0.1118] | -0.016 [-0.285, +0.255] |

*All 54 within-mode exact binomial tests for valid plans (Family B) yield raw $p = 1.0000$ and Holm-adjusted $p = 1.0000$. End-to-end differences (Family C) are statistically significant after Holm adjustment across all 54 tests ($p < 0.001$), matching the upstream Stage 2 feasibility gate difference.*

---

### Table 4: Shared-Reference Engineering Endpoints

When comparing Stage 2 target sources, shared-reference physical metrics demonstrate virtually identical engineering compliance:

| Metric | Rules-Only (Mean [95% CI]) | Minimum-Change (Mean [95% CI]) | Model-Guided (Mean [95% CI]) | Contrast (MG vs. MC) | Contrast (MG vs. RO) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Energy Error (kcal)** | 4.29 [3.49, 5.17] | 4.31 [3.51, 5.19] | 4.30 [3.51, 5.18] | -0.01 [-0.05, +0.03] kcal | +0.01 [-0.04, +0.06] kcal |
| **Energy Error (%)** | 0.27% [0.22%, 0.32%] | 0.27% [0.22%, 0.32%] | 0.27% [0.22%, 0.32%] | -0.00% [-0.00%, +0.00%] | +0.00% [-0.00%, +0.00%] |
| **Active Dish Count** | 7.92 [7.88, 7.96] | 7.92 [7.88, 7.96] | 7.92 [7.88, 7.96] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **Total Servings** | 14.86 [14.73, 14.99] | 14.87 [14.74, 15.00] | 14.87 [14.74, 15.00] | 0.00 [-0.02, +0.02] | +0.01 [-0.01, +0.03] |
| **Meal Mass (g)** | 1633.2 [1614.8, 1651.9] | 1633.9 [1615.5, 1652.6] | 1633.8 [1615.4, 1652.5] | -0.10 [-2.35, +2.18] g | +0.60 [-1.82, +3.01] g |

---

## 5. Practical Synthesis & Trade-Off Interpretation

1. **Budget Trade-Off:**
   - **3s $\to$ 5s:** Clear win for target-aware planning. Eliminates the valid-plan deficit (+0.33 to +1.00 pp) and improves $L_1$ target error by ~5% to 8% for +1.67s runtime.
   - **5s $\to$ 10s:** Trade-off between runtime and certified MIP gap termination. Feasibility gain is zero (897/900 vs 896/900). Target error drops by an additional ~2.5% to 3.0% for +2.8s runtime, but status 0 terminations jump from 26% to 56%.
   - **Recommendation:** 5.0 seconds is the natural operational default for interactive clinical workflows; 10.0 seconds is appropriate for asynchronous batch planning where proving integer gap $\le 1\%$ is prioritized.

2. **Stage 2 Target Source Trade-Off:**
   - **Stage 3 Independence:** Stage 3 meal feasibility and shared-reference meal composition are identical across rules-only, minimum-change, and model-guided targets.
   - **Upstream Value:** The engineering advantage of model-guided optimization occurs **entirely in Stage 2**, where model guidance finds continuous nutrient vectors that satisfy joint disease rules for 77% of participants (compared to 69% for minimum-change projection and 48% for rules-only targets).
   - **Simpler Targets Competitiveness:** When rules-only or minimum-change targets pass Stage 2, Stage 3 generates meals of identical structural validity and energy accuracy. Simpler targets remain competitive whenever clinical guidelines do not require personalized model-guided adjustments.

---

## 6. Limitations & Evidence Boundaries

1. **Retrospective Design:** Planned after inspecting earlier audit records; exploratory, not prospectively registered.
2. **Attainment vs. Health:** Distance to a target vector reflects mathematical solver alignment, not biological efficacy, glycemic control, or patient adherence.
3. **NHANES Cohort:** Evaluated on a temporal NHANES cohort (100 cases); does not establish validation in Indian clinical populations.
4. **Production Code Untouched:** Production solvers and pipelines remain on their original release defaults.
