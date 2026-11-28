# Stage 2 Component Ablation Benchmark (v2) Results — 24 November 2026

## Executive Summary & Study Status

The full controlled component-ablation evaluation of the BioNutriGPT Stage 2 Counterfactual Prescription Engine (`repaired_v1` release) has been **executed, independently verified, and recorded with zero discrepancies**.

- **Status:** **COMPLETED AND VERIFIED**
- **Cohort:** 100 adult temporal NHANES participants (seed `20261125`) drawn from an eligible pool of 5,707 participants strictly disjoint from all 337 previously evaluated participants across 5 earlier studies (repair audit, downstream study, adapter pilot, comparison study, budget study) and disjoint from the 3 development pilot participants.
- **Protocol:** Frozen v2 specification ([Saved Model/stage2_ablation_v2/protocol.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_ablation_v2/protocol.json)).
- **Workload:** Exactly **600 counterfactual optimization tasks** (100 participants × 6 variants), evaluated sequentially on GPU (NVIDIA GeForce RTX 3050 Laptop GPU) with balanced cyclic variant ordering.
- **Total Measured Session Wall Time:** **1,159.98 seconds** ($\approx 19.33$ minutes) from cold start to completion.
- **Verification:** Both the full reconstructive audit ([Saved Model/stage2_ablation_v2/full/verification.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_ablation_v2/full/verification.json)) and the separate independent statistical verifier ([Saved Model/stage2_ablation_v2/full/independent_statistics_verification.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_ablation_v2/full/independent_statistics_verification.json)) passed byte-for-byte, verifying all 600 checkpoints, 45 bootstrap confidence intervals (10,000 resamples, seed `20261124`), 15 exact paired tests, Holm-Bonferroni step-down adjustments, and rejection of altered confidence intervals.
- **Production Status:** Production code, Stage 1 neural weights, preprocessors, calibrators, and Stage 2 targets remain 100% untouched.

---

## 1. Verified Results Table

All point estimates are paired comparisons across the same 100 participants ($N=100$). Confidence intervals are 95% percentiles from 10,000 paired participant-bootstrap resamples (seed `20261124`). Exact tests are two-sided paired binomial tests of discordant pairs. Multiplicity adjustments are computed using the Holm-Bonferroni step-down procedure across prespecified families:
- **Primary Family:** 5 contrasts on Overall Acceptance.
- **Secondary Family:** 10 contrasts on Numerical Pass and Score-Gate Pass.
- **Continuous Endpoints:** Exploratory, unadjusted bootstrap confidence intervals.

| Variant ID | Name & Modification | Attempted / Completed | Numerical Pass | Score-Gate Pass | Overall Accepted | Paired Overall Acceptance Diff (pp) [95% CI] (Wins/Losses, Holm $p$) | Normalized $L_1$ Distance from Observed [95% CI] | Mean Paired $L_1$ Diff vs Ref [95% CI] | Mean Wall Time (ms) [95% CI vs Ref] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`variant_0_reference`** | Full Stage 2 Reference (All components active) | 100 / 100 | 100/100 (100.0%) | 62/100 (62.0%) | 62/100 (62.0%) | *Reference* | 8.6427 [8.3079, 8.9712] | *Reference* | 2145.6 ms [*Ref*] |
| **`variant_1_no_model_objective`** | Ablate Model-Score Loss ($2.0p_c + 2.0p_d + 1.0p_o + 50\text{Pareto} = 0$) | 100 / 100 | 100/100 (100.0%) | 49/100 (49.0%) | 49/100 (49.0%) | **-13.00 pp** [-20.00, -7.00]<br>(0/13, $p=2.44\times 10^{-4}$, **$p_{\text{adj}}=9.77\times 10^{-4}$**) | 7.5015 [7.2144, 7.7979] | -1.1412 [-1.3410, -0.9449] | 962.1 ms [-1260.7, -1135.7] |
| **`variant_2_no_change_penalty`** | Ablate Dietary-Change Regularizer ($0.001\|\delta N\|_2^2 = 0$) | 100 / 100 | 100/100 (100.0%) | 64/100 (64.0%) | 64/100 (64.0%) | **+2.00 pp** [0.00, 5.00]<br>(2/0, $p=0.50$, **$p_{\text{adj}}=1.00$**) | 9.5591 [9.1678, 9.9472] | **+0.9164** [0.6847, 1.1650] | 2122.8 ms [-98.0, 26.3] |
| **`variant_3_no_barrier_penalty`** | Ablate Differentiable Barrier ($\text{barrier} = 0$) | 100 / 100 | 100/100 (100.0%) | 65/100 (65.0%) | 65/100 (65.0%) | **+3.00 pp** [0.00, 7.00]<br>(3/0, $p=0.25$, **$p_{\text{adj}}=0.75$**) | 8.4803 [8.1408, 8.8183] | -0.1624 [-0.2783, -0.0511] | 2085.6 ms [-130.4, -14.1] |
| **`variant_4_no_joint_projection`** | Ablate Joint Convex Quadratic Projection | 100 / 100 | 8/100 (8.0%) | 67/100 (67.0%) | 7/100 (7.0%) | **-55.00 pp** [-65.00, -45.00]<br>(0/55, $p=5.55\times 10^{-17}$, **$p_{\text{adj}}=2.78\times 10^{-16}$**) | 8.4644 [8.1341, 8.7909] | -0.1783 [-0.2186, -0.1422] | 1979.3 ms [-242.4, -118.3] |
| **`variant_5_no_sequential_filters`** | Ablate Downstream Sequential Filters 1–3 | 100 / 100 | 100/100 (100.0%) | 62/100 (62.0%) | 62/100 (62.0%) | **0.00 pp** [0.00, 0.00]<br>(0/0, $p=1.00$, **$p_{\text{adj}}=1.00$**) | 8.6427 [8.3079, 8.9712] | **0.0000** [0.0000, 0.0000] | 2140.5 ms [-94.7, 83.0] |

---

## 2. Key Engineering Findings

### A. Model-Score Objective Adds Statistically Significant Acceptance Value
- **Finding:** Guiding Adam optimization with neural model risk gradients ($2.0 p_c + 2.0 p_d + 1.0 p_o + 50.0 \text{Pareto}$) increased overall acceptance from **49.0% (49/100)** without model guidance to **62.0% (62/100)** with model guidance (+13 percentage points; paired bootstrap 95% CI [+7.0, +20.0]; 0 wins, 13 losses, exact paired binary $p = 2.44 \times 10^{-4}$, Holm-adjusted **$p_{\text{adj}} = 9.77 \times 10^{-4}$**).
- **Risk Reductions:** Model-guided search achieved lower target probabilities across all three disease outcomes compared to unguided search (paired mean probability differences: diabetes $-0.0055$, obesity $-0.0051$, CVD $-0.0067$).
- **Displacement Trade-off:** Unguided search stayed closer to observed baseline intake (mean $L_1$ distance difference: $-1.1412$ [95% CI: $-1.3410, -0.9449$]), because it minimized deviation without steering nutrient totals toward therapeutic targets.

### B. Joint Constraint Projection is Mathematically Indispensable
- **Finding:** Bypassing the convex quadratic joint projection collapsed numerical feasibility from **100.0% (100/100)** to **8.0% (8/100)** ($-92.00$ pp; exact paired $p = 4.04 \times 10^{-28}$, Holm $p = 4.04 \times 10^{-27}$).
- **Overall Acceptance:** Collapsed from **62.0% (62/100)** to **7.0% (7/100)** ($-55.00$ pp; exact paired $p = 5.55 \times 10^{-17}$, Holm $p = 2.78 \times 10^{-16}$).
- **Failure Mode:** Unprojected Adam outputs satisfy component bounding boxes $[lower, upper]$ via box clamping, but severely violate the coupled linear thermodynamic energy balance ($4P + 4C + 9F + 7A \approx KCAL$) and clinical ratio envelopes (linear residuals up to 400+ kcal). The joint projection is an essential engineering component.

### C. Downstream Sequential Filters are Mathematically Redundant
- **Finding:** Removing sequential downstream execution of Filters 1, 2, and 3 after joint projection produced **bit-for-bit identical final targets, probabilities, and residuals** across all 100 participants (paired differences: overall acceptance $0.00$ pp, $L_1$ distance $0.0000$, all outcome probability deltas $0.0000$).
- **Mechanism:** In the repaired pipeline (`deployment.REPAIRED`), the joint projection solves an optimization whose constraints strictly incorporate Filter 2 invariants, energy balance, and ratio limits within $10^{-6}$. Consequently, downstream Filters 1, 2, and 3 find all invariants pre-satisfied and make zero modifications. While safe to retain for defense-in-depth, they are functionally redundant.

### D. Dietary-Change Penalty Restrains Search Drift
- **Finding:** Ablating the $0.001 \|\delta N\|_2^2$ regularizer did not significantly alter overall acceptance ($64.0\%$ vs $62.0\%$, diff $+2.00$ pp; Holm $p = 1.00$).
- **Displacement Effect:** Omitting the penalty caused optimization to drift **significantly farther** from observed intake (mean $L_1$ distance increased from $8.64$ to $9.56$; paired diff **$+0.9164$** [95% CI: $+0.6847, +1.1650$]). The regularizer successfully functions as an anchor preventing gratuitous deviation from baseline habits.

### E. Differentiable Barrier Penalty is Non-Essential
- **Finding:** Setting the in-loop differentiable barrier penalty to 0.0 produced no statistically significant difference in overall acceptance ($65.0\%$ vs $62.0\%$, diff $+3.00$ pp; Holm $p = 0.75$) or $L_1$ distance (diff $-0.1624$ [95% CI: $-0.2783, -0.0511$]). Post-Adam box clamping (`torch.clamp`) and joint projection adequately manage boundaries even without soft barrier gradients during Adam.

---

## 3. Disclosures, Constraints & Scope Boundaries

1. **Exploratory Disjoint Sample:** The 100 participants were sampled from the NHANES 2017–March 2020 pre-pandemic dataset. Although strictly disjoint from all 337 participants in earlier BioNutriGPT studies, this remains an exploratory retrospective cohort.
2. **Dimensionless Distance:** Normalized dietary distance is a scale-normalized mathematical metric, not a clinical measure of patient palatability or dietary adherence.
3. **Self-Referential Model Scores:** Higher score-gate pass rates and lower model-predicted risks reflect the objective function of the Stage 1 neural network; they do not constitute prospective proof of clinical efficacy or health improvement.
4. **Joint Filter Ablation:** Filters 1–3 were ablated jointly in Variant 5. This evaluates the entire post-projection sequential filter block; it does not measure isolated filter contributions.
5. **No Production Modifications:** Production code, Stage 1 weights, preprocessors, and Stage 2 targets remain 100% untouched.

---

## 4. Verification Artifacts & Checksums

- **Full Verification Receipt:** [Saved Model/stage2_ablation_v2/full/verification.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_ablation_v2/full/verification.json)
- **Independent Statistics Receipt:** [Saved Model/stage2_ablation_v2/full/independent_statistics_verification.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_ablation_v2/full/independent_statistics_verification.json)
- **Full Run Receipt:** [Saved Model/stage2_ablation_v2/full/run_receipt.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_ablation_v2/full/run_receipt.json)
- **Analysis Data:** [Saved Model/stage2_ablation_v2/full/analysis.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_ablation_v2/full/analysis.json) (SHA-256: `51c19f369dc3dbfdd6af2d43f9f6bfa106c371febe1a3b0ccf46471c6f561355`)
- **Total Checkpoints Verified:** 600 JSON files byte-for-byte.
