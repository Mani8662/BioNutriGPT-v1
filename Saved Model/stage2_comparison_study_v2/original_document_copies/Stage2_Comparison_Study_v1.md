# Stage 2 Three-Arm Comparative Added-Value Study: Empirical Results and Verification

**Date:** 23 November 2026  
**Cohort:** 100 Reserved Complete-Input Temporal NHANES Adult Profiles (disjoint from 32-case development repair, 100-profile downstream study, and 5-case pilot)  
**Execution Environment:** CUDA (NVIDIA GeForce RTX 3050 Laptop GPU, PyTorch 12-thread CPU backend)  
**Protocol Version:** `stage2_comparison_study_v1` (Frozen prior to execution; SHA-256 bindings verified)  
**Verification Receipt:** `Saved Model/stage2_comparison_study_v1/verification.json` (`verification_passed: true`)  

---

## 1. Executive Summary & Answer to Research Question

> **Research Question:** Does model-guided optimization add measurable engineering value beyond simpler ways of constructing nutrient targets?

**Key Empirical Finding:**
- **On the Primary Endpoint (All-Check Acceptance Rate):** Model-guided optimization (Arm C) achieved an acceptance rate of **77.0%** (77/100), compared to **70.0%** (70/100) for minimum-change projection (Arm B) and **48.0%** (48/100) for deterministic clinical rules (Arm A).
- **Primary Contrast (Arm C vs. Arm B):** Model-guided optimization provided a statistically significant improvement of **+7.00 percentage points** (+7.0 pp; 95% paired bootstrap CI: **[+2.00, +13.00] pp**, SE = 2.91 pp, two-sided $p = 0.0154$).
- **Secondary Contrasts:**
  - Arm C vs. Arm A (Rules-only): **+29.00 pp** (95% CI: **[+19.00, +39.00] pp**, $p < 0.0001$).
  - Arm B vs. Arm A (Rules-only): **+22.00 pp** (95% CI: **[+11.00, +33.00] pp**, $p = 0.0002$).
- **The Engineering Trade-Off:**
  - **Computational Cost:** Arm C required a median runtime of **5.74 seconds/profile** (total: 581.99s on CUDA), whereas Arm B completed in **0.20 seconds/profile** (total: 20.44s) — a **28.5-fold computational overhead** for Arm C.
  - **Dietary Perturbation:** Arm B remained substantially closer to what participants currently eat (median normalized distance **1.8494**, IQR 1.3720) than Arm C (median **2.1992**, IQR 1.3024) or Arm A (median **4.8190**, IQR 3.7350).
- **Conclusion:** Model-guided search adds measurable, statistically significant value specifically for satisfying the closed-loop Stage 1 risk score gate (+7.0 pp over minimum-change projection). However, 70% of participants can be successfully steered to valid, risk-compliant targets using convex quadratic projection alone at a fraction of the computational budget.

---

## 2. Experimental Design & Arm Definitions

All three arms evaluate the exact same 100 participants under identical inputs, immutable non-dietary features, calorie prescriptions, nutrient intervals, thermodynamic energy coupling, clinical ratio limits, and final acceptance criteria.

```
+---------------------------------------------------------------------------------------------------+
|                                  SHARED PARTICIPANT ENVELOPE                                      |
|  - Observed 87-feature vector (canonicalized via canonicalize_dietary_ratios)                     |
|  - Mifflin-St Jeor TDEE with sedentary assumption (PAL = 1.20)                                     |
|  - Energy prescription: T_kcal in [0.95*T_kcal, 1.05*T_kcal] (deficit 500 kcal if obese)          |
|  - Component bounds: [L_i, U_i] tightened with Filter 2 invariants                                |
|  - Linear inequalities: Energy balance (+-0.99% margin) & 3 clinical ratio envelopes              |
+---------------------------------------------------------------------------------------------------+
                                                  |
           +--------------------------------------+--------------------------------------+
           |                                      |                                      |
           v                                      v                                      v
  [Arm A: Rules-Only]                   [Arm B: Minimum-Change]                [Arm C: Model-Guided]
  Pure closed-form clinical             Convex quadratic projection of         50-step Adam gradient search
  rules based on AMDR & DRI             observed intake n_obs onto P:          on multi-objective loss:
  guidelines. Zero gradient,            min 0.5 * sum ((n_i - n_obs_i)/s_i)^2  min 2*p_cvd + 2*p_diab + 1*p_obes
  zero projection.                      Zero model-guided search.              followed by identical QP projection.
           |                                      |                                      |
           +--------------------------------------+--------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                    SHARED UNIFORM EVALUATION                                      |
|  1. Independent Numerical Checks: Bounds [L-1e-6, U+1e-6], Energy error <= 1%, Ratios, Filter 2    |
|  2. Frozen-Model Rescoring: Calibrated Stage 1 inference -> Delta p = p_cand - p_0                |
|  3. Closed-Loop Gate: Delta p <= +0.0015 for Diabetes, Obesity, and CVD                           |
|  4. All-Check Acceptance: Joint Numerical Pass AND Score Gate Pass                                |
+---------------------------------------------------------------------------------------------------+
```

### Exact Arm Formulations

1. **Arm A — Deterministic Rules-Only Nutrient Targets:**
   - Closed-form target setting without optimization.
   - Energy: $T_{\text{kcal}}$.
   - Protein: $1.0\text{ g/kg}$, clipped to $[L_{\text{PROT}}, U_{\text{PROT}}]$.
   - Total Fat: $25\%$ of energy ($30\%$ under diabetes routing).
   - Carbohydrate: Derived by energy conservation: $n_{\text{CARB}} = (T_{\text{kcal}} - 4\,n_{\text{PROT}} - 9\,n_{\text{TFAT}}) / 4$.
   - Saturated Fat: $0.05 \times T_{\text{kcal}} / 9$ (CVD) or $0.07 \times T_{\text{kcal}} / 9$ (general), capped by ratio $\frac{a_{\text{su}}}{1 + a_{\text{su}}} (n_{\text{TFAT}} + 0.1) \times 0.99$.
   - Polyunsaturated Fat: $0.07 \times T_{\text{kcal}} / 9$; Monounsaturated Fat: $n_{\text{TFAT}} - n_{\text{SFAT}} - n_{\text{PFAT}}$.
   - Fiber: Guideline floor $\max(14\text{ g}/1000\text{ kcal}, 28/35\text{ g})$; Sugar: $\min(10\%\text{ kcal}, a_{\text{sf}} (n_{\text{FIBE}} + 1) \times 0.99)$.
   - Sodium: $1500/2000\text{ mg}$ capped by ratio $a_{\text{nk}} (n_{\text{POTA}} + 1) \times 0.99$; Potassium: $3500/2600\text{ mg}$.
   - Non-essential carotenoids, caffeine, alcohol, synthetic folic acid: $0.0$.
   - All other 17 micronutrients: Age/sex-specific DRI RDA/AI lower bound.
   - *Disclosure:* Baseline model probabilities are used solely to establish the common routing envelope (diabetes/obesity/CVD routing), as required by the shared experimental design.

2. **Arm B — Minimum-Change Joint Constraint Projection:**
   - Projects the participant's observed intake $\mathbf{n}_{\text{obs}}$ directly onto the joint feasible polytope $\mathcal{P}$:
     $$\min_{\mathbf{n} \in \mathcal{P}} \frac{1}{2} \sum_{i=0}^{38} \left( \frac{n_i - n_{\text{obs}, i}}{s_i} \right)^2$$
     where $s_i$ are fixed engineering scales (`DAILY_REFERENCE_SCALES`).
   - Solved via SLSQP warm-started with a HiGHS linear programming witness.
   - Contains zero model risk guidance.

3. **Arm C — Full Model-Guided Optimization + Joint Projection:**
   - 50-step Adam optimizer ($\text{lr} = 0.05$) minimizing:
     $$\mathcal{L} = 2.0\,\hat{p}_{\text{cvd}} + 2.0\,\hat{p}_{\text{diab}} + 1.0\,\hat{p}_{\text{obes}} + 50.0 \sum_k \text{ReLU}(\hat{p}_k - \hat{p}_{k, 0}) + 0.001\,\|\Delta \mathbf{N}\|_2^2 + \mathcal{B}(\mathbf{n}, \mathbf{l}, \mathbf{u})$$
   - Followed by the identical scale-normalized convex quadratic projection onto $\mathcal{P}$.

---

## 3. Verified Results

### Table 1: Main Performance Across Arms ($N = 100$ per arm)

| Metric | Arm A: Rules-Only | Arm B: Minimum-Change | Arm C: Model-Guided |
| :--- | :---: | :---: | :---: |
| **Attempted Profiles** | 100 | 100 | 100 |
| **Completed Targets** | 100 | 100 | 100 |
| **Execution Errors** | 0 | 0 | 0 |
| **Numerical Feasibility (%)** | 100 / 100 (100.0%) | 99 / 100 (99.0%) | **100 / 100 (100.0%)** |
| **Score Gate Pass Rate (%)** | 48 / 100 (48.0%) | 70 / 100 (70.0%) | **77 / 100 (77.0%)** |
| **All-Check Acceptance Rate (%)** | 48 / 100 (48.0%) | 70 / 100 (70.0%) | **77 / 100 (77.0%)** |
| **LP Solver Fallbacks** | 0 (N/A) | 0 (0.0%) | 0 (0.0%) |
| **Distance from Observed (Median)** | 4.8190 | **1.8494** | 2.1992 |
| **Distance from Observed (IQR)** | 3.7350 | **1.3720** | 1.3024 |
| **Distance from Observed (Range)** | [2.2751, 20.4594] | **[0.7599, 17.3659]** | [0.8546, 17.3906] |
| **Runtime per Profile (Median)** | **0.031 s** | 0.199 s | 5.735 s |
| **Total Arm Runtime** | **3.70 s** | 20.44 s | 581.99 s |

*Note on Case 49 in Arm B:* The single numerical failure in Arm B was a floating-point boundary artifact where protein landed on $0.7999999999999999\text{ g/kg}$, triggering Filter 2's strict $< 0.8\text{ g/kg}$ check. Arm C moved protein slightly away from the lower boundary ($0.85\text{ g/kg}$) and achieved 100% numerical feasibility.

---

### Table 2: Paired Bootstrap Contrasts (10,000 Resamples, Seed `20261123`)

| Contrast | Endpoint | Point Diff (pp) | 95% Bootstrap CI | SE (pp) | Two-Sided $p$-value | Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary: Arm C vs Arm B** | **All-Check Acceptance** | **+7.00** | **[+2.00, +13.00]** | **2.91** | **0.0154** | **Significant ($p < 0.05$)** |
| | Numerical Feasibility | +1.00 | [0.00, +3.00] | 1.00 | 0.7334 | Not significant |
| | Score-Gate Pass | +7.00 | [+2.00, +13.00] | 2.91 | 0.0154 | Significant |
| **Secondary: Arm C vs Arm A** | **All-Check Acceptance** | **+29.00** | **[+19.00, +39.00]** | **5.31** | **< 0.0001** | **Significant ($p < 0.001$)** |
| | Numerical Feasibility | +0.00 | [0.00, 0.00] | 0.00 | 1.0000 | Equivalence |
| | Score-Gate Pass | +29.00 | [+19.00, +39.00] | 5.31 | < 0.0001 | Significant |
| **Secondary: Arm B vs Arm A** | **All-Check Acceptance** | **+22.00** | **[+11.00, +33.00]** | **5.38** | **0.0002** | **Significant ($p < 0.001$)** |
| | Numerical Feasibility | -1.00 | [-3.00, 0.00] | 1.00 | 0.7334 | Not significant |
| | Score-Gate Pass | +22.00 | [+11.00, +33.00] | 5.38 | 0.0002 | Significant |

---

### Table 3: Major Dietary Nutrient Shifts Across Arms ($N = 100$)

| Nutrient | Observed Mean | Arm A Target Mean | Arm A $\Delta$ (Median) | Arm B Target Mean | Arm B $\Delta$ (Median) | Arm C Target Mean | Arm C $\Delta$ (Median) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Energy (kcal)** | 2,212.1 | 1,661.7 | -324.0 | 1,687.7 | -244.8 | 1,685.0 | -244.8 |
| **Protein (g)** | 80.5 | 84.9 | +9.6 | 84.3 | +0.5 | 85.6 | +0.8 |
| **Carbohydrate (g)** | 255.6 | 225.4 | -9.0 | 201.8 | -19.0 | 204.0 | -16.6 |
| **Total Fat (g)** | 91.3 | 46.7 | -31.6 | 58.9 | -15.3 | 57.1 | -14.5 |
| **Sodium (mg)** | 3,465.4 | 1,990.0 | -1,245.5 | 2,120.0 | -948.5 | 2,120.0 | -948.5 |
| **Potassium (mg)** | 2,625.2 | 2,618.0 | +116.5 | 3,031.4 | +116.5 | 3,031.3 | +116.5 |

*Nutrient Shift Insights:*
- Both Arm B and Arm C produce remarkably congruent macronutrient shifts: reducing calories by ~525 kcal, moderating fat by ~33 g, curtailing sodium by ~1,345 mg, and elevating potassium by ~406 mg.
- Arm A imposes substantially larger reductions on total fat (-44.6 g vs -32.4 g in Arm B) and sodium (-1,475 mg), forcing participants into rigid generic targets that disregard baseline eating habits.

---

## 4. Independent Verification Outcome

The independent replay script (`verify_stage2_comparison_study_v1.py`) evaluated all 300 saved checkpoint files without trusting precomputed booleans or differences:
1. **Source Integrity:** All 14 source files and `cases_LOCAL.json` matched their exact frozen SHA-256 digests.
2. **Cohort Disjointness:** Confirmed 0 overlap with the 32 repair cases, 100 downstream study participants, and 5 pilot participants.
3. **Replay Residuals:** Recomputed all 39 bounds, thermodynamic energy balance ($\le 0.01 + 10^{-9}$), and 3 clinical ratio limits.
4. **Fresh Neural Forward Passes:** Loaded frozen checkpoint `model.pt` and calibrators. Evaluated baseline and target probabilities. Maximum absolute difference between fresh neural replay and saved probabilities was **0.00e+00**.
5. **Exact Summary Parity:** All 100 attempted cases per arm were verified; recomputed acceptance counts and rates matched `summary.json` exactly.
6. **Bootstrap Replay:** Re-ran 10,000 bootstrap resamples with seed `20261123`; confirmed point differences and confidence intervals match Table 2 with zero discrepancy.
7. **Verification Receipt:** Saved to `Saved Model/stage2_comparison_study_v1/verification.json`.

---

## 5. Methodological Limitations & Explicit Boundaries

1. **Self-Referential Score Gate:** Evaluation of cardiometabolic risk scores uses the exact model that guided Arm C. While Arm C achieves a +7.0 pp advantage over Arm B on this metric, this is partly an artifact of optimizing for the evaluation criterion. It does not establish clinical safety, disease prevention, or therapeutic efficacy in living human patients.
2. **Fixed Engineering Scales:** Distance calculations rely on `DAILY_REFERENCE_SCALES`. Although grounded in US RDA/AI values for core nutrients, non-essential phytochemicals, caffeine, and fatty acid fractions use engineering choices.
3. **Sedentary Assumption:** All participants are evaluated under a standardized sedentary physical activity level ($1.20 \times \text{BMR}$).
4. **Stage 3 Independence:** Feasibility in nutrient space $\mathcal{P}$ does not guarantee that Stage 3's MILP integer solver can identify a discrete, palatable food basket satisfying all constraints.
5. **Not Indian Clinical Data:** Evaluated on complete-input US NHANES temporal profiles. Does not constitute clinical validation in Indian cohorts.
