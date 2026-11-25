# Stage 3 Solver-Budget Matched Comparisons Specification (v1)

**Date:** 24 November 2026  
**Status:** Frozen Immutable Analysis Specification  
**Directory:** `Saved Model/stage3_budget_comparisons_v1/`  
**Input Manifest:** `Saved Model/stage3_budget_comparisons_v1/input_manifest.json`  

---

## 1. Retrospective Provenance & Disclosure
- **Retrospective Analysis:** This analysis was designed post-hoc after inspecting the results of the Stage 3 solver-budget experiment and its reconstructive audit (`Saved Model/stage3_budget_audit_v3/`).
- **Cohort Reuse:** The cohort of 100 adult temporal NHANES participants (seed `20261124`, SHA-256: `fed1ab71134c26586a4b55af1ab64eb995c9912f4269e3ce0e91214c25f869bf`) is reused for additional matched endpoints.
- **Exploratory Nature:** These are exploratory matched engineering comparisons, not a prospectively registered confirmatory trial.

---

## 2. Statistical Estimator & Weighting Contract
To avoid cell-weighting confounding across dietary modes and participants:
1. **Participant-Mode Matching:**
   For any pair of matched conditions $(A, B)$ evaluated on participant $i$ and dietary mode $m$:
   - Binary endpoints: $\Delta_{i, m} = (Y_{A, i, m} - Y_{B, i, m}) \times 100$ (percentage points).
   - Continuous endpoints: $\Delta_{i, m} = X_{A, i, m} - X_{B, i, m}$.
2. **Within-Participant Averaging:**
   Average eligible matched modes within participant $i$:
   $$D_i = \frac{1}{|M_i|} \sum_{m \in M_i} \Delta_{i, m}$$
3. **Equal Participant Weighting:**
   The population point estimate is the unweighted mean over all eligible participants:
   $$\bar{D} = \frac{1}{N_{\text{part}}} \sum_{i \in \text{eligible}} D_i$$
4. **Participant-Clustered Bootstrap:**
   10,000 resamples of whole participant vectors $(D_i)$ with replacement (fixed seed `20261124`). Both point estimate and 95% bootstrap percentile CI $[q_{0.025}, q_{0.975}]$ share the exact same participant weighting.
5. **No Pooled McNemar Tests:**
   Pooled p-values across repeated modes within participants are omitted. Mode-specific exact two-sided binomial (McNemar) tests are reported with declared Holm step-down adjustments.

---

## 3. Comparison Factorial Matrix

### A. Budget Comparisons (6 Series)
Within each Stage 2 arm (`rules_only`, `minimum_change`, `model_guided`) and each Stage 3 objective (`variant_a` = bounds-based, `variant_b` = target-aware):
- **5s vs. 3s** (Primary)
- **10s vs. 5s** (Primary)
- **10s vs. 3s** (Secondary)

### B. Target-Source Comparisons (6 Conditions)
Within each Stage 3 objective (`variant_a`, `variant_b`) and each budget (3.0s, 5.0s, 10.0s):
- **Model-Guided vs. Minimum-Change**
- **Model-Guided vs. Rules-Only**
- **Minimum-Change vs. Rules-Only**

Retaining mode-level breakdowns across all three dietary modes (`pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`).

---

## 4. Multiplicity Families & Controls
- **Family A (Budget Comparisons):** 54 tests = 6 series $\times$ 3 mode-level tests $\times$ 3 comparisons (36 primary + 18 secondary). Adjusted via Holm step-down.
- **Family B (Target-Source Valid-Plan Comparisons):** 54 tests = 6 conditions $\times$ 3 contrasts $\times$ 3 mode-level tests. Adjusted via Holm step-down.
- **Family C (Target-Source End-to-End Comparisons):** 54 tests = 6 conditions $\times$ 3 contrasts $\times$ 3 mode-level tests. Adjusted via Holm step-down.

---

## 5. Endpoints & Critical Interpretation Rules
1. **Primary Endpoints:** Stage 3 valid-plan rate (`accepted`), end-to-end acceptance (`e2e`), actual measured runtime (`runtime`).
2. **Secondary Endpoints:** Normalized target error (`l1`), energy error ($|E - Target|$ in kcal and %), dish count, servings, mass, solver status breakdown.
3. **Critical Interpretation Contract for Target Sources:**
   When comparing Stage 2 target sources, each source constructs its own target vector. Lower distance to its own target does not demonstrate a superior target or clinical health benefit. All such comparisons are explicitly labeled **"Attainment relative to each source's own target"**. Shared-reference outcomes (validity, energy error, runtime, mass, servings, dishes) are evaluated separately.
4. **End-to-End Gate Separation:**
   End-to-end differences inherit the upstream Stage 2 feasibility gate. Stage 2 acceptance and conditional Stage 3 success are reported separately to prevent conflating meal generation capability with upstream mathematical constraint filters.
