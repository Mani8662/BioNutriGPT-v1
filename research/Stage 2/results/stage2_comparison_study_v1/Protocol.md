# Protocol: Stage 2 Three-Arm Comparative Added-Value Study

**Protocol Version:** `stage2_comparison_study_v1`  
**Date:** 23 November 2026  
**Status:** Frozen prior to execution on reserved cohort  
**Sample:** 100 reserved temporal NHANES adult profiles (Disjoint from 32-case development repair, 100-profile downstream study, and 5-case pilot)  
**Sample SHA-256:** `8be8ae24cc66269b57cf49d1f70e049cbb46f6703a219f97255e05c1aa85193c`  

## 1. Research Question
Does model-guided optimization add measurable engineering value beyond simpler ways of constructing nutrient targets?

A negative or inconclusive result is completely acceptable. The experiment is not tuned to favor BioNutriGPT.

## 2. Experimental Arms
All three arms evaluate under the exact same participant envelope, calorie prescriptions, nutrient bounds, ratio rules, and closed-loop score gate.

- **Arm A (Rules-only):** Deterministic, closed-form clinical target construction derived strictly from dietary guidelines, AMDRs, and DRI references. Contains zero gradient descent, zero prediction-assisted candidate search, and zero mathematical programming projection. Uses baseline model probabilities solely to define the common routing envelope.
- **Arm B (Minimum-change):** Scale-normalized convex quadratic projection of the participant's observed nutrient intake vector onto the joint feasible polytope. Minimizes deviation from observed intake without model-guided loss or risk gradients.
- **Arm C (Full model-guided):** Frozen deployed Stage 2 implementation: 50-step Adam search on the multi-objective counterfactual risk loss, followed by the identical scale-normalized convex quadratic projection onto the joint feasible polytope.

## 3. Endpoints & Statistical Analysis Plan
- **Primary Endpoint:** All-check acceptance rate (proportion of cases meeting all 39 bounds, energy balance, 3 ratios, Filter 2 invariants, and closed-loop score gate $\Delta p \le 0.0015$).
- **Primary Contrast:** Arm C vs. Arm B (isolates the value of model-score guidance).
- **Secondary Contrasts:** Arm C vs. Arm A and Arm B vs. Arm A.
- **Statistical Method:** Participant-level paired bootstrap (10,000 resamples, fixed random seed 20261123). Reports absolute differences in percentage points and paired 95% percentile confidence intervals.

## 4. Integrity and Safeguards
- All 100 attempted profiles remain in the denominator.
- Atomic per-case checkpointing: resume permitted only when inputs, protocol, and code hashes match.
- Solver failure and LP fallback are explicitly distinguished from proven infeasibility.
- Model score improvements evaluated with the optimization model are explicitly disclosed as self-referential.
- Stage 3 meal planning is explicitly outside the scope of this experiment.
