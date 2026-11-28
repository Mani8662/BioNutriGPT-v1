# Stage 2 Component Ablation Protocol (v1)

**Date:** 24 November 2026  
**Status:** Protocol Frozen; Pilot Prepared; Full Study Unexecuted  
**Location:** `Saved Model/stage2_ablation_v1/`

---

## 1. Purpose and Research Questions

The purpose of this study is to perform a controlled, one-at-a-time component ablation of the BioNutriGPT Stage 2 Counterfactual Prescription Engine under the repaired release (`repaired_v1`).

### Primary Research Questions:
1. **Model-Score Objective:** Does guiding Adam optimization with Stage 1 neural risk gradients produce higher model-score acceptance rates and lower target risk compared to an optimization guided purely by distance and bounds?
2. **Dietary-Change Penalty:** Does the $0.001 \|\delta N\|_2^2$ regularizer in Adam prevent unnecessary deviation from baseline intake without harming risk reduction?
3. **Barrier Penalty:** Does the differentiable quadratic barrier penalty in Adam add value over relying purely on post-Adam box clamping (`torch.clamp`) and joint projection?
4. **Joint Constraint Projection:** What is the exact numerical feasibility failure rate when joint constraint projection is ablated?
5. **Sequential Filters (Invariance Test):** Are downstream sequential Filters 1–3 mathematically redundant when preceded by the joint constraint projection?

---

## 2. Experimental Design & Ablation Variants

All variants operate on the exact same patient profiles, baseline probabilities, caloric prescriptions, component envelopes, random seeds, and external evaluation standards.

| Variant ID | Name | What Changes | What Remains Fixed | Expected Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **`variant_0_reference`** | Full Stage 2 Reference | None (Full current deployed pipeline). | Full pipeline. | Serves as the matched reference arm. |
| **`variant_1_no_model_objective`** | Ablate Model-Score Objective | Sets model risk terms ($2.0 p_c + 2.0 p_d + 1.0 p_o$) and Pareto penalty ($50.0 \sum \text{ReLU}$) to $0.0$ in Adam loss. Loss is $0.001 \|\delta N\|_2^2 + \text{barrier}$. | Adam search (50 steps), barrier, change penalty, clamping, joint projection, score gate, validator. | Isolates the engineering value of neural model risk gradients. |
| **`variant_2_no_change_penalty`** | Ablate Dietary-Change Penalty | Sets the coefficient of $0.001 \|\delta N\|_2^2$ to $0.0$ in Adam loss. | Model loss, Pareto penalty, barrier, Adam steps, clamping, joint projection, score gate, validator. | Tests whether unconstrained model optimization drifts farther from baseline intake without improving score gate passage. |
| **`variant_3_no_barrier_penalty`** | Ablate Differentiable Barrier | Sets $\text{barrier}(N, lower, upper) = 0.0$ in Adam loss. | Model loss, Pareto penalty, change penalty, Adam steps, clamping, joint projection, score gate, validator. | Tests whether smooth soft-envelope guidance during search prevents trajectory saturation against boundary clamps. |
| **`variant_4_no_joint_projection`** | Ablate Joint Projection | Bypasses Step 4.5 convex quadratic joint projection. `n_adam_clipped` is passed directly to downstream filters and validator. | Full Adam optimization (all terms), post-Adam clamping, downstream filter audits, score gate, validator. | Directly quantifies the necessity of joint projection for thermodynamic energy and ratio feasibility. |
| **`variant_5_no_sequential_filters`** | Ablate Downstream Sequential Filters | Bypasses sequential execution of Filters 1, 2, and 3 after joint projection. Vector goes directly from projection to rescoring. | Full Adam optimization, clamping, joint projection, score gate, validator. | Verifies the code-path finding that Filters 1–3 are mathematically redundant when preceded by joint projection. |

---

## 3. Cohort Sampling & Disjointness Enforcement

To eliminate data leakage, cohorts are sampled from the complete temporal NHANES clean dataset ($N=7,004$; eligible $N=6,033$) while strictly excluding all 337 participants evaluated across any previous study:
- 32-profile development repair cohort (`Saved Model/stage2_constraint_audit_repaired_v1`)
- 100-profile downstream translation cohort (`Saved Model/stage1_downstream_study_v1`)
- 5-profile adapter pilot cohort (`Saved Model/stage2_research_adapter_pilot_v1`)
- 100-profile comparison study cohort (`Saved Model/stage2_comparison_study_v2`)
- 100-profile budget experiment cohort (`Saved Model/stage3_budget_experiment_v1`)

### Cohorts:
1. **Pilot Development Sample ($N=3$):**
   - Seed: `20261124`
   - SEQNs: `[116727, 117285, 118239]`
   - Frozen in `development_participants_LOCAL.json`.
   - Used exclusively for engineering verification and pilot execution.
2. **Full Evaluation Sample ($N=100$):**
   - Seed: `20261125`
   - Sampled from the remaining 5,704 eligible participants disjoint from both prior studies and the pilot cohort.
   - Frozen in `evaluation_participants_LOCAL.json`.
   - **NOT executed** in this task.

---

## 4. Endpoints & Common Evaluation

Every output is evaluated by the **same external reconstructive validator** (`stage2_comparison_checks_v2.check`) and the **same closed-loop model-score acceptance gate**:
1. **Numerical Pass Rate:** $100\%$ compliance with all 39 component bounds, thermodynamic energy balance ($\pm 1.00\%$), Filter 2 invariants, and clinical ratio envelopes under a documented $10^{-6}$ tolerance.
2. **Score-Gate Pass Rate:** $\Delta p_d \le 0.0015$, $\Delta p_o \le 0.0015$, $\Delta p_c \le 0.0015$.
3. **Overall Accepted Rate:** Joint satisfaction of numerical validity AND score-gate passage.
4. **Normalized Distance from Observed Intake:** Scale-normalized $L_2$ distance using `DAILY_REFERENCE_SCALES`.
5. **Runtime (ms):** Wall-clock duration of Adam search, projection, and total pipeline.
6. **Optimizer Failures / Fallback Frequency:** Rate of SLSQP non-convergence or HiGHS LP fallback.

---

## 5. Statistical Pre-Specification

- **Unit of Inference:** Participant ($N=100$ in full study; $N=3$ in pilot).
- **Contrasts:** 5 paired contrasts of each ablation variant against `variant_0_reference`.
- **Continuous Endpoints:** Mean paired differences with 95% bootstrap confidence intervals (10,000 resamples, seed `20261124`).
- **Binary Endpoints:** Exact two-sided paired binomial tests (discordant pairs).
- **Multiplicity Family:** The 5 primary contrasts form a prespecified family adjusted using the Holm-Bonferroni step-down procedure.
- **Pilot Scope:** Pilot statistics are strictly mechanical pipeline sanity checks and do not constitute population evidence.

---

## 6. Verification and Negative Control Safeguards

All artifacts and checkpoints are verified independently:
- Full-method bit-for-bit parity test with reference implementation.
- Detection of deliberate bound, energy, and ratio corruptions.
- Negative corruption fixtures using the standard rejection pattern:
  ```python
  try:
      validator(corrupted)
  except ExpectedValidationError:
      record_expected_rejection()
  else:
      raise AssertionError("Corrupted artifact was accepted")
  ```
- Active rejection of altered statistical estimates and confidence intervals.
