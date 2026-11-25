# Stage 2 Joint Constraint Repair and 32-Profile Verification

**Date:** 22 November 2026  
**Status:** Verification Passed  
**Artifact Directory:** `Saved Model/stage2_joint_constraint_repair_v1/`  
**Baseline Comparator:** `Saved Model/stage2_constraint_audit_repaired_v1/`  

---

## 1. Executive Summary

A previous observational audit of BioNutriGPT's Stage 2 counterfactual prescription engine (`Saved Model/stage2_constraint_audit_repaired_v1/`) revealed a systemic failure mode: the 50-step Adam optimizer clipped nutrients component-wise into separate intervals without coupling macronutrients to total caloric budget. Subsequently, sequential Filter 1 recalculated energy as $E_{\text{calc}} = 4\text{PROT} + 4\text{CARB} + 9\text{TFAT} + 7\text{ALCO}$ and overwrote calories ($n_{\text{KCAL}} = E_{\text{calc}}$) whenever relative disagreement exceeded 1%. In 28 out of 32 frozen audit profiles, this unconstrained overwrite displaced the calorie target outside the prescribed interval (by 50.84 to 871.55 kcal), causing downstream Filter 3 and final certification checks to fail. Only 3 of 32 cases were accepted.

To eliminate this defect without relaxing clinical boundaries or widening acceptance thresholds, we implemented a mathematically exact **scale-normalized convex quadratic projection** onto the joint feasible polytope $\mathcal{P} \subset \mathbb{R}^{39}$. The projection simultaneously enforces:
1. All 39 nutrient component lower and upper bounds;
2. Filter 2 clinical invariants (minimum energy $\ge 1200\text{ kcal}$, protein $0.8W \le n_{\text{PROT}} \le 2.0W$, sodium ceiling $\le 1500/2300\text{ mg}$, potassium floor $\ge 3500/2600\text{ mg}$);
3. Linear thermodynamic energy consistency ($|E_{\text{calc}} - n_{\text{KCAL}}| \le 0.01 \cdot n_{\text{KCAL}}$);
4. Three clinical ratio envelopes (sugar-to-fiber, sodium-to-potassium, saturated-to-unsaturated fat).

Regression evaluation across the frozen 32 NHANES audit profiles and independent replay verification confirmed:
- **Calorie envelope violations:** Reduced from 28 to **0** (100% elimination).
- **Nutrient bound violations:** Reduced from 28 to **0** across all 39 nutrients.
- **Energy arithmetic inconsistencies:** Maintained at **0**.
- **Ratio violations:** Maintained at **0**.
- **Stage 2 Certification:** Increased from **3 / 32 (9.4%)** to **25 / 32 (78.1%)** (+22 newly accepted cases, +733% relative gain).
- **False accepts:** **0** (independently verified; every accepted case strictly satisfies all 39 bounds, energy consistency, ratio envelopes, and the closed-loop score gate).
- **Preservation of baseline accepted cases:** All 3 originally accepted cases (Cases 07, 21, 27) remain certified.
- **Remaining uncertified cases:** Exactly **7** cases (Cases 01, 08, 09, 10, 12, 24, 31) are rejected solely because they fail the closed-loop Stage 1 model-score gate ($\Delta p \le 0.0015$). In accordance with project safeguards, no score gates were widened and no constraints were relaxed to force acceptance.

---

## 2. Mathematical Formulation of Joint Constraint Projection

### 2.1 The Joint Feasible Polytope $\mathcal{P} \subset \mathbb{R}^{39}$

Let $n \in \mathbb{R}^{39}$ represent the 39 dietary nutrients defined in `DIET_RAW_COLS`. The joint feasible space is defined as the convex intersection of box constraints and linear inequality constraints:

$$\mathcal{P} = \left\{ n \in \mathbb{R}^{39} \;\middle|\; l^{\text{tight}} \le n \le u^{\text{tight}}, \; A_{\text{ub}} n \le b_{\text{ub}} \right\}$$

#### 1. Tightened Box Bounds ($l^{\text{tight}}, u^{\text{tight}}$)
For each nutrient $i \in \{0, \dots, 38\}$, component bounds $[l_i, u_i]$ from `_build_nutrient_bounds` are tightened by incorporating Filter 2 clinical invariants:
$$l^{\text{tight}}_{\text{KCAL}} = \max(l_{\text{KCAL}}, 1200.0)$$
$$l^{\text{tight}}_{\text{PROT}} = \max(l_{\text{PROT}}, 0.8 \cdot W), \quad u^{\text{tight}}_{\text{PROT}} = \min(u_{\text{PROT}}, 2.0 \cdot W)$$
$$u^{\text{tight}}_{\text{SODI}} = \min\left(u_{\text{SODI}}, \begin{cases} 1500.0 & \text{if } p_{\text{CVD}, 0} \ge 0.603 \\ 2300.0 & \text{otherwise} \end{cases}\right)$$
$$l^{\text{tight}}_{\text{POTA}} = \max\left(l_{\text{POTA}}, \begin{cases} 3500.0 & \text{if } p_{\text{CVD}, 0} \ge 0.603 \\ 2600.0 & \text{otherwise} \end{cases}\right)$$
For all other 35 nutrients $i$, $l^{\text{tight}}_i = l_i$ and $u^{\text{tight}}_i = u_i$. If $l^{\text{tight}}_i > u^{\text{tight}}_i$ for any $i$, $\mathcal{P} = \emptyset$ and the system raises `Stage2InfeasibleConstraintsError`.

#### 2. Linear Inequality Constraints ($A_{\text{ub}} n \le b_{\text{ub}}$)
The 5 linear inequalities comprise thermodynamic energy consistency and three clinical ratios:

1. **Thermodynamic Energy Balance:**
   $$4 n_{\text{PROT}} + 4 n_{\text{CARB}} + 9 n_{\text{TFAT}} + 7 n_{\text{ALCO}} - 1.0099 n_{\text{KCAL}} \le 0$$
   $$-4 n_{\text{PROT}} - 4 n_{\text{CARB}} - 9 n_{\text{TFAT}} - 7 n_{\text{ALCO}} + 0.9901 n_{\text{KCAL}} \le 0$$
   *(Note: The energy margin is tightened to $\pm 0.99\%$ to ensure that floating-point boundary rounding strictly satisfies the $1.00\%$ acceptance criterion with zero margin for numerical overshoot.)*

2. **Sugar-to-Fiber Ratio Envelope:**
   $$a_{\text{sf}} = \max(\text{baseline\_sugar\_to\_fiber}, 1.0)$$
   $$n_{\text{SUGR}} - a_{\text{sf}} n_{\text{FIBE}} \le a_{\text{sf}}$$

3. **Sodium-to-Potassium Ratio Envelope:**
   $$a_{\text{nk}} = \max(\text{baseline\_sodium\_potassium}, 1.0)$$
   $$n_{\text{SODI}} - a_{\text{nk}} n_{\text{POTA}} \le a_{\text{nk}}$$

4. **Saturated-to-Unsaturated Fat Ratio Envelope:**
   $$a_{\text{su}} = \max(\text{baseline\_sat\_to\_unsat}, 0.40)$$
   $$n_{\text{SFAT}} - a_{\text{su}} n_{\text{MFAT}} - a_{\text{su}} n_{\text{PFAT}} \le 0.1 \cdot a_{\text{su}}$$

### 2.2 Objective Function and Scale Normalization

Let $\tilde{n} \in \mathbb{R}^{39}$ denote the candidate nutrient target produced by the 50-step Adam optimizer. The projection seeks the point $n^* \in \mathcal{P}$ minimizing weighted Euclidean distance:

$$\min_{n \in \mathcal{P}} \frac{1}{2} \sum_{i=0}^{38} w_i (n_i - \tilde{n}_i)^2 = \min_{n \in \mathcal{P}} \frac{1}{2} \sum_{i=0}^{38} \left( \frac{n_i - \tilde{n}_i}{s_i} \right)^2$$

where $w_i = 1 / s_i^2$ and $s_i > 0$ represents the characteristic physiological intake scale for nutrient $i$.

**Justification for Scale Normalization:**
Nutrient units span multiple orders of magnitude (from micrograms for Vitamin B12 and Vitamin K, to milligrams for Sodium and Potassium, to grams for Macronutrients, to thousands for Kilocalories). Unweighted Euclidean distance would heavily penalize adjustments to KCAL and SODI while permitting massive unconstrained swings in micronutrients. 

Conversely, normalizing by upper search bounds is mathematically unacceptable: nutrients with `NUMERICAL_SEARCH_CAP = 1,000,000.0` would receive weights $w_i = 10^{-12}$, effectively removing them from the objective.

To provide a defensible, clinically anchored metric, characteristic scales $s_i$ are defined directly from the official **US FDA Reference Daily Intakes (RDIs) / Daily Values (DVs)** for general adults (2,000 kcal diet reference):
- Energy & Macros: KCAL = 2,000 kcal; PROT = 50 g; CARB = 275 g; TFAT = 78 g; FIBE = 28 g; SUGR = 50 g; SFAT = 20 g; MFAT = 30 g; PFAT = 20 g; ALCO = 14 g.
- Minerals: SODI = 2,300 mg; POTA = 4,700 mg; CALC = 1,300 mg; PHOS = 1,250 mg; MAGN = 420 mg; IRON = 18 mg; ZINC = 11 mg; COPP = 0.9 mg; SELE = 55 $\mu$g.
- Vitamins: VB1 = 1.2 mg; VB2 = 1.3 mg; NIAC = 16 mg; VB6 = 1.7 mg; FOLA/FA/DFE = 400 $\mu$g; VB12 = 2.4 $\mu$g; VC = 90 mg; VK = 120 $\mu$g; ATOC = 15 mg; RET/VARA = 900 $\mu$g.

Under this weighting, a 1-unit deviation in normalized coordinate space represents exactly 100% of the recommended daily intake, guaranteeing unit-invariant physiological trade-offs.

### 2.3 Optimization Procedure and Surrogate Forward Scope

1. **Feasibility Gate:** HiGHS LP solves $\min_{n \in \mathcal{P}} c^T n$ with $c = W (\text{clip}(\tilde{n}, l^{\text{tight}}, u^{\text{tight}}) - \tilde{n})$. If the LP is infeasible, the procedure terminates immediately with `Stage2InfeasibleConstraintsError`.
2. **Warm-Started SLSQP Projection:** Using the LP solution $x_{\text{LP}} \in \mathcal{P}$ as a strictly feasible warm start, Sequential Least Squares Programming (SLSQP; `ftol=1e-12`, `maxiter=500`) computes $n^* = \arg\min_{n \in \mathcal{P}} \frac{1}{2} \| W^{1/2} (n - \tilde{n}) \|_2^2$.
3. **Boundary Precision Fallback:** If boundary rounding in SLSQP ever produces a linear residual $> 10^{-6}$, the solver automatically falls back to $x_{\text{LP}}$.
4. **Surrogate Forward Scope:** Gradient backpropagation terminates at the 50-step Adam output $\tilde{n}$. The projection $n^*$ is a detached, non-differentiable forward mapping. Derived dietary features are canonicalized from $n^*$, non-dietary features are verified immutable, and $n^*$ is rescored through the frozen calibrated Stage 1 model to evaluate Filter 4 ($\Delta p \le 0.0015$).

---

## 3. Paired Comparison Results (Baseline vs. Repaired)

The 32 frozen profiles (sampled with seed `20261122` from complete repaired NHANES temporal participants) were evaluated under identical hardware (RTX 3050 GPU, 12 CPU threads), identical thresholds (0.5), and identical Stage 1 model checkpoints.

### Table 1: Comprehensive Paired Regression Audit (N = 32)

| Evaluation Check / Stage | Baseline Audit (`v1`) | Repaired Release (`joint_v1`) | Net Change | Replay Verification |
|---|---:|---:|---:|:---:|
| **Attempted Profiles** | 32 | 32 | 0 | Exact match |
| **Returned Dossiers** | 32 | 32 | 0 | Exact match |
| **Execution Errors** | 0 | 0 | 0 | None |
| **Bound Failures: Post-Optimizer Clipping** | 0 | 0 | 0 | Confirmed |
| **Bound Failures: Post-Projection** | *N/A* | **0** | **0** | Confirmed |
| **Bound Failures: Post-Filter 1** | 28 | **0** | **-28** | Confirmed |
| **Bound Failures: Post-Filter 2** | 28 | **0** | **-28** | Confirmed |
| **Bound Failures: Post-Filter 3 (Final Target)** | 28 | **0** | **-28** | Confirmed |
| **Final Calorie Envelope Violations** | 28 | **0** | **-28** | Confirmed |
| **Final Energy Arithmetic Inconsistencies** | 0 | **0** | 0 | Confirmed |
| **Final Clinical Ratio Violations** | 0 | **0** | 0 | Confirmed |
| **Closed-Loop Score Gate Failures ($\Delta p > 0.0015$)** | 7 | **7** | 0 | Confirmed |
| **Total Certified for Stage 3** | **3 (9.4%)** | **25 (78.1%)** | **+22 (+68.7 pp)** | Confirmed |
| **False Accepts (Certified but Infeasible)** | **0** | **0** | **0** | Confirmed |
| **Linear Continuous Feasibility Witnesses** | 32 | 32 | 0 | Confirmed |
| **Mean Execution Time per Case** | 7.40 s | 3.31 s | -4.09 s | GPU accelerated |

---

## 4. Analysis of Remaining Non-Certified Cases

Of the 32 evaluated profiles, exactly **7 cases (21.9%)** were rejected. All 7 cases strictly satisfied all 39 nutrient bounds, thermodynamic energy consistency, and all three clinical ratio envelopes. Their rejection was caused solely by the closed-loop Stage 1 model-score gate:

$$\max(\Delta p_{\text{diab}}, \Delta p_{\text{obes}}, \Delta p_{\text{cvd}}) > 0.0015$$

### Table 2: Detailed Breakdown of Score-Gate Failures

| Case ID | SEQN | Age / Sex | BMI (kg/m²) | Baseline CVD | $\Delta p_{\text{diab}}$ | $\Delta p_{\text{obes}}$ | $\Delta p_{\text{cvd}}$ | Primary Failing Outcome |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **01** | 109384 | 64 / M | 28.4 | 0.0461 | **+0.009422** | -0.058693 | -0.001100 | Diabetes score increased |
| **08** | 114389 | 42 / F | 31.2 | 0.0089 | -0.002614 | -0.033999 | **+0.002240** | CVD score increased |
| **09** | 115202 | 47 / M | 34.6 | 0.0321 | -0.017557 | -0.070238 | **+0.024776** | CVD score increased |
| **10** | 115481 | 75 / M | 26.8 | 0.1420 | **+0.019946** | -0.039614 | **+0.052174** | CVD & Diabetes increased |
| **12** | 117395 | 55 / M | 29.1 | 0.0210 | **+0.019524** | **+0.002845** | **+0.014062** | All 3 scores increased |
| **24** | 121966 | 58 / F | 35.8 | 0.0185 | -0.034334 | **+0.017398** | -0.013409 | Obesity score increased |
| **31** | 124707 | 68 / M | 30.5 | 0.0512 | -0.078255 | -0.099383 | **+0.013099** | CVD score increased |

### Scientific and Engineering Implications
1. **Gate Integrity Maintained:** In earlier development notes, algorithms occasionally adjusted thresholds or removed gating to force 100% acceptance. Here, the strict threshold ($\Delta p \le 0.0015$) was preserved without modification. Rejection is an essential safety feature protecting downstream meal synthesis from counterfactuals that increase predicted risk.
2. **Trade-off Conflicts in Neural Space:** In multi-task disease models, certain nutrient modifications that reduce obesity (such as major carbohydrate reductions) can induce feature combinations that elevate predicted CVD or diabetes logits. For these 7 individuals, exploratory 50-step gradient descent failed to discover a Pareto-improving direction within the programmed clinical envelope.

---

## 5. Pre-Regression Unit Test Verification

The unit test suite `test_stage2_joint_repair.py` was executed prior to the full regression run. All 10 tests passed (10/10 OK in 9.07s):
1. `test_01_feasible_target_already_satisfying_constraints_unchanged`: Feasible target returned with $\Delta = 0$ ($nit = 0$).
2. `test_02_inconsistent_energy_projected_to_feasible_energy_within_bounds`: $30\%$ energy error projected to $\le 0.01$ within bounds.
3. `test_03_calorie_correction_exceeding_upper_bound_constrained`: Inflated macros constrained to $KCAL \le u^{\text{tight}}_{\text{KCAL}}$.
4. `test_04_calorie_correction_below_lower_bound_constrained`: Deflated macros constrained to $KCAL \ge l^{\text{tight}}_{\text{KCAL}}$.
5. `test_05_conflicting_bounds_raise_explicit_failure`: Contradictory bounds raise `Stage2InfeasibleConstraintsError`.
6. `test_06_ratio_violations_corrected_within_bounds`: Sugar/fiber ratio violation brought into compliance within envelope.
7. `test_07_nonfinite_inputs_handled_cleanly`: NaN and inf inputs rejected with explicit errors.
8. `test_08_final_target_rescoring_uses_repaired_calibrated_model`: Confirmed Platt calibrator binding.
9. `test_09_non_dietary_features_remain_strictly_immutable`: Input biology and demographics strictly unchanged.
10. `test_10_rejection_when_score_gate_fails`: Case 10 rejected due to $\Delta p_{\text{CVD}} > 0.0015$.

Backward compatibility was verified against legacy unit tests (`verify_stage2_contracts.py`), with all 10 tests passing under `BIONUTRIGPT_STAGE1_RELEASE="legacy"`.

---

## 6. Scope, Integrity Bindings, and Clinical Limitations

### 6.1 Artifact Checksums and Integrity Bindings

```json
{
  "stage2_counterfactual_engine.py": "57f5979bc6bbdd40d7045b85e05d21a9a4e21a2eb4eeffcc6c7ea6e2162cf0cf",
  "stage1_deployment.py": "b6f849cf271d49c693a2e1d731bc994683a48e78044dc8100ff7343e8ea50eb0",
  "nutrient_rule_policy.py": "173873e3a4e929fa77884d5f9d78465b16e4fe079717cb4ab9fb355df107bfec",
  "cases_LOCAL.json": "7b9019620ca755651973a38dbb91d5b1ae5674637b23773eb46924860be19036",
  "bionutrigpt_stage1_best.pt": "54cb1fb9e5746c1fbe2492fec2eb5d8cbb82683ba56f6cf3a7fcb8f7203b5550",
  "calibrators.joblib": "4be6b7afa46d00b461f6d65e4e076220ceb16ebe7cbe8e77ad2b33fdd2bf3d4f",
  "verify_stage2_joint_constraint_repair_v1.py": "0d9566b0534c82169faadd5b94de8142fd853c91622a8b2d0be55d6250a69649"
}
```

### 6.2 Clinical and Methodological Scope
- **Engineering Diagnostic Study:** This regression audit demonstrates mathematical constraint satisfaction on 32 complete-input temporal NHANES profiles. It does not establish clinical efficacy, patient tolerability, or disease risk reduction in real patients.
- **Sedentary Engineering Assumption:** Activity was fixed to sedentary for computational benchmarking; individualized clinical prescribing requires objective physical activity measurement.
- **Discrete Feasibility Distinction:** Demonstrating continuous nutrient-space feasibility does not guarantee that a discrete cooked Indian meal plan exists in the food catalog for every profile (which is evaluated separately in Stage 3).
