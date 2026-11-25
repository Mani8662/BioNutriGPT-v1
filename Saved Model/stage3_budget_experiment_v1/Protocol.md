# Protocol: stage3_budget_experiment_v1

## Research Question
**Can additional computation recover the target-aware optimizer's valid-plan rate while retaining its lower nutrient-target error?**

## Study Architecture & Factorial Dimensions
- **Cohort**: 100 adult temporal NHANES participants (strictly disjoint from 32-case repair, 100-case downstream, 5-case pilot, and 100-case comparison cohorts; total 237 exclusions).
- **Stage 2 Target Sources (3 arms)**: Rules-Only (RO), Minimum-Change (MC), Model-Guided (MG).
- **Dietary Modes (3 modes)**: Pure Vegetarian, Veg/Non-Veg Combo, Pure Non-Vegetarian.
- **Solver Objectives (2 variants)**:
  - **Variant A (Current Bounds-Based)**: $\min \text{Cost}_{\text{selection}} + \lambda_{\text{cal}} |\text{kcal} - T_{\text{kcal}}| + M \cdot \text{slack}$ (with slack clamped to 0).
  - **Variant B (Target-Aware)**: $\min \text{Cost}_{\text{selection}} + \lambda_{\text{cal}} |\text{kcal} - T_{\text{kcal}}| + \lambda_{\text{target}} \sum_k \frac{|y_k - T_k|}{s_k}$ (with slack clamped to 0).
- **Time Budgets (3 levels)**: 3.0 seconds, 5.0 seconds, 10.0 seconds.
- **Total Paired Tasks**: 100 participants $\times$ 3 arms $\times$ 3 dietary modes $\times$ 3 budgets = **2,700 paired tasks**.
- **Total Optimization Solves**: $2,700 \times 2 = \mathbf{5,400}$ individual solves.

## Mathematical Invariants & Implementation Safeguards
1. **Un-clamped Research Wrapper**: `stage3_budget_solver.py` enforces `RESEARCH_MAX_TIME_LIMIT_SEC = 15.0`, passing exactly 3.0s, 5.0s, and 10.0s directly to HiGHS.
2. **Production Baseline Untouched**: `stage3_milp_solver.py` default limits (1.5s default, 3.0s max clamp) remain untouched for clinical callers.
3. **Zero Active Slack**: Component bound slack upper bounds are fixed strictly at 0.0 for all active nutrients.
4. **Identical Feasible Polytope**: Discrete decision variables $(q, z)$, bounds, category constraints, and integrality are identical between Variant A and Variant B.
5. **Equal Participant Weighting**: Statistical contrasts weight each participant equally by averaging within-participant cells first.
6. **Jointly Successful Conditioning**: Target error comparisons and food basket comparisons are strictly conditioned on pairs where both variants found valid plans.
7. **Cryptographic Binding**: All code, manifest, targets, and catalog digests are frozen in `protocol.json` prior to execution.