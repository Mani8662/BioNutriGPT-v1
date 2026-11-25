# Protocol: stage3_target_aware_experiment_v1

## Overview
Controlled engineering evaluation comparing the experimental target-aware Stage 3 optimizer (Variant B)
against the current bounds-based Stage 3 optimizer (Variant A) across 100 participants x 3 Stage 2 arms x 3 dietary modes.

## Key Mathematical Invariants
- Identical discrete feasible polytope: Proj_{(q,z)}(X_B) = X_A.
- Zero-slack invariant: Deviation variables cannot purchase violations of hard component bounds.
- Energy conservation: Calorie target penalized once via cal_over/cal_under; excluded from d_k^+/d_k^-.
- Normalized L1 goal programming: Standard FDA Daily Reference Scales s_k with positive, finite units.

## Study Dimensions
- Cohort: 100 adult temporal NHANES participants (disjoint from pilot and training sets)
- Arms: Rules-Only (RO), Minimum-Change (MC), Model-Guided (MG)
- Dietary Modes: Pure Vegetarian, Veg/Non-Veg Combo, Pure Non-Vegetarian
- Total Paired Tasks: 900 (1,800 total solves)
- Time Limit: 3.0 seconds per solve
- Target Weight (lambda): 1.0

## Primary Metrics
1. Discrete Meal Feasibility Rate (% accepted by independent verifier)
2. Mean Normalized L1 Target Error (NTE_L1)
3. Normalized L2 Target Distance (NTD_L2)
4. Core Nutrients MAPE (%)
5. Solve Time (seconds) and MIP Gap
6. Food Basket Composition (dish count, total servings, total mass)