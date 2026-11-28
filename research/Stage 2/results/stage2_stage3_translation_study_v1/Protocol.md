# Protocol: stage2_stage3_translation_study_v1

**Date:** 2026-11-23T11:05:58+0530  
**Cohort:** 100 reserved temporal NHANES adult participants (`cases_LOCAL.json`, SHA-256: `a29ebab45295090d...`)  
**Catalog:** 49 cooked dishes (`food_composition_cooked_sourced.json`, SHA-256: `a66b145f088f02c8...`)  
**Total Tasks:** 900 (100 cases x 3 arms x 3 dietary modes)  
**Time Limit:** 3.0s per solve via HiGHS MILP on 10 CPU worker threads.  

## Evaluated Arms
1. **Arm A (Rules-Only):** Deterministic clinical reference target vector.
2. **Arm B (Minimum-Change):** Scale-normalized convex quadratic projection of observed intake.
3. **Arm C (Model-Guided):** 50-step Adam search on risk loss + scale-normalized QP projection.

## Evaluated Dietary Modes
1. `pure_vegetarian`: lacto-vegetarian, zero non-veg foods.
2. `veg_nonveg_combo`: combination of vegetarian staples with non-veg protein dish.
3. `pure_non_vegetarian`: non-veg protein prioritized with cereal and vegetable sabzi.

## Dual Evaluation Protocol
1. **Evaluation A (Diagnostic Translation):** Evaluates all 900 target-to-meal solves regardless of upstream score gate. Evaluates discrete meal feasibility, dish variety, portion compliance, and soft target attainment.
2. **Evaluation B (End-to-End Acceptance):** A case succeeds if and only if upstream Stage 2 accepted AND downstream Stage 3 independently passes all checks. Denominator is all 100 participants.