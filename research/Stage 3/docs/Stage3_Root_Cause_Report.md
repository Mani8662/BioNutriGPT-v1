# Stage 3 root cause audit (saved example dossier)

## Subsequent B12 repair

The findings below describe the original catalog, preserved in
`Data/food_sources/USDA_SR_Legacy/before_b12_repair/`. The active catalog now has
31 explicit USDA SR Legacy proxy matches and 555 unknown B12 values instead of
placeholders. The original 1.0496 microgram maximum no longer applies to the
active catalog. After the repair, the continuous strict-bound model is feasible
and a 30-second integer run found a strict-bound incumbent. Unknown contributions
still block complete B12 verification; recipe ingredient data and the other
legacy nutrient estimates still need review. See the source README, mapping
report, and post-repair feasibility output in that source directory.

The saved example's Stage 3 result is not a fully compliant meal plan. Its output has 26/39 nutrients within Stage 2 bounds and fails the sodium-to-potassium post-check. The result was a heuristic basket selected after the 0.5-second HiGHS run timed out, despite an incumbent being available (`mip_gap = 0.9821`). The heuristic primarily chooses low-penalty foods and fills calories; it does not solve the 39-nutrient problem.

## Proven hard infeasibility: vitamin B12 data

I set all nutrient slack variables to zero and tested the *same* 586-food catalog, nutrient targets, category rules, and serving limits in `audit_stage3_feasibility.py`. The continuous relaxation was infeasible. Relaxing only the B12 bound made it feasible. Maximizing B12 while retaining every other nutrient bound produced **1.0496 micrograms**, below the Stage 2 minimum of **2.4 micrograms**. Therefore, no integer meal plan can meet every recorded nutrient bound under these data and rules; more solver time alone cannot fix that.

The direct data cause is in `build_indian_food_database.py`: all imported IFCT records receive `DR1TVB12 = 0.05` if categorized as dairy and `0.0` otherwise. Recipe records receive `0.50` if dairy and `0.0` otherwise. The builder does not read a measured B12 source field. The resulting catalog assigns zero B12 even to its meat and fish entries. Only 23 of 586 food rows have a positive B12 value, and the maximum is 0.5 micrograms per serving. These values are placeholders; they cannot justify a B12 adequacy claim.

## Why the saved basket misses more than B12

The solver permits two-sided slack on every nutrient. Its 0.5-second run stopped at the time limit, and the code selected a calorie-oriented heuristic instead of the available solver incumbent because that heuristic had a lower internal score. The saved basket delivers 80.46 g sugar against a 50 g ceiling and misses 12 additional nutrient bounds, including B12. This is a property of the selected basket, not proof that each of those 12 other bounds is impossible.

With B12 slack allowed and the other 38 nutrient slacks fixed to zero, a 30-second integer search found a candidate satisfying those 38 bounds. That candidate delivered 1.0 microgram B12. Its sodium-to-potassium ratio was approximately 0.435, below the saved baseline ratio of 0.543. The search timed out with an incumbent, so it did not prove optimality. It did demonstrate that the other nutrient failures and ratio failure of the saved basket were avoidable within the current mathematical model given more search time and different selection.

The sodium-to-potassium condition is evaluated in `verify_basket_safety` after optimization; `_make_sparse_constraints` does not include it as a constraint. Therefore, Stage 3 can return a basket that fails this check even when individual sodium and potassium bounds pass.

## Required repairs before presenting clinical adequacy

1. Replace placeholder B12 values with traceable measured or defensibly calculated values. Preserve missing B12 as unknown rather than zero; review every food's preparation and serving basis. Rebuild the catalog and rerun the strict feasibility audit.
2. Mark nutrient rules as hard or soft. If a hard rule cannot be satisfied, return a clear infeasible result rather than a meal plan described as compliant.
3. Include any mandatory sodium-to-potassium rule in optimization, or explicitly label it a post-check and reject plans that fail it.
4. Rework the heuristic and time budget. Record whether the selected plan is an optimal MILP solution, a time-limited incumbent, or a fallback. Compare each candidate against the same verification rules, not only the internal penalty.
5. Validate cookability of the food catalog. The 30-second candidate included raw ingredient records such as zucchini and fruit alongside cooked recipes; mathematical feasibility alone does not establish a realistic household meal.

This audit uses the current saved example and project food catalog. It is an engineering feasibility finding, not a nutritional recommendation.
