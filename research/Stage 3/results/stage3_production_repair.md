# Stage 3 acceptance repair

Production now returns either an explicitly accepted plan or an empty failure.
The canonical `solve_meal_plan` checks data coverage before solving. Foods with
unknown active nutrient contributions cannot satisfy those rules. A lack of
documented candidates for required meal categories is reported as
`insufficient_nutrient_data`, not mathematical infeasibility.

For eligible data, active nutrient slack is fixed to zero. Energy, macronutrient,
sodium-to-potassium, and fiber post-checks are represented in the search and
independently recalculated afterward. Portion and category checks use catalog
IDs and quantities, rather than trusting a claimed nutrient summary. The solver
candidate and heuristic both pass the same validator before scoring selects
between them. A cheap invalid heuristic cannot displace a valid incumbent.

Failures distinguish data coverage, proven infeasibility of the configured
model, timeout without a passing candidate, solver error, and validation failure.
Failure outputs have no solved basket, delivered meal nutrition, or schedule.
Acceptance concerns configured active arithmetic rules; it is not clinical
certification. Report-only fields and missing data are still disclosed.

FastAPI and the CLI block Stage 4 for rejected/unverified plans. Stage 4 also
revalidates an alleged accepted plan against current catalog arithmetic before
loading an LLM. The legacy Stage 3 filename delegates to the canonical solver.
Chatbot proposals and acceptance use that validator; a failed revision keeps the
current plan unchanged. The frontend displays the failure, withholds plan-change
controls and never shows a rejected basket as an accepted plan.

## Verification

`python verify_stage3_production.py` passes nine regression tests, including a
real SciPy solve on an explicitly synthetic complete-data fixture, failed real
catalog coverage, candidate tampering, a cheaper invalid heuristic, timeout
classification, pipeline blocking, and chatbot success/failure flows. Synthetic
tests are control-flow evidence, not real-food validation.

`python run_stage3_live_sample.py` executes the real sample with Stage 1/2 on an
NVIDIA RTX 3050 Laptop GPU. The measured run took about 8.63 seconds. Stage 2's
programmed checks passed, Stage 3 returned `insufficient_nutrient_data`, and
Stage 4 was not called. Only nine of 29 candidates have all active data; these
lack pulse and vegetable categories. The failure lists eight missing active
fields and all fourteen incomplete catalog fields. Exact output is recorded in
`stage3_production_live_sample.json`.

The sourced-catalog audit can still find a candidate for the 24 covered active
intervals. That restricted result does not pass the full production gate because
eight other active fields remain incompletely observed. Additional compatible
data or an explicitly reviewed change in scope is needed; no rule was silently
waived to make the production run appear successful.
