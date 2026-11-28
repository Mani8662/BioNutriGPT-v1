# LASI evaluation audit

Audit of saved results and current benchmark source; no inference rerun or participant-level export.

## Verified saved counts

- Merged 66,859; eligible 65,368; evaluated 100 unique profiles (25 per disease stratum, 52 male/48 female, 29 states/UTs).
- Stage 3: 300 solves on 100 profiles; 285 accepted, 10 reported infeasible and 5 timed out without a valid plan. These are repeated dietary-mode evaluations, not 300 people.
- Acceptance: vegetarian 91/100; combined 96/100; non-vegetarian 98/100. Vegetarian below 1500 kcal: 21/27 (77.8%).
- Guarded chatbot: script-scored 60/60; of 30 swap requests, 26 have validated proposals and four receive no proposal but are scored successful refusals.
- Matched subset: guarded script-scored 24/24 versus Qwen 2/24. Guarded produced 10 validated proposals from 12 swaps; Qwen produced none; its two scored successes are no-proposal refusals.
- Change-penalty ablation: mean serving delta 9.46 versus 16.08 among 26 accepted swaps per system (41.1% lower descriptively).

## Publication interpretation

This is useful real-profile computational evidence with important remaining design limitations. The numerical summary counts reconcile. It is not a nationally representative Indian effectiveness study or Stage 1 external validation.

## Corrections and remaining evidence repairs

- Balanced 100-profile benchmark is not a nationally representative prevalence sample; counts and summaries are unweighted.
- Stage 2 adapter assigns 0.85/0.10 disease-status routing scores; these are not predicted probabilities. Sodium/potassium baselines are assumed 2800/2600 mg.
- This tests adapted Stage 2 bounds plus Stages 3/4, not Stage 1 external validity or the full Stage 2 counterfactual search.
- Every saved chatbot scenario is diabetes_only; first accepted profiles were selected from a stratum-ordered file. Distinct participant count is not saved in chat scenario logs.
- Oracle failure within two seconds is treated as grounds for a successful refusal, although timeout does not prove infeasibility.
- Qwen catalog prompt supplies kcal, protein and B12 but omits most of the 30 active nutrient coefficients; it also omits full plate rules/objective. This is not an equal-information comparison.
- Second swap prompts adapt to system-specific baskets. Ablation traces are consequently not identical interventions after divergence.
- Raw full baskets/verification certificates and full raw LLM responses are not retained in these new reports; chat logs contain truncated answer excerpts.
- Grounding checks use number proximity on selected nutrient turns, not exhaustive semantic hallucination detection.
- No human acceptability, clinical benefit, Stage 1 validation, or weighted national performance is established.

Do not overwrite the saved raw reports to make them appear corrected or rerun. This audit is the interpretive correction. Before a new scientific run, freeze a disease-balanced scenario list, full shared information, feasible/timeout/refusal definitions and complete evidence logging.
