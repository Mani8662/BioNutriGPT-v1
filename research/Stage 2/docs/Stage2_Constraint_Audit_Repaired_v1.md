# Stage 2 constraint consistency audit

Updated 22 November 2026. A fixed 32-profile audit of the repaired application release found that sequential post-processing frequently breaks the calorie envelope. Only 3 profiles passed all programmed acceptance checks. The production engine was observed without modification; rejected cases were not replaced.

## Protocol and scope

The sample was drawn with seed 20261122 from 6,033 complete-input participants in the repaired NHANES temporal cohort with measured weight 30–180 kg and height 120–210 cm. All 32 selected cases were processed. Observed age, sex and anthropometry were used; sedentary activity was a fixed engineering assumption, not inferred measured activity. Renal disease, pregnancy and lactation declarations remained unknown. This is an unweighted engineering pilot, not a representative patient study or clinical safety evaluation.

The frozen repaired_v1 checkpoint, preprocessing and reserved-set calibration were used on CUDA, with 12 PyTorch CPU threads. Research routing thresholds were 0.5. Each case used the current 50-step optimizer. Total recorded case runtime was 236.85 seconds, excluding setup. Source and artifact hashes remained unchanged.

Snapshots were recorded after the optimizer's final clipping and after each of the three modifying filters. An independent numerical checker tested all 39 search bounds, distinguished active nutrient fields from reporting fields, reconstructed energy and ratios, and checked final score changes. A separate archive verifier replayed the snapshots and final dossiers without importing the Stage 2 engine. Five hand-constructed fixtures tested a valid case and energy, ratio, bound and protein violations.

The bound tolerance was 1e-6 in each field's native units. Energy consistency was the engine's 1% criterion, with 1e-9 numerical tolerance on the fractional error. Ratio checks used the implemented limits: the greater of the baseline ratio and 1.0 for sugar/fiber and sodium/potassium, or 0.4 for saturated/unsaturated fat. These are the programmed engineering rules, not newly validated clinical rules.

## Results

| Check | Cases |
|---|---:|
| Attempted / returned dossiers | 32 / 32 |
| Execution errors | 0 |
| Passed all Stage 2 acceptance checks | 3 |
| Violated a bound immediately after optimizer clipping | 0 |
| Violated a bound after energy filter 1 | 28 |
| Violated a bound after filter 2 | 28 |
| Violated a bound after filter 3 | 28 |
| Final calorie-envelope violations | 28 |
| Final energy-arithmetic inconsistencies | 0 |
| Final ratio-rule violations | 0 |
| Final closed-loop model-score gate failures | 7 |
| Both calorie-bound and model-score failure | 6 |
| Accepted outputs failing independent checks | 0 |

Of the 28 calorie-envelope violations, 17 were below the lower bound and 11 above the upper bound. The distance outside the interval ranged from 50.84 to 871.55 kcal, with median 244.67 kcal. These are not rounding artifacts. The independent replay confirmed every bound-violation count, all final targets and score differences, and all three accepted cases.

## Actual failure mechanism

1. The gradient optimizer uses component-wise nutrient-bound penalties and then clips each nutrient into its permitted interval. This does not jointly enforce energy consistency across calories, protein, carbohydrate, fat and alcohol.
2. Filter 1 calculates energy as `4*protein + 4*carbohydrate + 9*fat + 7*alcohol`. When disagreement exceeds 1%, it replaces the calorie target with that calculated value.
3. This replacement can leave the original calorie interval. Subsequent filters do not jointly restore the calorie interval and macro-energy relationship.
4. The final acceptance check detects the out-of-bound target and rejects the case. Thus rejection protects the pipeline, but does not repair the prescription.

Example audit case 000: the prescribed calorie interval was 1,800.92–1,990.48 kcal. Filter 1 changed calories to 2,304.80 kcal, 314.32 kcal above the maximum. The filter's energy-arithmetic check passed because calories now equalled calculated energy; the overall envelope check correctly failed.

## Independent feasibility check

A continuous linear feasibility program found a nutrient-space witness for all 32 profiles under the same search bounds, energy-consistency inequality, ratio inequalities and filter-2 numerical rules. Every witness was independently replayed at 1e-6 feasibility tolerance.

This distinguishes a feasible set of nutrient rules from the particular target returned by sequential adjustment. It does not prove that every rejected case can pass the nonlinear model-score gate, that a discrete food basket exists, or that a target benefits health. There were 29 rejected cases with a feasible nutrient-space witness; six had both calorie and score failures, and one had a score failure without a calorie-bound failure.

## Required next experiment

Repair Stage 2 so that energy consistency and nutrient bounds are enforced jointly, instead of repairing calories through an unconstrained overwrite. Preserve the current frozen implementation/results as the comparator. Check feasibility before attempting model-guided optimization and retain explicit rejection if the final independently checked target is invalid.

Re-run these 32 profiles as regression cases, without treating them as untouched evaluation data. Then lock a separate evaluation sample for the matched comparison of rules-only targets, minimum-change constrained optimization and full model-guided optimization. Hold the envelope and disease-routing flags fixed across optimization methods. Compare joint constraint success, change magnitude, runtime, model-score changes and downstream Stage 3 acceptance. Threshold routing and component-removal ablations should remain separate experiments.

The seven score-gate failures require their own analysis; joint numeric feasibility alone will not establish a repair for those cases. A failure to improve a model's score is not evidence of harm, and a score decrease is not evidence of clinical benefit.

## Reproduction

- Runner: `audit_stage2_constraints_repaired_v1.py`.
- Independent archive verifier: `verify_stage2_constraint_audit_repaired_v1.py`.
- Frozen protocol, selected local cases, per-step snapshots, original dossiers, summary, source hashes and verification: `Saved Model/stage2_constraint_audit_repaired_v1/`.
- Re-running the same bound runner resumes saved cases; changed source/artifact bindings are rejected. A different intervention or repaired engine must use a new output version.

This audit did not retrain Stage 1, change the production optimizer, run Stage 3, or run a language model. It begins Stage 2 evaluation; it does not complete the planned baselines and ablations.
