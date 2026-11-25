# Stage 1 downstream optimizer connection pilot

Completed 2026-11-20. Five deterministic NHANES temporal profiles, 80 solves: one rules-only reference per participant and three calibrated prediction arms over five exploratory thresholds (.05/.10/.20/.30/.50). Repeated thresholds are not independent patients. Mixed-diet catalog, same three-second limit, same production objective and fixed rules-only nutrient targets. Four worker processes; wall times under concurrent load are not isolated latency benchmarks.

| Arm | Solves | Accepted | Baskets changed versus rules-only | Median wall seconds |
|---|---:|---:|---:|---:|
| rules_only | 5 | 5 | 0 | 3.014 |
| bionutrigpt | 25 | 25 | 11 | 3.014 |
| catboost | 25 | 25 | 13 | 3.014 |
| ft_transformer | 25 | 25 | 12 | 3.013 |

All 80 returned plans passed independent recalculation of nutrient totals, integer servings, active bounds, plate counts, energy, macro proportions, fiber density and sodium-to-potassium checks. All also passed the common rules-only reference checks. Unknown/report-only nutrient fields are not certified adequate. The independent checker reuses configured category limits, but computes quantities/totals and checks separately from production acceptance.

The rules-only arm was feasible for all five participants. Therefore this pilot demonstrates no feasibility improvement from Stage 1. Different baskets are not evidence of improved quality; solver ties/time limits can also change baskets. No cost, palatability, health benefit, causal risk reduction or clinical safety claim follows. Same-model rescoring was not used as evidence. Predicted rules can only tighten selected branches; known diagnoses and measured BMI retain precedence.

This is an implementation pilot, not the full added-value study. No inferential statistics are appropriate for five selected records. Next: freeze a broader paired evaluation sample, dietary modes, threshold policy and common-reference endpoints, then execute with all failures retained. Do not select thresholds based on favourable pilot results. The repaired differentiable counterfactual path remains separate and unimplemented.

Artifacts: protocol.json with source/catalog hashes; case_000_LOCAL.json through case_079_LOCAL.json with full inputs/baskets/solver reports/replay results; aggregate_summary.json and verification.json. Local participant-level files should not be published. The pilot sends research envelopes directly to the solver without claiming Stage 2 clinical certification. Production unchanged.
