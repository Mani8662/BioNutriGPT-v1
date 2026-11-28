# Stage 2 joint constraint repair: corrected verification and regression

Updated 22 November 2026. This report supersedes the earlier interpretation; original source/report and verification artifacts are retained in `Saved Model/stage2_joint_repair_correction_v2/before_correction/` and the original v1 study directory.

## Problem and implementation

In the original 32-profile audit, sequential energy adjustment introduced calorie-envelope violations in 28 cases. Only three cases passed all programmed acceptance checks. The repair jointly enforces nutrient intervals, implemented protein/energy/sodium/potassium rules, energy arithmetic and three ratio envelopes after the 50-step Adam candidate search. Energy arithmetic uses 4 kcal/g protein, 4 carbohydrate, 9 fat and 7 alcohol. The internal energy margin is ±0.99%; final acceptance uses 1%. The repaired deployment uses descriptive research routing thresholds of 0.5, including CVD; the former report's 0.603 was incorrect.

The objective minimizes the sum of squared candidate changes divided by fixed nutrient-specific engineering scales. These weights are not uniformly official FDA/DRI recommendations. SLSQP numerically solves the convex quadratic problem using a HiGHS linear-feasibility witness as initialization. Numerical convergence is not a mathematical exactness certificate. If SLSQP fails or returns an invalid candidate, the engine can return a checked feasible LP witness, explicitly marked `feasible_lp_fallback` and `minimum_distance_not_established`. LP infeasibility is distinguished from time limits or other solver failures. The final model-score gate remains separate; nutrient feasibility does not guarantee that gate or discrete meal feasibility.

## Measured results

| Check | Original audit | Corrected repair regression |
| --- | ---: | ---: |
| Profiles | 32 | 32 |
| Accepted by all programmed checks | 3 | 25 |
| Calorie-envelope violations | 28 | 0 |
| Model-score gate failures | 7 | 7 |

All final nutrient bounds, energy and ratio checks passed in the corrected regression. The three baseline accepted cases (7, 14, 19) remained accepted; 22 additional cases passed. Cases 1, 8, 9, 10, 12, 24 and 31 remained rejected by the score gate. No accepted output failed the independent checks in this sample. The gate limits increases in each calibrated outcome probability to 0.0015; it is not a demonstrated clinical benefit criterion. Outcomes are diabetes, obesity and CVD, not prediabetes.

The corrected implementation produced identical final target vectors and acceptance decisions to the original joint-repair run. No LP fallback occurred in these 32 runs. Total recorded regression runtime was 116.14 seconds; this is descriptive and is not evidence of a speed improvement.

All 16 tests passed, including six tests covering SLSQP nonconvergence, nonfinite output, exceptions, invalid LP witnesses and distinctions between solver failure and infeasibility. Independent frozen-model/calibrator replay verified baseline and candidate probabilities for all 32 profiles, with maximum absolute discrepancy 5.74e-8. Replay uses the deployed preprocessing/candidate-construction path; it does not claim an independently implemented preprocessing system.

## Size of target changes

The following dimensionless L2 distances use the same fixed engineering scales as the objective; they do not measure clinical benefit or practical meal acceptability.

- observed to final: median 2.3796, range 1.2302–9.5280.
- optimizer clipped to final: median 0.1656, range 0.0063–0.6428.

## Evidence and next study

- Historical corrected verification: `Saved Model/stage2_joint_repair_correction_v2/verification.json`.
- Corrected engine regression: `Saved Model/stage2_joint_constraint_repair_v2/summary.json` and per-case artifacts.
- Fresh regression model replay: `Saved Model/stage2_joint_repair_correction_v2/model_replay_v2/independent_review.json`.
- Tests and target-change summary: `Saved Model/stage2_joint_repair_correction_v2/`.
- Reserved sample and frozen source copies: `Saved Model/stage2_comparison_design_v1/`.

These 32 complete-input NHANES profiles are development/regression cases with sedentary activity assumed. They do not establish Indian validity, nutritional appropriateness, causal risk reduction, global optimality, or superiority over a rules-only approach.

A 100-profile sample is reserved with zero overlap with the 32 repair cases, earlier 100-profile downstream study or five-case pilot. It is not globally unseen Stage 1 data. No comparison outcomes have been run. Next, implement and freeze the exact rules-only construction, minimum-change projection arm and full model-guided arm, shared evaluation rules and downstream solver settings, using development cases before inspecting reserved outcomes.
