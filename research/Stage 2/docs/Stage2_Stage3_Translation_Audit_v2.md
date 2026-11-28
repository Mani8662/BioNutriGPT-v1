# Stage 2 to Stage 3 translation: corrected existing-record audit v2

Updated 23 November 2026. No solver rerun, model retraining or architecture change. Original study records and historical document copies are preserved.

## Corrected findings

Reanalysis of 900 existing records (100 participants, three arms and three dietary modes) changed no acceptance decisions. Model-guided versus minimum-change end-to-end acceptance differed by 7.33 percentage points (participant-cluster bootstrap 95% CI 2.00 to 13.33). The earlier pooled exact p-value was invalid because dietary modes were treated as independent people; it is withdrawn. No pooled exact p-value is reported.

| Method | Verified Stage 3 plans | End-to-end accepted | Conditional success after Stage 2 | Median recorded call time |
| --- | ---: | ---: | ---: | ---: |
| Rules-only | 298/300 | 144/300 | 144/144 | 3.031 s |
| Minimum-change | 297/300 | 209/300 | 209/210 | 3.031 s |
| Model-guided | 298/300 | 231/300 | 231/231 | 3.032 s |

The 300 cells per arm are repeated dietary-mode evaluations of 100 people, not 300 independent participants. Conditional denominators are displayed separately and are not the primary paired comparison.

| Mode: model-guided vs minimum-change end-to-end | Difference (pp) | Exact p | Holm-adjusted p (nine comparisons) |
| --- | ---: | ---: | ---: |
| pure_vegetarian | +7.00 | 0.0390625 | 0.0781250 |
| veg_nonveg_combo | +8.00 | 0.0214844 | 0.0644531 |
| pure_non_vegetarian | +7.00 | 0.0390625 | 0.0781250 |

The protocol did not fix a primary mode or a multiplicity family. This correction is explicitly retrospective: pooled confidence intervals resample 100 participants with their modes together; per-mode exact tests are exploratory, with Holm adjustment across nine mode-by-arm comparisons per binary endpoint. The pointwise 95% intervals are not multiplicity-adjusted. Model-guided versus minimum-change is not significant at .05 in any individual mode after this adjustment. This does not prove equivalence.

## Solver classification

Archived solver metadata show 881 time-limited solves with independently verified plans, 12 solver-status-0 results, three solver-reported infeasibilities and four timeouts without a verified plan. Status 0 is interpreted under the configured solver gap tolerance (1%), not as an independently proven exact optimum. The audit does not rerun or independently certify infeasibility. All 893 returned baskets passed fresh arithmetic and structural checks.

The approximately 3.03-second medians describe recorded call durations, not proof of optimal solutions. The original aggregate wall time is historical runtime metadata; it cannot be independently reconstructed by summing concurrent task durations. No new wall-clock benchmark is claimed.

## Architecture interpretation

With active nutrient slack fixed to zero and the same energy prescription and bounds, arm-specific non-energy targets do not change the effective selection objective or mathematical feasible set. They can affect coefficients on fixed-zero slack variables, so the unreduced numerical model need not be byte-identical. Identical problems also need not produce identical incumbents within a time limit. Only 274 of 300 participant-mode groups had identical baskets across all three arms; 26 differed. The one-plan operational difference is not evidence that model guidance expands feasibility. End-to-end differences primarily inherit the upstream model-score gate, with a one-cell downstream timeout difference.

## Target-distance correction

The original all-field weighted distance is withdrawn from comparative interpretation: it was incorrectly labeled as using Stage 2 scales and active nutrients. A new explicitly post-hoc metric uses all 30 active nutrients, the exact fixed Stage 2 scales and only jointly successful participant-mode pairs. Unknown contributions are excluded rather than treated as known zero. Model-guided minus minimum-change paired mean distance difference: -0.0841, pointwise bootstrap 95% CI [-0.1648, -0.0043], over 100 participants and 297 paired mode cells. Each participant contributes the mean over their available matched modes. This exploratory geometric difference is not evidence of better meals or clinical benefit; it compares distances to different target vectors. Per-nutrient absolute errors and units are retained locally.

## Verification performed

The new replay reconstructs inputs from the unchanged Stage 2 checkpoints and participant metadata, checks all 900 input fingerprints, recalculates food quantities, integrality, category/serving limits, dietary exclusions, active nutrient coverage and bounds, energy, macro fractions, fiber density and sodium/potassium checks. It recomputes Stage 3 acceptance instead of trusting saved flags. Stage 2 gate decisions are recomputed from saved verified probabilities and nutrient vectors; neural inference is not repeated.

Maximum nutrient accounting residual: 4.99982357e-07; acceptance mismatches: zero. Five corruption/safeguard checks passed (invalid serving count, changed nutrient total, changed bound, false saved acceptance flag and unknown food ID). A separate statistics implementation verified 27 paired confidence intervals, 18 exact tests, both Holm-adjustment families, distances, counts and timing summaries.

Shared dependencies are disclosed: food parsing, Stage 2 bound construction, rule metadata and fingerprint serialization use existing helpers. Basket acceptance logic and statistical replay are separately implemented. This does not validate the underlying catalog measurements or clinical appropriateness.

## Provenance limits

Current hashes match the historical source bindings. All original JSON artifacts and upstream checkpoints remained unchanged throughout replay. The historical runner rewrote its protocol and did not bind its own source or all upstream checkpoints; retrospective replay cannot establish that a prospectively immutable protocol existed. The new audit captures an input hash manifest and source hashes but does not erase that limitation. These are reused US profiles, not Indian prediction validation.

## Evidence files

`Saved Model/stage2_stage3_translation_audit_v2/` contains input_manifest.json, recomputed_records_LOCAL.json, acceptance_mismatches.json, analysis_summary.json, replay_verification.json and verification.json. The original 900 checkpoints remain in `Saved Model/stage2_stage3_translation_study_v1/checkpoints/`.

## Next decision

This repair is complete. Testing an explicit target-deviation objective in Stage 3 would be a separate architectural experiment, with the present bounds-based objective as comparator. It has not been implemented or run by this audit.
