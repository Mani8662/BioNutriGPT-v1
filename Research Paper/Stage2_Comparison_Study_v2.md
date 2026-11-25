# Stage 2 three-method comparison: corrected evidence v2

Updated 23 November 2026. Post-result correction on the same 100 participants; this is not an independent replication cohort. Original v1 results and document copies are preserved.

## Methods and scope

All arms share participant inputs, baseline calibrated predictions for routing, energy prescriptions, nutrient envelopes and final score gate. Rules-only refers to target construction without model-guided search, not a model-free pipeline. Its exact values are project reference policies and must not be represented as uniformly established clinical guidelines. Minimum-change projects observed nutrients; model-guided runs the existing 50-step search then the same projection. No model assets, routing thresholds, search objective or production engine were changed.

## Corrected results

| Method | Completed | Numerical pass | All-check accepted | Median runtime (s) | Median normalized change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules-only construction | 100/100 | 100/100 | 48/100 | 0.031 | 4.8190 |
| Minimum-change projection | 100/100 | 100/100 | 70/100 | 0.181 | 1.8494 |
| Model-guided search | 100/100 | 100/100 | 77/100 | 2.879 | 2.1992 |

Model-guided search accepted 77/100 profiles versus 70/100 with minimum-change projection and 48/100 with rules-only construction. The primary paired difference was +7.00 percentage points (10,000-resample percentile bootstrap 95% CI +2.00 to +13.00); the exact two-sided discordance test gave p=0.0390625 (8 wins, 1 losses). This is a model-score acceptance result, not demonstrated health improvement.

The paired mean additional normalized displacement for model-guided search was 0.2717 (bootstrap 95% CI 0.2325 to 0.3153). The scales are fixed engineering weights; this does not measure meal acceptability.

## Corrections and verification

- Removed duplicate Adam execution from the timing runner. The saved pre-projection vector now comes from the actual search used for the final target.
- Applied 1e-6 absolute tolerance in nutrient units to all Filter 2 inequalities in the versioned comparison checker. Case 49 previously failed because 0.7999999999999999 was compared strictly against 0.8 g/kg. The historical production diagnostic remains archived in per-arm audits; authoritative comparison decisions use the corrected common checker. Production code itself is unchanged.
- Replaced bootstrap tail proportions with exact paired binary tests. Corrected the distance confidence interval to bootstrap the paired mean rather than reporting percentiles of individual differences.
- Fresh model replay checked all 300 outputs, baseline probabilities and immutable features. The verifier shares deployed preprocessing and a numerical-check helper; statistical/distance calculations were independently reproduced from saved vectors and records.
- Seven targeted correction tests passed. Maximum target change versus v1: 0; changed overall acceptance decisions: 0.
- Actual run device: cuda; GPU: NVIDIA GeForce RTX 3050 Laptop GPU; PyTorch CPU threads: 12. Timings include baseline inference and candidate evaluation, exclude model loading, and use CUDA synchronization and an untimed inference warm-up. Fixed arm order and one timing run limit performance claims. The old 5.74-second and 28.5-fold figures are superseded.

## Paired acceptance contrasts

| Contrast | Difference (pp) | Pointwise bootstrap 95% CI (pp) | Exact p | Secondary Bonferroni-adjusted p |
| --- | ---: | --- | ---: | ---: |
| Primary: Model-Guided (C) vs Minimum-Change (B) | +7.00 | [+2.00, +13.00] | 0.0390625 | Not applicable |
| Secondary: Model-Guided (C) vs Rules-Only (A) | +29.00 | [+19.00, +39.00] | 1.084394e-06 | 2.168788e-06 |
| Secondary: Minimum-Change (B) vs Rules-Only (A) | +22.00 | [+11.00, +33.00] | 0.0001951256 | 0.0003902512 |

Primary acceptance contrast uses alpha .05. Two secondary acceptance contrasts use Bonferroni correction; their displayed 95% intervals are pointwise, not simultaneous. Other endpoints are exploratory. Statistical corrections were made after reviewing v1 results and are disclosed as amendments.

## Limits and next step

The acceptance gate is self-referential: the prediction model also guides optimization. Neither clinical benefit nor Indian prediction validity is established. Continuous nutrient feasibility does not ensure a realizable meal. Zeroed or independently assigned nutrient components in the rules-only construction are engineering targets, not a demonstrated food composition. No cascade was deployed or prospectively tested. Next: a matched Stage 3 translation study using frozen food data, constraints, objectives and solver budgets.

## Artifacts

`Saved Model/stage2_comparison_study_v2/` contains protocol.json, cases_LOCAL.json, 300 checkpoints, summary.json, analysis_summary.json, paired_contrasts.csv, correction_tests.json, verification.json and audit_complete.json. The latter verifies full paired intervals, exact tests, numerical counts, distances and timing summaries. Source scripts are versioned with v2 suffixes.
