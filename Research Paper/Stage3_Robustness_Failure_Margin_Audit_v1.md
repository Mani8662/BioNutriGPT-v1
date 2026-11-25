# Stage 3 all-rule failure and margin audit

Date: 24 November 2026. Retrospective analysis of existing records; zero solver or model calls. Original results and checkpoints preserved.

## Verification

All 4,200 classifications agree with a new all-rule reconstruction. All 4,200 checkpoint hashes remained unchanged. A positive control and a synthetic simultaneous energy/protein/sodium violation fixture passed. Separate aggregation and interval formulas reproduced 94 paired margin confidence intervals. Shared category-policy helpers are disclosed; this is not an independent clinical-policy review. Infinite category upper limits are unconstrained and are not exported as numeric bounds.

## Perturbation findings

Each objective has 299 available nominal plans out of 300 participant-mode cells. Under the fixed 5% perturbation, A has 236 compliant plans and B 215 (63 versus 84 available plans fail). Under 10%, A has 192 and B 159 (107 versus 140 fail). One missing nominal plan per objective is not assigned a nutrient violation. These are fixed proportional per-food perturbations, not measured error distributions.

| Rule | A violations at 5% | B violations at 5% | A violations at 10% | B violations at 10% |
|---|---:|---:|---:|---:|
| PUFA upper bound | 8 | 26 | 19 | 39 |
| Folate DFE lower bound | 1 | 11 | 1 | 16 |
| Niacin lower bound | 5 | 10 | 9 | 16 |
| Energy upper bound (fraction of target) | 0 | 1 | 2 | 14 |
| Potassium upper bound | 8 | 1 | 21 | 3 |
| Carbohydrate upper bound | 9 | 14 | 15 | 20 |

Counts overlap: a plan may fail several rules. Energy envelope and energy-prescription checks can flag the same event; do not add their counts as independent failures. Full per-record rules and residuals are in all_rule_records.json.

## Matched nominal margins

Margin means distance from the relevant bound, with positive values indicating room before violation. Comparisons use 299 jointly valid mode cells across 100 participants, equal participant weighting after averaging available modes. Below, nutrient margins are converted back to native units using the fixed normalization scale; energy remains a fraction of the prescribed target. Intervals are retrospective, exploratory and unadjusted across 94 rules, not confirmatory significance claims.

| Rule | Mean margin A | Mean margin B | B minus A [95% interval] |
|---|---:|---:|---|
| PUFA upper bound | 3.6573 | 2.3534 | -1.3039 [-1.7295, -0.8874] |
| Folate DFE lower bound | 467.7366 | 157.0030 | -310.7335 [-339.9308, -281.8552] |
| Niacin lower bound | 9.8596 | 8.6130 | -1.2465 [-2.3454, -0.1337] |
| Energy upper bound (fraction of target) | 0.0500 | 0.0486 | -0.0015 [-0.0023, -0.0006] |
| Potassium upper bound | 482.1441 | 982.6286 | 500.4845 [443.2748, 557.4310] |
| Carbohydrate upper bound | 30.3723 | 38.4656 | 8.0933 [5.4935, 10.7064] |

PUFA margins are in grams, folate DFE in micrograms, niacin in milligrams, potassium in milligrams, and carbohydrate in grams. Reported engineering bounds are not thereby established as clinically appropriate limits.

B has smaller average margins for PUFA upper limits and folate lower limits, alongside more violations of those rules under the fixed perturbations. B has larger potassium-upper-bound margins and fewer such failures. Carbohydrate shows why a single mean is insufficient: B has a larger average upper-bound margin but more perturbed violations. Therefore the data support rule-specific vulnerability, not a universal closer-to-bound mechanism or a causal explanation of all failures. Variation in plans and per-food factors also matters.

## Solver failures, kept separate from nutrient replay

Counts below are per objective; A and B share these failure counts:
- Nominal: 1 pure-vegetarian case, code infeasible_active_rules (not a timeout).
- 10% removal: 4 pure-vegetarian cases, code infeasible_active_rules.
- 20% removal: 76 pure-vegetarian cases, code infeasible_active_rules.
- Dairy exclusion: 100 pure-vegetarian cases, code infeasible_active_rules. Mixed and non-vegetarian cases succeeded. The previous opposite-mode explanation was incorrect. These records alone do not identify a minimal conflicting nutrient set.
- Legume exclusion: 100 pure-vegetarian and 100 mixed cases, code insufficient_nutrient_data. The scenario removes a mandatory dal/pulse category, a structural conflict rather than demonstrated timeout or a measured nutrient deficit. Non-vegetarian cases succeeded.

No timeout-coded failure occurred among these saved rejected solves. An infeasible solver status pertains to the encoded model, not an independent nutritional impossibility proof. Rejected solves have no final basket from which all nutrient deficits can be diagnosed.

## Corrections and next step

Removed the unsupported target-error p<0.001 claim; retain measured target-error intervals. Corrected the claim that scaling enforces 4P+4C+9F=E: scaling preserves existing catalog relationships. Removed the untested universal margin mechanism and erroneous nominal-timeout/dairy-mode descriptions.

Next candidate experiment: prespecify a research-only objective with a penalty for proximity to selected hard bounds, then compare it against the current target-aware objective under identical conditions. Select weights on separate development data and test on fresh cases/perturbation draws. Do not relax hard rules or deploy a change based on this audit alone. Catalog/constraint feasibility for dairy-free vegetarian plans is a separate unresolved diagnostic question.
