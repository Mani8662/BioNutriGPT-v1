# Stage 3 robustness repair v3 — 24 November 2026

Status: runner repaired and one-participant engineering pilot verified. The full 100-participant robustness benchmark has NOT run. This supersedes the v1 preparation-readiness claims. V1 and the intermediate v2 evidence remain preserved.

## Repairs

- New working CLI and complete sequential execution loop, with atomic per-objective checkpoints. No four-worker solver timing comparison.
- Resume reconstructs plan validity and arithmetic before reuse, rejects fingerprint/identity changes, and does not overwrite corrupt checkpoints. Quarantined disagreements require review.
- Nutrient replays are recalculated from the nominal accepted plan and perturbed catalog, including nutrient bounds, energy, macro fractions, sodium/potassium and fibre checks. Missing active data cannot become valid zero values. Rejected nominal plans have no replay nutrient estimate.
- Hypothetical perturbations use one common factor per food, preserving nominal energy and nutrient relationships, unknowns and zeros. Five/ten-percent severities use nested directions. There is ONE fixed catalog draw per severity, not an empirical uncertainty distribution.
- Target error uses active non-energy nutrients. Unknown report-only nutrients no longer suppress target error. Solver-reported target error is cross-checked when available.
- Equal-participant contrasts average modes within participant. Target-error contrasts use jointly valid cells; runtime contrasts include all measured attempts. Bootstrap uses 10,000 draws with seed 20261124. Exact tests are within scenario/mode; Holm family has 21 tests. Replay binary outcomes represent available compliant plans, not conditional compliance among successful nominal plans. Intervals are exploratory/unadjusted.
- Separate statistical verifier reproduces point estimates, intervals, exact probabilities and Holm correction with separate formulas. Catalog checking shares legacy constraint/category helpers; full software independence is not claimed.

## Completed checks

One reused real participant (case 0), three modes, five solver scenarios and two replay scenarios: 30 measured solver calls and 12 arithmetic replays. The original v2 pilot exposed a target-error omission; v3 retained those measured solver outputs with source hashes and recomputed audits/replays. No replacement participants or solver reruns were used to obtain more favorable outcomes.

All 42 v3 pilot records passed reconstruction. Nineteen nonempty confidence-interval calculations and 21 exact tests passed the separate statistical verifier. With one participant these intervals are mechanical checks ONLY, not generalization evidence.

Eleven checkpoint/record corruption checks passed, plus perturbation-bound/unknown/zero checks, changed-CI rejection, actual changed-file manifest rejection, exclusion rejection and corrupt-resume refusal. Resume reloaded all 42 checkpoints with solver calls disabled and preserved their bytes.

Nominal: both objectives returned valid plans in all three modes. Stress outcomes include rejected plans; rejection itself is not automatically proof of infeasibility. Failures and raw metadata remain in the checkpoints. No claim of clinical safety, benefit, Q1/Q2 readiness or broad robustness follows.

## Frozen full study and commands

100 reused participants; model-guided cached targets; three modes; two objectives; five-second budget. Planned workload: 3,000 solver calls plus at most 1,200 nutrient replays. The nominal solve-budget sum is 15,000 seconds (4.17 hours), plus wrapper/startup/verification overhead; this is NOT a strict wall-clock upper bound. No 5–10 minute parallel runtime claim is retained. Pilot artifacts are isolated from full-run checkpoints.

Dry run:
`python stage3_robustness_v3.py`

Future full run (not performed in this repair):
`python stage3_robustness_v3.py --run-full`

Full independent statistical verification:
`python verify_stage3_robustness_v3.py`

Pilot/resume checks:
`python stage3_robustness_v3.py --pilot`
`python test_stage3_robustness_v3.py`
`python test_stage3_robustness_resume_v3.py`
`python verify_stage3_robustness_v3.py --pilot`

## Limits

This is an exploratory extension using a previously inspected cohort; case 0 was used during development. Dairy exclusion explicitly permits zero dairy under existing policy and is not an unchanged-category test. Constraints are audited with the documented shared legacy checker; it records the first failing constraint rather than an exhaustive list of every residual. Missing solver diagnostics remain unavailable; no dual bound or optimality result is invented. No model training, Stage 2 target regeneration, production deployment or full-cohort run occurred. Frozen source changes require a new study version.

Evidence: Saved Model/stage3_robustness_v3/{protocol.json,input_manifest.json,scenario_manifest.json,repair_tests.json,resume_tests.json,pilot/verification.json,pilot/independent_statistics_verification.json,pilot/migration.json}.
