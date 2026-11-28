# Stage 3 budget study: reporting correction v3

Updated 24 November 2026. Production unchanged; no solver reruns.

| Budget | Current valid plans | Target-aware valid plans | Difference (pp), participant-cluster 95% CI |
| --- | ---: | ---: | --- |
| 3s | 895/900 | 891/900 | -0.44 [-1.22, +0.33] |
| 5s | 897/900 | 897/900 | +0.00 [+0.00, +0.00] |
| 10s | 897/900 | 896/900 | -0.11 [-0.33, +0.00] |

At five seconds, both objectives returned 897/900 valid plans with no discordant cells in this sample. This is identical observed acceptance, not demonstrated equivalence or noninferiority. The [0, 0] empirical bootstrap interval occurs because every sampled paired difference is zero; it does not establish zero population difference. Each budget contains nine repeated target-source/mode cells for each of 100 participants. Pooled McNemar p-values have been removed; participant-cluster intervals and the 27 within-cell exact tests with their Holm adjustments are retained.

At ten seconds, the target-aware solver recorded status 0 for 501/900 tasks under a configured 1% relative-gap tolerance. Describe this as solver termination within that tolerance, not mathematical proof of an exact integer optimum. The three common failures per budget are solver-reported infeasibilities; additional no-plan timeouts exist at three seconds and for one target-aware task at ten seconds. Their cause must not be generalized as proven extreme-calorie infeasibility without further evidence.

At ten seconds, recorded mean runtime was approximately 7.214 seconds for target-aware and 9.091 seconds for current-objective solves. This is an observed timing difference. No experiment established earlier branch-and-bound pruning as its cause. No causal explanation for the runtime difference is claimed.

The corrected statistical verifier reproduced 42 confidence intervals and 27 valid within-cell exact tests and checked all 3,006 input manifest hashes. Three invalid pooled exact tests were withdrawn. This correction reused the existing reconstructed records and prior arithmetic audit; it did not rerun meal optimization or repeat the full basket reconstruction. Existing verifier fixture checks are retained without treating them as new end-to-end corruption tests. Original v2 evidence remains unchanged.

Target-error findings remain conditional on jointly successful matched pairs and use the declared engineering scales. They describe target attainment, not health outcomes. Formal matched comparisons across budgets and between Stage 2 target sources should precede a deployment decision; this correction does not claim those additional analyses have been completed.

Evidence: `Saved Model/stage3_budget_audit_v3/analysis_summary.json`, `verification.json`, `correction_manifest.json`. Original reports are archived in `original_documents/`.
