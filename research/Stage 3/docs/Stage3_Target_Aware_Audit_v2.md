# Stage 3 target-aware experiment: corrected audit v2

Updated 23 November 2026. Existing-record correction; no production changes or solver reruns.

The target-aware experimental optimizer reduced target error among jointly successful pairs, but produced fewer valid plans under the tested three-second budget. Acceptance counts were unchanged by fresh constraint verification. These are exploratory results on the reused 100-participant cohort, which also appeared in development tests; they are not independent validation.

| Target source | A valid plans | B valid plans | A mean target error | B mean target error | Paired error difference, 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| rules_only | 294/300 | 261/300 | 0.9072 | 0.4851 | -0.4221 [-0.4536, -0.3909] |
| minimum_change | 292/300 | 261/300 | 0.9434 | 0.5347 | -0.4086 [-0.4395, -0.3776] |
| model_guided | 294/300 | 260/300 | 0.9199 | 0.5133 | -0.4065 [-0.4378, -0.3761] |

Continuous means and 10,000-resample confidence intervals now both weight participants equally. Within each participant, only jointly successful matched dietary modes contribute. The three within-arm distance contrasts use 259, 257 and 257 paired cells respectively; available participant counts are recorded in analysis_summary.json. Missing plans have no target-error value. Binary endpoints retain all 100 participants and all modes; pooled intervals resample participant clusters. Per-mode exact tests are exploratory with retrospective Holm adjustment across 18 contrasts per endpoint, and intervals remain pointwise. The earlier p < 1e-15 statement is withdrawn.

Among 773 jointly successful pairs, 771 baskets differed and 2 were identical. The old 887/900 statistic conflated meal changes with failed empty outputs.

The archive contains status labels and gaps but omits raw solver codes, messages and incumbent flags. Therefore labels cannot independently prove optimality, lack of any incumbent, or mathematical feasibility for timed-out cases. The 299/300 mathematical-feasibility claim and the untested 5–10-second deployment recommendation are withdrawn. Variant A always ran before B; recorded runtimes are descriptive. The historical protocol was rewritable, so current hash agreement does not retrospectively establish preregistration. Engineering normalization scales are not uniformly official FDA/RDA values.

Fresh replay reconstructed all 900 input fingerprints, checked 1,800 outputs against food quantities, serving/category rules, dietary exclusions, active nutrient coverage/bounds, energy, macro fractions, fiber density and sodium/potassium rules. Target errors were recomputed from catalog totals and upstream targets. Four corruption checks passed. A separate implementation verified 30 confidence intervals, 36 exact paired tests, multiplicity adjustments and summary counts/timings. Food parsing, bound construction and the earlier independent arithmetic checker are shared dependencies; neither production acceptance flags nor saved replay flags determine audit decisions. Original records remained unchanged and no solver was called.

Evidence: `Saved Model/stage3_target_aware_audit_v2/analysis_summary.json`, `recomputed_LOCAL.json`, `replay_verification.json`, `verification.json` and `input_manifest.json`. Historical documents are retained in `original_documents/`; original experimental checkpoints are untouched.

The new objective improves the metric it directly optimizes, conditional on successful plans. That is not proof of clinical benefit or that model-guided targets are superior to simpler targets. Keep this optimizer experimental. A separately frozen budget experiment on an independent sample would be needed to assess longer-runtime trade-offs; none was run here.
