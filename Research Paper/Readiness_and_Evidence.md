STAGE 1 COMPARISON UPDATE — 2026-11-19
Logistic regression, histogram boosting, CatBoost and GPU MLP completed and verified (4/6 model families). MLP temporal AUROC on N=7,004: 0.7996 diabetes, 0.7519 obesity, 0.8275 CVD. CatBoost retains highest diabetes/obesity AUROC point estimates so far; logistic is slightly higher for CVD. MLP recalibration lowers Brier error for all outcomes, without improving every calibration statistic. Paired uncertainty and two Transformer runs remain pending. See Research Paper/Stage1_Comparison_v2_Current_Results.md. BioNutriGPT GPU is next, pending approval. Production remains unchanged.

STAGE 1 RESEARCH REPAIR v2 — 2026-11-18
The isolated repaired dataset/preprocessor is ready: 16,894 training, 3,621 validation, 3,621 calibration and 7,004 temporal participants. It uses harmonized labels and training-only preprocessing. Repaired logistic-regression, histogram-boosting, CatBoost and MLP baselines are now trained and verified; BioNutriGPT and reference FT-Transformer remain pending. Production and manuscript performance remain legacy. See Research Paper/Stage1_Repair_v2_Report.md. Older audit findings below describe legacy artifacts and remaining limitations.

LATEST STAGE 1 AUDIT — 2026-11-18
Stage 1 foundation audit found inconsistent development/test disease definitions, internal preprocessing leakage and calibration overlap. Existing Stage 1 results remain historical pending repair. See Research Paper/Stage1_Foundation_Audit/Stage1_Foundation_Audit.md. This audit did not change architecture, data or weights.

# BioNutriGPT current evidence and submission readiness

Updated 2026-11-18 after verification of the checkpointed v4 run. The current working manuscript is BioNutriGPT_Manuscript_v2.md; the root BioNutriGPT_Research_Manuscript.md is synchronized. A journal and author details remain unconfirmed.

## Completed evidence

| Area | Evidence | Interpretation |
|---|---|---|
| Stage 1 | Saved NHANES temporal results, N=7,046 | Unweighted AUROC 0.7999 diabetes, 0.7518 obesity, 0.8260 CVD; no overall Transformer advantage established |
| Food data | 49 records: 20 INDB variants + 29 USDA proxies | 43 active-complete before mode filters; incomplete records excluded consistently in v4 |
| LASI optimizer pilot | 100 unique selected profiles, 300 mode-specific solves | Acceptance 91/100, 96/100 and 98/100; unweighted selected-profile evidence |
| Final v4 dialogue study | 36 distinct people, four disease strata; 78 turns per guarded variant | Full system passes 76/78; 29 validated revisions, 11 justified refusals, two timeout failures |
| Matched v4 comparison | 12 participants, one per mode-by-stratum cell; 24 turns per system | Guarded 24/24 versus Qwen 2/24; feasible-request valid proposals 9/9 versus 0/9 |
| Same-state change-penalty ablation | 25 paired successful revisions | Mean change 7.72 versus 16.56, 53.38% lower; paired difference 8.84, bootstrap 95% interval 6.72–11.12 |
| Reproducibility checks | All 24 prompts, journals, four stage checkpoints and source hashes verified | Full raw records remain local; aggregate evidence is packaged |
| Software tests | Eight benchmark-correction checks passed | Earlier application suite rerun was 10/11; this separate contract issue is not silently marked fixed |

## What remains

1. Repair Stage 1 calibration splits, matched comparator tuning/training budgets and survey-design uncertainty. Run paired model comparisons, ablations and meaningful robustness/subgroup analyses. LASI meal planning does not externally validate Stage 1.
2. Complete the current 49-entry continuous-versus-integer optimizer comparison under identical active rules, objectives and limits. The older 29-entry seven-nutrient comparison remains historical.
3. Broaden dialogue testing with an independently reserved test set, more paraphrases and stronger tool-using comparators where appropriate. The present cases were part of a development/correction sequence and are not untouched prospective validation.
4. Add memory/independent-validator ablations if making specific claims for those components. The change-penalty ablation is now completed.
5. Reconcile the outstanding application regression expectation about an explicit allergy declaration ending in a question mark. No application-code repair was part of the benchmark run.
6. Review rule applicability, food proxies, serving realism and allergen data. Human acceptability and clinical effectiveness require their own evidence.
7. Confirm authorship, affiliations, funding, conflicts, institutional ethics wording, data terms, public release and journal-specific requirements. Reporting maps remain working drafts rather than completed official checklists.

## Boundaries

The LASI sample is deliberately balanced, unweighted and selected for initial planning feasibility. Stage 2 uses assumed 0.85/0.10 status-routing scores and Na/K baselines. This is downstream computational testing, not Stage 1 Indian prediction validation or the full Stage 2 counterfactual search. Protocol success includes correct refusals; low violation counts alone do not establish request fulfillment. Bootstrap and Wilson intervals are conditional working analyses, not national-population uncertainty. The tested comparator is a specific quantized Qwen 3B configuration; broader LLM superiority is not established.

Historical reports and the failed-export attempt are preserved separately. The authoritative current results are Saved Model/stage4_lasi_v4_verified_summary.json and the detailed v4 report. Clinical benefit, allergen-free certification and journal-tier guarantees are not claimed.
