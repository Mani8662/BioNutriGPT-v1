# Implementation verification evidence index — 2026-11-20

This update separates repaired Stage1 research results from legacy production inference and version-bound historical downstream experiments. Targeted regression tests are not pooled into research success rates. The simple browser demonstration was interrupted and is not recorded as completed.

| Evidence | Location relative to Research Paper | Scope |
|---|---|---|
| Repaired cohort | Stage1_Repair_v2_Report.md | 16,894 / 3,621 / 3,621 / 7,004 participants |
| Six-family comparison | Stage1_Comparison_v2_Current_Results.md | Repaired research artifacts; not deployed |
| Paired uncertainty | Stage1_Paired_Uncertainty_v2/Report.md | Conditional participant uncertainty |
| Five-seed architecture ablations | Stage1_Ablation_v2_Results.md | 25 runs; no adjusted primary architecture advantage established |
| Latest chatbot regression | ../Saved Model/stage4_understanding_v7/Report.md | 62 automated checks; 5 scenarios/8 turns, 3 GPU calls; targeted development evidence |
| Stage2 audit | ../Saved Model/stage2_current_audit_v1/Report.md | 39 rules; 216 boundary combinations; historical pre-repair findings |
| Input/rule repair | ../Saved Model/stage2_contract_repair_v1/verification.json | Before/after engineering evidence |
| Consistent-input stability pilot | ../Saved Model/stage2_stability_v2/Report.md | One fixture, six runs; legacy model |
| Ratio/model identity contract | ../Saved Model/stage2_ratio_contract_v3/Report.md | Two GPU runs and source binding |
| Cross-stage agreement | ../Saved Model/stage12_agreement_v1/Report.md | Two cases, exact baseline probability agreement |

These local references are an evidence map, not a public data-release package. Do not release participant-level artifacts without reviewing data terms. Full multi-profile counterfactual stability, guideline-only comparison, current-catalog matched optimizer benchmark, reserved dialogues/tool-parity comparison and clinical/human assessment remain incomplete.

## Submission artifact status

BioNutriGPT_Manuscript_v2.md was selectively reconciled with verified results. Its backup preserves the pre-update text. The existing Submission_Pack, figures, Word/PDF derivatives, evidence_snapshot.json and BioNutriGPT_Submission_Pack_DRAFT.zip were not regenerated and may contain older values. They must not be submitted as synchronized current outputs. The root manuscript outside this folder was not changed in this folder-scoped update. Before submission, reconcile all derived tables, reporting maps and metadata, then rebuild and verify the package.
