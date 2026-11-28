# Stage 2 stability retrospective audit (24 November 2026)

The 800-record stability study passed saved numerical checks and identical-input repeats matched in this environment. However, target sensitivity remains: the largest single-nutrient differences in each perturbation scenario occurred without routing crossings or solver fallback and were already present in the saved post-Adam, box-clipped vectors. Projection was not their primary origin. Exact gradient-level causes require a separate diagnostic experiment; saved records do not include optimizer trajectories. Fixed research routing cutoffs are 0.5 for all outcomes; 39 nutrient targets enter displacement metrics. These are not diagnostic thresholds or clinical benefit estimates.

## Largest single-nutrient displacement per scenario

| Scenario | Nutrient | Baseline → perturbed | Scaled max displacement | L1 before projection → final | Routing crossing / fallback |
|---|---|---|---|---|---|
| diet_minus1 | DR1TVB2 | 2.7169 → 1.2920 | 1.0961 | 1.3271 → 1.3289 | False / False |
| diet_minus5 | DR1TVB2 | 2.7169 → 1.1075 | 1.2380 | 2.4513 → 2.4607 | False / False |
| diet_plus1 | DR1TVB12 | 5.3191 → 2.8101 | 1.0454 | 1.2612 → 1.2638 | False / False |
| diet_plus5 | DR1TCOPP | 2.4112 → 1.1102 | 1.4456 | 3.7089 → 3.7089 | False / False |
| lipid_minus5 | DR1TCOPP | 2.0850 → 0.9680 | 1.2411 | 1.3850 → 1.3805 | False / False |
| lipid_plus5 | DR1TVB2 | 2.9261 → 1.1091 | 1.3977 | 2.3808 → 2.3803 | False / False |

All 600 nonbaseline/nonrepeat comparisons were inspected. No baseline/perturbed pair reported a projection fallback. Largest single-nutrient cases above had no crossing. This does not mean projection never affects other cases. L1 stage differences locate the discrepancy before projection but do not prove a specific optimization mechanism. Post-Adam records are already box-clipped; separate Adam and clipping effects cannot be isolated here.

Local participant identifiers and full comparison diagnostics remain in Saved Model/stage2_stability_repaired_v1/retrospective_audit/case_diagnostics_LOCAL.json. No identifiers are included in the table above.

Next experiment to prepare: instrument a small diagnostic subset to capture pre-clipping Adam trajectories, gradients, loss components and any routing/gating changes inside the predictive model. Compare prespecified scale-normalized update/penalty alternatives only after locating the mechanism; do not tune on these 100 evaluation participants. Gemini should run any new experiments.

Original study JSON, frozen code and verification receipts were preserved. Corrections affect narrative documents only. Existing DOCX/PDF submission exports were not regenerated and remain older drafts.