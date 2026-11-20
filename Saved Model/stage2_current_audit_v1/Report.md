# Current Stage 2 audit — v1

Date: 2026-11-20. Scope: current-code rule provenance, function-level boundary checks, metadata validation and local threshold sensitivity. No training, Stage 1 inference, counterfactual search or real-food feasibility study was performed. Production rules and saved historical evidence were not modified. Clinical source content was not independently revalidated in this step.

## Findings and repair order

1. **Population contract is not enforced at the Stage 2 boundary.** The metadata validator accepts age 0 and 12 despite adult-only rule metadata. Unknown keys `renal_disease=True` and `pregnant=True` are discarded by this function. This demonstrates lack of handling at this boundary, not proof that every frontend route accepts these cases. Define a shared eligibility contract and unknown/unsupported status before claiming general clinical applicability. Do not treat absent information as confirmed absence.
2. **Cross-rule protein conflict.** All 216 age/sex/weight/calorie combinations returned finite ordered intervals, but 27 have an envelope protein minimum above the later filter's 2 g/kg maximum. All are the deliberately extreme 25 kg male cases (9 ages x 3 calorie targets): minimum 56 g versus maximum 50 g. The grid is a stress test, not prevalence evidence. Resolve or explicitly reject contradictory prescriptions; do not silently lower requirements or override limits. These checks do not show an invalid final plan was certified.
3. **Threshold discontinuity.** At fixed 2,000 kcal, diabetes probability changes from 0.616999 to 0.617001 reduce carbohydrate upper bound 300 to 225 g and increase fibre minimum 28 to 35 g. At the CVD cutoff 0.603, sodium upper bound changes 2,300 to 1,500 mg, potassium minimum 2,600 to 3,500 mg, cholesterol ceiling 300 to 150 mg, and saturated-fat upper bound 22.22 to 13.33 g. These are project routing assumptions, not validated therapeutic thresholds. Report-only sugar/exposure fields are distinguished in rule metadata.
4. **Energy sensitivity.** The production energy formula for the explicit 70 kg, 170 cm, 42-year male sedentary example changes from 1,869 to 1,369 kcal across obesity cutoff 0.445. This is a formula-level reproduction, not a generated prescription. Compare routing-policy alternatives and downstream feasibility before choosing a replacement policy.
5. **Override validation gap.** Contradictory and nonfinite micronutrient overrides are rejected; a negative B12 lower bound is accepted. Reject negative nutrient domains explicitly and test this at the artifact boundary.
6. **Documentation drift.** Historical stage2_rule_audit reports v3; current policy is adult-food-rule-review-v4-exposure-separated. Current 39-rule table is captured here without overwriting historical results.
7. **Model integration remains legacy.** Stage 2 defaults to Saved Model/bionutrigpt_stage1_best.pt and hardcoded cutoffs 0.617/0.445/0.603. This audit did not establish equivalence to repaired v2 comparison artifacts. Bind checkpoint, preprocessing, calibration and threshold provenance explicitly before integrated claims.
8. **Missing-input behavior.** Missing weight, nonfinite height, negative weight and unknown activity are rejected in the metadata function. Missing sex is inferred from the input feature vector. The CLI separately supplies 75 kg/170 cm defaults; document or remove silent assumptions before claiming user-specific anthropometric prescription.

## Next experiments

First repair the input/eligibility and contradictory-bound contracts, retaining before/after evidence. Then run full counterfactual stability checks, immutable-feature tests, guideline-only comparator and current-catalog feasibility under baseline/perturbed scores. These require a frozen model-artifact selection; do not substitute invented LASI predictors. Food-proxy sensitivity and matched optimizer comparisons remain separate unfinished tasks. Threshold jumps alone do not justify inventing smoother clinical rules.

## Current rule inventory

Values below are evaluated for age 42, male, 70 kg, 2,000 kcal and all three risk inputs 0.1. They are not universal targets. The source URLs and source-support qualifiers are inherited from code, not newly verified clinical guidance.

| Nutrient | Unit | Lower | Upper | Role | Stage 3 enforcement |
|---|---|---:|---:|---|---|
| DR1TKCAL | kcal | 1900 | 2100 | project_assumption | required_for_acceptance |
| DR1TPROT | g | 56 | 140 | project_assumption | required_for_acceptance |
| DR1TCARB | g | 130 | 300 | project_assumption | required_for_acceptance |
| DR1TSUGR | g | 0 | 300 | report_only | none |
| DR1TFIBE | g | 28 | 60 | project_assumption | required_for_acceptance |
| DR1TTFAT | g | 44.4444 | 77.7778 | project_assumption | required_for_acceptance |
| DR1TSFAT | g | 0 | 22.2222 | project_assumption | required_for_acceptance |
| DR1TMFAT | g | 0 | 77.7778 | reference_interval | required_for_acceptance |
| DR1TPFAT | g | 11.1111 | 22.2222 | project_assumption | required_for_acceptance |
| DR1TCHOL | mg | 0 | 300 | project_assumption | required_for_acceptance |
| DR1TATOC | mg | 15 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TRET | ug | 0 | 3000 | reference_interval | required_for_acceptance |
| DR1TVARA | ug | 900 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TACAR | ug | 0 | 1e+06 | report_only | none |
| DR1TBCAR | ug | 0 | 1e+06 | report_only | none |
| DR1TLYCO | ug | 0 | 1e+06 | report_only | none |
| DR1TLZ | ug | 0 | 1e+06 | report_only | none |
| DR1TVB1 | mg | 1.2 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TVB2 | mg | 1.3 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TNIAC | mg | 16 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TVB6 | mg | 1.3 | 100 | reference_interval | required_for_acceptance |
| DR1TFOLA | ug | 0 | 1e+06 | report_only | none |
| DR1TFA | ug | 0 | 1000 | reference_interval | required_for_acceptance |
| DR1TFF | ug | 0 | 1e+06 | report_only | none |
| DR1TFDFE | ug | 400 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TVB12 | ug | 2.4 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TVC | mg | 90 | 2000 | reference_interval | required_for_acceptance |
| DR1TVK | ug | 120 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TCALC | mg | 1000 | 2500 | reference_interval | required_for_acceptance |
| DR1TPHOS | mg | 700 | 4000 | reference_interval | required_for_acceptance |
| DR1TMAGN | mg | 420 | 1e+06 | reference_interval | required_for_acceptance |
| DR1TIRON | mg | 8 | 45 | reference_interval | required_for_acceptance |
| DR1TZINC | mg | 11 | 40 | reference_interval | required_for_acceptance |
| DR1TCOPP | mg | 0.9 | 10 | reference_interval | required_for_acceptance |
| DR1TSODI | mg | 500 | 2300 | project_assumption | required_for_acceptance |
| DR1TPOTA | mg | 2600 | 4700 | project_assumption | required_for_acceptance |
| DR1TSELE | ug | 55 | 400 | reference_interval | required_for_acceptance |
| DR1TCAFF | mg | 0 | 400 | exposure_reporting | none |
| DR1TALCO | g | 0 | 14 | exposure_reporting | none |

Artifacts: current_rule_table.json includes source/applicability metadata; boundary_grid.json contains every case; threshold_sensitivity.json, metadata_checks.json and override_checks.json retain measured outputs; source_hashes.json binds implementation. Reproduce in a fresh output directory using audit_stage2_current_v1.py; existing evidence is protected against overwrite.
