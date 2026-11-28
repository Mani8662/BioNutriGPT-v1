# Stage 1 version 2 data and preprocessing repair

Status: completed data preparation and verification; no model training, calibration fitting or new performance evaluation. Production still uses the legacy checkpoint and preprocessing. New artifacts are isolated under `Data/stage1_repaired_v2` and refuse overwrite on rerun.

## Changes and rationale

1. Primary diabetes phenotype: reported diabetes (DIQ010=1) or HbA1c >=6.5%. A negative requires measured HbA1c below threshold and DIQ010=2 or 3. Refused, unknown and missing responses cannot establish a negative. Borderline/prediabetes (code 3) is distinguished from diabetes; this is not a claim of normal metabolism. A stricter borderline-exclusion sensitivity is recorded.
2. Glucose is excluded from the primary phenotype to avoid differential ascertainment from irregular fasting-subsample availability. It remains in separately labelled sensitivity counts. This is an explicit endpoint change from the old temporal definition, made before new training; old/new AUROC results are not directly comparable. In the repaired temporal cohort, adding available glucose changes 99 labels to positive. Fasting duration is absent from the available 2015–2016 and 2017–2020 merged rows, so this sensitivity cannot establish fully verified fasting ascertainment. Requiring all negative components including glucose leaves 2,885 temporal participants unclassified.
3. CVD uses heart failure, coronary heart disease, angina, heart attack and stroke. Any reported positive is sufficient; all five must be explicitly negative for a negative label. Unknown components remain unknown.
4. The complete-outcome joint study now explicitly targets adults age >=20, matching the CVD questionnaire scope. BMI-defined obesity remains a secondary exploratory outcome. Outcome construction is independent of legacy target columns.
5. Development is split 70/15/15, stratified on the three-label pattern, with seed 42. Preprocessing fits only training rows. Validation is for selection/early stopping; calibration is reserved for later prespecified recalibration. No new clinical thresholds are chosen.

These are operational research phenotypes. A single HbA1c measurement is not a confirmed clinical diagnosis, and the diabetes outcome does not distinguish type 1 from type 2.

## Frozen partitions

| Partition | Participants | Diabetes positive | Obesity positive | CVD positive |
|---|---:|---:|---:|---:|
| Training | 16,894 | 2,616 | 6,427 | 1,773 |
| Validation | 3,621 | 562 | 1,377 | 380 |
| Calibration | 3,621 | 561 | 1,377 | 379 |
| Temporal evaluation | 7,004 | 1,301 | 3,057 | 845 |

Development total: 24,136. Rebuilding from available merged raw data admits 539 previously excluded development participants and excludes 97 previously included participants. Among the 23,597 shared development participants, 622 diabetes and 225 CVD labels change; obesity labels do not change. Temporal selection excludes 42 previously included participants and admits none; 99 diabetes labels change among retained participants. These numbers differ from the initial audit's 910 diabetes disagreements because the repaired primary endpoint deliberately excludes glucose and the cohort/missingness rules changed.

## Evidence and checks

- Eight focused tests pass: missing negatives, positive overrides, controlled diagnosed diabetes, borderline sensitivity, glucose scope, invalid measurements, independence from legacy labels and reproducible disjoint splits.
- All partitions have unique IDs with zero pairwise overlap.
- All 87 predictors are finite and stored in exact model order. Outcome-defining HbA1c/glucose/BMI/diagnosis fields are excluded from model inputs.
- The scaler saw exactly 16,894 training rows. Medians, modes, skew-selected transforms and clipping bounds were also fitted on those rows only.
- Reloading the saved preprocessing artifact reproduces every partition exactly in memory. Altering a reserved-set input does not mutate the fitted preprocessing state.
- Independent persisted-file verification checked every primary label, negative ascertainment, age eligibility, feature order and output hash.
- Hashes confirm legacy data, weights and preprocessing were unchanged.

The protocol, source hashes, environment versions, output hashes, partition membership, clean/ML-ready CSVs, cohort flow, outcome counts and survey metadata are saved. Survey fields are outside predictors; storing them does not resolve the correct analysis weights. Sensitivity counts concern the retained primary cohort and do not constitute a separately trained sensitivity model or quantify all excluded-profile re-entry.

## Required next steps

1. Build a shared training/evaluation runner that consumes these explicit partitions and refuses mismatched feature order/artifact versions. Do not use the old training script unchanged: it expects old counts and performs its own split.
2. Freeze model/tuning budgets, primary comparisons, calibration method and metric definitions before running new models. Fit preprocessing inside any additional inner cross-validation folds.
3. Train new models using GPU where supported and available CPU parallelism otherwise. Obtain the user's approval before this independent long run.
4. Fit prespecified calibrators on the reserved calibration set and evaluate on the temporal cohort without tuning on its outcomes. It remains an already-inspected historical period, not untouched prospective confirmation.
5. Complete form/TXT preprocessing parity and only deploy a newly trained checkpoint with its matching versioned artifact. Current production is unchanged.

## Remaining scope limitations

The source development assembly file remains unavailable, so lineage begins at the merged raw file. This is not a reconstruction of all NHANES-eligible adults. Complete-outcome, complete-recall and laboratory selection can bias the analytic sample. Survey weighting, cycle-specific assay comparability, medication and pregnancy applicability, and independent Indian prediction evaluation remain pending. No claim of improved performance, clinical benefit or journal readiness follows from this data repair alone. Existing manuscript results and submission ZIP remain historical and predate the repaired evaluation.

## Sources and access

- [NHANES diabetes codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DIQ.htm)
- [NHANES CVD questionnaire codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_MCQ.htm)
- [NHANES glucose documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GLU.htm)
- [NIDDK HbA1c interpretation and confirmation](https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test)

This version is a forward repair after legacy results were inspected, not a preregistration. Participant membership and raw metadata are retained locally and have not been copied into the submission archive.
