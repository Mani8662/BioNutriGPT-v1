# Stage 1 study specification — version 1

Frozen scope: 2026-11-18. This is a documented prospective work specification after existing results have been inspected, not a preregistration or completed evaluation repair. No model, dataset, thresholds or application behavior were changed in this task.

## Purpose and population

Stage 1 estimates three prevalent-outcome screening probabilities from recorded participant information. It is laboratory-assisted screening: measured HDL and total cholesterol are required inputs. Removing HbA1c from the predictor list does not make the model non-invasive.

The target research setting is the selected NHANES adult analytic sample (age >=18): development 2007–2016, N=23,694; temporal evaluation 2017–March 2020 pre-pandemic, N=7,046. These counts describe the existing analytic files, not all eligible US adults. The source selection script requires complete binary labels, measured HbA1c/HDL/total cholesterol and reliable complete day-one nutrient recall. HbA1c remains in selection even though it is excluded from predictors. Cohort lineage, missing component labels, exclusions and nonresponse handling still need the next audit; these rules are not yet certified as reproducing every saved row. The temporal cohort has already been inspected and must not be called untouched in future analyses.

## Outputs, usefulness and intended actions

| Output | Research role | Intended interpretation and action |
|---|---|---|
| Prevalent diabetes score | Primary endpoint | Research screening estimate; flag for professional review and appropriate diagnostic assessment. No diabetes subtype or future-onset claim. |
| Prevalent CVD score | Primary endpoint | Research estimate of the recorded composite outcome; professional review of history and appropriate assessment, not prediction of a future cardiac event. |
| Obesity score | Secondary exploratory endpoint | Retained to evaluate multi-task learning and reproduce the existing model. When measured height and weight are available, use directly calculated BMI to describe BMI-defined obesity rather than claiming prediction adds clinical value. A missing-anthropometry use case remains untested. |

The primary/secondary designation above is a forward work-plan decision, not a claim that the old analyses were prospectively specified. Outputs are sigmoid model scores in [0,1]; reliable probability interpretation awaits calibration repair. The tensor order remains diabetes, obesity, CVD.

The application may display estimates with their limitations for professional review. No score alone establishes a diagnosis, treatment, referral cutoff or clinically validated meal prescription. Existing score-driven Stage 2 rules are computational research assumptions. Future action-specific thresholds must be chosen using development data; inference currently labels 0.617/0.445/0.603 as thresholds selected from temporal evaluation. Those are exploratory and cannot support unbiased clinical threshold claims. This task records the issue without silently changing deployment.

## Outcome definitions requiring verification

The available temporal-label construction script defines diabetes by HbA1c >=6.5%, fasting glucose >=126 mg/dL or reported diabetes; obesity by measured BMI >=30 kg/m²; CVD by reported heart failure, coronary heart disease, angina, heart attack or stroke. These are descriptions of current code, not new clinical recommendations. Development-cycle consistency and negative-label assignment under incomplete composite ascertainment remain unaudited. The diabetes label does not establish specifically type 2 diabetes.

## Exact predictor contract

The following is extracted from current training-source constants, preserving deployed model order: 5 biomarker, 46 diet, 19 lifestyle and 17 demographic/missingness fields. Diet includes 39 nutrient variables and 7 derived features. Names are model identifiers; units, coding and transformations require the preprocessing audit. The companion JSON records the same order, cohort-file hashes and source hashes.

### BIO_COLS

1. `LBDHDD`
2. `LBXTC`
3. `NON_HDL_CHOL`
4. `CHOL_HDL_RATIO`
5. `NON_HDL_TO_HDL_RATIO`

### DIET_COLS

6. `DR1TKCAL`
7. `DR1TPROT`
8. `DR1TCARB`
9. `DR1TSUGR`
10. `DR1TFIBE`
11. `DR1TTFAT`
12. `DR1TSFAT`
13. `DR1TMFAT`
14. `DR1TPFAT`
15. `DR1TCHOL`
16. `DR1TATOC`
17. `DR1TRET`
18. `DR1TVARA`
19. `DR1TACAR`
20. `DR1TBCAR`
21. `DR1TLYCO`
22. `DR1TLZ`
23. `DR1TVB1`
24. `DR1TVB2`
25. `DR1TNIAC`
26. `DR1TVB6`
27. `DR1TFOLA`
28. `DR1TFA`
29. `DR1TFF`
30. `DR1TFDFE`
31. `DR1TVB12`
32. `DR1TVC`
33. `DR1TVK`
34. `DR1TCALC`
35. `DR1TPHOS`
36. `DR1TMAGN`
37. `DR1TIRON`
38. `DR1TZINC`
39. `DR1TCOPP`
40. `DR1TSODI`
41. `DR1TPOTA`
42. `DR1TSELE`
43. `DR1TCAFF`
44. `DR1TALCO`
45. `PROT_PCT_KCAL`
46. `CARB_PCT_KCAL`
47. `FAT_PCT_KCAL`
48. `FIBER_PER_1000_KCAL`
49. `SUGAR_TO_FIBER_RATIO`
50. `SODIUM_POTASSIUM_RATIO`
51. `SAT_TO_UNSAT_FAT_RATIO`

### LIFE_COLS

52. `SLEEP_HOURS`
53. `SLEEP_DEV_FROM_7_5`
54. `PAD680`
55. `PA_VIG_WORK_MIN_WK`
56. `PA_MOD_WORK_MIN_WK`
57. `PA_WALK_BIKE_MIN_WK`
58. `PA_VIG_REC_MIN_WK`
59. `PA_MOD_REC_MIN_WK`
60. `PA_TOTAL_MET_MIN_WK`
61. `SLQ050`
62. `PAQ605`
63. `PAQ620`
64. `PAQ635`
65. `PAQ650`
66. `PAQ665`
67. `SMOKE_NEVER`
68. `SMOKE_FORMER`
69. `SMOKE_SOMEDAYS`
70. `SMOKE_EVERYDAY`

### DEMO_COLS

71. `RIDAGEYR`
72. `INDFMPIR`
73. `DMDEDUC2`
74. `SEX_FEMALE`
75. `RACE_MEX_AMER`
76. `RACE_OTHER_HISP`
77. `RACE_NH_WHITE`
78. `RACE_NH_BLACK`
79. `RACE_OTHER_MULTI`
80. `MARITAL_MARRIED`
81. `MARITAL_PREV_MARRIED`
82. `MARITAL_NEVER_MARRIED`
83. `INDFMPIR_MISS`
84. `SLEEP_HOURS_MISS`
85. `PAD680_MISS`
86. `PA_TOTAL_MET_MIN_WK_MISS`
87. `OTHER_FEATURE_MISS`

## Boundaries and change control

HbA1c, glucose, BMI, height, weight and diagnosis variables are not in the 87-feature contract. This statement is not a completed indirect-leakage audit. US demographic encodings must not be presented as Indian validation; arbitrary neutral values do not solve population transportability. Stage 1 performance in India remains unestablished. LASI planning profiles cannot validate this predictor without compatible observed inputs.

Any change to population, label definitions, primary outcomes or feature order requires a new specification version with rationale. This freeze does not endorse existing preprocessing or split validity. Next independent task: audit cohort construction and outcome labels, then leakage/preprocessing and recoverable partitions before any model comparisons. No training or GPU run is needed to define this specification.
