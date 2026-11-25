# Stage 1 survey reference verification and subgroup evaluation

Completed 2026-11-20 using R survey 4.5. All 54 overall estimates/SEs/intervals and 45 overall paired contrasts matched the earlier Python implementation within 1e-10. Independently recomputed all 594 overall/subgroup point estimates using Python sklearn and weighted Brier arithmetic.

Design: WTDRD1PP, 24 masked strata, 49 masked PSUs, JKn with mse=TRUE and 25 design degrees of freedom. Full design is retained for domain estimates using zero domain weights outside the subgroup. All three outcomes, six frozen models, 10 demographic groups plus overall: 594 metric rows and 495 paired contrasts. Metrics are AUROC, Brier score and average precision on calibrated probabilities. No fitting or calibration updates.

| Group | Outcome | BioNutriGPT AUROC | Pointwise 95% interval |
|---|---|---:|---|
| Overall | TARGET_DIABETES | 0.8402 | [0.8203, 0.8601] |
| Overall | TARGET_CVD | 0.8468 | [0.8281, 0.8656] |
| Overall | TARGET_OBESITY | 0.7330 | [0.7077, 0.7583] |
| Female | TARGET_DIABETES | 0.8355 | [0.8068, 0.8643] |
| Female | TARGET_CVD | 0.8245 | [0.7970, 0.8519] |
| Female | TARGET_OBESITY | 0.7523 | [0.7223, 0.7822] |
| Male | TARGET_DIABETES | 0.8432 | [0.8101, 0.8763] |
| Male | TARGET_CVD | 0.8629 | [0.8427, 0.8832] |
| Male | TARGET_OBESITY | 0.7157 | [0.6825, 0.7488] |
| Age20_40 | TARGET_DIABETES | 0.7788 | [0.6837, 0.8738] |
| Age20_40 | TARGET_CVD | 0.8081 | [0.7169, 0.8992] |
| Age20_40 | TARGET_OBESITY | 0.7725 | [0.7403, 0.8046] |
| Age41_65 | TARGET_DIABETES | 0.7790 | [0.7506, 0.8074] |
| Age41_65 | TARGET_CVD | 0.7622 | [0.6966, 0.8278] |
| Age41_65 | TARGET_OBESITY | 0.7156 | [0.6830, 0.7483] |
| AgeOver65 | TARGET_DIABETES | 0.7661 | [0.7368, 0.7955] |
| AgeOver65 | TARGET_CVD | 0.7079 | [0.6653, 0.7504] |
| AgeOver65 | TARGET_OBESITY | 0.7270 | [0.6806, 0.7734] |
| RIDRETH1_1 | TARGET_DIABETES | 0.8290 | [0.7987, 0.8593] |
| RIDRETH1_1 | TARGET_CVD | 0.9067 | [0.8653, 0.9481] |
| RIDRETH1_1 | TARGET_OBESITY | 0.6996 | [0.6421, 0.7570] |
| RIDRETH1_2 | TARGET_DIABETES | 0.8420 | [0.7943, 0.8896] |
| RIDRETH1_2 | TARGET_CVD | 0.8695 | [0.8268, 0.9122] |
| RIDRETH1_2 | TARGET_OBESITY | 0.6577 | [0.5656, 0.7498] |
| RIDRETH1_3 | TARGET_DIABETES | 0.8534 | [0.8260, 0.8807] |
| RIDRETH1_3 | TARGET_CVD | 0.8357 | [0.8086, 0.8628] |
| RIDRETH1_3 | TARGET_OBESITY | 0.7408 | [0.7051, 0.7766] |
| RIDRETH1_4 | TARGET_DIABETES | 0.8010 | [0.7679, 0.8342] |
| RIDRETH1_4 | TARGET_CVD | 0.8045 | [0.7470, 0.8620] |
| RIDRETH1_4 | TARGET_OBESITY | 0.7365 | [0.7113, 0.7617] |
| RIDRETH1_5 | TARGET_DIABETES | 0.8279 | [0.7972, 0.8586] |
| RIDRETH1_5 | TARGET_CVD | 0.8835 | [0.8367, 0.9302] |
| RIDRETH1_5 | TARGET_OBESITY | 0.7418 | [0.6835, 0.8000] |

RIDRETH1 codes: 1 Mexican American, 2 Other Hispanic, 3 Non-Hispanic White, 4 Non-Hispanic Black, 5 Other Race including Multi-Racial. Obesity and subgroup contrasts are exploratory. Intervals are unadjusted for multiple comparisons and untruncated; no overall superiority, equivalence or fairness claim follows. Complete-case/laboratory/recall selection remains a limitation despite weighting. This is not Indian external validation or clinical-benefit evidence. Weighted calibration curves/slopes, threshold clinical utility and training-variation uncertainty are outside this run.

Verification covers overall uncertainty via independent package implementation and subgroup point estimates independently; subgroup uncertainty is produced by R survey rather than independently replicated by a third implementation. All estimates/intervals were finite. R startup emitted locale-setting warnings, with successful numerical cross-checks. Initial ID type comparison was repaired before successful evaluation; a slow initial calculation was restarted with equivalent tied-score aggregation. No prior result files were replaced.

Sources: [R survey replicate design documentation](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/as.svrepdesign.html), [CDC dietary day-one weight guidance](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DR1TOT.htm).
