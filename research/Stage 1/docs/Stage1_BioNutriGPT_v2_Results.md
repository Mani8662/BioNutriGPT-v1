# Repaired Stage 1: BioNutriGPT GPU results

Reported 2026-11-19 (Asia/Calcutta). All four configurations completed and independent artifact/prediction verification passed. Five of six model families are now complete.

The common 87 predictors and repaired partitions were used: 16,894 training, 3,621 validation, 3,621 calibration and 7,004 temporal participants. Validation-only mean diabetes/CVD AUROC selected trial 03 (learning rate 0.0003, weight decay 0.001), best checkpoint epoch 4. Configurations stopped after 17, 11, 21 and 11 epochs. The model has 1,295,623 parameters; seed 42, four configurations, 40-epoch cap and seven-epoch patience were retained.

## Temporal results

| Outcome | AUROC | Average precision | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|
| DIABETES | 0.8107 | 0.4858 | 0.121065 | 0.120914 |
| OBESITY | 0.7467 | 0.6823 | 0.204283 | 0.204624 |
| CVD | 0.8288 | 0.3947 | 0.087336 | 0.087353 |

Recalibration reduced diabetes Brier error, but slightly increased obesity and CVD Brier error. Both variants remain reported; no temporal-result-based choice of calibrator was made. Calibrated slopes are 1.0030, 1.0506 and 0.9321 for diabetes, obesity and CVD, respectively; corresponding intercepts are 0.0544, 0.2569 and -0.0989. Obesity calibration remains imperfect.

BioNutriGPT does not lead on AUROC among the five completed families. CatBoost leads diabetes/obesity and logistic regression leads CVD by point estimate. BioNutriGPT has the lowest CVD Brier error in both raw and calibrated comparisons. This does not establish statistical superiority or justify architectural complexity by itself. Paired confidence intervals, multiple seeds, ablations and clinical utility remain pending.

## Compute and verification

Recorded training/validation/checkpointing intervals total 650.62 seconds on the RTX 3050. This excludes interrupted attempts, startup, calibration, exports and verification; it is not total elapsed project time. Interrupted logs were retained and completed trials reused. Warm synchronized raw-model GPU prediction medians are 17.032 ms for one participant and 20.796 ms for a batch of 128 (20 repetitions). Loading, preprocessing and recalibration are excluded. CPU and GPU latency measurements must retain their device labels.

Verification passed for final/trial hashes, current source/data/environment bindings, frozen partitions and schema, validation-only selection, all exported participant IDs/labels, reloaded model/calibrator predictions and recomputed AUROC/AP/Brier/log-loss. Evidence: Stage1_bionutrigpt_v2_Verification.json and Saved Model/stage1_comparison_v2/bionutrigpt/completed.json.

These are unweighted single-seed results on selected participants from an already-inspected historical temporal period. They do not provide Indian external validation or establish a clinical referral threshold. Reference FT-Transformer is the next independent run, requiring user approval. Production weights and manuscript/submission performance remain unchanged until reconciliation.
