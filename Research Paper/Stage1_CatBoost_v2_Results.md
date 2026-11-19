# Repaired Stage 1: CatBoost GPU baseline

Reported 2026-11-19 (Asia/Calcutta). Status: completed and independently verified; three of six planned families are complete. This GPU run was explicitly approved as an independent task.

Four predefined configurations each fitted three binary heads using 16,894 training participants. The selected configuration was depth=4, learning_rate=0.03, l2_leaf_reg=3.0, with a maximum of 600 iterations and validation-based early stopping. Selection used mean diabetes/CVD validation AUROC on 3,621 participants. Prespecified logit recalibration then used a separate 3,621-person calibration set. Temporal evaluation used the same 7,004 participants as the two earlier repaired baselines.

## Temporal results

| Outcome | AUROC | Average precision | Raw Brier | Calibrated Brier | Raw calibration slope | Calibrated slope |
|---|---:|---:|---:|---:|---:|---:|
| Diabetes | 0.8138 | 0.4918 | 0.120351 | 0.120133 | 1.0331 | 0.9800 |
| Obesity | 0.7578 | 0.6856 | 0.201964 | 0.200574 | 1.1675 | 1.0315 |
| Cvd | 0.8288 | 0.3853 | 0.087834 | 0.087991 | 1.0559 | 0.9234 |

Calibration reduces Brier error for diabetes and obesity, but increases CVD Brier from 0.087834 to 0.087991 and slightly increases CVD log-loss. The prespecified calibrated result is retained alongside the raw result; no post-test choice of whichever version performs better is made. Positive recalibration slopes preserve AUROC/AP.

## Three-model comparison so far

| Outcome | Logistic AUROC | Histogram boosting AUROC | CatBoost AUROC |
|---|---:|---:|---:|
| Diabetes | 0.7965 | 0.8124 | 0.8138 |
| Obesity | 0.7445 | 0.7557 | 0.7578 |
| Cvd | 0.8291 | 0.8252 | 0.8288 |

CatBoost has the highest diabetes and obesity AUROC point estimates among these three models. Logistic regression is slightly higher for CVD. This does not establish statistical or practical superiority: paired confidence intervals, multi-seed stability and the remaining three neural families are pending. CatBoost does not lead every metric; histogram boosting has higher obesity average precision, and logistic regression has lower calibrated CVD Brier.

The fixed descriptive 0.5 calibrated threshold yields sensitivity of 23.6% diabetes, 50.6% obesity and 17.0% CVD. It is not a validated screening threshold. AUROC is not percent accuracy.

## Compute and verification

Training ran on the RTX 3050 Laptop GPU; a live GPU sample showed 3,207 MiB used and 35% utilization. CatBoost printed that AUC evaluation is not implemented on GPU; this concerns metric calculation, not the training device. The four recorded training/validation/checkpointing intervals total 111.06 seconds. This is not the full command duration and excludes later calibration/export/verification.

Warm raw prediction was measured on CPU, as specified: median 3.269 ms for one participant and 4.196 ms for a batch of 128. These values exclude loading, preprocessing and recalibration and must not be described as GPU inference latency.

Independent verification passed for completion/trial hashes, current data/source/environment bindings, frozen split membership, model selection from validation predictions, all exported IDs/labels, checkpoint/calibrator prediction reproduction and AUROC/AP/Brier/log-loss recomputation. GPU fitting can vary slightly across repeated runs despite the fixed seed.

Artifacts: Saved Model/stage1_comparison_v2/catboost. Verification: Research Paper/Stage1_catboost_v2_Verification.json. Local participant-level predictions remain outside the submission archive.

## Next task and limitations

Next independent model: MLP on GPU, pending approval; BioNutriGPT and reference FT-Transformer follow separately. All current results are unweighted selected-sample estimates from one seed and four configurations. The temporal period was previously inspected. Clinical utility, Indian validation, survey uncertainty and robustness are not established. Legacy and repaired endpoints/cohorts differ, so their scores are not a controlled before/after comparison. Production and manuscript/submission ZIP remain unchanged pending full reconciliation.
