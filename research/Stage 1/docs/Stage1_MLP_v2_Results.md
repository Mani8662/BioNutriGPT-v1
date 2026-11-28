# Repaired Stage 1: MLP GPU baseline

Reported 2026-11-19 (Asia/Calcutta). Status: completed and independently verified; four of six planned model families completed. This MLP GPU run was separately authorized after CatBoost.

Four predefined configurations trained shared three-output MLPs on 16,894 participants, with validation on 3,621. Validation-only mean diabetes/CVD AUROC selected hidden layers [256,128], dropout=0.1, learning rate=0.0003 and weight decay=0.0001. This is a LayerNorm/ReLU MLP trained with AdamW and unweighted mean binary cross-entropy. Separate logit recalibrators were fitted on 3,621 calibration participants; temporal evaluation used the common 7,004 participants.

The four configurations stopped after 12, 10, 17 and 12 epochs respectively. The first configuration's best checkpoint was epoch 5. Each checkpoint was chosen on validation performance, not the last epoch or temporal performance. The common 40-epoch cap and seven-epoch patience were unchanged.

## Temporal results

| Outcome | AUROC | Average precision | Raw Brier | Calibrated Brier | Raw calibration slope | Calibrated slope |
|---|---:|---:|---:|---:|---:|---:|
| Diabetes | 0.7996 | 0.4748 | 0.124715 | 0.123189 | 0.8302 | 0.9454 |
| Obesity | 0.7519 | 0.6867 | 0.205542 | 0.203210 | 1.0097 | 1.0885 |
| Cvd | 0.8275 | 0.3816 | 0.088440 | 0.087865 | 0.8832 | 0.9592 |

Calibration reduced Brier error for all three outcomes. Diabetes and CVD slopes moved closer to one; obesity's slope moved from 1.0097 to 1.0885, so not every calibration statistic improved. Calibrated intercepts are 0.0374 diabetes, 0.3059 obesity and -0.0298 CVD. Obesity calibration remains imperfect. AUROC and AP are unchanged by the monotonic recalibrators.

## Four-model comparison so far

| Outcome | Logistic AUROC | Histogram AUROC | CatBoost AUROC | MLP AUROC |
|---|---:|---:|---:|---:|
| Diabetes | 0.7965 | 0.8124 | 0.8138 | 0.7996 |
| Obesity | 0.7445 | 0.7557 | 0.7578 | 0.7519 |
| Cvd | 0.8291 | 0.8252 | 0.8288 | 0.8275 |

The MLP does not have the highest AUROC point estimate for any of the three outcomes. It improves on logistic regression's diabetes/obesity AUROC but falls below both boosted-tree methods for those outcomes. This is a descriptive comparison, not proof that one family is statistically superior. It establishes a simpler neural baseline for the upcoming Transformer runs.

## Compute and verification

All four recorded training/validation/checkpointing intervals total 59.52 seconds on the RTX 3050; full command time includes additional startup, calibration, exports and verification. Warm raw-model GPU inference median: 0.673 ms for one participant and 0.913 ms for a batch of 128, synchronized with CUDA. These timings exclude loading, preprocessing and recalibration. Classical models' latency was measured on CPU, so device differences must remain explicit.

Independent verification passed: trial/final artifact hashes, current code/data/environment bindings, exact partitions, validation-only selection, all exported IDs/labels, checkpoint/calibrator prediction reproduction and metric recomputation. Artifacts are under Saved Model/stage1_comparison_v2/mlp; verification is Research Paper/Stage1_mlp_v2_Verification.json. Local prediction files remain outside the submission archive.

## Next task and limits

Next independent model: BioNutriGPT Transformer on GPU, pending approval. Reference FT-Transformer remains after that. All current results use one seed and four configurations, unweighted selected participants and an already-inspected temporal period. Paired confidence intervals, seed variability, architectural ablations and clinical utility are pending. The fixed 0.5 threshold remains descriptive, not a justified referral rule. Production and manuscript/submission ZIP remain unchanged until full reconciliation.
