# Stage 1 comparison and calibration — version 2

Status updated 2026-11-19: all six model families are completed and verified, including reference FT-Transformer. Paired participant uncertainty analysis is complete and verified; see Stage1_Paired_Uncertainty_v2/Report.md. Repeated-seed and ablation studies remain pending. See Stage1_Comparison_v2_Current_Results.md and the individual model result reports. Nine synthetic-only verification tests passed, including CPU/GPU training loops, checkpoint reload, completed-run reuse, schema/ID rejection, calibration isolation and end-to-end exports. These are software checks, not evidence of model performance.

## Fixed design

All six families consume the repaired 16,894 training, 3,621 validation, 3,621 calibration and 7,004 temporal participants with identical 87-predictor information. The runner verifies frozen file hashes, feature order and disjoint IDs. All models use the same saved training-only preprocessing. No existing checkpoint is reused.

| Family | Training device | Search |
|---|---|---|
| Logistic regression | CPU, all 12 logical cores available to numerical libraries | Four C values |
| Histogram gradient boosting | CPU, all 12 logical cores available | Four learning-rate/leaf-size combinations |
| CatBoost | RTX 3050 CUDA GPU | Four depth/learning-rate combinations |
| MLP | RTX 3050 CUDA GPU | Four architecture/learning-rate combinations |
| BioNutriGPT | RTX 3050 CUDA GPU | Four learning-rate/weight-decay combinations |
| Reference FT-Transformer | RTX 3050 CUDA GPU | Four block-count/learning-rate combinations |

Each classical configuration fits three outcome heads; each neural configuration fits one shared three-output model. Four configurations per family is a modest equal-count budget, not equal compute or an exhaustive benchmark. Actual training time is saved. FT-Transformer uses the installed reference rtdl-revisiting-models implementation; all supplied columns, including encoded categories, are treated as continuous tokens to preserve identical predictor representation. CatBoost likewise receives the common numeric matrix; native categorical-feature advantages are not evaluated.

Neural models use float32, batch size 64, at most 40 epochs, patience 7, AdamW and unweighted binary cross-entropy. BioNutriGPT retains the current architecture and MoE auxiliary term; the old hardcoded class-weighted training loss is deliberately not reused. Full hyperparameters and coefficients are in stage1_comparison_protocol_v2.json. HistGradientBoosting has a fixed 300-iteration budget and no internal random split. CatBoost uses validation AUC for per-head early stopping with patience 40 and at most 600 iterations. GPU floating-point execution, especially CatBoost, need not be bitwise reproducible.

## Selection and calibration boundaries

Select one configuration per family using validation mean AUROC for the two primary outcomes (diabetes and CVD); tie-breaking uses configuration order. Obesity is secondary. The classical heads share the selected configuration. Validation selects neural epochs; temporal and calibration labels are never supplied to trial training or model selection.

After selection is saved, fit one prespecified unpenalized logistic recalibrator per outcome on calibration-set probability logits. No calibration-method search is performed. Export raw and recalibrated probabilities separately. The evaluation calibration slope/intercept is a diagnostic fit on temporal predictions; it does not replace their recalibrator or modify temporal predictions.

Report AUROC, average precision, Brier, log-loss, calibration intercept/slope and equal-width calibration-bin counts. Fixed 0.5 threshold metrics are descriptive only; no clinical referral cutoff or decision-curve claim is established. A model can remain better discriminating but less calibrated, so the report must retain both views. Paired confidence intervals, multi-seed stability, survey inference and ablations are later work blocks; this single-seed comparison alone cannot establish superiority.

## Runtime, memory and audit outputs

Warm raw-model prediction latency is measured at batch sizes 1 and 128 with three warmups and 20 timed repeats. CUDA timing is synchronized. It excludes model loading, preprocessing and recalibration. Classical prediction runs on CPU, including CatBoost prediction; GPU neural prediction and CPU timings therefore represent their stated deployment devices, not identical hardware execution.

Each trial saves its model and validation predictions before its atomic completion record. Neural epochs are journalled and best weights saved. An interrupted incomplete trial restarts from its beginning; completed trials are reused only after hash verification. Each family saves selection, calibrators, participant-level predictions, metrics, calibration curves and final completion metadata. Participant-level outputs remain local. CPU RSS is sampled process-wide; PyTorch allocator peak memory covers only PyTorch, not CatBoost GPU memory.

Source, protocol, environment and data bindings must match to resume. A changed completed artifact is rejected. Failures remain failures; CUDA unavailability is not silently replaced by CPU training. Running without arguments performs preflight only.

## Execution controls

The user requested approval before each independent long run. Run exactly one family per approved command, then report its result before asking to start the next. Recommended first run:

```powershell
python -B run_stage1_comparison_v2.py --run-model logistic_regression
```

Other family names: hist_gradient_boosting, catboost, mlp, bionutrigpt, ft_transformer. Do not invoke all families in a shell loop. No study output directory has been created by synthetic tests; their temporary artifacts were removed.

The existing manuscript metrics and production pipeline remain legacy. New results must be labelled by their repaired outcome/cohort definition and cannot be numerically compared with the old evaluation as if only the model changed.

References: [FT-Transformer reference implementation](https://github.com/yandex-research/rtdl-revisiting-models), [CatBoost GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu).
