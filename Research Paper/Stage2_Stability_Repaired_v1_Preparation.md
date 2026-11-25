# Stage 2 repaired-model stability: prepared, not executed

This addresses item 6 of the Stage 2 evidence plan. It is an engineering study, not clinical validation. No benchmark or inference was run during preparation.

## Frozen design

Reuse the 100 evaluation participants and three separate development participants from the ablation study. This deliberately supports within-cohort comparisons; these are not newly independent participants. Do not pool the development cases into full-study statistics.

Eight conditions per participant: baseline, identical-input repeat, dietary nutrients scaled by -1%, +1%, -5%, +5%, and both measured lipids scaled by -5%, +5%. Recompute all derived features. Leave anthropometry and demographics unchanged. These proportional perturbations are prespecified engineering stresses, not estimates of actual measurement error. They do not test nutrient-specific composition errors.

Full reference Stage 2, frozen repaired model/preprocessor/calibration, actual operating thresholds, fresh 50-step optimizer per condition, same initialization seed. Rotate condition execution order by participant position. Use CUDA and the existing 12-thread CPU policy. No tuning, retraining, threshold changes, or production modifications.

Pilot: 3 x 8 = 24 optimizations. Full: 100 x 8 = 800. Full execution requires a verified pilot. Commands and stopping rules are in Gemini_Stage2_Stability_Repaired_v1_Prompt.txt.

## Endpoints

Paired target displacement from each participant's new baseline: sum and maximum absolute nutrient changes divided by fixed daily reference scales. Also report calorie-budget change, numerical acceptance change and overall engineering acceptance change, with participant bootstrap 95% intervals (10,000 resamples, seed 20261124). Report medians, 95th percentiles, maxima, counts and denominators. All intervals are exploratory and pointwise, without multiplicity adjustment.

Separate naturally occurring routing-threshold crossings from noncrossings. Count baseline profiles within 0.02 probability units of any cutoff. These are descriptive strata, not validated clinical margins. Zero crossings does not establish threshold-boundary robustness. Forced near-boundary testing and anthropometric perturbations remain outside this version.

Repeat differences must be reported even if verification passes: verification checks integrity, not desirable stability. Do not invent a universal acceptable displacement threshold. Execution errors stop the run; numerical or score-gate failures are valid findings and remain in denominators.

## Verification and limits

Manifest binds source files recursively, model assets inherited from the frozen ablation manifest, cohort files and protocol. Each checkpoint binds transformed input, scenario and manifest. Resume validates existing checkpoints. Verifier reconstructs model predictions and acceptance, recomputes the summary, separately reproduces 40 interval estimates and rejects altered CI and identity fixtures.

The verifier shares the transformation, numerical validator and model loader. It does not independently implement the full optimizer or replay Adam. Completion requires a saved verification receipt tied to the output analysis hash. Static parsing during preparation is not evidence that GPU execution succeeds.
