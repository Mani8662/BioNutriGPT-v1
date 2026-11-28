# Repaired Stage 1 application deployment

Updated 22 November 2026. The backend and combined Stage 1–2/Stage 1–4 pipeline now default to the versioned repaired BioNutriGPT release. This is a research deployment, not Indian external validation or clinical deployment validation. No model was retrained.

## Model and release policy

The release retains BioNutriGPT's existing architecture to preserve the counterfactual and attention interfaces. It uses comparison trial 03, selected using development validation performance, and the reserved-set calibrators from the same experiment. This choice does not claim superiority over CatBoost or FT-Transformer. The original checkpoints and research outputs remain unchanged.

The bundle is `Saved Model/deployment/repaired_stage1_v1/`: model weights, repaired training-fitted preprocessing, calibrators, selection record and a manifest containing feature order, targets and artifact hashes. Loaders verify hashes and reject mixing a legacy preprocessor or checkpoint with the repaired release. Both Stage 1 and Stage 2 use calibrated logits, including counterfactual rescoring.

All three operating thresholds are fixed at 0.5, matching the comparison's descriptive threshold. These are explicitly experimental routing thresholds, not clinically selected treatment thresholds. Legacy thresholds 0.617/0.445/0.603 are not transferred to calibrated probabilities. Changing the research threshold can change energy/rule branches, nutrient targets and feasibility; historical downstream benchmark numbers do not measure this new release.

## Input and preprocessing contract

The deployed schema remains `nhanes-87-v2`: complete finite values in raw units with categorical features already encoded. Age must be at least 20; categorical groups must be valid one-hot values. Nonzero missingness flags and incomplete named requests are rejected. The app no longer silently fills missing measurements with sample-patient defaults in repaired mode. This release does not deploy the research missing-modality experiments.

Dietary ratios, lipid derivatives and sleep deviation are recomputed. Continuous values use the repaired log transforms, clipping, scaler and six-decimal rounding. The counterfactual path uses rounded forward values and straight-through gradients through rounding; these are exploratory surrogate gradients, not causal effects. Attention weights are not causal explanations.

## Verification results

| Check | Result |
|---|---|
| All 7,004 frozen temporal calibrated predictions | Matched; maximum absolute error 1.67e-16 |
| 128 complete raw-input records against research preprocessing | Exact normalized-input match; prediction tolerance 1e-6 passed |
| Real backend Stage 1–Stage 2 baseline probabilities and model binding | Matched |
| Six invalid-input/mixed-release checks | Rejected |
| FastAPI screening with valid inputs | HTTP 200; repaired release metadata preserved |
| FastAPI screening with incomplete inputs | HTTP 422 |
| Second integration profile through the real Stage 3 solver | Accepted meal plan |
| Explicit legacy rollback in a separate process | Legacy inference passed |
| Frontend production build | TypeScript and Vite build passed |

The first synthetic integration profile (70 kg, 170 cm, age 42, male, moderate activity) failed Stage 2 filter 3: a post-filter target was outside the nutrient envelope. It was not allowed through the full-pipeline acceptance gate. The second profile (65 kg, 165 cm, age 42, male, light activity; same synthetic input vector) passed Stage 2 and produced an accepted meal plan. Neither example estimates population-level feasibility. The local Qwen compiler and a browser interaction session were not rerun for this migration.

Receipts: `Saved Model/stage1_deployment_verification_v1/verification.json`, `sample_dossier.json` and `stage3_integration.json`. Reproduce with `python verify_stage1_deployment.py`; this uses the GPU when available and performs no training. The verifier includes live in-process API calls and separate-process legacy inference.

## Activation and rollback

The repaired release is the default on the next backend start. No backend was listening on port 8000 during the migration; no existing process was restarted. Start the existing backend launcher normally, or from the backend directory run `python main.py`.

To roll back, set `BIONUTRIGPT_STAGE1_RELEASE=legacy` in the backend process environment before starting it. Restart the process after changing releases; do not change releases inside a running worker. Remove that override or set `BIONUTRIGPT_STAGE1_RELEASE=repaired_v1` to restore the repaired release. Individual checkpoint/preprocessor overrides are refused in repaired mode. Existing stored sessions retain their original model provenance and are not retrospectively rescored.

The standalone historical `infer_bionutrigpt_stage1.py` utility remains separate and is not the application's repaired inference entry point. Use the backend or combined pipeline for the versioned release.

## Evidence boundaries

This verification establishes numerical and integration consistency for the stated cases. Indian prediction validation remains unavailable. It does not validate the 0.5 rule policy clinically, establish a Transformer advantage, rerun historical LASI/chatbot benchmarks, or guarantee that every input produces an accepted meal plan. Submission-package derivatives must be regenerated separately before submission if they describe the deployment state.
