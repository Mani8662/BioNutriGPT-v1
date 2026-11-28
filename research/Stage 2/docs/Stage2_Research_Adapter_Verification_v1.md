# Stage 2 repaired-prediction research adapter — implementation verification

Completed 2026-11-20. Separate module stage2_research_rule_adapter.py connects frozen calibrated research predictions to explicit condition-rule flags. No production files or checkpoints were modified. The original nutrient-bound equations were copied into a research function accepting booleans and checked against all eight legacy branch combinations. Future rule edits require rerunning parity tests.

Rules-only rejects prediction inputs. Known reported diagnoses cannot be negated by model predictions. Obesity energy rules use BMI calculated from measured weight/height, never predicted obesity. Unknown history/clinical exclusions remain explicitly unknown; positive renal/pregnancy/lactation exclusions reject construction. Age scope is >=20 for this NHANES experiment. Unknown eligibility does not imply clinical safety.

Contract checks passed: eight branch combinations, no model loader/inference calls, diagnosis precedence, BMI invariance, exact threshold boundary, unknown history, no input mutation, invalid prediction/metadata rejection and explicit clinical exclusions.

Connection pilot: five lowest temporal SEQN records, with raw source measurements and saved calibrated BioNutriGPT/CatBoost/FT-Transformer predictions. One rules-only envelope plus three models across five thresholds per participant = 80 envelopes. All have finite ordered bounds and measured baseline nutrients; energy is identical across arms for a given participant. 36 of 75 prediction-arm envelopes changed at least one bound versus the paired rules-only envelope. This is descriptive adapter behavior, not plan-quality improvement.

Threshold grid .05/.10/.20/.30/.50 is exploratory, specified after previous model evaluation; no preferred threshold was chosen. Activity is a common sedentary engineering scenario, not claimed as observed. DIQ010=3 is retained in raw provenance and treated as no reported diabetes diagnosis, not proof of normal metabolism. Baseline source HbA1c/composite labels do not select clinical-history branches.

No optimizer was run. Envelopes explicitly have certified_for_stage3=False: targets, common-reference evaluation, frozen solver settings, independent nutrient replay and paired failure reporting must be implemented before a plan comparison. No claim of prospective benefit, causal risk reduction or Indian external prediction validation follows. The differentiable repaired-model counterfactual adapter remains a separate later experiment.

Artifacts: protocol.json, envelopes_LOCAL.json, verification.json. Participant records stay local. Reproducible scripts: verify_stage2_research_rule_adapter.py and run_stage2_research_adapter_pilot.py.
