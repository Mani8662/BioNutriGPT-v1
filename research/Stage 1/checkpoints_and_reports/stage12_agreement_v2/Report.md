# Stage 1 – Stage 2 Baseline Prediction Agreement Report (v2)

**Execution Date:** 2026-11-20  
**Device:** NVIDIA GeForce RTX 3050 Laptop GPU (`cuda`)  
**Tolerance:** $\le 1e-07$  
**Overall Status:** PASS

---

## 1. Objective & Scope

Establish whether Stage 1 screening (`infer_bionutrigpt_pipeline_Stage_1_and_2.py`) and Stage 2 baseline inference (`stage2_counterfactual_engine.py`) produce matching disease probabilities (Type 2 Diabetes, Obesity, Cardiovascular Disease) when evaluated on the identical patient input, model weights, and preprocessing artifacts.

**Scope & Non-Claims:**
- Evaluated on the standard benchmark patient fixture (`SAMPLE_PATIENT_VALUES`, 87 features) and metadata.
- Compares Stage 1 diagnostics against Stage 2 **patient baseline** inference only. (Stage 2's final counterfactual targets optimize nutrition and legitimately differ).
- Confirms mathematical and implementation agreement for the tested cases. Does **not** claim clinical validity, national representativeness, or prospective risk reduction.

---

## 2. Model & Preprocessing Asset Binding

Production strictly uses the legacy model checkpoint and preprocessor, separate from the repaired research artifacts:

| Component | Active Production Asset | SHA-256 Digest |
| :--- | :--- | :--- |
| **Stage 1 Checkpoint** | `Saved Model/bionutrigpt_stage1_best.pt` | `ccb47ba51760335cc56974705ed2e66ca9a45780bc272fb53cd5288b420250ae` |
| **Preprocessor** | `Data/preprocessing_artifacts.joblib` | `04c69228edbf0ac4fb50c7e65f73e2c017b9eb56288f05cb3da091a0562278c4` |
| **Probability Transform** | `raw_sigmoid_no_calibration` | *Raw sigmoid logits without temperature or isotonic recalibration* |
| **Operating Cutoffs** | T2D: `0.617`, Obesity: `0.445`, CVD: `0.603` | *Legacy empirical thresholds preserved* |
| **Feature Order** | 87 features (`FEATURE_COLS`) | `2299737c3ed80cb462a7a5716bead6cf515b63e19f6cbec8fe26238377f6e0cf` |

*Note: Repaired research assets (`Saved Model/stage1_comparison_v2/` and `Data/stage1_repaired_v2/`) remain strictly isolated and are not deployed in production.*

---

## 3. Measured Case Results

| Case | Disease | Stage 1 Probability | Stage 2 Baseline Probability | Absolute Difference | Outcome Status | Pass/Fail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case A: Supplied Ratios** | Diabetes | 0.34435883 | 0.34435883 | 0.00e+00 | HIGH RISK | PASS |
| | Obesity | 0.23807271 | 0.23807271 | 0.00e+00 | HIGH RISK | PASS |
| | CVD | 0.03901693 | 0.03901693 | 0.00e+00 | HIGH RISK | PASS |
| **Case B: Canonical Ratios** | Diabetes | 0.34435883 | 0.34435883 | 0.00e+00 | HIGH RISK | PASS |
| | Obesity | 0.23807271 | 0.23807271 | 0.00e+00 | HIGH RISK | PASS |
| | CVD | 0.03901693 | 0.03901693 | 0.00e+00 | HIGH RISK | PASS |
| **Case C: Repeat (Determinism)** | Diabetes | 0.34435883 | 0.34435883 | 0.00e+00 | HIGH RISK | PASS |
| | Obesity | 0.23807271 | 0.23807271 | 0.00e+00 | HIGH RISK | PASS |
| | CVD | 0.03901693 | 0.03901693 | 0.00e+00 | HIGH RISK | PASS |

---

## 4. Contract & Invariant Verifications

1. **Max Probability Difference:** `0.00e+00` (well below tolerance `1e-07`).
2. **Asset Binding Identity:** Checkpoint SHA-256, Preprocessor SHA-256, and feature ordering matched identically across Stage 1 and Stage 2 in all three runs.
3. **Threshold Agreement:** Operating cutoffs matched between `screening["classifications"]` and `planning["model_binding"]["operating_cutoffs"]` exactly (`diabetes: 0.617`, `obesity: 0.445`, `cvd: 0.603`).
4. **Classification Agreement:** Categorical risk classifications agreed across all cases.
5. **Effective Ratio Audit:** Both stages logged identical `derived_input_audit` structures, recording recomputation of the 7 derived dietary ratios.
6. **Input Immutability:** Neither `canonicalize_dietary_ratios`, `run_stage1_diagnostics`, nor `generate_counterfactual_prescription` mutated caller arrays.
7. **Non-Derived Feature Conservation:** Exactly 80 non-derived clinical features remained identical before and after canonical ratio recomputation.
8. **Nutrient Envelope Invariance:** Stage 2 final nutrient envelopes between Case A (supplied) and Case B (canonical) were identical (`envelope_identical_a_b = True`), because Stage 2 internally applies the shared canonical ratio contract.
9. **Determinism:** Results between Case B and Case C matched 100.0% (`determinism_identical_b_c = True`).

---

## 5. Artifacts and Evidence Files

Saved in `Saved Model/stage12_agreement_v2/`:
- `protocol.json`
- `binding.json`
- `results.json`
- `summary.json`
- `supplied_ratios_stage1.json` & `supplied_ratios_stage2.json`
- `canonical_ratios_stage1.json` & `canonical_ratios_stage2.json`
- `repeat_canonical_ratios_stage1.json` & `repeat_canonical_ratios_stage2.json`
- `Report.md` (this file)
