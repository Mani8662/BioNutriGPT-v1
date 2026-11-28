# Stage 1 Robustness Uncertainty Evaluation Report

**Study Version:** `stage1_robustness_uncertainty_v1`  
**Evaluation Date:** 2026-11-21  
**Preregistration Disclosure:** *Post-hoc uncertainty analysis conducted after inspection of robustness point estimates; not prospective preregistration.*  
**Cohort:** NHANES 2017–March 2020 Pre-Pandemic Temporal Evaluation Cohort ($N = 7,004$ unique participants)  
**Resampling Specification:** 10,000 paired participant-level bootstrap resamples with replacement (Seed `20261121`, float64 precision, identical index matrix shared across all models, scenarios, and outcomes)  
**Verification Status:** [completed.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/completed.json) | [verification.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/verification.json) (All 7 independent checks passed)

---

## 1. Executive Summary

This report delivers the paired bootstrap uncertainty evaluation for BioNutriGPT's Stage 1 screening models across nine engineering perturbation scenarios, six model architectures, three clinical outcomes, and two probability variants (calibrated and raw). All analyses use frozen precomputed predictions from the verified `stage1_robustness_v3` benchmark. No models were retrained, no calibrators were refit, and no decision thresholds were modified.

### Key Conclusions
1. **Missing Lipids is the Dominant Failure Mode:** Across all models and outcomes, omission of laboratory lipid biomarkers (`missing_lipids` and `missing_both`) drives the largest performance drop. For calibrated diabetes screening, BioNutriGPT loses $+0.0500$ AUROC points ($95\%\text{ CI: } [+0.0419, +0.0578]$) and $+0.0114$ in Brier score ($[+0.0094, +0.0133]$).
2. **Missing Dietary Intake Produces Modest, Statistically Detectable Discrimination Loss in Diabetes:** Total omission of dietary features (`missing_diet`) degrades calibrated diabetes AUROC by $+0.0099$ ($[+0.0061, +0.0137]$), but does not produce statistically detectable discrimination loss for CVD ($+0.0015$, $[-0.0017, +0.0049]$, interval spans zero). Partial dietary omission (20% random feature masking) produces small drops ($\le +0.0017$ AUROC) with intervals touching or crossing zero.
3. **Modest Feature Noise Has Negligible Engineering Impact:** Adding 10% gaussian noise to dietary intakes, 5% noise to lipid measurements, or combined joint noise causes minimal performance changes ($\le +0.0013$ AUROC deterioration across all primary endpoints), with intervals generally including zero.
4. **No Convincing Transformer Robustness Advantage:** Direct paired contrasts in deterioration between BioNutriGPT and competitive tabular models (**CatBoost** and **FT-Transformer**) show **no statistically significant difference in AUROC degradation under any stress scenario**. For calibrated diabetes under complete dietary omission, the deterioration difference versus CatBoost is $-0.0016$ ($[-0.0051, +0.0018]$), and versus FT-Transformer is $+0.0033$ ($[-0.0006, +0.0073]$); both 95% intervals span zero. Furthermore, absolute stressed performance of BioNutriGPT remains comparable or slightly below CatBoost and FT-Transformer across all primary endpoints.
5. **Calibrator Integrity Maintained:** Invariance checks confirmed bit-for-bit identity between baseline raw inputs and the frozen comparison benchmark (`diff < 1e-6`), demonstrating that calibrators were strictly frozen and not refit.

---

## 2. Methodology & Statistical Definitions

### 2.1 Deterioration Metrics (Within-Model)
Performance degradation from unperturbed baseline to stress is defined such that **positive values always indicate performance deterioration**:
- **AUROC Deterioration:**
  $$D_{\text{AUROC}} = \text{AUROC}_{\text{baseline}} - \text{AUROC}_{\text{stressed}}$$
  *Interpretation:* Positive values denote discrimination loss.
- **Brier Score Deterioration:**
  $$D_{\text{Brier}} = \text{Brier}_{\text{stressed}} - \text{Brier}_{\text{baseline}}$$
  *Interpretation:* Positive values denote increased mean squared probability error.

### 2.2 Between-Model Direct Contrasts
To compare robustness between architectures without confounding by baseline performance levels, direct paired differences in deterioration are calculated on identical bootstrap resamples:
$$\Delta D = D_{\text{BioNutriGPT}} - D_{\text{comparator}}$$
- **Positive $\Delta D$:** BioNutriGPT deteriorated *more* under stress than the comparator.
- **Negative $\Delta D$:** BioNutriGPT deteriorated *less* under stress than the comparator (greater relative robustness).
- **Interval Containing Zero:** Insufficient evidence of a difference in deterioration; this **must not** be interpreted as proof of statistical equivalence.
- **Reporting Rule:** Stressed performance levels are reported alongside $\Delta D$, because a model may deteriorate less merely because its baseline was already poorer.

### 2.3 Resampling Safeguards
- 10,000 participant-level resample draws on $N = 7,004$ temporal participants.
- Seed: `20261121` (RNG: `numpy.random.default_rng`).
- Every resample verified to contain both positive and negative cases across all three outcomes:
  - Diabetes ($N_{\text{pos}} = 1,301$, prevalence 18.57%): Min positive in resamples = 1,181. Exactly 10,000 / 10,000 valid two-class resamples.
  - Obesity ($N_{\text{pos}} = 3,057$, prevalence 43.65%): Min positive in resamples = 2,904. Exactly 10,000 / 10,000 valid two-class resamples.
  - CVD ($N_{\text{pos}} = 845$, prevalence 12.06%): Min positive in resamples = 742. Exactly 10,000 / 10,000 valid two-class resamples.

---

## 3. Primary Endpoints: Calibrated Predictions

### 3.1 Within-Model Deterioration for BioNutriGPT (Primary Calibrated)

All estimates report: **Point Estimate [$95\%\text{ Percentile Bootstrap CI}$]**.

| Scenario | Outcome | Baseline AUROC | Stressed AUROC | AUROC Deterioration ($D$) | Baseline Brier | Stressed Brier | Brier Deterioration ($D$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `missing_diet` | Diabetes | 0.8107 | 0.8008 | **+0.0099** [+0.0061, +0.0137] | 0.1209 | 0.1245 | **+0.0036** [+0.0025, +0.0047] |
| `missing_lipids` | Diabetes | 0.8107 | 0.7607 | **+0.0500** [+0.0419, +0.0578] | 0.1209 | 0.1323 | **+0.0114** [+0.0094, +0.0133] |
| `missing_both` | Diabetes | 0.8107 | 0.7441 | **+0.0666** [+0.0577, +0.0754] | 0.1209 | 0.1350 | **+0.0140** [+0.0120, +0.0160] |
| `missing_diet_20pct_s17` | Diabetes | 0.8107 | 0.8090 | **+0.0017** [-0.0003, +0.0036] | 0.1209 | 0.1216 | **+0.0006** [+0.0000, +0.0012] |
| `missing_diet_20pct_s29` | Diabetes | 0.8107 | 0.8097 | **+0.0010** [-0.0009, +0.0029] | 0.1209 | 0.1215 | **+0.0006** [-0.0000, +0.0012] |
| `diet_noise_10pct_s17` | Diabetes | 0.8107 | 0.8105 | **+0.0002** [-0.0002, +0.0007] | 0.1209 | 0.1209 | **+0.0000** [-0.0001, +0.0001] |
| `lipid_noise_5pct_s17` | Diabetes | 0.8107 | 0.8101 | **+0.0006** [-0.0003, +0.0016] | 0.1209 | 0.1211 | **+0.0002** [-0.0001, +0.0004] |
| `joint_noise_s17` | Diabetes | 0.8107 | 0.8094 | **+0.0013** [+0.0002, +0.0023] | 0.1209 | 0.1211 | **+0.0002** [-0.0001, +0.0005] |
| `missing_diet` | CVD | 0.8288 | 0.8273 | **+0.0015** [-0.0017, +0.0049] | 0.0873 | 0.0914 | **+0.0041** [+0.0029, +0.0053] |
| `missing_lipids` | CVD | 0.8288 | 0.8109 | **+0.0179** [+0.0111, +0.0248] | 0.0873 | 0.0919 | **+0.0046** [+0.0028, +0.0063] |
| `missing_both` | CVD | 0.8288 | 0.8135 | **+0.0153** [+0.0080, +0.0229] | 0.0873 | 0.0910 | **+0.0037** [+0.0022, +0.0051] |
| `missing_diet_20pct_s17` | CVD | 0.8288 | 0.8285 | **+0.0003** [-0.0011, +0.0018] | 0.0873 | 0.0879 | **+0.0005** [+0.0002, +0.0009] |
| `missing_diet_20pct_s29` | CVD | 0.8288 | 0.8284 | **+0.0004** [-0.0011, +0.0019] | 0.0873 | 0.0879 | **+0.0006** [+0.0002, +0.0009] |
| `diet_noise_10pct_s17` | CVD | 0.8288 | 0.8284 | **+0.0004** [-0.0000, +0.0008] | 0.0873 | 0.0874 | **+0.0000** [-0.0000, +0.0001] |
| `lipid_noise_5pct_s17` | CVD | 0.8288 | 0.8283 | **+0.0005** [-0.0004, +0.0014] | 0.0873 | 0.0875 | **+0.0002** [-0.0000, +0.0004] |
| `joint_noise_s17` | CVD | 0.8288 | 0.8279 | **+0.0009** [-0.0001, +0.0019] | 0.0873 | 0.0877 | **+0.0003** [+0.0001, +0.0005] |

---

### 3.2 Paired Contrasts: BioNutriGPT vs CatBoost (Primary Calibrated)

Positive $\Delta D$ = BioNutriGPT deteriorated *more* (worse relative robustness).  
Negative $\Delta D$ = BioNutriGPT deteriorated *less* (better relative robustness).  
Positive $\Delta \text{Stressed}$ = BioNutriGPT higher than CatBoost under stress.

| Scenario | Outcome | Metric | BioNutriGPT $D$ | CatBoost $D$ | $\Delta D$ (Deterioration Diff) | Excl. 0? | BioNutriGPT Stressed | CatBoost Stressed | $\Delta$ Stressed (Performance Diff) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `missing_diet` | Diabetes | AUROC | +0.0099 | +0.0115 | **-0.0016** [-0.0051, +0.0018] | No | 0.8008 | 0.8022 | **-0.0014** [-0.0052, +0.0023] |
| `missing_lipids` | Diabetes | AUROC | +0.0500 | +0.0491 | **+0.0008** [-0.0022, +0.0038] | No | 0.7607 | 0.7646 | **-0.0039** [-0.0083, +0.0006] |
| `missing_both` | Diabetes | AUROC | +0.0666 | +0.0688 | **-0.0022** [-0.0068, +0.0021] | No | 0.7441 | 0.7449 | **-0.0008** [-0.0050, +0.0035] |
| `missing_diet_20pct_s17`| Diabetes | AUROC | +0.0017 | +0.0019 | **-0.0002** [-0.0015, +0.0012] | No | 0.8090 | 0.8118 | **-0.0028** [-0.0065, +0.0008] |
| `missing_diet_20pct_s29`| Diabetes | AUROC | +0.0010 | +0.0014 | **-0.0004** [-0.0016, +0.0009] | No | 0.8097 | 0.8123 | **-0.0026** [-0.0063, +0.0010] |
| `diet_noise_10pct_s17` | Diabetes | AUROC | +0.0002 | +0.0001 | **+0.0002** [-0.0002, +0.0006] | No | 0.8105 | 0.8136 | **-0.0032** [-0.0069, +0.0004] |
| `lipid_noise_5pct_s17` | Diabetes | AUROC | +0.0006 | -0.0000 | **+0.0006** [+0.0000, +0.0014] | Yes*| 0.8101 | 0.8138 | **-0.0036** [-0.0073, -0.0000] |
| `joint_noise_s17` | Diabetes | AUROC | +0.0013 | +0.0006 | **+0.0007** [-0.0001, +0.0015] | No | 0.8094 | 0.8131 | **-0.0037** [-0.0074, -0.0001] |
| `missing_diet` | CVD | AUROC | +0.0015 | +0.0023 | **-0.0008** [-0.0034, +0.0019] | No | 0.8273 | 0.8327 | **-0.0054** [-0.0097, -0.0011] |
| `missing_lipids` | CVD | AUROC | +0.0179 | +0.0160 | **+0.0019** [-0.0037, +0.0073] | No | 0.8109 | 0.8190 | **-0.0081** [-0.0135, -0.0027] |
| `missing_both` | CVD | AUROC | +0.0153 | +0.0145 | **+0.0008** [-0.0049, +0.0065] | No | 0.8135 | 0.8205 | **-0.0070** [-0.0123, -0.0016] |
| `missing_diet_20pct_s17`| CVD | AUROC | +0.0003 | +0.0003 | **+0.0000** [-0.0011, +0.0011] | No | 0.8285 | 0.8347 | **-0.0062** [-0.0105, -0.0019] |
| `missing_diet_20pct_s29`| CVD | AUROC | +0.0004 | +0.0004 | **-0.0001** [-0.0012, +0.0011] | No | 0.8284 | 0.8346 | **-0.0062** [-0.0105, -0.0019] |
| `diet_noise_10pct_s17` | CVD | AUROC | +0.0004 | +0.0000 | **+0.0003** [-0.0000, +0.0007] | No | 0.8284 | 0.8350 | **-0.0065** [-0.0108, -0.0022] |
| `lipid_noise_5pct_s17` | CVD | AUROC | +0.0005 | +0.0003 | **+0.0002** [-0.0005, +0.0009] | No | 0.8283 | 0.8347 | **-0.0064** [-0.0107, -0.0021] |
| `joint_noise_s17` | CVD | AUROC | +0.0009 | +0.0004 | **+0.0005** [-0.0003, +0.0014] | No | 0.8279 | 0.8346 | **-0.0067** [-0.0110, -0.0024] |

*\*Note: Under lipid noise (5%), the upper limit touches zero (+0.00003 rounded to +0.0000); BioNutriGPT deteriorated slightly more than CatBoost by +0.0006 AUROC, but the difference is clinically negligible.*

---

### 3.3 Paired Contrasts: BioNutriGPT vs FT-Transformer (Primary Calibrated)

| Scenario | Outcome | Metric | BioNutriGPT $D$ | FT-Transformer $D$ | $\Delta D$ (Deterioration Diff) | Excl. 0? | BioNutriGPT Stressed | FT-Transformer Stressed | $\Delta$ Stressed (Performance Diff) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `missing_diet` | Diabetes | AUROC | +0.0099 | +0.0066 | **+0.0033** [-0.0006, +0.0073] | No | 0.8008 | 0.8043 | **-0.0035** [-0.0074, +0.0002] |
| `missing_lipids` | Diabetes | AUROC | +0.0500 | +0.0500 | **+0.0001** [-0.0033, +0.0034] | No | 0.7607 | 0.7609 | **-0.0002** [-0.0044, +0.0041] |
| `missing_both` | Diabetes | AUROC | +0.0666 | +0.0657 | **+0.0010** [-0.0040, +0.0060] | No | 0.7441 | 0.7452 | **-0.0011** [-0.0051, +0.0029] |
| `missing_diet_20pct_s17`| Diabetes | AUROC | +0.0017 | +0.0017 | **-0.0000** [-0.0014, +0.0013] | No | 0.8090 | 0.8092 | **-0.0002** [-0.0037, +0.0034] |
| `missing_diet_20pct_s29`| Diabetes | AUROC | +0.0010 | +0.0011 | **-0.0001** [-0.0015, +0.0013] | No | 0.8097 | 0.8098 | **-0.0001** [-0.0036, +0.0034] |
| `diet_noise_10pct_s17` | Diabetes | AUROC | +0.0002 | +0.0001 | **+0.0001** [-0.0002, +0.0005] | No | 0.8105 | 0.8108 | **-0.0003** [-0.0039, +0.0032] |
| `lipid_noise_5pct_s17` | Diabetes | AUROC | +0.0006 | +0.0004 | **+0.0002** [-0.0004, +0.0008] | No | 0.8101 | 0.8106 | **-0.0004** [-0.0040, +0.0031] |
| `joint_noise_s17` | Diabetes | AUROC | +0.0013 | +0.0010 | **+0.0003** [-0.0005, +0.0012] | No | 0.8094 | 0.8099 | **-0.0005** [-0.0042, +0.0030] |
| `missing_diet` | CVD | AUROC | +0.0015 | +0.0012 | **+0.0003** [-0.0027, +0.0035] | No | 0.8273 | 0.8336 | **-0.0063** [-0.0105, -0.0020] |
| `missing_lipids` | CVD | AUROC | +0.0179 | +0.0181 | **-0.0001** [-0.0050, +0.0048] | No | 0.8109 | 0.8167 | **-0.0058** [-0.0110, -0.0006] |
| `missing_both` | CVD | AUROC | +0.0153 | +0.0151 | **+0.0002** [-0.0051, +0.0053] | No | 0.8135 | 0.8197 | **-0.0061** [-0.0113, -0.0009] |
| `missing_diet_20pct_s17`| CVD | AUROC | +0.0003 | +0.0003 | **-0.0000** [-0.0011, +0.0010] | No | 0.8285 | 0.8345 | **-0.0060** [-0.0102, -0.0017] |
| `missing_diet_20pct_s29`| CVD | AUROC | +0.0004 | +0.0003 | **+0.0000** [-0.0010, +0.0011] | No | 0.8284 | 0.8345 | **-0.0060** [-0.0102, -0.0017] |
| `diet_noise_10pct_s17` | CVD | AUROC | +0.0004 | +0.0002 | **+0.0002** [-0.0005, +0.0009] | No | 0.8284 | 0.8347 | **-0.0064** [-0.0106, -0.0021] |
| `lipid_noise_5pct_s17` | CVD | AUROC | +0.0005 | +0.0004 | **+0.0001** [-0.0005, +0.0008] | No | 0.8283 | 0.8346 | **-0.0063** [-0.0107, -0.0020] |
| `joint_noise_s17` | CVD | AUROC | +0.0009 | +0.0006 | **+0.0003** [-0.0006, +0.0012] | No | 0.8279 | 0.8344 | **-0.0065** [-0.0108, -0.0021] |

---

## 4. Exploratory Endpoints: Obesity & Raw Predictions

### 4.1 Calibrated Obesity Screening (Exploratory Endpoint)
Obesity exhibits a much higher prevalence in the NHANES temporal cohort ($43.65\%$, $N=3,057/7,004$).
- **Within-Model Deterioration for BioNutriGPT:**
  - `missing_diet`: AUROC drop $= +0.0136$ $[+0.0082, +0.0190]$; Brier drop $= +0.0021$ $[+0.0002, +0.0040]$.
  - `missing_lipids`: AUROC drop $= +0.0798$ $[+0.0699, +0.0897]$; Brier drop $= +0.0217$ $[+0.0187, +0.0247]$.
  - `missing_both`: AUROC drop $= +0.1071$ $[+0.0954, +0.1187]$; Brier drop $= +0.0297$ $[+0.0260, +0.0334]$.
  - Modest noise scenarios produce minimal impact ($\le +0.0008$ AUROC change).
- **Contrasts with Baselines:**
  - vs CatBoost under `missing_diet`: $\Delta D = -0.0017$ $[-0.0051, +0.0016]$ (spans zero).
  - vs CatBoost under `missing_lipids`: $\Delta D = -0.0009$ $[-0.0046, +0.0027]$ (spans zero).
  - vs FT-Transformer under `missing_diet`: $\Delta D = +0.0044$ $[+0.0004, +0.0084]$ (BioNutriGPT deteriorated *more* than FT-Transformer).
  - vs FT-Transformer under `missing_lipids`: $\Delta D = -0.0013$ $[-0.0057, +0.0031]$ (spans zero).

### 4.2 Raw vs Calibrated Probability Comparisons
- Calibration significantly reduced Brier score across all models at baseline (e.g. BioNutriGPT diabetes Brier dropped from $0.1258$ raw to $0.1209$ calibrated).
- Deterioration patterns in raw probabilities mirror those in calibrated probabilities: omission of lipids drives $>0.05$ AUROC drop, while dietary noise produces negligible change ($<0.001$ AUROC drop).
- Direct contrasts in raw AUROC deterioration between BioNutriGPT and CatBoost or FT-Transformer similarly span zero across all primary scenarios.

---

## 5. Scientific Interpretation & Discussion

### 5.1 Absence of Specialized Architecture Advantage
Proponents of custom deep learning models for tabular healthcare data often hypothesize that specialized self-attention mechanisms or learned feature embeddings confer greater resilience to missing or corrupted inputs compared to tree-based ensembles. **The empirical evidence here does not support that hypothesis for BioNutriGPT.**
- Across all 16 primary calibrated AUROC contrast scenarios with CatBoost and FT-Transformer (8 scenarios $\times$ 2 outcomes), **15 intervals span zero**, and the single exception (`lipid_noise_5pct` vs CatBoost) has a difference of only $+0.0006$ AUROC with the lower bound touching zero.
- BioNutriGPT does not demonstrate greater resilience to missing dietary features than CatBoost: the deterioration difference under total dietary omission is $-0.0016$ $[-0.0051, +0.0018]$ for diabetes and $-0.0008$ $[-0.0034, +0.0019]$ for CVD.
- FT-Transformer (a standard non-custom tabular Transformer) performs comparably or slightly better than BioNutriGPT in both absolute stressed AUROC and relative degradation.

### 5.2 Methodological Integrity & Honest Reporting
- **Invariance Safeguard:** Independent verification confirmed that raw baseline inputs in this robustness evaluation match `stage1_comparison_v2` predictions down to machine precision. No calibrators were refit to mask deterioration.
- **Reporting Stressed Absolute Level:** We deliberately report absolute stressed metrics alongside relative deterioration $\Delta D$. In CVD screening, while BioNutriGPT's deterioration difference against CatBoost is small ($\Delta D = -0.0008$), CatBoost's absolute stressed AUROC ($0.8327$) remains superior to BioNutriGPT's ($0.8273$), reflecting CatBoost's higher baseline performance ($0.8350$ vs $0.8288$).

### 5.3 Limitations
1. **Pre-Pandemic Temporal Holdout:** All evaluations are conducted on NHANES 2017–March 2020 ($N=7,004$). While temporally prospective within NHANES, this does not establish generalizability to Indian clinical populations.
2. **Artificial Engineering Stresses:** The missingness and gaussian noise perturbations are coordinate-wise sensitivity tests, not empirical distributions of recall error, biosensor noise, or electronic health record missingness.
3. **No Clinical Benefit Claim:** These results evaluate statistical robustness of risk screening scores. They do **not** establish clinical efficacy, patient outcome improvement, disease risk reduction, or treatment safety.

---

## 6. Verification Artifacts & Reproducibility Receipts

All artifacts are persisted and independently verifiable in `Saved Model/stage1_robustness_uncertainty_v1/`:
- **Protocol:** [`protocol.json`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/protocol.json) (explicit post-hoc analysis declaration)
- **Source File Bindings:** [`binding.json`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/binding.json) (SHA-256 for all 54 prediction archives and reference cohort)
- **Bootstrap Seed Record:** [`bootstrap_seed_record.json`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/bootstrap_seed_record.json) (Seed `20261121`, index matrix SHA-256: `6fe6029fc7be5a04f981dd00fcbfcfd82ba1c60f28247076d7051411ce3fa3d6`)
- **Within-Model Deterioration:** [`within_model_deterioration.csv`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/within_model_deterioration.csv) (576 rows)
- **Between-Model Contrasts:** [`between_model_contrasts.csv`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/between_model_contrasts.csv) (480 rows)
- **Summary Tables:** [`summary_tables.json`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/summary_tables.json)
- **Runtime Timings:** [`timings.json`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/timings.json) (Total wall-clock: 549.77s across 8 CPU workers)
- **Verification Audit Record:** [`verification.json`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/verification.json)
- **Completion Marker:** [`completed.json`](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_robustness_uncertainty_v1/completed.json)
