# Stage 2 Repaired-Model Stability Benchmark (v1) Results — 24 November 2026

## Executive Summary & Study Status

The controlled engineering stability evaluation of the BioNutriGPT Stage 2 Counterfactual Prescription Engine (`repaired_v1` release) under proportional input perturbations has been **executed, independently verified, and recorded with zero discrepancies**.

- **Status:** **COMPLETED AND VERIFIED**
- **Hardware & Environment:**
  - Python Executable: `C:\Users\MANI KUMAR\AppData\Local\Python\pythoncore-3.14-64\python.exe`
  - PyTorch: `2.14.0+cu130`
  - Compute Device: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA)
  - CPU Thread Policy: 12 threads (`torch.set_num_threads(12)`)
- **Cohorts & Scope:**
  - Pilot Cohort: Exactly 3 development participants $\times$ 8 scenarios = 24 optimization tasks.
  - Full Evaluation Cohort: Exactly 100 adult temporal NHANES participants (seed `20261125`, reused from Stage 2 ablation studies) $\times$ 8 scenarios = 800 optimization tasks.
  - Cohorts are strictly disjoint (0 overlap between development and evaluation participants). Cohort reuse from ablations is intentional to enable matched within-cohort comparisons; these are not newly sampled independent participants.
- **Total Optimization Tasks:** Exactly **824 counterfactual optimization tasks** (24 pilot + 800 full), executed sequentially on GPU with condition rotation.
- **Measured Wall Durations:**
  - Full Execution Session Wall Time (`stage2_stability_repaired_v1.py --run full`): **6,123.0 seconds** ($\approx 102.05$ minutes) from cold start, averaging $\approx 7.65$ seconds per task (Adam 50 steps + convex quadratic projection + reconstructive audit).
  - Full Independent Verification Wall Time (`verify_stage2_stability_repaired_v1.py --verify full`): **32.0 seconds** (800 forward model passes on GPU + 40 bootstrap interval reconstructions).
- **Verification Receipts & Hashes:**
  - Manifest Digest: `9a0487a7d33bc4afb962e48e93ac8c0b8ece1401052948d08cd63b3edf458806`
  - Pilot Verification Receipt: [Saved Model/stage2_stability_repaired_v1/pilot/verification.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/pilot/verification.json) (`status: "passed"`, 40 intervals, `analysis_sha256: 134bdc67...`)
  - Full Verification Receipt: [Saved Model/stage2_stability_repaired_v1/full/verification.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/full/verification.json) (`status: "passed"`, 40 intervals, `analysis_sha256: 0e6e1e63dae634a4aac3a71414d086695ea43c4ebe70fbb4f464205ac0cacc3f`)
  - Both verifiers confirmed active rejection of corrupted confidence intervals (`changed_ci_rejected: true`) and corrupted study identities (`changed_identity_rejected: true`).
- **Production Status:** Production code, Stage 1 weights, preprocessors, and deployed targets remain 100% untouched.

---

## 1. Full Evaluation Cohort Results ($N=100$, 800 Tasks)

All displacement metrics evaluate paired changes from each participant's own unperturbed baseline target. Target displacements are normalized by fixed daily engineering reference scales (`s2.DAILY_REFERENCE_SCALES`):
- Scaled $L_1$: $\sum_{j=1}^{14} \frac{|\Delta T_j|}{\text{Scale}_j}$
- Scaled $L_\infty$: $\max_{j=1}^{14} \frac{|\Delta T_j|}{\text{Scale}_j}$

Confidence intervals are 95% percentiles from 10,000 paired participant-bootstrap resamples (seed `20261124`). All intervals are exploratory and pointwise.

| Scenario | Perturbation Description | Numerical Pass Rate | Overall Accepted Rate | Scaled $L_1$ Displacement Mean [95% CI] | Scaled $L_1$ Median (p95 / Max) | Scaled $L_\infty$ Displacement Mean [95% CI] | Scaled $L_\infty$ Median (p95 / Max) | Prescribed Energy Delta (kcal) Mean [95% CI] | Overall Acceptance Delta Mean [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | Reference unperturbed input | 100/100 (100%) | 62/100 (62%) | 0.0000 [0.0000, 0.0000] | 0.0000 (0.0000 / 0.0000) | 0.0000 [0.0000, 0.0000] | 0.0000 (0.0000 / 0.0000) | 0.0 [0.0, 0.0] | 0.0000 [0.0000, 0.0000] |
| **`repeat`** | Identical inputs, same seed | 100/100 (100%) | 62/100 (62%) | **0.0000** [0.0000, 0.0000] | 0.0000 (0.0000 / 0.0000) | **0.0000** [0.0000, 0.0000] | 0.0000 (0.0000 / 0.0000) | 0.0 [0.0, 0.0] | 0.0000 [0.0000, 0.0000] |
| **`diet_minus1`** | Raw dietary nutrients $\times 0.99$ | 100/100 (100%) | 64/100 (64%) | **0.3294** [0.2775, 0.3888] | 0.2559 (0.8006 / 1.7323) | **0.0953** [0.0631, 0.1350] | 0.0409 (0.2451 / 1.0961) | 0.0 [0.0, 0.0] | +0.0200 [0.0000, +0.0500] |
| **`diet_plus1`** | Raw dietary nutrients $\times 1.01$ | 100/100 (100%) | 62/100 (62%) | **0.2978** [0.2585, 0.3418] | 0.2471 (0.6785 / 1.3043) | **0.0753** [0.0532, 0.1046] | 0.0418 (0.1860 / 1.0454) | 0.0 [0.0, 0.0] | 0.0000 [0.0000, 0.0000] |
| **`diet_minus5`** | Raw dietary nutrients $\times 0.95$ | 100/100 (100%) | 68/100 (68%) | **1.3066** [1.1546, 1.4677] | 1.0962 (3.0669 / 3.9896) | **0.2452** [0.2048, 0.2905] | 0.1866 (0.7957 / 1.2380) | 0.0 [0.0, 0.0] | +0.0600 [+0.0200, +0.1100] |
| **`diet_plus5`** | Raw dietary nutrients $\times 1.05$ | 100/100 (100%) | 64/100 (64%) | **1.3518** [1.2023, 1.5089] | 1.2187 (2.9747 / 3.9925) | **0.2858** [0.2350, 0.3416] | 0.1769 (0.8794 / 1.4456) | 0.0 [0.0, 0.0] | +0.0200 [-0.0300, +0.0800] |
| **`lipid_minus5`** | Measured lipids $\times 0.95$ | 100/100 (100%) | 63/100 (63%) | **0.2554** [0.1882, 0.3310] | 0.1177 (1.0253 / 2.0558) | **0.1298** [0.0906, 0.1776] | 0.0514 (0.5390 / 1.2411) | -10.99 [-26.98, 0.0] | +0.0100 [-0.0300, +0.0500] |
| **`lipid_plus5`** | Measured lipids $\times 1.05$ | 100/100 (100%) | 70/100 (70%) | **0.3008** [0.2123, 0.4012] | 0.1327 (1.3195 / 2.6213) | **0.1616** [0.1091, 0.2200] | 0.0594 (0.9737 / 1.3977) | +15.80 [+2.76, +32.87] | +0.0800 [+0.0300, +0.1400] |

---

## 2. Clinical Routing Threshold Crossings ($N=100$)

The stability protocol monitors naturally occurring threshold crossings across actual production cutoffs:
- Diabetes Cutoff: $\ge 0.1762$
- Obesity Cutoff: $\ge 0.4180$
- CVD Cutoff: $\ge 0.3846$

Baseline profiles within $0.02$ probability units of any cutoff are designated as **near-threshold baselines**. In the evaluation cohort ($N=100$), exactly **11 participants** were near-threshold baselines.

| Scenario | Routing Crossings Observed | Near-Threshold Baselines ($n=11$) | Mean Scaled $L_1$ Among Crossing Profiles | Mean Scaled $L_1$ Among Noncrossing Profiles | Clinical Impact & Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`baseline`** | 0 / 100 | 11 / 100 | N/A | 0.0000 | Reference baseline state. |
| **`repeat`** | 0 / 100 | 11 / 100 | N/A | 0.0000 | Zero routing shifts under identical inputs. |
| **`diet_minus1`** | 0 / 100 | 11 / 100 | N/A | 0.3294 | $\pm 1\%$ dietary shifts do not trigger discrete rule boundary transitions. |
| **`diet_plus1`** | 0 / 100 | 11 / 100 | N/A | 0.2978 | $\pm 1\%$ dietary shifts do not trigger discrete rule boundary transitions. |
| **`diet_minus5`** | 0 / 100 | 11 / 100 | N/A | 1.3066 | Dietary shifts alter nutrient distance without crossing model disease cutoffs. |
| **`diet_plus5`** | 0 / 100 | 11 / 100 | N/A | 1.3518 | Dietary shifts alter nutrient distance without crossing model disease cutoffs. |
| **`lipid_minus5`** | **6 / 100** | 11 / 100 | **0.8851** | **0.2152** | $\pm 5\%$ lipid changes cross cutoffs in 6 participants, increasing $L_1$ displacement by $4.1\times$. |
| **`lipid_plus5`** | **5 / 100** | 11 / 100 | **1.3940** | **0.2432** | $\pm 5\%$ lipid changes cross cutoffs in 5 participants, increasing $L_1$ displacement by $5.7\times$. |

*Key Finding on Routing:* Proportional shifts in measured serum lipids (HDL and total cholesterol) directly affect baseline model probabilities, causing 5 to 6 participants near the diagnostic thresholds to cross routing cutoffs. When a profile crosses a cutoff, discrete clinical rule constraints change, resulting in substantially larger target displacement ($\text{Mean } L_1 \approx 0.89\text{--}1.39$) compared to participants who do not cross cutoffs ($\text{Mean } L_1 \approx 0.22\text{--}0.24$).

---

## 3. Pilot Cohort Reference Results ($N=3$, 24 Tasks)

The separate 3-participant development pilot was executed and verified prior to launching the full benchmark. As required by the protocol, pilot statistics are reported separately and not pooled into cohort totals:

| Scenario | Evaluated $N$ | Numerical Pass | Overall Accepted | Scaled $L_1$ Mean [95% CI] | Scaled $L_1$ Median (Max) | Scaled $L_\infty$ Mean [95% CI] | Scaled $L_\infty$ Median (Max) | Routing Crossings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | 3 | 3 / 3 | 2 / 3 | 0.0000 [0.0000, 0.0000] | 0.0000 (0.0000) | 0.0000 [0.0000, 0.0000] | 0.0000 (0.0000) | 0 |
| **`repeat`** | 3 | 3 / 3 | 2 / 3 | **0.0000** [0.0000, 0.0000] | 0.0000 (0.0000) | **0.0000** [0.0000, 0.0000] | 0.0000 (0.0000) | 0 |
| **`diet_minus1`** | 3 | 3 / 3 | 2 / 3 | 0.4676 [0.0723, 0.7584] | 0.5720 (0.7584) | 0.1024 [0.0088, 0.1944] | 0.1040 (0.1944) | 0 |
| **`diet_plus1`** | 3 | 3 / 3 | 2 / 3 | 0.4094 [0.0696, 0.7184] | 0.4402 (0.7184) | 0.0815 [0.0071, 0.1944] | 0.0431 (0.1944) | 0 |
| **`diet_minus5`** | 3 | 3 / 3 | 2 / 3 | 1.9939 [0.3861, 3.6075] | 1.9882 (3.6075) | 0.4722 [0.0623, 0.9718] | 0.3825 (0.9718) | 0 |
| **`diet_plus5`** | 3 | 3 / 3 | 2 / 3 | 2.0034 [0.3979, 3.6528] | 1.9593 (3.6528) | 0.4241 [0.0851, 0.9718] | 0.2156 (0.9718) | 0 |
| **`lipid_minus5`** | 3 | 3 / 3 | 2 / 3 | 0.1816 [0.0381, 0.3978] | 0.1088 (0.3978) | 0.0978 [0.0099, 0.2335] | 0.0499 (0.2335) | 0 |
| **`lipid_plus5`** | 3 | 3 / 3 | 2 / 3 | 0.0650 [0.0462, 0.1023] | 0.0467 (0.1023) | 0.0288 [0.0244, 0.0316] | 0.0303 (0.0316) | 0 |

---

## 4. Key Engineering Conclusions

1. **Numerical Repeatability is Exact:**
   Re-running the identical input profile with the same initialization seed produces bit-for-bit identical targets across all 100 participants (max scaled $L_\infty = 0.0000$, scaled $L_1 = 0.0000$). The optimizer is mathematically deterministic on fixed inputs.
2. **Proportional Dietary Scaling Displacements:**
   Small dietary shifts produce smooth, proportional target displacements:
   - $\pm 1\%$ dietary shifts displace targets by an average of $\approx 0.31$ daily reference units (median $\approx 0.25$, p95 $\approx 0.74$, max $1.73$).
   - $\pm 5\%$ dietary shifts displace targets by an average of $\approx 1.33$ daily reference units (median $\approx 1.16$, p95 $\approx 3.02$, max $3.99$).
   - The scaling is roughly linear ($\approx 4.2\times$ increase in displacement for a $5\times$ increase in input perturbation).
3. **Calorie Invariance Under Dietary Perturbation:**
   Energy delta is exactly $0.0\text{ kcal}$ across all dietary perturbation scenarios (`diet_minus1`, `diet_plus1`, `diet_minus5`, `diet_plus5`). The energy prescription calculation is derived from sex, age, height, weight, and physical activity, which remain unperturbed in dietary scenarios.
4. **Boundary Discontinuity from Measured Biomarkers:**
   Perturbing serum lipid biomarkers ($\pm 5\%$ on HDL and total cholesterol) shifts baseline model probabilities, inducing 5 to 6 participants near cutoffs to cross diagnostic thresholds. When routing thresholds are crossed, discrete clinical rule constraints activate/deactivate, producing discontinuous jumps in target nutrient limits and energy envelopes (up to $+500\text{ kcal}$ energy delta in `lipid_plus5`). Non-crossing profiles experience small smooth shifts ($L_1 \approx 0.23$).

---

## 5. Scope Boundaries, Limitations & Disclosures

1. **Defined Engineering Scope Only:** This study evaluates uniform proportional scaling of raw dietary nutrients and measured serum lipids. It does **not** test:
   - Anthropometric measurement errors (height, weight, waist circumference).
   - Missing feature imputations or data dropouts.
   - Independent nutrient-specific noise or composition uncertainty.
   - Adversarial or forced near-boundary stress profiles.
2. **Confidence Intervals Spanning Zero $\ne$ Statistical Equivalence:** Acceptance deltas whose 95% bootstrap intervals span zero reflect lack of detectable acceptance change on this sample, not proven equivalence.
3. **Integrity $\ne$ Clinical Favourability:** Verification receipts confirm that all 800 optimization records were computed, audited, and summarized without execution faults or hash mismatches. Verification does not establish clinical safety, biological optimality, or therapeutic effectiveness.
4. **No External Clinical Claims:** These numerical findings on NHANES observational profiles do not establish clinical validity, dietary adherence, or applicability to Indian populations.

---

## 6. Artifact & Verification Manifest

- **Protocol:** [Saved Model/stage2_stability_repaired_v1/protocol.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/protocol.json)
- **Manifest:** [Saved Model/stage2_stability_repaired_v1/manifest.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/manifest.json) (`SHA-256: 9a0487a7d33bc4afb962e48e93ac8c0b8ece1401052948d08cd63b3edf458806`)
- **Pilot Execution Receipt:** [Saved Model/stage2_stability_repaired_v1/pilot/execution.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/pilot/execution.json)
- **Pilot Verification Receipt:** [Saved Model/stage2_stability_repaired_v1/pilot/verification.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/pilot/verification.json)
- **Pilot Analysis:** [Saved Model/stage2_stability_repaired_v1/pilot/analysis.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/pilot/analysis.json) (`SHA-256: 134bdc6708d8e5287e0951e37aa3313bcf01dbe4d4288ea133ba82c96407e05f`)
- **Full Execution Receipt:** [Saved Model/stage2_stability_repaired_v1/full/execution.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/full/execution.json)
- **Full Verification Receipt:** [Saved Model/stage2_stability_repaired_v1/full/verification.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/full/verification.json)
- **Full Analysis:** [Saved Model/stage2_stability_repaired_v1/full/analysis.json](file:///C:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage2_stability_repaired_v1/full/analysis.json) (`SHA-256: 0e6e1e63dae634a4aac3a71414d086695ea43c4ebe70fbb4f464205ac0cacc3f`)
