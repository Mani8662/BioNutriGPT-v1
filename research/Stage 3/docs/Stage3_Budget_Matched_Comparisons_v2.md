# Stage 3 matched comparisons: corrected interpretation and verification (v2)

### Corrected matched budget and target-source interpretation (24 November 2026)

Retrospective analysis uses 5,400 existing solve records from 100 participants, not 5,400 independent participants. No new optimization or training was performed. Five seconds is a candidate engineering setting on this sample, not an established clinical operating point or user-latency preference.

For the target-aware objective, observed valid plans were 891/900 at three seconds, 897/900 at five seconds and 896/900 at ten seconds. At five seconds both objectives accepted the same 897 cells; this does not establish population equivalence. At ten seconds, target-source success rates are not all identical. Across arms, the matched five-minus-three-second target-error differences were -0.0181 to -0.0145, with mean runtime differences +1.66 to +1.69 seconds. Ten-minus-five-second target-error differences were -0.0141 to -0.0115, at +2.79 to +2.92 seconds. These continuous comparisons condition on joint success. Common-success sensitivity retained the direction of lower target error; this does not establish clinical benefit. Normalization uses engineering reference scales, not a verified FDA nutrient standard. Recorded status 0 indicates termination under the configured 1% relative MIP-gap tolerance, not exact integer optimality.

At five seconds with the target-aware objective, model-guided versus minimum-change end-to-end acceptance differed by +7.67 percentage points (exploratory participant-bootstrap 95% CI [1.33, 14.33]). However, the per-mode exact tests have Holm-adjusted p=0.6943359375 across the declared 54-test end-to-end family. An advantage over minimum-change is therefore not established after multiplicity correction. The unadjusted bootstrap interval and adjusted tests answer different inferential questions. The prior claim that all 54 adjusted tests had p<0.001 was incorrect.

Saved Stage 2 flags distinguish numerical constraint checks from the model-score gate: rules-only 99/100 numerical, 49/100 score-gate and 48/100 overall; minimum-change 100/100, 69/100 and 69/100; model-guided 100/100, 77/100 and 77/100. These are counts of saved flags, not freshly recomputed Stage 2 predictions. The higher model-guided overall acceptance is principally associated with the model-score gate, not evidence of uniquely resolving clinical-rule conflicts. No external clinical benefit, target-source independence, practical equivalence or clinical superiority is demonstrated. Comparing error to different sources' own targets cannot rank their clinical quality.

Verification v2 replays statistical calculations from the existing reconstructed records; it is not a fresh food-catalog or solver audit. Six corruption fixtures now invoke the same validator used for clean data, and unexpected acceptance raises an uncaught failure. The original v1 fixture harness could catch its own failure sentinel and was not adequate evidence of rejection. Original results and historical artifacts are retained. See Stage3_Budget_Matched_Comparisons_v2.md and Saved Model/stage3_budget_comparisons_v2/verification.json. Production behavior remains unchanged by this repair.


## Statistical scope

Participant-weighted differences and 10,000-resample bootstrap intervals use seed 20261124. Continuous contrasts use jointly accepted cells. Exact tests are per mode with families of 54 budget, 54 target-source valid-plan and 54 target-source end-to-end tests; no pooled repeated-mode exact test is used. Confidence intervals are exploratory and not multiplicity-adjusted. Absolute means in the common-success table are retained from the original analysis; the verifier checks its paired differences, not those absolute-mean intervals.

### Table 1: Budget Contrasts Across Stage 2 Arms and Stage 3 Objectives

| Objective | Stage 2 Arm | Contrast | Valid Plans (pp) [95% CI] | E2E Acceptance (pp) [95% CI] | Runtime Difference (s) [95% CI] | $L_1$ Target Error Difference [95% CI] | Contributing Cells |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A** (Current) | rules_only | 5s vs. 3s | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +1.939 [+1.896, +1.975] | +0.0189 [+0.0027, +0.0351] | 298 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +4.447 [+4.294, +4.593] | +0.0243 [+0.0059, +0.0431] | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +6.386 [+6.207, +6.554] | +0.0421 [+0.0174, +0.0664] | 298 / 300 |
| | minimum_change | 5s vs. 3s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +1.936 [+1.897, +1.970] | +0.0146 [-0.0028, +0.0329] | 299 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +4.396 [+4.230, +4.553] | +0.0179 [-0.0003, +0.0368] | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +6.332 [+6.144, +6.509] | +0.0325 [+0.0119, +0.0532] | 299 / 300 |
| | model_guided | 5s vs. 3s | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +1.939 [+1.889, +1.979] | +0.0284 [+0.0086, +0.0488] | 298 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +4.493 [+4.333, +4.642] | +0.0155 [-0.0005, +0.0316] | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.33 [0.00, +1.00] | +0.33 [0.00, +1.00] | +6.432 [+6.246, +6.603] | +0.0445 [+0.0223, +0.0666] | 298 / 300 |
| **Variant B** (Target-Aware) | rules_only | 5s vs. 3s | +0.33 [0.00, +1.00] | +0.00 [0.00, 0.00] | +1.694 [+1.618, +1.767] | **-0.0181 [-0.0243, -0.0129]** | 298 / 300 |
| | | 10s vs. 5s | -0.33 [-1.00, 0.00] | -0.33 [-1.00, 0.00] | +2.831 [+2.573, +3.083] | **-0.0141 [-0.0195, -0.0092]** | 298 / 300 |
| | | 10s vs. 3s (sec.) | +0.00 [-1.00, +1.00] | -0.33 [-1.00, 0.00] | +4.524 [+4.214, +4.829] | **-0.0316 [-0.0410, -0.0238]** | 297 / 300 |
| | minimum_change | 5s vs. 3s | +0.67 [0.00, +1.67] | +0.67 [0.00, +1.67] | +1.663 [+1.579, +1.741] | **-0.0145 [-0.0183, -0.0109]** | 297 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +2.916 [+2.649, +3.171] | **-0.0115 [-0.0154, -0.0079]** | 299 / 300 |
| | | 10s vs. 3s (sec.) | +0.67 [0.00, +1.67] | +0.67 [0.00, +1.67] | +4.579 [+4.247, +4.895] | **-0.0257 [-0.0301, -0.0215]** | 297 / 300 |
| | model_guided | 5s vs. 3s | +1.00 [0.00, +2.67] | +1.00 [0.00, +2.67] | +1.678 [+1.591, +1.759] | **-0.0159 [-0.0221, -0.0110]** | 296 / 300 |
| | | 10s vs. 5s | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] | +2.789 [+2.495, +3.069] | **-0.0130 [-0.0171, -0.0093]** | 299 / 300 |
| | | 10s vs. 3s (sec.) | +1.00 [0.00, +2.67] | +1.00 [0.00, +2.67] | +4.483 [+4.137, +4.813] | **-0.0291 [-0.0382, -0.0221]** | 296 / 300 |

### Table 2: Sensitivity Analysis — Common-Success Cells Across All Three Budgets

| Objective | Arm | Common Cells | 3s $L_1$ Mean [95% CI] | 5s $L_1$ Mean [95% CI] | 10s $L_1$ Mean [95% CI] | 5s - 3s Diff [95% CI] | 10s - 5s Diff [95% CI] | 10s - 3s Diff [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A** | rules_only | 298 / 300 | 0.9430 [0.8988, 0.9882] | 0.9619 [0.9168, 1.0076] | 0.9851 [0.9392, 1.0322] | +0.0189 [+0.0027, +0.0351] | +0.0232 [+0.0049, +0.0418] | +0.0421 [+0.0174, +0.0664] |
| | minimum_change | 299 / 300 | 0.9678 [0.9174, 1.0184] | 0.9824 [0.9320, 1.0329] | 1.0004 [0.9507, 1.0494] | +0.0146 [-0.0028, +0.0329] | +0.0179 [-0.0003, +0.0368] | +0.0325 [+0.0119, +0.0532] |
| | model_guided | 298 / 300 | 0.9430 [0.8967, 0.9897] | 0.9714 [0.9234, 1.0197] | 0.9875 [0.9398, 1.0347] | +0.0284 [+0.0086, +0.0488] | +0.0161 [+0.0001, +0.0322] | +0.0445 [+0.0223, +0.0666] |
| **Variant B** | rules_only | 297 / 300 | 0.4508 [0.4332, 0.4705] | 0.4330 [0.4162, 0.4517] | 0.4192 [0.4026, 0.4371] | **-0.0178 [-0.0236, -0.0129]** | **-0.0137 [-0.0190, -0.0090]** | **-0.0316 [-0.0410, -0.0238]** |
| | minimum_change | 297 / 300 | 0.5016 [0.4729, 0.5312] | 0.4872 [0.4590, 0.5165] | 0.4759 [0.4485, 0.5044] | **-0.0145 [-0.0183, -0.0109]** | **-0.0112 [-0.0150, -0.0076]** | **-0.0257 [-0.0301, -0.0215]** |
| | model_guided | 296 / 300 | 0.4935 [0.4633, 0.5239] | 0.4775 [0.4497, 0.5057] | 0.4644 [0.4374, 0.4917] | **-0.0159 [-0.0221, -0.0110]** | **-0.0131 [-0.0172, -0.0094]** | **-0.0291 [-0.0382, -0.0221]** |

### Table 3: Target-Source Contrasts Under Matched Objective and Budget

| Objective | Budget | Contrast | Valid Plans (pp) [95% CI] | E2E Acceptance (pp) [95% CI] | Upstream Stage 2 Gate Diff (pp) | Own-Target $L_1$ Error Diff [95% CI] | Runtime Difference (s) [95% CI] |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Variant A** | **3.0s** | Model-Guided vs. Minimum-Change | -0.33 [-1.00, 0.00] | +7.33 [+1.00, +14.00] | +8.0 pp (77 vs. 69) | -0.0245 [-0.0424, -0.0073] | +0.004 [-0.007, +0.015] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | -0.0000 [-0.0438, +0.0610] | +0.002 [-0.008, +0.013] |
| | | Minimum-Change vs. Rules-Only | +0.33 [0.00, +1.00] | +21.67 [+12.67, +31.33] | +21.0 pp (69 vs. 48) | +0.0244 [-0.0207, +0.0864] | -0.002 [-0.013, +0.009] |
| | **5.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0107 [-0.0240, +0.0034] | +0.006 [-0.013, +0.024] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0095 [-0.0334, +0.0681] | +0.002 [-0.019, +0.023] |
| | | Minimum-Change vs. Rules-Only | 0.00 [0.00, 0.00] | +21.33 [+12.00, +31.00] | +21.0 pp (69 vs. 48) | +0.0202 [-0.0208, +0.0773] | -0.004 [-0.024, +0.016] |
| | **10.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0131 [-0.0246, -0.0009] | +0.103 [-0.070, +0.273] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0007 [-0.0431, +0.0622] | +0.048 [-0.134, +0.229] |
| | | Minimum-Change vs. Rules-Only | 0.00 [0.00, 0.00] | +21.33 [+12.00, +31.00] | +21.0 pp (69 vs. 48) | +0.0138 [-0.0305, +0.0755] | -0.055 [-0.228, +0.117] |
| **Variant B** | **3.0s** | Model-Guided vs. Minimum-Change | -0.33 [-2.33, +1.33] | +7.33 [+1.00, +14.00] | +8.0 pp (77 vs. 69) | -0.0079 [-0.0155, -0.0000] | +0.019 [-0.013, +0.051] |
| | | Model-Guided vs. Rules-Only | -0.67 [-2.33, +0.67] | +28.00 [+18.67, +37.67] | +29.0 pp (77 vs. 48) | +0.0418 [+0.0023, +0.0980] | +0.002 [-0.033, +0.038] |
| | | Minimum-Change vs. Rules-Only | -0.33 [-1.33, +0.67] | +20.67 [+11.67, +30.33] | +21.0 pp (69 vs. 48) | +0.0510 [+0.0107, +0.1071] | -0.017 [-0.049, +0.017] |
| | **5.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0107 [-0.0166, -0.0046] | +0.034 [-0.061, +0.126] |
| | | Model-Guided vs. Rules-Only | 0.00 [0.00, 0.00] | +29.00 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0438 [+0.0050, +0.0993] | -0.014 [-0.114, +0.090] |
| | | Minimum-Change vs. Rules-Only | 0.00 [0.00, 0.00] | +21.33 [+12.00, +31.00] | +21.0 pp (69 vs. 48) | +0.0545 [+0.0148, +0.1102] | -0.047 [-0.147, +0.053] |
| | **10.0s** | Model-Guided vs. Minimum-Change | 0.00 [0.00, 0.00] | +7.67 [+1.33, +14.33] | +8.0 pp (77 vs. 69) | -0.0122 [-0.0173, -0.0066] | -0.024 [-0.297, +0.244] |
| | | Model-Guided vs. Rules-Only | +0.33 [0.00, +1.00] | +29.33 [+20.00, +39.00] | +29.0 pp (77 vs. 48) | +0.0437 [+0.0043, +0.1003] | -0.040 [-0.316, +0.231] |
| | | Minimum-Change vs. Rules-Only | +0.33 [0.00, +1.00] | +21.67 [+12.67, +31.33] | +21.0 pp (69 vs. 48) | +0.0554 [+0.0155, +0.1118] | -0.016 [-0.285, +0.255] |