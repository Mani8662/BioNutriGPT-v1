# Stage 2 + Stage 3 Evaluation on Real Indian Adults (LASI Wave 1, N = 100)

- **Source Cohort:** IIPS Longitudinal Ageing Study in India (LASI) Wave 1 (2017-2019) (`65dc53fced0231708938236.zip`, SHA256 `2cdb47d5dc6fa617...`)
- **Full Valid Indian Cohort:** $N = 65,368$ adults across 36 Indian states/UTs
- **Stratified Evaluation Cohort:** $N = 100$ real Indian adults (25 Diabetes, 25 Obesity BMI ≥ 30, 25 CVD, 25 Multimorbid; 52 Male, 48 Female; 29 states/UTs)
- **Food Catalog:** `sourced-cooked-v5-veg-nonveg` (49 sourced dishes; 20 INDB + 29 USDA SR Legacy)
- **Ruleset:** `adult-food-rule-review-v4-exposure-separated` (30 active solid-food rules, zero active slack required)
- **Scope Note:** LASI Wave 1 supplies real Indian anthropometry, age, sex, activity, and clinical disease status for Stage 2–4 evaluation; it does not contain serum lipids or 24-hour dietary recalls and does not externally validate Stage 1.

## Strict 30-Rule Acceptance by Dietary Mode (300 Real-Patient MILP Solves)

| Dietary Mode | Accepted Plans | Mean Cal Err (%) | Median Solve (ms) | Male Acceptance | Female Acceptance | Top Failure Modes |
|---|---:|---:|---:|---:|---:|---|
| `pure_vegetarian` | **91/100 (91.0%)** | 0.46% | 2003.1 | 48/52 (92.3%) | 43/48 (89.6%) | `{'infeasible_active_rules': 9}` |
| `veg_nonveg_combo` | **96/100 (96.0%)** | 0.72% | 2005.6 | 50/52 (96.2%) | 46/48 (95.8%) | `{'timeout_no_valid_plan': 3, 'infeasible_active_rules': 1}` |
| `pure_non_vegetarian` | **98/100 (98.0%)** | 0.49% | 2006.1 | 50/52 (96.2%) | 48/48 (100.0%) | `{'timeout_no_valid_plan': 2}` |

## Acceptance by Target Energy Band (Real Indian Anthropometry)

| Dietary Mode | `< 1,500 kcal/day` | `1,500–1,999 kcal/day` | `>= 2,000 kcal/day` |
|---|---:|---:|---:|
| `pure_vegetarian` | 21/27 (77.8%) | 48/50 (96.0%) | 22/23 (95.7%) |
| `veg_nonveg_combo` | 27/27 (100.0%) | 47/50 (94.0%) | 22/23 (95.7%) |
| `pure_non_vegetarian` | 27/27 (100.0%) | 49/50 (98.0%) | 22/23 (95.7%) |
