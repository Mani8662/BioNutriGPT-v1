# Stage 4 Repaired v4 Multi-Turn Conversational Benchmark on Real LASI Wave 1 Indian Adults

- **Benchmark Version:** `stage4-lasi-indian-conversational-v4`
- **Catalog:** `sourced-cooked-v5-veg-nonveg` (`49` sourced dishes)
- **Full Benchmark Cohort:** `36` distinct LASI Wave 1 Indian adults (`36` scenarios, `78` turns) — balanced `{'diabetes_only': 9, 'obesity_only': 9, 'cvd_only': 9, 'multimorbid': 9}` across `{'pure_vegetarian': 12, 'veg_nonveg_combo': 12, 'pure_non_vegetarian': 12}` (exact 3 per mode x stratum cell).
- **Matched Equal-Information Qwen Subset:** `12` distinct LASI Wave 1 Indian adults (`12` scenarios, `24` turns) — exact 1 per `(mode, stratum)` cell (`{'diabetes_only': 3, 'obesity_only': 3, 'cvd_only': 3, 'multimorbid': 3}` across `{'pure_vegetarian': 4, 'veg_nonveg_combo': 4, 'pure_non_vegetarian': 4}`).
- **Non-Adaptive Protocol:** All Turn 1–3 prompts are frozen upfront before execution (`Saved Model/stage4_lasi_v4_frozen_scenarios.json`); zero runtime prompt rewriting.
- **Equal-Information Qwen Prompt:** Full 30 active nutrient coefficients per serving for all eligible dishes + exact Stage 2 bounds + exact plate/category rules + post-checks + full uncompressed history (mean prompt `16599.9` tokens, max `17159` tokens, `max_new_tokens=512`).
- **Scope Statement:** Unweighted computational stress test of adapted Stage 2 bounds plus Stages 3 and 4 on real LASI Wave 1 Indian adult anthropometry and prevalent disease status (using deterministic 0.85/0.10 status routing and assumed 2800/2600 mg Na/K baselines). Does NOT establish Stage 1 external validation, nationally representative weighted effectiveness, exhaustive semantic hallucination absence, or clinical benefit. Grounding checks verify numeric proximity (+/-5% rel or +/-0.3 abs) on targeted nutrient queries.

## 1. Full 36-Scenario (78-Turn) Minimal-Change Penalty Ablation

| Metric | Guarded Stage 4 (Full, $L_1=0.35$) | Guarded Stage 4 (No $L_1=0.0$) |
|---|---:|---:|
| Strict Protocol Pass Rate (timeouts count as failures) | 76/78 (97.4%) | 76/78 (97.4%) |
| Validated Swap Proposals (all revisions) | 29/42 (69.0%) | 29/42 (69.0%) |
| Validated Swap Proposals (given oracle feasible) | 29/30 (96.7%) | 29/30 (96.7%) |
| Valid Proven-Infeasible Refusals | 11 | 11 |
| Timeout / No Proposal (never counted as pass) | 2 (missed feasible: 1) | 2 (missed feasible: 1) |
| Invalid / Constraint-Violating Proposals | 0 | 0 |
| Nutrient Query Numeric Proximity Grounding | 12/12 | 12/12 |
| Locked-Dish Violations | 0/6 | 0/6 |
| Rollback Exact Restoration | 12/12 | 12/12 |
| Safety Refusal & Allergy Flagging | 12/12 | 12/12 |
| Mean Serving Delta (all validated swaps) | 8.34 (n=29) | 16.76 (n=29) |
| Mean Unchanged Dish Ratio | 0.5683 | 0.225 |
| Mean / P95 Turn Latency (ms) | 770.7 / 2235.7 | 1127.79 / 2421.8 |

**Strictly Paired Ablation Comparison (`n=29` turns where both systems produced validated swaps):** Full $L_1=0.35$ mean serving delta = `8.34` (median `8.0`) vs. No-$L_1=0.0$ mean serving delta = `16.76` (median `17.0`), a **`50.21%` reduction** (Full lower on `26` turns, tied on `1`, higher on `2`).

## 2. Matched 12-Scenario (24-Turn) Equal-Information Comparison: Guarded Stage 4 vs. Prompt-Only Qwen 2.5 3B

| Metric | Guarded Stage 4 (Full) | Prompt-Only Qwen 2.5 3B (Equal-Info, NF4 GPU) |
|---|---:|---:|
| Strict Protocol Pass Rate (timeouts count as failures) | 24/24 (100.0%) | 2/24 (8.3%) |
| Validated Swap Proposals (all revisions) | 9/12 (75.0%) | 0/12 (0.0%) |
| Validated Swap Proposals (given oracle feasible) | 9/9 (100.0%) | 0/9 (0.0%) |
| Valid Proven-Infeasible Refusals | 3 | 0 |
| Timeout / Missed Feasible (no proposal) | timeout=0, other=0 | timeout=0, other=3 |
| Invalid / Constraint-Violating Proposals | 0 | 8 |
| Unknown Food ID Proposals | 0 | 0 |
| Nutrient Query Numeric Proximity Grounding | 3/3 | 2/3 |
| Locked-Dish Violations | 0/3 | 0/3 |
| Rollback Exact Restoration | 3/3 | 0/3 |
| Safety Refusal & Allergy Flagging | 6/6 | 0/6 |
| Mean / P95 Turn Latency (ms) | 664.14 / 2091.98 | 46111.91 / 59922.13 |

### Disaggregated Revision Breakdown
- **Guarded Full (36 scenarios):** validated=29/42 (oracle_feasible=30), proven_infeasible_refusals=11, timeouts=2 (missed_feasible_timeout=1), invalid_proposals=0
- **Guarded No-L1 Ablation (36 scenarios):** validated=29/42 (oracle_feasible=30), proven_infeasible_refusals=11, timeouts=2 (missed_feasible_timeout=1), invalid_proposals=0
- **Guarded Subset (12 scenarios):** validated=9/12 (oracle_feasible=9), proven_infeasible_refusals=3, timeouts=0 (missed_feasible_timeout=0), invalid_proposals=0
- **Equal-Info Qwen Subset (12 scenarios):** validated=0/12 (oracle_feasible=9), proven_infeasible_refusals=0, timeouts=0 (missed_feasible_timeout=0), invalid_proposals=8
