# Stage 4 Conversational Meal-Revision Benchmark on Real Indian Adults (LASI Wave 1)

- **Catalog:** `sourced-cooked-v5-veg-nonveg` (49 sourced dishes; 20 INDB + 29 USDA SR Legacy)
- **Full Evaluation Suite:** 30 multi-turn clinical scenarios (60 turns) across real LASI Wave 1 Indian adults and all 3 dietary modes (`pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`)

## Table A: Full 30-Scenario (60-Turn) Minimal-Change $L_1$ Ablation

| Metric | Guarded Stage 4 (Full, $L_1$ Penalty = 0.35) | Guarded Stage 4 Ablation ($L_1$ Penalty = 0.0) |
|---|---:|---:|
| Task Completion Rate | **100.0% (60/60)** | 100.0% (60/60) |
| Active Constraint Violation Rate | **0.0% (0/30)** | 0.0% (0/30) |
| Unsupported Numeric/Food Claim Rate | **0.0% (0/12)** | 0.0% (0/12) |
| Unintended Locked-Dish Change Rate | **0.0% (0/6)** | 0.0% (0/6) |
| Safety & Conflict Accuracy | **100.0% (16/16)** | 100.0% (16/16) |
| Mean Serving Delta per Swap (lower = less disruption) | **9.46** | 16.08 |
| Mean Unchanged Dish Ratio per Swap (higher = more stable) | **51.0%** | 28.8% |
| Mean Turn Latency (ms) | 917.5 ms | 1047.4 ms |

## Table B: Matched 12-Scenario (24-Turn) Head-to-Head vs. Real Prompt-Only Qwen 2.5 3B (NF4, RTX 3050 GPU)

| Metric | Guarded Tool-Backed Stage 4 | Prompt-Only Qwen 2.5 3B (NF4) |
|---|---:|---:|
| Task Completion Rate | **100.0% (24/24)** | 8.3% (2/24) |
| Active Constraint Violation Rate | **0.0% (0/12)** | 0.0% (0/12) |
| Unsupported Numeric/Food Claim Rate | **0.0% (0/3)** | 100.0% (3/3) |
| Safety & Conflict Accuracy | **100.0% (8/8)** | 25.0% (2/8) |
| Mean Turn Latency (ms) | **887.7 ms** | 6969.8 ms |
