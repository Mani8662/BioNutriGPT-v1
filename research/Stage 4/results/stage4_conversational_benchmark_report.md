# Stage 4 Conversational Meal-Revision Benchmark Report

- **Catalog:** `sourced-cooked-v4-lactoveg-complete` (31 sourced dishes)
- **Suite:** 8 frozen multi-turn scenarios (16 turns total)
- **Research Question:** Can conversational meal revision improve practical personalization while preserving declared constraints across repeated changes?

## System Comparison

| Metric | Guarded Tool-Backed Stage 4 | Prompt-Only Qwen 2.5 3B (NF4) |
|---|---:|---:|
| Task Completion Rate | 93.8% (15/16) | 6.2% (1/16) |
| Active Constraint Violation Rate (lower is better) | 0.0% (0/7) | 0.0% (0/7) |
| Unsupported Numeric/Food Claim Rate (lower is better) | 0.0% (0/4) | 75.0% (3/4) |
| Unintended Locked-Dish Change Rate (lower is better) | 0.0% (0/2) | 0.0% (0/2) |
| Safety & Conflict Handling Accuracy | 100.0% (8/8) | 14.3% (1/7) |
| Mean Turn Latency (ms) | 214.4 ms | 18500.4 ms |
| P95 Turn Latency (ms) | 804.4 ms | 27284.9 ms |

## Scope Note
All constraint checks verify the declared Stage 2 arithmetic/portion rules on the 31-entry sourced catalog; clinical validation has not been established.