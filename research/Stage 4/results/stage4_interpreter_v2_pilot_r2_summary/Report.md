# Stage 4 revised Qwen interpretation pilot

Completed 2026-11-19. Six real local Qwen calls; same six synthetic requests as the first pilot.

- Correct complete interpretations: **2/6**.
- Structurally accepted: **4/6**, including two semantically incorrect interpretations.
- Rejected: **2/6**.
- First call including loading: 31.26 seconds.
- Five warm calls: mean 3.34 seconds, median 3.51 seconds, range 2.49–3.76 seconds.

| Request | Actual interpretation | Outcome |
|---|---|---|
| Protein in named dal | Remove the dal | Incorrect; passed structural validation |
| Energy in one serving | Clarification, with irrelevant protein code | Incorrect; passed current clarification branch |
| Protein at lunch | Correct nutrient and meal | Correct |
| Leave named dish out | Correct removal | Correct |
| Exact two-serving replacement | Added an unrequested lunch scope | Rejected |
| Do not remove named dish | Plan intent with missing fields | Rejected |

Full-field scoring was checked against explicit expected requests, including nutrient code, meal, quantity and per-serving scope. Original pilot flags checked only intent and food and must not be mistaken for complete semantic validation. Raw outputs and timing are retained in the r2 JSONL, with source snapshots and hashes alongside this summary.

The experimental interpreter remains unsuitable for dependable use and off by default. The confirmation gate would prevent immediate execution of the mistaken removal, but does not make the interpretation correct. This pilot called interpretation only; no solver or plan modification was performed. The strict parser remains insufficient to establish semantic correctness.

Recommended next work: route clear supported questions through deterministic tools first; use the model only for unresolved language; reduce candidate/context size; require action evidence and negation checks; validate unused fields for clarification/unsupported intents; expand semantic and multi-turn tests. Do not weaken validation merely to increase acceptance. Evaluate any revised approach on fresh held-out paraphrases as well as these regression cases. Another GPU run requires user approval.
