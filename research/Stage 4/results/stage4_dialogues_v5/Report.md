# Stage 4 real-catalog dialogue evaluation v5

Date: 2026-11-20. One saved Stage 2 profile, real 49-item catalog and optimizer, temporary sessions, real Qwen GPU interpretation for unresolved wording. This is an engineering evaluation, not a held-out population study or a general chatbot accuracy estimate. No alternate food search or retries were used.

| Check | Observed result |
| --- | --- |
| Conversations / turns | 9 / 17 |
| Accepted plan unchanged before acceptance | 17/17 turns |
| Returned proposals | 3: group exclusion, exclusion plus placement, two placements |
| Independent validation of returned proposals | 3/3 passed |
| Tampered proposals rejected | 3/3 |
| Original proposals accepted and rollback verified | 3/3 |
| Cancellation, unsupported cost condition, negative quantity | No proposals returned |
| Free-form model calls | 2 actual GPU calls |
| Free-form plan question | Correctly interpreted and answered |
| Free-form removal question | Rejected: missing required JSON fields; no proposal |

The removal model output correctly identified the intended food and remove intent but omitted replacement_id, servings and nutrients. Current schema requires every field, so validation rejected it. Its subsequent confirmation found no pending action. This is an incomplete user task, not a successful revision or proof of optimizer infeasibility. Historical results are retained without rescoring them after repairs.

Latency: the plan question took 66.38 seconds including cold loading; removal took 2.91 seconds with the model already loaded. These two timings do not establish typical or percentile latency. Deterministic turns took approximately 0.05–0.71 seconds in this run.

Next work: improve intent-specific structured generation or explicitly define neutral optional fields while retaining required action arguments, unknown-ID rejection, source-action evidence and confirmation. Validate any repair on fresh phrasing and preserve this failed result. Broader patient-profile coverage and browser end-to-end tests remain outstanding.

Artifacts: protocol.json freezes selected dialogues and scope; binding.json records source hashes; initial.json records the starting plan; turns.jsonl retains all responses, raw model outputs, timings and checks; summary.json gives machine-readable counts. The two nutrient answers were food-scoped, including a per-serving follow-up; no independent arithmetic assertion was included in this run.
