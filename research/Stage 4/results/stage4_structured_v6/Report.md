# Structured-response repair evaluation v6

2026-11-20. Four fresh-wording scenarios, six turns, four actual Qwen GPU calls, one saved Stage 2 profile and the real catalog/optimizer. Engineering regression only; not broad chatbot accuracy or clinical evidence.

The parser fills only inapplicable fields with neutral values by intent. Required action arguments remain required. Supplied conflicting values are never overwritten. Nutrient scope fields remain explicit to prevent omitted scope from becoming daily totals. Unknown fields, IDs, invalid types, source-action mismatch and negation remain rejected; revisions still require confirmation.

58 automated checks passed. Replaying the historical v5 failed removal payload now passes schema and source-action validation; original v5 evidence was preserved unchanged.

| Fresh scenario | Result |
| --- | --- |
| Omit a named food from next daily basket | Correct interpretation; confirmation produced one independently validated proposal. Tamper rejection, acceptance and rollback passed. |
| Overview of today's menu | Correct plan response. |
| Tell me about the first dish | Model incorrectly emitted a removal. Source-action evidence rejected it; clarification returned, no proposal. Information task not completed. |
| Do not omit a named food | Model incorrectly emitted a removal. Negation check rejected it; no pending confirmation or proposal. |

All six turns preserved the accepted basket/version before explicit acceptance. One proposal passed all recorded checks. This repairs missing inapplicable fields, not the model's semantic reliability. Two of four model calls were rejected for semantic errors. Cold removal interpretation took 38.87 seconds including loading; other GPU calls took 2.26–3.37 seconds. These few timings do not establish production latency.

Remaining work: improve language understanding and grounded information responses, test on a separately reserved dialogue set and multiple profiles, and complete browser/deployment checks. Keep confirmation and source-action guards enabled. Full raw outputs, timing, input/source hashes and checks are retained in turns.jsonl, protocol.json, binding.json and summary.json.
