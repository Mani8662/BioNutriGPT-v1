# Stage 4 language-understanding repair v7

2026-11-20. Five scenarios, eight turns, one saved Stage 2 profile, real catalog and optimizer, three real Qwen GPU calls. Engineering regression, not a held-out population or clinical benchmark.

Changes: added a read-only describe intent with catalog-grounded portion, energy, protein and fibre facts. Missing nutrients are identified as unavailable. Explicit description wording routes locally; ordinal references ask for food selection rather than guessing displayed order. Negated omission/substitution requests route locally without creating pending actions. Interpreter prompt distinguishes informational descriptions from removals; strict validation, source-action evidence and confirmation remain enabled. These are generic operations over catalog IDs, not food-specific patches.

62 automated checks passed. Fresh real-catalog results:

| Scenario | Outcome |
| --- | --- |
| Describe first dish, then select a food | Asked for selection, then returned catalog description without GPU or solver |
| Never omit named food, then confirm | No change requested; no pending confirmation; no GPU |
| Brief factual overview of named food | Qwen correctly selected describe; grounded answer |
| Details regarding named food | Qwen correctly selected describe; grounded answer |
| Kindly omit named food, then confirm | Qwen selected remove; confirmed proposal independently validated; tamper rejection, acceptance and rollback passed |

All eight turns preserved accepted basket/version before explicit acceptance. Three of three model calls matched the intended operation and food in this small run. One returned proposal passed all recorded checks. First model call took 36.75 seconds including cold loading; warm calls took 2.59 and 2.63 seconds. No typical-latency claim follows from these few samples.

Historical v5/v6 failures remain unchanged. This is targeted regression after observed failures, not a reserved unbiased benchmark. General education/medical claims, broad paraphrase coverage, multiple patient-profile evaluation, browser QA and deployment hardening remain outside the completed work. See protocol.json, binding.json, turns.jsonl and summary.json for source hashes and complete raw outputs.
