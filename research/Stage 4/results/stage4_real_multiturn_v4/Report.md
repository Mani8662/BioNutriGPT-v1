# Real-catalog multi-turn evaluation

Completed 2026-11-19. One saved Stage 2 sample, current 49-food catalog, real Stage 3 solver, five scenarios and nine turns. Temporary sessions only.

## Findings
- Food-specific protein and the follow-up calorie question resolved to the same food.
- Missing replacement quantity prompted clarification; “2 servings” produced a real proposal.
- The alternative “I'd rather have ... instead of ...” wording produced a real proposal.
- Negated replacement and unsupported dinner-specific replacement requested clarification.
- Natural “take it off my plan” failed: Qwen omitted required fields. The scripted confirmation also failed, because no pending interpretation existed and the confirmation text unnecessarily reached Qwen.

Both proposals excluded okra and contained exactly two servings of broccoli. Independently validated structural, portion and 30 active nutrient checks passed. Full 39-field completeness and clinical validation were explicitly false. The optimizer also changed other foods to preserve constraints: the requested swap is not guaranteed to leave the rest of the daily basket unchanged.

For both proposals, modified/tampered copies were rejected, original proposals were accepted into new versions, accepted baskets matched the proposals, and rollback restored the original basket. All nine turns left the accepted plan unchanged before explicit test acceptance. No uncaught per-turn exceptions were recorded. This is not a nine-of-nine conversational success claim: the natural-language scenario failed.

The two proposal responses took 1.19 and 1.24 seconds. Standard read/clarification responses took approximately 48–72 ms. The first failed model response took 37.29 seconds including loading; the second took 2.75 seconds. Times exclude subsequent verification/acceptance/rollback and HTTP/browser overhead.

## Evidence and limits
The nine-turn journal, initial plan and validation, fixed food pair, protocol and input hashes are saved here. Source hashes were checked after execution and matching code snapshots preserved. Independent review confirmed source exclusion and exact replacement quantity for both proposals, in addition to the saved acceptance/rollback checks.

This small engineering run uses one existing sample and regression wording. It does not establish population performance, clinical safety, general conversational reliability or real user satisfaction. No-proposal read/clarification turns are not feasibility tests. Raw rejected Qwen generations were not captured by this conversation runner; exception diagnostics reported missing replacement_id, servings and nutrients. No outputs were reconstructed.

## Next repairs
Handle confirmation/cancel messages locally when there is no pending interpretation. Improve supported take-off/leave-out wording without dropping negation or added constraints. Expand the frozen evaluation with fresh multi-turn cases; request approval before another independent GPU experiment. Source-grounded nutrition education remains separate unfinished work. Natural mode remains experimental and off by default.
