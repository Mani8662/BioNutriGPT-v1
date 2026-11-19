# Stage 4 chatbot correctness repairs — 2026-11-18

This records software repairs following the chatbot audit, not clinical validation.

## Behavior changes

- Patient-level reported allergies survive acceptance and rollback. Initial intake
  allergies and existing warning-only records are retained when creating sessions.
- Known allergen-containing foods are excluded from revision solves and checked
  again at acceptance. Incomplete allergen records still require review: numeric
  acceptance never removes the allergy warning or establishes allergen safety.
- Allergy declarations without a session create a persistent session returned to
  the frontend. Allergy questions and tested negative statements do not mutate
  patient preferences.
- Saved exclusions, preferred foods, food quantity locks, ingredient exclusions,
  and dish limits carry into subsequent revision requests. Rollback restores the
  target version's plan preferences while retaining patient-level allergies.
- Persistent-session acceptance requires the exact server-stored pending proposal
  for the current version, followed by independent Stage 3 validation. Reporting
  an allergy invalidates previously pending proposals.
- The frontend derives allergy warnings from authoritative accepted/restored
  dossiers and refreshed session state, rather than unconditionally clearing them.
- Optional chatbot LLM rephrasing returns deterministic verified text. Number-set
  equality cannot enforce semantic preservation, so unrestricted rewriting is
  disabled. The separate Stage 4 report compiler is unchanged.

## Verification

`python -B -m unittest verify_stage4_chat_repairs verify_stage3_production -v`

24 checks passed: 11 targeted chatbot regression checks and 13 existing production
checks, including the real-catalog sample/food-swap check. Synthetic fixtures and
temporary session directories are used for the new regression tests.

Frontend: `node_modules/.bin/tsc.cmd --noEmit` passed.

No browser interaction test or new real-LLM benchmark was performed. The existing
16-turn conversational benchmark report describes the previous implementation;
its percentages are not new measurements of this repair. These checks establish
the tested software behavior, not clinical safety or publication readiness.
