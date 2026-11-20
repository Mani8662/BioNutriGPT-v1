# Hybrid Stage 4 pilot results

Completed 2026-11-19. Eight synthetic conversation requests; real local Qwen called only for three unresolved requests. Five requests bypassed the model. Revision solver was mocked and sessions were temporary.

**7/8 matched expected handling**, including clarification/refusal behavior. This is not a 7/8 nutrition-question answer rate or general chatbot accuracy estimate.

| Case | Handling | Result |
|---|---|---|
| Protein in named food | Backend calculated 20 g from synthetic data | Expected |
| Energy per serving | Backend calculated 200 kcal from synthetic data | Expected |
| Do not remove food | Clarification, no model or solver | Expected |
| Exact replacement | Called mocked solver once with controlled request | Expected routing; feasibility not tested |
| Tired of food, leave it out | Model correctly interpreted removal; confirmation requested | Expected; not executed |
| Why is this a good choice? | Clarification instead of invented explanation | Expected boundary; educational capability still absent |
| Prefer another food instead, two servings | Model added lunch; validator rejected | Failed understanding; safely clarified |
| Do not leave food out | Clarification, no model or solver | Expected |

## Timing

Five standard-path responses averaged 19.34 ms, range 13.51–30.39 ms. These use synthetic data and a mocked solver, so they are not production optimizer latency estimates.

First model-path response took 29.59 s including loading. The other two averaged 2.65 s. These are Python conversation-function timings, excluding HTTP/browser overhead.

## Verification and limits

The saved journal contains eight complete entries. Model-call positions and solver-call counts were checked; the only solver call was the explicit replacement scenario. The natural removal stayed behind its confirmation gate. Raw model outputs, errors and elapsed time are retained. Source snapshots and hashes are included alongside this report.

The pilot does not exercise confirmation acceptance, real solver feasibility, cross-turn sessions, real patient plans, browser interaction or clinical outcomes. Prior synthetic regression tests cover some of these control paths but do not replace broader evaluation. One model error remains; general education is not implemented. Keep natural wording experimental and off by default.

## Next work

Resolve supported replacement paraphrases with explicit quantity/food/meal extraction and reject unsupported extra scope. Do not silently discard a model-invented restriction. Add fresh held-out paraphrases and multi-turn cases before another approved model evaluation. Build source-grounded explanations separately for questions such as why a food was selected.
