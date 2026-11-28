# Verified checkpointed Stage 4 v4 results

Measured run verified against saved prompts, journals, stage checkpoints, frozen requests and source hashes.

- Full guarded run: 76/78 protocol passes; timeouts count as failures.
- Matched guarded vs Qwen: 24/24 versus 2/24.
- Validated proposals on feasible requests: 9/9 versus 0/9.
- Mean measured response time: 664.14 ms guarded; 46111.91 ms Qwen. Warm-model generation excludes loading and prompt construction; GPU work is synchronized.
- Same-input-state successful ablation pairs: n=25; mean serving changes 7.72 versus 16.56; descriptive reduction 53.38%.
- Scenario-cluster bootstrap 95% interval for serving-change difference (no penalty minus full): [6.72, 11.12].
- Verified all 24 Qwen prompts: complete eligible records only, 30 active nutrient fields, canonical 5% calorie check, canonical macronutrient bands and Na/(K+1) baseline check.
- All four completed-stage checkpoints and per-turn journals match the final detailed report.
- Source/input hashes are unchanged. Frozen requests match the logged requests. Actual model/device configuration is retained in JSON.

## Interpretation

Selected scripted LASI profiles; no Stage 1 external validation, national weighted effectiveness, clinical benefit or proof of general LLM superiority. Successful-pair ablation and empirical bootstrap are conditional analyses.
The earlier failed-export attempt remains separately archived and is not pooled with this run. No missing raw outputs were fabricated or reconstructed from aggregate counts.
