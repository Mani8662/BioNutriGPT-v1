# Stage 1 / Stage 2 GPU agreement

2026-11-20. Existing backend sample tested with supplied dietary ratios and with explicitly recomputed ratios. Both paths ran real production model inference and Stage2 counterfactual optimization on CUDA, using explicitly selected default assets and isolated output files.

| Check | Supplied ratios | Recomputed ratios |
|---|---|---|
| Diabetes, obesity and CVD baseline score differences | All exactly 0 | All exactly 0 |
| Model/preprocessor binding | Matches | Matches |
| Operating cutoffs | Match | Match |
| Input-ratio audit across stages | Matches | Matches |

Final nutrient envelopes were identical between the two cases. First case took 5.18 seconds; second 1.87 seconds, including Stage1 and Stage2 calls. These are engineering observations, not latency benchmarks. This confirms shared normalization and inference agreement on one fixture, not clinical validity, repaired-v2 equivalence, or broad stability. Production still uses legacy model assets, raw sigmoid probabilities and fixed thresholds. No migration occurred.

Protocol/source hashes, stage-specific outputs and machine-readable results are retained beside this report. No production dossier was overwritten. Next research work: frozen multi-profile stability/guideline-target comparison and Stage3 realization; model/calibration migration requires a separate explicit compatibility assessment. CLI anthropometry defaults and threshold discontinuities remain outstanding.
