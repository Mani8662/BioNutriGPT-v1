# Stage 2 ablation execution repair v2

24 November 2026. Preparation and development pilot verified; full 100-participant study NOT run.

## Implemented

- Full 600-task execution and full-cohort verification paths, using the same six research variants and evaluation cohort as v1.
- Balanced cyclic variant ordering across participants; sequential GPU calls and CUDA-synchronized wrapper timings. Untimed forward warmup; 12 CPU threads on this machine. Model loading and verifier time are excluded from individual execution wall_ms.
- Hash-bound current research code, statically imported local modules, models, preprocessing, calibrators, policies, variant definitions and participant manifests. Fail closed on changed bindings.
- Checkpoint validation before resume: identities, variant configuration, probabilities, numerical/score gates, normalized distances and bypass consistency. Corrupt files remain intact. Exceptions and quarantines require review, not automatic retries.
- Genuine reconstructed statistics. Primary overall-acceptance comparisons use Holm across five contrasts; numerical-pass and score-gate comparisons use a separate secondary family of ten. Continuous paired participant-bootstrap intervals are exploratory and unadjusted. These v2 definitions supersede the ambiguous v1 family description.
- Separate statistical verifier recalculates points, intervals, exact probabilities and Holm adjustments and rejects altered intervals.

## Executed checks

Fresh development pilot: 3 participants x 6 variants = 18 actual counterfactual runs, CUDA. Session measured 32.92 seconds including checkpoint checks, excluding model loading. Mean individual synchronized execution time 1.806 seconds. Extrapolation to 600 tasks: 18.1 minutes of execution alone; startup, verification and harder cases add time. This is not a guaranteed runtime range, and no unmeasured CPU runtime estimate is provided.

All 18 checkpoints validated; 45 paired intervals and 15 exact tests independently reproduced. Thirteen corruption checks, 18-record optimization-disabled resume, immutable checkpoint byte checks, and reference parity passed. Reference final targets matched the preserved v1 pilot for all three cases; an additional one-case parity test executed the research and comparison reference paths (two additional counterfactual executions). The old v1 statistics receipt was not genuine statistical reconstruction and is superseded.

The 100 evaluation identities were checked for overlap against the Stage 2 audit, comparison, adapter pilot, Stage 1 downstream study, Stage 3 budget cohort and three development cases: zero overlap in these checked sets. This does not claim disjointness from every conceivable historical use or independent clinical validation.

## Interpretation corrections

Identical full and no-sequential-filter outputs in three cases do not prove universal redundancy. Projection omission failures in a pilot do not establish necessity for every person. Dietary normalized-distance changes are dimensionless values, not percentage points. Model-score changes remain self-referential engineering endpoints. Filters 1-3 are ablated jointly; individual-filter causal contributions remain outside this six-variant design. Shared preprocessing and numerical audit helpers limit software independence. Names containing '88' in legacy functions do not establish the active feature count.

## Commands for Gemini (full run not launched here)

Dry run: `python stage2_ablation_runner_v2.py`

Full experiment: `python -u stage2_ablation_runner_v2.py --run-full`

Full independent verification: `python -u verify_stage2_ablation_v2.py --verify-full`

Do not use the v1 full-run or full-verifier commands. Do not re-run --prepare against frozen v2. A failed hash or quarantine needs investigation, not refreshed hashes or deletion. Pilot and full artifacts have separate directories. Full execution exceptions halt the run with saved evidence; do not call incomplete work completed.

Production code, model assets and previous study results were not modified. No full-cohort optimization or manuscript result insertion occurred.
