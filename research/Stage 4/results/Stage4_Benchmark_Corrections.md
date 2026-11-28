# Stage 4 benchmark corrections

Post-hoc reanalysis of unchanged v3 logs. Does not repair or rerun the old Qwen prompt.

## Corrections calculated from saved logs

- Qwen truly feasible requests with no proposal, excluding timeouts: 7 (old ambiguous count was 9). Two invalid proposals occurred on feasible requests.
- Separate diagnostics: 2 malformed no-proposal outputs; 1 unexplained null on proven-infeasible request. Diagnostic counts may overlap feasibility categories.
- Same preceding-basket/frozen-request pairs: 25 of 29 successful trajectory pairs. Mean serving changes 7.72 vs 17.28, descriptive reduction 55.32%.
- Added participant/scenario-clustered bootstrap intervals (1,000 draws, seed 42), paired comparison intervals, and Wilson bounds on whole-scenario success. These quantify selected-case uncertainty, not national-population performance.
- A 100% observed pass rate yields a degenerate empirical bootstrap interval; Wilson bounds are provided under an independent-scenario assumption instead of implying no uncertainty.

## Corrected future runner

- benchmark_stage4_lasi_conversational_v4.py retains the v3 roster, frozen messages and initial baskets. It validates those initial plans; it does not silently reselect cases.
- Incomplete foods are excluded using the same active-nutrient completeness rule as Stage 3. Known zeros remain legitimate; unknowns never become prompt zeros.
- Prompt post-checks come directly from canonical validator output, with coefficient precision and the full selection/energy/change objective retained.
- Exact input state and full model prompts are recorded per turn. CUDA execution is required for the LLM run; synchronized GPU timing and context-limit checks are used.
- Source/catalog/scenario hashes and environment/config records are captured before execution; source changes are checked at the end. Actual model revision/device metadata are recorded when the model is loaded.
- Separate same-input-state and trajectory-level ablation summaries and uncertainty are saved in the machine-readable v4 report.

## Status

v3 files remain unchanged. No corrected-prompt v4 performance result exists yet. A new GPU run is necessary before replacing the v3 head-to-head figures in the manuscript. An already existing v4 frozen manifest is not silently overwritten.


## v4 execution and export correction
The first approved v4 attempt completed all model calls but failed while assembling the final report: a local variable shadowed the imported paired-ablation analysis function. Only the printed aggregate summaries were recoverable; detailed per-turn logs had not yet been saved. They are explicitly archived as incomplete evidence under Saved Model/stage4_v4_failed_export_attempt/RECOVERED_PRINTED_SUMMARIES_ONLY.json.
The exporter now uses an unambiguous function alias. Per-turn JSONL journals and atomic completed-stage checkpoints are written before final report assembly. Eight benchmark regression checks now pass, including an end-to-end simulated main/export test and checkpoint persistence test. The simulated test used temporary files and no model calls. A fresh scientific run requires user approval under the requested long-run workflow. The previous aggregate 2/24 Qwen pass rate is not a substitute for missing detailed evidence.


## 2026-11-18 — Checkpointed v4 rerun completed and verified
The explicitly approved checkpointed rerun completed in 1548.32 seconds. All four completed-stage checkpoints and all per-turn journals were saved before final export. The final JSON and Markdown reports were successfully generated. verify_stage4_v4_results.py verified frozen request equality, unique participant counts, disease balance, reported pass counts, all 24 Qwen prompt eligibility/rule checks, checkpoint consistency and unchanged source/input hashes. Actual Qwen model revision: aa8e72537993ba99e69dfaafa59ed015b17504d1, CUDA RTX 3050, 4-bit NF4.
Full guarded evaluation: 76/78 protocol passes, 29/42 validated revisions (29/30 oracle-feasible), 11 justified refusals and two timeout-related failures, with no invalid proposals. Matched 12-participant comparison: guarded 24/24 versus Qwen 2/24 protocol passes; validated proposals on feasible requests 9/9 versus 0/9. Qwen made eight invalid revision proposals and passed 2/3 targeted nutrient queries. Mean matched response time: guarded 664.14 ms versus Qwen 46111.91 ms; timing excludes model loading and prompt construction.
Same-input-state ablation: n=25 successful pairs; mean serving changes 7.72 with change penalty versus 16.56 without it (53.38% lower descriptively). Mean change difference 8.84, scenario-cluster bootstrap 95% interval [6.72, 11.12]. This successful-pair conditional analysis is not clinical acceptability evidence or national inference.
Authoritative aggregate summary: Saved Model/stage4_lasi_v4_verified_summary.md/.json. Full measured report: Saved Model/stage4_lasi_conversational_benchmark_v4_report.json. Measured hashed Python source files preserved under Saved Model/stage4_lasi_v4_measured_source_snapshot. The failed-export attempt remains separately archived and is not pooled with this complete run. Eight benchmark correction checks passed before execution; this does not assert that every unrelated application test passes.
