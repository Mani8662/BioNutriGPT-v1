# BioNutriGPT paper: evidence map and remaining work

Prepared 2026-11-18. Working manuscript: `BioNutriGPT_Manuscript_v2.md`.
Original root manuscripts are preserved. This directory is the new working draft,
not a submission-ready release. Journal quartile is not a property established
by completing the application or rewriting its claims.

## Primary research position

**Research question:** Can tool-backed, stateful meal-plan revision complete users'
requests while retaining declared nutritional and preference constraints?

**Candidate contribution:** an auditable revision workflow connecting food
provenance, whole-portion optimization, independent acceptance validation, and
persistent plan state. Whether this combination improves outcomes is an
experimental question. Do not claim a new Transformer, a new MILP algorithm,
causal risk reduction, an allergen-free diet, or the first nutrition chatbot.

The source review confirms direct overlap with ChatDiet and an integer
goal-programming preprint. The literature section is a starting comparison,
not a systematic review or proof that the remaining combination is novel.

## Evidence map

| Claim | Source | Current use |
|---|---|---|
| Development N=23,694; temporal holdout N=7,046; zero SEQN overlap | Direct CSV audit in `evidence_snapshot.json` | Current descriptive evidence; does not reconstruct every cohort exclusion |
| 87 predictors and exact feature names | Training-code AST and CSV headers in snapshot | Current architecture description |
| Unweighted Stage 1 metrics | `Saved Model/stage1_test_evaluation_report.json` | Saved temporal-holdout results; model predictions were not regenerated for this draft |
| Classical Stage 1 comparisons | `Saved Model/tripod_baseline_evaluation_report.json` and its evaluation script | Context only; training-budget/split differences prevent a strong superiority claim |
| Survey-weighted CIs and calibration | Same report and `evaluate_tripod_and_baselines.py` | Requires audit: participant bootstrap ignores survey clusters/strata; missing-weight imputation and split mismatch need repair |
| 49 catalog entries: 20 INDB variants + 29 USDA proxies | Current catalog and evidence snapshot | Current descriptive data; not 49 laboratory-validated Indian dishes |
| 43 entries have no declared missing active field | Catalog missingness and current rule roles | Pre-filter completeness, not per-patient feasibility or food-assay accuracy |
| 29-entry, 20-case LP/MILP/Qwen comparison | `Saved Model/matched_milp_lp_qwen_benchmark_report.json` | Historical result only; seven scored bounds and older soft-deviation policy |
| 31-entry, 16-turn conversational comparison | `Saved Model/stage4_conversational_benchmark_report.json` | Historical result only; two base profiles; predates latest repairs |
| 24 software checks passed | `Stage4_Chatbot_Repair_Report.md`; two verification suites | Software verification, including synthetic regression checks; not 24 independent clinical cases |
| 100 independent measured Stage 4 executions | `benchmark_stage4_edge_cases_and_statistics.py` assigns counts/latencies | Excluded as empirical evidence |
| Dietitian collaboration, preregistration, triple blinding | Protocol file only | Planned work; no such completed activity is claimed |
| Indian prediction validation or longitudinal benefit | No completed supporting artifact established | Not claimed |

## What must be completed before submission

### 1. Freeze one current version

Freeze code commit, checkpoint hash, preprocessing artifacts, 49-entry catalog,
rule version, model prompts, hardware, and study cases. The snapshot hashes some
existing evidence files but is not yet a complete executable release manifest.
Archive the earlier 29/31-entry results as historical studies. Do not combine
their percentages with current results.

### 2. Run the current-system study

The next independent execution task is a benchmark on the frozen current
version. Agree its sample and endpoints before running it; do not use new test
cases repeatedly for development and still describe them as untouched.

Use matched continuous and integer optimizers with identical food data, active
rules, category rules, objectives, and time limits. Include the continuous model
with binary dish-selection variables to separate portion integrality from dish
selection. Include a prompt-only real LLM baseline and a tool-backed orchestration
variant if making claims about LLM orchestration. Different model roles are
different systems and should be named accordingly.

For chat, vary profiles, paraphrases, repeated edits, negative statements,
ambiguous food names, locks, dish caps, allergies introduced after a proposal,
stale acceptance, rollback, solver timeouts, and missing data. Predefine allowed
clarifications and correct refusals. Report mode-specific results for vegetarian,
combined, and non-vegetarian planning when those modes are included.

Primary endpoint: **joint task success and validated constraint retention**.
Use every eligible request as a denominator and report feasibility, completion,
refusal, and timeout separately. Do not report a low violation rate as an
advantage if the system usually refuses or produces no plan.

Secondary endpoints: preference-retention failures, stale-proposal acceptance,
unsupported claims, serving-change magnitude, objective value, integer portions,
and latency distributions. Use scenario-level resampling for dependent multi-turn
dialogues. Report uncertainty and exact counts rather than “zero hallucinations.”

Ablations: remove persistent preference memory, independent acceptance validation,
and the revision change penalty separately. Those comparisons test the proposed
contribution more directly than adding more architectural complexity.

### 3. Repair prediction reporting

Recover original training/validation row IDs and fit any calibrator on genuinely
reserved predictions. Check deployment uses the same calibrated outputs if those
are reported. Align comparator training/tuning budgets and held-out evaluation.
Audit survey weights and design-aware uncertainty. Generate paired comparisons
and subgroup summaries from shared predictions. Add direct BMI assessment as an
obesity comparator when height and weight are already available.

Do not launch new expensive model training as part of manuscript editing; it is
a separate study task under the user's requested execution workflow.

### 4. Make the scope clinically honest

Review Stage 2 thresholds and food-proxy mappings. State which targets are project
choices and which have a directly applicable reference. Keep allergy warnings
because all 49 entries retain possible unknown allergens. Limit current claims
to tested numerical/structural properties. Expert acceptability review would
support stronger usability claims; a clinical-effectiveness claim requires
different evidence.

### 5. Finish the submission package

Obtain real author names, affiliations, corresponding author, contributions,
funding/conflict disclosures, institutional ethics determination, and repository
release link. Complete a page-mapped TRIPOD+AI checklist for prediction reporting
and a separate risk-of-bias/applicability assessment; checklists do not establish
that bias is absent. Finish related-work coverage, architecture/flow figures,
source permissions, and journal-specific formatting after the results are frozen.

## Draft controls

- No guaranteed Q1/Q2 acceptance or quartile claim.
- No prospective clinical trial, Indian transfer validation, or human review is
  implied by temporal testing, synthetic profiles, or unit tests.
- Do not label routine lipid-assisted screening “non-invasive.”
- Do not label a numerical search cap as a clinical upper intake limit.
- Do not equate whole portions with demonstrated acceptability.
- Do not describe disabled optional chatbot rewriting as a measured real-LLM
  conversational feature.
- The current manuscript is about 4 stages; the strongest proposed contribution
  is the controlled revision workflow, not the number of stages.

## LASI evaluation update 2026-11-18

A real-profile Indian pilot now exists: 100 unique LASI profiles, 25 per disease stratum, evaluated in three modes (300 solves). Acceptance is 91/100 vegetarian, 96/100 combined and 98/100 non-vegetarian. This updates the earlier statement that only historical catalog comparisons existed. It does not complete the matched continuous-versus-integer study.

The saved chatbot suite contains 30 diabetes-only scenarios, not four disease groups. Its 60/60 scripted success includes four no-proposal swap outcomes. Change penalty reduces mean serving delta from 16.08 to 9.46 among 26 valid swaps per arm. The Qwen subset is not equal-information: most nutrient coefficients and full plate/objective rules are omitted from its prompt. Repair selection, oracle timeout/refusal scoring, full evidence logging and equal-information comparison before relying on these results for strong superiority claims.

Stage 2 uses assigned 0.85/0.10 disease-routing values and assumed sodium/potassium baselines; this is adapted rule-based profile testing, not Stage 1 external validation or full Stage 2 counterfactual validation. The balanced sample and eligible-cohort summaries are unweighted, not national estimates.

The current audit reran 11 chatbot regressions: ten pass and one allergy-question/declaration expectation fails. Do not describe the historical 24-check pass as proof of the current source. See Saved Model/LASI_Evaluation_Audit.md in the project and its aggregate copy in the submission pack. No scientific benchmark was rerun in this audit.


## 2026-11-18 — Checkpointed v4 rerun completed and verified
The explicitly approved checkpointed rerun completed in 1548.32 seconds. All four completed-stage checkpoints and all per-turn journals were saved before final export. The final JSON and Markdown reports were successfully generated. verify_stage4_v4_results.py verified frozen request equality, unique participant counts, disease balance, reported pass counts, all 24 Qwen prompt eligibility/rule checks, checkpoint consistency and unchanged source/input hashes. Actual Qwen model revision: aa8e72537993ba99e69dfaafa59ed015b17504d1, CUDA RTX 3050, 4-bit NF4.
Full guarded evaluation: 76/78 protocol passes, 29/42 validated revisions (29/30 oracle-feasible), 11 justified refusals and two timeout-related failures, with no invalid proposals. Matched 12-participant comparison: guarded 24/24 versus Qwen 2/24 protocol passes; validated proposals on feasible requests 9/9 versus 0/9. Qwen made eight invalid revision proposals and passed 2/3 targeted nutrient queries. Mean matched response time: guarded 664.14 ms versus Qwen 46111.91 ms; timing excludes model loading and prompt construction.
Same-input-state ablation: n=25 successful pairs; mean serving changes 7.72 with change penalty versus 16.56 without it (53.38% lower descriptively). Mean change difference 8.84, scenario-cluster bootstrap 95% interval [6.72, 11.12]. This successful-pair conditional analysis is not clinical acceptability evidence or national inference.
Authoritative aggregate summary: Saved Model/stage4_lasi_v4_verified_summary.md/.json. Full measured report: Saved Model/stage4_lasi_conversational_benchmark_v4_report.json. Measured hashed Python source files preserved under Saved Model/stage4_lasi_v4_measured_source_snapshot. The failed-export attempt remains separately archived and is not pooled with this complete run. Eight benchmark correction checks passed before execution; this does not assert that every unrelated application test passes.
