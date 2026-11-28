"""Documentation-only reconciliation; no model/solver imports or benchmark runs."""
from pathlib import Path
import hashlib
import json
import shutil
import re

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'Documentation_History' / '2026-11-26_before_sync'
ARCHIVE.mkdir(parents=True, exist_ok=True)
names = ['Current_Model_Explanation.txt', 'Progress.txt',
         'Challenges faced and how we solved them.txt',
         'challenges_faced_and_how_we_solved_them.txt', 'Stage 3 Challenges.txt']
for name in names:
    src = ROOT / name
    dst = ARCHIVE / name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

overview = '''BIONUTRIGPT — CURRENT ARCHITECTURE AND COMPLETE EVIDENCE GUIDE
Updated: 26 November 2026

READING GUIDE AND EVIDENCE POLICY
This is the current account, replacing the accumulated historical snapshots formerly
in this file. Sections 1–8 explain the application and results. Section 9 contains
the numerical tables from the named experiment reports, with source paths and scope.
Section 10 contains the complete predictor inventory and tuning configurations.
Original text is preserved in Documentation_History/2026-11-26_before_sync/.
Audit/ and older manuscript exports are historical copies, not current app specifications.
This update reviewed code and saved reports; it did not rerun experiments or certify
every underlying record anew. A recorded verification is attributed to its original run.
Participant counts, optimization tasks, bootstrap draws and dialogue turns are different
units. Repeated modes/arms are not independent patients. Corrections supersede earlier
claims; cross-cohort or cross-release scores are not direct before/after comparisons.

1. WHAT THE APPLICATION DOES
Input: a structured form or TXT upload; both feed the same validated raw-input contract.
The current repaired application requires 87 finite raw-unit features, valid encoded
categories, age >=20, and no nonzero missingness flags. Weight, height, activity and
food preferences supply planning metadata. Ratios and lipid derivatives are recomputed.
Incomplete input is rejected rather than filled with the demonstration patient's values.
Backend: FastAPI/Pydantic. Frontend: React 18, TypeScript, Vite and Tailwind.
Main routes: /api/parse-txt, /api/upload-txt, /api/screen, /api/optimize,
/api/compile, /api/compile/llm, /api/pipeline/full, /api/pipeline/upload,
/api/chat, /api/chat/accept, /api/chat/rollback and /api/sessions.

Flow:
Form/TXT -> validation and canonical preprocessing -> Stage 1 calibrated screening
 -> Stage 2 hypothetical nutrient search, joint projection and final checks
 -> Stage 3 discrete food optimization and independent basket validation
 -> Stage 4 grounded explanation and controlled conversational revisions.
A revision is a proposal. Explicit acceptance validates and stores a new plan version.
Rollback restores a recorded plan; patient-level allergy warnings remain persistent.
Observed patient measurements, hypothetical nutrient targets and delivered food totals
are separate records. A revised meal is never written back as a measured laboratory test.
Failed Stage 2 certification blocks downstream planning; failed Stage 3 validation does
not produce an accepted plan. A feasible daily basket is not proof of clinical safety.

2. STAGE 1 — CURRENT PREDICTION ARCHITECTURE
Application default: stage1_deployment.py selects repaired_v1, unless an explicit
BIONUTRIGPT_STAGE1_RELEASE=legacy override is set before process startup.
Bundle: Saved Model/deployment/repaired_stage1_v1/. Checkpoint, preprocessing,
feature order and reserved-set calibrators are hash-bound. Stage 1 and Stage 2 share it.
Selected comparison trial 03 preserves existing interfaces; selection does not establish
the custom model as the best classifier. This describes code defaults, not a newly
verified running server. Historical standalone inference utilities can still be legacy.

Feature groups: 5 lipid measurements/derivatives, 39 nutrient quantities plus 7 dietary
ratios (46 diet features), 19 lifestyle features, 17 demographic/missingness features.
Total 87. HbA1c and glucose are NOT prediction inputs. HbA1c is used in the diabetes
outcome definition. Lipids require a blood sample: call this non-glycaemic laboratory-
assisted screening, not non-invasive screening. Exact feature names appear in section 10.

Layers:
- Feature-specific scalar tokenization plus modality embeddings: 87 x 128 tokens.
- Three pre-normalized Transformer encoder layers: 4 attention heads, feed-forward
  dimension 256, GELU, dropout 0.1.
- Learned query pooling produces one 128-dimensional vector for each of four modalities.
- Concatenation gives 512 dimensions. Four-expert top-2 sparse MoE gives a 256-dimensional
  patient state; balancing auxiliary loss coefficient in the comparison recipe is 0.01.
- Diabetes and CVD heads use patient state plus bio context (384 ->64 ->1).
  Obesity uses patient state plus diet and lifestyle context (512 ->64 ->1).
- Reserved-set logistic calibration transforms logits to reported probabilities.
Full model: 1,295,623 trainable parameters. Attention is not a causal explanation.

Repaired study: NHANES 2007–2016 development N=24,136, split into training 16,894,
validation 3,621 and calibration 3,621. Temporal 2017–March 2020 N=7,004.
Temporal positive counts: diabetes 1,301; obesity 3,057; CVD 845.
Diabetes: reported diagnosis or HbA1c >=6.5%; negatives need observed eligible evidence.
CVD: any of five reported conditions; negative requires all five no responses.
Obesity: measured BMI >=30, secondary exploratory outcome. This is prevalent-state
screening, not incident disease forecasting. Source lineage starts at the merged file;
complete recall/laboratory/outcome selection limits generalization.
Training-only preprocessing uses repaired transforms, clipping/scaling and rounding.
Six families each had four configurations and seed 42, selected on validation mean
diabetes/CVD AUROC. Neural training: unweighted BCE, AdamW, batch 64, at most 40 epochs,
early stopping patience 7, gradient clipping 1.0; selected BioNutriGPT lr=0.0003,
weight decay=0.001. Do not substitute the legacy trainer's weighted-loss recipe.

Main UNWEIGHTED temporal AUROC (diabetes / obesity / CVD):
Logistic regression     0.7965 / 0.7445 / 0.8291
Histogram boosting      0.8124 / 0.7557 / 0.8252
CatBoost                0.8138 / 0.7578 / 0.8288
MLP                     0.7996 / 0.7519 / 0.8275
BioNutriGPT             0.8107 / 0.7467 / 0.8288
FT-Transformer          0.8109 / 0.7539 / 0.8350
BioNutriGPT calibrated Brier: 0.120914 / 0.204624 / 0.087353.
FT-Transformer is a tabular reference architecture; it does not use this model's modality
query pooling/MoE design. CatBoost leads diabetes/obesity point estimates; FT leads CVD.
20,000 paired participant bootstrap draws and 20 adjusted primary contrasts detect
BioNutriGPT diabetes AUROC gains over LR/MLP and diabetes Brier gain over LR, but no
overall advantage over the stronger comparators. Equal configuration counts are not
equal compute budgets. Conditional fitted-model uncertainty is not retraining uncertainty.
Warm single-participant raw inference: LR 0.298 ms, HGB 10.586, CatBoost 3.269,
MLP 0.673, BioNutriGPT 17.032, FT 3.065. Devices differ; this is not a matched speed ranking.
Four-configuration training totals: 3.94, 19.70, 111.06, 59.52, 650.62, 488.32 seconds
respectively; exclude startup/calibration/export and interrupted attempts.

Five variants x five seeds =25 fits:
Full mean AUROC diabetes/CVD/obesity =0.8090/0.8316/0.7501.
No MoE =0.8086/0.8315/0.7521; parameters 700,675 (about 46% fewer).
Mean pooling =0.8075/0.8305/0.7509.
No modality embeddings =0.8073/0.8303/0.7483.
No head shortcuts =0.8074/0.8313/0.7485.
None of 16 adjusted primary contrasts establishes a full-model advantage. Absence of a
detected difference is not equivalence. Full MoE remains in the deployed bundle; the
ablation does not itself deploy the smaller architecture.

Robustness: six frozen families, complete diet/lipid removal, two 20%-diet-cell masks,
diet/lipid/joint noise. Training-fitted imputation and derived-feature rebuilding used.
BioNutriGPT diabetes AUROC: baseline .8107; diet missing .8008; lipids missing .7607;
both .7441; joint noise .8094. Corresponding Brier: .120911 (rounded report baseline
is .120914), .124530, .132341, .134954, .121111. Use exact source table for precision.
All-diet removal changes 12.78% of obesity classifications at fixed 0.5.
The corrected statistical audit uses 10,000 resamples: 2,688 interval pairs/5,376
endpoints independently reproduced. This differs from the 20,000-draw main comparison.
Missing-modality research does NOT mean the deployed form accepts missing inputs.

Subgroups: sex, age20–40/41–65/>65, five race/ethnicity groups. Six models, 7,004 rows.
Unweighted subgroup verification: 360 calibration rows, 360 model intervals and 300
paired intervals; 2,000 participant resamples. Young CVD has only 37 positives.
Survey reference: WTDRD1PP, 24 strata, 49 PSUs, R survey JKn, 25 design degrees of freedom.
594 overall/subgroup metric rows and 495 contrasts; all 54 overall estimates/SEs/CIs
and 45 overall contrasts matched the Python implementation within 1e-10.
WEIGHTED BioNutriGPT AUROC: diabetes .8402 [.8203,.8601], CVD .8468 [.8281,.8656],
obesity .7330 [.7077,.7583]. Do not mix these with the main unweighted table.
Weighted calibration offset/slope: diabetes .0512/1.0873, CVD .0273/.9602,
obesity .2447/.9465. Obesity shows average underprediction; no fairness guarantee.
DCA: thresholds .05–.50 in .01 steps; at .20 net benefit .05269 diabetes/.03133 CVD.
Paired CatBoost/FT contrasts include zero there. These are hypothetical further-assessment
curves on a cohort including diagnosed disease, not validated treatment decisions.
108 calibration estimates, 360 bins, 736 curve estimates/intervals and 644 contrasts checked.

Deployment/agreement: 7,004 saved calibrated predictions matched (max error1.67e-16);
128 complete raw records had exact preprocessing parity; Stage1/2 baseline binding and
probabilities matched, six invalid/mixed-release cases rejected, API200/422 and frontend
build passed. One integration case blocked, another accepted. Older one-fixture legacy
agreement is historical, not the scope of repaired deployment evidence.

Downstream prediction-added-value: 100 participants x3 modes x16 arms =4,800 solves;
rules-only plus three frozen models at .05/.10/.20/.30/.50. Rules-only feasibility296/300
(98.67%); BioNutriGPT95.80%, CatBoost96.07%, FT96.00% across1,500 solves/model.
BioNutriGPT vs rules: -2.87pp CI[-4.33,-1.53]; vs CatBoost -.27pp[-.60,0]; vs FT
-.20pp[-.53,.13]. No prediction arm rescued an infeasible rules-only case.
Stricter routing rules can reduce feasibility; this does not establish poorer health
outcomes. This historical adapter study is distinct from the deployed counterfactual
pipeline and the later Stage2 target-construction experiment.
Indian external prediction validation remains unavailable. LASI is a planning study.

3. STAGE 2 — NUTRIENT SEARCH AND CONSISTENCY
Implementation: stage2_counterfactual_engine.py. Frozen repaired model and calibrators;
only 39 dietary quantities are optimized, with derived ratios recalculated. Lab,
lifestyle and demographic observations are not optimized. Descriptive route cutoffs
are .5 for all three outcomes, not clinical decision thresholds. Application energy
routing uses predicted obesity; the separate research rule adapter uses measured BMI.
Do not conflate these two policies. Energy policy includes a 500-kcal reduction branch
and 1,200-kcal floor. These are project rules, not universal individual prescriptions.
Search: 50 Adam steps lr .05, weighted model scores (2*CVD+2*diabetes+obesity),
non-increase penalty weight50, dietary change penalty .001, and barrier term.
Joint projection enforces coupled bounds/energy/ratios; LP feasibility witness and
SLSQP projection have explicit statuses. A feasible LP fallback is not proof of
minimum-distance optimality. Sequential checks are re-audited after later edits.
Final score gate allows <=.0015 probability increase for each outcome.
Numerical consistency, model-score acceptance and food feasibility are separate.

Repair pilot32: before joint repair only3 fully accepted; 28 calorie-envelope failures,
7 score-gate failures (6 overlap). After repair25 accepted,7 score-gate rejected,
zero final nutrient/energy/ratio violations. 16 correction tests including6 solver-failure
tests; frozen rescoring max discrepancy5.74e-8. Development evidence, not a population rate.

Matched comparison100 x3 methods: numerical100/100 each; all-check acceptance rules48,
minimum-change70, guided77. Median time .031/.181/2.879 s; median normalized change
4.8190/1.8494/2.1992. Guided-minus-minimum acceptance+7pp CI[2,13], exactp .0390625;
extra mean displacement .2717 CI[.2325,.3153]. Shared score gate makes this partly
self-referential; rules-only here means no model-guided target search, not no prediction.

Ablations100 x6=600 tasks on a separate cohort:
variant                   numerical pass / all-check accepted / mean normalized L1
full                      100 /62 /8.6427
no model objective        100 /49 /7.5015
no change penalty         100 /64 /9.5591
no barrier                100 /65 /8.4803
no joint projection         8 / 7 /8.4644
no sequential filters     100 /62 /8.6427
Joint projection provides the clearest numerical contribution. Filters1–3 removed
together, not individually. Full62 here and77 in the comparison are different cohorts.

Cross-model check: 600 candidates plus100 baselines =700 evaluated rows per frozen
scorer (not700 unique people). Full-variant mean diabetes score changes: BioNutriGPT
-.0098 CI[-.0143,-.0054]; CatBoost+.0132[.0061,.0207]; FT-.0024[-.0113,.0067].
Gate passes62/42/53 out of100 respectively. Mean obesity/CVD changes are negative for
all three scorers; this does not establish individual agreement or health improvement.
171 pointwise paired intervals independently reproduced with10,000 resamples.

Stability: pilot3x8=24 separate; full100x8=800 optimization tasks. Identical-input
repeats matched; all800 passed numerical rules. Diet +/-1% mean scaled L1=.3294/.2978;
diet +/-5%=1.3066/1.3518. Lipid +/-5% caused6/5 routing crossings and up to500kcal
energy change. Repeatability is limited to this environment, not a universal guarantee.
Retrospective audit of600 perturbation comparisons found large single-nutrient changes
without routing/fallback, already present after Adam/box clipping: B2 2.7169->1.2920mg
under -1% diet; B12 5.3191->2.8101mcg under +1%. Numerical feasibility is not stability.
Gradient/clipping mechanisms remain unresolved without new trajectory experiments.

Threshold sensitivity is completed for prediction-dependent rule routing only; it is
not a full counterfactual-search threshold sweep. Individual filter ablations, broad
anthropometric/missingness stress and clinical validation remain outside current evidence.

4. STAGE 3 — FOOD CATALOG AND INTEGER OPTIMIZATION
Production module stage3_milp_solver.py; backend calls this bounds-based objective A.
Catalog49 =20 INDB recipe variants +29 USDA proxies, with documented nutrient gap sources.
43 fully cover active fields before preference filtering; 6 have missing active data.
Not49 laboratory-validated Indian recipes, not44 validated IFCT cooked dishes.
Unknown values remain unknown. IFCT-based historical CSV and estimated44-recipe builder
must not be confused with the sourced catalog used in current planning.
39 tracked nutrients:30 active,7 report-only,2 exposure-reporting fields. Full39-field
adequacy is not certified. Allergen information is incomplete; known allergens excluded
but absence of a flag cannot prove an allergen-free meal.

Variables: whole servings q_i in{0,1,2,3}, binary dish selection z_i; normally4–8 dishes,
with food-category and dietary-mode rules, persistent exclusions, locks and meal requests.
SciPy MILP/HiGHS enforces active nutrient bounds, calories and post-check ratios.
Current objective penalizes selection cost and calorie deviation; active nutrient-bound
slack is constrained to zero in the accepted formulation. Selection costs are engineering
weights, not measured market rupees. Revision penalties favor smaller serving changes.
Default time1.50s, production cap3s, relative MIP gap1%. A validated time-limited
incumbent may be accepted; solver status0 under1% gap is not exact-optimum proof.
No plan is accepted unless independently reconstructed totals and structural checks pass.
Meal placement is not per-meal glycaemic-load or nutrient-adequacy validation.

Experimental B adds normalized absolute deviation from Stage2 nutrient targets (lambda1),
while keeping hard constraints. Report-only/exposure fields are excluded; energy handled
separately. stage3_target_aware_solver.py and stage3_budget_solver.py are research paths.
The budget wrapper permits up to15s; five-second findings do not change app default1.5s.

Translation comparison:100 participants x3 target methods x3 modes=900 records;
893 returned baskets passed reconstruction. Guided vs minimum-change end-to-end
acceptance+7.33pp CI[2.00,13.33]. Earlier pooled exactp withdrawn because modes repeat
people. 881 valid time-limited solves,12 status0,3 reported infeasible,4 no-plan timeouts.

Initial target-aware comparison on reused100: A/B valid counts rules294/261,
minimum292/261,guided294/260 out of300 each. Mean target error A/B .9072/.4851,
.9434/.5347,.9199/.5133 on paired successes. 771/773 paired successful baskets differ.
Fixed run order and limited solver metadata restrict runtime/optimality conclusions.

Controlled budget study: fresh100 for that study,3 arms x3 modes x2 objectives x3
budgets(3/5/10s)=5,400 solves,2,700 paired tasks; randomized execution and cold starts.
At5s both A/B accept897/900 identical cells: observed agreement, not equivalence.
B counts at3/5/10s=891/897/896. Mean target errors A/B by budget .9520/.4808,
.9722/.4655,.9918/.4529 (conditional study summaries). Longer budgets improve B's
target objective but do not monotonically improve accepted-plan counts.
Matched B five-minus-three-second error differences -.0181 to-.0145 at+1.66–1.69s;
ten-minus-five -.0141 to-.0115 at+2.79–2.92s. Five-second guided/minimum end-to-end
difference+7.67pp CI[1.33,14.33], but Holm-adjusted per-modep=.6943; no adjusted
superiority established. This cohort's minimum-change gate69 differs from comparison70.

Robustness100 reused budget participants x3 modes x2 objectives x7 scenarios=4,200
records:3,000 actual solves +1,200 composition replays, not4,200 people or solver calls.
Scenario                       A valid / B valid (denominator300 per objective)
nominal                        299 /299
10% foods removed              296 /296
20% foods removed              224 /224
dairy excluded                 200 /200
legumes excluded               100 /100
5% nutrient composition        236 /215
10% nutrient composition       192 /159
Composition changes replay nominal baskets without reoptimizing; one fixed per-food
scale draw per severity is shared across participants/objectives. Missing values stay
missing. Not independent per-nutrient errors or repeated Monte Carlo realizations.
Nominal B-A target error -.4877 CI[-.5122,-.4629] among jointly accepted baskets.
Composition B-A acceptance -7pp CI[-14,0], -11pp[-19.67,-2.33]. Pointwise exploratory
cluster intervals; all21 within-mode Holm tests nonsignificant (minimump about.279).
All4,200 classifications reproduced by all-rule audit;94 margin intervals reproduced.
Under10% stress A/B violations: PUFA19/39,folateDFE1/16,niacin9/16,energyupper2/14,
potassium21/3,carbohydrate15/20; overlapping counts must not be summed as unique failures.
Dairy/legume exclusion can conflict with mandatory category rules; a clean refusal
is not proof that no medically acceptable alternative diet exists outside these rules.
No dietitian-rated whole-portion acceptability or real food-cost advantage established.
Older continuous-LP/LLM comparisons are historical, not a new fully matched comparator
study of the repaired target-aware pipeline; do not claim that experiment is complete.

5. STAGE 4 — CURRENT CHATBOT AND EVIDENCE BY SNAPSHOT
Current code: backend/conversation_service.py over chat_service.py, chat_interpreter.py,
compound_requests.py, food_groups.py and plan_store.py. Structured deterministic tools
answer totals, food descriptions, serving weights, named nutrients, meal scope,
screening limitations and version history. Catalog-driven groups support fish exclusion;
the whole group is excluded, rather than allowing tuna to be replaced by salmon.
Generic meal requirements support including a named catalog food/quantity at breakfast,
lunch or dinner. Feasibility is attempted, not promised. No egg-specific optimizer rule.
Limited compound grammar assembles supported constraints atomically; unresolved clauses
clarify instead of silently executing a partial request. Quantities are whole servings;
unsupported units/modifiers require clarification. This is not unrestricted understanding.
Same-version conversational context supports follow-ups and food/quantity clarification.
Deterministic-first routing bypasses Qwen for recognized intents. Experimental natural
wording is off by default; optional local Qwen2.5-3B interpretation must pass typed-schema,
catalog-ID, scope, negation and semantic guards. Generated JSON is not executed as code.
LLM-inferred mutations require interpretation confirmation and then plan acceptance.
Arithmetic and final factual responses come from tools; unrestricted chatbot paraphrase
is disabled. The optional separate report compiler is not the conversation interpreter.
Session state stores the pending proposal, version and preferences. Stale/tampered
proposals are rejected. Acceptance revalidates the exact server-stored proposal.
Allergies invalidate pending proposals and persist through rollback. Local JSON storage
still needs concurrency/access hardening for multi-user deployment.

Measured LASI dialogue benchmark belongs to its preserved earlier source snapshot:
36 profiles/78 turns, guarded76/78 protocolpasses (97.44%; scenario-bootstrapCI93.67–100).
42 revision requests:29 valid proposals,11 justified infeasibility refusals,2 timeouts;
one request judged feasible by the reference oracle was missed.
Matched12 profiles/24 turns: guarded24/24 vs solver-free Qwen2/24; feasible proposals9/9
vs0/9. Qwen had8 invalid proposals plus1 malformed response, NOT9 invalid proposals.
Matched mean measured response664.14ms vs46,111.91ms; Qwen warm generation excludes
loading/prompt construction. This compares different tool architectures, not equal agents.
Rollback12/12; read queries12/12; allergy/safety checks12/12; observed lock violations0/6.
Successful same-state serving-change ablation25pairs: full7.72 vs no-penalty16.56;
53.38% descriptive reduction, differenceCI[6.72,11.12]. Conditional on successful pairs;
not measured human acceptability and not evidence of zero hallucinations generally.

Later chatbot implementation checks (do not overwrite historical failures):
- Initial Qwen interpreter6requests:0/6 accepted structured interpretations.
- Revised6requests:4 structurally accepted but only2/6 full semantic matches;
  cold31.26s, five warm mean3.34s. Valid JSON is not correct interpretation.
- Hybrid8cases:7 expected-handling matches,3 Qwen calls, mocked revision solver.
- Real-catalog v4:5scenarios/9turns,2validated proposals; natural removal failed.
  Active plan unchanged before acceptance9/9; acceptance/tamper/rollback checked.
- v5:9scenarios/17turns,2Qwen calls,3validated proposals; free-form removal failed
  schema; active state unchanged17/17.
- v6:4scenarios/6turns,4Qwen calls,1proposal; named removal repaired, but information
  and negation misinterpretations were blocked by guards; active state unchanged6/6.
- v7:5scenarios/8turns,3Qwen calls,1proposal; two descriptions and one removal interpreted
  correctly; ordinal clarification/negation local; state unchanged8/8; acceptance,
  tamper rejection and rollback passed. Cold36.75s, warm2.59/2.63s, not latency estimates
  for a population.62 automated checks recorded at this repair stage.
These are targeted engineering checks on one saved profile, not a new held-out multi-
profile accuracy benchmark. Earlier19/25/35/49/56/58 checks are overlapping suites,
not additional independent tests to sum. No fresh browser validation claimed here.
Grounded broad nutrition education, reserved natural-dialogue/profile evaluation,
broader real-catalog coverage and multi-user hardening remain incomplete.

6. INDIAN PROFILE EVIDENCE AND TRANSFER LIMITS
LASI extraction66,859 merged records;65,368 eligible in this extraction. Selected100
profiles:25 diabetes-only,25 obesity-only,25 CVD-only,25 multimorbid;52male/48female,
29states/territories. This deliberately balanced subset is not nationally representative.
Adapted planning uses reported diagnoses and measured anthropometry, assumed sedentary
activity and rule-derived missing diet values. It does NOT run external Stage1 validation.
300planning solves: accepted91/100purevegetarian,96/100mixed,98/100nonvegetarian.
These results are distinct from NHANES counterfactual-search experiments. Do not pool.
Training in the US plus Indian food selection does not establish Indian predictive
validity. Required harmonized Indian predictor/outcome data remain unavailable.

7. WHAT IMPROVED, WHAT DID NOT, AND COMPARISON RULES
Legacy -> repaired Stage1: labels/splits/preprocessing/calibration and deployment binding
repaired. Old N=7,046 and AUROC .95/.84 values belong to different pipelines; not evidence
of a measured loss/gain against current N=7,004. Current six-model comparisons are matched.
Full -> component removals:25fit matched ablation provides no adjusted primary advantage.
Sequential -> joint Stage2 constraints: pilot numerical defects fixed; model-score gate
can still refuse, and cross-model disagreement remains.
Bounds-based A -> target-aware B: matched studies show improved target distance conditional
on success, with budget and nutrient-uncertainty trade-offs; B is experimental.
Original chat -> hybrid/contextual tools: documented regression repairs and small real
GPU checks; no fresh broad result establishes general language reliability.
The project's supported contribution is an auditable integration and measured trade-offs,
not a new Transformer breakthrough, causal diet treatment or proven clinical safety.

8. PUBLICATION AND OUTSTANDING EVIDENCE
Latest manuscript: Research Paper/IJIES_Final/BioNutriGPT_IJIES.docx and .pdf;
12pages,9tables,2researchfigures,18references,139-word abstract, six supplied authors.
Exact predictors/tuning settings in supplementary files; author declarations remain
unconfirmed (contributions, conflicts, funding, ethics/data-access details). No journal
acceptance or Q1/Q2 classification is guaranteed. This documentation update does not
change the manuscript or create new evidence.
Open scientific limits: Indian prediction validation, prospective outcomes/dietitian
acceptability, independent-filter ablations, gradient-level stability causes, broader
noise realizations/robust optimization, matched current LP comparator and a fresh broad
conversation benchmark. Completed experiments can have negative findings; that does
not make them incomplete. No new runs were launched for this documentation task.

9. NUMERICAL EVIDENCE TABLES AND SOURCE INDEX
The following tables are copied from saved reports so that confidence intervals,
calibration, timing and alternative-variant numbers are available in this document.
They retain each study's units and cohort. Section summaries and named later corrections
govern interpretation. Historical report statements about future work/deployment are
not copied as current status. Full methodological prose remains at each source path.
'''
# Avoid rounding/conflicting baseline phrasing: use the authoritative comparison value.
overview = overview.replace('Corresponding Brier: .120911 (rounded report baseline\nis .120914),', 'Corresponding Brier: .120914,')
overview = overview.replace('-2.87pp CI[-4.33,-1.53]', '-2.87pp CI[-4.33,-1.60]')

sources = [
('Stage 1 repaired six-family comparison', 'Research Paper/Stage1_Comparison_v2_Current_Results.md'),
('Stage 1 paired uncertainty; adjusted primary family', 'Research Paper/Stage1_Paired_Uncertainty_v2/Report.md'),
('Stage 1 five-seed component ablations', 'Research Paper/Stage1_Ablation_v2_Results.md'),
('Stage 1 robustness; corrected endpoint addendum governs intervals', 'Research Paper/Stage1_Robustness_v3_Results.md'),
('Stage 1 unweighted demographic subgroups; pointwise intervals', 'Research Paper/Stage1_Subgroup_Results_v1.md'),
('Stage 1 complex-survey reference and subgroups', 'Research Paper/Stage1_Survey_Reference_v1_Results.md'),
('Stage 1 weighted calibration and hypothetical DCA', 'Research Paper/Stage1_Calibration_DCA_v1_Results.md'),
('Stage 1 downstream prediction-rule study; audit corrections govern', 'Research Paper/Stage1_Downstream_Added_Value_Results_v1.md'),
('Repaired application deployment checks', 'Research Paper/Stage1_Deployment_v1.md'),
('Stage 2 initial32-profile audit before joint repair', 'Research Paper/Stage2_Constraint_Audit_Repaired_v1.md'),
('Stage 2 joint repair regression', 'Research Paper/Stage2_Joint_Constraint_Repair_v1.md'),
('Stage 2 corrected matched comparison100; not ablation cohort', 'Research Paper/Stage2_Comparison_Study_v2.md'),
('Stage 2 ablation100x6', 'Research Paper/Stage2_Ablation_v2_Results.md'),
('Stage 2 cross-model evaluation; pointwise exploratory intervals', 'Research Paper/Stage2_Cross_Model_v1_Results.md'),
('Stage 2 stability100x8; retrospective audit qualifies stability claims', 'Research Paper/Stage2_Stability_Repaired_v1_Results.md'),
('Stage 2 retrospective sensitivity audit', 'Research Paper/Stage2_Stability_Retrospective_Audit_v1.md'),
('Stage 2 to3 translation corrected statistical audit', 'Research Paper/Stage2_Stage3_Translation_Audit_v2.md'),
('Stage 3 initial target-aware experiment corrected audit', 'Research Paper/Stage3_Target_Aware_Audit_v2.md'),
('Stage 3 controlled budget corrected audit; repeated participant modes', 'Research Paper/Stage3_Budget_Audit_v3.md'),
('Stage 3 matched budgets/target sources corrected interpretation', 'Research Paper/Stage3_Budget_Matched_Comparisons_v2.md'),
('Stage 3 robustness4200records; 3000solves1200replays', 'Research Paper/Stage3_Robustness_v3_Results.md'),
('Stage 3 all-rule failure margins; failures can overlap', 'Research Paper/Stage3_Robustness_Failure_Margin_Audit_v1.md'),
('Stage 4 historical measured checkpointed LASI comparison', 'Saved Model/stage4_lasi_v4_verified_summary.md'),
('Stage 4 real-catalog v4 engineering study', 'Saved Model/stage4_real_multiturn_v4/Report.md'),
('Stage 4 v5 engineering study', 'Saved Model/stage4_dialogues_v5/Report.md'),
('Stage 4 v6 engineering study', 'Saved Model/stage4_structured_v6/Report.md'),
('Stage 4 v7 engineering study', 'Saved Model/stage4_understanding_v7/Report.md'),
]
parts = [overview]
manifest = []
for title, relative in sources:
    p = ROOT / relative
    raw = p.read_bytes()
    text = raw.decode('utf-8-sig')
    tables, active = [], []
    for line in text.splitlines() + ['']:
        if line.lstrip().startswith('|'):
            active.append(line)
        elif active:
            tables.append('\n'.join(active)); active = []
    parts.append('\n' + '='*78 + '\n' + title + '\nSource: '+relative+'\n')
    parts.append('\n\n'.join(tables) if tables else 'No tabular block in this report; numerical findings are summarized above. See source for receipts and full detail.')
    manifest.append({'path':relative,'sha256':hashlib.sha256(raw).hexdigest(),'table_blocks':len(tables)})
supp = ROOT/'Research Paper/IJIES_Final/Supplementary_Methods.md'
parts.append('\n\n10. COMPLETE FEATURE INVENTORY AND COMPARISON CONFIGURATIONS\n\n'+supp.read_text(encoding='utf-8-sig'))
(ROOT/'Current_Model_Explanation.txt').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')

challenges = '''BIONUTRIGPT — CHALLENGES, REPAIRS AND REMAINING LIMITS
Updated26November2026. Current status, not a claim that all challenges are solved.
Original historical narrative is preserved under Documentation_History/2026-11-26_before_sync/.
Evidence numbers below refer to saved studies; this documentation pass ran no experiments.

1. Diabetes leakage, ambiguous outcomes and inconsistent cohorts — repaired for current study.
HbA1c was historically a predictor while also defining diabetes. It is now label-only;
glucose is excluded from the primary phenotype and predictors. Missing/refused outcomes
cannot establish negatives. Current development24,136 and temporal7,004 replace legacy
counts for current results. Earlier high AUROCs are not directly comparable. Lipids still
require blood: removed the non-invasive claim. Imputation is not inherently fabricated
biology; the real issues are appropriate measurement handling, leakage and honest scope.
Source: Research Paper/Stage1_Repair_v2_Report.md.

2. Preprocessing/calibration and application/research mismatch — repaired release deployed in code.
Training-fitted transforms, explicit validation/calibration partitions and versioned hashes
prevent asset mixing. Default repaired_v1 aligns Stage1/2;7,004prediction matches,
128raw-input parity checks and6negative checks recorded. Incomplete requests reject.
Historical source assembly remains unavailable; code default is not proof of a running
server or Indian clinical validation. Source: Stage1_Deployment_v1.md in Research Paper/.

3. Extra neural complexity without clear gain — measured, not solved by claim inflation.
Six-model matched comparison and25ablation fits show no overall Transformer advantage.
Removing MoE cuts parameters1,295,623->700,675 without a detected adjusted primary loss.
MoE remains in deployed model; no unsupported statement that it is necessary or superior.
Sources: Stage1_Comparison_v2_Current_Results.md, Stage1_Ablation_v2_Results.md.

4. Incorrect statistical units/intervals and weak corruption checks — corrected in named audits.
Repeated modes/arms share participants; pooled exact tests were withdrawn where invalid.
Use paired/participant-cluster inference and declared multiplicity families. Stage1
robustness10,000-resample audit reproduced2,688intervalpairs; budget audit42intervals
and27within-cell tests; Stage3 margins94intervals. Fixture checks must invoke real
validators and fail if corruption is accepted. CIs containing zero do not prove equivalence.
Verification can share parsing/preprocessing dependencies and is not always independent
end-to-end reproduction. Sources: Stage3_Budget_Audit_v3.md, Stage3_Budget_Matched_Comparisons_v2.md,
Stage1_Robustness_v3_Results.md, Stage3_Robustness_Failure_Margin_Audit_v1.md.

5. Demographic selection, missingness and survey effects — evaluated within a limited scope.
Missing both lipids/diet drops diabetes AUROC .8107->.7441. Survey and subgroup evaluation
completed, but weighting does not undo cohort selection or demonstrate fairness.
Weighted diabetes AUROC .8402 is not interchangeable with unweighted .8107.
Application still requires complete inputs. Indian external prediction data remain missing.

6. Sequential nutrient repairs broke earlier rules — numerical joint repair implemented.
32-profile development audit originally3accepted/28calorie violations. Joint projection
and final re-audit yielded25accepted,7score-gate refusals,0numerical violations.
LP fallback can establish feasibility, not nearest-target optimality.100x6ablation
numerical passes drop100->8 without projection. Rules are engineering prescriptions,
not a clinically validated guideline implementation for every patient.

7. Model exploitation and unstable targets — unresolved scientific limitations exposed.
Full100proposal diabetes score change is -.0098 in BioNutriGPT but+.0132 in CatBoost;
FT interval includes zero.800stability tasks pass numerical rules, yet1%input changes
can strongly alter individual vitamin targets without routing/fallback. Largest changes
already exist post-Adam/box clipping; exact gradient mechanism is not established.
Remaining: trajectory diagnostics on development cases and independent validation.
Sources: Stage2_Cross_Model_v1_Results.md, Stage2_Stability_Retrospective_Audit_v1.md.

8. Food provenance, B12 placeholders and overclaimed cooked-dish validation — repaired provenance.
Historical44estimated recipes/IFCT CSV are not the present food data. IFCT lacks a B12
column; uniform .05/.50 placeholders cannot be used as measured values. Current sourced
catalog49records (20INDB variants29USDA proxies),43complete for active fields. Missing
nutrients stay unknown; source proxies and estimated servings are disclosed. Not all
39nutrients or allergens are known. Full laboratory/recipe validation remains absent.

9. MUFA prescription exceeded catalog capability — historical rule conflict reviewed/repaired.
Old independent21.8g MUFA minimum exceeded17.593g attainable in that diagnostic catalog.
Replaced standalone minimum with a total-fat component envelope; this is not a MUFA
RDA/UL or proof that every new profile is feasible. Historical diagnostic catalog differs
from current49foods; do not reuse those feasibility counts as present performance.

10. Calorie pass mislabeled as overall nutrient compliance — fail-closed acceptance implemented.
Stage3 rechecks nutrient totals, quantities, categories, calories and ratios; unknown
active values exclude foods. Active slack cannot buy acceptance.4,200robustness records
were reconstructed without classification disagreement. Timeouts and infeasibility
remain different; solver status0 at1%gap is not proof of exact optimum.

11. Closer targets versus solver budget and robustness — measured trade-off remains.
Initial3s B produced fewer valid plans; controlled5s A/B both897/900, not equivalence.
Target-aware B improves target distance but under10%composition uncertainty has159/300
valid baskets vs192/300 for A. Failures are nutrient-specific; no universal boundary
mechanism proven. B remains experimental; production defaults1.5s/cap3s unchanged.
Whole servings do not establish human palatability or real cost savings.

12. Food exclusions conflict with mandatory plate categories — detected, not always repaired.
Dairy removal leaves200/300 valid; legume removal100/300 per objective in the study.
Rules can require excluded categories. Return explicit refusal instead of silently
relaxing nutrient rules. This does not establish that no alternative healthy diet exists.

13. Fish removal replaced fish with another fish; meal requests were narrow — generic data-driven repair.
Group metadata excludes all matching catalog IDs; meal placement/quantity requirements
are generic rather than breakfast-egg special cases. Supported compound clauses execute
atomically. Unsupported restrictions clarify rather than silently disappear. Real-catalog
and synthetic checks exist; all possible phrasing/food combinations are not validated.

14. LLM valid JSON was mistaken for correct intent — guarded hybrid interpretation implemented.
Initial0/6structured successes; revised only2/6semantic matches despite4valid schemas.
Deterministic-first routing, semantic/negation checks and confirmation stop unintended
mutations. v7recorded8turns/3actual model calls/1validated proposal on one profile,
with62automated checks. This does not establish general-chatbot reliability.

15. State, allergy and rollback bugs — targeted repairs and checks completed.
Server-stored proposals/version fingerprints are checked before acceptance; allergy
updates invalidate pending plans; allergy warnings persist across rollback. Current
catalog has incomplete allergen data. Never claim allergen-free or clinically safe from
numerical acceptance alone. Multi-user concurrency/access hardening remains pending.

16. Synthetic/fixed-count LLM reports and failed exports — excluded from current empirical claims.
The old claimed100-case report was not100measured LLM calls. Corrected checkpointed
run retains prompts/raw outputs/journals. Historical measured78turns:76protocol passes;
matched guarded24/24vsQwen2/24. Qwen8invalid proposals plus1malformed response.
Later chatbot changes are separate targeted checks, not a retrospective improvement of
those results. No missing raw logs are fabricated; cold/warm latency scopes remain explicit.

17. US screening plus Indian food planning — scope clarified, not external validation.
LASI100selected profiles/300plans yielded91/96/98 accepted by mode using adapted rules.
This is not Stage1 prediction validation or a nationally representative patient sample.
No longitudinal health outcomes, clinical trial or dietitian-rated acceptability exists.

18. Conflicting historical documents and manuscript readiness — current account consolidated.
Old narratives are archived; current explanation separates deployment, research and
historical snapshots. Latest IJIES package has12pages/9tables/2figures and six authors.
Funding/conflicts/contributions/ethics/data-access declarations await confirmation.
Publication readiness and journal acceptance cannot be guaranteed by a tier label.
'''
for name in ['Challenges faced and how we solved them.txt','challenges_faced_and_how_we_solved_them.txt']:
    (ROOT/name).write_text(challenges,encoding='utf-8')
(ROOT/'Stage 3 Challenges.txt').write_text('STAGE 3 — CURRENT CHALLENGES (26 November 2026)\n\n'+
    challenges[challenges.index('8. Food provenance'):challenges.index('13. Fish removal')]+
    '\nSee Current_Model_Explanation.txt sections4/9 for complete tables.\n'+
    'Historical text preserved in Documentation_History/2026-11-26_before_sync/.\n',encoding='utf-8')
progress_header = '''CURRENT STATUS — 26 NOVEMBER 2026
This is a cumulative historical log. Dated claims below describe their original snapshot;
this status block and later corrected reports take precedence over obsolete "pending",
legacy-production, perfect-safety and journal-tier claims. Earlier entries are preserved.

Stage1: repaired cohort/labels/preprocessing; six-family four-configuration comparison;
20,000-resample paired uncertainty;25architecture fits; missing/noisy-input evaluation
and corrected10,000-resample endpoint audit; unweighted subgroups, survey reference,
calibration/DCA;4,800-task prediction-routing downstream study; repaired deployment and
Stage1/2 binding checks completed within their stated scopes. No overall Transformer
advantage and no Indian external prediction validation established.
Stage2:32-profile audit/repair;100x3matched methods;100x6ablations;700rows/scorer frozen
cross-model evaluation;100x8stability tasks and600comparison retrospective audit complete.
Filters1–3 only removed jointly; full counterfactual threshold sweep not done. Instability
and scorer disagreement remain findings, not repaired-away clinical risks.
Stage3:900translation records,1,800initial target-aware outputs,5,400controlled-budget
solves and corrected paired analysis;4,200robustness records (3,000solves+1,200replays)
and94margin intervals audited. Production bounds-based vs experimental target-aware
paths remain distinct;5s results did not deploy a5s default.
Stage4: historical measured LASI78turns and matched24turns plus successful-pair ablation;
subsequent contextual/group/meal/compound/guard repairs and v4/v5/v6/v7small real-catalog
evaluations documented. Latest v7:8turns,3Qwen calls,1proposal;62automated checks recorded.
Broader current-code dialogue generalization, nutrition education and multi-user hardening
remain incomplete. Do not call historical78turns a fresh benchmark of today's code.
Publication: IJIES_Final Word/PDF/supplement prepared25November; six authors and shared
affiliation included. Author-confirmed administrative declarations remain outstanding.

Documentation reconciliation completed26November:
- Replaced Current_Model_Explanation.txt with current architecture, deployment defaults,
  all-stage results, numeric source tables, exact87features and tuning settings.
- Rewrote both root challenge-file spellings consistently and refreshed Stage3challenges.
- Preserved previous copies in Documentation_History/2026-11-26_before_sync/.
- Added Documentation_Evidence_Index_20261126.json with source report hashes.
- No training, GPU inference, solver benchmark, browser test or manuscript changes.

---------------------- ORIGINAL CUMULATIVE HISTORY ----------------------
'''
old_progress = (ARCHIVE/'Progress.txt').read_text(encoding='utf-8-sig')
(ROOT/'Progress.txt').write_text(progress_header+'\n'+old_progress,encoding='utf-8')
(ROOT/'Documentation_Evidence_Index_20261126.json').write_text(json.dumps({
    'updated':'2026-11-26','scope':'Documentation-only review of saved evidence and current code',
    'experiments_run':False,'sources':manifest,
    'outputs':names,'historical_backup':str(ARCHIVE.relative_to(ROOT)),
},indent=2),encoding='utf-8')
# Improve plain-text readability without modifying technical IDs, paths or table values.
units = r'(?:participants|profiles|records|tasks|solves|rows|scenarios|turns|requests|cases|proposals|pairs|fits|tables|figures|references|pages|foods|nutrients|features|methods|variants|arms|modes|budgets|checks|tests|intervals|intervalpairs|endpoints|contrasts|resamples|prediction|raw|negative|positive|calorie|score|numerical|accepted|rejected|model|within|research|comparison|scientific|parameter|actual|matched|food|group|sourced|validated|valid|invalid|malformed|structured|semantic|expected|joint|full|corrected|saved|separate|mean|median|warm|cold|automated|selected|merged|eligible|diabetes|obesity|CVD|multimorbid|male|female|states|age|repeat|history|table|seconds|GPU|Qwen|INDB|USDA|mg|mcg|kcal|ms|pp)'
for name in names:
    p = ROOT/name
    value = p.read_text(encoding='utf-8')
    # Do not reformat the original cumulative progress history.
    marker = '---------------------- ORIGINAL CUMULATIVE HISTORY ----------------------'
    head, sep, tail = value.partition(marker) if name == 'Progress.txt' else (value,'','')
    head = re.sub(r'(?<=\d)(?='+units+r'\b)', ' ', head)
    head = re.sub(r'\b(\d+)x(\d+)\b', r'\1 x \2', head)
    head = re.sub(r'(?<=\d)(?=CI\[)', ' ', head)
    head = head.replace('26November2026','26 November 2026').replace('25November','25 November')
    head = re.sub(r'\bStage([1234])(?![\w/])', r'Stage \1', head)
    p.write_text(head+sep+tail,encoding='utf-8')
print(json.dumps({'updated':names,'evidence_reports':len(manifest),
                  'numeric_table_blocks':sum(x['table_blocks'] for x in manifest)},indent=2))
