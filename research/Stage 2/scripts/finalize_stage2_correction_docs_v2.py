"""Synchronize repair documentation from verified local evidence, without inference runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from stage2_counterfactual_engine import DAILY_REFERENCE_SCALES, DIET_RAW_COLS, FEATURE_COLS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Saved Model/stage2_joint_repair_correction_v2'
RUN = ROOT / 'Saved Model/stage2_joint_constraint_repair_v2'
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

tests = read(OUT / 'tests.json')
verification = read(OUT / 'verification.json')
replay = read(OUT / 'model_replay_v2/independent_review.json')
reservation = read(ROOT / 'Saved Model/stage2_comparison_design_v1/verification.json')
assert tests['passed'] and tests['tests_run'] == 16
assert verification['verification_passed']
assert verification['verifier_sha256'] == sha(ROOT / 'verify_stage2_joint_constraint_repair_v1.py')
assert replay['passed'] and reservation['passed'] and reservation['no_comparison_run']
cases = read(RUN / 'cases_LOCAL.json')
scales = np.array([DAILY_REFERENCE_SCALES[k] for k in DIET_RAW_COLS])
distances = {'observed_to_final': [], 'optimizer_clipped_to_final': []}
for case in cases:
    result = read(RUN / f"case_{case['case']:03d}_LOCAL.json")
    final = np.array(result['snapshots']['after_filter_3'])
    observed = np.array([case['vector'][FEATURE_COLS.index(k)] for k in DIET_RAW_COLS])
    clipped = np.array(result['snapshots']['optimizer_clipped'])
    for key, source in [('observed_to_final', observed), ('optimizer_clipped_to_final', clipped)]:
        distances[key].append(float(np.linalg.norm((final-source)/scales)))
stats = {k: {'median': float(np.median(v)), 'min': min(v), 'max': max(v)} for k,v in distances.items()}
(OUT / 'target_change_summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

note = '''STAGE 2 REPAIR CORRECTION AND REGRESSION — 2026-11-22
The corrected solver distinguishes numerical SLSQP convergence, a verified feasible LP fallback (minimum-distance optimality not established), infeasible constraints, and solver failures. Fixed engineering normalization scales are not uniformly FDA/DRI values. Clinical thresholds and model assets were not changed. All 16 tests passed, including six solver-failure tests. The corrected 32-profile regression reproduced all original repair targets and decisions: 25 accepted, seven rejected by the model-score gate, zero final nutrient-bound, energy or ratio violations. The original three accepted cases are 7, 14 and 19. Independent frozen-model rescoring verified all 32 cases (maximum probability discrepancy 5.74e-8). These are development/regression findings, not clinical efficacy or population generalization evidence. A disjoint 100-profile comparison sample is reserved; no comparison outcomes have been run. The rules-only construction and executable analysis protocol still need freezing before that study. Evidence: Saved Model/stage2_joint_repair_correction_v2/ and Saved Model/stage2_joint_constraint_repair_v2/. Report: Research Paper/Stage2_Joint_Constraint_Repair_v1.md.

'''
for name in ['Progress.txt', 'Current_Model_Explanation.txt']:
    path = ROOT / name
    text = path.read_text(encoding='utf-8-sig')
    if text.startswith('STAGE 2 JOINT CONSTRAINT REPAIR & 32-PROFILE VERIFICATION COMPLETE'):
        text = text.split('\n\n', 1)[1]
    if not text.startswith(note):
        path.write_text(note + text, encoding='utf-8')

report = '''# Stage 2 joint constraint repair: corrected verification and regression

Updated 22 November 2026. This report supersedes the earlier interpretation; original source/report and verification artifacts are retained in `Saved Model/stage2_joint_repair_correction_v2/before_correction/` and the original v1 study directory.

## Problem and implementation

In the original 32-profile audit, sequential energy adjustment introduced calorie-envelope violations in 28 cases. Only three cases passed all programmed acceptance checks. The repair jointly enforces nutrient intervals, implemented protein/energy/sodium/potassium rules, energy arithmetic and three ratio envelopes after the 50-step Adam candidate search. Energy arithmetic uses 4 kcal/g protein, 4 carbohydrate, 9 fat and 7 alcohol. The internal energy margin is ±0.99%; final acceptance uses 1%. The repaired deployment uses descriptive research routing thresholds of 0.5, including CVD; the former report's 0.603 was incorrect.

The objective minimizes the sum of squared candidate changes divided by fixed nutrient-specific engineering scales. These weights are not uniformly official FDA/DRI recommendations. SLSQP numerically solves the convex quadratic problem using a HiGHS linear-feasibility witness as initialization. Numerical convergence is not a mathematical exactness certificate. If SLSQP fails or returns an invalid candidate, the engine can return a checked feasible LP witness, explicitly marked `feasible_lp_fallback` and `minimum_distance_not_established`. LP infeasibility is distinguished from time limits or other solver failures. The final model-score gate remains separate; nutrient feasibility does not guarantee that gate or discrete meal feasibility.

## Measured results

| Check | Original audit | Corrected repair regression |
| --- | ---: | ---: |
| Profiles | 32 | 32 |
| Accepted by all programmed checks | 3 | 25 |
| Calorie-envelope violations | 28 | 0 |
| Model-score gate failures | 7 | 7 |

All final nutrient bounds, energy and ratio checks passed in the corrected regression. The three baseline accepted cases (7, 14, 19) remained accepted; 22 additional cases passed. Cases 1, 8, 9, 10, 12, 24 and 31 remained rejected by the score gate. No accepted output failed the independent checks in this sample. The gate limits increases in each calibrated outcome probability to 0.0015; it is not a demonstrated clinical benefit criterion. Outcomes are diabetes, obesity and CVD, not prediabetes.

The corrected implementation produced identical final target vectors and acceptance decisions to the original joint-repair run. No LP fallback occurred in these 32 runs. Total recorded regression runtime was 116.14 seconds; this is descriptive and is not evidence of a speed improvement.

All 16 tests passed, including six tests covering SLSQP nonconvergence, nonfinite output, exceptions, invalid LP witnesses and distinctions between solver failure and infeasibility. Independent frozen-model/calibrator replay verified baseline and candidate probabilities for all 32 profiles, with maximum absolute discrepancy 5.74e-8. Replay uses the deployed preprocessing/candidate-construction path; it does not claim an independently implemented preprocessing system.

## Size of target changes

The following dimensionless L2 distances use the same fixed engineering scales as the objective; they do not measure clinical benefit or practical meal acceptability.

'''
for key, value in stats.items():
    report += f"- {key.replace('_', ' ')}: median {value['median']:.4f}, range {value['min']:.4f}–{value['max']:.4f}.\n"
report += '''
## Evidence and next study

- Historical corrected verification: `Saved Model/stage2_joint_repair_correction_v2/verification.json`.
- Corrected engine regression: `Saved Model/stage2_joint_constraint_repair_v2/summary.json` and per-case artifacts.
- Fresh regression model replay: `Saved Model/stage2_joint_repair_correction_v2/model_replay_v2/independent_review.json`.
- Tests and target-change summary: `Saved Model/stage2_joint_repair_correction_v2/`.
- Reserved sample and frozen source copies: `Saved Model/stage2_comparison_design_v1/`.

These 32 complete-input NHANES profiles are development/regression cases with sedentary activity assumed. They do not establish Indian validity, nutritional appropriateness, causal risk reduction, global optimality, or superiority over a rules-only approach.

A 100-profile sample is reserved with zero overlap with the 32 repair cases, earlier 100-profile downstream study or five-case pilot. It is not globally unseen Stage 1 data. No comparison outcomes have been run. Next, implement and freeze the exact rules-only construction, minimum-change projection arm and full model-guided arm, shared evaluation rules and downstream solver settings, using development cases before inspecting reserved outcomes.
'''
(ROOT / 'Research Paper/Stage2_Joint_Constraint_Repair_v1.md').write_text(report, encoding='utf-8')
path = ROOT / 'Research Paper/Readiness_and_Evidence.md'
text = path.read_text(encoding='utf-8-sig')
header = '## Stage 2 repair correction — 22 November 2026\n\n'
if header not in text:
    path.write_text(header + note.split('\n',1)[1] + text, encoding='utf-8')
path = ROOT / 'Research Paper/BioNutriGPT_Manuscript_v2.md'
text = path.read_text(encoding='utf-8-sig')
paragraph = '\n\nA subsequent joint-constraint repair and corrected regression on these same 32 development profiles increased acceptance from 3 to 25, with no final nutrient-bound, energy or ratio violations. All three previously accepted cases remained accepted; seven still failed the unchanged model-score gate. Sixteen software tests passed, and frozen-model rescoring verified all 32 cases (maximum probability discrepancy 5.74e-8). The solver now distinguishes numerical convergence from a checked feasible LP fallback; no fallback occurred in the regression. Fixed engineering scales define the projection objective, rather than a uniformly guideline-derived weighting scheme. These reused development cases establish a repair regression, not comparative efficacy or clinical benefit. A separate 100-profile sample is reserved for a three-method comparison, which has not yet been executed. See Stage2_Joint_Constraint_Repair_v1.md.'
anchor = 'See Stage2_Constraint_Audit_Repaired_v1.md.'
assert anchor in text
if paragraph not in text:
    path.write_text(text.replace(anchor, anchor + paragraph, 1), encoding='utf-8')
receipt = {'passed': True, 'tests': tests, 'historical_verifier_hash_current': True, 'regression_model_replay_passed': replay['passed'], 'comparison_sample': reservation, 'target_changes': stats, 'comparison_executed': False}
receipt['document_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'Progress.txt', ROOT/'Current_Model_Explanation.txt', ROOT/'Research Paper/Stage2_Joint_Constraint_Repair_v1.md', ROOT/'Research Paper/Readiness_and_Evidence.md', ROOT/'Research Paper/BioNutriGPT_Manuscript_v2.md']}
(OUT / 'completion.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
