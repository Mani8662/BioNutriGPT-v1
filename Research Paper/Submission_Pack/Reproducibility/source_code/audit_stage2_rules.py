"""Reproducible review of all 39 rules, with historical comparison and sources."""
import copy
import json
from pathlib import Path

import numpy as np

import stage2_counterfactual_engine as s2
from nutrient_rule_policy import RULESET_VERSION, reviewed_rule_metadata

ROOT = Path(__file__).resolve().parent


def review_saved_envelope():
    historical = json.loads((ROOT / 'bionutrigpt_pipeline_dossier.json').read_text(encoding='utf-8'))
    stage2 = copy.deepcopy(historical['stage2_counterfactual_prescription'])
    metadata = historical['patient_metadata']
    age = float(metadata.get('age_years', 42))
    female = bool(metadata.get('is_female', False))
    weight = float(metadata.get('weight_kg', 74))
    probabilities = historical['stage1_diagnostic_assessment']['probabilities']
    lower, upper = s2._build_nutrient_bounds(
        artifacts={}, t_kcal=float(stage2['energy_prescription']['target_calories']),
        weight_kg=weight, is_female=female, age_years=age,
        p_diab_0=probabilities['diabetes'], p_obes_0=probabilities['obesity'], p_cvd_0=probabilities['cvd'])
    assert len(lower) == len(upper) == 39 and np.isfinite(lower).all() and np.isfinite(upper).all()
    rows = []
    for record in stage2['nutrient_constraint_envelope']:
        code = record['nutrient']
        i = s2.NUTRIENT_INDEX[code]
        provenance = reviewed_rule_metadata(code, float(lower[i]), float(upper[i]))
        rows.append({'nutrient': code, 'historical_bounds': [record['min_bound'], record['max_bound']],
                     'reviewed_bounds': [float(lower[i]), float(upper[i])], **provenance})
        record.update(min_bound=float(lower[i]), max_bound=float(upper[i]), rule_provenance=provenance)
    # This fixture is a rule evaluation, not a re-executed Stage 2 search.
    stage2['nutrient_ruleset_version'] = RULESET_VERSION
    stage2['certified_for_stage3'] = False
    stage2['clinical_prescribing_validated'] = False
    stage2['audit_fixture_only'] = True
    stage2['historical_risk_scores_not_recomputed'] = True
    stage2['rule_review_note'] = 'Only bounds and rule metadata were recomputed. Historical model scores, targets and filters do not certify this revised fixture.'
    return stage2, rows, metadata


def main():
    stage2, rows, metadata = review_saved_envelope()
    report = {'ruleset_version': RULESET_VERSION, 'rule_count': len(rows), 'patient_metadata': metadata,
              'changed_interval_count': sum(r['historical_bounds'] != r['reviewed_bounds'] for r in rows),
              'clinical_validation_established': False, 'stage1_features_changed': False,
              'scope': 'US food-reference interpretation for generally healthy adults; not an ICMR-validated Indian treatment protocol.',
              'mandatory_controls': ['finite nonnegative inputs', 'known schema and units', 'portion and plate rules',
                                     'explicit missing-data status', 'no clinical certification from report-only fields'],
              'advisory_controls': 'Reference values and project assumptions are not validated clinical prescriptions. Operationally, Stage 3 requires complete data and interval compliance for active fields before acceptance. The diagnostic checks only covered, active fields.',
              'remaining_review': ['risk-score-based sodium/potassium/carb/protein/cholesterol/exposure policies',
                                   'renal disease, pregnancy/lactation, medication use, supplements and smoking',
                                   'population-specific Indian applicability and cooking yield',
                                   'food niacin vs niacin equivalents; vitamin K1 vs total vitamin K AI',
                                   'Stage 2 filters still use legacy project rules and do not prove clinical safety'],
              'rules': rows}
    out = ROOT / 'Saved Model'
    (out / 'stage2_rule_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'Data/food_sources/INDB/reviewed_stage2_envelope.json').write_text(json.dumps(stage2, indent=2), encoding='utf-8')
    lines = ['# Stage 2 nutrient-rule review', '',
             f'Ruleset: {RULESET_VERSION}. All 39 fields reviewed. Stage 1 inputs are unchanged.',
             'These are US reference interpretations and labeled project assumptions, not an Indian clinical protocol.',
             'References describe generally healthy people. Pregnancy/lactation, renal disease, medicines,',
             'supplements, smoking and individual clinical treatment require separate assessment.', '',
             '## Mandatory versus advisory', '',
             'Mandatory controls concern input validity, units, portion/plate structure and honest missing-data reporting.',
             'The production solver requires active-interval compliance before acceptance. An RDA/AI is a planning reference,',
             'not a proven one-day clinical safety boundary. The diagnostic may enforce active covered intervals',
             'strictly, but that mathematical check does not establish clinical validity.', '',
             '## Corrections', '',
             '- Total sugar is reported separately; its former 25/50 g limit cannot be justified by WHO free-sugar guidance.',
             '- Folate adequacy is assessed in DFE. Total folate and food folate no longer duplicate the DFE minimum.',
             '- The folic-acid ceiling is kept on synthetic folic acid, not total food folate or DFE.',
             '- Vitamin A RAE retains its adequacy reference; the preformed-vitamin-A ceiling is screened separately.',
             '- Four individual carotenoids are reporting/research fields without invented clinical upper limits.',
             '- Supplemental-only or unestablished upper limits are removed from ordinary food intervals.',
             '- Selected age/sex B6, C, K, calcium, phosphorus, magnesium, iron, zinc and copper values are corrected.',
             '- Risk-score-based macronutrient, sodium, potassium, cholesterol and exposure policies remain explicitly',
             '  labeled project assumptions needing review. No universal safety claim is made for them.', '',
             'A finite numerical search cap (1,000,000 in the field unit) is used where the vector optimizer',
             'requires a finite domain. It is never a UL or recommended dose. Report-only fields have zero',
             'nutrient penalty and no nutrient constraint in Stage 3; unknown data still prevent full verification.', '',
             '| Field | Historical interval | Reviewed interval | Role | Evidence / applicability |',
             '| --- | --- | --- | --- | --- |']
    for r in rows:
        def fmt(values): return '–'.join(f'{x:g}' for x in values)
        lines.append(f"| {r['nutrient']} ({r['unit']}) | {fmt(r['historical_bounds'])} | {fmt(r['reviewed_bounds'])} | {r['planning_role']} | {r['review_note']} [Source]({r['source_url']}) |")
    lines += ['', '## Reproduction', '', 'Run `python audit_stage2_rules.py`, then `python audit_sourced_cooked_catalog.py`.',
              'The saved envelope is an audit fixture: only rules were recomputed. Historical model scores and',
              'targets were not recalculated, and its prior Stage 2 certification is explicitly revoked.', '']
    (out / 'stage2_rule_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"Reviewed {len(rows)} rules; {report['changed_interval_count']} intervals corrected; clinical validation remains false")


if __name__ == '__main__':
    main()
