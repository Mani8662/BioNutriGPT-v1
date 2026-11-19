"""Check the sourced catalog and run feasibility only on fully covered fields."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, milp

import stage3_milp_solver as s3
from nutrient_rule_policy import mufa_component_bounds, mufa_rule_metadata
from audit_stage2_rules import review_saved_envelope

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / 'food_composition_cooked_sourced.json'
    rows = json.loads(path.read_text(encoding='utf-8'))
    foods = s3.load_food_database()
    assert {food.food_id for food in foods} == {row['food_id'] for row in rows}
    assert all(row['food_id'].startswith('recipe_') for row in rows)
    assert all(row['ingredients'] for row in rows)
    for row in rows:
        if 'recipe_code' in row['recipe_source']:
            assert abs(sum(i['grams_per_standardized_portion'] for i in row['ingredients']) - row['serving_weight_g']) < 1e-6
    dossier = json.loads((ROOT / 'bionutrigpt_pipeline_dossier.json').read_text(encoding='utf-8'))
    stage2, rule_review, _ = review_saved_envelope()
    # The full reviewed policy is applied explicitly to an audit-only copy.
    mufa = next(row for row in stage2['nutrient_constraint_envelope'] if row['nutrient'] == 'DR1TMFAT')
    historical_mufa = next(row for row in dossier['stage2_counterfactual_prescription']['nutrient_constraint_envelope'] if row['nutrient'] == 'DR1TMFAT')
    stage = s3._parse_stage2_input(stage2)
    c, integrality, bounds, constraints, info = s3._make_sparse_constraints(foods, stage, gamma=.1)
    covered = [code for code in s3.NUTRIENT_CODES if all(code not in f.unknown_nutrients for f in foods)]
    missing = [code for code in s3.NUTRIENT_CODES if code not in covered]
    enforced_covered = [code for code in covered if stage.rule_metadata.get(code, {}).get('planning_role') != 'report_only']
    upper = bounds.ub.copy()
    # Only fully covered nutrient fields are asserted as hard constraints.
    # Uncovered fields are explicitly omitted from this diagnostic objective.
    for k, code in enumerate(s3.NUTRIENT_CODES):
        if code in enforced_covered:
            upper[info['indices']['e_plus'] + k] = 0
            upper[info['indices']['e_minus'] + k] = 0
        else:
            c[info['indices']['e_plus'] + k] = 0
            c[info['indices']['e_minus'] + k] = 0
    report = {'candidate_count': len(foods), 'covered_fields': covered,
              'enforced_covered_fields': enforced_covered,
              'ruleset_version': stage2['nutrient_ruleset_version'],
              'rule_audit_fixture_only': True,
              'mufa_policy': mufa_rule_metadata(),
              'historical_mufa_bounds': [historical_mufa['min_bound'], historical_mufa['max_bound']],
              'current_mufa_bounds': [mufa['min_bound'], mufa['max_bound']],
              'uncovered_fields': missing, 'all_39_adequacy_established': False,
              'source_registry': 'Data/food_sources/INDB/cooked_catalog_coverage.json'}
    print('Candidates', len(foods), 'covered', len(covered), 'active covered rules', len(enforced_covered), 'uncovered', len(missing), flush=True)
    result = milp(c=c, integrality=integrality, bounds=Bounds(bounds.lb, upper),
                  constraints=constraints, options={'time_limit': 30, 'mip_rel_gap': .01})
    report['covered_fields_milp_status'] = int(result.status)
    report['covered_fields_solver_message'] = str(result.message)
    report['covered_fields_feasible_candidate'] = result.x is not None
    if result.status == 2:
        lp = milp(c=np.zeros_like(c), integrality=np.zeros_like(integrality),
                  bounds=Bounds(bounds.lb, upper), constraints=constraints)
        report['covered_fields_continuous_status'] = int(lp.status)
        report['single_nutrient_relaxations_restoring_continuous_feasibility'] = []
        for k, code in enumerate(s3.NUTRIENT_CODES):
            if code not in covered:
                continue
            trial_upper = upper.copy()
            trial_upper[info['indices']['e_plus'] + k] = np.inf
            trial_upper[info['indices']['e_minus'] + k] = np.inf
            trial = milp(c=np.zeros_like(c), integrality=np.zeros_like(integrality),
                         bounds=Bounds(bounds.lb, trial_upper), constraints=constraints)
            if trial.x is not None:
                report['single_nutrient_relaxations_restoring_continuous_feasibility'].append(code)
                objective = np.zeros_like(c)
                objective[:len(foods)] = -info['food_matrix'][k]
                maximum = milp(c=objective, integrality=np.zeros_like(integrality),
                               bounds=Bounds(bounds.lb, trial_upper), constraints=constraints)
                objective[:len(foods)] *= -1
                minimum = milp(c=objective, integrality=np.zeros_like(integrality),
                               bounds=Bounds(bounds.lb, trial_upper), constraints=constraints)
                report.setdefault('conflict_ranges', {})[code] = {
                    'required_min': stage.lower_bounds[code], 'required_max': stage.upper_bounds[code],
                    'attainable_min': None if minimum.x is None else float(minimum.fun),
                    'attainable_max': None if maximum.x is None else float(-maximum.fun)}
        print('Continuous status', lp.status, 'single-nutrient conflict candidates',
              report['single_nutrient_relaxations_restoring_continuous_feasibility'], flush=True)
    if result.x is not None:
        q = np.rint(result.x[:len(foods)]).astype(int)
        out = s3._build_output_from_quantities(q, foods, stage, {'status': 'sourced_catalog_diagnostic'})
        report['candidate'] = out.to_dict()
        assert not out.clinical_verification['all_clinical_checks_passed']
        report['constraint_residual_max'] = float(max(
            np.max(np.maximum(constraints.lb - constraints.A @ result.x, 0)),
            np.max(np.maximum(constraints.A @ result.x - constraints.ub, 0))))
        assert report['constraint_residual_max'] < 1e-4
    # Production requires all active fields, unlike this covered-only diagnostic.
    production = s3.solve_meal_plan(stage2, foods)
    report['production_output'] = production.to_dict()
    report['note'] = 'Diagnostic feasibility concerns covered fields only. Production requires complete active data and all active checks, and returns no accepted basket on failure. Gram portions use modeled input mass, not measured cooked yield.'
    (ROOT / 'Data/food_sources/INDB/cooked_catalog_feasibility.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Covered-field status', result.status, result.message, flush=True)
    print('Candidate found', result.x is not None, 'full adequacy', False, flush=True)
    print('Production accepted', production.accepted, 'all_active_checks_passed', production.clinical_verification.get('all_active_checks_passed'), flush=True)


if __name__ == '__main__':
    main()
