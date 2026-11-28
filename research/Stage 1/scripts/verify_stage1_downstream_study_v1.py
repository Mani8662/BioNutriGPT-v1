"""Independent verifier and contract tests for Stage 1 downstream study v1.

Independently recalculates all nutrient totals, servings, variety, calorie deviations,
macro proportions, ratios, fiber density, and bounds from raw catalog foods.
Performs adversarial contract tests exercising real validation behavior.
"""
import json
import math
import hashlib
import numpy as np
from pathlib import Path
import pandas as pd

import stage3_milp_solver as s3
from stage2_research_rule_adapter import build_research_prescription, DIET_RAW_COLS

R = Path(__file__).resolve().parent
O = R / 'Saved Model/stage1_downstream_study_v1'
CASES_DIR = O / 'cases'

def verify_single_plan(case_dict, foods):
    """Independently recompute all nutrient totals and check constraints."""
    accepted = case_dict.get('accepted', False)
    if not accepted:
        return {
            'has_plan': False,
            'passed': False,
            'reason': 'solver_rejected',
            'mismatch_with_solver': accepted is True
        }
        
    basket = case_dict.get('output', {}).get('solved_food_basket') or case_dict.get('solved_food_basket', [])
    delivered = case_dict.get('output', {}).get('delivered_nutrition') or case_dict.get('delivered_nutrition', [])
    dietary_mode = case_dict['dietary_mode']
    env = case_dict.get('input', case_dict)
    
    lookup = {f.food_id: f for f in foods}
    tot = dict.fromkeys(s3.NUTRIENT_CODES, 0.0)
    errors = []
    mass = 0.0
    counts = {}
    active = {}
    
    for item in basket:
        fid = item['food_id']
        q = item['quantity']
        if fid not in lookup:
            errors.append(f'unknown_food_id:{fid}')
            continue
        f = lookup[fid]
        if type(q) is not int or not 1 <= q <= 3:
            errors.append(f'invalid_portion:{fid}:{q}')
        if dietary_mode == 'pure_vegetarian' and f.dietary_type == 'non_vegetarian':
            errors.append(f'non_veg_in_pure_veg:{fid}')
        mass += q * f.serving_weight_g
        counts[f.category] = counts.get(f.category, 0) + q
        active[f.category] = active.get(f.category, 0) + 1
        for k in tot:
            tot[k] += q * f.nutrients_per_serving[k]
            
    # Serving and variety rules
    cat_limits, serv_limits = s3._category_rules_for_mode(dietary_mode)
    for k, (lo, hi) in serv_limits.items():
        c = counts.get(k, 0)
        if not lo <= c <= hi:
            errors.append(f'serving_rule_failed:{k}:{c}_not_in_[{lo},{hi}]')
    for k, (lo, hi) in cat_limits.items():
        a = active.get(k, 0)
        if not lo <= a <= hi:
            errors.append(f'variety_rule_failed:{k}:{a}_not_in_[{lo},{hi}]')
            
    # Dish counts: 4 to 8
    if not 4 <= len(basket) <= 8:
        errors.append(f'dish_count_out_of_bounds:{len(basket)}')
        
    # Active nutrient envelope bounds
    for row in env['nutrient_constraint_envelope']:
        k = row['nutrient']
        role = row['rule_provenance'].get('planning_role')
        if role in ('report_only', 'exposure_reporting'):
            continue
        if any(k in lookup[x['food_id']].unknown_nutrients for x in basket if x['food_id'] in lookup):
            errors.append(f'unknown_nutrient_selected:{k}')
        val = tot[k]
        if not (row['min_bound'] - 1e-4 <= val <= row['max_bound'] + 1e-4):
            errors.append(f'active_bound_violation:{k}:{val:.2f}_outside_[{row["min_bound"]:.2f},{row["max_bound"]:.2f}]')
            
    # Calorie tolerance: +/- 5%
    kcal = tot['DR1TKCAL']
    target = env['energy_prescription']['target_calories']
    err = abs(kcal - target) / target
    if err > 0.05 + 1e-9:
        errors.append(f'calorie_tolerance_exceeded:{err:.4f}')
        
    # Macro energy percentages
    for k, factor, lo, hi in [('DR1TPROT', 4.0, 0.10, 0.35), ('DR1TCARB', 4.0, 0.45, 0.65), ('DR1TTFAT', 9.0, 0.20, 0.35)]:
        frac = tot[k] * factor / kcal
        if not (lo - 1e-9 <= frac <= hi + 1e-9):
            errors.append(f'macro_fraction_out_of_bounds:{k}:{frac:.3f}_not_in_[{lo},{hi}]')
            
    # Sodium-to-potassium ratio
    base = {x['nutrient']: x['baseline'] for x in env['nutrient_constraint_envelope']}
    sodi_pot_actual = tot['DR1TSODI'] / (tot['DR1TPOTA'] + 1.0)
    sodi_pot_base = base['DR1TSODI'] / (base['DR1TPOTA'] + 1.0)
    if sodi_pot_actual > sodi_pot_base + 1e-9:
        errors.append(f'sodium_potassium_ratio_worse_than_baseline:{sodi_pot_actual:.3f}_gt_{sodi_pot_base:.3f}')
        
    # Fiber density >= 14 g / 1000 kcal
    if tot['DR1TFIBE'] / (kcal / 1000.0) < 14.0 - 1e-9:
        errors.append('fiber_density_below_14g_per_1000kcal')
        
    # Delivered vs recomputed totals accounting check
    deliv_dict = {r['nutrient']: r['delivered'] for r in delivered}
    for k in s3.NUTRIENT_CODES:
        if k in deliv_dict:
            if abs(deliv_dict[k] - tot[k]) > 1e-4:
                errors.append(f'accounting_mismatch:{k}:deliv={deliv_dict[k]:.4f}_vs_recomputed={tot[k]:.4f}')
                
    passed = len(errors) == 0
    return {
        'has_plan': True,
        'passed': passed,
        'errors': errors,
        'totals': tot,
        'food_mass_g': mass,
        'dish_count': len(basket),
        'calorie_error_fraction': err,
        'mismatch_with_solver': accepted != passed
    }

def run_contract_tests(foods):
    """Adversarial tests exercising real validation behavior."""
    print("--- Running Adversarial Contract Tests ---")
    tests_passed = 0
    
    # Base dummy profile
    base_profile = dict(
        weight_kg=75.0, height_cm=175.0, age_years=45.0, is_female=False,
        activity_level='sedentary', has_diabetes=False, has_cvd=False,
        renal_disease=None, pregnant=None, lactating=None,
        baseline_nutrients={k: 10.0 for k in DIET_RAW_COLS}
    )
    base_profile['baseline_nutrients']['DR1TKCAL'] = 2000.0
    base_profile['baseline_nutrients']['DR1TPOTA'] = 3000.0
    base_profile['baseline_nutrients']['DR1TSODI'] = 2000.0
    
    # 1. Missing predictions rejected
    try:
        build_research_prescription(base_profile, arm='bionutrigpt', probabilities=None, thresholds={'diabetes': 0.1, 'cvd': 0.1})
        assert False, "Should have failed on None probabilities"
    except ValueError:
        tests_passed += 1
        print("  [PASS] Missing predictions rejected")
        
    # 2. Out-of-range / nonfinite probabilities rejected
    for bad_p in [-0.1, 1.5, float('nan'), float('inf')]:
        try:
            build_research_prescription(base_profile, arm='bionutrigpt', probabilities={'diabetes': bad_p, 'cvd': 0.1}, thresholds={'diabetes': 0.1, 'cvd': 0.1})
            assert False, f"Should have failed on p={bad_p}"
        except ValueError:
            pass
    tests_passed += 1
    print("  [PASS] Out-of-range / nonfinite probabilities rejected")
    
    # 3. Known diagnosis takes precedence over low predicted risk
    prof_diab = dict(base_profile, has_diabetes=True)
    env_diab = build_research_prescription(prof_diab, arm='bionutrigpt', probabilities={'diabetes': 0.01, 'cvd': 0.01}, thresholds={'diabetes': 0.5, 'cvd': 0.5})
    assert env_diab['rule_decisions']['diabetes']['selected_rule'] is True
    assert env_diab['rule_decisions']['diabetes']['source'] == 'reported_diagnosis'
    tests_passed += 1
    print("  [PASS] Known diagnosis precedence verified")
    
    # 4. Predicted obesity does not override measured BMI
    # Normal BMI = 75 / 1.75^2 = 24.49 (not obese)
    env_norm = build_research_prescription(base_profile, arm='bionutrigpt', probabilities={'diabetes': 0.1, 'cvd': 0.1}, thresholds={'diabetes': 0.5, 'cvd': 0.5})
    assert env_norm['obesity_rule'] is False
    assert env_norm['measured_bmi'] < 30.0
    tests_passed += 1
    print("  [PASS] Measured BMI precedence over predicted obesity verified")
    
    # 5. Unknown clinical history stays unknown
    prof_unk = dict(base_profile, has_diabetes=None)
    env_unk = build_research_prescription(prof_unk, arm='rules_only')
    assert 'has_diabetes' in env_unk['clinical_unknowns']
    assert env_unk['rule_decisions']['diabetes']['observed_history'] is None
    assert env_unk['rule_decisions']['diabetes']['selected_rule'] is False
    tests_passed += 1
    print("  [PASS] Unknown clinical history preserved")
    
    # 6. Deliberately corrupted food portion caught by verifier
    dummy_case = {
        'accepted': True,
        'dietary_mode': 'veg_nonveg_combo',
        'solved_food_basket': [{'food_id': foods[0].food_id, 'quantity': 5}],  # portion > 3
        'delivered_nutrition': [],
        'input': env_norm
    }
    rep = verify_single_plan(dummy_case, foods)
    assert not rep['passed']
    assert any('invalid_portion' in e for e in rep['errors'])
    tests_passed += 1
    print("  [PASS] Corrupted food portion caught by verifier")
    
    # 7. Deliberately corrupted saved nutrient total caught by verifier
    # Create valid dummy basket with valid portions
    dummy_basket = [
        {'food_id': foods[0].food_id, 'quantity': 1},
        {'food_id': foods[1].food_id, 'quantity': 1},
        {'food_id': foods[2].food_id, 'quantity': 1},
        {'food_id': foods[3].food_id, 'quantity': 1},
    ]
    # Compute correct delivered
    tot_kcal = sum(f.nutrients_per_serving['DR1TKCAL'] for f in foods[:4])
    corrupt_delivered = [{'nutrient': 'DR1TKCAL', 'delivered': tot_kcal + 100.0}]
    dummy_case_corrupt = {
        'accepted': True,
        'dietary_mode': 'veg_nonveg_combo',
        'solved_food_basket': dummy_basket,
        'delivered_nutrition': corrupt_delivered,
        'input': env_norm
    }
    rep_corr = verify_single_plan(dummy_case_corrupt, foods)
    assert any('accounting_mismatch' in e for e in rep_corr['errors'])
    tests_passed += 1
    print("  [PASS] Corrupted nutrient total caught by verifier")
    
    # 8. Non-veg dish in pure_vegetarian caught
    non_veg_foods = [f for f in foods if f.dietary_type == 'non_vegetarian']
    if non_veg_foods:
        dummy_veg_case = {
            'accepted': True,
            'dietary_mode': 'pure_vegetarian',
            'solved_food_basket': [{'food_id': non_veg_foods[0].food_id, 'quantity': 1}],
            'delivered_nutrition': [],
            'input': env_norm
        }
        rep_veg = verify_single_plan(dummy_veg_case, foods)
        assert any('non_veg_in_pure_veg' in e for e in rep_veg['errors'])
        tests_passed += 1
        print("  [PASS] Non-veg food in pure_vegetarian mode caught")
        
    print(f"All {tests_passed} adversarial contract tests passed successfully.\n")
    return True

def verify_all_study_cases():
    """Verify all 4,800 saved cases from Saved Model/stage1_downstream_study_v1/cases/."""
    foods = s3.load_food_database(str(R / 'food_composition_cooked_sourced.json'))
    
    # First run contract tests
    run_contract_tests(foods)
    
    case_files = sorted(list(CASES_DIR.glob('case_*_LOCAL.json')))
    print(f"Verifying {len(case_files)} study case files...")
    assert len(case_files) == 4800, f"Expected 4800 case files, found {len(case_files)}"
    
    hashes = {}
    accepted_count = 0
    replay_passed_count = 0
    common_ref_passed_count = 0
    mismatches = []
    
    for i, cf in enumerate(case_files):
        content = cf.read_bytes()
        h = hashlib.sha256(content).hexdigest()
        hashes[cf.name] = h
        
        cdata = json.loads(content.decode('utf-8'))
        is_acc = cdata.get('accepted', False)
        if is_acc:
            accepted_count += 1
            
        ver = verify_single_plan(cdata, foods)
        if ver['passed']:
            replay_passed_count += 1
            
        if cdata.get('reference_replay', {}).get('passed', False):
            common_ref_passed_count += 1
            
        if ver['mismatch_with_solver']:
            mismatches.append({'case': cdata.get('case'), 'SEQN': cdata.get('SEQN'), 'errors': ver['errors']})
            
        if (i + 1) % 500 == 0 or (i + 1) == len(case_files):
            print(f"  Verified {i+1}/{len(case_files)} cases | Accepted: {accepted_count} | Replay Passed: {replay_passed_count}")
            
    assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches between solver and replay: {mismatches[:5]}"
    assert accepted_count == replay_passed_count, f"Accepted ({accepted_count}) != Replay Passed ({replay_passed_count})"
    
    report = {
        'complete': True,
        'total_cases': len(case_files),
        'accepted_cases': accepted_count,
        'rejected_cases': len(case_files) - accepted_count,
        'own_replay_all_passed': True,
        'common_reference_passed_count': common_ref_passed_count,
        'mismatch_count': 0,
        'scope': 'Controlled 100-participant engineering study across 3 dietary modes and 16 arms (4,800 solves)',
        'case_file_hashes': hashes
    }
    
    (O / 'verification.json').write_text(json.dumps(report, indent=2))
    (O / 'completed.json').write_text(json.dumps({
        'status': 'COMPLETED',
        'study': 'stage1_downstream_study_v1',
        'cases': 4800,
        'accepted': accepted_count,
        'verified_all_pass': True,
        'timestamp': pd.Timestamp.now().isoformat()
    }, indent=2))
    
    print("\nSUCCESS: All 4,800 cases verified with 100% agreement. verification.json and completed.json written.")

if __name__ == '__main__':
    verify_all_study_cases()
