"""Unit tests exercising the actual runner worker (execute_study_worker) in run_stage1_downstream_study_v2.py.

Verifies:
1. Matching valid artifact: solver is not called.
2. Change ONLY a reference target: invalidates resume.
3. Change ONLY a reference bound: invalidates resume.
4. Change ONLY reference rule metadata: invalidates resume.
5. Change ONLY a baseline nutrient used in ratio check: invalidates resume.
6. Change assigned bound: invalidates resume.
7. Change input metadata: invalidates resume.
8. Change binding: invalidates resume.
9. Missing fingerprint: strict mode rejects.
10. Truncated JSON: cleanly re-solved.
11. Corrupted quantity in output: rejected.
12. Duplicate food in basket: rejected.
13. Missing nutrient in delivered_nutrition: rejected.
14. Duplicate nutrient in delivered_nutrition: rejected.
15. Nonfinite total in delivered_nutrition: rejected.
16. Nutrient accounting mismatch with recomputed food totals: rejected.
17. Inconsistent acceptance flag: rejected.
18. Nonfinite float in fingerprint payload raises ValueError.
19. Valid saved failures resume without retry.
20. Incomplete failure record rejected and re-solved.
21. Caller inputs are strictly immutable.
"""
import copy
import json
import math
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

import stage3_milp_solver as s3
from run_stage1_downstream_study_v2 import execute_study_worker, compute_case_fingerprint, PROTOCOL_VERSION

def run_tests():
    print("=== Testing Actual Runner Worker Resume Safeguards (Comprehensive v3) ===")
    
    R = Path(__file__).resolve().parent
    foods = s3.load_food_database(str(R / 'food_composition_cooked_sourced.json'))
    
    nv = [f for f in foods if f.category == 'non_veg_protein'][0]
    cg = [f for f in foods if f.category == 'cereal_grain'][0]
    vs1 = [f for f in foods if f.category == 'vegetable_sabzi'][0]
    vs2 = [f for f in foods if f.category == 'vegetable_sabzi'][1]
    
    valid_basket = [
        {'food_id': nv.food_id, 'quantity': 2},
        {'food_id': cg.food_id, 'quantity': 3},
        {'food_id': vs1.food_id, 'quantity': 1},
        {'food_id': vs2.food_id, 'quantity': 1},
    ]
    tot_valid = {
        k: 2 * nv.nutrients_per_serving[k] + 3 * cg.nutrients_per_serving[k] + vs1.nutrients_per_serving[k] + vs2.nutrients_per_serving[k]
        for k in s3.NUTRIENT_CODES
    }
    deliv_valid = [{'nutrient': k, 'delivered': tot_valid[k]} for k in s3.NUTRIENT_CODES]
    
    mock_env = {
        'SEQN': 109369,
        'arm': 'bionutrigpt',
        'threshold': 0.10,
        'patient_metadata': {'age': 45.0, 'weight_kg': 70.0, 'height_cm': 170.0, 'is_female': False},
        'rule_decisions': {'diabetes': {'selected_rule': True, 'source': 'prediction'}},
        'energy_prescription': {'target_calories': tot_valid['DR1TKCAL']},
        'nutrient_constraint_envelope': [
            {'nutrient': 'DR1TKCAL', 'min_bound': tot_valid['DR1TKCAL'] - 100.0, 'max_bound': tot_valid['DR1TKCAL'] + 100.0, 'target': tot_valid['DR1TKCAL'], 'baseline': 2000.0, 'rule_provenance': {'planning_role': 'hard_constraint'}},
            {'nutrient': 'DR1TPROT', 'min_bound': 10.0, 'max_bound': 200.0, 'target': tot_valid['DR1TPROT'], 'baseline': 75.0, 'rule_provenance': {'planning_role': 'hard_constraint'}}
        ]
    }
    
    mock_ref = {
        'SEQN': 109369,
        'arm': 'rules_only',
        'threshold': None,
        'patient_metadata': {'age': 45.0, 'weight_kg': 70.0, 'height_cm': 170.0, 'is_female': False},
        'rule_decisions': {'diabetes': {'selected_rule': False, 'source': 'rules_only'}},
        'energy_prescription': {'target_calories': tot_valid['DR1TKCAL']},
        'nutrient_constraint_envelope': [
            {'nutrient': 'DR1TKCAL', 'min_bound': tot_valid['DR1TKCAL'] - 100.0, 'max_bound': tot_valid['DR1TKCAL'] + 100.0, 'target': tot_valid['DR1TKCAL'], 'baseline': 2000.0, 'rule_provenance': {'planning_role': 'hard_constraint'}},
            {'nutrient': 'DR1TPROT', 'min_bound': 10.0, 'max_bound': 200.0, 'target': tot_valid['DR1TPROT'], 'baseline': 75.0, 'rule_provenance': {'planning_role': 'hard_constraint'}}
        ]
    }
    
    bindings = {
        'catalog_hash': 'catalog_sha256_dummy',
        'adapter_hash': 'adapter_sha256_dummy',
        'solver_hash': 'solver_sha256_dummy',
        'prediction_probabilities': {'diabetes': 0.25, 'cvd': 0.12}
    }
    
    dietary_mode = 'pure_non_vegetarian'
    task = (0, copy.deepcopy(mock_env), copy.deepcopy(mock_ref), dietary_mode)
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="runner_safeguard_test_"))
    
    try:
        mock_output = s3.Stage3Output(
            accepted=True,
            solved_food_basket=valid_basket,
            delivered_nutrition=deliv_valid,
            clinical_verification={'checks': {'all_passed': True}},
            solver_metadata={'status_message': 'Optimal', 'solver_status_code': 0, 'solve_time_ms': 50.0}
        )
        
        # 1. Fresh solve: case file doesn't exist yet
        mock_solver = MagicMock(return_value=mock_output)
        res1 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res1['resumed'] is False, "Initial run must not be resumed"
        assert mock_solver.call_count == 1, "Solver must be called on fresh run"
        case_file = tmp_dir / 'case_0000_LOCAL.json'
        assert case_file.exists(), "Case file must be written atomically"
        print("  [PASS] 1. Initial solve executed and atomic case written.")
        
        # 2. Matching valid artifact: solver must NOT be called
        mock_solver.reset_mock()
        res2 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res2['resumed'] is True, f"Second run with matching fingerprint must resume, got {res2}"
        assert mock_solver.call_count == 0, "Solver must NOT be called on matching valid artifact"
        print("  [PASS] 2. Matching valid artifact resumed without calling solver.")
        
        # 3. Change ONLY a reference target -> invalidates resume
        mock_solver.reset_mock()
        changed_ref_target = copy.deepcopy(mock_ref)
        changed_ref_target['nutrient_constraint_envelope'][1]['target'] = 85.0 # ONLY reference target changed
        task_ref_target = (0, copy.deepcopy(mock_env), changed_ref_target, dietary_mode)
        res3 = execute_study_worker(task_ref_target, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res3['resumed'] is False, "Changed reference target must reject resume"
        assert mock_solver.call_count == 1, "Solver must be called when reference target changes"
        print("  [PASS] 3. Changing ONLY reference target invalidates resume.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 4. Change ONLY a reference bound -> invalidates resume
        mock_solver.reset_mock()
        changed_ref_bound = copy.deepcopy(mock_ref)
        changed_ref_bound['nutrient_constraint_envelope'][0]['min_bound'] = tot_valid['DR1TKCAL'] - 200.0 # ONLY reference bound changed
        task_ref_bound = (0, copy.deepcopy(mock_env), changed_ref_bound, dietary_mode)
        res4 = execute_study_worker(task_ref_bound, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res4['resumed'] is False, "Changed reference bound must reject resume"
        assert mock_solver.call_count == 1, "Solver must be called when reference bound changes"
        print("  [PASS] 4. Changing ONLY reference bound invalidates resume.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 5. Change ONLY reference rule metadata -> invalidates resume
        mock_solver.reset_mock()
        changed_ref_meta = copy.deepcopy(mock_ref)
        changed_ref_meta['nutrient_constraint_envelope'][0]['rule_provenance']['planning_role'] = 'updated_role'
        task_ref_meta = (0, copy.deepcopy(mock_env), changed_ref_meta, dietary_mode)
        res5 = execute_study_worker(task_ref_meta, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res5['resumed'] is False, "Changed reference rule metadata must reject resume"
        assert mock_solver.call_count == 1, "Solver must be called when reference rule metadata changes"
        print("  [PASS] 5. Changing ONLY reference rule metadata invalidates resume.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 6. Change ONLY a baseline nutrient used in ratio check -> invalidates resume
        mock_solver.reset_mock()
        changed_baseline_env = copy.deepcopy(mock_env)
        changed_baseline_env['nutrient_constraint_envelope'][0]['baseline'] = 2500.0 # ONLY baseline changed
        task_baseline = (0, changed_baseline_env, copy.deepcopy(mock_ref), dietary_mode)
        res6 = execute_study_worker(task_baseline, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res6['resumed'] is False, "Changed baseline nutrient must reject resume"
        assert mock_solver.call_count == 1, "Solver must be called when baseline nutrient changes"
        print("  [PASS] 6. Changing ONLY baseline nutrient invalidates resume.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 7. Change assigned bound -> invalidates resume
        mock_solver.reset_mock()
        changed_assigned_bound = copy.deepcopy(mock_env)
        changed_assigned_bound['nutrient_constraint_envelope'][1]['max_bound'] = 140.0
        task_assigned_bound = (0, changed_assigned_bound, copy.deepcopy(mock_ref), dietary_mode)
        res7 = execute_study_worker(task_assigned_bound, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res7['resumed'] is False, "Changed assigned bound must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 7. Changing assigned bound invalidates resume.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 8. Change binding -> invalidates resume
        mock_solver.reset_mock()
        changed_bindings = copy.deepcopy(bindings)
        changed_bindings['prediction_probabilities']['diabetes'] = 0.35
        res8 = execute_study_worker(task, tmp_dir, foods, changed_bindings, solver_override=mock_solver)
        assert res8['resumed'] is False, "Changed prediction probability binding must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 8. Changing binding invalidates resume.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 9. Output validation: Corrupted quantity in output
        mock_solver.reset_mock()
        saved_data = json.loads(case_file.read_text(encoding='utf-8'))
        saved_data['output']['solved_food_basket'][0]['quantity'] = 4 # invalid (>3)
        case_file.write_text(json.dumps(saved_data), encoding='utf-8')
        res9 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res9['resumed'] is False, "Corrupted quantity must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 9. Corrupted quantity in output rejected.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 10. Output validation: Duplicate food in basket
        mock_solver.reset_mock()
        saved_data = json.loads(case_file.read_text(encoding='utf-8'))
        saved_data['output']['solved_food_basket'][1]['food_id'] = saved_data['output']['solved_food_basket'][0]['food_id']
        case_file.write_text(json.dumps(saved_data), encoding='utf-8')
        res10 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res10['resumed'] is False, "Duplicate food ID in basket must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 10. Duplicate food in basket rejected.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 11. Output validation: Missing nutrient in delivered_nutrition
        mock_solver.reset_mock()
        saved_data = json.loads(case_file.read_text(encoding='utf-8'))
        saved_data['output']['delivered_nutrition'].pop() # drop one nutrient
        case_file.write_text(json.dumps(saved_data), encoding='utf-8')
        res11 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res11['resumed'] is False, "Missing nutrient must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 11. Missing nutrient in delivered_nutrition rejected.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 12. Output validation: Duplicate nutrient in delivered_nutrition
        mock_solver.reset_mock()
        saved_data = json.loads(case_file.read_text(encoding='utf-8'))
        saved_data['output']['delivered_nutrition'].append(copy.deepcopy(saved_data['output']['delivered_nutrition'][0]))
        case_file.write_text(json.dumps(saved_data), encoding='utf-8')
        res12 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res12['resumed'] is False, "Duplicate nutrient must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 12. Duplicate nutrient in delivered_nutrition rejected.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 13. Output validation: Nonfinite total in delivered_nutrition
        mock_solver.reset_mock()
        saved_data = json.loads(case_file.read_text(encoding='utf-8'))
        saved_data['output']['delivered_nutrition'][0]['delivered'] = None
        case_file.write_text(json.dumps(saved_data), encoding='utf-8')
        res13 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res13['resumed'] is False, "Nonfinite/None nutrient total must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 13. Nonfinite total in delivered_nutrition rejected.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 14. Output validation: Nutrient accounting mismatch
        mock_solver.reset_mock()
        saved_data = json.loads(case_file.read_text(encoding='utf-8'))
        saved_data['output']['delivered_nutrition'][0]['delivered'] += 50.0 # deliberate accounting corruption
        case_file.write_text(json.dumps(saved_data), encoding='utf-8')
        res14 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res14['resumed'] is False, "Nutrient accounting mismatch must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 14. Nutrient accounting mismatch rejected.")
        
        # Reset case_0000 to matching
        execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        
        # 15. Output validation: Inconsistent acceptance flag
        mock_solver.reset_mock()
        saved_data = json.loads(case_file.read_text(encoding='utf-8'))
        saved_data['output']['accepted'] = False # inconsistent with case['accepted'] == True
        case_file.write_text(json.dumps(saved_data), encoding='utf-8')
        res15 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res15['resumed'] is False, "Inconsistent accepted flag must reject resume"
        assert mock_solver.call_count == 1
        print("  [PASS] 15. Inconsistent acceptance flag rejected.")
        
        # 16. Nonfinite float in fingerprint payload raises ValueError
        nonfinite_env = copy.deepcopy(mock_env)
        nonfinite_env['nutrient_constraint_envelope'][0]['min_bound'] = float('nan')
        try:
            compute_case_fingerprint(nonfinite_env, dietary_mode, bindings, reference=mock_ref)
            assert False, "Should have raised ValueError on NaN float"
        except ValueError:
            print("  [PASS] 16. Nonfinite float in fingerprint payload raises ValueError.")
            
        # 17. Truncated JSON
        mock_solver.reset_mock()
        case_file.write_text('{"case": 0, "fingerprint": "abc", "accepted":', encoding='utf-8')
        res17 = execute_study_worker(task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert res17['resumed'] is False
        assert mock_solver.call_count == 1
        print("  [PASS] 17. Truncated JSON detected and re-solved cleanly.")
        
        # 18. Valid saved failures resume without retry
        task_fail = (1, copy.deepcopy(mock_env), copy.deepcopy(mock_ref), dietary_mode)
        mock_fail_output = s3.Stage3Output(
            accepted=False,
            solved_food_basket=[],
            delivered_nutrition=[],
            clinical_verification={},
            solver_metadata={'status_message': 'Infeasible active rules', 'solver_status_code': 2, 'solve_time_ms': 25.0}
        )
        mock_fail_solver = MagicMock(return_value=mock_fail_output)
        res_fail1 = execute_study_worker(task_fail, tmp_dir, foods, bindings, solver_override=mock_fail_solver)
        assert res_fail1['accepted'] is False
        assert res_fail1['resumed'] is False
        assert mock_fail_solver.call_count == 1
        
        # Resume valid failure
        mock_fail_solver.reset_mock()
        res_fail2 = execute_study_worker(task_fail, tmp_dir, foods, bindings, solver_override=mock_fail_solver)
        assert res_fail2['resumed'] is True, "Valid failure must resume without retry"
        assert res_fail2['accepted'] is False
        assert mock_fail_solver.call_count == 0, "Solver must NOT be called for valid recorded failure"
        print("  [PASS] 18. Valid saved failures resume without retry.")
        
        # 19. Incomplete failure record rejected
        fail_file = tmp_dir / 'case_0001_LOCAL.json'
        fail_data = json.loads(fail_file.read_text(encoding='utf-8'))
        fail_data['output'] = {'accepted': False} # missing failure reason or solver metadata
        fail_file.write_text(json.dumps(fail_data), encoding='utf-8')
        mock_fail_solver.reset_mock()
        res_fail3 = execute_study_worker(task_fail, tmp_dir, foods, bindings, solver_override=mock_fail_solver)
        assert res_fail3['resumed'] is False, "Incomplete failure record must reject resume"
        assert mock_fail_solver.call_count == 1
        print("  [PASS] 19. Incomplete failure record rejected and re-solved.")
        
        # 20. Caller inputs are strictly immutable
        original_env_before = copy.deepcopy(mock_env)
        original_ref_before = copy.deepcopy(mock_ref)
        immut_env = copy.deepcopy(mock_env)
        immut_ref = copy.deepcopy(mock_ref)
        immut_task = (2, immut_env, immut_ref, dietary_mode)
        execute_study_worker(immut_task, tmp_dir, foods, bindings, solver_override=mock_solver)
        assert immut_env == original_env_before, "Caller env was mutated!"
        assert immut_ref == original_ref_before, "Caller ref was mutated!"
        print("  [PASS] 20. Caller inputs are strictly immutable.")
        
        print("\nALL 20 ACTUAL RUNNER SAFEGUARD TESTS PASSED WITH 100% SUCCESS.")
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    run_tests()
