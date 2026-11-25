"""Stage 1 Downstream Study Runner v2 with Canonical Fingerprinting and Strict Resume Safeguards.

Implements:
1. Canonical fingerprint computation covering participant, arm, mode, threshold,
   complete effective solver inputs, rule decisions, assigned bounds, reference bounds,
   targets, baselines, energy prescriptions, clinical flags, bindings, and solver settings.
2. Strict resume verification: only resumes when fingerprint, identifiers, output integrity,
   and independent recomputation match.
3. Output validation before resume: food IDs, integer quantities, exact nutrient coverage,
   finite totals, independent accounting, and rule replay.
4. Preservation of recorded solver failures without silent retry.
5. Complete caller input immutability.
6. Atomic file writing (.tmp -> replace).
"""
import os
import sys
import json
import time
import copy
import math
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

import stage3_milp_solver as s3
from stage2_research_rule_adapter import build_research_prescription, DIET_RAW_COLS

PROTOCOL_VERSION = 'stage1_downstream_study_v2'
TIME_LIMIT_SEC = 3.0

def validate_finite(obj, path=''):
    """Recursively validate that all numeric values in obj are finite.
    
    Rejects NaN, Inf, and -Inf with ValueError.
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Nonfinite float encountered at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            validate_finite(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            validate_finite(v, f"{path}[{i}]")

def compute_case_fingerprint(env, dietary_mode, bindings, reference=None, protocol_version=PROTOCOL_VERSION):
    """Compute deterministic SHA-256 fingerprint for a downstream meal planning case.
    
    Covers:
    - Participant, arm, dietary_mode, threshold
    - Patient metadata, clinical flags, and complete rule decisions
    - Effective assigned bounds, targets, baselines, and rule provenance metadata
    - Common-reference bounds, targets, baselines, and rule provenance metadata
    - Energy prescriptions
    - Catalog, adapter, solver, ruleset, and prediction bindings
    - Solver settings and protocol version
    
    Uses strict canonical serialization without silent lossy rounding and rejects nonfinite values.
    """
    if isinstance(dietary_mode, dict) and isinstance(bindings, str):
        # Called as (env, reference, dietary_mode, bindings)
        reference, dietary_mode, bindings = dietary_mode, bindings, reference
    if reference is None:
        reference = env

    def _extract_envelope(envelope_list):
        rows = []
        for r in envelope_list:
            nk = str(r.get('nutrient', ''))
            row_dict = {
                'nutrient': nk,
                'min_bound': float(r['min_bound']),
                'max_bound': float(r['max_bound']),
                'target': float(r['target']),
                'baseline': float(r['baseline']) if 'baseline' in r and r['baseline'] is not None else None,
                'rule_provenance': r.get('rule_provenance', {})
            }
            rows.append(row_dict)
        rows.sort(key=lambda x: x['nutrient'])
        return rows

    assigned_envelope = _extract_envelope(env.get('nutrient_constraint_envelope', []))
    reference_envelope = _extract_envelope(reference.get('nutrient_constraint_envelope', []))

    patient_meta = copy.deepcopy(env.get('patient_metadata', {}))
    bindings_dict = bindings or {}

    payload = {
        'protocol_version': str(protocol_version),
        'SEQN': int(env['SEQN']),
        'arm': str(env['arm']),
        'dietary_mode': str(dietary_mode),
        'threshold': float(env['threshold']) if env.get('threshold') is not None else None,
        'patient_metadata': patient_meta,
        'rule_decisions': env.get('rule_decisions', {}),
        'energy_prescription': env.get('energy_prescription', {}),
        'reference_energy_prescription': reference.get('energy_prescription', {}),
        'assigned_envelope': assigned_envelope,
        'reference_envelope': reference_envelope,
        'bindings': {
            'catalog_hash': str(bindings_dict.get('catalog_hash', '')),
            'adapter_hash': str(bindings_dict.get('adapter_hash', '')),
            'solver_hash': str(bindings_dict.get('solver_hash', '')),
            'prediction_probabilities': bindings_dict.get('prediction_probabilities', {}),
            'nutrient_ruleset_version': str(env.get('nutrient_ruleset_version', '')),
            'constraint_spec_version': str(env.get('constraint_spec_version', ''))
        },
        'solver_settings': {
            'time_limit_sec': float(TIME_LIMIT_SEC),
            'slack_penalty': 0.0,
            'objective_policy': 'calorie_deviation_plus_food_selection'
        }
    }

    validate_finite(payload)
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=s3._json_safe)
    return hashlib.sha256(dumped.encode('utf-8')).hexdigest()

def replay_solution(out, foods, env, dietary_mode):
    """Independent replay audit of a solved meal plan against catalog and envelope."""
    accepted = getattr(out, 'accepted', None) if not isinstance(out, dict) else out.get('accepted')
    if not accepted:
        return {'has_plan': False, 'passed': False, 'errors': ['solver_rejected']}
        
    lookup = {f.food_id: f for f in foods}
    tot = dict.fromkeys(s3.NUTRIENT_CODES, 0.0)
    errors = []
    mass = 0.0
    counts = {}
    active = {}
    
    basket = getattr(out, 'solved_food_basket', None) if not isinstance(out, dict) else out.get('solved_food_basket')
    if not basket:
        return {'has_plan': False, 'passed': False, 'errors': ['empty_food_basket']}
        
    for item in basket:
        fid = item.get('food_id')
        q = item.get('quantity')
        if fid not in lookup:
            errors.append(f'unknown_food:{fid}')
            continue
        f = lookup[fid]
        if type(q) is not int or not 1 <= q <= 3:
            errors.append(f'portion:{fid}:{q}')
        if dietary_mode == 'pure_vegetarian' and f.dietary_type == 'non_vegetarian':
            errors.append(f'non_veg_in_pure_veg:{fid}')
        mass += q * f.serving_weight_g
        counts[f.category] = counts.get(f.category, 0) + q
        active[f.category] = active.get(f.category, 0) + 1
        for k in tot:
            tot[k] += q * f.nutrients_per_serving[k]
            
    cat_limits, serv_limits = s3._category_rules_for_mode(dietary_mode)
    for k, (lo, hi) in serv_limits.items():
        c = counts.get(k, 0)
        if not lo <= c <= hi:
            errors.append(f'serving:{k}:{c}_not_in_[{lo},{hi}]')
    for k, (lo, hi) in cat_limits.items():
        a = active.get(k, 0)
        if not lo <= a <= hi:
            errors.append(f'variety:{k}:{a}_not_in_[{lo},{hi}]')
            
    if not 4 <= len(basket) <= 8:
        errors.append(f'dish_count:{len(basket)}')
        
    for row in env.get('nutrient_constraint_envelope', []):
        k = row['nutrient']
        role = row.get('rule_provenance', {}).get('planning_role')
        if role in ('report_only', 'exposure_reporting'):
            continue
        val = tot[k]
        if not (row['min_bound'] - 1e-4 <= val <= row['max_bound'] + 1e-4):
            errors.append(f'bound:{k}:{val:.2f}_not_in_[{row["min_bound"]:.2f},{row["max_bound"]:.2f}]')
            
    delivered = getattr(out, 'delivered_nutrition', None) if not isinstance(out, dict) else out.get('delivered_nutrition')
    if delivered:
        for row in delivered:
            n_code = row['nutrient']
            if abs(row['delivered'] - tot[n_code]) > 1e-4:
                errors.append(f'accounting:{n_code}')
                
    kcal = tot['DR1TKCAL']
    target = env.get('energy_prescription', {}).get('target_calories', 2000.0)
    err = abs(kcal - target) / target if target > 0 else 0.0

    return {
        'has_plan': True,
        'passed': len(errors) == 0,
        'errors': errors,
        'totals': tot,
        'food_mass_g': mass,
        'dish_count': len(basket),
        'calorie_error_fraction': err
    }

def execute_study_worker(task, cases_dir, foods, bindings, legacy_policy='reject', solver_override=None):
    """Execute a single study solve task with strict canonical fingerprint resume safeguards.
    
    Parameters:
    - task: (index, env, reference, dietary_mode)
    - cases_dir: Path to case files directory
    - foods: List of FoodItem objects
    - bindings: dict containing hashes and prediction probabilities
    - legacy_policy: 'reject' (default strict mode) or 'accept_with_warning'
    - solver_override: callable replacing s3.solve_meal_plan (for unit testing)
    """
    index, raw_env, raw_ref, dietary_mode = task
    case_file = Path(cases_dir) / f'case_{index:04d}_LOCAL.json'
    lookup = {f.food_id: f for f in foods}

    # Snapshot caller inputs to verify caller immutability
    orig_env_snapshot = json.dumps(raw_env, sort_keys=True, default=s3._json_safe)
    orig_ref_snapshot = json.dumps(raw_ref, sort_keys=True, default=s3._json_safe)

    # Construct effective solver input without mutating caller inputs
    effective_env = copy.deepcopy(raw_env)
    effective_ref = copy.deepcopy(raw_ref)

    # Synchronize reference targets before fingerprinting
    ref_target_map = {
        row['nutrient']: row['target']
        for row in effective_ref.get('nutrient_constraint_envelope', [])
        if 'nutrient' in row and 'target' in row
    }
    for row in effective_env.get('nutrient_constraint_envelope', []):
        if row.get('nutrient') in ref_target_map:
            row['target'] = ref_target_map[row['nutrient']]

    expected_fp = compute_case_fingerprint(
        effective_env, dietary_mode, bindings, reference=effective_ref, protocol_version=PROTOCOL_VERSION
    )

    can_resume = False
    reject_reason = None

    if case_file.exists():
        try:
            raw_text = case_file.read_text(encoding='utf-8')
            data = json.loads(raw_text)

            # 1. Fingerprint validation
            saved_fp = data.get('fingerprint')
            if saved_fp is None:
                if legacy_policy == 'reject':
                    reject_reason = 'missing_fingerprint_legacy_cache_rejected'
                else:
                    reject_reason = 'missing_fingerprint_unverified'
            elif saved_fp != expected_fp:
                reject_reason = f'fingerprint_mismatch:saved={saved_fp[:8]}..._expected={expected_fp[:8]}...'
            else:
                # 2. Case identifiers matching requested task
                if (data.get('case') != index or
                    data.get('SEQN') != effective_env.get('SEQN') or
                    data.get('dietary_mode') != dietary_mode or
                    data.get('arm') != effective_env.get('arm') or
                    data.get('threshold') != effective_env.get('threshold')):
                    reject_reason = 'identifier_mismatch_with_task'
                # 3. Consistent acceptance across saved fields
                elif 'accepted' not in data or 'output' not in data or 'own_replay' not in data:
                    reject_reason = 'incomplete_artifact_missing_core_fields'
                elif not isinstance(data.get('accepted'), bool):
                    reject_reason = 'invalid_nonboolean_accepted_field'
                elif data.get('output', {}).get('accepted') != data.get('accepted'):
                    reject_reason = 'inconsistent_accepted_flag_between_case_and_output'
                elif data['accepted'] is True:
                    out = data['output']
                    basket = out.get('solved_food_basket')
                    delivered = out.get('delivered_nutrition')

                    # Food IDs validation
                    if not isinstance(basket, list) or not (4 <= len(basket) <= 8):
                        reject_reason = f'invalid_dish_count:{len(basket) if isinstance(basket, list) else None}'
                    else:
                        food_ids = [item.get('food_id') for item in basket if isinstance(item, dict)]
                        if len(food_ids) != len(basket):
                            reject_reason = 'basket_item_missing_food_id'
                        elif len(food_ids) != len(set(food_ids)):
                            reject_reason = 'duplicate_food_in_basket'
                        elif not all(fid in lookup for fid in food_ids):
                            reject_reason = 'unrecognized_food_id_in_basket'
                        elif dietary_mode == 'pure_vegetarian' and any(lookup[fid].dietary_type == 'non_vegetarian' for fid in food_ids):
                            reject_reason = 'non_vegetarian_food_in_pure_vegetarian_mode'
                        # Quantities validation
                        elif not all(isinstance(item.get('quantity'), int) and not isinstance(item.get('quantity'), bool) and 1 <= item.get('quantity') <= 3 for item in basket):
                            reject_reason = 'invalid_serving_quantity'
                        # Delivered nutrients validation
                        elif not isinstance(delivered, list):
                            reject_reason = 'delivered_nutrition_not_a_list'
                        else:
                            deliv_codes = [r.get('nutrient') for r in delivered if isinstance(r, dict)]
                            if len(deliv_codes) != len(s3.NUTRIENT_CODES) or set(deliv_codes) != set(s3.NUTRIENT_CODES):
                                reject_reason = f'delivered_nutrition_key_mismatch:count={len(deliv_codes)}_unique={len(set(deliv_codes))}'
                            elif not all(isinstance(r.get('delivered'), (int, float)) and math.isfinite(r.get('delivered')) for r in delivered):
                                reject_reason = 'nonfinite_or_missing_delivered_nutrient'
                            else:
                                # Nutrient totals recomputation
                                recomputed_totals = {
                                    code: sum(item['quantity'] * lookup[item['food_id']].nutrients_per_serving[code] for item in basket)
                                    for code in s3.NUTRIENT_CODES
                                }
                                accounting_errors = [
                                    r['nutrient'] for r in delivered
                                    if abs(r['delivered'] - recomputed_totals[r['nutrient']]) > 1e-4
                                ]
                                if accounting_errors:
                                    reject_reason = f'nutrient_accounting_mismatch:{accounting_errors}'
                                else:
                                    # Assigned-rule replay
                                    own_replay = replay_solution(out, foods, effective_env, dietary_mode)
                                    if not own_replay['passed']:
                                        reject_reason = f'assigned_rule_replay_failed:{own_replay["errors"]}'
                                    elif data.get('own_replay', {}).get('passed') is not True:
                                        reject_reason = 'saved_own_replay_not_passed'
                                    else:
                                        # Common-reference replay consistency
                                        ref_replay = replay_solution(out, foods, effective_ref, dietary_mode)
                                        saved_ref = data.get('reference_replay', {})
                                        if saved_ref.get('passed') != ref_replay['passed'] or saved_ref.get('has_plan') != ref_replay['has_plan']:
                                            reject_reason = 'reference_replay_mismatch_with_recomputation'
                                        else:
                                            can_resume = True
                else:
                    # Accepted is False: valid recorded failure
                    out = data.get('output', {})
                    if not isinstance(out, dict):
                        reject_reason = 'corrupted_failure_output_not_a_dict'
                    elif data.get('own_replay', {}).get('has_plan') is True:
                        reject_reason = 'inconsistent_failure_has_plan_is_true'
                    else:
                        has_failure_info = (
                            out.get('failure_reason') is not None or
                            out.get('solver_metadata', {}).get('status_message') is not None or
                            out.get('failure') is not None or
                            data.get('error') is not None
                        )
                        if not has_failure_info:
                            reject_reason = 'incomplete_failure_record_missing_diagnostics'
                        else:
                            can_resume = True

        except json.JSONDecodeError as exc:
            reject_reason = f'truncated_or_corrupted_json:{repr(exc)}'
        except Exception as exc:
            reject_reason = f'read_error:{repr(exc)}'

    if can_resume:
        # Verify caller immutability before returning
        assert json.dumps(raw_env, sort_keys=True, default=s3._json_safe) == orig_env_snapshot, "Caller env was mutated!"
        assert json.dumps(raw_ref, sort_keys=True, default=s3._json_safe) == orig_ref_snapshot, "Caller reference was mutated!"

        return {
            'case': index,
            'SEQN': data['SEQN'],
            'dietary_mode': data['dietary_mode'],
            'arm': data['arm'],
            'threshold': data['threshold'],
            'accepted': data['accepted'],
            'own_replay': data['own_replay'],
            'reference_replay': data.get('reference_replay', {}),
            'seconds': data.get('seconds', 0.0),
            'fingerprint': data['fingerprint'],
            'resumed': True,
            'resume_status': 'resumed_valid_fingerprint'
        }

    # Fresh solve required using effective_env
    solve_fn = solver_override if solver_override is not None else s3.solve_meal_plan
    start = time.perf_counter()

    try:
        out = solve_fn(effective_env, foods, time_limit_sec=TIME_LIMIT_SEC, revision_spec={'dietary_mode': dietary_mode})
        own = replay_solution(out, foods, effective_env, dietary_mode)
        common = replay_solution(out, foods, effective_ref, dietary_mode)
        elapsed = time.perf_counter() - start

        result = {
            'case': index,
            'SEQN': int(effective_env['SEQN']),
            'dietary_mode': dietary_mode,
            'arm': effective_env['arm'],
            'threshold': effective_env['threshold'],
            'accepted': out.accepted,
            'own_replay': own,
            'reference_replay': common,
            'output': out.to_dict() if hasattr(out, 'to_dict') else out,
            'input': effective_env,
            'seconds': elapsed,
            'fingerprint': expected_fp,
            'protocol_version': PROTOCOL_VERSION,
            'error': None
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        result = {
            'case': index,
            'SEQN': int(effective_env['SEQN']),
            'dietary_mode': dietary_mode,
            'arm': effective_env['arm'],
            'threshold': effective_env['threshold'],
            'accepted': False,
            'own_replay': {'has_plan': False, 'passed': False, 'errors': ['exception']},
            'reference_replay': {'has_plan': False, 'passed': False, 'errors': ['exception']},
            'output': {'accepted': False, 'failure': {'exception': repr(exc)}},
            'input': effective_env,
            'seconds': elapsed,
            'fingerprint': expected_fp,
            'protocol_version': PROTOCOL_VERSION,
            'error': repr(exc)
        }

    # Atomic write: write to .tmp, then atomic replace
    tmp_file = case_file.with_suffix('.tmp')
    tmp_file.write_text(json.dumps(result, indent=2, default=s3._json_safe), encoding='utf-8')
    tmp_file.replace(case_file)

    # Verify caller immutability
    assert json.dumps(raw_env, sort_keys=True, default=s3._json_safe) == orig_env_snapshot, "Caller env was mutated!"
    assert json.dumps(raw_ref, sort_keys=True, default=s3._json_safe) == orig_ref_snapshot, "Caller reference was mutated!"

    return {
        'case': index,
        'SEQN': result['SEQN'],
        'dietary_mode': result['dietary_mode'],
        'arm': result['arm'],
        'threshold': result['threshold'],
        'accepted': result['accepted'],
        'own_replay': result['own_replay'],
        'reference_replay': result['reference_replay'],
        'seconds': result['seconds'],
        'fingerprint': expected_fp,
        'resumed': False,
        'resume_status': 'solved_fresh'
    }
