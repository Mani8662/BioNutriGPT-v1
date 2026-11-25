"""Large controlled study of Stage 1 downstream added value in Stage 2-3 meal planning.

Controlled engineering study across 100 participants, 3 dietary modes, and 16 arms
(1 rules-only + 3 models x 5 thresholds = 16 solves per participant/mode; 4,800 total).
Zero active nutrient slack; baseline clipped into rules-only reference bounds.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import json
import copy
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

import stage3_milp_solver as s3
from stage2_research_rule_adapter import build_research_prescription, DIET_RAW_COLS

R = Path(__file__).resolve().parent
O = R / 'Saved Model/stage1_downstream_study_v1'
CASES_DIR = O / 'cases'
CASES_DIR.mkdir(parents=True, exist_ok=True)

def replay(out, foods, env, dietary_mode):
    if not out.accepted:
        return {'has_plan': False, 'passed': False, 'errors': ['solver_rejected']}
    lookup = {f.food_id: f for f in foods}
    tot = dict.fromkeys(s3.NUTRIENT_CODES, 0.0)
    errors = []
    mass = 0.0
    counts = {}
    active = {}
    
    for item in out.solved_food_basket:
        fid = item['food_id']
        q = item['quantity']
        f = lookup[fid]
        if type(q) is not int or not 1 <= q <= 3:
            errors.append('portion')
        if dietary_mode == 'pure_vegetarian' and f.dietary_type == 'non_vegetarian':
            errors.append('non_veg_in_pure_veg')
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
            
    if not 4 <= len(out.solved_food_basket) <= 8:
        errors.append(f'dish_count:{len(out.solved_food_basket)}')
        
    for row in env['nutrient_constraint_envelope']:
        k = row['nutrient']
        role = row['rule_provenance'].get('planning_role')
        if role in ('report_only', 'exposure_reporting'):
            continue
        if any(k in lookup[x['food_id']].unknown_nutrients for x in out.solved_food_basket):
            errors.append(f'unknown:{k}')
        val = tot[k]
        if not (row['min_bound'] - 1e-4 <= val <= row['max_bound'] + 1e-4):
            errors.append(f'bound:{k}:{val:.2f}_not_in_[{row["min_bound"]:.2f},{row["max_bound"]:.2f}]')
            
    kcal = tot['DR1TKCAL']
    target = env['energy_prescription']['target_calories']
    err = abs(kcal - target) / target
    if err > 0.05 + 1e-9:
        errors.append(f'energy_deviation:{err:.4f}')
        
    for k, factor, lo, hi in [('DR1TPROT', 4.0, 0.10, 0.35), ('DR1TCARB', 4.0, 0.45, 0.65), ('DR1TTFAT', 9.0, 0.20, 0.35)]:
        frac = tot[k] * factor / kcal
        if not (lo - 1e-9 <= frac <= hi + 1e-9):
            errors.append(f'macro:{k}:{frac:.3f}_not_in_[{lo},{hi}]')
            
    base = {x['nutrient']: x['baseline'] for x in env['nutrient_constraint_envelope']}
    sodi_pot_actual = tot['DR1TSODI'] / (tot['DR1TPOTA'] + 1.0)
    sodi_pot_base = base['DR1TSODI'] / (base['DR1TPOTA'] + 1.0)
    if sodi_pot_actual > sodi_pot_base + 1e-9:
        errors.append(f'sodium_potassium_ratio:{sodi_pot_actual:.3f}_gt_{sodi_pot_base:.3f}')
        
    if tot['DR1TFIBE'] / (kcal / 1000.0) < 14.0 - 1e-9:
        errors.append('fiber_density_lt_14')
        
    for row in out.delivered_nutrition:
        n_code = row['nutrient']
        if abs(row['delivered'] - tot[n_code]) > 1e-4:
            errors.append(f'accounting:{n_code}')
            
    return {
        'has_plan': True,
        'passed': len(errors) == 0,
        'errors': errors,
        'totals': tot,
        'food_mass_g': mass,
        'dish_count': len(out.solved_food_basket),
        'calorie_error_fraction': err
    }

def worker(task):
    index, env, reference, dietary_mode = task
    case_file = CASES_DIR / f'case_{index:04d}_LOCAL.json'
    
    # Check if already completed and valid
    if case_file.exists():
        try:
            data = json.loads(case_file.read_text())
            if 'accepted' in data and 'own_replay' in data:
                return {
                    'case': index,
                    'SEQN': data['SEQN'],
                    'dietary_mode': data['dietary_mode'],
                    'arm': data['arm'],
                    'threshold': data['threshold'],
                    'accepted': data['accepted'],
                    'own_replay': data['own_replay'],
                    'reference_replay': data['reference_replay'],
                    'seconds': data['seconds'],
                    'resumed': True
                }
        except Exception:
            pass  # Recompute if file was corrupted

    foods = s3.load_food_database(str(R / 'food_composition_cooked_sourced.json'))
    
    # Targets fixed to reference bounds
    target = {x['nutrient']: x['target'] for x in reference['nutrient_constraint_envelope']}
    for row in env['nutrient_constraint_envelope']:
        row['target'] = target[row['nutrient']]
        
    start = time.perf_counter()
    try:
        out = s3.solve_meal_plan(env, foods, time_limit_sec=3.0, revision_spec={'dietary_mode': dietary_mode})
        own = replay(out, foods, env, dietary_mode)
        common = replay(out, foods, reference, dietary_mode)
        
        result = {
            'case': index,
            'SEQN': int(env['SEQN']),
            'dietary_mode': dietary_mode,
            'arm': env['arm'],
            'threshold': env['threshold'],
            'accepted': out.accepted,
            'own_replay': own,
            'reference_replay': common,
            'output': out.to_dict(),
            'input': env,
            'seconds': time.perf_counter() - start,
            'error': None
        }
    except Exception as exc:
        result = {
            'case': index,
            'SEQN': int(env['SEQN']),
            'dietary_mode': dietary_mode,
            'arm': env['arm'],
            'threshold': env['threshold'],
            'accepted': False,
            'own_replay': {'has_plan': False, 'passed': False, 'errors': ['exception']},
            'reference_replay': {'has_plan': False, 'passed': False, 'errors': ['exception']},
            'output': {'accepted': False, 'failure': {'exception': repr(exc)}},
            'input': env,
            'seconds': time.perf_counter() - start,
            'error': repr(exc)
        }
        
    tmp_file = CASES_DIR / f'case_{index:04d}_LOCAL.tmp'
    tmp_file.write_text(json.dumps(result, indent=2, default=s3._json_safe))
    tmp_file.replace(case_file)
    
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
        'resumed': False
    }

def main():
    print("=== Initializing Stage 1 Downstream Controlled Study ===")
    
    # 1. Load selected 100 participants
    sel_df = pd.read_csv(O / 'selected_participants_LOCAL.csv')
    assert len(sel_df) == 100, f"Expected 100 participants, got {len(sel_df)}"
    selected_seqns = sel_df['SEQN'].astype(int).tolist()
    print(f"Loaded {len(selected_seqns)} participants.")
    
    # 2. Load raw inputs
    needed_cols = ['SEQN', 'BMXWT', 'BMXHT', 'RIDAGEYR', 'RIAGENDR', 'DIQ010',
                   'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F'] + list(DIET_RAW_COLS)
    raw_df = pd.read_csv(R / 'Data/NHANES_2017_2020_PrePandemic_Raw.csv', usecols=needed_cols).set_index('SEQN')
    
    # 3. Load model predictions
    models = ['bionutrigpt', 'catboost', 'ft_transformer']
    pred_dfs = {}
    for m in models:
        pdf = pd.read_csv(R / f'Saved Model/stage1_comparison_v2/{m}/temporal_predictions_LOCAL.csv').set_index('SEQN')
        pred_dfs[m] = pdf
        
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    dietary_modes = ['pure_vegetarian', 'veg_nonveg_combo', 'pure_non_vegetarian']
    
    # 4. Generate all task envelopes deterministically
    tasks = []
    task_index = 0
    
    for seqn in selected_seqns:
        x = raw_df.loc[seqn]
        cv = x[['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']]
        diabetes = True if x.DIQ010 == 1 else False if x.DIQ010 in (2, 3) else None
        cvd = True if cv.eq(1).any() else False if cv.eq(2).all() else None
        profile = dict(
            weight_kg=float(x.BMXWT),
            height_cm=float(x.BMXHT),
            age_years=float(x.RIDAGEYR),
            is_female=bool(x.RIAGENDR == 2),
            activity_level='sedentary',
            has_diabetes=diabetes,
            has_cvd=cvd,
            renal_disease=None,
            pregnant=None,
            lactating=None,
            baseline_nutrients={k: float(x[k]) for k in DIET_RAW_COLS}
        )
        
        # Build rules_only reference prescription
        base_ref = build_research_prescription(profile, arm='rules_only')
        base_ref.update(SEQN=seqn, threshold=None, raw_DIQ010=float(x.DIQ010) if pd.notna(x.DIQ010) else None)
        for row in base_ref['nutrient_constraint_envelope']:
            row['target'] = base_ref['energy_prescription']['target_calories'] if row['nutrient'] == 'DR1TKCAL' else min(max(row['baseline'], row['min_bound']), row['max_bound'])
            
        for mode in dietary_modes:
            # 1 solve for rules_only
            tasks.append((task_index, copy.deepcopy(base_ref), copy.deepcopy(base_ref), mode))
            task_index += 1
            
            # 15 solves for prediction arms (3 models x 5 thresholds)
            for m in models:
                p_row = pred_dfs[m].loc[seqn]
                probs = {
                    'diabetes': float(p_row['TARGET_DIABETES_calibrated']),
                    'cvd': float(p_row['TARGET_CVD_calibrated'])
                }
                for t in thresholds:
                    env = build_research_prescription(profile, arm=m, probabilities=probs, thresholds={'diabetes': t, 'cvd': t})
                    env.update(SEQN=seqn, threshold=t, raw_DIQ010=float(x.DIQ010) if pd.notna(x.DIQ010) else None)
                    tasks.append((task_index, env, copy.deepcopy(base_ref), mode))
                    task_index += 1
                    
    assert len(tasks) == 4800, f"Expected 4800 tasks, got {len(tasks)}"
    print(f"Generated {len(tasks)} task configurations.")
    
    # 5. Deterministic balanced execution ordering
    rng = np.random.RandomState(20261120)
    permuted_order = rng.permutation(len(tasks))
    ordered_tasks = [tasks[i] for i in permuted_order]
    
    # 6. Execute in parallel with 10 workers (utilizing 10 of 12 CPU cores)
    num_workers = min(10, os.cpu_count() or 1)
    print(f"Launching pool with {num_workers} worker processes...")
    t0 = time.time()
    
    results = []
    completed_count = 0
    accepted_count = 0
    resumed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(worker, t) for t in ordered_tasks]
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            completed_count += 1
            if res['accepted']:
                accepted_count += 1
            if res.get('resumed'):
                resumed_count += 1
            if completed_count % 100 == 0 or completed_count == 4800:
                elapsed = time.time() - t0
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(f"[{completed_count:4d}/4800] Accepted: {accepted_count} | Resumed: {resumed_count} | Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} cases/s", flush=True)
                
    # Sort results by original task index (0..4799)
    results.sort(key=lambda x: x['case'])
    
    # Save compact summary_LOCAL.json
    (O / 'summary_LOCAL.json').write_text(json.dumps(results, indent=2))
    print(f"All 4800 tasks finished. Saved summary_LOCAL.json.")
    
if __name__ == '__main__':
    main()
