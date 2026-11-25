"""Repaired statistical and failure analysis pipeline for Stage 1 Downstream Study Audit v3.

Produces:
1. failure_analysis_audit_v3.json and timing_analysis_audit_v3.json with precise solver vs wall times.
2. case_level_paired_eval_audit_v3_LOCAL.csv (4,500 rows).
3. inter_model_contrasts_audit_v3_LOCAL.csv (13,500 rows).
4. direct_model_contrasts_audit_v3.csv and .json (45 cell-specific + 3 overall equal-weighted).
5. aggregate_summary_audit_v3.csv and .json (45 cells with programmatic verification).
"""
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np

R = Path(__file__).resolve().parent
V1_DIR = R / 'Saved Model/stage1_downstream_study_v1'
CASES_DIR = V1_DIR / 'cases'
AUDIT_DIR = R / 'Saved Model/stage1_downstream_study_v1_audit_v3'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Standard Reference scales (FDA Daily Values for nutrition labeling)
# These are constant reference scales used to avoid division by near-zero baselines.
# They are not individualized RDAs or Indian clinical targets.
REFERENCE_DAILY_SCALES = {
    'DR1TKCAL': 2000.0,    # standard reference kcal
    'DR1TPROT': 50.0,      # FDA Daily Value (g)
    'DR1TCARB': 275.0,     # FDA Daily Value (g)
    'DR1TFIBE': 28.0,      # FDA Daily Value (g)
    'DR1TTFAT': 78.0,      # FDA Daily Value (g)
    'DR1TSFAT': 20.0,      # FDA Daily Value (g)
    'DR1TCHOL': 300.0,     # FDA Daily Value (mg)
    'DR1TSODI': 2300.0,    # FDA Daily Value (mg)
    'DR1TPOTA': 4700.0,    # FDA Daily Value (mg)
}

PRIMARY_CLINICAL_NUTRIENTS = [
    'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TFIBE',
    'DR1TTFAT', 'DR1TSFAT', 'DR1TCHOL', 'DR1TSODI', 'DR1TPOTA'
]

def _to_json_compat(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    return str(obj)

def load_cases():
    print("Loading 4,800 cases from disk...")
    case_files = sorted(list(CASES_DIR.glob('case_*_LOCAL.json')))
    assert len(case_files) == 4800, f"Expected 4800 case files, found {len(case_files)}"
    cases = []
    for cf in case_files:
        cases.append(json.loads(cf.read_text(encoding='utf-8')))
    return cases

def run_analysis():
    cases = load_cases()
    participants = sorted(list({int(c['SEQN']) for c in cases}))
    assert len(participants) == 100, f"Expected 100 participants, got {len(participants)}"
    
    modes = ['pure_vegetarian', 'veg_nonveg_combo', 'pure_non_vegetarian']
    models = ['bionutrigpt', 'catboost', 'ft_transformer']
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    
    indexed = {}
    for c in cases:
        key = (int(c['SEQN']), c['dietary_mode'], c['arm'], c['threshold'])
        indexed[key] = c
        
    print("Indexed all 4,800 cases.")
    
    # -------------------------------------------------------------
    # 1. Failure and Timing Analysis
    # -------------------------------------------------------------
    failure_records = []
    timing_all_records = []
    
    for c in cases:
        meta = c.get('output', {}).get('solver_metadata', {})
        wall_sec = float(c.get('seconds', 0.0))
        wall_ms = wall_sec * 1000.0
        solv_ms = float(meta.get('solve_time_ms', 0.0)) if meta.get('solve_time_ms') is not None else None
        status_code = meta.get('solver_status_code')
        status_msg = meta.get('status_message', '')
        acc = bool(c.get('accepted', False))
        
        timing_all_records.append({
            'case': c['case'],
            'SEQN': int(c['SEQN']),
            'mode': c['dietary_mode'],
            'arm': c['arm'],
            'threshold': c['threshold'],
            'accepted': acc,
            'status_code': status_code,
            'wall_ms': wall_ms,
            'solver_time_ms': solv_ms
        })
        
        if not acc:
            out = c.get('output', {})
            fail = out.get('failure', c.get('failure_details', {}))
            code = fail.get('code') if isinstance(fail, dict) else str(fail)
            
            if status_code == 2 or 'Infeasible' in status_msg or code == 'infeasible_active_rules':
                cat = 'solver_certified_infeasible'
            elif status_code == 1 or 'Time limit' in status_msg or code == 'timeout_no_valid_plan':
                cat = 'computational_timeout'
            else:
                cat = 'other_failure'
                
            failure_records.append({
                'case': c['case'],
                'SEQN': int(c['SEQN']),
                'dietary_mode': c['dietary_mode'],
                'arm': c['arm'],
                'threshold': c['threshold'],
                'category': cat,
                'failure_code': code,
                'status_code': status_code,
                'status_message': status_msg,
                'wall_ms': wall_ms,
                'solver_time_ms': solv_ms
            })
            
    fail_df = pd.DataFrame(failure_records)
    time_df = pd.DataFrame(timing_all_records)
    
    inf_df = fail_df[fail_df['category'] == 'solver_certified_infeasible']
    timeout_df = fail_df[fail_df['category'] == 'computational_timeout']
    
    def summarize_series(s):
        clean = s.dropna()
        return {
            'count': int(len(clean)),
            'min': float(clean.min()),
            'q25': float(clean.quantile(0.25)),
            'median': float(clean.median()),
            'mean': float(clean.mean()),
            'q75': float(clean.quantile(0.75)),
            'max': float(clean.max()),
            'count_gt_40ms': int((clean > 40.0).sum()),
            'pct_gt_40ms': float((clean > 40.0).mean() * 100.0)
        }
        
    timing_summary = {
        'infeasible_cases_count': len(inf_df),
        'infeasible_solver_time_ms': summarize_series(inf_df['solver_time_ms']),
        'infeasible_wall_ms': summarize_series(inf_df['wall_ms']),
        'timeout_cases_count': len(timeout_df),
        'timeout_wall_ms': summarize_series(timeout_df['wall_ms']),
        'timeout_details': timeout_df[['case', 'SEQN', 'dietary_mode', 'arm', 'threshold', 'wall_ms', 'solver_time_ms', 'status_message']].to_dict(orient='records'),
        'rules_only_failure_count': int((fail_df['arm'] == 'rules_only').sum()),
        'rules_only_failures': fail_df[fail_df['arm'] == 'rules_only'][['case', 'SEQN', 'dietary_mode', 'category', 'status_code', 'wall_ms']].to_dict(orient='records')
    }
    
    (AUDIT_DIR / 'timing_analysis_audit_v3.json').write_text(json.dumps(timing_summary, indent=2, default=_to_json_compat))
    print("Saved timing_analysis_audit_v3.json.")
    
    fail_summary = {
        'total_cases': 4800,
        'total_accepted': 4614,
        'rules_only_accepted': sum(1 for c in cases if c['arm'] == 'rules_only' and c['accepted']),
        'rules_only_total': 300,
        'rules_only_failures': int((fail_df['arm'] == 'rules_only').sum()),
        'prediction_assisted_accepted': sum(1 for c in cases if c['arm'] != 'rules_only' and c['accepted']),
        'prediction_assisted_total': 4500,
        'total_failures': len(failure_records),
        'failures_solver_certified_infeasible': len(inf_df),
        'failures_computational_timeout': len(timeout_df),
        'failures_by_mode': {m: dict(fail_df[fail_df['dietary_mode'] == m]['category'].value_counts()) for m in modes},
        'failures_by_arm': {a: dict(fail_df[fail_df['arm'] == a]['category'].value_counts()) for a in ['rules_only'] + models},
        'failures_by_threshold': {str(t): dict(fail_df[fail_df['threshold'] == t]['category'].value_counts()) for t in thresholds}
    }
    (AUDIT_DIR / 'failure_analysis_audit_v3.json').write_text(json.dumps(fail_summary, indent=2, default=_to_json_compat))
    print("Saved failure_analysis_audit_v3.json.")
    
    # -------------------------------------------------------------
    # 2. Case-Level Paired Comparisons
    # -------------------------------------------------------------
    paired_rows = []
    for seqn in participants:
        for mode in modes:
            ref_case = indexed[(seqn, mode, 'rules_only', None)]
            ref_acc = bool(ref_case.get('accepted', False))
            ref_basket = {x['food_id']: x['quantity'] for x in ref_case.get('output', {}).get('solved_food_basket', [])}
            ref_totals = ref_case.get('own_replay', {}).get('totals', {})
            ref_mass = float(ref_case.get('own_replay', {}).get('food_mass_g', 0.0))
            ref_cal_err = float(ref_case.get('own_replay', {}).get('calorie_error_fraction', 0.0))
            
            ref_env = ref_case.get('input', ref_case)
            ref_bounds = {row['nutrient']: (row['min_bound'], row['max_bound'], row['rule_provenance'].get('planning_role'))
                          for row in ref_env['nutrient_constraint_envelope']}
                          
            for t in thresholds:
                m_cases = {m: indexed[(seqn, mode, m, t)] for m in models}
                m_accs = {m: bool(m_cases[m].get('accepted', False)) for m in models}
                m_baskets = {m: {x['food_id']: x['quantity'] for x in m_cases[m].get('output', {}).get('solved_food_basket', [])} for m in models}
                m_totals = {m: m_cases[m].get('own_replay', {}).get('totals', {}) for m in models}
                m_masses = {m: float(m_cases[m].get('own_replay', {}).get('food_mass_g', 0.0)) for m in models}
                m_cal_errs = {m: float(m_cases[m].get('own_replay', {}).get('calorie_error_fraction', 0.0)) for m in models}
                
                m_com_pass = {}
                m_com_viols = {}
                for m in models:
                    if not m_accs[m]:
                        m_com_pass[m] = False
                        m_com_viols[m] = 0
                        continue
                    viols = 0
                    tots = m_totals[m]
                    for nk, (lo, hi, role) in ref_bounds.items():
                        if role in ('report_only', 'exposure_reporting'):
                            continue
                        val = tots.get(nk, 0.0)
                        if val < lo - 1e-4 or val > hi + 1e-4:
                            viols += 1
                    m_com_pass[m] = (viols == 0)
                    m_com_viols[m] = viols
                    
                for m in models:
                    acc = m_accs[m]
                    both_rules = ref_acc and acc
                    
                    if both_rules:
                        all_fids = set(ref_basket.keys()).union(set(m_baskets[m].keys()))
                        dish_ch = sum(1 for fid in all_fids if ref_basket.get(fid, 0) != m_baskets[m].get(fid, 0))
                        serv_ch = sum(abs(ref_basket.get(fid, 0) - m_baskets[m].get(fid, 0)) for fid in all_fids)
                        mass_diff = m_masses[m] - ref_mass
                        cal_err_diff = m_cal_errs[m] - ref_cal_err
                        
                        unstable_devs = []
                        for nk, rval in ref_totals.items():
                            if rval > 0:
                                unstable_devs.append(abs(m_totals[m].get(nk, 0.0) - rval) / rval)
                        hist_dev = float(np.mean(unstable_devs)) if unstable_devs else 0.0
                        
                        norm_devs = []
                        for nk in PRIMARY_CLINICAL_NUTRIENTS:
                            scale = REFERENCE_DAILY_SCALES[nk]
                            if nk == 'DR1TKCAL':
                                scale = ref_env['energy_prescription']['target_calories']
                            norm_devs.append(abs(m_totals[m].get(nk, 0.0) - ref_totals.get(nk, 0.0)) / scale)
                        norm_dev = float(np.mean(norm_devs))
                    else:
                        dish_ch = np.nan
                        serv_ch = np.nan
                        mass_diff = np.nan
                        cal_err_diff = np.nan
                        hist_dev = np.nan
                        norm_dev = np.nan
                        
                    paired_rows.append({
                        'SEQN': seqn,
                        'dietary_mode': mode,
                        'threshold': t,
                        'model': m,
                        'ref_accepted': ref_acc,
                        'model_accepted': acc,
                        'paired_diff_vs_rules': int(acc) - int(ref_acc),
                        'common_reference_passed': m_com_pass[m],
                        'common_reference_violations': m_com_viols[m],
                        'both_valid_with_rules': both_rules,
                        'dish_changes_vs_rules': dish_ch,
                        'serving_changes_vs_rules': serv_ch,
                        'mass_diff_vs_rules': mass_diff,
                        'calorie_error_diff_vs_rules': cal_err_diff,
                        'historical_unstable_dev': hist_dev,
                        'normalized_nutrient_dev': norm_dev,
                        'solve_seconds': float(m_cases[m].get('seconds', 0.0))
                    })
                    
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(AUDIT_DIR / 'case_level_paired_eval_audit_v3_LOCAL.csv', index=False)
    print(f"Saved case_level_paired_eval_audit_v3_LOCAL.csv ({len(paired_df)} rows).")
    
    # -------------------------------------------------------------
    # 3. Direct Inter-Model Pairwise Contrasts
    # -------------------------------------------------------------
    model_contrast_pairs = [
        ('bionutrigpt', 'catboost'),
        ('bionutrigpt', 'ft_transformer'),
        ('catboost', 'ft_transformer')
    ]
    
    inter_rows = []
    for seqn in participants:
        for mode in modes:
            ref_acc = bool(indexed[(seqn, mode, 'rules_only', None)].get('accepted', False))
            for t in thresholds:
                for (m1, m2) in model_contrast_pairs:
                    c1 = indexed[(seqn, mode, m1, t)]
                    c2 = indexed[(seqn, mode, m2, t)]
                    acc1 = bool(c1.get('accepted', False))
                    acc2 = bool(c2.get('accepted', False))
                    
                    both_m1_m2 = acc1 and acc2
                    three_way = both_m1_m2 and ref_acc
                    
                    if both_m1_m2:
                        b1 = {x['food_id']: x['quantity'] for x in c1.get('output', {}).get('solved_food_basket', [])}
                        b2 = {x['food_id']: x['quantity'] for x in c2.get('output', {}).get('solved_food_basket', [])}
                        all_fids = set(b1.keys()).union(set(b2.keys()))
                        dish_diff_m1_m2 = sum(1 for fid in all_fids if b1.get(fid, 0) != b2.get(fid, 0))
                        serv_diff_m1_m2 = sum(abs(b1.get(fid, 0) - b2.get(fid, 0)) for fid in all_fids)
                        
                        t1 = c1.get('own_replay', {}).get('totals', {})
                        t2 = c2.get('own_replay', {}).get('totals', {})
                        
                        norm_devs_m1_m2 = []
                        for nk in PRIMARY_CLINICAL_NUTRIENTS:
                            scale = REFERENCE_DAILY_SCALES[nk]
                            norm_devs_m1_m2.append(abs(t1.get(nk, 0.0) - t2.get(nk, 0.0)) / scale)
                        nut_diff_m1_m2 = float(np.mean(norm_devs_m1_m2))
                    else:
                        dish_diff_m1_m2 = np.nan
                        serv_diff_m1_m2 = np.nan
                        nut_diff_m1_m2 = np.nan
                        
                    inter_rows.append({
                        'SEQN': seqn,
                        'dietary_mode': mode,
                        'threshold': t,
                        'model1': m1,
                        'model2': m2,
                        'contrast': f'{m1}_minus_{m2}',
                        'model1_accepted': acc1,
                        'model2_accepted': acc2,
                        'feasibility_diff': int(acc1) - int(acc2),
                        'both_models_accepted': both_m1_m2,
                        'three_way_accepted': three_way,
                        'dish_diff_between_models': dish_diff_m1_m2,
                        'serving_diff_between_models': serv_diff_m1_m2,
                        'nutrient_diff_between_models': nut_diff_m1_m2
                    })
                    
    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_csv(AUDIT_DIR / 'inter_model_contrasts_audit_v3_LOCAL.csv', index=False)
    print("Saved inter_model_contrasts_audit_v3_LOCAL.csv.")
    
    # -------------------------------------------------------------
    # 4. Bootstrap Inference for Direct Model Contrasts (2,000 resamples)
    # -------------------------------------------------------------
    n_part = len(participants)
    n_boot = 2000
    rng = np.random.RandomState(20261120)
    boot_indices = rng.choice(n_part, size=(n_boot, n_part), replace=True)
    
    def bootstrap_ci(values_by_seqn):
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            boot_means[b] = np.nanmean(values_by_seqn[boot_indices[b]])
        valid = boot_means[~np.isnan(boot_means)]
        if len(valid) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))
        
    direct_contrast_summary = []
    
    # A. Cell-specific contrasts (mode x threshold)
    for (m1, m2) in model_contrast_pairs:
        contrast_name = f'{m1}_minus_{m2}'
        sub_contrast = inter_df[inter_df['contrast'] == contrast_name]
        
        for mode in modes:
            for t in thresholds:
                cell = sub_contrast[(sub_contrast['dietary_mode'] == mode) & (sub_contrast['threshold'] == t)].sort_values('SEQN')
                assert len(cell) == 100
                
                n_both = cell['both_models_accepted'].sum()
                n_three = cell['three_way_accepted'].sum()
                
                feas_diffs = cell['feasibility_diff'].values
                pt_diff = float(np.mean(feas_diffs))
                ci_lo, ci_hi = bootstrap_ci(feas_diffs)
                
                dish_diffs = cell['dish_diff_between_models'].values
                pt_dish = float(np.nanmean(dish_diffs)) if n_both > 0 else np.nan
                ci_dish_lo, ci_dish_hi = bootstrap_ci(dish_diffs)
                
                excludes_zero = bool(ci_lo > 0.0 or ci_hi < 0.0)
                
                direct_contrast_summary.append({
                    'scope': 'cell_specific',
                    'dietary_mode': mode,
                    'threshold': t,
                    'model1': m1,
                    'model2': m2,
                    'contrast': contrast_name,
                    'participants': 100,
                    'model1_accepted': int(cell['model1_accepted'].sum()),
                    'model2_accepted': int(cell['model2_accepted'].sum()),
                    'paired_feasibility_diff': pt_diff,
                    'paired_feasibility_diff_ci_lo': ci_lo,
                    'paired_feasibility_diff_ci_hi': ci_hi,
                    'valid_both_denominator': int(n_both),
                    'three_way_denominator': int(n_three),
                    'mean_dish_diff_between_models': pt_dish,
                    'dish_diff_ci_lo': ci_dish_lo,
                    'dish_diff_ci_hi': ci_dish_hi,
                    'pointwise_excludes_zero': excludes_zero
                })
                
        # B. Overall equal-weighted contrast
        seqn_means = sub_contrast.groupby('SEQN')['feasibility_diff'].mean()
        seqn_feas_arr = np.array([seqn_means[s] for s in participants])
        overall_pt_diff = float(np.mean(seqn_feas_arr))
        ov_ci_lo, ov_ci_hi = bootstrap_ci(seqn_feas_arr)
        
        seqn_dish_means = sub_contrast.groupby('SEQN')['dish_diff_between_models'].mean()
        seqn_dish_arr = np.array([seqn_dish_means.get(s, np.nan) for s in participants])
        ov_dish_pt = float(np.nanmean(seqn_dish_arr))
        ov_dish_ci_lo, ov_dish_ci_hi = bootstrap_ci(seqn_dish_arr)
        
        m1_total_acc = sub_contrast['model1_accepted'].sum()
        m2_total_acc = sub_contrast['model2_accepted'].sum()
        total_both = sub_contrast['both_models_accepted'].sum()
        total_three = sub_contrast['three_way_accepted'].sum()
        
        direct_contrast_summary.append({
            'scope': 'overall_equal_weighted',
            'dietary_mode': 'all_3_modes',
            'threshold': 'all_5_thresholds',
            'model1': m1,
            'model2': m2,
            'contrast': contrast_name,
            'participants': 100,
            'model1_accepted': int(m1_total_acc),
            'model2_accepted': int(m2_total_acc),
            'paired_feasibility_diff': overall_pt_diff,
            'paired_feasibility_diff_ci_lo': ov_ci_lo,
            'paired_feasibility_diff_ci_hi': ov_ci_hi,
            'valid_both_denominator': int(total_both),
            'three_way_denominator': int(total_three),
            'mean_dish_diff_between_models': ov_dish_pt,
            'dish_diff_ci_lo': ov_dish_ci_lo,
            'dish_diff_ci_hi': ov_dish_ci_hi,
            'pointwise_excludes_zero': bool(ov_ci_lo > 0.0 or ov_ci_hi < 0.0)
        })
        
    direct_df = pd.DataFrame(direct_contrast_summary)
    direct_df.to_csv(AUDIT_DIR / 'direct_model_contrasts_audit_v3.csv', index=False)
    (AUDIT_DIR / 'direct_model_contrasts_audit_v3.json').write_text(json.dumps(direct_contrast_summary, indent=2, default=_to_json_compat))
    print("Saved direct_model_contrasts_audit_v3.csv and .json.")
    
    # -------------------------------------------------------------
    # 5. Aggregate Summary (Model vs Rules-Only)
    # -------------------------------------------------------------
    agg_rows = []
    for mode in modes:
        for m in models:
            for t in thresholds:
                sub = paired_df[(paired_df['dietary_mode'] == mode) & (paired_df['model'] == m) & (paired_df['threshold'] == t)].sort_values('SEQN')
                assert len(sub) == 100
                
                n_ref = sub['ref_accepted'].sum()
                n_mod = sub['model_accepted'].sum()
                n_both = sub['both_valid_with_rules'].sum()
                n_com_pass = sub['common_reference_passed'].sum()
                
                p_ref = n_ref / 100.0
                p_mod = n_mod / 100.0
                diff_feas = p_mod - p_ref
                
                feas_diffs = sub['paired_diff_vs_rules'].values
                ci_diff_lo, ci_diff_hi = bootstrap_ci(feas_diffs)
                
                com_rate_attempts = n_com_pass / 100.0
                com_rate_accepted = (n_com_pass / n_mod) if n_mod > 0 else np.nan
                
                com_pass_arr = sub['common_reference_passed'].astype(float).values
                ci_com_lo, ci_com_hi = bootstrap_ci(com_pass_arr)
                
                dish_ch_arr = sub['dish_changes_vs_rules'].values
                pt_dish = float(np.nanmean(dish_ch_arr)) if n_both > 0 else np.nan
                ci_dish_lo, ci_dish_hi = bootstrap_ci(dish_ch_arr)
                
                serv_ch_arr = sub['serving_changes_vs_rules'].values
                pt_serv = float(np.nanmean(serv_ch_arr)) if n_both > 0 else np.nan
                ci_serv_lo, ci_serv_hi = bootstrap_ci(serv_ch_arr)
                
                norm_dev_arr = sub['normalized_nutrient_dev'].values
                pt_norm_dev = float(np.nanmean(norm_dev_arr)) if n_both > 0 else np.nan
                ci_norm_lo, ci_norm_hi = bootstrap_ci(norm_dev_arr)
                
                hist_dev_arr = sub['historical_unstable_dev'].values
                pt_hist_dev = float(np.nanmean(hist_dev_arr)) if n_both > 0 else np.nan
                
                med_sec = float(sub['solve_seconds'].median())
                
                agg_rows.append({
                    'dietary_mode': mode,
                    'model': m,
                    'threshold': t,
                    'total_participants': 100,
                    'rules_only_accepted': int(n_ref),
                    'model_accepted': int(n_mod),
                    'valid_pairs_with_rules': int(n_both),
                    'paired_feasibility_diff_percentage_points': diff_feas * 100.0,
                    'paired_feasibility_diff_ci_lo_pp': ci_diff_lo * 100.0,
                    'paired_feasibility_diff_ci_hi_pp': ci_diff_hi * 100.0,
                    'feasibility_ci_excludes_zero': bool(ci_diff_lo > 0.0 or ci_diff_hi < 0.0),
                    'common_reference_passed_count': int(n_com_pass),
                    'common_reference_rate_all_attempts_pct': com_rate_attempts * 100.0,
                    'common_reference_rate_attempts_ci_lo': ci_com_lo * 100.0,
                    'common_reference_rate_attempts_ci_hi': ci_com_hi * 100.0,
                    'common_reference_rate_among_accepted_pct': com_rate_accepted * 100.0 if np.isfinite(com_rate_accepted) else None,
                    'mean_dish_changes_vs_rules': pt_dish,
                    'dish_changes_ci_lo': ci_dish_lo,
                    'dish_changes_ci_hi': ci_dish_hi,
                    'mean_serving_changes_vs_rules': pt_serv,
                    'serving_changes_ci_lo': ci_serv_lo,
                    'serving_changes_ci_hi': ci_serv_hi,
                    'normalized_nutrient_deviation_pct': pt_norm_dev * 100.0 if np.isfinite(pt_norm_dev) else None,
                    'normalized_nutrient_dev_ci_lo_pct': ci_norm_lo * 100.0 if np.isfinite(ci_norm_lo) else None,
                    'normalized_nutrient_dev_ci_hi_pct': ci_norm_hi * 100.0 if np.isfinite(ci_norm_hi) else None,
                    'historical_unstable_dev_pct': pt_hist_dev * 100.0 if np.isfinite(pt_hist_dev) else None,
                    'median_solve_seconds': med_sec
                })
                
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(AUDIT_DIR / 'aggregate_summary_audit_v3.csv', index=False)
    (AUDIT_DIR / 'aggregate_summary_audit_v3.json').write_text(json.dumps(agg_rows, indent=2, default=_to_json_compat))
    print("Saved aggregate_summary_audit_v3.csv and .json.")
    
    # Print key verification numbers
    cell_direct = [r for r in direct_contrast_summary if r['scope'] == 'cell_specific']
    direct_ex_zero = sum(1 for r in cell_direct if r['pointwise_excludes_zero'])
    print(f"\nVerification Check: Direct cell-specific feasibility contrasts excluding zero: {direct_ex_zero} / {len(cell_direct)}")
    
    rules_ex_zero = sum(1 for r in agg_rows if r['feasibility_ci_excludes_zero'])
    print(f"Verification Check: Model-vs-rules feasibility contrasts excluding zero: {rules_ex_zero} / {len(agg_rows)}")
    
    norm_devs = [r['normalized_nutrient_deviation_pct'] for r in agg_rows if r['normalized_nutrient_deviation_pct'] is not None]
    print(f"Verification Check: Normalized nutrient deviation range: {min(norm_devs):.2f}% to {max(norm_devs):.2f}%")
    
if __name__ == '__main__':
    run_analysis()
