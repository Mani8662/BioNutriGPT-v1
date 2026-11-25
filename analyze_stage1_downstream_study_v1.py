"""Statistical analysis and paired bootstrap evaluation for Stage 1 downstream study v1.

Evaluates primary paired engineering feasibility difference, common-reference compliance,
dish and serving changes, calorie errors, and nutrient deviations.
Uses 2,000 participant-level bootstrap resamples with seed 20261120 accelerated on CUDA GPU.
"""
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch

R = Path(__file__).resolve().parent
O = R / 'Saved Model/stage1_downstream_study_v1'
CASES_DIR = O / 'cases'

def load_all_cases():
    print("Loading all 4,800 cases...")
    case_files = sorted(list(CASES_DIR.glob('case_*_LOCAL.json')))
    assert len(case_files) == 4800, f"Expected 4800 cases, found {len(case_files)}"
    cases = []
    for cf in case_files:
        cases.append(json.loads(cf.read_text(encoding='utf-8')))
    return cases

def analyze_study():
    cases = load_all_cases()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bootstrap acceleration device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    
    indexed = {}
    participants = sorted(list({c['SEQN'] for c in cases}))
    assert len(participants) == 100, f"Expected 100 participants, got {len(participants)}"
    
    modes = ['pure_vegetarian', 'veg_nonveg_combo', 'pure_non_vegetarian']
    models = ['bionutrigpt', 'catboost', 'ft_transformer']
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    
    for c in cases:
        key = (c['SEQN'], c['dietary_mode'], c['arm'], c['threshold'])
        indexed[key] = c
        
    records = []
    for seqn in participants:
        for mode in modes:
            ref_case = indexed[(seqn, mode, 'rules_only', None)]
            ref_acc = bool(ref_case.get('accepted', False))
            ref_basket = {x['food_id']: x['quantity'] for x in ref_case.get('output', {}).get('solved_food_basket', [])}
            ref_totals = ref_case.get('own_replay', {}).get('totals', {})
            ref_mass = float(ref_case.get('own_replay', {}).get('food_mass_g', 0.0))
            ref_cal_err = float(ref_case.get('own_replay', {}).get('calorie_error_fraction', 0.0))
            
            for mod in models:
                for t in thresholds:
                    m_case = indexed[(seqn, mode, mod, t)]
                    m_acc = bool(m_case.get('accepted', False))
                    com_acc = bool(m_case.get('reference_replay', {}).get('passed', False))
                    
                    m_basket = {x['food_id']: x['quantity'] for x in m_case.get('output', {}).get('solved_food_basket', [])}
                    m_totals = m_case.get('own_replay', {}).get('totals', {})
                    m_mass = float(m_case.get('own_replay', {}).get('food_mass_g', 0.0))
                    m_cal_err = float(m_case.get('own_replay', {}).get('calorie_error_fraction', 0.0))
                    
                    both_valid = ref_acc and m_acc
                    
                    if both_valid:
                        all_fids = set(ref_basket.keys()).union(set(m_basket.keys()))
                        dish_changes = sum(1 for fid in all_fids if ref_basket.get(fid, 0) != m_basket.get(fid, 0))
                        serv_changes = sum(abs(ref_basket.get(fid, 0) - m_basket.get(fid, 0)) for fid in all_fids)
                        mass_diff = m_mass - ref_mass
                        cal_err_diff = m_cal_err - ref_cal_err
                        
                        devs = []
                        for nk, rval in ref_totals.items():
                            if rval > 0:
                                mval = m_totals.get(nk, 0.0)
                                devs.append(abs(mval - rval) / rval)
                        nut_dev = float(np.mean(devs)) if devs else 0.0
                    else:
                        dish_changes = np.nan
                        serv_changes = np.nan
                        mass_diff = np.nan
                        cal_err_diff = np.nan
                        nut_dev = np.nan
                        
                    records.append({
                        'SEQN': seqn,
                        'dietary_mode': mode,
                        'model': mod,
                        'threshold': t,
                        'ref_accepted': ref_acc,
                        'model_accepted': m_acc,
                        'paired_feasibility_diff': int(m_acc) - int(ref_acc),
                        'common_reference_passed': com_acc,
                        'both_valid': both_valid,
                        'dish_changes': dish_changes,
                        'serv_changes': serv_changes,
                        'mass_diff': mass_diff,
                        'cal_err_diff': cal_err_diff,
                        'nut_dev': nut_dev,
                        'model_seconds': float(m_case.get('seconds', 0.0)),
                        'ref_seconds': float(ref_case.get('seconds', 0.0))
                    })
                    
    df = pd.DataFrame(records)
    df.to_csv(O / 'case_level_paired_eval_LOCAL.csv', index=False)
    print(f"Saved case_level_paired_eval_LOCAL.csv ({len(df)} rows).")
    
    # GPU-accelerated participant bootstrap
    n_part = len(participants)
    n_boot = 2000
    rng = np.random.RandomState(20261120)
    boot_indices_np = rng.choice(n_part, size=(n_boot, n_part), replace=True)
    boot_idx_t = torch.as_tensor(boot_indices_np, dtype=torch.long, device=device)
    
    summary_rows = []
    grouped = df.groupby(['dietary_mode', 'model', 'threshold'])
    print(f"Computing {n_boot} participant-level bootstrap replicates on GPU ({device})...")
    
    for (mode, mod, t), sub in grouped:
        n_cases = len(sub)
        n_ref_acc = sub['ref_accepted'].sum()
        n_mod_acc = sub['model_accepted'].sum()
        n_com_pass = sub['common_reference_passed'].sum()
        n_both = sub['both_valid'].sum()
        
        p_ref = n_ref_acc / n_cases
        p_mod = n_mod_acc / n_cases
        p_com = n_com_pass / n_cases
        diff_feas = p_mod - p_ref
        
        both_sub = sub[sub['both_valid']]
        mean_dish_changes = both_sub['dish_changes'].mean() if len(both_sub) > 0 else np.nan
        mean_serv_changes = both_sub['serv_changes'].mean() if len(both_sub) > 0 else np.nan
        mean_nut_dev = both_sub['nut_dev'].mean() if len(both_sub) > 0 else np.nan
        mean_cal_err_diff = both_sub['cal_err_diff'].mean() if len(both_sub) > 0 else np.nan
        
        # Sort by SEQN so ordering aligns with participants
        sub_sorted = sub.sort_values('SEQN')
        
        diff_feas_t = torch.as_tensor(sub_sorted['paired_feasibility_diff'].values, dtype=torch.float32, device=device)
        com_pass_t = torch.as_tensor(sub_sorted['common_reference_passed'].values, dtype=torch.float32, device=device)
        
        # GPU bootstrap for feasibility diff and common-reference rate
        diff_feas_boot = diff_feas_t[boot_idx_t].mean(dim=1).cpu().numpy()
        com_pass_boot = com_pass_t[boot_idx_t].mean(dim=1).cpu().numpy()
        
        # For dish changes and nutrient deviation (only both_valid cases)
        dish_vals = sub_sorted['dish_changes'].values
        nut_vals = sub_sorted['nut_dev'].values
        
        # CPU-safe CI handling for NaN values where one arm failed
        dish_boot = []
        nut_boot = []
        for b in range(n_boot):
            sample_idx = boot_indices_np[b]
            d_s = dish_vals[sample_idx]
            n_s = nut_vals[sample_idx]
            d_valid = d_s[~np.isnan(d_s)]
            n_valid = n_s[~np.isnan(n_s)]
            dish_boot.append(np.mean(d_valid) if len(d_valid) > 0 else np.nan)
            nut_boot.append(np.mean(n_valid) if len(n_valid) > 0 else np.nan)
            
        def get_ci(arr):
            valid = np.array(arr)[~np.isnan(arr)]
            if len(valid) == 0:
                return (np.nan, np.nan)
            return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))
            
        ci_diff = (float(np.percentile(diff_feas_boot, 2.5)), float(np.percentile(diff_feas_boot, 97.5)))
        ci_com = (float(np.percentile(com_pass_boot, 2.5)), float(np.percentile(com_pass_boot, 97.5)))
        ci_dish = get_ci(dish_boot)
        ci_dev = get_ci(nut_boot)
        
        summary_rows.append({
            'dietary_mode': mode,
            'model': mod,
            'threshold': t,
            'participants': n_cases,
            'rules_only_accepted': int(n_ref_acc),
            'model_accepted': int(n_mod_acc),
            'paired_feasibility_diff': float(diff_feas),
            'paired_feasibility_diff_ci_lo': ci_diff[0],
            'paired_feasibility_diff_ci_hi': ci_diff[1],
            'common_reference_passed': int(n_com_pass),
            'common_reference_rate': float(p_com),
            'common_reference_rate_ci_lo': ci_com[0],
            'common_reference_rate_ci_hi': ci_com[1],
            'valid_pairs_both_accepted': int(n_both),
            'mean_dish_changes': float(mean_dish_changes) if np.isfinite(mean_dish_changes) else None,
            'dish_changes_ci_lo': ci_dish[0],
            'dish_changes_ci_hi': ci_dish[1],
            'mean_serving_changes': float(mean_serv_changes) if np.isfinite(mean_serv_changes) else None,
            'mean_nutrient_deviation': float(mean_nut_dev) if np.isfinite(mean_nut_dev) else None,
            'nutrient_deviation_ci_lo': ci_dev[0],
            'nutrient_deviation_ci_hi': ci_dev[1],
            'mean_calorie_error_diff': float(mean_cal_err_diff) if np.isfinite(mean_cal_err_diff) else None,
            'median_wall_seconds': float(sub['model_seconds'].median())
        })
        
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(O / 'aggregate_summary.csv', index=False)
    (O / 'aggregate_summary.json').write_text(json.dumps(summary_rows, indent=2))
    print(f"Generated aggregate_summary.csv and aggregate_summary.json ({len(summary_rows)} rows) with GPU acceleration.")

if __name__ == '__main__':
    analyze_study()
