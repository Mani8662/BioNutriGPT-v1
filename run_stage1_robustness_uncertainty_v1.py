"""Stage 1 Robustness Uncertainty Evaluation Runner (10,000 Paired Participant Resamples).

Computes:
1. Within-model deterioration from baseline to stress across 6 models, 8 stress scenarios,
   3 outcomes, 2 probability variants, and 2 metrics (AUROC, Brier).
2. Direct paired inter-model contrasts in deterioration (BioNutriGPT minus comparator),
   alongside direct contrasts in absolute stressed performance.
3. 95% pointwise percentile bootstrap confidence intervals from 10,000 paired resamples (seed 20261121).
4. Full validation and verification records.
"""
import os
import sys
import json
import time
import hashlib
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

MODELS = ['bionutrigpt', 'catboost', 'ft_transformer', 'hist_gradient_boosting', 'logistic_regression', 'mlp']
SCENARIOS = [
    'baseline',
    'missing_diet',
    'missing_lipids',
    'missing_both',
    'missing_diet_20pct_seed17',
    'missing_diet_20pct_seed29',
    'diet_noise_10pct_seed17',
    'lipid_noise_5pct_seed17',
    'joint_noise_seed17'
]
STRESS_SCENARIOS = SCENARIOS[1:]
OUTCOMES = ['TARGET_DIABETES', 'TARGET_CVD', 'TARGET_OBESITY']
OUTCOME_INDICES = {'TARGET_DIABETES': 0, 'TARGET_OBESITY': 1, 'TARGET_CVD': 2}
VARIANTS = ['calibrated', 'raw']
METRICS = ['auroc', 'brier']

BOOTSTRAP_SEED = 20261121
BOOTSTRAP_REPLICATES = 10000

def fast_auc(y_true, y_score):
    """Exact trapezoidal AUROC identical to sklearn.metrics.roc_auc_score down to 1e-14."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    desc = np.argsort(y_score, kind='mergesort')[::-1]
    y_true_s = y_true[desc]
    y_score_s = y_score[desc]
    distinct = np.where(np.diff(y_score_s))[0]
    thresh = np.r_[distinct, y_true_s.size - 1]
    tps = np.cumsum(y_true_s)[thresh]
    fps = 1 + thresh - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return np.trapezoid(tpr, fpr)

def _worker_bootstrap_chunk(args):
    """Worker task computing bootstrap replicate metrics for a batch of replicates."""
    p_chunk, y_chunk, boot_idx_chunk = args
    n_reps = boot_idx_chunk.shape[0]
    auc_reps = np.empty(n_reps, dtype=np.float64)
    brier_reps = np.empty(n_reps, dtype=np.float64)
    
    sq = (p_chunk - y_chunk) ** 2
    brier_reps = np.mean(sq[boot_idx_chunk], axis=1)
    
    for b in range(n_reps):
        auc_reps[b] = fast_auc(y_chunk[boot_idx_chunk[b]], p_chunk[boot_idx_chunk[b]])
        
    return auc_reps, brier_reps

def compute_bootstrap_distribution(p, y, boot_idx, pool, chunk_size=1000):
    """Compute 10,000 bootstrap AUROC and Brier values using bounded multiprocessing."""
    B = boot_idx.shape[0]
    tasks = []
    for start_idx in range(0, B, chunk_size):
        end_idx = min(start_idx + chunk_size, B)
        tasks.append((p, y, boot_idx[start_idx:end_idx]))
        
    results = pool.map(_worker_bootstrap_chunk, tasks)
    auc_all = np.concatenate([r[0] for r in results])
    brier_all = np.concatenate([r[1] for r in results])
    return auc_all, brier_all

def main():
    start_total = time.perf_counter()
    print("=== Starting Stage 1 Robustness Paired Uncertainty Evaluation ===")
    
    root = Path(__file__).resolve().parent
    rob_dir = root / 'Saved Model/stage1_robustness_v3'
    out_dir = root / 'Saved Model/stage1_robustness_uncertainty_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load reference dataset and verify integrity
    ref_df = pd.read_csv(root / 'Data/stage1_repaired_v2/temporal_ml_ready.csv')
    ref_seqns = ref_df['SEQN'].to_numpy(dtype=np.int64)
    N = len(ref_seqns)
    assert N == 7004, f"Expected 7004 participants, found {N}"
    assert len(np.unique(ref_seqns)) == 7004, "Duplicate SEQNs found"
    
    # 2. Load all 54 prediction files
    print(f"Loading 54 prediction archives from {rob_dir}...")
    preds = {}
    labels = None
    
    for m in MODELS:
        preds[m] = {}
        for s in SCENARIOS:
            fpath = rob_dir / f'{m}_{s}_predictions_LOCAL.npz'
            assert fpath.exists(), f"Missing prediction file: {fpath}"
            data = np.load(fpath)
            assert np.array_equal(data['ids'], ref_seqns), f"Participant ID mismatch in {m}_{s}"
            if labels is None:
                labels = data['labels'].astype(np.float64)
            else:
                assert np.array_equal(data['labels'], labels), f"Label mismatch in {m}_{s}"
            
            # Check finite and bounds
            for var in VARIANTS:
                arr = data[var].astype(np.float64)
                assert np.isfinite(arr).all(), f"Nonfinite predictions in {m}_{s}_{var}"
                assert ((arr >= 0.0) & (arr <= 1.0)).all(), f"Out-of-bounds predictions in {m}_{s}_{var}"
                
            preds[m][s] = {
                'raw': data['raw'].astype(np.float64),
                'calibrated': data['calibrated'].astype(np.float64)
            }
            
    print("  [PASS] All 54 prediction archives verified: identical IDs, identical labels, finite [0, 1] bounds.")
    
    # 3. Generate 10,000 bootstrap resample index matrix
    print(f"Generating {BOOTSTRAP_REPLICATES} participant-level resample draws (seed={BOOTSTRAP_SEED})...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, N, size=(BOOTSTRAP_REPLICATES, N), dtype=np.int32)
    boot_bytes_sha256 = hashlib.sha256(boot_idx.tobytes()).hexdigest()
    
    # Check valid replicate counts for all outcomes
    valid_replicates = {}
    for outcome in OUTCOMES:
        j = OUTCOME_INDICES[outcome]
        y_col = labels[:, j]
        pos_counts = np.sum(y_col[boot_idx], axis=1)
        valid_mask = (pos_counts > 0) & (pos_counts < N)
        valid_replicates[outcome] = int(np.sum(valid_mask))
        assert valid_replicates[outcome] == BOOTSTRAP_REPLICATES, f"Found invalid one-class samples for {outcome}"
    print(f"  [PASS] Bootstrap resample contract verified: 10,000 / 10,000 valid two-class samples across all outcomes.")
    
    # Save seed and index contract
    seed_record = {
        'seed': BOOTSTRAP_SEED,
        'replicates': BOOTSTRAP_REPLICATES,
        'n_participants': N,
        'resample_sha256': boot_bytes_sha256,
        'valid_replicates': valid_replicates
    }
    (out_dir / 'bootstrap_seed_record.json').write_text(json.dumps(seed_record, indent=2))
    
    # 4. Multiprocessing execution across all combinations
    num_cpus = min(8, os.cpu_count() or 1)
    print(f"Executing bootstrap calculations with {num_cpus} parallel CPU worker processes...")
    
    within_records = []
    between_records = []
    summary_tables = {'calibrated': {}, 'raw': {}}
    timing_records = {}
    
    with mp.Pool(num_cpus) as pool:
        for var in VARIANTS:
            summary_tables[var] = {}
            for outcome in OUTCOMES:
                out_start = time.perf_counter()
                j = OUTCOME_INDICES[outcome]
                y_true = labels[:, j]
                summary_tables[var][outcome] = {}
                
                print(f"  Processing {var} | {outcome}...")
                
                # Compute baseline distributions and points for all 6 models
                base_boot = {}
                base_pt = {}
                for m in MODELS:
                    p_base = preds[m]['baseline'][var][:, j]
                    auc_boot, brier_boot = compute_bootstrap_distribution(p_base, y_true, boot_idx, pool, chunk_size=1250)
                    base_boot[m] = {'auroc': auc_boot, 'brier': brier_boot}
                    base_pt[m] = {
                        'auroc': float(fast_auc(y_true, p_base)),
                        'brier': float(brier_score_loss(y_true, p_base))
                    }
                    
                # Compute stress distributions and within-model deteriorations
                stress_boot = {m: {} for m in MODELS}
                stress_pt = {m: {} for m in MODELS}
                det_boot = {m: {} for m in MODELS}
                det_pt = {m: {} for m in MODELS}
                
                for s in STRESS_SCENARIOS:
                    for m in MODELS:
                        p_stress = preds[m][s][var][:, j]
                        auc_boot, brier_boot = compute_bootstrap_distribution(p_stress, y_true, boot_idx, pool, chunk_size=1250)
                        stress_boot[m][s] = {'auroc': auc_boot, 'brier': brier_boot}
                        stress_pt[m][s] = {
                            'auroc': float(fast_auc(y_true, p_stress)),
                            'brier': float(brier_score_loss(y_true, p_stress))
                        }
                        
                        # AUROC deterioration = baseline - stress (positive = worse)
                        det_auc_b = base_boot[m]['auroc'] - stress_boot[m][s]['auroc']
                        det_auc_p = base_pt[m]['auroc'] - stress_pt[m][s]['auroc']
                        
                        # Brier deterioration = stress - baseline (positive = worse)
                        det_brier_b = stress_boot[m][s]['brier'] - base_boot[m]['brier']
                        det_brier_p = stress_pt[m][s]['brier'] - base_pt[m]['brier']
                        
                        det_boot[m][s] = {'auroc': det_auc_b, 'brier': det_brier_b}
                        det_pt[m][s] = {'auroc': det_auc_p, 'brier': det_brier_p}
                        
                        # Save within-model records
                        for met in METRICS:
                            b_dist = base_boot[m][met]
                            s_dist = stress_boot[m][s][met]
                            d_dist = det_boot[m][s][met]
                            
                            within_records.append({
                                'model': m,
                                'scenario': s,
                                'outcome': outcome,
                                'variant': var,
                                'metric': met,
                                'baseline_point': base_pt[m][met],
                                'baseline_ci_lower': float(np.percentile(b_dist, 2.5)),
                                'baseline_ci_upper': float(np.percentile(b_dist, 97.5)),
                                'stressed_point': stress_pt[m][s][met],
                                'stressed_ci_lower': float(np.percentile(s_dist, 2.5)),
                                'stressed_ci_upper': float(np.percentile(s_dist, 97.5)),
                                'deterioration_point': det_pt[m][s][met],
                                'deterioration_ci_lower': float(np.percentile(d_dist, 2.5)),
                                'deterioration_ci_upper': float(np.percentile(d_dist, 97.5)),
                                'valid_replicates': valid_replicates[outcome],
                                'total_replicates': BOOTSTRAP_REPLICATES
                            })
                            
                    # Compute between-model direct contrasts: BioNutriGPT minus comparator
                    focal = 'bionutrigpt'
                    summary_tables[var][outcome][s] = {}
                    
                    for comp in MODELS:
                        if comp == focal:
                            continue
                        for met in METRICS:
                            # Difference in deterioration: BioNutriGPT - Comparator (positive = BioNutriGPT deteriorated more)
                            diff_det_b = det_boot[focal][s][met] - det_boot[comp][s][met]
                            diff_det_p = det_pt[focal][s][met] - det_pt[comp][s][met]
                            diff_det_ci_lo = float(np.percentile(diff_det_b, 2.5))
                            diff_det_ci_hi = float(np.percentile(diff_det_b, 97.5))
                            
                            # Difference in absolute stressed performance: BioNutriGPT - Comparator
                            diff_stress_b = stress_boot[focal][s][met] - stress_boot[comp][s][met]
                            diff_stress_p = stress_pt[focal][s][met] - stress_pt[comp][s][met]
                            diff_stress_ci_lo = float(np.percentile(diff_stress_b, 2.5))
                            diff_stress_ci_hi = float(np.percentile(diff_stress_b, 97.5))
                            
                            between_records.append({
                                'comparator': comp,
                                'scenario': s,
                                'outcome': outcome,
                                'variant': var,
                                'metric': met,
                                'bionutrigpt_deterioration_point': det_pt[focal][s][met],
                                'comparator_deterioration_point': det_pt[comp][s][met],
                                'diff_deterioration_point': diff_det_p,
                                'diff_deterioration_ci_lower': diff_det_ci_lo,
                                'diff_deterioration_ci_upper': diff_det_ci_hi,
                                'excludes_zero': bool(diff_det_ci_lo > 0 or diff_det_ci_hi < 0),
                                'bionutrigpt_stressed_point': stress_pt[focal][s][met],
                                'comparator_stressed_point': stress_pt[comp][s][met],
                                'diff_stressed_point': diff_stress_p,
                                'diff_stressed_ci_lower': diff_stress_ci_lo,
                                'diff_stressed_ci_upper': diff_stress_ci_hi,
                                'valid_replicates': valid_replicates[outcome],
                                'total_replicates': BOOTSTRAP_REPLICATES
                            })
                            
                            if comp in ('catboost', 'ft_transformer'):
                                if comp not in summary_tables[var][outcome][s]:
                                    summary_tables[var][outcome][s][comp] = {}
                                summary_tables[var][outcome][s][comp][met] = {
                                    'diff_deterioration_point': diff_det_p,
                                    'diff_deterioration_ci': [diff_det_ci_lo, diff_det_ci_hi],
                                    'excludes_zero': bool(diff_det_ci_lo > 0 or diff_det_ci_hi < 0),
                                    'bionutrigpt_stressed': stress_pt[focal][s][met],
                                    'comparator_stressed': stress_pt[comp][s][met],
                                    'diff_stressed_point': diff_stress_p,
                                    'diff_stressed_ci': [diff_stress_ci_lo, diff_stress_ci_hi]
                                }
                                
                timing_records[f'{var}_{outcome}'] = time.perf_counter() - out_start
                print(f"    Completed in {timing_records[f'{var}_{outcome}']:.2f}s")
                
    # 5. Export results
    print("Exporting results to disk...")
    df_within = pd.DataFrame(within_records)
    df_between = pd.DataFrame(between_records)
    
    assert len(df_within) == 576, f"Expected 576 within records, got {len(df_within)}"
    assert len(df_between) == 480, f"Expected 480 between records, got {len(df_between)}"
    
    df_within.to_csv(out_dir / 'within_model_deterioration.csv', index=False)
    (out_dir / 'within_model_deterioration.json').write_text(json.dumps(within_records, indent=2))
    
    df_between.to_csv(out_dir / 'between_model_contrasts.csv', index=False)
    (out_dir / 'between_model_contrasts.json').write_text(json.dumps(between_records, indent=2))
    
    (out_dir / 'summary_tables.json').write_text(json.dumps(summary_tables, indent=2))
    
    total_elapsed = time.perf_counter() - start_total
    timing_records['total_wall_seconds'] = total_elapsed
    timing_records['worker_cpus'] = num_cpus
    (out_dir / 'timings.json').write_text(json.dumps(timing_records, indent=2))
    
    print(f"=== Analysis Successfully Completed in {total_elapsed:.2f}s ===")
    print(f"  Within-model records: {len(df_within)} rows saved.")
    print(f"  Between-model contrasts: {len(df_between)} rows saved.")

if __name__ == '__main__':
    main()
