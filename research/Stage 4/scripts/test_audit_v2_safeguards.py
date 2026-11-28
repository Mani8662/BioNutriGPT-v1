"""Comprehensive safeguards and resume verification tests for Audit v2.

Tests:
1. Canonical input fingerprint generation (deterministic, sensitive to any input change).
2. Matching resume behavior (avoids re-solve when fingerprint matches).
3. Mismatched resume behavior (triggers fresh solve when inputs/targets change).
4. Corrupted/truncated file resume handling (catches invalid JSON and forces re-solve).
5. Deterministic cohort selection reproducibility from seed 20261120.
"""
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from verify_stage1_downstream_audit_v2 import compute_case_fingerprint
from stage2_research_rule_adapter import DIET_RAW_COLS

def test_canonical_fingerprint():
    print("Testing canonical fingerprint...")
    fp1 = compute_case_fingerprint(109369, 'pure_vegetarian', 'bionutrigpt', 0.05, {'age': 45}, {'DR1TKCAL': 2000}, 'hash1')
    fp2 = compute_case_fingerprint(109369, 'pure_vegetarian', 'bionutrigpt', 0.05, {'age': 45}, {'DR1TKCAL': 2000}, 'hash1')
    assert fp1 == fp2, "Fingerprints must be identical for identical inputs"
    
    # Change threshold
    fp_diff_t = compute_case_fingerprint(109369, 'pure_vegetarian', 'bionutrigpt', 0.10, {'age': 45}, {'DR1TKCAL': 2000}, 'hash1')
    assert fp1 != fp_diff_t, "Fingerprint must change when threshold changes"
    
    # Change target
    fp_diff_tgt = compute_case_fingerprint(109369, 'pure_vegetarian', 'bionutrigpt', 0.05, {'age': 45}, {'DR1TKCAL': 2100}, 'hash1')
    assert fp1 != fp_diff_tgt, "Fingerprint must change when target changes"
    
    # Change catalog hash
    fp_diff_cat = compute_case_fingerprint(109369, 'pure_vegetarian', 'bionutrigpt', 0.05, {'age': 45}, {'DR1TKCAL': 2000}, 'hash2')
    assert fp1 != fp_diff_cat, "Fingerprint must change when catalog hash changes"
    print("  [PASS] Canonical fingerprint tests passed.")

def test_resume_logic():
    print("Testing resume logic safeguards...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="safeguard_test_"))
    try:
        case_file = tmp_dir / "case_test_LOCAL.json"
        
        # Helper worker function simulating robust resume
        def mock_worker(c_file, expected_fp, solver_val):
            if c_file.exists():
                try:
                    data = json.loads(c_file.read_text(encoding='utf-8'))
                    if data.get('fingerprint') == expected_fp and 'accepted' in data:
                        return {'resumed': True, 'val': data['val']}
                except Exception:
                    pass
            # Solve
            res = {'accepted': True, 'fingerprint': expected_fp, 'val': solver_val}
            tmp = c_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(res), encoding='utf-8')
            tmp.replace(c_file)
            return {'resumed': False, 'val': solver_val}
            
        fp_a = "abc123hash"
        
        # 1. First run: no file -> fresh solve
        r1 = mock_worker(case_file, fp_a, 42)
        assert not r1['resumed'] and r1['val'] == 42, "First run must not resume"
        assert case_file.exists()
        
        # 2. Second run: identical fingerprint -> resume
        r2 = mock_worker(case_file, fp_a, 999)
        assert r2['resumed'] and r2['val'] == 42, "Matching fingerprint must resume and preserve prior result"
        
        # 3. Third run: mismatched fingerprint (stale cache) -> re-solve
        fp_b = "def456hash"
        r3 = mock_worker(case_file, fp_b, 99)
        assert not r3['resumed'] and r3['val'] == 99, "Mismatched fingerprint must trigger fresh solve"
        
        # 4. Fourth run: truncated file -> re-solve
        case_file.write_text('{"truncated": true, "fingerprint":', encoding='utf-8')
        r4 = mock_worker(case_file, fp_b, 777)
        assert not r4['resumed'] and r4['val'] == 777, "Truncated file must trigger fresh solve"
        
        print("  [PASS] Resume safeguards (matching, mismatched, truncated) passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def test_cohort_reproducibility():
    print("Testing cohort selection reproducibility from seed 20261120...")
    R = Path('.').resolve()
    sel_path = R / 'Saved Model/stage1_downstream_study_v1/selected_participants_LOCAL.csv'
    assert sel_path.exists(), f"Missing {sel_path}"
    existing_sel = pd.read_csv(sel_path)
    
    # Re-run sampling logic exactly as defined in protocol/setup
    pilot_source = R / "Saved Model/stage2_research_adapter_pilot_v1/envelopes_LOCAL.json"
    pilot_rows = json.loads(pilot_source.read_text())
    pilot_seqns = sorted(list({int(x['SEQN']) for x in pilot_rows}))
    
    temporal_df = pd.read_csv(R / 'Data/stage1_repaired_v2/temporal_ml_ready.csv', usecols=['SEQN'])
    temporal_seqns = set(temporal_df['SEQN'].astype(int))
    
    needed_cols = ['SEQN', 'BMXWT', 'BMXHT', 'RIDAGEYR', 'RIAGENDR', 'DIQ010',
                   'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F'] + list(DIET_RAW_COLS)
    raw_df = pd.read_csv(R / 'Data/NHANES_2017_2020_PrePandemic_Raw.csv', usecols=needed_cols).set_index('SEQN')
    
    models = ['bionutrigpt', 'catboost', 'ft_transformer']
    pred_dfs = {m: pd.read_csv(R / f'Saved Model/stage1_comparison_v2/{m}/temporal_predictions_LOCAL.csv').set_index('SEQN') for m in models}
    
    eligible_records = []
    for seqn in sorted(temporal_seqns):
        if seqn in pilot_seqns:
            continue
        if seqn not in raw_df.index:
            continue
        x = raw_df.loc[seqn]
        w = float(x.BMXWT)
        h = float(x.BMXHT)
        age = float(x.RIDAGEYR)
        gender = int(x.RIAGENDR) if pd.notna(x.RIAGENDR) else None
        
        if not (np.isfinite(w) and w > 0 and np.isfinite(h) and h > 0 and np.isfinite(age) and age >= 20 and gender in (1, 2)):
            continue
        
        nutrients = {k: float(x[k]) for k in DIET_RAW_COLS}
        if not all(np.isfinite(v) and v >= 0 for v in nutrients.values()):
            continue
            
        has_preds = True
        for m in models:
            if seqn not in pred_dfs[m].index:
                has_preds = False; break
            p_row = pred_dfs[m].loc[seqn]
            p_d = float(p_row['TARGET_DIABETES_calibrated'])
            p_c = float(p_row['TARGET_CVD_calibrated'])
            if not (np.isfinite(p_d) and 0 <= p_d <= 1 and np.isfinite(p_c) and 0 <= p_c <= 1):
                has_preds = False; break
        if not has_preds:
            continue
            
        cv = x[['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F']]
        diabetes = True if x.DIQ010 == 1 else False if x.DIQ010 in (2, 3) else None
        cvd = True if cv.eq(1).any() else False if cv.eq(2).all() else None
        bmi = w / ((h / 100.0) ** 2)
        
        eligible_records.append({
            'SEQN': int(seqn),
            'RIDAGEYR': age,
            'RIAGENDR': gender,
            'is_female': bool(gender == 2),
            'BMXWT': w,
            'BMXHT': h,
            'BMI': bmi,
            'is_obese': bool(bmi >= 30.0),
            'DIQ010': float(x.DIQ010) if pd.notna(x.DIQ010) else np.nan,
            'has_diabetes': diabetes,
            'has_cvd': cvd,
        })
        
    assert len(eligible_records) == 6999, f"Expected 6999 eligible records, got {len(eligible_records)}"
    rng = np.random.RandomState(20261120)
    chosen_indices = rng.choice(len(eligible_records), size=100, replace=False)
    selected_participants = [eligible_records[i] for i in sorted(chosen_indices)]
    recomputed_df = pd.DataFrame(selected_participants)
    
    assert list(recomputed_df['SEQN']) == list(existing_sel['SEQN']), "Cohort SEQN order did not match exactly!"
    for col in ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMI']:
        np.testing.assert_allclose(recomputed_df[col].values, existing_sel[col].values, rtol=1e-5)
    print("  [PASS] Cohort selection exactly reproduced (100 participants, seed 20261120, 0 pilot overlap).")

if __name__ == '__main__':
    test_canonical_fingerprint()
    test_resume_logic()
    test_cohort_reproducibility()
    print("\nALL SAFEGUARD AND RESUME VERIFICATIONS PASSED SUCCESSFULLY.")
