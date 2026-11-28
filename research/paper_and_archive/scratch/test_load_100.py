import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json, pandas as pd
from prepare_stage1_repair_v2 import load
from dataclasses import fields
import numpy as np
import stage2_counterfactual_engine as s2
import stage1_deployment as dep

pp = load('test_preprocessing', 'Data/preprocess_for_training.py')
artifacts = s2._load_preprocessing_artifacts(dep.PREPROCESSOR)
state = pp.PreprocessorState(**{f.name: artifacts[f.name] for f in fields(pp.PreprocessorState)})

clean = pd.read_csv('Data/stage1_repaired_v2/temporal_clean.csv')
rawsource = 'Data/NHANES_2017_2020_PrePandemic_Raw.csv'
measured = pd.read_csv(rawsource, usecols=['SEQN','BMXWT','BMXHT','RIDAGEYR','RIAGENDR']).set_index('SEQN')

ids = json.loads((ROOT / 'Saved Model/stage2_comparison_design_v1/selected_participants_LOCAL.json').read_text())
chosen = clean[clean.SEQN.isin(ids)].sort_values('SEQN')
assert len(chosen) == 100

inputs = [c for c in clean if c not in set(pp.META_COLS+pp.TARGET_COLS) and c not in pp.MODEL_EXCLUDED_RAW_FEATURES]
flags = pp.compute_missingness_indicators(clean, inputs)

cont = chosen[state.base_continuous_cols].astype(float)
raw_df = pd.concat([cont, pp.add_clinical_features(cont), pp.encode_binary_and_nominal(chosen, state.binary_modes, state.nominal_modes), flags.loc[chosen.index]], axis=1)[list(s2.FEATURE_COLS)]

cases = []
for i, row in chosen.iterrows():
    x = measured.loc[row.SEQN]
    cases.append({
        'case': len(cases),
        'SEQN': int(row.SEQN),
        'vector': raw_df.loc[i].tolist(),
        'metadata': {
            'weight_kg': float(x.BMXWT),
            'height_cm': float(x.BMXHT),
            'age_years': float(x.RIDAGEYR),
            'is_female': bool(x.RIAGENDR == 2),
            'activity_level': 'sedentary',
            'renal_disease': None,
            'pregnant': None,
            'lactating': None,
        }
    })
print(f"Successfully built {len(cases)} cases. First case SEQN: {cases[0]['SEQN']}, last: {cases[-1]['SEQN']}")
print("Vector length of case 0:", len(cases[0]['vector']))
print("Number of FEATURE_COLS:", len(s2.FEATURE_COLS))
print("FEATURE_COLS == raw_df.columns:", list(s2.FEATURE_COLS) == list(raw_df.columns))
assert len(cases[0]['vector']) == len(s2.FEATURE_COLS)
assert all(np.isfinite(c['vector']).all() for c in cases)
print("All 100 vectors have length 88 and are strictly finite!")
