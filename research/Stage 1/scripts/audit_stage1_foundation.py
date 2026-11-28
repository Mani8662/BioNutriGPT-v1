"""Read-only audit of existing Stage 1 evidence; never trains or overwrites inputs."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Research Paper' / 'Stage1_Foundation_Audit'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b):
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return {'same_schema_shape': False, 'left_shape': list(a.shape), 'right_shape': list(b.shape)}
    bad = {}
    for c in a:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            equal = np.isclose(a[c], b[c], atol=1e-6, rtol=1e-8, equal_nan=True)
        else:
            equal = (a[c].fillna('<NA>').to_numpy() == b[c].fillna('<NA>').to_numpy())
        if not np.all(equal):
            bad[c] = int((~equal).sum())
    return {'same_schema_shape': True, 'mismatched_cells_by_column': bad, 'matches_within_tolerance': not bad}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cleanmod = load('audit_cleaner', 'Data/preprocess_raw_genuinely.py')
    pp = load('audit_preprocessor', 'Data/preprocess_for_training.py')
    targets = cleanmod.TARGET_COLS
    tree = ast.parse((ROOT / 'train_bionutrigpt_stage1.py').read_text(encoding='utf-8-sig'))
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ['BIO_COLS','DIET_COLS','LIFE_COLS','DEMO_COLS']:
                    constants[t.id] = ast.literal_eval(n.value)
    features = sum([constants[x] for x in ['BIO_COLS','DIET_COLS','LIFE_COLS','DEMO_COLS']], [])
    sources = ['Data/BioNutriGPT_merged_raw_2005_2020.csv', 'Data/preprocessing_artifacts.joblib',
               'Data/preprocess_raw_genuinely.py', 'Data/preprocess_for_training.py',
               'Data/download_nhanes_2017_2020.py', 'Data/merge_raw_datasets.py',
               'train_bionutrigpt_stage1.py', 'evaluate_tripod_and_baselines.py',
               'infer_bionutrigpt_stage1.py', 'backend/pipeline_service.py',
               'stage2_counterfactual_engine.py', 'Saved Model/bionutrigpt_stage1_best.pt']
    frames = {}
    for partition, suffix in [('development','train_2007_2016'),('temporal','test_2017_2020')]:
        for kind in ['clean','ml_ready']:
            name = f'Data/BioNutriGPT_{suffix}_{kind}.csv'
            sources.append(name)
            frames[partition,kind] = pd.read_csv(ROOT/name)
    before = {p: sha(ROOT/p) for p in sources}
    raw = pd.read_csv(ROOT/sources[0], low_memory=False)
    rebuilt, log = cleanmod.clean_dataset(raw)
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Audit only; no model training or production mutations. Reconstructed splits are conditional on current code/data, not checkpoint-attested original memberships.',
              'source_hashes': before, 'raw_shape': list(raw.shape), 'flow': log.steps,
              'raw_duplicate_ids': int(raw.SEQN.duplicated().sum()),
              'upstream_development_source_available': (ROOT.parent/'P/BioNutriGPT_final_dataset_2005_2018.csv').exists(),
              'clean_reproduction': {}, 'preprocessing_reproduction': {}}
    pd.DataFrame(log.steps).to_csv(OUT/'cohort_flow.csv', index=False)
    art = joblib.load(ROOT/'Data/preprocessing_artifacts.joblib')
    state = pp.PreprocessorState(**{k:art[k] for k in pp.PreprocessorState.__dataclass_fields__})
    refit = pp.fit_preprocessor(frames['development','clean'])
    report['preprocessor_fit'] = {'saved_scaler_n_samples_seen': int(art['scaler'].n_samples_seen_),
        'development_n': len(frames['development','clean']),
        'refit_scaler_mean_max_difference': float(np.max(np.abs(refit.scaler.mean_-state.scaler.mean_))),
        'refit_scaler_scale_max_difference': float(np.max(np.abs(refit.scaler.scale_-state.scaler.scale_))),
        'refit_medians_match': refit.continuous_medians == state.continuous_medians,
        'refit_modes_match': refit.binary_modes == state.binary_modes and refit.nominal_modes == state.nominal_modes,
        'refit_log_columns_match': refit.log1p_cols == state.log1p_cols,
        'refit_clip_bounds_match': refit.clip_bounds == state.clip_bounds}
    miss = []
    for partition, cycles in [('development',cleanmod.TRAIN_CYCLES),('temporal',cleanmod.TEST_CYCLES)]:
        current = frames[partition,'clean']
        report['clean_reproduction'][partition] = compare(rebuilt[rebuilt.NHANES_CYCLE.isin(cycles)].reset_index(drop=True),current)
        transformed,_ = pp.transform_dataset(current,state)
        report['preprocessing_reproduction'][partition] = compare(transformed,frames[partition,'ml_ready'])
        for c in current:
            miss.append({'partition':partition,'field':c,'n':len(current),'missing':int(current[c].isna().sum())})
    pd.DataFrame(miss).to_csv(OUT/'clean_input_missingness.csv',index=False)
    dev=frames['development','ml_ready']; test=frames['temporal','ml_ready']
    tr,va=train_test_split(np.arange(len(dev)),test_size=3554,random_state=42,stratify=dev.TARGET_DIABETES)
    _,ca=train_test_split(np.arange(len(dev)),test_size=.15,random_state=42)
    parts={'development':dev,'reconstructed_training':dev.iloc[tr], 'reconstructed_validation':dev.iloc[va], 'temporal':test}
    rows=[]
    for name,df in parts.items():
        for target in targets:
            rows.append({'partition':name,'outcome':target,'n':len(df),'positive':int((df[target]==1).sum()),'negative':int((df[target]==0).sum()),'missing':int(df[target].isna().sum())})
    pd.DataFrame(rows).to_csv(OUT/'partition_outcome_counts.csv',index=False)
    split=dev[['SEQN','NHANES_CYCLE']].copy()
    split['partition']='reconstructed_training';split.loc[va,'partition']='reconstructed_validation'
    split['legacy_calibration_member']=False;split.loc[ca,'legacy_calibration_member']=True
    split['source_row']=np.arange(len(split))
    split=pd.concat([split,test[['SEQN','NHANES_CYCLE']].assign(partition='temporal',legacy_calibration_member=False,source_row=np.arange(len(test)))],ignore_index=True)
    split.to_csv(OUT/'reconstructed_split_membership_LOCAL.csv',index=False)
    report['split_audit']={'training_n':len(tr),'validation_n':len(va),'legacy_calibration_n':len(ca),
        'legacy_calibration_overlaps_training':len(set(ca)&set(tr)),
        'legacy_calibration_overlaps_validation':len(set(ca)&set(va)),
        'development_temporal_id_overlap':len(set(dev.SEQN)&set(test.SEQN)),
        'development_duplicate_ids':int(dev.SEQN.duplicated().sum()),'temporal_duplicate_ids':int(test.SEQN.duplicated().sum()),
        'split_manifest_sha256':sha(OUT/'reconstructed_split_membership_LOCAL.csv'),
        'checkpoint_attestation':'Training saves weights only; original membership cannot be independently attested from checkpoint.'}
    report['feature_contract']={'count':len(features),'unique_count':len(set(features)),
        'missing_features':sorted(set(features)-set(dev.columns)),
        'direct_outcome_defining_predictors':[x for x in features if x in ['LBXGH','LBXGLU','BMXBMI','BMXWT','BMXHT','DIQ010'] or x.startswith('MCQ')],
        'artifact_order_equals_model_order':state.feature_cols==features,
        'artifact_feature_set_equals_model_set':set(state.feature_cols)==set(features),
        'note':'Artifact order differs legitimately if columns are selected by name; positional conversion must use model order.'}
    labelrows=[]
    disagreements=[]
    for name,df in parts.items():
        cols=['SEQN','NHANES_CYCLE','LBXGH','LBXGLU','DIQ010','BMXBMI','MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F']
        src=raw[cols].set_index('SEQN').loc[df.SEQN].reset_index()
        h=src.LBXGH; g=src.LBXGLU; d=src.DIQ010
        cv=src[['MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F']]
        positive=(h>=6.5)|(g>=126)|(d==1)
        observed=h.notna()|g.notna()|d.isin([1,2])
        reconstructed={'TARGET_DIABETES':pd.Series(np.where(observed,positive.astype(float),np.nan)),
            'TARGET_OBESITY':pd.Series(np.where(src.BMXBMI.notna(),(src.BMXBMI>=30).astype(float),np.nan)),
            'TARGET_CVD':pd.Series(np.where(cv.isin([1,2]).any(axis=1),(cv==1).any(axis=1).astype(float),np.nan))}
        for target,rec in reconstructed.items():
            saved=df[target].to_numpy()
            if name in ['development','temporal']:
                mismatch=~np.isclose(saved,rec,equal_nan=True)
                for cycle in src.NHANES_CYCLE.unique():
                    cm=src.NHANES_CYCLE.eq(cycle).to_numpy()
                    disagreements.append({'partition':name,'cycle':cycle,'outcome':target,
                        'saved_negative_reconstructed_positive':int((cm&(saved==0)&(rec.to_numpy()==1)).sum()),
                        'saved_positive_reconstructed_negative':int((cm&(saved==1)&(rec.to_numpy()==0)).sum()),
                        'mismatch_total':int((cm&mismatch).sum())})
            labelrows.append({'partition':name,'outcome':target,'n':len(df),'reconstructed_mismatch':int((~np.isclose(saved,rec,equal_nan=True)).sum()),
                'negative_incomplete_components':int(((saved==0)&(~(h.notna()&g.notna()&d.isin([1,2])) if target=='TARGET_DIABETES' else ~cv.isin([1,2]).all(axis=1) if target=='TARGET_CVD' else src.BMXBMI.isna())).sum()),
                'negative_missing_hba1c_or_valid_diagnosis':int(((saved==0)&~(h.notna()&d.isin([1,2]))).sum()) if target=='TARGET_DIABETES' else None})
    pd.DataFrame(labelrows).to_csv(OUT/'label_ascertainment_sensitivity.csv',index=False)
    pd.DataFrame(disagreements).to_csv(OUT/'label_disagreement_by_cycle.csv',index=False)
    src=raw[raw.SEQN.isin(dev.SEQN)]
    report['development_definition_diagnostics']={
        'diabetes_matches_diagnosis_only':bool(np.array_equal(src.TARGET_DIABETES.to_numpy(),(src.DIQ010==1).astype(float).to_numpy())),
        'cvd_matches_composite_excluding_heart_failure':bool(np.array_equal(src.TARGET_CVD.to_numpy(),(src[['MCQ160C','MCQ160D','MCQ160E','MCQ160F']]==1).any(axis=1).astype(float).to_numpy())),
        'interpretation':'Empirical equivalence on selected rows; upstream original construction code has not been recovered.'}
    report['labels']=labelrows;report['partition_counts']=rows
    cyc=[]
    selected=raw[raw.SEQN.isin(set(dev.SEQN)|set(test.SEQN))]
    for cycle,df in selected.groupby('NHANES_CYCLE'):
        for c in ['LBXGH','LBXGLU','DIQ010','BMXBMI','MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F']:
            cyc.append({'cycle':cycle,'field':c,'n':len(df),'missing':int(df[c].isna().sum()),'refused_or_unknown_7_9':int(df[c].isin([7,9]).sum()) if c.startswith(('DIQ','MCQ')) else None})
    pd.DataFrame(cyc).to_csv(OUT/'outcome_component_missingness_by_cycle.csv',index=False)
    report['sentinel_replacements']=log.sentinels_replaced
    report['inputs_unchanged']=all(sha(ROOT/p)==h for p,h in before.items())
    assert report['inputs_unchanged']
    (OUT/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['clean_reproduction','preprocessing_reproduction','preprocessor_fit','split_audit','labels','feature_contract']},indent=2))


if __name__=='__main__':
    main()
