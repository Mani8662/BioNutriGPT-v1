"""Build isolated, versioned Stage 1 cohorts and training-only preprocessing.

Never loads or trains the neural model. Never overwrites existing outputs.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'stage1_repair_protocol_v2.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def model_features():
    tree = ast.parse((ROOT/'train_bionutrigpt_stage1.py').read_text(encoding='utf-8-sig'))
    names = ['BIO_COLS','DIET_COLS','LIFE_COLS','DEMO_COLS']
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    result = sum([values[n] for n in names], [])
    assert len(result) == len(set(result)) == 87
    return result


def harmonized_labels(raw, protocol):
    """Missing negatives are kept unknown; observed positives override unknown components."""
    get = lambda c: pd.to_numeric(raw[c], errors='coerce')
    h = get('LBXGH')
    h = h.where(h.between(*protocol['diabetes_primary']['valid_hba1c_range_percent']))
    d = get('DIQ010')
    positive = (d == 1) | (h >= 6.5)
    negative = d.isin([2,3]) & (h < 6.5)
    diabetes = pd.Series(np.select([positive,negative],[1.,0.],default=np.nan),index=raw.index)
    cv = raw[protocol['cvd_primary']['components']].apply(pd.to_numeric,errors='coerce')
    cvd = pd.Series(np.select([(cv==1).any(axis=1),(cv==2).all(axis=1)],[1.,0.],default=np.nan),index=raw.index)
    bmi = get('BMXBMI')
    obesity = (bmi>=30).astype(float).where(bmi.notna() & np.isfinite(bmi) & (bmi>0))
    labels = pd.DataFrame({'TARGET_DIABETES':diabetes,'TARGET_OBESITY':obesity,'TARGET_CVD':cvd})
    g = get('LBXGLU').where(lambda x: np.isfinite(x) & (x>0))
    gp = positive | (g>=126)
    # Sensitivity only: available glucose may create differential ascertainment.
    glucose_available = pd.Series(np.select([gp,negative],[1.,0.],default=np.nan),index=raw.index)
    glucose_complete = pd.Series(np.select([gp,negative & g.notna() & (g<126)],[1.,0.],default=np.nan),index=raw.index)
    strict_diq = diabetes.mask((d==3)&~positive)
    cv_observed = pd.Series(np.select([(cv==1).any(axis=1),cv.isin([1,2]).any(axis=1)],[1.,0.],default=np.nan),index=raw.index)
    sensitivity = pd.DataFrame({'diabetes_primary':diabetes,'diabetes_exclude_borderline_negatives':strict_diq,
        'diabetes_add_available_glucose':glucose_available,'diabetes_require_all_components_for_negative':glucose_complete,
        'cvd_primary':cvd,'cvd_observed_components_only':cv_observed})
    return labels,sensitivity


def make_splits(dev, targets, protocol):
    pattern = dev[targets].astype(int).astype(str).agg(''.join,axis=1)
    if pattern.value_counts().min()<7:
        raise ValueError('Joint-label strata too small; revise and version splitting policy explicitly.')
    seed = protocol['split']['seed']
    rest,cal = train_test_split(np.arange(len(dev)),test_size=.15,random_state=seed,stratify=pattern)
    train,val = train_test_split(rest,test_size=.15/.85,random_state=seed,stratify=pattern.iloc[rest])
    result = {'training':np.sort(train),'validation':np.sort(val),'calibration':np.sort(cal)}
    assert len(set(np.concatenate(list(result.values()))))==len(dev)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'Data/stage1_repaired_v2')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists; preserve it and choose a new explicit version: {out}')
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    cleaner=load('stage1_v2_cleaner','Data/preprocess_raw_genuinely.py')
    pp=load('stage1_v2_preprocessor','Data/preprocess_for_training.py')
    feature_order=model_features()
    input_files=[PROTOCOL,Path(__file__),ROOT/'Data/preprocess_raw_genuinely.py',ROOT/'Data/preprocess_for_training.py',
        ROOT/'Data/BioNutriGPT_merged_raw_2005_2020.csv',ROOT/'Data/preprocessing_artifacts.joblib',
        ROOT/'Saved Model/bionutrigpt_stage1_best.pt',ROOT/'train_bionutrigpt_stage1.py']
    input_files += [ROOT/'Data'/f'BioNutriGPT_{part}_{kind}.csv' for part in ['train_2007_2016','test_2017_2020'] for kind in ['clean','ml_ready']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in input_files}
    print('Loading available merged source and rebuilding outcome labels...',flush=True)
    raw=pd.read_csv(ROOT/'Data/BioNutriGPT_merged_raw_2005_2020.csv',low_memory=False)
    assert raw.SEQN.notna().all() and not raw.SEQN.duplicated().any(), 'Source IDs must be unique.'
    labels,sensitivity=harmonized_labels(raw,protocol)
    rebuilt=raw.copy()
    rebuilt[cleaner.TARGET_COLS]=labels
    eligible_age=rebuilt.RIDAGEYR.ge(protocol['minimum_age'])
    clean,audit=cleaner.clean_dataset(rebuilt.loc[eligible_age].copy())
    dev=clean[clean.NHANES_CYCLE.isin(protocol['development_cycles'])].reset_index(drop=True)
    temporal=clean[clean.NHANES_CYCLE.isin(protocol['temporal_cycles'])].reset_index(drop=True)
    assert len(dev) and len(temporal) and set(dev.SEQN).isdisjoint(temporal.SEQN)
    split=make_splits(dev,cleaner.TARGET_COLS,protocol)
    partitions={name:dev.iloc[idx].reset_index(drop=True) for name,idx in split.items()}
    partitions['temporal']=temporal
    print('Fitting preprocessing on training partition only...',flush=True)
    state=pp.fit_preprocessor(partitions['training'])
    assert set(state.feature_cols)==set(feature_order)
    artifact={f.name:getattr(state,f.name) for f in fields(pp.PreprocessorState)}
    artifact.update({'version':protocol['version'],'model_feature_order':feature_order,
        'meta_cols':pp.META_COLS,'target_cols':pp.TARGET_COLS,'continuous_cols':state.all_continuous_cols,
        'binary_and_onehot_cols':[c for c in state.feature_cols if c not in state.all_continuous_cols],
        'fit_participant_ids_sha256':hashlib.sha256(partitions['training'].SEQN.to_numpy(dtype='<i8').tobytes()).hexdigest(),
        'compatible_model_checkpoint':None,'warning':'Requires newly trained models. Do not pair with legacy checkpoint.'})
    out.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    joblib.dump(artifact,out/'preprocessing_artifacts.joblib')
    counts=[]; members=[]; stats={}; transformed={}
    for name,df in partitions.items():
        ml,st=pp.transform_dataset(df,state)
        ml=ml[pp.META_COLS+pp.TARGET_COLS+feature_order]
        assert np.isfinite(ml[feature_order].to_numpy(dtype=float)).all()
        df.to_csv(out/f'{name}_clean.csv',index=False)
        ml.to_csv(out/f'{name}_ml_ready.csv',index=False)
        transformed[name]=ml;stats[name]=st
        members.append(df[pp.META_COLS].assign(partition=name))
        for target in pp.TARGET_COLS:
            counts.append({'partition':name,'outcome':target,'n':len(df),'positive':int(df[target].sum()),'negative':int((df[target]==0).sum())})
    membership=pd.concat(members,ignore_index=True)
    assert not membership.SEQN.duplicated().any()
    membership.to_csv(out/'split_membership_LOCAL.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'partition_outcome_counts.csv',index=False)
    pd.DataFrame(audit.steps).to_csv(out/'cohort_flow.csv',index=False)
    # Retain survey-design variables separately, never as predictors. No weight choice is implied.
    survey_cols=['SEQN','NHANES_CYCLE','RIDAGEYR','RIAGENDR','RIDRETH1']+[c for c in raw if c.startswith(('WT','SDMV','PHAFST'))]
    raw.loc[raw.SEQN.isin(membership.SEQN),survey_cols].to_csv(out/'survey_metadata_LOCAL.csv',index=False)
    sensitivity_rows=[]
    for name,df in partitions.items():
        ss=sensitivity.loc[raw.SEQN.isin(df.SEQN)]
        for col in ss:
            base=ss.diabetes_primary if col.startswith('diabetes') else ss.cvd_primary
            sensitivity_rows.append({'partition':name,'definition':col,'n':len(ss),'positive':int((ss[col]==1).sum()),
                'negative':int((ss[col]==0).sum()),'unknown':int(ss[col].isna().sum()),
                'changed_known_label_vs_primary':int((ss[col].notna()&base.notna()&(ss[col]!=base)).sum())})
    pd.DataFrame(sensitivity_rows).to_csv(out/'label_sensitivity_counts.csv',index=False)
    # Changes on old selected participants, plus newly admitted / excluded participant counts.
    changes=[]
    for oldpart,new in [('train_2007_2016',dev),('test_2017_2020',temporal)]:
        old=pd.read_csv(ROOT/'Data'/f'BioNutriGPT_{oldpart}_clean.csv')
        shared=old[['SEQN']+pp.TARGET_COLS].merge(new[['SEQN']+pp.TARGET_COLS],on='SEQN',suffixes=('_old','_new'))
        changes.append({'old_partition':oldpart,'old_n':len(old),'new_n':len(new),'shared':len(shared),
            'newly_included':len(set(new.SEQN)-set(old.SEQN)),'excluded_old':len(set(old.SEQN)-set(new.SEQN)),
            'label_changes_shared':{t:int((shared[t+'_old']!=shared[t+'_new']).sum()) for t in pp.TARGET_COLS}})
    # Actual persisted artifact round-trip; applies training-fitted state to all reserved sets.
    reloaded=joblib.load(out/'preprocessing_artifacts.joblib')
    reloaded_state=pp.PreprocessorState(**{f.name:reloaded[f.name] for f in fields(pp.PreprocessorState)})
    for name,df in partitions.items():
        reproduced,_=pp.transform_dataset(df,reloaded_state)
        assert np.array_equal(reproduced[feature_order].to_numpy(),transformed[name][feature_order].to_numpy())
    before_fit=joblib.hash(state)
    altered=partitions['validation'].copy()
    altered['INDFMPIR']=99999.
    pp.transform_dataset(altered,state)
    assert joblib.hash(state)==before_fit, 'Reserved-set transformation mutated fitted state.'
    assert int(state.scaler.n_samples_seen_)==len(partitions['training'])
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in input_files},'Source mutation detected.'
    report={'version':protocol['version'],'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'Data repair built and checked; model training/calibration/evaluation not executed',
        'protocol_sha256':sha(PROTOCOL),'source_hashes':hashes,'environment':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__},
        'raw_n':len(raw),'age_20_plus_n_before_cycle_filter':int(eligible_age.sum()),'flow':audit.steps,
        'development_n':len(dev),'temporal_n':len(temporal),'counts':counts,'changes_from_legacy':changes,
        'preprocessing_stats':stats,'feature_order':feature_order,'target_order':pp.TARGET_COLS,
        'checks':{'all_pairwise_partitions_disjoint':True,'all_features_finite':True,'feature_count':87,
            'scaler_training_rows_seen':int(state.scaler.n_samples_seen_),'persisted_transform_reproduces_all_partitions':True,
            'reserved_transform_does_not_mutate_state':True,'legacy_inputs_unchanged':True},
        'limitations':protocol['limitations'],
        'output_hashes':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'repair_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'development_n':len(dev),'temporal_n':len(temporal),'partition_n':{k:len(v) for k,v in partitions.items()},'changes':changes,'checks':report['checks']},indent=2))


if __name__=='__main__':
    main()
