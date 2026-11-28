"""Check frozen release parity and real Stage 1/2 application integration."""
from pathlib import Path
from dataclasses import fields
import json, os, tempfile, subprocess, sys
import numpy as np
import pandas as pd
import torch
import joblib
import stage1_deployment as dep
import stage2_counterfactual_engine as s2
import run_stage1_comparison_v2 as study
from prepare_stage1_repair_v2 import load

def main():
    torch.set_num_threads(os.cpu_count() or 1)
    manifest=dep.verify_bundle()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    artifacts=s2._load_preprocessing_artifacts(dep.PREPROCESSOR)
    model=s2._load_stage1_model(dep.CHECKPOINT,artifacts,device)
    report=json.loads((study.DATA/'repair_report.json').read_text())
    for name in ['temporal_ml_ready.csv','temporal_clean.csv']:
        assert study.sha(study.DATA/name)==report['output_hashes'][name]
    df=pd.read_csv(study.DATA/'temporal_ml_ready.csv')
    x=df[manifest['feature_order']].to_numpy(dtype=np.float32)
    expected=pd.read_csv(study.RESULTS/'bionutrigpt/temporal_predictions_LOCAL.csv')
    assert np.array_equal(df.SEQN,expected.SEQN)
    actual=study.predict('bionutrigpt',model,x,device)
    target=expected[[t+'_calibrated' for t in study.TARGETS]].to_numpy()
    np.testing.assert_allclose(actual,target,atol=1e-6,rtol=0)
    print('PASS all temporal calibrated probabilities',len(x),flush=True)
    pp=load('deployment_preprocessor','Data/preprocess_for_training.py')
    state=pp.PreprocessorState(**{f.name:artifacts[f.name] for f in fields(pp.PreprocessorState)})
    clean=pd.read_csv(study.DATA/'temporal_clean.csv')
    inputs=[c for c in clean if c not in set(pp.META_COLS+pp.TARGET_COLS) and c not in pp.MODEL_EXCLUDED_RAW_FEATURES]
    flags=pp.compute_missingness_indicators(clean,inputs)
    complete=flags.sum(axis=1)==0
    selected=clean[complete].iloc[:128]
    cont=selected[state.base_continuous_cols].astype(float)
    raw=pd.concat([cont,pp.add_clinical_features(cont),pp.encode_binary_and_nominal(selected,state.binary_modes,state.nominal_modes),flags.loc[selected.index]],axis=1)[manifest['feature_order']]
    transformed=[]
    for row in raw.to_numpy():
        canonical,_=s2.canonicalize_dietary_ratios(row)
        transformed.append(s2._normalize_numpy(canonical,artifacts)[0])
    transformed=np.array(transformed)
    np.testing.assert_allclose(transformed,x[selected.index],atol=1.1e-6,rtol=0)
    raw_preds=study.predict('bionutrigpt',model,transformed,device)
    np.testing.assert_allclose(raw_preds,target[selected.index],atol=1e-6,rtol=0)
    print('PASS raw-unit preprocessing/prediction parity',len(raw),flush=True)
    # Reject ambiguous/missing deployment inputs and mixed artifact releases.
    from backend import pipeline_service as service
    rejected=0
    cases=[dict(named_features={'RIDAGEYR':42}),dict(features=[0.]*88),dict(features=service.SAMPLE_PATIENT_VALUES,schema_version='unknown')]
    for kwargs in cases:
        try:service.run_screening(**kwargs)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid request accepted')
    for flag in [float('nan'),1.]:
        v=raw.iloc[0].to_numpy().copy();v[-1]=flag
        try:s2.canonicalize_dietary_ratios(v)
        except ValueError:rejected+=1
        else:raise AssertionError('Missing input accepted')
    try:dep.verify_bundle(preprocessor=dep.ROOT/'Data/preprocessing_artifacts.joblib')
    except ValueError:rejected+=1
    else:raise AssertionError('Mixed release accepted')
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['BIONUTRIGPT_STAGE2_JSON']=str(Path(tmp)/'stage2.json')
        result=service.run_screening(features=service.SAMPLE_PATIENT_VALUES,metadata_overrides={'weight_kg':70.,'height_cm':170.,'age_years':42.,'is_female':False,'activity_level':'moderate'})
    dossier=result['dossier'];b=dossier['stage2_counterfactual_prescription']['patient_baseline']['probabilities']
    np.testing.assert_allclose(list(result['probabilities'].values()),[b[k] for k in result['probabilities']],atol=1e-7,rtol=0)
    assert dossier['stage1_diagnostic_assessment']['model_binding']==dossier['stage2_counterfactual_prescription']['model_binding']
    assert all(v['threshold']==.5 for v in result['classifications'].values())
    print('PASS real backend Stage 1/2 calibrated agreement',flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['BIONUTRIGPT_STAGE2_JSON']=str(Path(tmp)/'stage2.json')
        supported=service.run_screening(features=service.SAMPLE_PATIENT_VALUES,metadata_overrides={'weight_kg':65.,'height_cm':165.,'age_years':42.,'is_female':False,'activity_level':'light'})
        assert supported['certified_for_stage3']
        meal=service.run_optimization(supported['dossier'])
        assert meal['accepted']
        sys.path.insert(0,str(dep.ROOT/'backend'))
        from fastapi.testclient import TestClient
        import main as api
        client=TestClient(api.app)
        bad=client.post('/api/screen',json={'named_features':{'RIDAGEYR':42}})
        assert bad.status_code==422
        response=client.post('/api/screen',json={'features':service.SAMPLE_PATIENT_VALUES,'weight_kg':65.,'height_cm':165.,'age_years':42.,'is_female':False,'activity_level':'light'})
        assert response.status_code==200,response.text
        assert response.json()['dossier']['stage1_diagnostic_assessment']['model_metadata']['release']=='repaired_v1'
        np.testing.assert_allclose(list(response.json()['probabilities'].values()),list(supported['probabilities'].values()),atol=1e-7,rtol=0)
    env=dict(os.environ,BIONUTRIGPT_STAGE1_RELEASE='legacy')
    rollback=subprocess.run([sys.executable,'-c',"import stage1_deployment as d; import infer_bionutrigpt_pipeline_Stage_1_and_2 as p; import numpy as np; from backend.pipeline_service import SAMPLE_PATIENT_VALUES; a,_=p.run_stage1_diagnostics(np.array(SAMPLE_PATIENT_VALUES)); assert d.RELEASE=='legacy'; assert p.CLINICAL_THRESHOLDS['diabetes']==.617; assert a['model_binding']['probability_transform']=='raw_sigmoid_no_calibration'; print('ROLLBACK PASS')"],env=env,capture_output=True,text=True,check=True)
    assert 'ROLLBACK PASS' in rollback.stdout
    output={'passed':True,'release':'repaired_v1','device':str(device),'temporal_rows':len(x),'max_calibrated_probability_error':float(abs(actual-target).max()),'raw_input_rows':len(raw),'max_raw_preprocess_error':float(abs(transformed-x[selected.index]).max()),'rejected_bad_inputs':rejected,'stage12_probability_agreement':True,'same_model_binding':True,'stage2_certified_for_stage3':result['certified_for_stage3'],'no_training':True,'limits':'Research threshold 0.5; no clinical or Indian validation. Raw-input parity covers complete observed records only.'}
    output.update(second_input_stage3_accepted=True,api_screening_200=True,api_incomplete_input_422=True,legacy_rollback_inference=True,source_hashes={str(p.relative_to(dep.ROOT)):dep.sha(p) for p in [dep.ROOT/'stage1_deployment.py',dep.ROOT/'stage2_counterfactual_engine.py',dep.ROOT/'backend/pipeline_service.py',Path(__file__)]})
    out=dep.ROOT/'Saved Model/stage1_deployment_verification_v1';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    (out/'sample_dossier.json').write_text(json.dumps(dossier,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()

