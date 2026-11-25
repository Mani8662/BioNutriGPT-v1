"""Frozen-model robustness stress study; no fitting or temporal selection."""
import copy,gc,hashlib,json,os,time
from dataclasses import fields
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import run_stage1_comparison_v2 as comp
from prepare_stage1_repair_v2 import load
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Saved Model/stage1_robustness_v2'

def main():
 OUT.mkdir(exist_ok=False)
 def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
 pp=load('robust_preprocessor','Data/preprocess_for_training.py')
 data,features=comp.load_data()
 artifact=joblib.load(comp.DATA/'preprocessing_artifacts.joblib')
 state=pp.PreprocessorState(**{f.name:artifact[f.name] for f in fields(pp.PreprocessorState)})
 raw=pd.read_csv(comp.DATA/'temporal_clean.csv')
 assert np.array_equal(raw.SEQN.to_numpy(),data['temporal']['ids'])
 diet=[c for c in state.base_continuous_cols if c.startswith('DR1T')]
 assert len(diet)==39
 scenarios=['baseline','missing_diet','missing_lipids','missing_both','missing_diet_20pct_seed17','missing_diet_20pct_seed29','diet_noise_10pct_seed17','lipid_noise_5pct_seed17','joint_noise_seed17']
 protocol={'version':'stage1-robustness-v2','status':'Prospectively fixed engineering stress scenarios after historical temporal results were inspected; not preregistration.','models':list(json.loads(comp.PROTOCOL_PATH.read_text())['models']), 'scenarios':scenarios,'n':len(raw),'missing_policy':'Mark source nutrients/lipids NaN; existing frozen training medians and missingness flags; regenerate engineered ratios using frozen transform. Missing lipid group includes stored NON_HDL_CHOL.','perturbation_policy':'Multiplicative independent uniform errors: dietary [-10%,+10%], HDL and total cholesterol [-5%,+5%]; recompute non-HDL=max(total-HDL,0) after lipid noise; regenerate ratios. Chosen engineering magnitudes, not validated device/recall error distributions.','partial_missing':'Independent 20% nutrient cells per row at fixed seeds 17 and 29; labels/participants unchanged.','scoring':'Raw and frozen-calibrated AUROC, AP, Brier, diagnostic calibration slope/intercept; absolute probability change and decisions at fixed descriptive 0.5. No clinical cutoff or recalibration. Descriptive point estimates; no robustness significance claim.','limits':'Selected NHANES cohort, unweighted; no Indian external validation; simulated missingness is not observed MNAR; no retraining or best-case selection.'}
 write('protocol.json',protocol)
 frozen_paths=[comp.DATA/'preprocessing_artifacts.joblib',comp.DATA/'temporal_clean.csv',comp.PROTOCOL_PATH,ROOT/'Data/preprocess_for_training.py',Path(__file__)]
 binding={str(p.relative_to(ROOT)):comp.sha(p) for p in frozen_paths}
 matrices={};stress_checks={}
 for label in scenarios:
  frame=raw.copy(deep=True);rng=np.random.default_rng(29 if '29' in label else 17)
  if label in ['missing_diet','missing_both']:frame[diet]=np.nan
  if label in ['missing_lipids','missing_both']:frame[['LBDHDD','LBXTC','NON_HDL_CHOL']]=np.nan
  if label.startswith('missing_diet_20pct'):frame[diet]=frame[diet].mask(rng.random((len(frame),len(diet)))<.2)
  if label.startswith('diet_noise') or label=='joint_noise_seed17':frame[diet]=frame[diet].to_numpy()*(1+rng.uniform(-.1,.1,(len(frame),len(diet))))
  if label.startswith('lipid_noise') or label=='joint_noise_seed17':
   frame[['LBDHDD','LBXTC']]=frame[['LBDHDD','LBXTC']].to_numpy()*(1+rng.uniform(-.05,.05,(len(frame),2)))
   frame['NON_HDL_CHOL']=(frame.LBXTC-frame.LBDHDD).clip(lower=0)
  transformed,stats=pp.transform_dataset(frame,state)
  x=transformed[features].to_numpy(dtype=np.float32).copy()
  assert np.isfinite(x).all()
  assert np.array_equal(transformed[comp.TARGETS].to_numpy(),data['temporal']['y'])
  if label=='baseline':assert np.array_equal(x,data['temporal']['x']),'Baseline preprocessing mismatch'
  matrices[label]=x;stress_checks[label]={'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'stats':stats}
 write('scenario_checks.json',stress_checks)
 configs=json.loads(comp.PROTOCOL_PATH.read_text())['models']
 allmetrics=[];timings=[]
 torch.set_num_threads(os.cpu_count() or 1)
 for name in protocol['models']:
  folder=comp.RESULTS/name
  comp.verify_completion(folder,'completed.json')
  selected=json.loads((folder/'selection.json').read_text())['selected_trial'];trial=folder/f'trial_{selected:02d}'
  comp.verify_completion(trial,'completed.json')
  modelpath=trial/('model.pt' if name in comp.NEURAL else 'model.joblib')
  for p in [modelpath,folder/'calibrators.joblib',folder/'selection.json']:binding[str(p.relative_to(ROOT))]=comp.sha(p)
  write('binding.json',binding)
  device='cuda' if name in comp.NEURAL and torch.cuda.is_available() else 'cpu'
  model=comp.load_trial(name,configs[name][selected],trial,device)
  calibrators=joblib.load(folder/'calibrators.joblib')
  saved=pd.read_csv(folder/'temporal_predictions_LOCAL.csv')
  assert np.array_equal(saved.SEQN.to_numpy(),data['temporal']['ids'])
  predictions={};baselines={}
  for label,x in matrices.items():
   start=time.perf_counter()
   with threadpool_limits(limits=os.cpu_count() or 1):pred=comp.predict(name,model,x,device)
   cal=comp.apply_calibrators(calibrators,pred)
   predictions[label+'_raw']=pred;predictions[label+'_calibrated']=cal
   if label=='baseline':
    assert np.allclose(pred,saved[[t+'_raw' for t in comp.TARGETS]].to_numpy(),rtol=0,atol=1e-6),'Baseline model reload mismatch'
    assert np.allclose(cal,saved[[t+'_calibrated' for t in comp.TARGETS]].to_numpy(),rtol=0,atol=1e-6),'Calibration reload mismatch'
   for variant,probs in [('raw',pred),('calibrated',cal)]:
    metrics,_=comp.metrics(data['temporal']['y'],probs)
    if label=='baseline':baselines[variant]=metrics
    ref=predictions['baseline_'+variant]
    for i,row in enumerate(metrics):
     row.update(model=name,scenario=label,variant=variant,delta_auroc=row['auroc']-baselines[variant][i]['auroc'],delta_brier=row['brier']-baselines[variant][i]['brier'],mean_absolute_probability_change=float(abs(probs[:,i]-ref[:,i]).mean()),p95_absolute_probability_change=float(np.quantile(abs(probs[:,i]-ref[:,i]),.95)),decision_flip_fraction=float(np.mean((probs[:,i]>=.5)!=(ref[:,i]>=.5))))
     allmetrics.append(row)
   timings.append(dict(model=name,scenario=label,device=device,seconds=time.perf_counter()-start))
   pd.DataFrame(allmetrics).to_csv(OUT/'metrics.csv',index=False)
   np.savez_compressed(OUT/(name+'_'+label+'_predictions_LOCAL.npz'),ids=data['temporal']['ids'],labels=data['temporal']['y'],raw=pred,calibrated=cal)
   print(name,label,'saved',flush=True)
  del model;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
 for path,digest in binding.items():assert comp.sha(ROOT/path)==digest,'Frozen artifact changed'
 write('timings.json',timings)
 write('completed.json',{'models':len(configs),'scenarios':len(scenarios),'metric_rows':len(allmetrics),'n':len(raw),'frozen_artifacts_unchanged':True,'baseline_reproduction_passed':True,'metrics_sha256':comp.sha(OUT/'metrics.csv'),'environment':comp.environment()})
if __name__=='__main__':main()
