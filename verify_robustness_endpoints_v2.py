"""Independent weighted-pair AUROC bootstrap replay, without analysis imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
R=Path(__file__).resolve().parent;S=R/'Saved Model/stage1_robustness_uncertainty_v1';O=R/'Saved Model/stage1_robustness_endpoint_audit_v2';P=R/'Saved Model/stage1_robustness_v3'
J={'TARGET_DIABETES':0,'TARGET_OBESITY':1,'TARGET_CVD':2}
def work(task):
 model,scenario=task
 data=np.load(P/f'{model}_{scenario}_predictions_LOCAL.npz');counts=np.load(O/'counts.npy',mmap_mode='r');result={}
 for variant in ['raw','calibrated']:
  for outcome,j in J.items():
   y=data['labels'][:,j].astype(np.float64);p=data[variant][:,j].astype(float);order=np.argsort(p);starts=np.r_[0,np.flatnonzero(np.diff(p[order]))+1];a=[];b=[]
   for lo in range(0,10000,128):
    c=counts[lo:lo+128];ordered=c[:,order]
    pos=np.add.reduceat(ordered*y[order],starts,axis=1);neg=np.add.reduceat(ordered*(1-y[order]),starts,axis=1)
    den=pos.sum(1)*neg.sum(1);num=(pos*(np.cumsum(neg,axis=1)-neg/2)).sum(1)
    a.extend(np.divide(num,den,out=np.full(len(c),np.nan),where=den>0));b.extend(c@((y-p)**2)/len(y))
   result[variant+'__'+outcome+'__auroc']=np.array(a);result[variant+'__'+outcome+'__brier']=np.array(b)
 np.savez_compressed(O/f'{model}__{scenario}.npz',**result);return model,scenario
def main():
 O.mkdir(exist_ok=True);w=pd.read_csv(S/'within_model_deterioration.csv');d=pd.read_csv(S/'between_model_contrasts.csv');seed=json.loads((S/'bootstrap_seed_record.json').read_text());n=seed['n_participants']
 idx=np.random.default_rng(seed['seed']).integers(0,n,size=(10000,n),dtype=np.int32);assert hashlib.sha256(idx.tobytes()).hexdigest()==seed['resample_sha256']
 counts=np.lib.format.open_memmap(O/'counts.npy',mode='w+',dtype=np.int16,shape=(10000,n))
 for k in range(10000):counts[k]=np.bincount(idx[k],minlength=n)
 counts.flush();del counts,idx
 tasks=[(m,s) for m in w.model.unique() for s in ['baseline',*w.scenario.unique()]]
 ref=np.load(P/'bionutrigpt_baseline_predictions_LOCAL.npz');bindings={}
 for m,s in tasks:
  path=P/f'{m}_{s}_predictions_LOCAL.npz';z=np.load(path);np.testing.assert_array_equal(z['ids'],ref['ids']);np.testing.assert_array_equal(z['labels'],ref['labels']);assert len(np.unique(z['ids']))==n
  for v in ['raw','calibrated']:assert np.isfinite(z[v]).all() and ((z[v]>=0)&(z[v]<=1)).all()
  bindings[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 if '--verify-only' not in __import__('sys').argv:
  with ProcessPoolExecutor(max_workers=6) as pool:
   for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks])):print(i+1, f.result(),flush=True)
 cache={(m,s):np.load(O/f'{m}__{s}.npz') for m,s in tasks}
 def check(arr,low,high):
  assert np.isfinite(arr).sum()==10000
  np.testing.assert_allclose(np.percentile(arr,[2.5,97.5]),[low,high],atol=1e-10,rtol=0)
 def arr(m,s,row):return cache[m,s][row.variant+'__'+row.outcome+'__'+row.metric]
 for row in w.itertuples():
  a=arr(row.model,'baseline',row);b=arr(row.model,row.scenario,row);delta=a-b if row.metric=='auroc' else b-a
  for v,prefix in [(a,'baseline'),(b,'stressed'),(delta,'deterioration')]:check(v,getattr(row,prefix+'_ci_lower'),getattr(row,prefix+'_ci_upper'))
  points=[]
  for s in ['baseline',row.scenario]:
   z=np.load(P/f'{row.model}_{s}_predictions_LOCAL.npz');y=z['labels'][:,J[row.outcome]];p=z[row.variant][:,J[row.outcome]].astype(np.float64);points.append(roc_auc_score(y,p) if row.metric=='auroc' else brier_score_loss(y,p))
  np.testing.assert_allclose(points,[row.baseline_point,row.stressed_point],atol=1e-12,rtol=0)
 for row in d.itertuples():
  a=arr('bionutrigpt','baseline',row);b=arr('bionutrigpt',row.scenario,row);c=arr(row.comparator,'baseline',row);e=arr(row.comparator,row.scenario,row)
  delta=(a-b)-(c-e) if row.metric=='auroc' else (b-a)-(e-c)
  check(delta,row.diff_deterioration_ci_lower,row.diff_deterioration_ci_upper);check(b-e,row.diff_stressed_ci_lower,row.diff_stressed_ci_upper)
 receipt={'passed':True,'replicates':10000,'within_rows':len(w),'between_rows':len(d),'intervals_checked':len(w)*3+len(d)*2,'method':'Independent count-weighted tied-score pair counting; full endpoint reproduction','source_hashes':bindings,'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (O/'verification.json').write_text(json.dumps(receipt,indent=2));print('ALL ENDPOINTS MATCH',flush=True)
if __name__=='__main__':main()
