"""Frozen-model weighted evaluation with stratified delete-PSU jackknife."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(__file__).resolve().parent
O=R/'Saved Model/stage1_survey_v1'
def evaluate(y,p,w):
 return np.array([roc_auc_score(y,p,sample_weight=w),np.average((y-p)**2,weights=w),average_precision_score(y,p,sample_weight=w)])
def main():
 O.mkdir(exist_ok=False)
 protocol={'weight':'WTDRD1PP','reason':'39 DR1T nutrient predictors and reliable day-one recall selection; no fasting subsample predictor in primary model','source':'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DR1TOT.htm','scope':'Overall six frozen models; calibrated probabilities; diabetes/CVD primary, obesity exploratory. Unweighted comparison on identical participants. No retraining.','variance':'Stratified delete-one-PSU jackknife; surviving PSU weights in deleted stratum multiplied by nh/(nh-1). Variance sum_h (nh-1)/nh sum_i (estimate_hi-estimate_full)^2. Student t intervals with PSU-minus-strata degrees of freedom, no finite population correction; intervals untruncated.','limitations':'Sensitivity analysis of selected complete-outcome/recall/laboratory cohort; weights do not correct additional analytic selection. Conditional on fitted models. Not Indian validation. Exploratory paired intervals without multiplicity adjustment. Subgroup survey evaluation remains pending.'}
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2))
 meta=pd.read_csv(R/'Data/stage1_repaired_v2/survey_metadata_LOCAL.csv').set_index('SEQN')
 truth=pd.read_csv(R/'Data/stage1_repaired_v2/temporal_ml_ready.csv').sort_values('SEQN')
 a=meta.loc[truth.SEQN]; assert len(a)==7004 and a.index.is_unique
 w=a.WTDRD1PP.to_numpy(); assert np.isfinite(w).all() and (w>0).all()
 assert a[['SDMVSTRA','SDMVPSU']].notna().all().all()
 reps=[]; scales=[]
 for h,g in a.groupby('SDMVSTRA'):
  psus=g.SDMVPSU.unique(); n=len(psus); assert n>=2
  for psu in psus:
   wr=w.copy(); inside=(a.SDMVSTRA==h).to_numpy(); deleted=inside&(a.SDMVPSU==psu).to_numpy()
   wr[inside]*=n/(n-1); wr[deleted]=0; reps.append(wr); scales.append((n-1)/n)
 reps=np.array(reps); scales=np.array(scales); df=len(reps)-a.SDMVSTRA.nunique(); critical=t.ppf(.975,df)
 rows=[]; cache={}; hashes={}; targets=['TARGET_DIABETES','TARGET_CVD','TARGET_OBESITY']; names=['auroc','brier','average_precision']
 for model in ['logistic_regression','hist_gradient_boosting','catboost','mlp','bionutrigpt','ft_transformer']:
  path=R/'Saved Model/stage1_comparison_v2'/model/'temporal_predictions_LOCAL.csv'; hashes[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
  f=pd.read_csv(path).sort_values('SEQN'); np.testing.assert_array_equal(f.SEQN,truth.SEQN)
  for target in targets:
   y=f[target].values; np.testing.assert_array_equal(y,truth[target]); p=f[target+'_calibrated'].values
   assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
   full=evaluate(y,p,w); unweighted=evaluate(y,p,np.ones(len(w))); rr=np.array([evaluate(y,p,wr) for wr in reps]); cache[model,target]=(full,rr)
   se=np.sqrt((scales[:,None]*(rr-full)**2).sum(0))
   assert np.isclose(full[1],np.sum(w*(y-p)**2)/w.sum())
   for k,metric in enumerate(names): rows.append(dict(model=model,outcome=target,metric=metric,weighted=full[k],unweighted=unweighted[k],se=se[k],lower=full[k]-critical*se[k],upper=full[k]+critical*se[k]))
  print('Completed '+model,flush=True)
 pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False)
 differences=[]
 for (model,target),(point,rr) in cache.items():
  if model=='bionutrigpt':continue
  bp,br=cache['bionutrigpt',target]; delta=bp-point; replicate=br-rr; se=np.sqrt((scales[:,None]*(replicate-delta)**2).sum(0))
  for k,metric in enumerate(names): differences.append(dict(comparator=model,outcome=target,metric=metric,difference=delta[k],se=se[k],lower=delta[k]-critical*se[k],upper=delta[k]+critical*se[k]))
 pd.DataFrame(differences).to_csv(O/'paired_differences.csv',index=False)
 np.savez_compressed(O/'replicate_metrics.npz',**{model+'__'+target:rr for (model,target),(point,rr) in cache.items()},scales=scales)
 audit={'n':len(w),'strata':int(a.SDMVSTRA.nunique()),'psus':len(reps),'design_df':int(df),'kish_weight_only_effective_n':float(w.sum()**2/(w*w).sum()),'note':'Kish weight-only diagnostic is not a design-based effective sample size','prediction_hashes':hashes,'metadata_sha256':hashlib.sha256((R/'Data/stage1_repaired_v2/survey_metadata_LOCAL.csv').read_bytes()).hexdigest()}
 (O/'audit.json').write_text(json.dumps(audit,indent=2)); print(audit['design_df'])
if __name__=='__main__':main()
