from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.optimize import root,brentq
from scipy.special import expit,logit
from scipy.stats import t
R=Path(__file__).resolve().parent;O=R/'Saved Model/stage1_calibration_dca_v1'
def main():
 assert (O/'run_finished.txt').exists()
 cal=pd.read_csv(O/'calibration.csv');dca=pd.read_csv(O/'decision_curves.csv');pairs=pd.read_csv(O/'decision_curve_pairs.csv');bins=pd.read_csv(O/'calibration_bins.csv')
 assert len(cal)==108 and len(dca)==736 and len(pairs)==644 and len(bins)==360
 models=cal.model.unique();pred={m:pd.read_csv(R/'Saved Model/stage1_comparison_v2'/m/'temporal_predictions_LOCAL.csv').sort_values('SEQN') for m in models}
 ref=pred[models[0]];meta=pd.read_csv(R/'Data/stage1_repaired_v2/survey_metadata_LOCAL.csv').set_index('SEQN').loc[ref.SEQN];w=meta.WTDRD1PP.values
 for f in pred.values():np.testing.assert_array_equal(f.SEQN,ref.SEQN)
 reps=[];scales=[]
 for h,g in meta.groupby('SDMVSTRA'):
  ps=g.SDMVPSU.unique();n=len(ps)
  for psu in ps:
   rw=w.copy();same=meta.SDMVSTRA.values==h;rw[same]*=n/(n-1);rw[same&(meta.SDMVPSU.values==psu)]=0;reps.append(rw);scales.append((n-1)/n)
 reps=np.array(reps);scales=np.array(scales);critical=t.ppf(.975,25)
 for (model,outcome,variant),g in cal.groupby(['model','outcome','variant']):
  y=pred[model][outcome].values;p=pred[model][outcome+'_'+variant].values;z=logit(np.clip(p,1e-7,1-1e-7));wn=w/w.mean()
  fit=root(lambda b:np.array([np.sum(wn*(expit(b[0]+b[1]*z)-y)),np.sum(wn*z*(expit(b[0]+b[1]*z)-y))]),[0,1]);assert fit.success
  off=brentq(lambda a:np.sum(wn*(expit(a+z)-y)),-30,30)
  for row in g.itertuples():
   v=off if row.fit=='offset' else fit.x[0 if row.term=='(Intercept)' else 1];assert abs(v-row.estimate)<1e-6
 for row in bins.itertuples():
  y=pred[row.model][row.outcome].values;p=pred[row.model][row.outcome+'_'+row.variant].values;mask=(p>=row.bin/10)&((p<(row.bin+1)/10)|((row.bin==9)&(p<=1)));assert mask.sum()==row.n
  if row.n:
   assert abs(np.average(y[mask],weights=w[mask])-row.observed)<1e-12
   assert abs(np.average(p[mask],weights=w[mask])-row.mean_predicted)<1e-12
 cache={}
 for row in dca.itertuples():
  y=ref[row.outcome].values;chosen=np.ones(len(y)) if row.model=='all' else np.zeros(len(y)) if row.model=='none' else pred[row.model][row.outcome+'_calibrated'].values>=row.threshold
  x=chosen*(y-(1-y)*row.threshold/(1-row.threshold));v=np.average(x,weights=w);rr=reps@x/reps.sum(1);se=np.sqrt(np.dot(scales,(rr-v)**2));cache[row.outcome,row.model,round(row.threshold,2)]=(v,rr)
  np.testing.assert_allclose([v,se,v-critical*se,v+critical*se],[row.net_benefit,row.se,row.lower,row.upper],atol=1e-10)
 for row in pairs.itertuples():
  a,ar=cache[row.outcome,'bionutrigpt',round(row.threshold,2)];b,br=cache[row.outcome,row.comparator,round(row.threshold,2)];v=a-b;se=np.sqrt(np.dot(scales,(ar-br-v)**2))
  np.testing.assert_allclose([v,se,v-critical*se,v+critical*se],[row.difference,row.se,row.lower,row.upper],atol=1e-10)
 # Simple independent DCA sanity cases: perfect classifier, all and none.
 y=np.array([0,1,0,1]);w0=np.ones(4);threshold=.2
 assert np.average(y*(y-(1-y)*threshold/(1-threshold)),weights=w0)==.5
 assert np.isclose(np.average(y-(1-y)*threshold/(1-threshold),weights=w0),.375)
 receipt={'passed':True,'calibration_point_checks':108,'bins_checked':360,'decision_curve_estimate_and_interval_checks':736,'paired_estimate_and_interval_checks':644,'calibration_interval_scope':'R survey JKn; point estimates independently checked; calibration SEs not separately refitted in Python','hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*O.glob('*.csv'),R/'run_stage1_calibration_dca_v1.R',Path(__file__),R/'Data/stage1_repaired_v2/survey_metadata_LOCAL.csv',* (R/'Saved Model/stage1_comparison_v2').glob('*/temporal_predictions_LOCAL.csv')]}}
 (O/'verification.json').write_text(json.dumps(receipt,indent=2));print('All independent checks passed.')
if __name__=='__main__':main()
