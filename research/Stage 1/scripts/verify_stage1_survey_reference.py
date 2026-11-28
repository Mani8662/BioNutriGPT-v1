from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent
O=R/'Saved Model/stage1_survey_reference_v1'
def main():
 assert (O/'run_finished.txt').exists()
 m=pd.read_csv(O/'metrics.csv');p=pd.read_csv(O/'paired_differences.csv')
 old=R/'Saved Model/stage1_survey_v1'
 for filename,new,keys,value in [('metrics.csv',m,['model','outcome','metric'],'weighted'),('paired_differences.csv',p,['comparator','outcome','metric'],'difference')]:
  prior=pd.read_csv(old/filename);current=new[new.group=='Overall'].drop(columns='group')
  joined=prior.merge(current,on=keys,validate='one_to_one',suffixes=('_old','_r'));assert len(joined)==len(prior)
  if value=='weighted':np.testing.assert_allclose(joined.weighted,joined.estimate,atol=1e-10)
  else:np.testing.assert_allclose(joined.difference_old,joined.difference_r,atol=1e-10)
  for col in ['se','lower','upper']:np.testing.assert_allclose(joined[col+'_old'],joined[col+'_r'],atol=1e-10)
 assert len(m)==594 and len(p)==495
 assert np.isfinite(m[['estimate','se','lower','upper']]).all().all()
 assert np.isfinite(p[['difference','se','lower','upper']]).all().all()
 # Independently check every subgroup weighted point using sklearn.
 from sklearn.metrics import roc_auc_score,average_precision_score
 part=pd.read_csv(R/'Saved Model/stage1_subgroups_v1/participant_subgroups_LOCAL.csv').sort_values('SEQN')
 meta=pd.read_csv(R/'Data/stage1_repaired_v2/survey_metadata_LOCAL.csv').set_index('SEQN').loc[part.SEQN]
 masks={'Overall':np.ones(len(part),bool),'Female':part.RIAGENDR.values==2,'Male':part.RIAGENDR.values==1,'Age20_40':part.RIDAGEYR.values<=40,'Age41_65':(part.RIDAGEYR.values>=41)&(part.RIDAGEYR.values<=65),'AgeOver65':part.RIDAGEYR.values>65}
 masks.update({f'RIDRETH1_{i}':part.RIDRETH1.values==i for i in range(1,6)})
 for model,rows in m.groupby('model'):
  f=pd.read_csv(R/'Saved Model/stage1_comparison_v2'/model/'temporal_predictions_LOCAL.csv').sort_values('SEQN');np.testing.assert_array_equal(f.SEQN,part.SEQN)
  for row in rows.itertuples():
   mask=masks[row.group];y=f[row.outcome].values[mask];prob=f[row.outcome+'_calibrated'].values[mask];w=meta.WTDRD1PP.values[mask]
   val=roc_auc_score(y,prob,sample_weight=w) if row.metric=='auroc' else average_precision_score(y,prob,sample_weight=w) if row.metric=='average_precision' else np.average((y-prob)**2,weights=w)
   assert abs(val-row.estimate)<1e-10
 receipt={'passed':True,'overall_metric_rows_matched':54,'overall_paired_rows_matched':45,'independent_subgroup_and_overall_point_checks':594,'reference':'R survey JKn mse=TRUE','hashes':{f:hashlib.sha256((O/f).read_bytes()).hexdigest() for f in ['metrics.csv','paired_differences.csv','protocol.txt','sessionInfo.txt']}}
 receipt['runner_sha256']=hashlib.sha256((R/'run_stage1_survey_reference.R').read_bytes()).hexdigest()
 receipt['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (O/'verification.json').write_text(json.dumps(receipt,indent=2));print(receipt)
if __name__=='__main__':main()
