"""Independent persisted-prediction arithmetic checks."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
root=Path(__file__).resolve().parent;p=root/'Saved Model/stage1_robustness_v3'
complete=json.loads((p/'completed.json').read_text())
metrics=pd.read_csv(p/'metrics.csv');assert len(metrics)==324
assert hashlib.sha256((p/'metrics.csv').read_bytes()).hexdigest()==complete['metrics_sha256']
reference=pd.read_csv(root/'Data/stage1_repaired_v2/temporal_ml_ready.csv')
outcomes=['TARGET_DIABETES','TARGET_OBESITY','TARGET_CVD']
for (model,scenario),rows in metrics.groupby(['model','scenario']):
 data=np.load(p/(model+'_'+scenario+'_predictions_LOCAL.npz'))
 base=np.load(p/(model+'_baseline_predictions_LOCAL.npz'))
 assert np.array_equal(data['ids'],reference.SEQN.to_numpy())
 assert np.array_equal(data['labels'],reference[outcomes].to_numpy())
 for _,row in rows.iterrows():
  i=outcomes.index(row.outcome);pred=data[row.variant][:,i];y=data['labels'][:,i]
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
  for key,value in [('auroc',roc_auc_score(y,pred)),('average_precision',average_precision_score(y,pred)),('brier',brier_score_loss(y,pred)),('decision_flip_fraction',np.mean((pred>=.5)!=(base[row.variant][:,i]>=.5)))]:
   assert abs(row[key]-value)<1e-10,(model,scenario,key)
for path,digest in json.loads((p/'binding.json').read_text()).items():
 assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
(p/'verification.json').write_text(json.dumps({'verified':True,'metric_rows':324,'model_scenario_pairs':54,'participant_count':len(reference),'checks':['frozen hashes','participant alignment','labels','finite probability bounds','AUROC','AP','Brier','decision flip arithmetic']},indent=2))
print('Verified all 324 metric rows and 54 prediction artifacts.')
