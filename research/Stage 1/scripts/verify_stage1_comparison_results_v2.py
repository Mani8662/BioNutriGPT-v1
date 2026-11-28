"""Verify a completed model against saved IDs, labels, predictions and checkpoints.

Read-only verification of scientific artifacts; writes only a separate audit report.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import joblib
import run_stage1_comparison_v2 as runner


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    protocol=json.loads(runner.PROTOCOL_PATH.read_text())
    if args.model not in protocol['models']:raise ValueError('Unknown model')
    folder=runner.RESULTS/args.model
    result=runner.verify_completion(folder,'completed.json')
    if json.loads((folder/'run_binding.json').read_text())!=runner.binding(protocol):
        raise ValueError('Run binding does not match current source/data/environment')
    data,features=runner.load_data()
    scores=[]
    for i in range(len(protocol['models'][args.model])):
        trial=folder/f'trial_{i:02d}';rec=runner.verify_completion(trial,'completed.json')
        vp=np.load(trial/'validation_probabilities.npy')
        score=runner.selection_score(data['validation']['y'],vp)
        np.testing.assert_allclose(score,rec['selection_score'],atol=1e-12,rtol=0)
        scores.append(score)
    selected=max(range(len(scores)),key=lambda i:scores[i])
    assert selected==result['selected_trial']
    device='cuda' if args.model in runner.NEURAL else 'cpu'
    model=runner.load_trial(args.model,protocol['models'][args.model][selected],folder/f'trial_{selected:02d}',device)
    calibrators=joblib.load(folder/'calibrators.joblib')
    for partition in ['validation','calibration','temporal']:
        df=pd.read_csv(folder/f'{partition}_predictions_LOCAL.csv')
        np.testing.assert_array_equal(df.SEQN.to_numpy(),data[partition]['ids'])
        np.testing.assert_array_equal(df[runner.TARGETS].to_numpy(),data[partition]['y'])
        raw=runner.predict(args.model,model,data[partition]['x'],device)
        adjusted=runner.apply_calibrators(calibrators,raw)
        saved_raw=df[[t+'_raw' for t in runner.TARGETS]].to_numpy()
        saved_adjusted=df[[t+'_calibrated' for t in runner.TARGETS]].to_numpy()
        np.testing.assert_allclose(raw,saved_raw,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(adjusted,saved_adjusted,atol=1e-6,rtol=1e-5)
        if partition=='temporal':
            for variant,p in [('raw',saved_raw),('calibrated',saved_adjusted)]:
                for i,t in enumerate(runner.TARGETS):
                    rec=next(m for m in result['metrics'] if m['variant']==variant and m['outcome']==t)
                    for metric,fn in [('auroc',roc_auc_score),('average_precision',average_precision_score),('brier',brier_score_loss),('log_loss',log_loss)]:
                        np.testing.assert_allclose(fn(data[partition]['y'][:,i],p[:,i]),rec[metric],atol=1e-10,rtol=0)
    output={'model':args.model,'status':'passed','completed_record_sha256':runner.sha(folder/'completed.json'),
        'checks':['All final and trial artifact hashes','Current source/data/environment binding','Frozen split membership and schema',
                  'Selection reproduced from all validation trial predictions','Participant IDs and outcome labels match all three exported partitions',
                  'Reloaded model and calibrators reproduce saved predictions','AUROC/AP/Brier/log-loss reproduced from exported temporal predictions'],
        'scope':'No training or new model selection; confidence intervals and comparative inference remain pending.'}
    path=runner.ROOT/'Research Paper'/f'Stage1_{args.model}_v2_Verification.json'
    runner.write_json(path,output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
