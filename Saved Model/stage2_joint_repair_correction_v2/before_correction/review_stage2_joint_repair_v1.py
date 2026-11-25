"""Read-only source/result review; independent frozen-model score replay."""
from pathlib import Path
import json,hashlib,os
import numpy as np
import torch
import joblib
import stage2_counterfactual_engine as s2
import run_stage1_comparison_v2 as study
from verify_stage2_constraint_audit_repaired_v1 import check

R=Path(__file__).resolve().parent
B=R/'Saved Model/stage2_constraint_audit_repaired_v1'
N=R/'Saved Model/stage2_joint_constraint_repair_v1'
O=R/'Saved Model/stage2_joint_repair_review_v1'

def main():
    torch.set_num_threads(os.cpu_count() or 1)
    protocol=json.loads((N/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    cases=json.loads((B/'cases_LOCAL.json').read_text());assert cases==json.loads((N/'cases_LOCAL.json').read_text())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    folder=study.RESULTS/'bionutrigpt';study.verify_completion(folder,'completed.json')
    choice=json.loads((folder/'selection.json').read_text())['selected_trial']
    cfg=json.loads(study.PROTOCOL_PATH.read_text())['models']['bionutrigpt'][choice]
    model=study.load_trial('bionutrigpt',cfg,folder/f'trial_{choice:02d}',device)
    calibrators=joblib.load(folder/'calibrators.joblib')
    artifacts=s2._load_preprocessing_artifacts(s2.DEFAULT_PREPROCESSOR)
    matrices=[];saved=[];rows=[]
    for case in cases:
        i=case['case'];d=json.loads((N/f'dossier_{i:03d}_LOCAL.json').read_text());old=json.loads((B/f'dossier_{i:03d}_LOCAL.json').read_text())
        assert d['energy_prescription']==old['energy_prescription']
        env=d['nutrient_constraint_envelope'];prev=old['nutrient_constraint_envelope']
        assert [(x['nutrient'],x['min_bound'],x['max_bound']) for x in env]==[(x['nutrient'],x['min_bound'],x['max_bound']) for x in prev]
        target=np.array([x['target'] for x in env]);assert np.isfinite(target).all()
        baseline={x['nutrient']:x['baseline'] for x in env}
        numerical=check(target,[x['min_bound'] for x in env],[x['max_bound'] for x in env],[x['nutrient'] for x in env],baseline,case['metadata'],d['patient_baseline']['probabilities']['cvd'])
        assert numerical['joint'],(i,numerical)
        raw,_=s2.canonicalize_dietary_ratios(case['vector'])
        matrices.append(s2._normalize_numpy(raw,artifacts)[0])
        raw_t=torch.tensor(raw,dtype=torch.float32,device=device)
        candidate=s2._construct_candidate_88(raw_t,torch.tensor(target,dtype=torch.float32,device=device))
        fixed=list(range(5))+list(range(51,87))
        np.testing.assert_array_equal(candidate[0,fixed].cpu().numpy(),raw.astype(np.float32)[fixed])
        matrices.append(s2._normalize_torch(candidate,artifacts)[0].detach().cpu().numpy())
        before=d['patient_baseline']['probabilities'];after=d['audit_filters']['filter_4_closed_loop_rescoring']['target_probabilities']
        saved.extend([[v[k] for k in ['diabetes','obesity','cvd']] for v in [before,after]])
        rows.append({'case':i,'before_accepted':old['certified_for_stage3'],'after_accepted':d['certified_for_stage3'],'numeric_constraints_pass':numerical['joint']})
    predicted=study.apply_calibrators(calibrators,study.predict('bionutrigpt',model,np.array(matrices,dtype=np.float32),device))
    np.testing.assert_allclose(predicted,saved,atol=1e-6,rtol=0)
    for i,row in enumerate(rows):
        delta=predicted[2*i+1]-predicted[2*i];accepted=bool(np.all(delta<=.0015))
        assert accepted==row['after_accepted']
        row['rescored_deltas']=dict(zip(['diabetes','obesity','cvd'],delta.tolist()))
    result={'passed':True,'n':len(cases),'device':device,'source_hashes_match':True,'same_cases_bounds_and_energy_prescriptions':True,'all_numeric_constraints_pass':True,'baseline_accepted':[r['case'] for r in rows if r['before_accepted']],'preserved':[r['case'] for r in rows if r['before_accepted'] and r['after_accepted']],'accepted_after':sum(r['after_accepted'] for r in rows),'max_score_replay_error':float(abs(predicted-np.array(saved)).max()),'model_score_gate_replayed_all_32':True,'rows':rows}
    O.mkdir(exist_ok=True);(O/'independent_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
