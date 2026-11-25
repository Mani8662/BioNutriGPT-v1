"""Repair tests on development cases only; no full evaluation optimization."""
import copy,tempfile
from pathlib import Path
import numpy as np
import stage2_ablation_runner_v2 as r
import verify_stage2_ablation_v2 as verifier

def reject(fn):
    try:fn()
    except r.ValidationError:return
    raise AssertionError('Corruption accepted')
def main():
    r.check_manifest();cases,variants=r.load(True);context=r.model_context()
    records,stats=r.verify(True,context);verifier.statistics(records,stats,cases,variants)
    c=cases[0];v='variant_0_reference';config=variants[v];folder=r.DIR/'pilot/checkpoints'
    original=r.read(r.checkpoint(folder,c,v))
    mutations={
      'identity':lambda x:x.update(SEQN=-1),
      'variant':lambda x:x.update(variant_id='unknown'),
      'binding':lambda x:x.update(binding='bad'),
      'config':lambda x:x['variant_config'].update(include_model_objective=False),
      'probability':lambda x:x['evaluation']['target_probabilities'].update(diabetes=.999),
      'acceptance':lambda x:x['evaluation'].update(overall_accepted=not x['evaluation']['overall_accepted']),
      'distance':lambda x:x['evaluation'].update(distance_from_observed_l1_norm=999),
      'target':lambda x:x['vectors']['final_target'].__setitem__(0,-100),
    }
    for name,mutate in mutations.items():
        corrupt=copy.deepcopy(original);mutate(corrupt);reject(lambda:r.validate(corrupt,c,v,config,context))
    reject(lambda:r.analyze(records[:-1],cases,variants));reject(lambda:r.analyze(records+[records[0]],cases,variants))
    altered=copy.deepcopy(stats);next(iter(altered.values()))['wall_ms']['mean']+=100
    reject(lambda:verifier.statistics(records,altered,cases,variants))
    # Resume must never call the engine or change bytes.
    hashes={p.name:r.sha(p) for p in folder.glob('*.json')};fn=r.engine.run_ablation_variant
    def forbidden(**kwargs):raise AssertionError('Resume called optimizer')
    r.engine.run_ablation_variant=forbidden
    try:
        for case in cases:
            for variant,cfg in variants.items():r.execute(folder,case,variant,cfg,context,sorted(variants))
    finally:r.engine.run_ablation_variant=fn
    assert hashes=={p.name:r.sha(p) for p in folder.glob('*.json')}
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td);corrupt=copy.deepcopy(original);corrupt['evaluation']['distance_from_observed_l1_norm']+=1
        path=r.checkpoint(temp,c,v);r.write(path,corrupt);h=r.sha(path)
        reject(lambda:r.execute(temp,c,v,config,context,sorted(variants)));assert r.sha(path)==h
        root_before,dir_before=r.ROOT,r.DIR
        f=temp/'source.txt';f.write_text('original');r.write(temp/'input_manifest.json',{'source.txt':r.sha(f)})
        try:
            r.ROOT=r.DIR=temp;r.check_manifest();f.write_text('modified');reject(r.check_manifest)
        finally:r.ROOT,r.DIR=root_before,dir_before
    # Reference targets match preserved v1 pilot; no claims about runtime parity.
    parity=[]
    for case in cases:
        old=r.read(r.checkpoint(r.OLD/'pilot/checkpoints',case,v));new=r.read(r.checkpoint(folder,case,v))
        delta=float(np.max(np.abs(np.asarray(old['vectors']['final_target'])-new['vectors']['final_target'])))
        assert delta<1e-6;parity.append(delta)
    # Current reference versus deployed comparison implementation, one development case.
    import test_stage2_ablation_v1 as prior
    prior.test_reference_parity()
    r.check_manifest()
    r.write(r.DIR/'repair_tests.json',{'status':'passed','corruption_checks':13,'validated_resume_records':18,'resume_solver_calls':0,'reference_target_max_differences':parity,'reference_parity_test':'passed_one_development_case','full_study_run':False,'script_sha256':r.sha(__file__)})
    print('Repair tests passed; full study not run')
if __name__=='__main__':main()
