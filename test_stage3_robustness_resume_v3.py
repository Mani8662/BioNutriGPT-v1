"""Resume and fail-closed checks using temporary files, preserving study evidence."""
import copy, tempfile
from pathlib import Path
import stage3_robustness_v3 as r

def reject(fn):
    try: fn()
    except r.ValidationError: return
    raise AssertionError('Corruption was accepted')

def main():
    r.check_manifest(); ts=r.tasks([0]); base=r.DIR/'pilot/checkpoints'
    before={p.name:r.sha(p) for p in base.glob('*.json')}
    original=r.legacy.rbs.solve_budget_meal_plan
    def forbidden(*args,**kwargs): raise AssertionError('Resume attempted another solver call')
    r.legacy.rbs.solve_budget_meal_plan=forbidden
    try:
        for t in ts:
            for v in ['a','b']: r.execute(t,v,base)
    finally: r.legacy.rbs.solve_budget_meal_plan=original
    assert before=={p.name:r.sha(p) for p in base.glob('*.json')}
    t=next(x for x in ts if x[4]=='nominal_baseline' and x[3]=='pure_vegetarian')
    record=r.read(r.path_for(base,t,'a'))
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        bad=copy.deepcopy(record); bad['audit']['target_error']+=1
        p=r.path_for(folder,t,'a'); r.write(p,bad); h=r.sha(p)
        reject(lambda:r.execute(t,'a',folder)); assert r.sha(p)==h
        # Real bound-file modification in an isolated manifest environment.
        root_before,dir_before=r.ROOT,r.DIR
        f=folder/'bound.txt'; f.write_text('original')
        r.write(folder/'input_manifest.json',{'bound.txt':r.sha(f)})
        try:
            r.ROOT=r.DIR=folder; r.check_manifest()
            f.write_text('changed'); reject(r.check_manifest)
        finally: r.ROOT,r.DIR=root_before,dir_before
    # Exclusion check: modify an otherwise valid fixture and its task together.
    modified=list(copy.deepcopy(t)); modified[5]['excluded_food_ids']=[record['basket'][0]['food_id']]
    bad=copy.deepcopy(record); bad['binding']=r.binding(modified); bad['exclusions']=r.excluded(modified)
    reject(lambda:r.validate_record(bad,modified,'a',base))
    r.write(r.DIR/'resume_tests.json',{'status':'passed','resumed_records':len(before),'solver_calls':0,'checkpoint_bytes_unchanged':True,'corrupt_checkpoint_refused_without_overwrite':True,'modified_bound_file_rejected':True,'excluded_food_rejected':True,'script_sha256':r.sha(__file__)})
    print('Resume, manifest and exclusion checks passed')
if __name__=='__main__': main()
