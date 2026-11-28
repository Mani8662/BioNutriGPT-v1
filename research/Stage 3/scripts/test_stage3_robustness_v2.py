"""Repair tests: positive controls and real validator rejection, no fabricated benchmark."""
import copy, tempfile
from pathlib import Path
import stage3_robustness_v2 as r

def reject(fn):
    try: fn()
    except (r.ValidationError, r.audit.RobustnessValidationError): return
    raise AssertionError('Corrupted input accepted')

def run():
    r.check_manifest()
    scenarios=r.read(r.DIR/'scenario_manifest.json')['scenarios']; foods=r.catalog()
    for sid,s in scenarios.items():
        if not s['replay_evaluation_only']: continue
        delta=.05 if '5pct' in sid else .1
        for fid,f in foods.items():
            for k,v in f['nutrients_per_serving'].items():
                p=s['perturbed_nutrients'][fid][k]
                if v is None or k in f.get('unknown_nutrients',[]): assert p is None
                elif v==0: assert p==0
                else: assert abs(p/v-1)<=delta+1e-12
    # Changed interval that STILL contains the mean must fail exact replay.
    expected=r.bootstrap([0.,1.,2.]); changed=copy.deepcopy(expected); changed['ci95']=[-1.,3.]
    reject(lambda:r.require(changed==r.bootstrap([0.,1.,2.]),'Changed CI'))
    assert r.bootstrap([0.,1.,2.])==expected
    pilot=r.DIR/'pilot/checkpoints'
    if pilot.exists() and list(pilot.glob('*.json')):
        ts=r.tasks([0]); r.verify(pilot,ts)
        t=next(t for t in ts if t[4]=='nominal_baseline' and t[3]=='pure_vegetarian')
        source=r.read(r.path_for(pilot,t,'a'))
        assert source['accepted'], 'Positive fixture must be valid; do not substitute an easier case'
        mutations={
          'identity':lambda x:x['identity'].update(SEQN=-1),
          'budget':lambda x:x.update(budget_seconds=10),
          'acceptance':lambda x:x.update(accepted=False),
          'quantity':lambda x:x['basket'][0].update(quantity=99),
          'total':lambda x:x['audit']['totals'].update(DR1TKCAL=0),
          'target_error':lambda x:x['audit'].update(target_error=999),
          'fingerprint':lambda x:x.update(binding='changed'),
          'exclusion':lambda x:x.update(exclusions=['unexpected']),
        }
        for name,mutate in mutations.items():
            x=copy.deepcopy(source); mutate(x); reject(lambda:r.validate_record(x,t,'a',pilot))
        replay=next(t for t in ts if t[5]['replay_evaluation_only'])
        x=r.read(r.path_for(pilot,replay,'a')); x['accepted']=not x['accepted']
        reject(lambda:r.validate_record(x,replay,'a',pilot))
        with tempfile.TemporaryDirectory() as tmp:
            reject(lambda:r.verify(Path(tmp),ts))
        rows=[r.read(p) for p in pilot.glob('*.json')]
        reject(lambda:r.analyze(rows+[copy.deepcopy(rows[0])]))
        r.write(r.DIR/'repair_tests.json',{'status':'passed','pilot_records':len(rows),'mutation_checks':11,'scope':'One reused real participant engineering pilot, not synthetic cohort evidence; CI and perturbation checks also passed'})
    print('Repair checks passed')
if __name__=='__main__': run()
