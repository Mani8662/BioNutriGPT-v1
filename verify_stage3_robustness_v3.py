"""Independent statistical formula replay; plan audit shares disclosed v2 policy code."""
import argparse
import math
import numpy as np
import stage3_robustness_v3 as r

def verify_statistics(records, saved):
    cases=sorted({x['identity']['case'] for x in records})
    index={(x['identity']['case'],x['identity']['mode'],x['identity']['scenario'],x['variant']):x for x in records}
    r.require(len(index)==len(records),'Duplicate records')
    refs=[]; intervals=0
    for scenario,result in saved.items():
        for endpoint in ['accepted_pp','target_error_joint','runtime_all_attempts']:
            people=[]
            for case in cases:
                modes=[]
                for mode in r.MODES:
                    a=index[case,mode,scenario,'a']; b=index[case,mode,scenario,'b']
                    if endpoint=='accepted_pp': modes.append((float(b['accepted'])-float(a['accepted']))*100)
                    elif endpoint=='runtime_all_attempts':
                        if 'runtime' in a and 'runtime' in b: modes.append(b['runtime']-a['runtime'])
                    elif a['accepted'] and b['accepted']:
                        x=a['audit']['target_error']; y=b['audit']['target_error']
                        if x is not None and y is not None: modes.append(y-x)
                if modes: people.append(sum(modes)/len(modes))
            got=result['contrasts'][endpoint]
            r.require(got['n']==len(people),'Participant denominator mismatch')
            if not people:
                r.require(got['mean'] is None and got['ci95'] is None,'Empty endpoint was imputed'); continue
            rng=np.random.default_rng(20261124)
            draws=rng.integers(0,len(people),(10000,len(people)))
            samples=np.asarray(people)[draws].mean(axis=1)
            ci=np.quantile(samples,[.025,.975])
            r.require(math.isclose(got['mean'],sum(people)/len(people),abs_tol=1e-12,rel_tol=0),'Point estimate mismatch')
            r.require(np.allclose(ci,got['ci95'],atol=1e-12,rtol=0),'CI mismatch'); intervals+=1
        for mode,test in result['tests'].items():
            wins=losses=0
            for case in cases:
                a=index[case,mode,scenario,'a']['accepted']; b=index[case,mode,scenario,'b']['accepted']
                wins+=int(b and not a); losses+=int(a and not b)
            n=wins+losses
            p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
            r.require((wins,losses)==(test['wins'],test['losses']),'Discordance mismatch')
            r.require(abs(test['p']-p)<1e-12,'Exact probability mismatch'); refs.append((p,test))
        for variant in ['a','b']:
            count=sum(index[c,m,scenario,variant]['accepted'] for c in cases for m in r.MODES)
            r.require(count==result['accepted'][variant],'Acceptance count mismatch')
    maximum=0.
    for rank,(p,test) in enumerate(sorted(refs,key=lambda x:x[0])):
        maximum=max(maximum,min(1.,(len(refs)-rank)*p))
        r.require(abs(test['holm_p']-maximum)<1e-12,'Holm mismatch')
    return {'intervals':intervals,'exact_tests':len(refs)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--pilot',action='store_true'); args=p.parse_args()
    r.check_manifest(); base=r.DIR/('pilot' if args.pilot else 'full')/'checkpoints'
    ts=r.tasks([0] if args.pilot else None); r.verify(base,ts)
    records=[r.read(x) for x in sorted(base.glob('*.json'))]
    result=r.read(base.parent/'analysis.json')
    counts=verify_statistics(records,result)
    # Alter an interval while retaining its estimate; reject through actual verifier.
    import copy
    altered=copy.deepcopy(result)
    first=next(iter(altered.values()))['contrasts']['accepted_pp']
    first['ci95']=[first['mean']-100,first['mean']+100]
    try: verify_statistics(records,altered)
    except r.ValidationError: pass
    else: raise AssertionError('Verifier accepted changed CI')
    r.write(base.parent/'independent_statistics_verification.json',{'status':'passed',**counts,'changed_ci_rejected':True,'scope':'Independent statistical formulas; shared plan-audit helpers. Pilot intervals are mechanical checks, not population evidence.','script_sha256':r.sha(__file__),'analysis_sha256':r.sha(base.parent/'analysis.json')})
    print('Independent statistical replay passed',counts)
if __name__=='__main__': main()
