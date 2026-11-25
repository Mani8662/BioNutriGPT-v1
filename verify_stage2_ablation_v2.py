"""Full/pilot reconstruction plus independently implemented paired statistics."""
import argparse,copy,math
import numpy as np
import stage2_ablation_runner_v2 as r

def statistics(records,saved,cases,variants):
    idx={(x['SEQN'],x['variant_id']):x for x in records}
    r.need(len(idx)==len(records)==len(cases)*len(variants),'Missing/duplicate records')
    families={'primary':[],'secondary':[]};count=0
    for variant,endpoints in saved.items():
        pairs=[(idx[c['SEQN'],'variant_0_reference'],idx[c['SEQN'],variant]) for c in cases]
        for name,e in endpoints.items():
            if name in ['overall_accepted','numerical_pass','score_gate_pass']:
                bits=[(a['evaluation'][name],b['evaluation'][name]) for a,b in pairs]
                values=[100*(int(b)-int(a)) for a,b in bits];w=sum(b and not a for a,b in bits);l=sum(a and not b for a,b in bits);n=w+l
                p=min(1.,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n) if n else 1.
                r.need((w,l)==(e['wins'],e['losses']) and abs(p-e['p'])<1e-12,'Exact test mismatch')
                families['primary' if name=='overall_accepted' else 'secondary'].append((p,e))
            elif name=='wall_ms':values=[b['wall_ms']-a['wall_ms'] for a,b in pairs]
            elif name.startswith('delta_'):values=[b['evaluation']['score_deltas'][name[6:]]-a['evaluation']['score_deltas'][name[6:]] for a,b in pairs]
            else:values=[b['evaluation'][name]-a['evaluation'][name] for a,b in pairs]
            arr=np.array(values);rng=np.random.default_rng(20261124)
            means=arr[rng.integers(0,len(arr),(10000,len(arr)))].mean(axis=1);ci=np.quantile(means,[.025,.975])
            r.need(e['n']==len(arr) and np.isclose(e['mean'],sum(values)/len(values),rtol=0,atol=1e-10),'Mean/denominator mismatch')
            r.need(np.allclose(ci,e['ci95'],rtol=0,atol=1e-10),'CI mismatch');count+=1
    for family in families.values():
        maximum=0.
        for rank,(p,e) in enumerate(sorted(family,key=lambda x:x[0])):
            maximum=max(maximum,min(1.,p*(len(family)-rank)));r.need(abs(e['holm_p']-maximum)<1e-12,'Holm mismatch')
    return {'intervals':count,'exact_tests':sum(map(len,families.values()))}
def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--verify-pilot',action='store_true');g.add_argument('--verify-full',action='store_true');args=p.parse_args()
    records,saved=r.verify(args.verify_pilot);cases,variants=r.load(args.verify_pilot)
    counts=statistics(records,saved,cases,variants)
    corrupt=copy.deepcopy(saved);e=next(iter(corrupt.values()))['overall_accepted'];e['ci95']=[e['mean']-100,e['mean']+100]
    try:statistics(records,corrupt,cases,variants)
    except r.ValidationError:pass
    else:raise AssertionError('Changed interval accepted')
    folder=r.DIR/('pilot' if args.verify_pilot else 'full')
    r.write(folder/'independent_statistics_verification.json',{'status':'passed',**counts,'participants':len(cases),'changed_ci_rejected':True,'script_sha256':r.sha(__file__),'analysis_sha256':r.sha(folder/'analysis.json'),'scope':'Separate statistical formulas; shared deployed preprocessing and numerical audit. Pilot is engineering evidence only.'})
    print('Independent statistics passed',counts)
if __name__=='__main__':main()
