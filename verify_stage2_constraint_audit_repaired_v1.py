"""Replay saved Stage 2 snapshots independently, without importing the engine."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Saved Model/stage2_constraint_audit_repaired_v1'

def check(values,lower,upper,keys,baseline,metadata,cvd):
    n=dict(zip(keys,values));lo=np.array(lower);hi=np.array(upper);v=np.array(values)
    bound_fail=(v<lo-1e-6)|(v>hi+1e-6)
    energy=4*n['DR1TPROT']+4*n['DR1TCARB']+9*n['DR1TTFAT']+7*n['DR1TALCO']
    energy_ok=n['DR1TKCAL']>0 and abs(energy-n['DR1TKCAL'])/n['DR1TKCAL']<=.01+1e-9
    ratios=[n['DR1TSUGR']/(n['DR1TFIBE']+1),n['DR1TSODI']/(n['DR1TPOTA']+1),n['DR1TSFAT']/(n['DR1TMFAT']+n['DR1TPFAT']+.1)]
    limits=[max(baseline['DR1TSUGR']/(baseline['DR1TFIBE']+1),1),max(baseline['DR1TSODI']/(baseline['DR1TPOTA']+1),1),max(baseline['DR1TSFAT']/(baseline['DR1TMFAT']+baseline['DR1TPFAT']+.1),.4)]
    ratio_ok=all(a<=b+1e-6 for a,b in zip(ratios,limits))
    f2=n['DR1TKCAL']>=1200 and n['DR1TSODI']<=(1500 if cvd>=.5 else 2300) and n['DR1TPOTA']>=(3500 if cvd>=.5 else 2600) and .8<=n['DR1TPROT']/metadata['weight_kg']<=2
    return {'bound_names':[k for k,b in zip(keys,bound_fail) if b], 'energy_ok':bool(energy_ok),'ratios_ok':bool(ratio_ok),'filter2_ok':bool(f2),'joint':bool(not bound_fail.any() and energy_ok and ratio_ok and f2)}

def main():
    protocol=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    cases=json.loads((OUT/'cases_LOCAL.json').read_text());assert len(cases)==32
    for p,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    first_bound_origin=Counter();counts=Counter();violations=Counter();errors=[];worst=[]
    for case in cases:
        i=case['case'];r=json.loads((OUT/f'case_{i:03d}_LOCAL.json').read_text())
        if r['error']:
            errors.append({'case':i,'error':r['error']});continue
        dossier=json.loads((OUT/f'dossier_{i:03d}_LOCAL.json').read_text())
        envelope=dossier['nutrient_constraint_envelope'];keys=[x['nutrient'] for x in envelope]
        assert len(keys)==len(set(keys))==39
        lower=[x['min_bound'] for x in envelope];upper=[x['max_bound'] for x in envelope]
        baseline={x['nutrient']:x['baseline'] for x in envelope};p=dossier['patient_baseline']['probabilities'];cvd=p['cvd']
        origin=None
        for stage,n in r['snapshots'].items():
            observed=check(n,lower,upper,keys,baseline,case['metadata'],cvd)
            assert observed['bound_names']==[v['nutrient'] for v in r['checks'][stage]['bound_violations']]
            assert observed['energy_ok']==r['checks'][stage]['energy_consistent']
            assert observed['ratios_ok']==r['checks'][stage]['ratios_pass']
            assert observed['joint']==r['checks'][stage]['joint_numeric_pass']
            counts[stage+'_bound_failures']+=bool(observed['bound_names'])
            if origin is None and observed['bound_names']:origin=stage
        if origin:first_bound_origin[origin]+=1
        np.testing.assert_array_equal(r['snapshots']['after_filter_3'],[x['target'] for x in envelope])
        final=observed
        scores=dossier['audit_filters']['filter_4_closed_loop_rescoring']
        delta=[scores['target_probabilities'][k]-p[k] for k in ['diabetes','obesity','cvd']]
        np.testing.assert_allclose(delta,[scores['delta_'+k] for k in ['diabetes','obesity','cvd']],atol=1e-12,rtol=0)
        closed=all(d<=.0015 for d in delta)
        assert closed==r['closed_loop_pass']
        independently_accepted=closed and final['joint']
        assert independently_accepted==r['independent_joint_pass']
        assert r['certified']==dossier['certified_for_stage3']
        counts['certified']+=r['certified'];counts['independent_joint_pass']+=independently_accepted
        counts['false_accepts']+=r['certified'] and not independently_accepted
        counts['false_rejects_numeric_only']+=not r['certified'] and independently_accepted
        violations.update(final['bound_names'])
        for row in r['checks']['after_filter_3']['bound_violations']:
            worst.append({'case':i,**row})
        oracle=r['linear_feasibility'];counts['linear_feasible']+=oracle['feasible']
        counts['linear_feasible_but_rejected']+=oracle['feasible'] and not r['certified']
        if oracle['feasible']:
            witness=np.array(oracle['witness']);lo=np.array(lower);hi=np.array(upper)
            assert np.all(witness>=lo-1e-6) and np.all(witness<=hi+1e-6)
            # Recheck all inequalities with LP feasibility tolerance to allow boundary rounding.
            w=dict(zip(keys,witness));e=4*w['DR1TPROT']+4*w['DR1TCARB']+9*w['DR1TTFAT']+7*w['DR1TALCO']
            assert .99*w['DR1TKCAL']-1e-6<=e<=1.01*w['DR1TKCAL']+1e-6
            a=max(baseline['DR1TSUGR']/(baseline['DR1TFIBE']+1),1);assert w['DR1TSUGR']<=a*(w['DR1TFIBE']+1)+1e-6
            a=max(baseline['DR1TSODI']/(baseline['DR1TPOTA']+1),1);assert w['DR1TSODI']<=a*(w['DR1TPOTA']+1)+1e-6
            a=max(baseline['DR1TSFAT']/(baseline['DR1TMFAT']+baseline['DR1TPFAT']+.1),.4);assert w['DR1TSFAT']<=a*(w['DR1TMFAT']+w['DR1TPFAT']+.1)+1e-6
            assert w['DR1TKCAL']>=1200-1e-6
            assert .8*case['metadata']['weight_kg']-1e-6<=w['DR1TPROT']<=2*case['metadata']['weight_kg']+1e-6
            assert w['DR1TSODI']<=(1500 if cvd>=.5 else 2300)+1e-6
            assert w['DR1TPOTA']>=(3500 if cvd>=.5 else 2600)-1e-6
    for k in ['certified','false_accepts','linear_feasible','linear_feasible_but_rejected']:assert counts[k]==summary[k]
    assert len(errors)==summary['errors']
    assert dict(violations)==summary['final_violated_nutrients']
    result={'verification_passed':True,'audit_cases':32,'errors':errors,'counts':dict(counts),'first_bound_violation_stage':dict(first_bound_origin),'violated_nutrients':dict(violations),'source_bindings_passed':True,
            'scope':'Archive replay; not rerun optimization or clinical validation','verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'violation_details_LOCAL.json').write_text(json.dumps(worst,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
