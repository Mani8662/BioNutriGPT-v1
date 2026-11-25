"""Independent arithmetic/statistical replay from target vectors and paired records."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from stage2_counterfactual_engine import DIET_RAW_COLS, DAILY_REFERENCE_SCALES
R=Path(__file__).resolve().parent; P=R/'Saved Model/stage2_comparison_study_v2'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases=read(P/'cases_LOCAL.json'); analysis=read(P/'analysis_summary.json'); receipt=read(P/'verification.json')
assert receipt['verification_passed']
protocol_hash=sha(P/'protocol.json'); keys=['rules_only','minimum_change','model_guided']; data={}; scales=np.array([DAILY_REFERENCE_SCALES[k] for k in DIET_RAW_COLS])
for arm in keys:
    rows=[read(P/'checkpoints'/f"case_{c['case']:03d}_{c['SEQN']}_{arm}.json") for c in cases]
    assert len(rows)==100 and all(x['success'] and x['protocol_sha256']==protocol_hash for x in rows)
    for x in rows:
        obs=np.array(x['observed_nutrients']); target=np.array(x['target_nutrients']); dist=np.linalg.norm((target-obs)/scales)
        assert abs(dist-x['distance_from_observed_normalized'])<1e-10
        for tag,item in x['major_nutrient_changes'].items():
            i=DIET_RAW_COLS.index('DR1T'+tag)
            assert abs(item['observed']-obs[i])<1e-10 and abs(item['target']-target[i])<1e-10
            assert abs(item['delta']-(target[i]-obs[i]))<1e-10
        for outcome in ['diabetes','obesity','cvd']:
            assert abs(x['score_deltas'][outcome]-(x['target_probabilities'][outcome]-x['baseline_probabilities'][outcome]))<1e-10
    stats=analysis['arm_statistics'][arm]
    for endpoint in ['numerical_pass','score_gate_pass','overall_accepted']:
        assert stats[endpoint]==sum(x[endpoint] for x in rows)
    times=[x['runtime_seconds'] for x in rows]
    for key,val in [('median',np.median(times)),('p95',np.percentile(times,95)),('total',sum(times))]:
        assert np.isclose(stats['runtime_seconds'][key],val,rtol=1e-12)
    ds=np.array([x['distance_from_observed_normalized'] for x in rows])
    for key,val in [('median',np.median(ds)),('mean',np.mean(ds)),('std',np.std(ds)),('min',np.min(ds)),('max',np.max(ds)),('iqr',np.percentile(ds,75)-np.percentile(ds,25))]:
        assert np.isclose(stats['distance_from_observed'][key],val,rtol=1e-12)
    assert stats['lp_fallbacks']==sum(bool(x['solver_audit'].get('projection_audit',{}).get('fallback_used')) for x in rows)
    for tag,group in analysis['major_nutrient_shifts'].items():
        vals=[x['major_nutrient_changes'][tag] for x in rows]; saved=group[arm]
        for field,key in [('observed','observed_mean'),('target','target_mean'),('delta','delta_mean')]:
            assert np.isclose(saved[key],np.mean([x[field] for x in vals]),rtol=1e-12)
        deltas=[x['delta'] for x in vals]
        assert np.isclose(saved['delta_median'],np.median(deltas))
        assert np.isclose(saved['delta_iqr'],np.percentile(deltas,75)-np.percentile(deltas,25))
    data[arm]=rows
for j,c in enumerate(cases):
    for arm in keys[1:]:
        assert data[arm][j]['observed_nutrients']==data[keys[0]][j]['observed_nutrients']
        assert data[arm][j]['baseline_probabilities']==data[keys[0]][j]['baseline_probabilities']
contrasts=[]
for entry in analysis['paired_contrasts']:
    a=data[entry['arm_1']]; b=data[entry['arm_2']]
    for field,endpoint in [('acceptance','overall_accepted'),('numerical_feasibility','numerical_pass'),('score_gate','score_gate_pass'),('distance_difference','distance_from_observed_normalized')]:
        diff=np.array([float(x[endpoint])-float(y[endpoint]) for x,y in zip(a,b)])
        rng=np.random.default_rng(20261123)
        # Separately implemented bootstrap loop, not the analysis helper.
        boots=np.array([np.mean(diff[rng.integers(0,len(diff),len(diff))]) for _ in range(10000)])
        lo,hi=np.quantile(boots,[.025,.975]); saved=entry[field]
        if field=='distance_difference':
            assert np.allclose([saved['point_diff'],saved['ci_lower'],saved['ci_upper']],[diff.mean(),lo,hi],atol=1e-12)
        else:
            wins=int(sum(diff>0)); losses=int(sum(diff<0)); n=wins+losses
            # Exact binomial probability using combinatorics, independent of scipy call in analysis.
            p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
            assert np.allclose([saved['point_diff_pct'],saved['ci_lower_pct'],saved['ci_upper_pct'],saved['two_sided_p']],[diff.mean()*100,lo*100,hi*100,p],atol=1e-12)
            assert np.isclose(saved['std_error_pct'],np.std(boots,ddof=1)*100)
            if field=='acceptance': contrasts.append({'contrast':entry['contrast_label'],'wins':wins,'losses':losses,'difference_pp':diff.mean()*100,'ci_pp':[lo*100,hi*100],'exact_p':p,'secondary_bonferroni_p':min(1,2*p) if entry['role']=='secondary' else None})
original=R/'Saved Model/stage2_comparison_study_v1/checkpoints'
max_change=0.; decisions_changed=0
for arm,rows in data.items():
    for x in rows:
        y=read(original/f"case_{x['case']:03d}_{x['SEQN']}_{arm}.json")
        max_change=max(max_change,float(np.max(np.abs(np.array(x['target_nutrients'])-y['target_nutrients']))))
        decisions_changed+=int(x['overall_accepted']!=y['overall_accepted'])
out={'passed':True,'n_records':300,'statistics_recomputed':True,'all_contrasts':contrasts,'max_target_difference_from_v1':max_change,'acceptance_changes_from_v1':decisions_changed,'fresh_model_replay':receipt,'scope':'Post-result correction, shared preprocessing/numerical helper, separate arithmetic/bootstrap implementation; no clinical efficacy claim','hashes':{p.name:sha(p) for p in [P/'analysis_summary.json',P/'protocol.json',R/'analyze_stage2_comparison_study_v2.py',R/'verify_stage2_comparison_study_v2.py',Path(__file__)]}}
(P/'audit_complete.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['fresh_model_replay','hashes']},indent=2))
