"""Independent replay of corrected tables, distances, paired intervals and tests."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from stage2_counterfactual_engine import DAILY_REFERENCE_SCALES
R=Path(__file__).resolve().parent;P=R/'Saved Model/stage2_stage3_translation_audit_v2'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify_ci(diff,saved):
    diff=np.asarray(diff,float);rng=np.random.default_rng(20261123)
    # Independent loop implementation; preserve participant pairing.
    means=[sum(diff[j] for j in rng.integers(len(diff),size=len(diff)))/len(diff) for _ in range(10000)]
    assert saved['n_participants']==len(diff)
    assert np.isclose(saved['mean_difference'],sum(diff)/len(diff),atol=1e-10)
    assert np.allclose(saved['ci95'],np.quantile(means,[.025,.975]),atol=1e-10)
def main():
    analysis=read(P/'analysis_summary.json');rows=read(P/'recomputed_records_LOCAL.json');replay=read(P/'replay_verification.json')
    assert replay['passed'] and replay['audit_source_sha256']==sha(R/'repair_translation_evidence_v2.py')
    ix={(x['case'],x['arm'],x['mode']):x for x in rows};assert len(ix)==900
    modes=['pure_vegetarian','veg_nonveg_combo','pure_non_vegetarian']
    for row in rows:
        if row['stage3']:
            distance=math.sqrt(sum((row['absolute_target_errors'][k]/DAILY_REFERENCE_SCALES[k])**2 for k in row['active_codes']))
            assert np.isclose(distance,row['distance_active_stage2_scales'],atol=1e-12)
            for k,v in row['absolute_target_errors'].items():
                if v is not None: assert np.isclose(v,abs(row['recomputed_totals'][k]-row['targets'][k]))
        else: assert row['distance_active_stage2_scales'] is None
    for arm,stats in analysis['arm_summaries'].items():
        armrows=[x for x in rows if x['arm']==arm]
        assert len(armrows)==stats['attempts']==300
        for ep in ['stage3','end_to_end']: assert stats[ep]==sum(r[ep] for r in armrows)
        times=[r['runtime'] for r in armrows]
        assert np.allclose([stats['runtime_median'],stats['runtime_p95'],stats['runtime_sum']],[np.median(times),np.quantile(times,.95),sum(times)])
        for mode,cell in stats['by_mode'].items():
            for ep in ['stage3','end_to_end']: assert cell[ep]==sum(ix[i,arm,mode][ep] for i in range(100))
    for c in analysis['contrasts']:
        a,b=c['arm1'],c['arm2']
        for ep in ['stage3','end_to_end']:
            ds=[sum(int(ix[i,a,m][ep])-int(ix[i,b,m][ep]) for m in modes)/3*100 for i in range(100)]
            verify_ci(ds,c['pooled'][ep])
            for mode in modes:
                d=[int(ix[i,a,mode][ep])-int(ix[i,b,mode][ep]) for i in range(100)];saved=c['per_mode'][mode][ep]
                verify_ci(np.array(d)*100,saved)
                w=d.count(1);l=d.count(-1);n=w+l
                p=min(1.,2*sum(math.comb(n,j) for j in range(min(w,l)+1))/2**n) if n else 1.
                assert (w,l)==(saved['wins'],saved['losses']) and abs(p-saved['exact_p'])<1e-12
        ds=[];cells=0
        for i in range(100):
            matched=[ix[i,a,m]['distance_active_stage2_scales']-ix[i,b,m]['distance_active_stage2_scales'] for m in modes if ix[i,a,m]['stage3'] and ix[i,b,m]['stage3']]
            if matched:ds.append(sum(matched)/len(matched));cells+=len(matched)
        verify_ci(ds,c['distance']);assert cells==c['distance']['n_joint_success_cells']
    for ep in ['stage3','end_to_end']:
        cells=[c['per_mode'][m][ep] for c in analysis['contrasts'] for m in modes]
        ranked=sorted(cells,key=lambda c:c['exact_p'])
        for j,cell in enumerate(ranked):
            adjusted=min(1.,max((9-i)*ranked[i]['exact_p'] for i in range(j+1)))
            assert abs(adjusted-cell['holm_p_9'])<1e-12
    manifest=read(P/'input_manifest.json');assert all(sha(R/f)==h for f,h in manifest.items())
    receipt={'passed':True,'original_artifacts_unchanged':True,'solver_calls':0,'records':900,'intervals_recomputed':27,'exact_tests_recomputed':18,'holm_families':2,'source_sha256':sha(Path(__file__)),'analysis_sha256':sha(P/'analysis_summary.json'),'replay_sha256':sha(P/'replay_verification.json'),'note':'Independent statistics implementation; model predictions inherited from verified Stage 2, not rerun.'}
    (P/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
