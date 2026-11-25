from pathlib import Path
import json,math,hashlib
import numpy as np
R=Path(__file__).resolve().parent;P=R/'Saved Model/stage3_target_aware_audit_v2'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=read(P/'analysis_summary.json');rows=read(P/'recomputed_LOCAL.json');idx={(x['case'],x['arm'],x['mode'],x['variant']):x for x in rows};modes=['pure_vegetarian','veg_nonveg_combo','pure_non_vegetarian'];count=0
 def verify(ds,saved):
  nonlocal count
  rng=np.random.default_rng(20261123);d=np.array(ds);boots=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)]
  assert saved['n_participants']==len(d) and np.isclose(saved['mean_difference'],d.mean()) and np.allclose(saved['ci95'],np.quantile(boots,[.025,.975]));count+=1
 for c in a['contrasts']:
  for ep in ['accepted','e2e']:
   ds=[sum(int(idx[i,c['a'],m,c['av']][ep])-int(idx[i,c['b'],m,c['bv']][ep]) for m in modes)/3*100 for i in range(100)];verify(ds,c['binary'][ep])
   for m in modes:
    d=[int(idx[i,c['a'],m,c['av']][ep])-int(idx[i,c['b'],m,c['bv']][ep]) for i in range(100)];w=d.count(1);l=d.count(-1);n=w+l;p=min(1,2*sum(math.comb(n,j) for j in range(min(w,l)+1))/2**n) if n else 1
    s=c['modes'][m][ep];assert (s['wins'],s['losses'])==(w,l) and abs(s['p']-p)<1e-12
  aa=[];bb=[];nc=0
  for i in range(100):
   pairs=[(idx[i,c['a'],m,c['av']],idx[i,c['b'],m,c['bv']]) for m in modes];pairs=[(x['l1'],y['l1']) for x,y in pairs if x['accepted'] and y['accepted']]
   if pairs:aa.append(sum(x for x,y in pairs)/len(pairs));bb.append(sum(y for x,y in pairs)/len(pairs));nc+=len(pairs)
  assert nc==c['distance']['joint_cells'];verify(aa,c['distance']['a_mean']);verify(bb,c['distance']['b_mean']);verify(np.array(aa)-bb,c['distance']['difference'])
 for ep in ['accepted','e2e']:
  cells=sorted([c['modes'][m][ep] for c in a['contrasts'] for m in modes],key=lambda x:x['p'])
  for j,s in enumerate(cells):assert abs(s['holm_p_18']-min(1,max((18-i)*cells[i]['p'] for i in range(j+1))))<1e-12
 for arm,variants in a['summary'].items():
  for v,s in variants.items():
   rs=[x for x in rows if x['arm']==arm and x['variant']==v];assert len(rs)==s['n']==300
   assert sum(x['accepted'] for x in rs)==s['accepted'] and sum(x['e2e'] for x in rs)==s['e2e'] and np.isclose(np.median([x['runtime'] for x in rs]),s['median_runtime'])
 manifest=read(P/'input_manifest.json');assert all(sha(R/f)==h for f,h in manifest.items())
 receipt={'passed':True,'intervals_verified':count,'exact_tests_verified':36,'holm_families':2,'solver_calls':0,'original_evidence_unchanged':True,'analysis_sha256':sha(P/'analysis_summary.json'),'verifier_sha256':sha(Path(__file__))}
 (P/'verification.json').write_text(json.dumps(receipt,indent=2));print(receipt)
if __name__=='__main__':main()
